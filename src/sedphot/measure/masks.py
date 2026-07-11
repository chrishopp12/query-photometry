"""
masks.py

Neighbor Masks for Aperture Photometry
---------------------------------------------------------

Boolean masks that keep neighboring sources out of the sky annulus and
the aperture: a deblended source detection for cleaning the sky
(source_mask), a two-channel interloper mask for the aperture
(neighbor_mask), and user-supplied masks loaded from disk and moved
between pixel grids by WCS (load_user_mask, reproject_mask).

The aperture mask is structural, not model-based. A difference of
Gaussians finds interloper CORES (PSF-scale power above the local
diffuse level), so the target's envelope -- asymmetric, patchy, or
disconnected at any isophote -- can never be masked as its own
neighbor. Each core's full EXTENT then comes from point-reflection
symmetry: light significantly brighter than its 180-degree twin through
the target center cannot belong to the target, and is masked where it
connects to a core -- wings and diffuse envelopes included.

Requirements:
    numpy, scipy, astropy, photutils

Notes:
    Masks are True where a pixel is EXCLUDED.
    A user mask may live on a different pixel grid than the band being
    measured; reproject_mask moves it by WCS, so the mask's native grid
    is immaterial.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from photutils.segmentation import deblend_sources, detect_sources
from scipy import ndimage
from scipy.ndimage import (binary_dilation, binary_propagation,
                           gaussian_filter, map_coordinates)


def _smooth(work: np.ndarray, sigma_px: float) -> np.ndarray:
    """Zero-boundary Gaussian smoothing.

    Separable scipy filter with support matched to the direct astropy
    kernel it replaced (truncate=4 = the 8-sigma kernel width); identical
    output to floating point, ~60x faster at the heavy scale.
    """
    return gaussian_filter(work, sigma_px, mode='constant', cval=0.0,
                           truncate=4.0)


# ------------------------------------
# Auto-masks
# ------------------------------------
def source_mask(
        stamp: np.ndarray,
        threshold_std: float,
        *,
        threshold_map: np.ndarray | None = None,
        npixels: int = 8,
        nlevels: int = 32,
        contrast: float = 0.001,
        dilate: int = 2,
        nodata: np.ndarray | None = None,
) -> np.ndarray:
    """Deblended source mask: an independent detection for cleaning the sky.

    Detects on a lightly smoothed copy at 2 x threshold_std and deblends, so
    a faint neighbor embedded in an extended envelope separates into its own
    segment instead of merging with the target. Every detected segment is
    masked (dilated).

    Parameters
    ----------
    stamp : np.ndarray
        Sky-subtracted image stamp.
    threshold_std : float
        Background rms; the detection threshold is 2 x threshold_std.
    threshold_map : np.ndarray, optional
        Per-pixel threshold override (e.g. a local-brightness floor);
        combined as max(2 x threshold_std, threshold_map).
    npixels, nlevels, contrast : int, int, float
        photutils detect/deblend parameters.
    dilate : int
        Binary-dilation iterations to grow the mask over source wings.
    nodata : np.ndarray (bool), optional
        Off-footprint / blank pixels; zeroed for the detection so they
        neither detect nor propagate NaN through the smoothing.

    Returns
    -------
    mask : np.ndarray (bool)
        True where a source is detected.
    """
    work = stamp if nodata is None else np.where(nodata, 0.0, stamp)
    smoothed = _smooth(np.nan_to_num(work), 1.5)
    threshold = 2.0 * threshold_std
    if threshold_map is not None:
        threshold = np.maximum(threshold, threshold_map)
    segm = detect_sources(smoothed, threshold=threshold, n_pixels=npixels)
    if segm is None:
        return np.zeros(stamp.shape, bool)
    segm = deblend_sources(smoothed, segm, n_pixels=npixels, n_levels=nlevels,
                           contrast=contrast)
    return binary_dilation(segm.data > 0, iterations=dilate)


def radii_arcsec(shape: tuple, cx: float, cy: float, pixscale: float) -> np.ndarray:
    """Radius map (arcsec) about a stamp-pixel center."""
    yy, xx = np.indices(shape)
    return np.hypot(xx - cx, yy - cy) * pixscale


# Difference-of-Gaussians dominance: a candidate is masked where its
# PSF-scale (fine-smoothed) brightness exceeds DOG_FACTOR x its own local
# diffuse level (heavy-smoothed). Diffuse structure -- an envelope,
# however asymmetric or patchy -- has fine ~ heavy and always fails; a
# star or galaxy core has fine >> heavy and passes. No profile model is
# involved, so there is no azimuthal-median reference to collapse on
# one-sided envelopes.
DOG_FACTOR = 1.5

# The heavy smoothing scale (FWHM, arcsec) that defines "diffuse": at
# least 3", or 2.5 seeing when the PSF is fat.
DOG_HEAVY_FWHM_MIN = 3.0


# A source whose smoothed peak clears this many detection sigmas is
# masked in FULL (segment + wings dilation) from the sky region: its
# skirt is individually significant structure, not ambient background.
SKY_BRIGHT_SIGMA = 10.0


# Symmetry-excess significance (units of the raw background rms): a
# pixel belongs to a neighbor where its fine-smoothed brightness exceeds
# its point-reflected twin through the target center by more than this.
# The fine smoothing suppresses the raw rms ~5x, so 1.0 raw-rms is a
# conservative threshold on the difference map while still reaching far
# down a bright neighbor's wings.
EXCESS_SIGMA = 1.0

# Light floor for the excess flood (units of the raw rms): masked pixels
# must also carry this much brightness on the fine map. The excess
# condition alone stops noise percolation (it is ~4 sigma of the
# smoothed difference map); the floor bounds how deep into a bright
# complex's low-surface-brightness fringe the flood may reach. On a
# crowded cluster-core test field, 1.5 masks visibly more halo fringe
# than 2.0 with an unchanged false-mask rate in source-free sectors
# (1.0 tested safe there too; kept conservative for field diversity).
EXCESS_LIGHT_FLOOR = 1.5


def sky_source_mask(
        stamp: np.ndarray,
        threshold_std: float,
        cx: float,
        cy: float,
        pixscale: float,
        *,
        npixels: int = 8,
        seeing_arcsec: float = 1.0,
        nodata: np.ndarray | None = None,
) -> np.ndarray:
    """Sky-region source mask, symmetric with the aperture's treatment.

    Aperture photometry measures the target only if the ambient
    background of faint sources cancels between the aperture and the sky
    region -- so the sky must exclude exactly what the aperture excludes,
    and keep what it keeps. Masking EVERY detected segment out of a deep
    annulus breaks the symmetry: the deep sky becomes true dark sky and
    under-subtracts the faint-source background still inside the
    aperture (the depth-dependent flux offsets between CFHT and SDSS
    measurements of one galaxy).

    Three exclusions, everything else stays:
    - the target's own connected segment (target light is never sky);
    - the FULL segment (dilated ~2") of every bright source
      (peak > SKY_BRIGHT_SIGMA x threshold_std): a bright galaxy's skirt
      is individually significant structure, not ambient background;
    - the DoG cores of everything fainter, grown by a seeing disk --
      mirroring the aperture-side interloper mask.
    """
    good = np.isfinite(stamp) if nodata is None else (np.isfinite(stamp) & ~nodata)
    work = np.where(good, np.nan_to_num(stamp), 0.0)
    fine = _smooth(work, 1.5)
    heavy_fwhm = max(DOG_HEAVY_FWHM_MIN, 2.5 * seeing_arcsec)
    heavy = _smooth(work, heavy_fwhm / 2.355 / pixscale)
    dog = fine - DOG_FACTOR * heavy

    mask = np.zeros(stamp.shape, bool)
    segm = detect_sources(fine, threshold=2.0 * threshold_std,
                          n_pixels=npixels)
    if segm is not None:
        iy = int(np.clip(round(cy), 0, stamp.shape[0] - 1))
        ix = int(np.clip(round(cx), 0, stamp.shape[1] - 1))
        target_label = int(segm.data[iy, ix])
        if target_label:
            mask |= segm.data == target_label
        labels = np.array([lab for lab in np.unique(segm.data)
                           if lab > 0 and lab != target_label])
        if labels.size:
            peaks = ndimage.maximum(fine, labels=segm.data, index=labels)
            bright = labels[np.asarray(peaks)
                            > SKY_BRIGHT_SIGMA * threshold_std]
            if bright.size:
                mask |= binary_dilation(
                    np.isin(segm.data, bright),
                    iterations=max(2, int(round(2.0 / pixscale))))

    cores = dog > 2.0 * threshold_std
    if nodata is not None:
        cores &= ~nodata
    grow = max(1, int(round(seeing_arcsec / pixscale))) + 2
    mask |= binary_dilation(cores, iterations=grow)
    return mask


def nontarget_parents(
        stamp: np.ndarray,
        threshold_std: float,
        cx: float,
        cy: float,
        *,
        npixels: int = 8,
        nodata: np.ndarray | None = None,
) -> np.ndarray:
    """Detected segments that do not contain the target position.

    Used to clean the DIAGNOSTIC curve of growth beyond the aperture: an
    unmasked bright neighbor outside the aperture is (correctly) in no
    flux, but once the growing radius reaches it the curve jumps and the
    outer-slope sky witness reads as a false alarm. Ambiguous
    disconnected envelope islands are filled there too -- the curve
    under-shows envelope growth beyond the aperture, which is the
    acceptable direction for a sky diagnostic.

    Returns
    -------
    mask : np.ndarray (bool)
        True on every detected segment except the target's own.
    """
    work = stamp if nodata is None else np.where(nodata, 0.0, stamp)
    smoothed = _smooth(np.nan_to_num(work), 1.5)
    segm = detect_sources(smoothed, threshold=2.0 * threshold_std,
                          n_pixels=npixels)
    if segm is None:
        return np.zeros(stamp.shape, bool)
    iy = int(np.clip(round(cy), 0, stamp.shape[0] - 1))
    ix = int(np.clip(round(cx), 0, stamp.shape[1] - 1))
    label = int(segm.data[iy, ix])
    return (segm.data > 0) & (segm.data != label if label else True)


def _dog_regions(
        stamp: np.ndarray,
        threshold_std: float,
        cx: float,
        cy: float,
        pixscale: float,
        *,
        seeing_arcsec: float = 1.0,
        nodata: np.ndarray | None = None,
) -> tuple:
    """DoG dominance regions and the target's own label.

    The shared front half of the interloper machinery: neighbor_mask
    grows and floods these regions into the aperture mask, and
    interloper_cores reads off their peak positions as deblend centers.

    Returns
    -------
    labels, n_regions, target_label, fine, good : tuple
        Connected dominance regions, their count, the label under (or
        nearest within a seeing disk of) the target position -- 0 when
        the target has none -- and the fine-smoothed map with its
        validity mask, for reuse.
    """
    good = np.isfinite(stamp) if nodata is None else (np.isfinite(stamp) & ~nodata)
    work = np.where(good, np.nan_to_num(stamp), 0.0)
    fine = _smooth(work, 1.5)
    heavy_fwhm = max(DOG_HEAVY_FWHM_MIN, 2.5 * seeing_arcsec)
    heavy = _smooth(work, heavy_fwhm / 2.355 / pixscale)
    exceed = (fine - DOG_FACTOR * heavy) > 2.0 * threshold_std
    if nodata is not None:
        exceed &= ~nodata
    labels, n_regions = ndimage.label(exceed)
    target_label = 0
    if n_regions:
        iy = int(np.clip(round(cy), 0, stamp.shape[0] - 1))
        ix = int(np.clip(round(cx), 0, stamp.shape[1] - 1))
        target_label = labels[iy, ix]
        if target_label == 0:
            rr = radii_arcsec(stamp.shape, cx, cy, pixscale)
            near = (rr < max(seeing_arcsec, 2.0 * pixscale)) & exceed
            if near.any():
                candidates, counts = np.unique(labels[near], return_counts=True)
                target_label = int(candidates[np.argmax(counts)])
    return labels, n_regions, int(target_label), fine, good


def interloper_cores(
        stamp: np.ndarray,
        threshold_std: float,
        cx: float,
        cy: float,
        pixscale: float,
        *,
        npixels: int = 8,
        seeing_arcsec: float = 1.0,
        nodata: np.ndarray | None = None,
) -> list[tuple[float, float, float]]:
    """Interloper core peaks, brightest first: [(peak, py, px), ...].

    The deblend centers: every DoG dominance region that is not the
    target's own and not a sub-npixels speck, located at its
    fine-smoothed peak.
    """
    labels, n_regions, target_label, fine, _ = _dog_regions(
        stamp, threshold_std, cx, cy, pixscale,
        seeing_arcsec=seeing_arcsec, nodata=nodata)
    if n_regions == 0:
        return []
    index = np.arange(1, n_regions + 1)
    sizes = ndimage.sum(labels > 0, labels, index=index)
    peaks = ndimage.maximum(fine, labels=labels, index=index)
    positions = ndimage.maximum_position(fine, labels=labels, index=index)
    cores = [(float(peaks[k]), float(positions[k][0]), float(positions[k][1]))
             for k in range(n_regions)
             if (k + 1) != target_label and sizes[k] >= npixels]
    return sorted(cores, reverse=True)


def neighbor_mask(
        stamp: np.ndarray,
        threshold_std: float,
        cx: float,
        cy: float,
        pixscale: float,
        *,
        protect_radius: float = 4.0,
        npixels: int = 8,
        dilate: int = 2,
        seeing_arcsec: float = 1.0,
        nodata: np.ndarray | None = None,
        symmetry_excess: bool = True,
) -> np.ndarray:
    """Two-channel neighbor mask: DoG cores plus seeded symmetry excess.

    Channel 1 -- scale selection. The stamp is smoothed twice, at the
    PSF scale and at a heavy "diffuse" scale, and their scaled
    difference is the dominance map:

        dog = fine - DOG_FACTOR x heavy

    Diffuse structure lives equally in both smoothings and cancels, so
    the target's envelope -- asymmetric, patchy, or disconnected at any
    isophote -- can never mask itself. The connected dog region under
    the target position is the target's own core (exempt); every other
    region is an interloper CORE, masked and grown by a seeing disk.

    Channel 2 -- ownership by symmetry. Cores alone under-mask: a bright
    companion's wings extend far past one seeing disk, and a neighbor's
    diffuse envelope is invisible to the dog by construction. Both are
    caught by point-reflection through the target center: where the
    fine-smoothed image exceeds its own 180-degree twin by more than
    EXCESS_SIGMA x rms, the light cannot belong to a source centered on
    the target. Excess alone is not enough -- the target's own envelope
    may be genuinely asymmetric (lumpy or one-sided envelopes) -- so
    excess pixels are masked only where they are CONNECTED to a channel-1 core: each
    interloper's mask flood-fills outward from its core through the
    significant-excess region and stops where the target's symmetric
    light takes over. An asymmetric envelope with no interloper core
    inside it is never touched.

    Light below both channels (a companion's sub-threshold skirt) stays
    in the flux: masking errs toward keeping target light, the leak is
    bounded, and the QA metrics betray it.

    Parameters
    ----------
    stamp : np.ndarray
        Sky-subtracted stamp.
    threshold_std : float
        Background rms; the dominance exceedance threshold is
        2 x threshold_std, the symmetry-excess threshold
        EXCESS_SIGMA x threshold_std.
    cx, cy : float
        Stamp-pixel center of the target.
    pixscale : float
        Arcsec per pixel.
    protect_radius : float
        Radius (arcsec) never masked -- keeps historical override
        semantics (--protect-radius). A companion inside it stays in the
        flux with only the QA metrics to flag it.
        [default: 4.0]
    npixels : int
        Minimum connected pixels for a dominance region.
    dilate : int
        Extra mask growth (pixels) on top of the seeing disk.
    seeing_arcsec : float
        Band PSF FWHM; sets the fine/heavy scale split and the mask
        growth. [default: 1.0]
    nodata : np.ndarray (bool), optional
        Off-footprint / blank pixels; zeroed for the smoothings and never
        counted as interloper pixels (the caller handles their fill and
        coverage accounting).
    symmetry_excess : bool
        Enable channel 2. [default: True]

    Returns
    -------
    mask : np.ndarray (bool)
        True where a neighbor's light is (exclude from the aperture and
        fill from the target's own profile).
    """
    rr = radii_arcsec(stamp.shape, cx, cy, pixscale)
    labels, n_regions, target_label, fine, good = _dog_regions(
        stamp, threshold_std, cx, cy, pixscale,
        seeing_arcsec=seeing_arcsec, nodata=nodata)
    if n_regions == 0:
        return np.zeros(stamp.shape, bool)

    # Drop sub-npixels specks, exempt the target's own core.
    sizes = ndimage.sum(labels > 0, labels, index=np.arange(1, n_regions + 1))
    keep = [lab for lab in range(1, n_regions + 1)
            if lab != target_label and sizes[lab - 1] >= npixels]
    if not keep:
        return np.zeros(stamp.shape, bool)

    cores = np.isin(labels, keep)
    grow = dilate + max(1, int(round(seeing_arcsec / pixscale)))
    mask = binary_dilation(cores, iterations=grow)

    if symmetry_excess:
        # Point-reflect the fine map through the target center; a twin
        # sampled off the stamp or on blank pixels proves nothing, so
        # ownership is only claimed where the twin is real data. The
        # flood is additionally gated at the EXCESS_LIGHT_FLOOR
        # isophote: a wing worth masking has significant LIGHT as well
        # as significant asymmetry. Without the light floor, the excess
        # set percolates through correlated noise and field-scale
        # diffuse light (ICL), and one seed annexes half the stamp.
        yy, xx = np.indices(stamp.shape)
        coords = np.array([2.0 * cy - yy, 2.0 * cx - xx])
        twin = map_coordinates(fine, coords, order=1, mode='constant',
                               cval=0.0)
        twin_good = map_coordinates(good.astype(float), coords, order=1,
                                    mode='constant', cval=0.0) > 0.99
        excess = ((fine - twin > EXCESS_SIGMA * threshold_std)
                  & (fine > EXCESS_LIGHT_FLOOR * threshold_std)
                  & twin_good & good)
        if target_label:
            excess &= labels != target_label
        mask |= binary_propagation(cores & excess, mask=excess)

    if target_label:
        mask &= labels != target_label
    mask &= rr > protect_radius
    if nodata is not None:
        mask &= ~nodata
    return mask


# ------------------------------------
# User-supplied masks
# ------------------------------------
def load_user_mask(
        path: str | Path,
        ref_image: str | Path | None = None,
) -> tuple[np.ndarray, WCS | None]:
    """Load a user mask: (mask, grid WCS or None).

    Two formats:
      - .npz with a 'neighbor_mask' array. The archive also stores the
        x0/y0 stamp center on its reference image; pair it with ref_image
        to reconstruct the mask's WCS. Without one, the mask can only be
        applied to stamps sharing its exact grid.
      - FITS whose primary HDU is nonzero where masked, with its own WCS.
    """
    path = Path(path)
    if path.suffix == ".npz":
        archive = np.load(path)
        if "neighbor_mask" not in archive:
            raise ValueError(f"{path} has no 'neighbor_mask' array "
                             f"(keys: {sorted(archive.keys())})")
        mask = archive["neighbor_mask"].astype(bool)
        if ref_image is not None:
            from astropy.nddata import Cutout2D
            from .calibrate import load_image
            image, image_wcs, _ = load_image(str(ref_image))
            cut = Cutout2D(image, (int(archive["x0"]), int(archive["y0"])),
                           mask.shape, wcs=image_wcs)
            return mask, cut.wcs
        return mask, None
    with fits.open(path) as hdul:
        mask = np.asarray(hdul[0].data) != 0
        wcs = WCS(hdul[0].header) if hdul[0].header.get("CTYPE1") else None
    return mask, wcs


def reproject_mask(
        stamp_wcs: WCS,
        shape: tuple,
        mask_wcs: WCS,
        mask: np.ndarray,
) -> np.ndarray:
    """Nearest-pixel reprojection of a boolean mask onto a stamp's grid."""
    yy, xx = np.indices(shape)
    mx, my = mask_wcs.world_to_pixel(stamp_wcs.pixel_to_world(xx, yy))
    mxr, myr = np.round(mx).astype(int), np.round(my).astype(int)
    inb = (mxr >= 0) & (mxr < mask.shape[1]) & (myr >= 0) & (myr < mask.shape[0])
    out = np.zeros(shape, bool)
    out[inb] = mask[myr[inb], mxr[inb]]
    return out
