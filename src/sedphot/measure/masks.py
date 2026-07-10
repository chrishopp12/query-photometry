"""
masks.py

Neighbor Masks for Aperture Photometry
---------------------------------------------------------

Boolean masks that keep neighboring sources out of the sky annulus and
the aperture: a deblended source detection for cleaning the sky
(source_mask), a difference-of-Gaussians interloper mask for the
aperture (neighbor_mask), and user-supplied masks loaded from disk and
moved between pixel grids by WCS (load_user_mask, reproject_mask).

The aperture mask is scale-selective, not model-based: only sources with
PSF-scale power above their own local diffuse level are interlopers, so
the target's envelope -- asymmetric, patchy, or disconnected at any
isophote -- can never be masked as its own neighbor, at any radius.

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
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.wcs import WCS
from photutils.segmentation import deblend_sources, detect_sources
from scipy.ndimage import binary_dilation


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
    smoothed = convolve(np.nan_to_num(work), Gaussian2DKernel(1.5))
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
# one-sided envelopes (the failure that ate c36/c38).
DOG_FACTOR = 1.5

# The heavy smoothing scale (FWHM, arcsec) that defines "diffuse": at
# least 3", or 2.5 seeing when the PSF is fat.
DOG_HEAVY_FWHM_MIN = 3.0


# A source whose smoothed peak clears this many detection sigmas is
# masked in FULL (segment + wings dilation) from the sky region: its
# skirt is individually significant structure, not ambient background.
SKY_BRIGHT_SIGMA = 10.0


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
    from scipy import ndimage

    good = np.isfinite(stamp) if nodata is None else (np.isfinite(stamp) & ~nodata)
    work = np.where(good, np.nan_to_num(stamp), 0.0)
    fine = convolve(work, Gaussian2DKernel(1.5))
    heavy_fwhm = max(DOG_HEAVY_FWHM_MIN, 2.5 * seeing_arcsec)
    heavy = convolve(work, Gaussian2DKernel(heavy_fwhm / 2.355 / pixscale))
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
    smoothed = convolve(np.nan_to_num(work), Gaussian2DKernel(1.5))
    segm = detect_sources(smoothed, threshold=2.0 * threshold_std,
                          n_pixels=npixels)
    if segm is None:
        return np.zeros(stamp.shape, bool)
    iy = int(np.clip(round(cy), 0, stamp.shape[0] - 1))
    ix = int(np.clip(round(cx), 0, stamp.shape[1] - 1))
    label = int(segm.data[iy, ix])
    return (segm.data > 0) & (segm.data != label if label else True)


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
) -> np.ndarray:
    """Difference-of-Gaussians neighbor mask: concentrated interlopers only.

    The stamp is smoothed twice -- at the PSF scale and at a heavy
    "diffuse" scale -- and their scaled difference is the dominance map:

        dog = fine - DOG_FACTOR x heavy

    Diffuse structure lives equally in both smoothings and cancels, so
    the target's envelope -- asymmetric, patchy, or disconnected at any
    isophote -- can never mask itself. Concentrated sources survive the
    difference; the connected dog region under the target position is
    the target's own core (exempt), every other region is an interloper
    and is masked, grown by roughly a seeing disk to cover its skirt.
    Light beyond that growth (a bright companion's outer wings) stays in
    the flux: masking errs toward keeping target light, the leak is
    bounded, and the QA metrics betray it.

    Parameters
    ----------
    stamp : np.ndarray
        Sky-subtracted stamp.
    threshold_std : float
        Background rms; the dominance exceedance threshold is
        2 x threshold_std.
    cx, cy : float
        Stamp-pixel center of the target.
    pixscale : float
        Arcsec per pixel.
    protect_radius : float
        Radius (arcsec) never masked -- keeps historical override
        semantics (--protect-radius). A companion inside it stays in the
        flux (the c41/c53/c58 class) with only the QA metrics to flag it.
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

    Returns
    -------
    mask : np.ndarray (bool)
        True where a concentrated interloper is (exclude from sky and
        aperture).
    """
    from scipy import ndimage

    rr = radii_arcsec(stamp.shape, cx, cy, pixscale)
    good = np.isfinite(stamp) if nodata is None else (np.isfinite(stamp) & ~nodata)
    work = np.where(good, np.nan_to_num(stamp), 0.0)

    fine = convolve(work, Gaussian2DKernel(1.5))
    heavy_fwhm = max(DOG_HEAVY_FWHM_MIN, 2.5 * seeing_arcsec)
    heavy_sigma_px = heavy_fwhm / 2.355 / pixscale
    heavy = convolve(work, Gaussian2DKernel(heavy_sigma_px))
    dog = fine - DOG_FACTOR * heavy

    exceed = dog > 2.0 * threshold_std
    if nodata is not None:
        exceed &= ~nodata
    labels, n_regions = ndimage.label(exceed)
    if n_regions == 0:
        return np.zeros(stamp.shape, bool)

    # Drop sub-npixels specks, exempt the target's own core: the region
    # under (or nearest within a seeing disk of) the target position.
    sizes = ndimage.sum(exceed, labels, index=np.arange(1, n_regions + 1))
    iy = int(np.clip(round(cy), 0, stamp.shape[0] - 1))
    ix = int(np.clip(round(cx), 0, stamp.shape[1] - 1))
    target_label = labels[iy, ix]
    if target_label == 0:
        near = (rr < max(seeing_arcsec, 2.0 * pixscale)) & exceed
        if near.any():
            candidates, counts = np.unique(labels[near], return_counts=True)
            target_label = int(candidates[np.argmax(counts)])
    keep = [lab for lab in range(1, n_regions + 1)
            if lab != target_label and sizes[lab - 1] >= npixels]
    if not keep:
        return np.zeros(stamp.shape, bool)

    mask = np.isin(labels, keep)
    grow = dilate + max(1, int(round(seeing_arcsec / pixscale)))
    mask = binary_dilation(mask, iterations=grow)
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
