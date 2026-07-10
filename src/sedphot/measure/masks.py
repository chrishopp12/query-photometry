"""
masks.py

Neighbor Masks for Aperture Photometry
---------------------------------------------------------

Boolean masks that keep neighboring sources out of the sky annulus and
the aperture: a deblended source detection (source_mask), structural
pixel ownership of the target's own segment (target_segment), the
two-channel neighbor mask built on both (neighbor_mask), and
user-supplied masks loaded from disk and moved between pixel grids by
WCS (load_user_mask, reproject_mask).

Ownership is structural, not radial: a detected segment that is not the
target's is a neighbor wherever it sits -- inside the photometry
aperture included -- while the target's own segment is never masked, no
matter how asymmetric its envelope. Only sources whose isophotes merge
with the target need the model-dependent residual channel.

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


def target_segment(
        stamp: np.ndarray,
        threshold_std: float,
        cx: float,
        cy: float,
        *,
        npixels: int = 8,
        nodata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Structural pixel ownership: the target's segment vs everyone else's.

    Segments the lightly smoothed stamp at 2 x threshold_std WITHOUT
    deblending: a connected envelope is one segment, so the target cannot
    be shredded into pieces here. The segment under the target position is
    the target's; every other segment belongs to a neighbor.

    Returns
    -------
    (target, neighbors) : tuple of np.ndarray (bool)
        Pixels of the target's own segment, and pixels of every other
        segment (undilated). Both all-False when nothing is detected.
    """
    work = stamp if nodata is None else np.where(nodata, 0.0, stamp)
    smoothed = convolve(np.nan_to_num(work), Gaussian2DKernel(1.5))
    segm = detect_sources(smoothed, threshold=2.0 * threshold_std,
                          n_pixels=npixels)
    if segm is None:
        return (np.zeros(stamp.shape, bool), np.zeros(stamp.shape, bool))
    iy = int(np.clip(round(cy), 0, stamp.shape[0] - 1))
    ix = int(np.clip(round(cx), 0, stamp.shape[1] - 1))
    label = int(segm.data[iy, ix])
    target = segm.data == label if label else np.zeros(stamp.shape, bool)
    neighbors = (segm.data > 0) & (segm.data != label if label else True)
    return target, neighbors


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
        n_iter: int = 2,
        nodata: np.ndarray | None = None,
) -> np.ndarray:
    """Neighbor mask with structural target/neighbor ownership.

    Two channels, by where a source sits:

    - OUTSIDE the target's segment: every other detected segment is a
      neighbor and is always masked -- radius is irrelevant, so a
      companion inside the photometry aperture is masked while the
      target's own envelope, however far it reaches, never is. (Deciding
      this radially via protect_radius either ate asymmetric envelopes or
      protected real companions, depending on which way it was set.)
    - INSIDE the target's segment: a companion whose isophotes merge with
      the target cannot be separated structurally without risking
      shredding the envelope. There an elliptical-annulus median model of
      the target is subtracted and the deblended detection runs on the
      RESIDUAL, with a local-brightness floor so the envelope's own lumpy
      substructure does not false-detect.

    Parameters
    ----------
    stamp : np.ndarray
        Sky-subtracted stamp.
    threshold_std : float
        Background rms; detection threshold is 2 x threshold_std.
    cx, cy : float
        Stamp-pixel center of the target.
    pixscale : float
        Arcsec per pixel.
    protect_radius : float
        Core radius (arcsec) the RESIDUAL channel never masks -- absorbs
        core mismatch of the smooth profile. Structural neighbors are
        masked regardless. [default: 4.0]
    npixels, dilate : int
        Passed through to source_mask.
    n_iter : int
        Profile/mask iterations (the profile is remeasured with the mask
        applied). [default: 2]
    nodata : np.ndarray (bool), optional
        Off-footprint / blank pixels; excluded from the moments and the
        profile, and never counted as neighbor pixels (the caller handles
        their fill and coverage accounting).

    Returns
    -------
    mask : np.ndarray (bool)
        True where a neighbor is (exclude from sky and aperture).

    Notes
    -----
    Any smooth-profile model has limits: lumpy structure no elliptical
    profile can follow (e.g. a lensed arc) still leaves residuals that
    detect as spurious neighbors inside the target's segment. Supply a
    user mask for such targets. A companion within protect_radius of the
    center remains unseparable (the c41/c53/c58 class): its light is in
    the flux and only the QA metrics flag it.
    """
    rr = radii_arcsec(stamp.shape, cx, cy, pixscale)
    yy, xx = np.indices(stamp.shape)
    good = np.isfinite(stamp) if nodata is None else (np.isfinite(stamp) & ~nodata)

    # Channel 1 -- structural: every segment that is not the target's.
    target_seg, neighbor_seg = target_segment(
        stamp, threshold_std, cx, cy, npixels=npixels, nodata=nodata)
    structural = binary_dilation(neighbor_seg, iterations=dilate)
    structural &= ~target_seg  # dilation must not creep onto the target

    # Channel 2 -- residual detection inside the target's segment.
    mask = structural.copy()
    # Moments over the target's own segment, not a blind circular region:
    # sky noise clipped positive across a 30" disk has enormous, perfectly
    # circular second moments, so including it dilutes and circularizes the
    # estimated ellipse -- and a wrong ellipse leaves a quadrupole residual
    # on the envelope that beats the local floor and masks real light.
    moment_region = target_seg if target_seg.any() \
        else (rr < min(30.0, float(rr.max()) * 0.5))
    for _ in range(n_iter):
        # Elliptical geometry from flux-weighted second moments of the
        # (unmasked) target light -- a circular profile leaves a quadrupole
        # residual on an elliptical galaxy that false-detects as neighbors.
        weight_sel = moment_region & ~mask & good
        weights = np.where(weight_sel, np.clip(stamp, 0, None), 0.0)
        total = weights.sum()
        if total > 0:
            dx, dy = xx - cx, yy - cy
            mxx = float((weights * dx * dx).sum() / total)
            myy = float((weights * dy * dy).sum() / total)
            mxy = float((weights * dx * dy).sum() / total)
            theta = 0.5 * np.arctan2(2 * mxy, mxx - myy)
            trace = mxx + myy
            det = np.sqrt(max((mxx - myy) ** 2 + 4 * mxy ** 2, 0.0))
            a2, b2 = (trace + det) / 2, max((trace - det) / 2, 1e-12)
            axis_ratio = float(np.clip(np.sqrt(b2 / a2), 0.3, 1.0))
        else:
            theta, axis_ratio = 0.0, 1.0
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = (xx - cx) * cos_t + (yy - cy) * sin_t
        y_rot = -(xx - cx) * sin_t + (yy - cy) * cos_t
        r_ell = np.hypot(x_rot, y_rot / axis_ratio) * pixscale

        # Elliptical-annulus median profile, then detect on the residual.
        edges = np.arange(0.0, float(r_ell.max()) + 2.0, 1.0)
        bin_index = np.clip(np.digitize(r_ell, edges) - 1, 0, len(edges) - 2)
        profile = np.zeros(len(edges) - 1)
        for i in range(len(edges) - 1):
            sel = (bin_index == i) & ~mask & good
            if sel.sum() > 5:
                profile[i] = np.median(stamp[sel])
        model = profile[bin_index]
        residual = np.where(good, stamp - model, 0.0)
        # Local-brightness floor: inside the galaxy-dominated region a
        # residual must also beat half the local profile level -- compact
        # neighbors do, the envelope's own lumpy substructure does not
        # (deep imaging detects that substructure at any sky-based
        # threshold, and masking it biases the flux low).
        embedded = source_mask(residual, threshold_std,
                               threshold_map=0.5 * np.clip(model, 0, None),
                               npixels=npixels, dilate=dilate, nodata=nodata)
        embedded &= target_seg          # channel 1 owns everything outside
        embedded &= rr > protect_radius
        mask = structural | embedded

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
