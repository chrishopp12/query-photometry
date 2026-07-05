"""
masks.py

Neighbor Masks for Aperture Photometry
---------------------------------------------------------
The auto-mask: a deblended detection on the stamp that protects the target
(every segment touching within protect_radius of the center is kept) and
masks everything else. Deblending is what keeps a neighbor that touches the
target's envelope from being absorbed into the target's segment -- without
it, masking the neighbor would gouge the galaxy.

Ported from a1925_nbcg/photometry/forced_phot.py (source_mask,
clean_neighbor_mask) and uniform_phot.py (_reproject_mask), plus the
user-supplied mask loader (--mask FILE) that replaces the A1925 staged
morphology masks.

Requirements:
    numpy, scipy, astropy, photutils

Notes:
    Masks are True where a pixel is EXCLUDED.
    A user mask may live on a different pixel grid than the band being
    measured; reproject_mask moves it by WCS, so the native grid is
    immaterial (the A1925 convention).
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
def source_mask(stamp: np.ndarray, threshold_std: float, *,
                threshold_map: np.ndarray | None = None,
                npixels: int = 8, nlevels: int = 32, contrast: float = 0.001,
                dilate: int = 2) -> np.ndarray:
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

    Returns
    -------
    mask : np.ndarray (bool)
        True where a source is detected.
    """
    smoothed = convolve(stamp, Gaussian2DKernel(1.5))
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


def neighbor_mask(stamp: np.ndarray, threshold_std: float, cx: float, cy: float,
                  pixscale: float, *, protect_radius: float = 4.0,
                  npixels: int = 8, dilate: int = 2, n_iter: int = 2) -> np.ndarray:
    """Profile-residual neighbor mask that protects the target's own light.

    Subtracts the target's azimuthal median profile (a circular, model-free
    "model" of the galaxy) and runs the deblended detection on the RESIDUAL,
    so neighbors -- including sources embedded in the envelope -- are
    detected while the smooth target itself is not. A plain deblend on the
    data shreds a bright extended envelope into segments and masks the
    galaxy's own light (a 10%-level flux bias observed on the A1925 BCG);
    the residual approach is the general, morphology-free equivalent of the
    A1925 staged model masks.

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
        Core radius (arcsec) never masked -- absorbs residual core
        mismatch of the circular profile. [default: 4.0]
    n_iter : int
        Profile/mask iterations (the profile is remeasured with the mask
        applied). [default: 2]

    Returns
    -------
    mask : np.ndarray (bool)
        True where a neighbor is (exclude from sky and aperture).
    """
    rr = radii_arcsec(stamp.shape, cx, cy, pixscale)
    yy, xx = np.indices(stamp.shape)
    mask = np.zeros(stamp.shape, bool)

    for _ in range(n_iter):
        # Elliptical geometry from flux-weighted second moments of the
        # (unmasked) target light -- a circular profile leaves a quadrupole
        # residual on an elliptical galaxy that false-detects as neighbors.
        weight_sel = (rr < min(30.0, float(rr.max()) * 0.5)) & ~mask & np.isfinite(stamp)
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
            sel = (bin_index == i) & ~mask & np.isfinite(stamp)
            if sel.sum() > 5:
                profile[i] = np.median(stamp[sel])
        model = profile[bin_index]
        residual = stamp - model
        # Local-brightness floor: inside the galaxy-dominated region a
        # residual must also beat half the local profile level -- compact
        # neighbors do, the envelope's own lumpy substructure does not
        # (deep imaging detects that substructure at any sky-based
        # threshold, and masking it biases the flux low).
        mask = source_mask(residual, threshold_std,
                           threshold_map=0.5 * np.clip(model, 0, None),
                           npixels=npixels, dilate=dilate)
        mask &= rr > protect_radius

    return mask


# ------------------------------------
# User-supplied masks
# ------------------------------------
def load_user_mask(path: str | Path,
                   ref_image: str | Path | None = None) -> tuple[np.ndarray, WCS | None]:
    """Load a user mask: (mask, grid WCS or None).

    Two formats:
      - .npz with a 'neighbor_mask' array (the A1925 staged-mask format,
        which also stores the x0/y0 stamp center on its reference image).
        Pair it with ref_image to reconstruct the mask's WCS (the
        uniform_phot load_mask recipe); without one it can only be applied
        to stamps sharing its exact grid.
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


def reproject_mask(stamp_wcs: WCS, shape: tuple, mask_wcs: WCS,
                   mask: np.ndarray) -> np.ndarray:
    """Nearest-pixel reprojection of a boolean mask onto a stamp's grid."""
    yy, xx = np.indices(shape)
    mx, my = mask_wcs.world_to_pixel(stamp_wcs.pixel_to_world(xx, yy))
    mxr, myr = np.round(mx).astype(int), np.round(my).astype(int)
    inb = (mxr >= 0) & (mxr < mask.shape[1]) & (myr >= 0) & (myr < mask.shape[0])
    out = np.zeros(shape, bool)
    out[inb] = mask[myr[inb], mxr[inb]]
    return out
