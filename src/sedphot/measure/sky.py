"""
sky.py

Annulus Sky Estimation
---------------------------------------------------------

Sigma-clipped sky from a source-masked annulus, with matched-filter
rejection of bright sources inside the annulus, and the radial sky
profile used for QA.

Requirements:
    numpy, scipy, astropy

Notes:
    The sky annulus is the dominant lever on the absolute flux level of an
    extended source; it must clear the galaxy's envelope. The radial
    profile is the tool that shows whether it does.
"""
from __future__ import annotations

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import gaussian_filter, maximum_filter

from .masks import radii_arcsec, source_mask


# ------------------------------------
# Sky measurement
# ------------------------------------
def annulus_source_mask(
        stamp: np.ndarray,
        cx: float,
        cy: float,
        pixscale: float,
        *,
        sky_in: float,
        sky_out: float,
        seeing_arcsec: float = 1.0,
        nodata: np.ndarray | None = None,
) -> np.ndarray:
    """Matched-filter detection of bright sources in the sky annulus.

    Smooths at the PSF scale, finds local maxima above 4 sigma, and masks a
    4 arcsec disk at each peak that falls in (or just outside) the annulus.

    Returns
    -------
    mask : np.ndarray (bool)
        True where an annulus source is masked.
    """
    yy, xx = np.indices(stamp.shape)
    work = stamp
    if nodata is not None:
        work = np.where(nodata, 0.0, stamp)
    finite = work[np.isfinite(work)]
    level, _, _ = sigma_clipped_stats(finite, sigma=3.0)
    smoothed = gaussian_filter(np.nan_to_num(work - level),
                               max(0.6, seeing_arcsec / 2.355 / pixscale))
    _, _, smooth_std = sigma_clipped_stats(smoothed, sigma=3.0)
    peaks = ((smoothed == maximum_filter(smoothed, size=int(round(3 / pixscale))))
             & (smoothed / smooth_std > 4))
    if nodata is not None:
        peaks &= ~nodata
    py, px = np.where(peaks)
    mask = np.zeros(stamp.shape, bool)
    for j in range(len(px)):
        if sky_in - 4 < np.hypot(px[j] - cx, py[j] - cy) * pixscale < sky_out + 2:
            mask |= (np.hypot(xx - px[j], yy - py[j]) < 4 / pixscale)
    return mask


def annulus_sky(
        stamp: np.ndarray,
        cx: float,
        cy: float,
        pixscale: float,
        *,
        sky_in: float,
        sky_out: float,
        seeing_arcsec: float = 1.0,
        nodata: np.ndarray | None = None,
        extra_mask: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray]:
    """Sigma-clipped sky level and rms from a source-masked annulus.

    Parameters
    ----------
    stamp : np.ndarray
        Image stamp (not yet sky-subtracted).
    cx, cy : float
        Stamp-pixel center.
    pixscale : float
        Arcsec per pixel.
    sky_in, sky_out : float
        Annulus radii in arcsec.
    seeing_arcsec : float
        PSF FWHM for the matched-filter source rejection. [default: 1.0]
    nodata : np.ndarray (bool), optional
        Off-footprint / blank pixels, excluded from every statistic. In a
        sky-subtracted archive image blank zeros sit exactly at the
        expected sky level, so the clip cannot reject them -- left in,
        they drag the median toward zero.
    extra_mask : np.ndarray (bool), optional
        Additional exclusion (e.g. the full segmentation mask on a second
        pass); the matched-filter peak rejection alone leaves the faint
        sources and bright-neighbor wings that bias a deep crowded
        annulus high.

    Returns
    -------
    sky_level : float
        Sigma-clipped median of the source-free annulus (image units/pixel).
    sky_std : float
        Sigma-clipped std (per-pixel background rms).
    annulus_mask : np.ndarray (bool)
        The source mask that was applied inside the annulus.
    """
    rr = radii_arcsec(stamp.shape, cx, cy, pixscale)
    srcmask = annulus_source_mask(stamp, cx, cy, pixscale,
                                  sky_in=sky_in, sky_out=sky_out,
                                  seeing_arcsec=seeing_arcsec, nodata=nodata)
    if extra_mask is not None:
        srcmask = srcmask | extra_mask
    region = (rr > sky_in) & (rr < sky_out) & ~srcmask & np.isfinite(stamp)
    if nodata is not None:
        region &= ~nodata
    # A pathological mask can empty the annulus; sigma_clipped_stats on an
    # empty array would return NaN and silently zero every flux downstream.
    if region.sum() < 50:
        raise ValueError(
            f"sky annulus {sky_in:g}-{sky_out:g}\" has only "
            f"{int(region.sum())} usable pixels after masking")
    sky_level, _, sky_std = sigma_clipped_stats(stamp[region], sigma=3.0)
    return float(sky_level), float(sky_std), srcmask


def radial_sky_profile(
        stamp: np.ndarray,
        cx: float,
        cy: float,
        pixscale: float,
        edges,
        *,
        srcmask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Source-masked, sigma-clipped sky in radial bins (QA).

    Parameters
    ----------
    stamp : np.ndarray
        Image stamp.
    cx, cy : float
        Stamp-pixel center.
    pixscale : float
        Arcsec per pixel.
    edges : sequence of float
        Bin edges in arcsec.
    srcmask : np.ndarray, optional
        Precomputed source mask; derived from the outer bin if None.

    Returns
    -------
    centers : np.ndarray
        Bin centers (arcsec).
    sky : np.ndarray
        Sigma-clipped median per bin (image units / pixel).
    """
    rr = radii_arcsec(stamp.shape, cx, cy, pixscale)
    if srcmask is None:
        _, _, outer_std = sigma_clipped_stats(
            stamp[(rr > edges[-2]) & (rr < edges[-1])], sigma=3.0)
        srcmask = source_mask(stamp, outer_std)
    centers, sky = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        region = (rr >= lo) & (rr < hi) & ~srcmask & np.isfinite(stamp)
        value = sigma_clipped_stats(stamp[region], sigma=3.0)[0] if region.sum() > 20 else np.nan
        centers.append(0.5 * (lo + hi))
        sky.append(value)
    return np.array(centers), np.array(sky)
