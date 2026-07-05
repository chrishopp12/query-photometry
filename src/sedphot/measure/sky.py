"""
sky.py

Annulus Sky Estimation
---------------------------------------------------------
Sigma-clipped sky from a source-masked annulus, with the matched-filter
bright-source rejection used inside the annulus, and the radial sky profile
QA. Ported from a1925_nbcg/photometry/forced_phot.py (clean_sky,
radial_sky_profile) and uniform_phot.py (the matched-filter annulus
detector inside measure()).

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
def annulus_source_mask(stamp: np.ndarray, cx: float, cy: float, pixscale: float,
                        *, sky_in: float, sky_out: float,
                        seeing_arcsec: float = 1.0) -> np.ndarray:
    """Matched-filter detection of bright sources in the sky annulus.

    Smooths at the PSF scale, finds local maxima above 4 sigma, and masks a
    4 arcsec disk at each peak that falls in (or just outside) the annulus.

    Returns
    -------
    mask : np.ndarray (bool)
        True where an annulus source is masked.
    """
    yy, xx = np.indices(stamp.shape)
    level, _, _ = sigma_clipped_stats(stamp, sigma=3.0)
    smoothed = gaussian_filter(stamp - level, max(0.6, seeing_arcsec / 2.355 / pixscale))
    _, _, smooth_std = sigma_clipped_stats(smoothed, sigma=3.0)
    peaks = ((smoothed == maximum_filter(smoothed, size=int(round(3 / pixscale))))
             & (smoothed / smooth_std > 4))
    py, px = np.where(peaks)
    mask = np.zeros(stamp.shape, bool)
    for j in range(len(px)):
        if sky_in - 4 < np.hypot(px[j] - cx, py[j] - cy) * pixscale < sky_out + 2:
            mask |= (np.hypot(xx - px[j], yy - py[j]) < 4 / pixscale)
    return mask


def annulus_sky(stamp: np.ndarray, cx: float, cy: float, pixscale: float, *,
                sky_in: float, sky_out: float,
                seeing_arcsec: float = 1.0) -> tuple[float, float, np.ndarray]:
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

    Returns
    -------
    sky_level : float
        Sigma-clipped median of the source-free annulus (image units/pixel).
    sky_std : float
        Sigma-clipped std (per-pixel background rms).
    annulus_mask : np.ndarray (bool)
        The bright-source mask that was applied inside the annulus.
    """
    rr = radii_arcsec(stamp.shape, cx, cy, pixscale)
    srcmask = annulus_source_mask(stamp, cx, cy, pixscale,
                                  sky_in=sky_in, sky_out=sky_out,
                                  seeing_arcsec=seeing_arcsec)
    region = (rr > sky_in) & (rr < sky_out) & ~srcmask & np.isfinite(stamp)
    sky_level, _, sky_std = sigma_clipped_stats(stamp[region], sigma=3.0)
    return float(sky_level), float(sky_std), srcmask


def radial_sky_profile(stamp: np.ndarray, cx: float, cy: float, pixscale: float,
                       edges, *, srcmask: np.ndarray | None = None,
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Source-masked, sigma-clipped sky in radial bins (QA).

    Parameters
    ----------
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
