"""
psf.py

PSF Resolution: Empirical Star Kernel, Moffat Fallback
---------------------------------------------------------
Resolve each band's PSF kernel and FWHM, empirical-first: the brightest
usable confirmed star on the stamp gives a circularized ring-median
profile, with analytic Moffat wings grafted on where the measured rings
fall below the wing S/N threshold. Only when no star qualifies does the
band fall back to an analytic Moffat sized by the resolved seeing -- a
beta=3 Moffat is measurably too soft through the core against real
survey PSFs, and a too-soft kernel rings every bright core on
subtraction.

Requirements:
    numpy, pandas, astropy

Notes:
    Kernels are unit-sum, square, odd-sized, and rendered at the stamp
    pixel scale; FWHMs are arcsec. Every resolver also returns a
    provenance string naming the kernel's source (star, catalog column,
    header keyword, or fallback) for the measurement record. Star
    confirmation happens upstream: stars arrives here as a DataFrame
    with columns ra, dec, phot_g_mean_mag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from . import recipe
from .sersic import MOFFAT_BETA, moffat_psf
from .stamp import Stamp


# ------------------------------------
# Implementation constants
# ------------------------------------
EDGE_MARGIN_AS = 3.0          # a star's CORE must sit this far inside the stamp

BG_ANNULUS_AS = (9.0, 13.0)   # local background annulus about the star
BG_ANNULUS_MIN_PX = 200       # annulus pixels required for its median

# Profile-ring schedule: pixel-aware core rings (never sub-pixel or
# narrower than the floor), then fixed-width wing rings to the end.
RING_STEP_PX = 2.0            # core ring width (pixels)
RING_STEP_MIN_AS = 0.5        # core ring width floor (arcsec)
RING_CORE_END_AS = 3.0        # core rings end here; wing rings begin
RING_WING_STEP_AS = 0.5       # wing ring width (arcsec)
RING_END_AS = 10.5            # schedule stop (last ring edge at 10.0)

MIN_FINITE_RINGS = 12         # measured rings required for a usable profile
PEAK_MIN_SIGMA = 20.0         # star peak floor, x the stamp's global sigma
FWHM_SANITY_AS = (0.4, 3.5)   # plausible measured-FWHM window
GRAFT_MAX_AS = 8.0            # graft only when rings fail inside this radius
KERNEL_MIN_PX = 25            # kernel size floor (odd pixels)


# ------------------------------------
# Seeing and the fallback kernel
# ------------------------------------
def resolve_seeing(
        cat: pd.DataFrame | None,
        header,
        *,
        psfsize_col: str | None = None,
        fallback_arcsec: float = 1.0,
        fallback_label: str = 'provider default',
) -> tuple[float, str]:
    """Resolve the band seeing FWHM; first hit wins.

    The chain: median of the catalog's own per-source PSF-size column
    (when the caller names one), then a survey seeing keyword in the
    image header, then the caller's fallback.

    Parameters
    ----------
    cat : pd.DataFrame or None
        Survey catalog of the field.
    header : astropy.io.fits.Header or None
        Image header, searched for seeing keywords.
    psfsize_col : str, optional
        Catalog column holding per-source PSF FWHM (arcsec); None skips
        the catalog step. [default: None]
    fallback_arcsec : float
        Seeing when nothing else answers. [default: 1.0]
    fallback_label : str
        Provenance label for the fallback. [default: 'provider default']

    Returns
    -------
    seeing : tuple
        (fwhm_arcsec, provenance).
    """
    if psfsize_col is not None and cat is not None and psfsize_col in cat:
        value = float(np.nanmedian(cat[psfsize_col]))
        if np.isfinite(value) and value > 0:
            return value, f'catalog median {psfsize_col}'
    for key in ('FINALIQ', 'IQFINAL', 'SEEING'):
        if header is not None and key in header:
            try:
                value = float(header[key])
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and 0.2 < value < 5.0:   # plausible values only
                return value, f'header {key}'
    return fallback_arcsec, fallback_label


def moffat_kernel(seeing_arcsec: float, pixscale: float) -> np.ndarray:
    """Unit-sum Moffat kernel sized by the seeing (no fixed-size truncation)."""
    fwhm_px = seeing_arcsec / pixscale
    size = int(round(recipe.MOFFAT_KERNEL_FWHM * fwhm_px)) | 1   # odd
    return moffat_psf(seeing_arcsec, pixscale, size=max(size, KERNEL_MIN_PX))


# ------------------------------------
# Empirical star kernel
# ------------------------------------
def empirical_psf(stamp: Stamp, stars: pd.DataFrame) -> tuple[np.ndarray, float, str] | None:
    """Measure the PSF from the brightest usable confirmed star.

    Candidates are confirmed stars inside the PSF_STAR_GMAG window whose
    center sits at least EDGE_MARGIN_AS inside the stamp -- only the
    core must be fully on-stamp, because the kernel is built from
    circularized ring medians, which tolerate partial outer rings.
    Brightest first, each candidate must pass every guard: a populated
    background annulus, enough measured rings, a monotone bright core,
    and a plausible FWHM. Rings below the wing S/N threshold hand off
    to a grafted Moffat continuation.

    Parameters
    ----------
    stamp : Stamp
        The prepared stamp.
    stars : pd.DataFrame
        Confirmed stars with columns ra, dec, phot_g_mean_mag.

    Returns
    -------
    psf : tuple or None
        (kernel, fwhm_arcsec, provenance) from the first star to pass
        every guard; None when no star qualifies.
    """
    data = stamp.data
    pixscale = stamp.pixscale
    good = stamp.good
    yy, xx = np.indices(data.shape)

    # Candidate stars: the G window first, then the edge-margin geometry.
    edge = EDGE_MARGIN_AS / pixscale
    candidates = []
    for _, star in stars.iterrows():
        gmag = float(star.get('phot_g_mean_mag', np.nan))
        if not (recipe.PSF_STAR_GMAG[0] <= gmag <= recipe.PSF_STAR_GMAG[1]):
            continue
        sx, sy = [float(v) for v in stamp.wcs.world_to_pixel(
            SkyCoord(float(star['ra']), float(star['dec']), unit='deg'))]
        if not (edge < sx < data.shape[1] - edge
                and edge < sy < data.shape[0] - edge):
            continue
        candidates.append((gmag, sx, sy))
    candidates.sort()

    for gmag, sx, sy in candidates:
        rr_star = np.hypot(yy - sy, xx - sx) * pixscale

        # Local background from an annulus around the star itself.
        bg_ring = good & (rr_star > BG_ANNULUS_AS[0]) & (rr_star < BG_ANNULUS_AS[1])
        if bg_ring.sum() < BG_ANNULUS_MIN_PX:
            continue
        background = float(np.median(data[bg_ring]))

        # Ring-median profile on the pixel-aware schedule; a ring votes
        # only with at least 3 pixels, and gaps are interpolated over.
        step = max(RING_STEP_MIN_AS, RING_STEP_PX * pixscale)
        edges = np.concatenate([np.arange(0.0, RING_CORE_END_AS, step),
                                np.arange(RING_CORE_END_AS, RING_END_AS,
                                          RING_WING_STEP_AS)])
        mids = 0.5 * (edges[1:] + edges[:-1])
        prof = np.full(len(mids), np.nan)
        ring_n = np.zeros(len(mids))
        for i in range(len(mids)):
            ring = good & (rr_star >= edges[i]) & (rr_star < edges[i + 1])
            ring_n[i] = ring.sum()
            if ring.sum() >= 3:
                prof[i] = np.median(data[ring]) - background
        if not np.isfinite(prof[0]) or np.isfinite(prof).sum() < MIN_FINITE_RINGS:
            continue
        finite = np.isfinite(prof)
        prof = np.interp(mids, mids[finite], prof[finite])

        # Saturated / blended cores are non-monotone at the center;
        # compare against the mean of the next two rings so single
        # noisy rings cannot veto a good star.
        if (np.mean(prof[1:3]) > prof[0]
                or prof[0] < PEAK_MIN_SIGMA * stamp.sigma):
            continue
        prof = np.clip(np.minimum.accumulate(prof), 0.0, None)
        peak = float(prof[0])
        prof = prof / prof[0]
        fwhm = 2.0 * float(np.interp(0.5, prof[::-1], mids[::-1]))
        if not (FWHM_SANITY_AS[0] < fwhm < FWHM_SANITY_AS[1]):
            continue

        # Hybrid wings: rings below the wing S/N threshold are noise,
        # and the monotone floor turns a faint star's noise wings into
        # zeros -- a zero-wing kernel pushes the solve to inflate
        # component cores to absorb real PSF wing light. Graft a Moffat
        # continuation at the measured FWHM, scaled to match at the
        # graft radius. Per-ring S/N of the NORMALIZED profile: ring
        # noise is sigma / sqrt(n_px), normalized by the
        # (pre-normalization) peak.
        ring_err = (stamp.sigma / np.sqrt(np.maximum(ring_n, 1)))
        snr = prof / np.maximum(ring_err / max(peak, 1e-12), 1e-12)
        graft_note = ''
        low = np.where(snr < recipe.PSF_WING_SNR)[0]
        first_low = int(low[0]) if len(low) else len(mids)
        if first_low < len(mids) and mids[min(first_low, len(mids) - 1)] < GRAFT_MAX_AS:
            graft_radius = mids[max(first_low - 1, 1)]
            gamma = fwhm / (2 * np.sqrt(2 ** (1 / MOFFAT_BETA) - 1))
            moffat = (1 + (mids / gamma) ** 2) ** -MOFFAT_BETA
            scale = prof[max(first_low - 1, 1)] / max(moffat[max(first_low - 1, 1)], 1e-12)
            prof = np.where(mids >= graft_radius, moffat * scale, prof)
            graft_note = f'+moffat wings r>{graft_radius:.1f}as'

        # Render the circular profile onto a square kernel and normalize.
        size = max(int(round(recipe.MOFFAT_KERNEL_FWHM * fwhm / pixscale)) | 1,
                   KERNEL_MIN_PX)
        center = size // 2
        kyy, kxx = np.indices((size, size))
        kr = np.hypot(kyy - center, kxx - center) * pixscale
        kernel = np.interp(kr, mids, prof, right=0.0)
        total = float(kernel.sum())
        if total <= 0:
            continue
        return kernel / total, fwhm, f'empirical star (G={gmag:.1f}){graft_note}'
    return None


# ------------------------------------
# Resolution chain
# ------------------------------------
def resolve_psf(
        stamp: Stamp,
        cat: pd.DataFrame | None,
        stars: pd.DataFrame,
        *,
        psfsize_col: str | None = None,
        fallback_arcsec: float = 1.0,
        fallback_label: str = 'provider default',
) -> tuple[np.ndarray, float, str]:
    """Resolve the band PSF: empirical star first, Moffat fallback.

    The empirical kernel is always preferred -- a beta=3 Moffat is
    measurably too soft through the core against real survey PSFs, and
    a too-soft kernel rings every bright core on subtraction. When no
    star qualifies, the fallback is a Moffat sized by resolve_seeing.

    Parameters
    ----------
    stamp : Stamp
        The prepared stamp.
    cat : pd.DataFrame or None
        Survey catalog of the field (for resolve_seeing).
    stars : pd.DataFrame
        Confirmed stars with columns ra, dec, phot_g_mean_mag.
    psfsize_col, fallback_arcsec, fallback_label
        Passed through to resolve_seeing.

    Returns
    -------
    psf : tuple
        (kernel, fwhm_arcsec, provenance); a Moffat fallback appends
        ' (moffat fallback)' to the seeing provenance.
    """
    empirical = empirical_psf(stamp, stars)
    if empirical is not None:
        return empirical
    seeing_arcsec, provenance = resolve_seeing(
        cat, stamp.header, psfsize_col=psfsize_col,
        fallback_arcsec=fallback_arcsec, fallback_label=fallback_label)
    kernel = moffat_kernel(seeing_arcsec, stamp.pixscale)
    return kernel, seeing_arcsec, provenance + ' (moffat fallback)'
