"""
psf.py

Stage 2: Empirical PSF, Moffat Fallback
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
# The core step must RESOLVE a marginally-sampled PSF: at 5 px FWHM,
# 2 px rings put the whole core in three medians and the rendered
# kernel broadens.
RING_STEP_PX = 1.0            # core ring width (pixels)
RING_STEP_MIN_AS = 0.25       # core ring width floor (arcsec)
RING_CORE_END_AS = 3.0        # core rings end here; wing rings begin
RING_WING_STEP_AS = 0.5       # wing ring width (arcsec)
RING_END_AS = 10.5            # schedule stop (last ring edge at 10.0)

MIN_FINITE_RINGS = 12         # measured rings required for a usable profile
PEAK_MIN_SIGMA = 20.0         # star peak floor, x the stamp's global sigma
FWHM_SANITY_AS = (0.4, 3.5)   # plausible measured-FWHM window
GRAFT_MAX_AS = 8.0            # graft only when rings fail inside this radius
KERNEL_MIN_PX = 25            # kernel size floor (odd pixels)
TAPER_START_FRAC = 0.8        # edge taper begins at this fraction of the radius


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


def _edge_taper(kernel: np.ndarray) -> np.ndarray:
    """Roll a kernel smoothly to exactly zero at its array edge.

    A finite kernel truncates its wings at a square boundary: every
    bright point source then renders as a box in the fitted scene, and
    the step survives into the residuals at the background plane's own
    scale on deep stacks. A cosine taper over the outer radius removes
    the discontinuity; the kernel returns renormalized to unit sum.
    """
    size = kernel.shape[0]
    center = size // 2
    yy, xx = np.indices(kernel.shape)
    rr = np.hypot(yy - center, xx - center)
    r_edge = float(center)
    r0 = TAPER_START_FRAC * r_edge
    taper = np.ones_like(kernel)
    outer = rr > r0
    taper[outer] = 0.5 * (1.0 + np.cos(
        np.pi * np.clip((rr[outer] - r0) / max(r_edge - r0, 1e-9),
                        0.0, 1.0)))
    taper[rr >= r_edge] = 0.0
    tapered = kernel * taper
    return tapered / tapered.sum()


def _kernel_fwhm(kernel: np.ndarray, pixscale: float) -> float:
    """Measured FWHM (arcsec) of a rendered kernel: azimuthal-median
    radial profile, interpolated at half peak."""
    size = kernel.shape[0]
    center = size // 2
    yy, xx = np.indices(kernel.shape)
    r_as = np.hypot(yy - center, xx - center) * pixscale
    edges = np.arange(0.0, (center - 1) * pixscale, 0.25 * pixscale)
    mids = 0.5 * (edges[1:] + edges[:-1])
    prof = np.array([np.median(kernel[(r_as >= lo) & (r_as < hi)])
                     if ((r_as >= lo) & (r_as < hi)).any() else np.nan
                     for lo, hi in zip(edges, edges[1:])])
    finite = np.isfinite(prof)
    prof = prof[finite] / kernel[center, center]
    mids = mids[finite]
    return 2.0 * float(np.interp(0.5, prof[::-1], mids[::-1]))


def moffat_kernel(seeing_arcsec: float, pixscale: float) -> np.ndarray:
    """Unit-sum Moffat kernel that MEASURES at the requested seeing.

    A kernel is only as sharp as it renders: pixel integration broadens
    a nominal-width Moffat by several percent at survey sampling, and a
    kernel broader than the true PSF is a floor no shape solve can get
    under -- the refit rails its size parameters trying. The rendered
    width is measured and the input width iterated until they agree;
    the box stays sized by the requested seeing.
    """
    fwhm_px = seeing_arcsec / pixscale
    size = max(int(round(recipe.MOFFAT_KERNEL_FWHM * fwhm_px)) | 1,
               KERNEL_MIN_PX)
    width = seeing_arcsec
    kernel = _edge_taper(moffat_psf(width, pixscale, size=size))
    for _ in range(3):
        measured = _kernel_fwhm(kernel, pixscale)
        if abs(measured - seeing_arcsec) <= 0.01 * seeing_arcsec:
            break
        width = max(width * seeing_arcsec / measured, 0.3 * pixscale)
        kernel = _edge_taper(moffat_psf(width, pixscale, size=size))
    return kernel


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
        # Never measure a kernel inside the target's structured zone:
        # the rings would measure the envelope, at high S/N.
        if np.hypot(sx - stamp.cx, sy - stamp.cy) * pixscale \
                < recipe.PSF_EXCLUDE_TARGET_AS:
            print(f"  psf: star G={gmag:.1f} is inside the target's "
                  f"{recipe.PSF_EXCLUDE_TARGET_AS:g}\" zone; not a "
                  f"kernel candidate")
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
                # The median VALUE of a monotone profile over a ring is
                # the profile at the ring's median RADIUS -- medians
                # commute with monotone maps. At marginal sampling a
                # ring holds a few discrete radii with unequal counts,
                # and the geometric mid is the wrong abscissa.
                mids[i] = float(np.median(rr_star[ring]))
        # Anchor r = 0 at the star's own background-subtracted peak.
        # Ring medians live at ring radii, and a render that clamps
        # inside the first one gets a flat-topped kernel broader than
        # the star it was measured from -- the median of a peaked core
        # is not its peak. The anchor IS the core measurement: sparse
        # inner rings (1-px rings can hold fewer than 3 pixels) are
        # interpolated against it.
        iy, ix = int(round(sy)), int(round(sx))
        core_peak = float(np.nanmax(
            data[max(iy - 1, 0):iy + 2, max(ix - 1, 0):ix + 2])) - background
        if not np.isfinite(core_peak) or core_peak <= 0:
            continue
        mids = np.concatenate([[0.0], mids])
        prof = np.concatenate([[core_peak], prof])
        ring_n = np.concatenate([[1.0], ring_n])
        if np.isfinite(prof).sum() < MIN_FINITE_RINGS:
            continue
        finite = np.isfinite(prof)
        prof = np.interp(mids, mids[finite], prof[finite])

        # Saturated / blended cores are non-monotone at the center --
        # at the anchor or between the first rings; means over two
        # rings keep single noisy rings from vetoing a good star.
        if (np.mean(prof[1:3]) > prof[0]
                or np.mean(prof[2:4]) > prof[1]
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
            j = max(first_low - 1, 1)
            graft_radius = mids[j]
            gamma = fwhm / (2 * np.sqrt(2 ** (1 / MOFFAT_BETA) - 1))
            # The continuation's SLOPE comes from the star itself: a
            # fixed beta=3 pastes transition-zone wings the measured
            # shoulder contradicts -- real survey PSFs fall steeper
            # there, and a unit-sum kernel pays for phantom wings by
            # suppressing its core, broadening every convolution.
            # Beta is fit on the reliable shoulder rings; the default
            # stands only when they are too few to say.
            beta = MOFFAT_BETA
            shoulder = ((snr >= recipe.PSF_WING_SNR)
                        & (mids >= 0.75 * fwhm) & (mids <= graft_radius))
            if shoulder.sum() >= 3:
                xw = np.log1p((mids[shoulder] / gamma) ** 2)
                yw = np.log(np.maximum(prof[shoulder], 1e-12))
                beta = float(np.clip(-np.polyfit(xw, yw, 1)[0], 2.0, 8.0))
            moffat = (1 + (mids / gamma) ** 2) ** -beta
            scale = prof[j] / max(moffat[j], 1e-12)
            prof = np.where(mids >= graft_radius, moffat * scale, prof)
            graft_note = (f'+moffat wings (b={beta:.1f}) '
                          f'r>{graft_radius:.1f}as')

        # Render the circular profile onto a square kernel. The box
        # never outgrows the measured profile's support: beyond the
        # last ring the interpolation would cut a hard circular step
        # inside the taper zone.
        def render(prof_use, fwhm_use):
            support_px = int(2 * RING_END_AS / pixscale)
            size = max(min(int(round(recipe.MOFFAT_KERNEL_FWHM
                                     * fwhm_use / pixscale)),
                           support_px) | 1, KERNEL_MIN_PX)
            center = size // 2
            kyy, kxx = np.indices((size, size))
            kr = np.hypot(kyy - center, kxx - center) * pixscale
            kernel = np.interp(kr, mids, prof_use, right=0.0)
            total = float(kernel.sum())
            if total <= 0:
                return None, 1.0
            wings = float(kernel[kr > 2 * fwhm_use].sum()) / total
            return kernel, wings

        kernel, wings = render(prof, fwhm)
        if kernel is not None and wings > recipe.PSF_WING_FRAC_MAX:
            # Contaminated wings on a (possibly) clean core: re-derive
            # the FWHM from the inner rings alone and force the graft
            # early -- the measurement keeps the core, the analytic
            # continuation takes the wings. High-S/N contamination
            # never trips the noise-triggered graft, so this is its
            # deliberate sibling.
            inner = mids < recipe.PSF_GRAFT_FORCE_AS
            if (np.isfinite(prof[inner]).sum() >= 4
                    and float(np.nanmin(prof[inner])) < 0.5):
                fwhm_core = 2.0 * float(np.interp(
                    0.5, prof[inner][::-1], mids[inner][::-1]))
                gamma = fwhm_core / (2 * np.sqrt(2 ** (1 / MOFFAT_BETA)
                                                 - 1))
                moffat = (1 + (mids / gamma) ** 2) ** -MOFFAT_BETA
                # Anchor the analytic wings to the CLEAN CORE, never
                # to the graft point: contamination at the graft
                # radius would scale itself straight back into the
                # wings it was meant to replace.
                win = (mids > 0.6 * fwhm_core) & (mids < 1.5 * fwhm_core)
                scale = (float(np.median(prof[win] / moffat[win]))
                         if win.sum() >= 2 else 1.0)
                j = int(np.argmin(np.abs(mids
                                         - recipe.PSF_GRAFT_FORCE_AS)))
                prof_fixed = np.where(mids >= mids[j], moffat * scale,
                                      prof)
                kernel2, wings2 = render(prof_fixed, fwhm_core)
                if kernel2 is not None \
                        and wings2 <= recipe.PSF_WING_FRAC_MAX:
                    kernel, wings, fwhm = kernel2, wings2, fwhm_core
                    graft_note = (f'+forced moffat wings r>'
                                  f'{recipe.PSF_GRAFT_FORCE_AS:g}as')
                else:
                    kernel = None
            else:
                kernel = None
        if kernel is None:
            print(f"  psf: star G={gmag:.1f} kernel rejected "
                  f"(contaminated wings); trying the next candidate")
            continue
        dx = (sx - stamp.cx) * pixscale
        dy = (sy - stamp.cy) * pixscale
        return (_edge_taper(kernel), fwhm,
                f'empirical star (G={gmag:.1f}, {dx:+.0f},{dy:+.0f}\" '
                f'from target, wings {wings:.0%}){graft_note}')
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
