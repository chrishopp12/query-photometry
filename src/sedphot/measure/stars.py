"""
stars.py

Stage 4: The Measured-Star Stage
---------------------------------------------------------
Confirmed stars leave the component list entirely: each is replaced by
its own measured circular clipped-median radial profile, subtracted
from the data before any fitting. The measurement is the model -- a
measured profile cannot absorb light it does not see, so its mask
needs no geometric cap.

Confirmation is astrometric, not positional: a Gaia row counts as a
star only with a 5-parameter solution at parallax or proper-motion
significance above recipe.STAR_ASTROM_SIG. Gaia membership alone is
not enough -- compact galaxy nuclei are in Gaia.

Requirements:
    numpy, scipy, astropy

Notes:
    Profiles are measured, returned, and subtracted in stamp counts on
    the stamp grid; star-log fluxes are microjanskys. Confirmed stars
    below recipe.STAR_MIN_UJY keep their catalog component, and the
    target is never treated as a star.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import binary_dilation

from . import recipe

if TYPE_CHECKING:
    import pandas as pd

    from .stamp import Stamp


# ------------------------------------
# Star confirmation
# ------------------------------------
def confirm_stars(gaia: pd.DataFrame) -> pd.DataFrame:
    """Select the Gaia rows that are astrometrically confirmed stars.

    A row is a star only when its 5-parameter astrometric solution
    detects parallax or proper motion above recipe.STAR_ASTROM_SIG.
    Rows without a solution (missing or non-positive errors) score
    zero significance and never pass.

    Parameters
    ----------
    gaia : pd.DataFrame
        Gaia rows with parallax, parallax_error, pmra, pmra_error,
        pmdec, and pmdec_error columns.

    Returns
    -------
    stars : pd.DataFrame
        The confirmed-star rows, reindexed from zero.
    """
    plx_sig = np.where(
        np.isfinite(gaia['parallax_error']) & (gaia['parallax_error'] > 0),
        np.abs(gaia['parallax']) / gaia['parallax_error'], 0.0)
    pm = np.hypot(gaia['pmra'], gaia['pmdec'])
    pm_err = np.hypot(gaia['pmra_error'], gaia['pmdec_error'])
    pm_sig = np.where(np.isfinite(pm_err) & (pm_err > 0), pm / pm_err, 0.0)
    confirmed = np.maximum(plx_sig, pm_sig) > recipe.STAR_ASTROM_SIG
    return gaia[confirmed].reset_index(drop=True)


# ------------------------------------
# The measured profile
# ------------------------------------
def measure_star_profile(
        raw: np.ndarray,
        good: np.ndarray,
        others_img: np.ndarray,
        level: float,
        sx: float,
        sy: float,
        pixscale: float,
        rr: np.ndarray,
        sigma: float,
        *,
        extra_exclude: np.ndarray | None = None,
) -> np.ndarray:
    """Measure one star's circular clipped-median radial profile.

    Ring medians are taken on the data with the rest of the scene and
    the background level subtracted, excluding the target region and
    the dilated bright pixels of the other components' models -- the
    profile must be the star's own light. The ring profile is
    interpolated across rings without a vote, forced monotone
    decreasing, floored at zero, and zero beyond the terminus.

    Parameters
    ----------
    raw : np.ndarray
        Stamp data (counts), with any brighter siblings' measured
        profiles already subtracted.
    good : np.ndarray
        Usable-pixel map.
    others_img : np.ndarray
        Summed model images of every other component (counts).
    level : float
        Background level under the star (counts).
    sx, sy : float
        Star position (stamp pixels).
    pixscale : float
        Pixel scale (arcsec/px).
    rr : np.ndarray
        Radius map about the target (arcsec).
    sigma : float
        Global pixel scatter (counts); sets the neighbor-exclusion
        threshold.
    extra_exclude : np.ndarray, optional
        Extra pixels to keep out of the rings (already-measured star
        light, for example).

    Returns
    -------
    profile_img : np.ndarray
        The measured profile evaluated on the stamp grid (counts);
        zeros when too few rings vote.
    """
    yy, xx = np.indices(raw.shape)
    r_star = np.hypot(yy - sy, xx - sx) * pixscale
    work = raw - others_img - level

    # Rings never see the target region or another component's bright
    # (dilated) footprint.
    exclude = (rr < recipe.BG_RMIN_AS) | binary_dilation(
        others_img > 1.0 * sigma, iterations=2)
    if extra_exclude is not None:
        exclude = exclude | extra_exclude
    usable = good & ~exclude

    # 1-arcsec rings through the core, 2.5-arcsec rings to the
    # terminus; a ring votes only with enough clean pixels.
    edges = np.concatenate([np.arange(0, 10, 1.0),
                            np.arange(10, recipe.STAR_PROF_MAX_AS + 1, 2.5)])
    mids, prof = [], []
    for i in range(len(edges) - 1):
        ring = usable & (r_star >= edges[i]) & (r_star < edges[i + 1])
        mids.append(0.5 * (edges[i] + edges[i + 1]))
        if ring.sum() < recipe.STAR_RING_MIN_PX:
            prof.append(np.nan)
            continue
        _, med, _ = sigma_clipped_stats(work[ring], sigma=3.0, maxiters=6)
        prof.append(med)
    prof, mids = np.array(prof), np.array(mids)

    finite = np.isfinite(prof)
    if finite.sum() < 4:
        return np.zeros_like(raw)
    prof = np.interp(mids, mids[finite], prof[finite])
    prof = np.clip(np.minimum.accumulate(prof), 0.0, None)
    return np.interp(r_star, mids, prof, right=0.0)


# ------------------------------------
# The measured-star stage
# ------------------------------------
def subtract_stars(
        stamp: Stamp,
        raw: np.ndarray,
        good: np.ndarray,
        comps: list[dict],
        stars: pd.DataFrame,
        level: float,
        *,
        tag: str = '',
) -> tuple[np.ndarray, list[tuple[str, np.ndarray]], list[dict], list[dict]]:
    """Replace confirmed-star components with measured profiles.

    Each confirmed star at or above recipe.STAR_MIN_UJY catalog flux
    is matched to the nearest component within recipe.TARGET_MATCH_AS
    (never the target) and treated brightest-first, sequentially: its
    profile is measured on data with brighter siblings' profiles
    already subtracted, with the treated stars' catalog bases removed
    from the reference scene, and with already-measured star light
    excluded above one sigma. A component is treated once, even when
    more than one Gaia row lands on it. Treated components leave the
    component list.

    Parameters
    ----------
    stamp : Stamp
        The prepared stamp (geometry, calibration, noise scale).
    raw : np.ndarray
        Stamp data (counts).
    good : np.ndarray
        Usable-pixel map.
    comps : list of dict
        Scene components; each carries at least name, cat (uJy), x, y
        (stamp px), and base (rendered image, counts).
    stars : pd.DataFrame
        Confirmed stars (confirm_stars) with ra, dec, and
        phot_g_mean_mag columns.
    level : float
        Background level (counts) under each measurement.
    tag : str
        Run-log prefix. [default: '']

    Returns
    -------
    star_img : np.ndarray
        Summed measured-star image (counts), to subtract from the data.
    star_masks : list of (str, np.ndarray)
        (component name, profile image) per treated star.
    comps : list of dict
        The component list with treated stars removed.
    star_log : list of dict
        One record per treated star: comp, cat_uJy, gmag, profile_uJy.
    """
    pix, cf = stamp.pixscale, stamp.cf
    star_img = np.zeros_like(raw)
    star_masks, star_log = [], []
    scene0 = sum(c['base'] for c in comps)
    treated_base = np.zeros_like(raw)

    # Match every confirmed star to its nearest component. The target
    # is never treated; fainter components keep their catalog model.
    matched = []
    for _, srow in stars.iterrows():
        ssky = SkyCoord(float(srow['ra']), float(srow['dec']), unit='deg')
        sx, sy = [float(v) for v in stamp.wcs.world_to_pixel(ssky)]
        best, bestd = None, recipe.TARGET_MATCH_AS
        for c in comps:
            d = np.hypot(c['x'] - sx, c['y'] - sy) * pix
            if d < bestd:
                best, bestd = c, d
        if (best is None or best['name'] == 'target'
                or best['cat'] < recipe.STAR_MIN_UJY):
            continue
        matched.append((best, sx, sy, srow))

    # Brightest first, each measured on data with the brighter
    # siblings' profiles already out; a claimed component never
    # measures twice.
    matched.sort(key=lambda t: -t[0]['cat'])
    claimed = set()
    for best, sx, sy, srow in matched:
        if best['name'] in claimed:
            continue
        claimed.add(best['name'])
        prof = measure_star_profile(
            raw - star_img, good, scene0 - best['base'] - treated_base,
            level, sx, sy, pix, stamp.rr, stamp.sigma,
            extra_exclude=star_img > stamp.sigma)
        star_img += prof
        treated_base += best['base']
        star_masks.append((best['name'], prof))
        flux = float(prof.sum() * cf)
        star_log.append(dict(comp=best['name'], cat_uJy=best['cat'],
                             gmag=float(srow['phot_g_mean_mag']),
                             profile_uJy=round(flux, 1)))
        print(f"    {tag}STAR {best['name']} ({best['cat']:.0f} uJy cat, "
              f"G={srow['phot_g_mean_mag']:.1f}): measured profile "
              f"{flux:.0f} uJy pre-subtracted")

    treated = {name for name, _ in star_masks}
    comps = [c for c in comps if c['name'] not in treated]
    return star_img, star_masks, comps, star_log
