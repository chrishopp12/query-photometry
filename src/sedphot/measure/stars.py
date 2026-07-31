"""
stars.py

Stage 4: The Star Stage
---------------------------------------------------------
A confirmed star is never subtracted from the data. It is placed in the
scene as a bounded component and removed by mask and fill, in one of two
ways decided by geometry alone:

  in the aperture zone   masked and twin-filled, with NO design column --
                         a free point-source column there absorbs target
                         light wholesale and its over-subtraction cannot
                         be masked after the fact.
  outside the zone       the catalog component stays, amplitude leashed to
                         the color-scaled catalog expectation and its mask
                         taken from the full uncapped model isophote.

Both routes bound the damage a wrong star model can do: the leash caps
the amplitude, and the mask-and-fill removes the region rather than
trusting the model over it. Neither can excavate light that was never
the star's.

Confirmation is astrometric, not positional: a Gaia row counts as a
star only with a 5-parameter solution at parallax or proper-motion
significance above recipe.STAR_ASTROM_SIG. Gaia membership alone is
not enough -- compact galaxy nuclei are in Gaia.

Requirements:
    numpy, astropy

Notes:
    Masks are returned in stamp counts on the stamp grid; star-log
    fluxes are microjanskys. Confirmed stars below recipe.STAR_MIN_UJY
    keep their catalog component untreated, and the target is never
    treated as a star.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from astropy.coordinates import SkyCoord

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
# The star stage
# ------------------------------------
def treat_stars(
        stamp: Stamp,
        comps: list[dict],
        stars: pd.DataFrame,
        *,
        colors: dict | None = None,
        aperture_arcsec: float | None = None,
        tag: str = '',
) -> tuple[list[tuple[str, np.ndarray]], list[dict], list[dict]]:
    """Route every confirmed star to a mask or to a leashed component.

    Each confirmed star at or above recipe.STAR_MIN_UJY catalog flux is
    matched to the nearest component within recipe.TARGET_MATCH_AS (never
    the target). A component is treated once, even when more than one Gaia
    row lands on it. The route is decided by geometry alone:

      inside the aperture zone   masked and filled, component dropped
      outside it                 component kept, amplitude leashed

    Nothing is subtracted from the data here. A star's light leaves the
    aperture either by mask-and-fill or by the bounded component the solve
    fits -- both cap the damage a wrong star model can do.

    Parameters
    ----------
    stamp : Stamp
        The prepared stamp (geometry, calibration, noise scale).
    comps : list of dict
        Scene components; each carries at least name, cat (uJy), x, y
        (stamp px), and base (rendered image, counts).
    stars : pd.DataFrame
        Confirmed stars (confirm_stars) with ra, dec, and
        phot_g_mean_mag columns.
    colors : dict, optional
        Catalog row index -> band/reference color factor; the leash and
        the masked footprint judge against color-scaled catalog fluxes.
    aperture_arcsec : float, optional
        Science aperture radius; with the target present it defines the
        no-column zone.
    tag : str
        Run-log prefix. [default: '']

    Returns
    -------
    star_masks : list of (str, np.ndarray)
        (component name, predicted footprint) per masked star.
    comps : list of dict
        The component list with masked stars removed.
    star_log : list of dict
        One record per treated star: comp, cat_uJy, gmag, mode.
    """
    pix = stamp.pixscale
    star_masks, star_log = [], []
    target = next((c for c in comps if c['name'] == 'target'), None)

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
        if best is None:
            inside = ((0 <= sx < stamp.data.shape[1])
                      and (0 <= sy < stamp.data.shape[0]))
            if inside:
                print(f"    {tag}Gaia star G="
                      f"{float(srow['phot_g_mean_mag']):.1f} has no "
                      f"catalog component within "
                      f"{recipe.TARGET_MATCH_AS:g}\"; unmodeled")
            continue
        if best['name'] == 'target' or best['cat'] < recipe.STAR_MIN_UJY:
            continue
        matched.append((best, srow))

    # Brightest first, so the log reads in order of consequence; a
    # claimed component is never treated twice.
    matched.sort(key=lambda t: -t[0]['cat'])
    claimed = set()
    for best, srow in matched:
        if best['name'] in claimed:
            continue
        claimed.add(best['name'])
        gmag = float(srow['phot_g_mean_mag'])
        color = (colors or {}).get(best.get('irow', -1), 1.0)
        # The expectation is the star's IN-STAMP flux (flux0: the
        # catalog-amplitude render's own integral), never its total:
        # amplitudes ARE in-stamp flux in this design, and for an
        # off-stamp star the two differ by orders of magnitude -- a
        # total-flux leash would FORCE that much wing light onto the
        # stamp. On-stamp stars render fully, so flux0 = cat there
        # and nothing changes.
        expected = color * best['flux0']
        head = (f"    {tag}STAR {best['name']} ({best['cat']:.0f} uJy "
                f"cat x{color:.2f} color, G={gmag:.1f})")
        in_zone = (aperture_arcsec is not None and target is not None
                   and np.hypot(best['x'] - target['x'],
                                best['y'] - target['y']) * pix
                   < aperture_arcsec + recipe.STAR_ZONE_BUFFER_AS)
        if in_zone:
            # Inside the aperture zone a free point-source column is an
            # absorber of target light: no column at all. The predicted
            # (color-scaled catalog) footprint is masked and the twin
            # fill reconstructs beneath it.
            star_masks.append((best['name'], best['base'] * color))
            star_log.append(dict(comp=best['name'], cat_uJy=best['cat'],
                                 gmag=gmag, mode='masked'))
            print(f"{head}: in the aperture zone -> masked + filled, "
                  f"no column")
        else:
            # Outside the zone the catalog component stays, amplitude
            # leashed to the color-scaled expectation so a wrong solve
            # can neither excavate nor absorb. It never gates, and
            # build_mask takes its full uncapped model isophote.
            best['gate'] = False
            best['star_reverted'] = True
            best['amp_lohi'] = (
                recipe.STAR_REVERT_AMP_BAND[0] * expected,
                recipe.STAR_REVERT_AMP_BAND[1] * expected)
            star_log.append(dict(comp=best['name'], cat_uJy=best['cat'],
                                 gmag=gmag, mode='leashed'))
            print(f"{head}: outside the aperture zone -> catalog "
                  f"component, amplitude leashed")

    treated = {name for name, _ in star_masks}
    comps = [c for c in comps if c['name'] not in treated]
    return star_masks, comps, star_log
