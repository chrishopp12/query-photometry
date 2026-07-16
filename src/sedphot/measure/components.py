"""
components.py

Stage 3: Catalog Rows to Scene Components
---------------------------------------------------------
Turn the scene catalog into rendered components: every catalog row
becomes a fixed profile at its catalog shape, rendered through the
band's PSF and normalized so a fitted amplitude reads directly in
microjanskys. The target is identified by position and is never gated.
Optional per-galaxy patches edit the catalog before any of this
(replacing a blended row with a known decomposition, for example).

Component dict (the currency of the whole engine):
    name       'target' or 'src<catalog row index>'
    irow       catalog row index (int; -1 for synthetic components)
    cat        catalog flux through the scene band (uJy)
    x, y       stamp-pixel position
    gate       True when the row qualifies for a shape solve (seats.py)
    base       rendered image at catalog shape and amplitude (counts)
    flux0      in-stamp flux of base (uJy), the design normalization
    shape      dict(reff_px, n, ellip, theta, pa) or None for point sources

Requirements:
    numpy, pandas, astropy

Notes:
    Registry consumption (seats.apply_registry) may append frozen
    components carrying two extra keys: reg=True and amp_lohi (a flux
    leash in uJy).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from ..catalogs.legacy import shape_from_tractor
from ..units import NANOMAGGY_TO_UJY
from . import recipe
from .render import ampl_from_total, conv_same, moffat_wings, sersic_profile
from .sersic import theta_from_pa
from .stamp import Stamp


# ------------------------------------
# The gate
# ------------------------------------
def gated_row(row: pd.Series, dist_arcsec: float) -> bool:
    """Whether a catalog row qualifies for the double-profile shape solve.

    The second profile exists to fix MISFIT, and the catalog's reduced
    chi-square is its own misfit statement -- the necessary condition.
    Point-source rows and the target itself never gate.
    """
    if dist_arcsec < recipe.TARGET_MATCH_AS:
        return False
    if str(row['type']).strip() in ('PSF', 'DUP'):
        return False
    return (row['uJy'] > recipe.GATE_FLUX_UJY
            and row['rchisq_r'] > recipe.GATE_RCHISQ)


# ------------------------------------
# Patches: the custom-input channel
# ------------------------------------
def apply_patches(cat: pd.DataFrame, patches: dict) -> pd.DataFrame:
    """Apply per-galaxy catalog edits from a patches file.

    'replace_rows' swaps one catalog row for one or more replacement
    rows (e.g. a known double-nucleus decomposition replacing the
    blended catalog row). Each replacement inherits every column of the
    replaced row and overrides the ones it names. Nothing here is
    required: no patch file means pure catalog behavior.

    Parameters
    ----------
    cat : pd.DataFrame
        Scene catalog (brightest-first, with a 'uJy' column).
    patches : dict
        Parsed patches file.

    Returns
    -------
    patched : pd.DataFrame
        The catalog with replacements applied, re-sorted brightest-first.
    """
    replacements = patches.get('replace_rows', [])
    if not replacements:
        return cat
    for rep in replacements:
        target = SkyCoord(rep['ra'], rep['dec'], unit='deg')
        rows = SkyCoord(cat['ra'].values, cat['dec'].values, unit='deg')
        sep = target.separation(rows).arcsec
        idx = int(np.argmin(sep))
        if sep[idx] > recipe.PATCH_MATCH_AS:
            print(f"  patch replace_rows: no row within "
                  f"{recipe.PATCH_MATCH_AS:g}\" of "
                  f"({rep['ra']:.5f},{rep['dec']:.5f}); skipped")
            continue
        old = cat.iloc[idx].to_dict()
        new_rows = []
        for row in rep['with']:
            record = dict(old)
            record.update(row)
            new_rows.append(record)
        cat = pd.concat([cat.drop(cat.index[idx]),
                         pd.DataFrame(new_rows)], ignore_index=True)
        print(f"  patch: replaced catalog row ({sep[idx]:.2f}\" match) "
              f"with {len(new_rows)} row(s)")
    cat['uJy'] = cat['flux_r'] * NANOMAGGY_TO_UJY
    return cat.sort_values('flux_r', ascending=False).reset_index(drop=True)


# ------------------------------------
# Target-substructure rows
# ------------------------------------
def drop_target_shreds(
        cat: pd.DataFrame,
        coord: SkyCoord,
        *,
        aperture_arcsec: float,
        patches: dict | None = None,
) -> pd.DataFrame:
    """Drop catalog rows that are really the target's own light.

    A row inside the science aperture whose fracflux exceeds
    recipe.SHRED_FRACFLUX sits where other sources' light outweighs its
    own flux -- the catalog's rendering of target substructure, not an
    independent neighbor. Subtracting (and masking) such rows steals
    target flux, so they leave the scene entirely and their light is
    measured as target. The target row itself never drops, and rows at
    patch-named positions are pinned -- declared human knowledge wins
    over the blind rule.

    Parameters
    ----------
    cat : pd.DataFrame
        Scene catalog (post-patches, with 'uJy' and 'fracflux_r').
    coord : SkyCoord
        Target position.
    aperture_arcsec : float
        Science aperture radius (the rule's scope).
    patches : dict, optional
        Per-galaxy custom inputs; their named positions are exempt.

    Returns
    -------
    kept : pd.DataFrame
        The catalog without the target-substructure rows, reindexed.
    """
    if not len(cat) or 'fracflux_r' not in cat:
        return cat
    rows = SkyCoord(cat['ra'].values, cat['dec'].values, unit='deg')
    dist = coord.separation(rows).arcsec
    shred = ((dist > recipe.TARGET_MATCH_AS)
             & (dist < aperture_arcsec)
             & (cat['fracflux_r'].values > recipe.SHRED_FRACFLUX))
    if patches:
        pinned = []
        for rep in patches.get('replace_rows', []):
            for row in rep['with']:
                pinned.append((row.get('ra', rep['ra']),
                               row.get('dec', rep['dec'])))
        for seat in patches.get('free_seats', []):
            pinned.append((seat['ra'], seat['dec']))
        for ra, dec in pinned:
            near = SkyCoord(ra, dec, unit='deg').separation(rows).arcsec
            shred &= ~(near < recipe.PATCH_MATCH_AS)
    if shred.any():
        fluxes = [round(v, 1) for v in cat.loc[shred, 'uJy']]
        print(f"  target substructure: {int(shred.sum())} catalog row(s) "
              f"inside the aperture with fracflux_r > "
              f"{recipe.SHRED_FRACFLUX:g} ({fluxes} uJy) leave the "
              f"scene; their light is measured as target flux")
    return cat[~shred].reset_index(drop=True)


# ------------------------------------
# Components
# ------------------------------------
def build_components(
        cat: pd.DataFrame,
        stamp: Stamp,
        psf: np.ndarray,
        seeing_arcsec: float,
        *,
        profile_cache: dict | None = None,
        gate_radius_arcsec: float | None = None,
) -> list[dict]:
    """Render every usable catalog row into a fixed scene component.

    On-stamp rows always enter. Extended rows within the off-stamp
    margin enter only when their catalog-shape render lands at least
    MARGIN_MIN_UJY on the stamp: normalized to unit in-stamp flux, a
    near-empty render is a numerically explosive design column whose
    amplitude rails at any bound. Off-stamp point sources at or above
    BRIGHT_PSF_UJY stay as analytic full-wing Moffat components -- their
    wings still reach across the edge.

    Parameters
    ----------
    cat : pd.DataFrame
        Scene catalog (post-patches).
    stamp : Stamp
        The band's prepared stamp.
    psf : np.ndarray
        Normalized PSF kernel for this band.
    seeing_arcsec : float
        Band PSF FWHM; sizes the analytic Moffat wings.
    profile_cache : dict, optional
        Unconvolved-profile cache shared by bands on an IDENTICAL
        instrument grid, keyed by component name (catalog shapes are
        band-independent; each band convolves with its own PSF). The
        caller owns grid-identity verification.
    gate_radius_arcsec : float, optional
        Radial gate reach: rows beyond this sky distance never gate,
        keeping the gate census identical on every instrument
        regardless of grid rotation. None gates by stamp membership
        alone.

    Returns
    -------
    comps : list of dict
        Scene components (see the module docstring for the schema).
    """
    comps = []
    shape_2d = stamp.shape
    pix = stamp.pixscale
    cf = stamp.cf
    margin_px = recipe.MARGIN_AS / pix
    for irow, row in cat.iterrows():
        x, y = [float(v) for v in stamp.wcs.world_to_pixel(
            SkyCoord(float(row['ra']), float(row['dec']), unit='deg'))]
        inside = (0 <= x < shape_2d[1]) and (0 <= y < shape_2d[0])
        near = (-margin_px <= x < shape_2d[1] + margin_px
                and -margin_px <= y < shape_2d[0] + margin_px)
        dist_arcsec = np.hypot(x - stamp.cx, y - stamp.cy) * pix
        # Components are named by CATALOG ROW, not running count: the
        # component list differs between bands (margin cuts on different
        # grids), and that must not shift the identity of every later
        # source -- solved shapes transfer across bands by name.
        name = ('target' if dist_arcsec < recipe.TARGET_MATCH_AS
                else f'src{irow}')
        shape = shape_from_tractor(row['type'], row['sersic'],
                                   row['shape_r'], row['shape_e1'],
                                   row['shape_e2'])
        counts = max(float(row['uJy']), 0.0) / cf
        # Off-stamp and beyond-reach rows never gate: the pixels that
        # would constrain a shape solve are not (reliably) on the
        # stamp, and the wing-level light such a source lands here is
        # served by its catalog profile. A monster beyond the edge
        # whose envelope truly reaches the target is patches territory,
        # never blind-scene machinery.
        in_reach = (gate_radius_arcsec is None
                    or dist_arcsec < gate_radius_arcsec)
        meta = dict(name=name, irow=int(irow), cat=float(row['uJy']),
                    x=x, y=y,
                    gate=inside and in_reach and gated_row(row, dist_arcsec))

        if shape is None:
            # Point source. On-stamp: a delta at the catalog position
            # convolved with the band PSF. Off-stamp: analytic Moffat
            # wings when bright enough to reach, dropped otherwise.
            if not inside:
                if not near or meta['cat'] < recipe.BRIGHT_PSF_UJY:
                    continue
                base = moffat_wings(counts, seeing_arcsec / pix, x, y,
                                    shape_2d)
            else:
                img = np.zeros(shape_2d)
                img[int(round(y)), int(round(x))] = counts
                base = conv_same(img, psf)
            comps.append(dict(base=base,
                              flux0=max(float(base.sum()) * cf, 1e-9),
                              shape=None, **meta))
            continue

        if not near:
            continue
        n = float(shape['n'])
        reff_arcsec = float(shape['reff_arcsec'])
        ellip = float(shape['ellip'])
        pa = float(shape['pa_deg'])
        theta = theta_from_pa(stamp.wcs, x, y, pa)
        reff_px = max(reff_arcsec / pix, 0.3)
        ampl = ampl_from_total(counts, reff_px, n, ellip) if counts else 0.0
        unconv = (profile_cache.get(name)
                  if profile_cache is not None else None)
        if unconv is None:
            unconv = sersic_profile([ampl, reff_px, n, ellip, theta, x, y],
                                    shape_2d)
            if profile_cache is not None:
                profile_cache[name] = unconv
        # The margin rule, no exemptions: an off-stamp source must
        # actually REACH the stamp at catalog shape and amplitude or it
        # does not exist for this field. An off-stamp giant whose light
        # truly reaches is patches territory -- components enter blind
        # scenes on data-supported presence only.
        if not inside and float(unconv.sum()) * cf < recipe.MARGIN_MIN_UJY:
            continue
        base = conv_same(unconv, psf)
        comps.append(dict(
            base=base, flux0=max(float(base.sum()) * cf, 1e-9),
            shape=dict(reff_px=reff_px, n=n, ellip=ellip, theta=theta,
                       pa=pa), **meta))
    return comps
