"""
engine.py

Stage 6: Per-Galaxy Scene Measurement Driver
---------------------------------------------------------
One galaxy, all bands: fetch the scene inputs once (survey catalog,
confirmed stars, optional patches and registry), then measure every
band through the same chain --

    stamp -> PSF -> components -> registry -> stars -> seats ->
    joint fit -> mask -> twin fill -> curve of growth -> witnesses

Bands are measured per instrument: the first band in preference order
is the REFERENCE -- it solves seat shapes -- and its siblings transfer
those shapes, re-solving neighbor seats warm with color-leashed fluxes.
Nothing is shared across instruments but the catalogs and patches.

Data products (per band, returned to the pipeline):
    measurement dict     flux, error, witnesses, and the images the QA
                         figure draws
    reference dict       seat shapes + fluxes for the instrument's
                         sibling bands

Requirements:
    numpy, pandas, astropy

Notes:
    A position with no scene catalog measures blind: no components, no
    masks, background and curve of growth only, with scene=none in the
    flags. Missing coverage still demotes through the stamp gates.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from ..catalogs import gaia
from ..catalogs.legacy import SCENE_COLS, query_scene
from ..units import NANOMAGGY_TO_UJY
from . import recipe
from .aperture import (build_mask, curve, enclosed_at, flux_error,
                       twin_fill, witness_row)
from .background import bin_plane
from .components import apply_patches, build_components
from .psf import resolve_psf
from .seats import apply_registry, build_seats, harvest_seats, load_registry
from .solve import joint_fit
from .stamp import check_coverage, load_stamp
from .stars import confirm_stars, subtract_stars


# ------------------------------------
# Scene inputs (once per galaxy)
# ------------------------------------
def prepare_scene(
        coord: SkyCoord,
        *,
        phot_dir: str | Path,
        out_dir: str | Path,
        legacy_dr: str = 'dr9',
        registry_path: str | Path | None = None,
) -> dict:
    """Fetch and assemble everything the scene needs, once per galaxy.

    Catalog queries are cache-first under <phot_dir>/scene/, so a
    re-measure never re-queries. A query that finds nothing (off the
    survey footprint) yields an EMPTY catalog and the engine measures
    blind; a query that FAILS raises -- a service outage must not
    silently downgrade the measurement.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    phot_dir : str or Path
        The galaxy's Photometry/ directory (scene cache lives under it).
    out_dir : str or Path
        The galaxy directory; an optional patches file
        (recipe.PATCH_FILENAME) is read from here.
    legacy_dr : str
        Legacy data release for the scene catalog. [default: 'dr9']
    registry_path : str or Path, optional
        Cross-field registry to consume (and update, if the caller
        saves it back).

    Returns
    -------
    scene : dict
        cat (DataFrame), stars (DataFrame), patches (dict), registry
        (dict), registry_path.
    """
    cache_dir = Path(phot_dir) / 'scene'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cat = query_scene(coord, recipe.QUERY_RADIUS_AS, dr=legacy_dr,
                      min_flux_nmgy=recipe.TRACTOR_MIN_NMGY,
                      cache_path=cache_dir / f'tractor_scene_{legacy_dr}.csv')
    gaia_cat = gaia.query_cone(coord, recipe.QUERY_RADIUS_AS,
                               cache_path=cache_dir / 'gaia_scene.csv')
    stars = confirm_stars(gaia_cat) if len(gaia_cat) else gaia_cat

    patches = {}
    patch_path = Path(out_dir) / recipe.PATCH_FILENAME
    if patch_path.exists():
        with open(patch_path) as handle:
            patches = json.load(handle)
        print(f"  patches: {sorted(patches.keys())}")
    if len(cat):
        cat = apply_patches(cat, patches)
    else:
        print("  scene catalog is empty here; measuring blind")

    registry = load_registry(registry_path)
    if registry:
        print(f"  registry: {sorted(registry.keys())}")
    return dict(cat=cat, stars=stars, patches=patches,
                registry=registry, registry_path=registry_path)


def order_bands(products: list) -> list:
    """Reference band first: sort an instrument's products so the first
    filter in recipe.REFERENCE_PREFERENCE leads and solves the seats."""
    def rank(product):
        band = product.band.lower()
        if band in recipe.REFERENCE_PREFERENCE:
            return recipe.REFERENCE_PREFERENCE.index(band)
        return len(recipe.REFERENCE_PREFERENCE)
    return sorted(products, key=rank)


def _seat_colors(seats: list[dict], cat: pd.DataFrame,
                 comps: list[dict], band: str) -> list[float]:
    """Color factor per transferred seat column: this band's catalog
    flux over the reference-band flux for the seat's owner (nearest
    catalog band). Keeps transfer flux leashes physical across colors;
    clamped, and neutral (1.0) when the catalog cannot say."""
    col = recipe.BAND_COLOR_COL.get(band.lower(), 'flux_r')
    irow_by_name = {c['name']: c['irow'] for c in comps}
    colors = []
    for seat in seats:
        irow = irow_by_name.get(seat['owner'], -1)
        if irow < 0 or irow >= len(cat):
            colors.append(1.0)
            continue
        flux_ref = float(cat.iloc[irow]['flux_r'])
        flux_band = float(cat.iloc[irow].get(col, np.nan))
        usable = (np.isfinite(flux_band) and np.isfinite(flux_ref)
                  and flux_ref > 0 and flux_band > 0)
        colors.append(float(np.clip(flux_band / flux_ref, 0.05, 20.0))
                      if usable else 1.0)
    return colors


# ------------------------------------
# One band through the chain
# ------------------------------------
def measure_band(
        product,
        coord: SkyCoord,
        scene: dict,
        ref: dict | None,
        caches: dict,
        *,
        aperture_arcsec: float,
        cutout_half_arcsec: float,
        rgrid: np.ndarray,
        target_shape: dict | None = None,
        registry_update: bool = False,
        dump_dir: str | Path | None = None,
) -> tuple[dict, dict | None]:
    """Measure one band; returns (measurement, reference-or-None).

    The returned reference is non-None only on a band that solved seat
    shapes (the instrument's reference band); the caller passes it back
    in for that instrument's remaining bands.

    Parameters
    ----------
    product : ImageProduct
        The fetched band image.
    coord : SkyCoord
        Target position.
    scene : dict
        Scene inputs (prepare_scene).
    ref : dict or None
        The instrument's reference (None on the reference band itself).
    caches : dict
        Cross-band scratch owned by the caller; holds the per-
        instrument unconvolved-profile cache.
    aperture_arcsec : float
        Science aperture radius.
    cutout_half_arcsec : float
        Stamp half-size.
    rgrid : np.ndarray
        Curve-of-growth radii.
    target_shape : dict, optional
        Explicit target shape (n, reff_arcsec, ellip, pa_deg): pins the
        target to a fixed profile instead of the standard refit
        (forced-photometry mode).
    registry_update : bool
        Harvest this band's solved seats into scene['registry'].
    dump_dir : str or Path, optional
        Write the per-band array bundle here (debug).

    Returns
    -------
    measurement : dict
        Everything the pipeline row, sidecar, and QA figure need.
    new_ref : dict or None
        Reference for sibling bands, when this band solved shapes.
    """
    cat, stars = scene['cat'], scene['stars']
    patches = dict(scene['patches'])
    band_key = f"{product.instrument}_{product.band}"
    tag = f"[{band_key}] "
    t0 = time.time()

    stamp = load_stamp(product.path, product.calib, coord,
                       cutout_half_arcsec=cutout_half_arcsec,
                       invvar_path=product.invvar_path)
    psf, seeing, seeing_src = resolve_psf(
        stamp, cat if len(cat) else None, stars,
        psfsize_col=f'psfsize_{product.band.lower()}',
        fallback_arcsec=product.seeing_arcsec,
        fallback_label='provider typical')
    check_coverage(stamp, aperture_arcsec=aperture_arcsec,
                   seeing_arcsec=seeing)
    raw = np.where(np.isfinite(stamp.data), stamp.data, 0.0)
    good = stamp.good

    # Unconvolved-profile cache, shared by bands on an identical
    # instrument grid (catalog shapes are band-independent).
    cache = caches.setdefault(product.instrument, {})
    signature = (stamp.shape, round(stamp.cx, 2), round(stamp.cy, 2))
    if cache.get('signature') != signature:
        cache.clear()
        cache['signature'] = signature
        cache['profiles'] = {}
    comps = build_components(cat, stamp, psf, seeing,
                             profile_cache=cache['profiles']) \
        if len(cat) else []
    comps, consumed = apply_registry(comps, scene['registry'], stamp,
                                     psf, band_key, product.instrument,
                                     tag=tag)

    # Forced mode: pin the target to the given shape and disable the
    # standard refit -- the amplitude stays free (the joint solve owns
    # every amplitude), the profile does not.
    if target_shape is not None:
        patches['target_refit'] = False
        _pin_target(comps, target_shape, stamp, psf)

    # Stars leave the problem here.
    bg0 = bin_plane(raw, good, stamp.rr, stamp.pixscale)
    star_img, star_masks, comps, star_log = subtract_stars(
        stamp, raw, good, comps, stars, bg0['const'], tag=tag)
    image = raw - star_img

    # Seats: the reference band builds them and re-solves shapes inside
    # the alternation; transfer bands reuse them by name with fluxes
    # leashed to the reference band's solution.
    seats, drops = [], set()
    if ref is None:
        if comps:
            seats, drops = build_seats(comps, patches, stamp, image,
                                       tag=tag)
    elif ref.get('seats'):
        seats, drops = ref['seats'], set(ref['drops'])

    fit_ref = None
    if ref is not None and seats:
        fit_ref = dict(ref)
        fit_ref['col_color'] = _seat_colors(seats, cat, comps,
                                            product.band)
    fit = joint_fit(image, good, stamp, psf, comps, seats, drops,
                    ref=fit_ref, tag=tag)
    bg, track = fit['bg'], fit['track']
    solve_info = fit['solve_info']
    if solve_info is not None:
        print(f"    {tag}seat solve [{', '.join(solve_info['seats'])}]: "
              f"nfev "
              f"{'+'.join(str(n) for n in solve_info['nfev_track'])}, "
              f"{solve_info['seconds']}s last, at_bound "
              f"{solve_info['at_bound']}")
    print(f"    {tag}background track: "
          + " -> ".join(f"{v * stamp.sb:+.4f}" for v in track)
          + f" uJy/as2 ({bg['n_rej']}/{bg['n_bins']} bins rejected"
          + (f"; far {stamp.farfield_sb:+.4f}"
             if stamp.farfield_sb is not None else "") + ")")

    bases = [c['base'] for c in fit['fixed']] + fit['cols']
    base_owner = [c['name'] for c in fit['fixed']] + fit['owners']
    scene_img = np.zeros_like(image)
    neighbors = np.zeros_like(image)
    target_img = np.zeros_like(image)
    fitted_by: dict[str, np.ndarray] = {}
    for mult, base, owner in zip(fit['mults'], bases, base_owner):
        contribution = max(mult, 0.0) * base
        scene_img += contribution
        if owner == 'target':
            target_img = target_img + contribution
        else:
            neighbors += contribution
            fitted_by[owner] = fitted_by.get(owner, 0.0) + contribution

    mask, flood_ujy = build_mask(comps, fitted_by, star_masks, stamp,
                                 seeing, scene_img, neighbors, image,
                                 good, tag=tag)
    model_fill = target_img + bg['img']
    fill = twin_fill(image, neighbors, mask, good, stamp, model_fill,
                     aperture_arcsec=aperture_arcsec, tag=tag)
    contributing = good | ((stamp.rr < aperture_arcsec) & ~good)
    enc = curve(np.where(contributing, fill['filled'] - bg['img'], 0.0),
                stamp.rr, stamp.cf, rgrid)
    model_cog = curve(target_img, stamp.rr, stamp.cf, rgrid)

    # Catalog-model comparison on the scene catalog's own native band:
    # the target's catalog base remains the survey anchor even when the
    # refit seat replaced it in the fit.
    target_comp = next((c for c in comps if c['name'] == 'target'), None)
    m_ap_cat = None
    if (target_comp is not None and product.instrument.lower() == 'legacy'
            and product.band.lower() == 'r'):
        in_ap = stamp.rr < aperture_arcsec
        m_ap_cat = float((target_comp['base'] * in_ap).sum() * stamp.cf)

    witness = witness_row(enc, model_cog, m_ap_cat, stamp, good, mask,
                          fill['twin_frac'], neighbors, star_img, bg,
                          track, flood_ujy, seeing, seeing_src,
                          rgrid=rgrid, aperture_arcsec=aperture_arcsec,
                          solve_info=solve_info)
    witness['stars'] = star_log
    witness['n_comps'] = len(comps)
    witness['gated'] = [c['name'] for c in comps
                        if c['shape'] is not None and c['gate']]
    witness['seat_owners'] = sorted(drops)
    witness['registry_consumed'] = consumed
    # Aperture attribution: f_ap - m_ap_fit = (scene residual on
    # unmasked pixels) + (fill vs model on masked pixels), exactly.
    ap_good = (stamp.rr < aperture_arcsec) & good
    witness['resid_unmasked_ap_uJy'] = round(float(
        (image - scene_img - bg['img'])[ap_good & ~mask].sum()
        * stamp.cf), 1)
    witness['fill_vs_model_ap_uJy'] = round(fill['fill_vs_model_ap'], 1)
    if 'target' in drops or target_shape is not None:
        witness['target_model_uJy'] = round(
            float(target_img.sum() * stamp.cf), 1)
        if target_comp is not None and target_comp['cat'] > 0:
            witness['target_refit_x_cat'] = round(
                witness['target_model_uJy'] / target_comp['cat'], 2)

    flux_ap = enclosed_at(rgrid, enc, aperture_arcsec)
    err_ujy, err_model = flux_error(stamp, good,
                                    aperture_arcsec=aperture_arcsec)
    print(f"    {tag}f_ap = {flux_ap:.1f} uJy, excess "
          f"{witness['excess_growth_uJy']:+.1f} (own "
          f"{witness['model_own_growth_uJy']:+.1f}), "
          f"{time.time() - t0:.0f}s")

    if registry_update and fit['seats_local']:
        harvest_seats(scene['registry'], fit['seats_local'],
                      fit['seat_params'], fit['seat_amps'], stamp,
                      band_key=band_key, tag=tag)

    if dump_dir is not None:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            Path(dump_dir) / f'{band_key}_arrays.npz',
            image=image, scene=scene_img, neighbors=neighbors,
            target=target_img, bg=bg['img'], mask=mask,
            filled=fill['filled'], good=good, star_img=star_img,
            amps=np.asarray(fit['amps']),
            owners=np.array(base_owner), cx=stamp.cx, cy=stamp.cy,
            pix=stamp.pixscale, cf=stamp.cf, sigma=stamp.sigma,
            seeing=seeing)
        print(f"    {tag}dumped arrays")

    new_ref = None
    if ref is None and solve_info is not None:
        n_fixed = len(fit['fixed'])
        # Amplitudes ARE microjanskys: the reference fluxes for sibling
        # -band leashes come straight off the solve.
        new_ref = dict(seats=seats, drops=sorted(drops),
                       p=solve_info['p'], pix=stamp.pixscale,
                       col_flux=[max(float(a), 0.0)
                                 for a in fit['amps'][n_fixed:]])

    measurement = dict(
        instrument=product.instrument, band=product.band,
        wave_um=product.wave_um,
        target_ra=float(coord.ra.deg), target_dec=float(coord.dec.deg),
        flux_ujy=float(flux_ap), flux_err_ujy=float(err_ujy),
        err_model=err_model,
        rgrid=rgrid, enclosed_ujy=enc, model_cog=model_cog,
        aperture_arcsec=float(aperture_arcsec),
        witness=witness, n_comps=len(comps),
        registry_consumed=consumed,
        seeing_arcsec=float(seeing),
        pixscale=stamp.pixscale, cf=stamp.cf,
        cx=stamp.cx, cy=stamp.cy,
        image=image, scene=scene_img + bg['img'],
        filled=np.where(contributing, fill['filled'], np.nan),
        mask=mask, good=good)
    return measurement, new_ref


def _pin_target(comps: list[dict], target_shape: dict, stamp,
                psf: np.ndarray) -> None:
    """Replace the target component's profile with an explicit shape.

    Used by forced mode: the shape (n, reff_arcsec, ellip, pa_deg) is
    given, the amplitude stays free in the joint solve.
    """
    from .render import ampl_from_total, conv_same, sersic_profile
    from .sersic import theta_from_pa

    target = next((c for c in comps if c['name'] == 'target'), None)
    if target is None:
        return
    pix = stamp.pixscale
    n = float(target_shape['n'])
    reff_px = max(float(target_shape['reff_arcsec']) / pix, 0.3)
    ellip = float(target_shape['ellip'])
    pa = float(target_shape['pa_deg'])
    theta = theta_from_pa(stamp.wcs, target['x'], target['y'], pa)
    counts = max(target['cat'], 0.0) / stamp.cf
    ampl = ampl_from_total(counts, reff_px, n, ellip) if counts else 0.0
    unconv = sersic_profile([ampl, reff_px, n, ellip, theta,
                             target['x'], target['y']], stamp.shape)
    target['base'] = conv_same(unconv, psf)
    target['flux0'] = max(float(target['base'].sum()) * stamp.cf, 1e-9)
    target['shape'] = dict(reff_px=reff_px, n=n, ellip=ellip,
                           theta=theta, pa=pa)
