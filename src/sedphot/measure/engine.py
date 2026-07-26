"""
engine.py

Stage 8: Per-Galaxy Scene Measurement Driver
---------------------------------------------------------
One galaxy, all bands: fetch the scene inputs once (survey catalog,
confirmed stars, optional patches and registry), then measure every
band through the same chain --

    stamp -> PSF -> components -> registry -> stars -> seats ->
    joint fit -> mask -> residual mesh -> twin fill ->
    curve of growth -> witnesses

Bands are measured per instrument: the first band in preference order
is the REFERENCE -- it solves seat shapes -- and its siblings transfer
those shapes, re-solving neighbor seats warm with color-leashed fluxes.
Nothing is shared across instruments but the catalogs and patches.

Each measured band returns two dicts to the pipeline: the measurement
(flux, error, witnesses, and the images the QA figure draws) and, from
a reference band, the seat shapes + fluxes its instrument's sibling
bands transfer.

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
from ..catalogs.legacy import LEGACY_DR_DEFAULT, query_scene
from . import recipe
from .aperture import (build_mask, curve, empap_error, enclosed_at,
                       flux_error, twin_fill, witness_row)
from .artifacts import find_artifacts
from .background import bin_plane, residual_mesh, eval_mesh
from .components import (apply_patches, build_components, drop_target_shreds,
                         gated_row)
from .psf import resolve_psf
from .render import ampl_from_total, conv_same, sersic_profile
from .seats import apply_registry, build_seats, harvest_seats, load_registry
from .sersic import theta_from_pa
from .solve import joint_fit, pinned_fit
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
        aperture_arcsec: float,
        cutout_half_arcsec: float,
        legacy_dr: str = LEGACY_DR_DEFAULT,
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
    aperture_arcsec : float
        Science aperture radius; scopes the target-substructure rule.
    cutout_half_arcsec : float
        Stamp half-width; sizes the query cone (see recipe.QUERY_PAD_AS).
    legacy_dr : str
        Legacy data release for the scene catalog.
        [default: LEGACY_DR_DEFAULT]
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
    radius = max(recipe.QUERY_RADIUS_AS,
                 cutout_half_arcsec * np.sqrt(2.0) + recipe.QUERY_PAD_AS)
    # The cache name carries the radius only when the stamp pushed it
    # past the recipe floor: recipe-default runs keep the plain name,
    # and a larger-stamp run cannot silently reuse a smaller cone.
    suffix = ('' if radius == recipe.QUERY_RADIUS_AS
              else f'_r{int(round(radius))}')
    cat = query_scene(coord, radius, dr=legacy_dr,
                      min_flux_nmgy=recipe.TRACTOR_MIN_NMGY,
                      cache_path=cache_dir
                      / f'tractor_scene_{legacy_dr}{suffix}.csv')
    gaia_cat = gaia.query_cone(coord, radius,
                               cache_path=cache_dir
                               / f'gaia_scene{suffix}.csv')
    stars = confirm_stars(gaia_cat) if len(gaia_cat) else gaia_cat

    patches = {}
    patch_path = Path(out_dir) / recipe.PATCH_FILENAME
    if patch_path.exists():
        try:
            with open(patch_path) as handle:
                patches = json.load(handle)
        except json.JSONDecodeError as e:
            raise ValueError(f"{patch_path} is not valid JSON: {e}") from e
        known = {'replace_rows', 'free_seats', 'snap_gated',
                 'target_refit', 'target_halo', 'harvest_target',
                 'target_system', 'comment'}
        unknown = sorted(set(patches) - known)
        if unknown:
            print(f"  WARNING {recipe.PATCH_FILENAME}: unrecognized "
                  f"key(s) {unknown} ignored (known: {sorted(known)})")
        for entry in patches.get('replace_rows', []):
            missing = sorted({'ra', 'dec', 'with'} - set(entry))
            if missing:
                raise ValueError(f"{patch_path}: replace_rows entry "
                                 f"missing {missing}")
        for entry in patches.get('free_seats', []):
            missing = sorted({'ra', 'dec'} - set(entry))
            if missing:
                raise ValueError(f"{patch_path}: free_seats entry "
                                 f"missing {missing}")
        for entry in patches.get('target_system', []):
            missing = sorted({'ra', 'dec'} - set(entry))
            if missing:
                raise ValueError(f"{patch_path}: target_system entry "
                                 f"missing {missing}")
        print(f"  patches: {sorted(patches.keys())}")
    if len(cat):
        cat = apply_patches(cat, patches)
        cat = drop_target_shreds(cat, coord,
                                 aperture_arcsec=aperture_arcsec,
                                 patches=patches)
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


def _system_names(comps: list[dict], patches: dict, stamp) -> set[str]:
    """Component names belonging to the target system.

    A patches "target_system" entry declares a component (a second
    nucleus of a dumbbell, a bound companion) to be the TARGET'S OWN
    light: it keeps its own component and seat for the solve and the
    registry, but its fitted model is attributed to the target -- never
    subtracted, never masked, integrated into the aperture flux.
    """
    names = {'target'}
    for member in patches.get('target_system', []):
        mx, my = [float(v) for v in stamp.wcs.world_to_pixel(
            SkyCoord(float(member['ra']), float(member['dec']),
                     unit='deg'))]
        best, dist = None, np.inf
        for comp in comps:
            d = np.hypot(comp['x'] - mx, comp['y'] - my) * stamp.pixscale
            if d < dist:
                best, dist = comp, d
        if best is not None and dist < recipe.PATCH_MATCH_AS \
                and best['name'] != 'target':
            names.add(best['name'])
        else:
            print(f"    target_system member at ({member['ra']:.5f},"
                  f"{member['dec']:.5f}) matches no component; ignored")
    return names


def _band_colors(cat: pd.DataFrame, band: str) -> dict[int, float]:
    """Catalog row index -> this band's flux over the reference flux
    (nearest listed catalog band); clamped, neutral when unknown."""
    col = recipe.BAND_COLOR_COL.get(band.lower(), 'flux_r')
    out = {}
    for irow in range(len(cat)):
        flux_ref = float(cat.iloc[irow]['flux_r'])
        flux_band = float(cat.iloc[irow].get(col, np.nan))
        usable = (np.isfinite(flux_band) and np.isfinite(flux_ref)
                  and flux_ref > 0 and flux_band > 0)
        out[irow] = (float(np.clip(flux_band / flux_ref, 0.05, 20.0))
                     if usable else 1.0)
    return out


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


def _fit_images(fit: dict, system: set[str],
                shape_2d: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(scene, target) images for a fit: every column at its solved amplitude.

    The target image collects the declared target SYSTEM, matching the
    science path's attribution rule.
    """
    bases = [c['base'] for c in fit['fixed']] + fit['cols']
    owners = [c['name'] for c in fit['fixed']] + fit['owners']
    scene = np.zeros(shape_2d)
    target = np.zeros(shape_2d)
    for mult, base, owner in zip(fit['mults'], bases, owners):
        contribution = max(mult, 0.0) * base
        scene += contribution
        if owner in system:
            target += contribution
    return scene, target


def _used_shapes(fit: dict, seats: list[dict], stamp) -> dict | None:
    """The seat shapes this band was MEASURED with, on this band's grid.

    Always the whole seat list, however each shape got there -- solved here,
    transferred from the reference band, or frozen from a previous solve.
    That is what a reconstruction has to re-render, and it is NOT what
    `solve` records: a solve reports only the parameters it varied, so on a
    transfer band its vector covers the free (neighbor) seats alone.
    """
    params = fit.get('seat_params')
    if params is None or not seats:
        return None
    return dict(seats=[f"{s['owner']}:{s['kind']}" for s in seats],
                params=[float(v) for v in params],
                pix=float(stamp.pixscale))


def _target_gates(comps: list[dict], cat: pd.DataFrame) -> bool:
    """Whether the target's own catalog row qualifies for a shape gate.

    A gating target is one that would be seated (a free shape solve)
    when it appears as a neighbor in another field. Its per-band shape,
    solved here from its own centered best view, is worth harvesting at
    target vantage so every consumer freezes it instead of re-fitting a
    clipped copy. Judged on the raw catalog row (gated_row wants a
    non-self distance); patch-forced targets ride their own
    'harvest_target' flag, not this.
    """
    target = next((c for c in comps if c['name'] == 'target'), None)
    if target is None:
        return False
    irow = target.get('irow', -1)
    if irow is None or irow < 0 or irow >= len(cat):
        return False
    return gated_row(cat.iloc[irow], recipe.TARGET_MATCH_AS + 1.0)


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
        pin: dict | None = None,
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
    pin : dict, optional
        Rebuild a stored fit instead of solving: seat shapes, amplitudes,
        plane coefficients, mesh, and the consumed-registry snapshot all
        come from the sidecar (see solve.pinned_fit). A pinned band neither
        solves nor harvests, and `ref` must be None -- the band is treated
        as self-contained.
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
    # The catalog's per-source PSF sizes describe the scene catalog's
    # own imaging; only that instrument's bands may read them. Every
    # other instrument falls through to its header keyword or its
    # provider-typical seeing.
    psfsize_col = (f'psfsize_{product.band.lower()}'
                   if product.instrument.lower() == 'legacy' else None)
    psf, seeing, seeing_src = resolve_psf(
        stamp, cat if len(cat) else None, stars,
        psfsize_col=psfsize_col,
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
                             profile_cache=cache['profiles'],
                             gate_radius_arcsec=cutout_half_arcsec
                             - recipe.GATE_EDGE_MARGIN_AS) \
        if len(cat) else []
    system = _system_names(comps, patches, stamp)
    comps, consumed = apply_registry(
        comps, scene['registry'], stamp, psf, band_key,
        product.instrument,
        protect_px=[(c['x'], c['y']) for c in comps
                    if c['name'] in system],
        snapshot=(pin.get('consumed') if pin else None),
        tag=tag)

    # Forced mode: pin the target to the given shape and disable the
    # standard refit -- the amplitude stays free (the joint solve owns
    # every amplitude), the profile does not.
    if target_shape is not None:
        patches['target_refit'] = False
        _pin_target(comps, target_shape, stamp, psf)

    # Artifacts: mask, never fit. Pixels far beyond both the sky and
    # the catalog scene's own claim (bleed trails, satellite streaks)
    # leave the usable map exactly like nodata. A catalog wreck
    # (rchisq_r past the gate ceiling) is the catalog's echo of the
    # artifact itself and may not claim pixels here; a component whose
    # claim the mask mostly OWNS leaves the scene with it, and the
    # coverage gate re-judges the aperture.
    artifact_as2, artifact_ujy = 0.0, 0.0
    artifact_mask = None
    if comps:
        chi_cols = [c for c in ('rchisq_g', 'rchisq_r', 'rchisq_i',
                                'rchisq_z') if c in cat]
        wrecks = (set(cat.index[(cat[chi_cols]
                                 > recipe.GATE_RCHISQ_MAX).any(axis=1)])
                  if chi_cols else set())
        pred = np.zeros_like(raw)
        for c in comps:
            if c['irow'] not in wrecks:
                pred += c['base']
        art, artifact_as2, flood_as2 = find_artifacts(
            raw, good, pred, stamp.rr, stamp.sigma, stamp.pixscale,
            sb=stamp.sb)
        if artifact_as2 > 0:
            artifact_mask = art
            artifact_ujy = float(raw[art].sum() * stamp.cf)
            good = good & ~art
            # Void by footprint OWNERSHIP, never center membership: a
            # component leaves the scene only when the mask covers the
            # MAJORITY of its claim (its render above sigma). A broad
            # source nicked by a narrow artifact keeps its column and
            # solves from its clean pixels -- the design excises the
            # masked rows -- while a row whose light sits wholly
            # inside the artifact is its catalog echo and goes with it.
            ny_a, nx_a = art.shape
            voided = []
            for c in comps:
                if c['name'] in system:
                    continue
                claim = c['base'] > stamp.sigma
                if claim.any():
                    owned = float(art[claim].mean())
                else:
                    cy_i, cx_i = int(round(c['y'])), int(round(c['x']))
                    owned = (float(art[cy_i, cx_i])
                             if 0 <= cy_i < ny_a and 0 <= cx_i < nx_a
                             else 0.0)
                if owned > 0.5:
                    voided.append(c['name'])
            if voided:
                comps = [c for c in comps if c['name'] not in voided]
            print(f"    {tag}artifact: {artifact_as2:.0f} as2 "
                  f"({artifact_ujy:.0f} uJy) masked, never fit"
                  + (f", {flood_as2:.0f} as2 of it skirt flood"
                     if flood_as2 > 0 else "")
                  + (f"; voids {voided}" if voided else ""))
            check_coverage(stamp, aperture_arcsec=aperture_arcsec,
                           seeing_arcsec=seeing, nodata=~good)

    # Stars leave the problem here.
    bg0 = bin_plane(raw, good, stamp.rr, stamp.pixscale)
    star_img, star_masks, comps, star_log = subtract_stars(
        stamp, raw, good, comps, stars, bg0['const'],
        colors=_band_colors(cat, product.band) if len(cat) else None,
        aperture_arcsec=aperture_arcsec, tag=tag)
    image = raw - star_img

    # A masked-mode star (in-zone revert) has no column and nothing
    # subtracted: its light is still in the pixels, and an unmodeled
    # bright source in the fitting pixels pulls the target refit and
    # the background. Its predicted footprint leaves the solve's pixel
    # set; the measurement-side mask and fill cover it after the fit.
    good_fit = good
    masked_mode = {rec['comp'] for rec in star_log
                   if rec.get('mode') == 'masked'}
    if masked_mode:
        exclude = np.zeros_like(good)
        for name, footprint in star_masks:
            if name in masked_mode:
                exclude |= footprint > recipe.K_ISO * stamp.sigma
        good_fit = good & ~exclude
        print(f"    {tag}solve excludes {int(exclude.sum())} px under "
              f"masked star(s) {sorted(masked_mode)}")

    # Seats: the reference band builds them and re-solves shapes inside
    # the alternation; transfer bands reuse them by name with fluxes
    # leashed to the reference band's solution.
    seats, drops = [], set()
    if ref is None:
        if comps:
            seats, drops = build_seats(comps, patches, stamp, image,
                                       tag=tag)
            # Mark which seats are the target's OWN light. A declared
            # target_system member is attributed to the target and never
            # subtracted, so its shape belongs to the measurement definition
            # and transfer bands freeze it with the target (solve._target_side).
            # Only the engine knows the system, so it tags the seats here and
            # the tag rides ref['seats'] to every sibling band.
            for seat in seats:
                seat['system'] = seat['owner'] in system
    elif ref.get('seats'):
        seats, drops = ref['seats'], set(ref['drops'])

    fit_ref = None
    if ref is not None and seats:
        fit_ref = dict(ref)
        fit_ref['col_color'] = _seat_colors(seats, cat, comps,
                                            product.band)
    if pin is not None:
        # Pinned reconstruction: rebuild the stored fit with no solve and
        # no harvest. Every shape, amplitude, and plane coefficient comes
        # from the sidecar; only the render is redone.
        fit = pinned_fit(image, good_fit, stamp, psf, comps, seats, drops,
                         pin=pin)
        harvest_fit, fit_free, target_gates = fit, None, False
    else:
        # The science flux always comes from `fit`: forced-target on transfer
        # bands, so colors stay shape-consistent. A GATING target's transfer
        # band also needs its own free per-band shape (Solve 1) -- the
        # data-driven envelope every neighbor's forced photometry consumes
        # must come from this object's own centered best view, not the
        # frozen reference shape. For a real color gradient that differs from
        # the transferred shape and fits the halo better (a genuinely compact
        # z-envelope under an extended r, say). The plane background cannot
        # absorb a radial halo, so the free solve cannot cheaply collapse
        # when the envelope is truly extended.
        #
        # The reference band solved the target free already, so it harvests
        # itself and needs no second pass.
        target_gates = _target_gates(comps, cat)
        two_solve = ref is not None and target_gates and bool(seats)
        fit_free = None
        if two_solve and recipe.NEIGHBOR_SHAPE_FROM_FREE_SOLVE:
            # Free first, then freeze: the free solve settles every seat, and
            # the science pass adopts its NEIGHBOR shapes and re-freezes the
            # target, leaving only amplitudes and background to solve. One
            # nonlinear solve, one neighbor shape per band -- stored and used
            # -- and the same treatment a neighbor gets when its shape came
            # from the registry instead.
            fit_free = joint_fit(image, good_fit, stamp, psf, comps, seats,
                                 drops, ref=fit_ref, free_target=True)
            fit = joint_fit(image, good_fit, stamp, psf, comps, seats, drops,
                            ref=fit_ref,
                            freeze_neighbors=fit_free['seat_params'])
            harvest_fit = fit_free
        else:
            fit = joint_fit(image, good_fit, stamp, psf, comps, seats, drops,
                            ref=fit_ref)
            if not two_solve:
                harvest_fit = fit
            else:
                fit_free = joint_fit(image, good_fit, stamp, psf, comps,
                                     seats, drops, ref=fit_ref,
                                     free_target=True)
                harvest_fit = fit_free
    bg, track = fit['bg'], fit['track']
    solve_info = fit['solve_info']
    # The free-target solve's own scene, built once and shared by the
    # witness and the QA dump. Its shape vector and model curve are the
    # ONLY record of the per-band free shape outside the mutable registry,
    # and a reconstruction must read this galaxy's own immutable sidecar.
    free_scene = free_target = free_cog = None
    if fit_free is not None:
        free_scene, free_target = _fit_images(fit_free, system, image.shape)
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
    # An amplitude pinned at its leash bound is a WITNESS: the solve wanted
    # a value the leash forbade (a mis-scaled star leash, a registry anchor
    # the data contradict, a frozen seat whose shape the band disagrees
    # with). Unrecorded, a column parked exactly at its bound is
    # indistinguishable from a converged fit. Checked over EVERY column,
    # seats included -- a frozen seat is precisely the case where the
    # amplitude is the only freedom left, so it is the one that must not go
    # unwatched.
    leashed_at_bound = []
    for owner, amp, band_lohi in zip(base_owner, fit['amps'],
                                     fit.get('amp_bounds') or []):
        lo, hi = band_lohi
        span = (hi - lo) if (lo is not None and hi is not None) else None
        if span and span > 0 and (amp - lo < 1e-3 * span
                                  or hi - amp < 1e-3 * span):
            leashed_at_bound.append(owner)
    if leashed_at_bound:
        print(f"    {tag}leash-bound amps: {sorted(leashed_at_bound)}")
    scene_img = np.zeros_like(image)
    neighbors = np.zeros_like(image)
    target_img = np.zeros_like(image)
    own_target_ujy = 0.0
    fitted_by: dict[str, np.ndarray] = {}
    for mult, base, owner in zip(fit['mults'], bases, base_owner):
        contribution = max(mult, 0.0) * base
        scene_img += contribution
        if owner in system:
            target_img = target_img + contribution
            if owner == 'target':
                own_target_ujy += float(contribution.sum()) * stamp.cf
        else:
            neighbors += contribution
            fitted_by[owner] = fitted_by.get(owner, 0.0) + contribution

    mask, flood_ujy = build_mask(comps, fitted_by, star_masks, stamp,
                                 seeing, scene_img, neighbors, image,
                                 good, tag=tag)
    # Post-fit residual mesh: the bin-level surface of the light no
    # model claimed, subtracted only inside the curve of growth. Every
    # fitted component (the target's own envelope included) is already
    # out of the residual, so a well-fit source leaves the mesh
    # nothing to take; its resolution floor (~2 bins) makes compact
    # light invisible to it either way.
    mesh_state: dict = {}
    if pin is not None:
        # Reconstruction reuses the stored mesh surface verbatim.
        mesh = eval_mesh(pin.get('mesh'), image.shape)
    else:
        mesh = residual_mesh(image - scene_img - bg['img'], good & ~mask,
                             stamp.pixscale, level_px=bg.get('keep_px'),
                             state=mesh_state)
    mesh_ap_ujy = float(mesh[stamp.rr < aperture_arcsec].sum()
                        * stamp.cf)
    model_fill = target_img + bg['img']
    # Artifact holes ride the SAME fill channel as neighbor masks, at
    # every radius: left out, the curve of growth would count masked
    # artifact pixels as zero flux and bias every witness measured
    # beyond them.
    fill_mask = mask if artifact_mask is None else (mask | artifact_mask)
    fill = twin_fill(image, neighbors, fill_mask, good, stamp, model_fill,
                     aperture_arcsec=aperture_arcsec, tag=tag)
    contributing = good | ((stamp.rr < aperture_arcsec) & ~good)
    if artifact_mask is not None:
        contributing = contributing | artifact_mask
    enc = curve(np.where(contributing,
                         fill['filled'] - bg['img'] - mesh, 0.0),
                stamp.rr, stamp.cf, rgrid)
    model_cog = curve(target_img, stamp.rr, stamp.cf, rgrid)
    if free_target is not None:
        free_cog = curve(free_target, stamp.rr, stamp.cf, rgrid)

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
                          solve_info=solve_info,
                          solve_free=(fit_free['solve_info']
                                      if fit_free is not None else None),
                          shapes=_used_shapes(fit, seats, stamp))
    witness['stars'] = star_log
    witness['n_comps'] = len(comps)
    witness['artifact_as2'] = round(artifact_as2, 1)
    witness['artifact_uJy'] = round(artifact_ujy, 1)
    if artifact_as2 > 0:
        witness['artifact_flood_as2'] = round(flood_as2, 1)
    witness['mesh_ap_uJy'] = round(mesh_ap_ujy, 1)
    witness['leash_bound'] = sorted(leashed_at_bound)
    # The fit state: everything a re-derivation needs without a solver
    # -- a new aperture reads the enclosed curve, a scene rebuild takes
    # the amplitudes + solve params + plane coefficients + mesh grid.
    witness['fit_state'] = dict(
        amps=[[owner, round(float(amp), 3)]
              for owner, amp in zip(base_owner, fit['amps'])],
        bg_coefs=[round(float(v), 8) for v in bg['coefs']],
        mesh=mesh_state or None,
        rgrid=[float(r) for r in rgrid],
        enclosed_uJy=[round(float(v), 2) for v in enc],
        model_cog_uJy=[round(float(v), 2) for v in model_cog])
    if free_cog is not None:
        # The free-target model's own curve: what a per-band-shape
        # re-report reads, where model_cog_uJy is the forced-shape one.
        witness['fit_state']['model_cog_free_uJy'] = [
            round(float(v), 2) for v in free_cog]
    witness['gated'] = [c['name'] for c in comps
                        if c['shape'] is not None and c['gate']]
    witness['seat_owners'] = sorted(drops)
    witness['registry_consumed'] = consumed
    # Aperture attribution: f_ap - m_ap_fit = (scene residual on
    # unmasked pixels) + (fill vs model on masked pixels), exactly.
    ap_good = (stamp.rr < aperture_arcsec) & good
    witness['resid_unmasked_ap_uJy'] = round(float(
        (image - scene_img - bg['img'] - mesh)[ap_good & ~mask].sum()
        * stamp.cf), 1)
    witness['fill_vs_model_ap_uJy'] = round(fill['fill_vs_model_ap'], 1)
    if ('target' in drops or target_shape is not None) \
            and target_comp is not None:
        # System total: declared members count as target light (this is
        # the flux a forced/sersic-mode row reports).
        witness['target_model_uJy'] = round(
            float(target_img.sum() * stamp.cf), 1)
        if free_target is not None:
            witness['target_model_free_uJy'] = round(
                float(free_target.sum() * stamp.cf), 1)
        # The refit-vs-catalog ratio only means something on the scene
        # catalog's own band, and compares the TARGET'S OWN seats to
        # its own catalog row -- system members would read as runaway
        # refit.
        if m_ap_cat is not None and target_comp['cat'] > 0:
            witness['target_refit_x_cat'] = round(
                own_target_ujy / target_comp['cat'], 2)

    flux_ap = enclosed_at(rgrid, enc, aperture_arcsec)
    err_ujy, err_model = flux_error(stamp, good,
                                    aperture_arcsec=aperture_arcsec)
    # The reported error is MEASURED: source-free aperture sums of the
    # final residual carry the correlation, structure, and confusion a
    # white-noise propagation is blind to. The larger of the two wins;
    # a stamp too small to place enough apertures keeps the white-
    # noise value under its own label.
    resid_final = np.where(good, image - scene_img - bg['img'] - mesh,
                           0.0)
    emp_err, n_emp = empap_error(resid_final, good & ~mask, stamp,
                                 aperture_arcsec=aperture_arcsec)
    if n_emp >= recipe.EMPAP_MIN_APS and emp_err > err_ujy:
        err_ujy, err_model = emp_err, 'empap'
    print(f"    {tag}f_ap = {flux_ap:.1f} +- {err_ujy:.1f} uJy "
          f"({err_model}), excess "
          f"{witness['excess_growth_uJy']:+.1f} (own "
          f"{witness['model_own_growth_uJy']:+.1f}), mesh "
          f"{mesh_ap_ujy:+.1f}, {time.time() - t0:.0f}s")

    if registry_update and pin is None and harvest_fit['seats_local']:
        h_si = harvest_fit['solve_info']
        health = dict(capped=h_si['status'] == 0) if h_si is not None else None
        harvest_seats(scene['registry'], harvest_fit['seats_local'],
                      harvest_fit['seat_params'], harvest_fit['seat_amps'],
                      stamp, band_key=band_key,
                      seat_col_flux=harvest_fit['col_flux'],
                      include_target=(bool(patches.get('harvest_target'))
                                      or target_gates),
                      solve_health=health, tag=tag)

    if dump_dir is not None:
        Path(dump_dir).mkdir(parents=True, exist_ok=True)
        # The RAW free-target solve (fit_free), engine-rendered, so QA
        # sees the free shape even when the guard rejected it; falls back
        # to the harvested fit where there was no free pass.
        src = fit_free if fit_free is not None else harvest_fit
        if free_target is None:
            free_scene, free_target = _fit_images(src, system, image.shape)
        np.savez_compressed(
            Path(dump_dir) / f'{band_key}_arrays.npz',
            image=image, scene=scene_img, neighbors=neighbors,
            target=target_img, bg=bg['img'], mask=mask,
            free_target=free_target, free_scene=free_scene,
            free_bg=src['bg']['img'],
            filled=fill['filled'], good=good, star_img=star_img,
            amps=np.asarray(fit['amps']),
            owners=np.array(base_owner), cx=stamp.cx, cy=stamp.cy,
            pix=stamp.pixscale, cf=stamp.cf, sigma=stamp.sigma,
            seeing=seeing)
        print(f"    {tag}dumped arrays")

    new_ref = None
    if ref is None and solve_info is not None:
        n_fixed = len(fit['fixed'])
        # Amplitudes ARE microjanskys: the reference fluxes for the
        # sibling-band leashes come straight off the solve.
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
        image=image, scene=scene_img + bg['img'], mesh=mesh,
        filled=np.where(contributing, fill['filled'], np.nan),
        mask=mask, good=good, artifact=artifact_mask)
    return measurement, new_ref


def _pin_target(comps: list[dict], target_shape: dict, stamp,
                psf: np.ndarray) -> None:
    """Replace the target component's profile with an explicit shape.

    Used by forced mode: the shape (n, reff_arcsec, ellip, pa_deg) is
    given, the amplitude stays free in the joint solve.
    """
    target = next((c for c in comps if c['name'] == 'target'), None)
    if target is None:
        return
    pix = stamp.pixscale
    n = float(target_shape['n'])
    reff_px = max(float(target_shape['reff_arcsec']) / pix, 0.3)
    ellip = float(target_shape['ellip'])
    pa = float(target_shape['pa_deg'])
    theta = theta_from_pa(stamp.wcs, target['x'], target['y'], pa)
    # Render at a positive amplitude even for a target undetected in
    # the scene band: the design normalizes every column to unit
    # in-stamp flux, and a zero render would pin the amplitude at zero
    # instead of leaving it free.
    counts = max(target['cat'] / stamp.cf, 1.0)
    ampl = ampl_from_total(counts, reff_px, n, ellip)
    unconv = sersic_profile([ampl, reff_px, n, ellip, theta,
                             target['x'], target['y']], stamp.shape)
    target['base'] = conv_same(unconv, psf)
    target['flux0'] = max(float(target['base'].sum()) * stamp.cf, 1e-9)
    target['shape'] = dict(reff_px=reff_px, n=n, ellip=ellip,
                           theta=theta, pa=pa)
