"""
seats.py

Stage 5: Seats and the Cross-Field Registry
---------------------------------------------------------
A "seat" is a component whose shape parameters enter the nonlinear
solve. Standard, catalog-driven seats: every gated component gets a
Sersic core (small center box), and the target gets a refit seat -- the
catalog informs the photometry only through the neighbors, never
through the target itself. recipe.GATED_HALO adds a Nuker halo (wide
center box under an ownership penalty) to every gated seat; by default
the halo belongs to the target alone, granted through the patches
file's "target_halo". Custom, per-galaxy seats (all optional, via the
patches file): free single-Sersic seats for companion nuclei or retyped
rows, center snapping, disabling the target refit.

Seat dict:
    kind       'sersic' or 'nuker'
    owner      component name whose fixed base this seat replaces
    ra, dec    seat anchor (sky degrees); centers resolve per band
    p0, lo, hi parameter seed and bounds, recipe.SEAT_NPARAMS each:
               sersic (reff_px, n, ellip, pa_deg, dx_px, dy_px)
               nuker  (rb_px, beta, ellip, pa_deg, dx_px, dy_px)
               radial and offset entries are reference-band pixels; pa
               is sky-frame degrees.

The registry has two sides: harvest_seats WRITES a solved seat's shape,
center, and per-band flux into it; apply_registry READS an entry back
into a later field as fixed components (consuming it).

The registry lets a source solved once be reused by every later field
that contains it: every stored record is consumed FROZEN -- fixed,
tightly amplitude-leashed components; gates die with the consumed
rows. The one unfreeze is the HOME field: an entry anchored on the
consuming field's own target is never consumed (the protect rule), so
the home solve runs fresh and its harvest UPGRADES the entry. The
VANTAGE grade ("the target fit is the best fit") governs write
priority, not consumption strength:

    target vantage     written by the source's own field
                       (include_target) -- the one centered view.
                       Never overwritten by a neighbor solve.
    neighbor vantage   written by a field that saw the source as a
                       neighbor; replaced when the home field runs.
    tombstone          the solve burned its evaluation cap. Consumers
                       kill the gate -- the catalog render stands --
                       so a doomed solve is never repeated field after
                       field; only the home vantage retries.

A record whose solved flux is below the margin floor is NEVER stored:
zero amplitude is a two-part verdict -- no such component here, and
the attached shape was never constrained -- so it can neither be
frozen into another scene nor offered new flux. Dropped or re-solved
from scratch, nothing in between.

Entries transport the full per-band decomposition: per-band shapes
(arcsec, sky PA), solved centers as sky coordinates, and per-band
solved fluxes as the amplitude anchor.

Data products:
    <registry>.json   the cross-field registry, rewritten whole by
                      save_registry when the caller asks for it

Requirements:
    numpy, scipy, astropy

Notes:
    Registry entries are keyed by a position-derived name, so re-solving
    the same physical source from any field updates the same entry.
    The registry is mutable state shared by every galaxy measured
    against it. harvest_seats updates the loaded dict in memory as each
    band solves; the file is rewritten once, at the end of the run, and
    only when the caller passes registry_update (CLI --registry-update).
    Two consequences: measuring galaxy B after galaxy A can change B's
    flux, because a shared neighbor A registered is frozen into B's
    scene instead of solved there; and updates are last-writer-wins with
    no locking, so sweeps must run one galaxy at a time.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from scipy.ndimage import gaussian_filter, maximum_filter

from . import recipe
from .render import (pa_map, render_nuker, render_sersic_boxed,
                     sersic_profile, sersic_total)
from .stamp import Stamp


# ------------------------------------
# Implementation constants
# ------------------------------------
# Smallest permitted seat size (reference-band pixels).
SEAT_REFF_MIN_PX = 0.8


def seat_slices(seats: list[dict]) -> list[slice]:
    """Parameter-vector slice of each seat, in seat order."""
    out = []
    for i in range(len(seats)):
        out.append(slice(i * recipe.SEAT_NPARAMS,
                         (i + 1) * recipe.SEAT_NPARAMS))
    return out


def snap_to_peak(image: np.ndarray, x0: float, y0: float,
                 pixscale: float) -> tuple[float, float]:
    """Nearest local maximum of the lightly smoothed image.

    Nearest, NOT brightest-in-box: the brightest pixel in a search box
    around a blended source is usually the slope toward a brighter
    neighbor, not the source's own peak.
    """
    smoothed = gaussian_filter(image, 1.0)
    box = int(round(recipe.SNAP_BOX_AS / pixscale))
    ys0 = max(int(y0) - box, 0)
    xs0 = max(int(x0) - box, 0)
    sub = smoothed[ys0:int(y0) + box + 1, xs0:int(x0) + box + 1]
    is_peak = sub == maximum_filter(sub, size=3)
    pys, pxs = np.where(is_peak)
    if len(pys) == 0:
        return float(x0), float(y0)
    dist = np.hypot(pys + ys0 - y0, pxs + xs0 - x0)
    j = int(np.argmin(dist))
    return float(pxs[j] + xs0), float(pys[j] + ys0)


def _seat(kind: str, owner: str, wcs, x: float, y: float,
          p0: list, lo: list, hi: list) -> dict:
    sky = wcs.pixel_to_world(x, y)
    return dict(kind=kind, owner=owner, ra=float(sky.ra.deg),
                dec=float(sky.dec.deg), p0=list(p0), lo=list(lo),
                hi=list(hi))


# ------------------------------------
# Seat construction
# ------------------------------------
def build_seats(
        comps: list[dict],
        patches: dict,
        stamp: Stamp,
        image: np.ndarray,
        *,
        tag: str = '',
) -> tuple[list[dict], set[str]]:
    """Build the seat list and the set of component names it replaces.

    Seat centers are stored as sky coordinates so every band renders
    the same scene on its own grid.

    Parameters
    ----------
    comps : list of dict
        Scene components (components.build_components).
    patches : dict
        Per-galaxy custom inputs ({} for pure catalog behavior).
    stamp : Stamp
        The reference band's stamp.
    image : np.ndarray
        Star-subtracted image (counts); only used by snap-to-peak.
    tag : str
        Run-log prefix.

    Returns
    -------
    seats : list of dict
        Seat definitions in solve order.
    drops : set of str
        Component names whose fixed catalog base the seats replace.
    """
    pix = stamp.pixscale
    wcs = stamp.wcs
    dxy_out = recipe.DXY_OUT_AS / pix
    dxy = recipe.SEAT_DXY_AS / pix
    n_lo, n_hi = recipe.SERSIC_N_RANGE
    seats: list[dict] = []
    drops: set[str] = set()

    snap_gated = bool(patches.get('snap_gated'))
    for comp in comps:
        if comp['shape'] is None or not comp['gate']:
            continue
        x, y = ((comp['x'], comp['y']) if not snap_gated
                else snap_to_peak(image, comp['x'], comp['y'], pix))
        if snap_gated:
            moved = np.hypot(x - comp['x'], y - comp['y']) * pix
            if moved > 0.05:
                print(f"    {tag}snap {comp['name']}: {moved:.2f}\" to peak")
        shape = comp['shape']
        core_hi = recipe.GATED_CORE_REFF_MAX_AS / pix
        p0_core = [min(shape['reff_px'], 0.95 * core_hi),
                   min(shape['n'], 5.9), min(shape['ellip'], 0.91),
                   shape['pa'], 0.0, 0.0]
        # A seat whose profile cannot land real light on the stamp must
        # not get a design column: normalized to unit in-stamp flux, an
        # empty render is a numerically explosive basis. Same rule as
        # the fixed-component margin prune.
        profile = sersic_profile([1.0] + p0_core[:3] + [p0_core[3], x, y],
                                 image.shape)
        frac = float(profile.sum()) / max(
            sersic_total(1.0, p0_core[0], p0_core[1], p0_core[2], 1.0),
            1e-12)
        seated = False
        if comp['cat'] * frac < recipe.MARGIN_MIN_UJY:
            print(f"    {tag}{comp['name']}: gated core seat has "
                  f"{comp['cat'] * frac:.2g} uJy in-stamp at catalog "
                  f"amplitude; seat pruned")
        else:
            seats.append(_seat(
                'sersic', comp['name'], wcs, x, y, p0_core,
                [SEAT_REFF_MIN_PX, n_lo, 0.0,
                 p0_core[3] - recipe.PA_BOX_DEG, -dxy, -dxy],
                [core_hi, n_hi, recipe.SERSIC_E_MAX,
                 p0_core[3] + recipe.PA_BOX_DEG, dxy, dxy]))
            seated = True
        if recipe.GATED_HALO:
            p0_halo = [recipe.NUKER_RB0_AS / pix, 2.0, shape['ellip'],
                       shape['pa'], 0.0, 0.0]
            seats.append(_seat(
                'nuker', comp['name'], wcs, x, y, p0_halo,
                [recipe.NUKER_RB_AS[0] / pix, recipe.NUKER_BETA[0], 0.0,
                 p0_halo[3] - recipe.PA_BOX_DEG, -dxy_out, -dxy_out],
                [recipe.NUKER_RB_AS[1] / pix, recipe.NUKER_BETA[1],
                 recipe.NUKER_E_MAX,
                 p0_halo[3] + recipe.PA_BOX_DEG, dxy_out, dxy_out]))
            seated = True
        # Drop the catalog base only if a seat replaced it; a pruned core
        # with no halo leaves the row as its fixed catalog render.
        if seated:
            drops.add(comp['name'])

    # Custom free seats: single-Sersic seats granted by the patches
    # file to named positions (companion nuclei, retyped rows).
    for request in patches.get('free_seats', []):
        fx, fy = [float(v) for v in wcs.world_to_pixel(
            SkyCoord(request['ra'], request['dec'], unit='deg'))]
        candidates = [c for c in comps if c['name'] not in drops
                      and c['name'] != 'target' and not c.get('reg')]
        if not candidates:
            continue
        best = min(candidates,
                   key=lambda c: np.hypot(c['x'] - fx, c['y'] - fy))
        if np.hypot(best['x'] - fx, best['y'] - fy) * pix > recipe.PATCH_MATCH_AS:
            print(f"    {tag}free seat at ({request['ra']:.5f},"
                  f"{request['dec']:.5f}) matches no component; skipped")
            continue
        if request.get('snap'):
            sx, sy = snap_to_peak(image, fx, fy, pix)
            target = next((c for c in comps if c['name'] == 'target'),
                          None)
            # On a coarse or blurry grid the peak of a tight blend IS
            # the target; snapping there would seat this component on
            # top of the target and solve it to nothing. Keep the
            # requested position instead -- declared positions beat a
            # blended peak.
            if target is not None and np.hypot(
                    sx - target['x'], sy - target['y']) * pix \
                    < recipe.TARGET_MATCH_AS:
                print(f"    {tag}free seat snap landed on the target "
                      f"(blended peak); keeping the requested position")
            else:
                fx, fy = sx, sy
        shape = best['shape']
        reff_hi = recipe.FREE_SEAT_REFF_MAX_AS / pix
        if shape is not None:
            p0 = [min(shape['reff_px'], 0.98 * reff_hi),
                  min(shape['n'], 5.9), min(shape['ellip'], 0.91),
                  shape['pa'], 0.0, 0.0]
        else:
            p0 = [1.2 / pix, 2.0, 0.1, 0.0, 0.0, 0.0]
        seats.append(_seat(
            'sersic', best['name'], wcs, fx, fy, p0,
            [SEAT_REFF_MIN_PX, n_lo, 0.0, p0[3] - recipe.PA_BOX_DEG,
             -dxy, -dxy],
            [reff_hi, n_hi, recipe.SERSIC_E_MAX,
             p0[3] + recipe.PA_BOX_DEG, dxy, dxy]))
        drops.add(best['name'])

    # The target refit is STANDARD: the target's shape is always solved
    # from the data, so the catalog informs the photometry only through
    # the neighbors, never through the target itself. A patch may set
    # "target_refit": false to disable it per galaxy, or
    # "target_halo": true to grant the target the gated-style core +
    # halo pair instead -- for a target whose own envelope must be
    # solved in its own centered field (the one field that sees the
    # whole envelope) rather than inferred from a neighbor's clipped
    # view. A catalog with no row at the target position has nothing
    # to refit.
    target = next((c for c in comps if c['name'] == 'target'), None)
    if patches.get('target_refit', True) and target is not None:
        shape = target['shape']
        x, y = target['x'], target['y']
        if patches.get('target_halo') and shape is not None:
            core_hi = recipe.GATED_CORE_REFF_MAX_AS / pix
            p0_core = [min(shape['reff_px'], 0.95 * core_hi),
                       min(shape['n'], 5.9), min(shape['ellip'], 0.91),
                       shape['pa'], 0.0, 0.0]
            seats.append(_seat(
                'sersic', 'target', wcs, x, y, p0_core,
                [SEAT_REFF_MIN_PX, n_lo, 0.0,
                 shape['pa'] - recipe.PA_BOX_DEG, -dxy, -dxy],
                [core_hi, n_hi, recipe.SERSIC_E_MAX,
                 shape['pa'] + recipe.PA_BOX_DEG, dxy, dxy]))
            seats.append(_seat(
                'nuker', 'target', wcs, x, y,
                [recipe.NUKER_RB0_AS / pix, 2.0, shape['ellip'],
                 shape['pa'], 0.0, 0.0],
                [recipe.NUKER_RB_AS[0] / pix, recipe.NUKER_BETA[0], 0.0,
                 shape['pa'] - recipe.PA_BOX_DEG, -dxy_out, -dxy_out],
                [recipe.NUKER_RB_AS[1] / pix, recipe.NUKER_BETA[1],
                 recipe.NUKER_E_MAX,
                 shape['pa'] + recipe.PA_BOX_DEG, dxy_out, dxy_out]))
            drops.add('target')
            return seats, drops
        if patches.get('target_halo'):
            print(f"    {tag}target_halo needs an extended target "
                  f"shape; falling back to the standard refit")
        reff_hi = recipe.REFIT_REFF_MAX_AS / pix
        if shape is not None:
            p0 = [min(shape['reff_px'], 0.98 * reff_hi),
                  min(shape['n'], 5.9), min(shape['ellip'], 0.91),
                  shape['pa'], 0.0, 0.0]
        else:
            p0 = [1.2 / pix, 2.0, 0.1, 0.0, 0.0, 0.0]
        seats.append(_seat(
            'sersic', 'target', wcs, x, y, p0,
            [SEAT_REFF_MIN_PX, n_lo, 0.0, p0[3] - recipe.PA_BOX_DEG,
             -dxy, -dxy],
            [reff_hi, n_hi, recipe.SERSIC_E_MAX,
             p0[3] + recipe.PA_BOX_DEG, dxy, dxy]))
        drops.add('target')
    return seats, drops


# ------------------------------------
# Registry: consume
# ------------------------------------
def apply_registry(
        comps: list[dict],
        registry: dict,
        stamp: Stamp,
        psf: np.ndarray,
        band_key: str,
        instrument: str,
        *,
        protect_px: list[tuple[float, float]] | None = None,
        snapshot: list | None = None,
        tag: str = '',
) -> tuple[list[dict], list[str]]:
    """Replace catalog rows with a registry entry's frozen components.

    Matched catalog rows are dropped in a first pass (so entries cannot
    eat each other's components), then every frozen component is added
    as a fixed, tightly-leashed design column. Gates die with the rows
    they belonged to. Entries anchored at the target -- or at any
    protect_px position (declared target-system members) -- are never
    consumed: this field measures those fresh.

    Parameters
    ----------
    comps : list of dict
        Scene components for this band.
    registry : dict
        Loaded registry ({} disables consumption).
    stamp : Stamp
        This band's stamp.
    psf : np.ndarray
        This band's PSF kernel.
    band_key : str
        Band label ('Legacy_r' style); the per-band component list is
        looked up by it, falling back to the instrument-level list.
    instrument : str
        Instrument label, the fallback lookup key.
    protect_px : list of (x, y), optional
        Extra stamp-pixel positions that are never consumed, on top of
        the target's own (declared target-system members).
    snapshot : list, optional
        A sidecar's registry_consumed record. When given it replaces the
        live registry entirely, so a pinned reconstruction consumes the
        shapes its fit actually used. [default: use `registry`]
    tag : str
        Run-log prefix.

    Returns
    -------
    comps : list of dict
        Component list with replacements applied.
    consumed : list of str
        Names of the registry entries consumed (frozen).
    """
    if not registry and not snapshot:
        return comps, []
    wcs = stamp.wcs
    pix = stamp.pixscale
    cf = stamp.cf
    shape_2d = stamp.shape
    ny, nx = shape_2d
    margin_px = recipe.MARGIN_AS / pix
    protect = [(c['x'], c['y']) for c in comps if c['name'] == 'target']
    protect.extend(protect_px or [])

    def _anchor_px(entry):
        return [float(v) for v in wcs.world_to_pixel(
            SkyCoord(entry['ra'], entry['dec'], unit='deg'))]

    def _protected(x0, y0):
        return any(np.hypot(x0 - px, y0 - py) * pix
                   < recipe.TARGET_MATCH_AS for px, py in protect)

    def _nearest_comp(pool, x0, y0):
        best, bestd = None, recipe.REGISTRY_MATCH_AS
        for comp in pool:
            if comp['name'] == 'target' or comp.get('reg'):
                continue
            d = np.hypot(comp['x'] - x0, comp['y'] - y0) * pix
            if d < bestd:
                best, bestd = comp, d
        return best

    live, tomb_entries = [], []
    # Sidecar-first: a pinned reconstruction consumes the frozen shapes the
    # fit stored (the registry_consumed snapshot), so it reproduces even
    # after the mutable registry is rewritten. No snapshot -> live registry.
    if snapshot is not None:
        for entry in snapshot:
            clist = entry.get('shapes')
            if not clist:
                continue
            x0, y0 = [float(v) for v in wcs.world_to_pixel(
                SkyCoord(clist[0]['ra'], clist[0]['dec'], unit='deg'))]
            live.append((entry['name'], clist, x0, y0))
    for name, entry in (registry.items() if snapshot is None else ()):
        by_band = entry.get('components') or {}
        clist = by_band.get(band_key) or by_band.get(instrument)
        tomb = ((entry.get('tombstones') or {}).get(band_key)
                or (entry.get('tombstones') or {}).get(instrument))
        if not clist and not tomb:
            if by_band:
                print(f"    {tag}registry: {name} has no components "
                      f"for {band_key} or {instrument}; not consumed")
            continue
        x0, y0 = _anchor_px(entry)
        # An entry at a protected position is this field's own target
        # (or a declared member of it) seen from another field. The
        # target is never frozen: it is measured fresh here, and
        # consuming its entry would add a leashed copy on top of the
        # free one and split the light between them. This is the one
        # UNFREEZE: the home field re-solves and upgrades the entry.
        if _protected(x0, y0):
            print(f"    {tag}registry: {name} is this field's target "
                  f"system; not consumed")
            continue
        if not (-margin_px <= x0 < nx + margin_px
                and -margin_px <= y0 < ny + margin_px):
            continue
        if not clist:
            tomb_entries.append((name, x0, y0, tomb))
        else:
            live.append((name, clist, x0, y0))

    # Tombstones: the solve failed where it was tried; kill the gate
    # so the doomed solve is not repeated -- the catalog render stands
    # until the home vantage rewrites the entry.
    out = list(comps)
    for name, x0, y0, reason in tomb_entries:
        comp = _nearest_comp(out, x0, y0)
        if comp is not None and comp['gate']:
            comp['gate'] = False
            print(f"    {tag}registry: {name} tombstone ({reason}); "
                  f"{comp['name']} keeps its catalog render, no seat")

    # Pass 1: drop every catalog row matched by a FROZEN entry (never
    # the target, never another entry's frozen components).
    for name, clist, x0, y0 in live:
        keep, dropped = [], []
        for comp in out:
            near = np.hypot(comp['x'] - x0, comp['y'] - y0) * pix
            if (comp['name'] != 'target' and not comp.get('reg')
                    and near < recipe.REGISTRY_MATCH_AS):
                dropped.append(comp['name'])
            else:
                keep.append(comp)
        out = keep
        if dropped:
            print(f"    {tag}registry: {name} drops {dropped}")

    # Pass 2: add the frozen components.
    consumed = []
    for name, clist, x0, y0 in live:
        for j, rc in enumerate(clist):
            x, y = [float(v) for v in wcs.world_to_pixel(
                SkyCoord(rc['ra'], rc['dec'], unit='deg'))]
            t0, slope = pa_map(wcs, x, y)
            theta = t0 + slope * rc['pa']
            # A SERSIC record carries the geometry its mask must be sized on.
            # aperture.build_mask caps a neighbor's mask at GEO_REFF_FACTOR x
            # reff and falls back to a seeing-sized disc for a shapeless
            # component, so leaving it None masks a resolved galaxy as a point
            # source -- and makes the mask depend on whether the source happens
            # to be registered rather than on the data. `gate` stays False, so
            # carrying a shape adds no seat.
            #
            # A NUKER halo gets none, deliberately. Masking is for compact
            # neighbor light the model may have got locally wrong; a cD
            # envelope is smooth, is already subtracted, and leaves a
            # background-scale residual the mesh owns. Nothing masks a halo
            # anywhere else either -- GATED_HALO is False, so the Nuker family
            # belongs to the TARGET, and build_mask skips the target outright.
            # Sized on rb it would cap at 2.5 x tens of arcsec, larger than the
            # stamp, and swallow the aperture of any galaxy embedded in an
            # envelope.
            shape = None
            if rc['kind'] == 'sersic':
                scale_px = rc['reff_as'] / pix
                base = render_sersic_boxed(scale_px, rc['n'], rc['ellip'],
                                           theta, x, y, shape_2d, psf)
                shape = dict(kind='sersic', reff_px=float(scale_px),
                             n=float(rc['n']), ellip=float(rc['ellip']),
                             theta=float(theta), pa=float(rc['pa']))
            else:
                base = render_nuker(rc['rb_as'] / pix, rc['beta'],
                                    rc['ellip'], theta, x, y, shape_2d,
                                    psf, pix)
            unit_sum = float(base.sum())          # fixed-norm shape integral
            unit_in_stamp = unit_sum * cf
            flux_ref = float(rc['flux_ref'])
            # The component is rendered at its FITTED PHYSICAL amplitude,
            # not renormalized to this stamp. The physical amplitude is
            # flux_ref / (home-field shape integral): scaling to
            # flux_ref / (THIS stamp's integral) instead would cram the
            # whole source's flux into whatever pixels are on-stamp, so
            # an off-stamp source (a BCG whose center is past a
            # neighbor's edge) has its thin wing multiplied many-fold.
            # flux_home is the home SHAPE integral (cf-free), so this
            # stamp's own cf sets the amplitude -- per-stack zeropoints
            # (PS1/SDSS) never leak across fields.
            flux_home = rc.get('flux_home')
            if flux_home:
                amp = flux_ref / (float(flux_home) * cf)
            else:
                amp = flux_ref / unit_in_stamp if unit_in_stamp > 0 else 1.0
            base = base * amp
            phys_in_stamp = amp * unit_in_stamp   # the wing that reaches
            if phys_in_stamp < recipe.MARGIN_MIN_UJY \
                    or flux_ref < recipe.MARGIN_MIN_UJY:
                continue
            # Leash around the RENDERED wing flux, not the home flux: an
            # off-stamp wing must not be re-solved up to the full source.
            out.append(dict(
                base=base, flux0=max(phys_in_stamp, 1e-9), shape=shape,
                name=f'{name}.{j}', irow=-1, cat=phys_in_stamp,
                amp_lohi=(recipe.REGISTRY_AMP_BAND[0] * phys_in_stamp,
                          recipe.REGISTRY_AMP_BAND[1] * phys_in_stamp),
                x=x, y=y, gate=False, reg=True))
        # Snapshot the frozen shapes, not just the name: the provenance then
        # records what each fit actually consumed, so the fit reconstructs from
        # its own sidecar even after the mutable registry is rewritten.
        consumed.append({'name': name, 'shapes': [dict(rc) for rc in clist]})
        print(f"    {tag}registry: {name} consumed "
              f"({len(clist)} frozen comps)")
    return out, consumed


# ------------------------------------
# Registry: write at solve time
# ------------------------------------
def registry_name(ra_deg: float, dec_deg: float) -> str:
    """Position-derived registry key (IAU style, JHHMMSS.s+DDMMSS).

    The key is deterministic in the seat's anchor position, so
    re-solving the same physical source from any overlapping field
    updates the same entry.
    """
    coord = SkyCoord(ra_deg, dec_deg, unit='deg')
    h, m, s = coord.ra.hms
    sign = '-' if coord.dec.deg < 0 else '+'
    _, dd, dm, ds = coord.dec.signed_dms
    return (f"J{int(h):02d}{int(m):02d}{s:04.1f}"
            f"{sign}{int(dd):02d}{int(dm):02d}{int(round(ds)):02d}")


def resolve_registry_key(registry: dict, ra_deg: float, dec_deg: float,
                         tol_arcsec: float = None) -> str:
    """Registry key for (ra, dec), reusing a near-duplicate entry.

    registry_name() alone fragments a source whose seat anchor drifts
    across bands: a coarse-pixel PS1 solve can land ~0.3" off the deep-
    band anchor and round to a different JHHMMSS+DDMMSS, so the source's
    bands split across two near-identical keys. Coalescing to the nearest
    existing entry within tol keeps one physical source on one key. Two
    keys for one source would both be consumed, and consumption places
    entries by position, so the same light would be subtracted twice in
    the same band.
    """
    tol = recipe.REGISTRY_KEY_COALESCE_AS if tol_arcsec is None else tol_arcsec
    fresh = registry_name(ra_deg, dec_deg)
    best, best_sep = fresh, tol
    c0 = SkyCoord(ra_deg, dec_deg, unit='deg')
    for key, entry in registry.items():
        try:
            sep = c0.separation(
                SkyCoord(entry['ra'], entry['dec'], unit='deg')).arcsec
        except (KeyError, TypeError, ValueError):
            continue
        if sep < best_sep:
            best, best_sep = key, sep
    return best


def harvest_seats(
        registry: dict,
        seats: list[dict],
        params: np.ndarray,
        seat_amps: list[float],
        stamp: Stamp,
        *,
        band_key: str,
        seat_col_flux: list[float] | None = None,
        include_target: bool = False,
        solve_health: dict | None = None,
        tag: str = '',
) -> list[str]:
    """Merge one band's solved seats into the registry.

    Every record carries its VANTAGE: 'target' for the target seat of
    its own field (include_target -- the one centered view, consumed
    frozen downstream), 'neighbor' for everything else (warm seeds
    only). A target-vantage record is never overwritten by a neighbor
    one -- the best fit stands; the home solve upgrades in the other
    direction. An UNHEALTHY solve -- one that burned its evaluation
    cap without converging -- harvests a TOMBSTONE for its entries'
    band instead of records: garbage must not propagate, and the
    doomed solve must not be re-attempted field after field. A
    parameter on a bound alone is NOT unhealthy: rails are routine
    witnesses on real envelopes (a Nuker beta pinned at its ceiling
    appears on validated fits throughout the sample) and ride the
    at_bound witness, not the registry's health verdict.

    Parameters
    ----------
    registry : dict
        Registry to update in place.
    seats : list of dict
        BAND-LOCAL seat list (radial bounds in this band's pixels).
    params : np.ndarray
        Solved parameter vector aligned with seats, in this band's
        pixels.
    seat_amps : list of float
        Solved per-seat fluxes (uJy), aligned with seats.
    seat_col_flux : list of float, optional
        Per-seat unit-column in-stamp flux (uJy), aligned with seats.
        Stored divided by cf as `flux_home`, the shape integral consumers
        use to recover the physical amplitude. None stores no anchor.
    stamp : Stamp
        This band's stamp (WCS and pixel scale).
    band_key : str
        Band label the components are stored under.
    include_target : bool
        Write the target seat too, at target vantage.
    solve_health : dict, optional
        {'capped': bool} from the solve; a cap taints every harvested
        owner of the band. None means healthy.
    tag : str
        Run-log prefix.

    Returns
    -------
    touched : list of str
        Registry entry names written.
    """
    wcs = stamp.wcs
    pix = stamp.pixscale
    health = solve_health or {}
    capped = bool(health.get('capped'))
    col_flux = seat_col_flux or [None] * len(seats)
    cf = stamp.cf

    fresh: dict[str, list[dict]] = {}
    anchors: dict[str, tuple[float, float]] = {}
    dead: dict[str, str] = {}
    for seat, sl, amp, col_f in zip(seats, seat_slices(seats), seat_amps,
                                    col_flux):
        if seat['owner'] == 'target' and not include_target:
            continue
        name = resolve_registry_key(registry, seat['ra'], seat['dec'])
        anchors[name] = (seat['ra'], seat['dec'])
        if capped:
            dead[name] = 'solve hit its evaluation cap'
            continue
        # Zero amplitude is a verdict, not a measurement: no such
        # component, and a shape that contributed nothing was never
        # constrained. Such records are not stored -- they can neither
        # be frozen into another scene nor offered new flux.
        if max(float(amp), 0.0) < recipe.MARGIN_MIN_UJY:
            continue
        q = np.asarray(params[sl], float)
        x, y = [float(v) for v in wcs.world_to_pixel(
            SkyCoord(seat['ra'], seat['dec'], unit='deg'))]
        center = wcs.pixel_to_world(x + q[4], y + q[5])
        # flux_home: the home shape integral, cf-FREE (col_flux / cf =
        # the unit-render's pixel sum). Consumers recover the physical
        # amplitude as flux_ref / (flux_home * their own cf), so an
        # off-stamp neighbor gets only the wing that reaches -- never
        # the whole source crammed into its edge -- and per-stack
        # zeropoints stay local.
        record = dict(kind=seat['kind'], ra=float(center.ra.deg),
                      dec=float(center.dec.deg), ellip=float(q[2]),
                      pa=float(q[3]), flux_ref=max(float(amp), 0.0),
                      flux_home=(float(col_f) / cf if col_f else None),
                      vantage=('target' if seat['owner'] == 'target'
                               else 'neighbor'))
        if seat['kind'] == 'sersic':
            record.update(reff_as=float(q[0] * pix), n=float(q[1]))
        else:
            record.update(rb_as=float(q[0] * pix), beta=float(q[1]))
        fresh.setdefault(name, []).append(record)

    # An entry is a JOINT decomposition: one tainted seat poisons its
    # entry's whole band.
    for name in dead:
        fresh.pop(name, None)

    written = []
    for name, records in fresh.items():
        entry = registry.setdefault(name, dict(
            ra=anchors[name][0], dec=anchors[name][1], components={}))
        existing = entry['components'].get(band_key)
        if (existing and existing[0].get('vantage') == 'target'
                and records[0]['vantage'] == 'neighbor'):
            continue      # the home fit stands; no downgrade
        entry['components'][band_key] = records
        (entry.get('tombstones') or {}).pop(band_key, None)
        written.append(name)
    for name, reason in dead.items():
        entry = registry.setdefault(name, dict(
            ra=anchors[name][0], dec=anchors[name][1], components={}))
        existing = entry['components'].get(band_key)
        if existing and existing[0].get('vantage') == 'target':
            continue      # a healthy home fit outranks a failed retry
        entry['components'].pop(band_key, None)
        entry.setdefault('tombstones', {})[band_key] = reason
        print(f"    {tag}registry: {name} tombstoned [{band_key}]: "
              f"{reason}")
    if written:
        print(f"    {tag}registry: harvested {sorted(written)} "
              f"[{band_key}]")
    return sorted(written)


def load_registry(path: str | Path | None) -> dict:
    """Read a registry file; {} when no path is given or none exists."""
    if path is None:
        return {}
    if not Path(path).exists():
        print(f"  registry: {path} does not exist; starting empty")
        return {}
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def save_registry(registry: dict, path: str | Path) -> None:
    """Write the registry atomically (write a sibling, then replace).

    Atomicity means an interrupted run can never leave a torn,
    half-written file behind. It does NOT serialize concurrent writers:
    two runs updating one registry finish last-writer-wins, so
    --registry-update sweeps must run one galaxy at a time.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(registry, handle, indent=1, sort_keys=True)
    tmp.replace(path)
