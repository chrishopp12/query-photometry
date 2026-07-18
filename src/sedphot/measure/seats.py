"""
seats.py

Stage 5: Seats and the Cross-Field Registry
---------------------------------------------------------
A "seat" is a component whose shape parameters enter the nonlinear
solve. Standard, catalog-driven seats: every gated component gets a
Sersic core (small center box) plus a Nuker halo (wide center box under
an ownership penalty), and the target always gets a refit seat -- the
catalog informs the photometry only through the neighbors, never
through the target itself. Custom, per-galaxy seats (all optional, via
the patches file): free single-Sersic seats for companion nuclei or
retyped rows, center snapping, disabling the target refit.

Seat dict:
    kind       'sersic' or 'nuker'
    owner      component name whose fixed base this seat replaces
    ra, dec    seat anchor (sky degrees); centers resolve per band
    p0, lo, hi parameter seed and bounds, recipe.SEAT_NPARAMS each:
               sersic (reff_px, n, ellip, pa_deg, dx_px, dy_px)
               nuker  (rb_px, beta, ellip, pa_deg, dx_px, dy_px)
               radial and offset entries are reference-band pixels; pa
               is sky-frame degrees.

The registry lets a source solved once (a bright galaxy's core + halo,
companion nuclei) be consumed by every later field that contains it as
FROZEN fixed components -- amplitude-only, tightly leashed -- instead
of being re-solved per field into as many different decompositions as
there are fields. Entries transport the full per-band decomposition:
per-band shapes (arcsec, sky PA), solved centers as sky coordinates,
and per-band solved fluxes as the amplitude anchor.

Requirements:
    numpy, scipy, astropy

Notes:
    Registry entries are keyed by a position-derived name, so re-solving
    the same physical source from any field updates the same entry.
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
        if comp['cat'] * frac < recipe.MARGIN_MIN_UJY:
            print(f"    {tag}{comp['name']}: gated core seat has "
                  f"{comp['cat'] * frac:.2g} uJy in-stamp at catalog "
                  f"amplitude; seat pruned")
        else:
            seats.append(_seat(
                'sersic', comp['name'], wcs, x, y, p0_core,
                [SEAT_REFF_MIN_PX, n_lo, 0.0,
                 shape['pa'] - recipe.PA_BOX_DEG, -dxy, -dxy],
                [core_hi, n_hi, recipe.SERSIC_E_MAX,
                 shape['pa'] + recipe.PA_BOX_DEG, dxy, dxy]))
        seats.append(_seat(
            'nuker', comp['name'], wcs, x, y,
            [recipe.NUKER_RB0_AS / pix, 2.0, shape['ellip'],
             shape['pa'], 0.0, 0.0],
            [recipe.NUKER_RB_AS[0] / pix, recipe.NUKER_BETA[0], 0.0,
             shape['pa'] - recipe.PA_BOX_DEG, -dxy_out, -dxy_out],
            [recipe.NUKER_RB_AS[1] / pix, recipe.NUKER_BETA[1],
             recipe.NUKER_E_MAX,
             shape['pa'] + recipe.PA_BOX_DEG, dxy_out, dxy_out]))
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
            fx, fy = snap_to_peak(image, fx, fy, pix)
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
    # "target_refit": false to disable it per galaxy. A catalog with no
    # row at the target position has nothing to refit.
    target = next((c for c in comps if c['name'] == 'target'), None)
    if patches.get('target_refit', True) and target is not None:
        shape = target['shape']
        reff_hi = recipe.REFIT_REFF_MAX_AS / pix
        if shape is not None:
            p0 = [min(shape['reff_px'], 0.98 * reff_hi),
                  min(shape['n'], 5.9), min(shape['ellip'], 0.91),
                  shape['pa'], 0.0, 0.0]
        else:
            p0 = [1.2 / pix, 2.0, 0.1, 0.0, 0.0, 0.0]
        seats.append(_seat(
            'sersic', 'target', wcs, target['x'], target['y'], p0,
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
        tag: str = '',
) -> tuple[list[dict], list[str]]:
    """Replace catalog rows with a registry entry's frozen components.

    Matched catalog rows are dropped in a first pass (so entries cannot
    eat each other's components), then every frozen component is added
    as a fixed, tightly-leashed design column. Gates die with the rows
    they belonged to.

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
    tag : str
        Run-log prefix.

    Returns
    -------
    comps : list of dict
        Component list with replacements applied.
    consumed : list of str
        Names of the registry entries consumed.
    """
    if not registry:
        return comps, []
    wcs = stamp.wcs
    pix = stamp.pixscale
    cf = stamp.cf
    shape_2d = stamp.shape
    ny, nx = shape_2d
    margin_px = recipe.MARGIN_AS / pix
    live = []
    for name, entry in registry.items():
        by_band = entry.get('components') or {}
        clist = by_band.get(band_key) or by_band.get(instrument)
        if not clist:
            if by_band:
                print(f"    {tag}registry: {name} has no components "
                      f"for {band_key} or {instrument}; not consumed")
            continue
        x0, y0 = [float(v) for v in wcs.world_to_pixel(
            SkyCoord(entry['ra'], entry['dec'], unit='deg'))]
        if -margin_px <= x0 < nx + margin_px \
                and -margin_px <= y0 < ny + margin_px:
            live.append((name, clist, x0, y0))

    # Pass 1: drop every catalog row matched by any live entry (never
    # the target, never another entry's frozen components).
    out = list(comps)
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
            if rc['kind'] == 'sersic':
                base = render_sersic_boxed(rc['reff_as'] / pix, rc['n'],
                                           rc['ellip'], theta, x, y,
                                           shape_2d, psf)
            else:
                base = render_nuker(rc['rb_as'] / pix, rc['beta'],
                                    rc['ellip'], theta, x, y, shape_2d,
                                    psf, pix)
            in_stamp = float(base.sum()) * cf
            if in_stamp < recipe.MARGIN_MIN_UJY:
                continue
            flux_ref = float(rc['flux_ref'])
            out.append(dict(
                base=base, flux0=max(in_stamp, 1e-9), shape=None,
                name=f'{name}.{j}', irow=-1, cat=flux_ref,
                amp_lohi=(recipe.REGISTRY_AMP_BAND[0] * flux_ref,
                          recipe.REGISTRY_AMP_BAND[1] * flux_ref),
                x=x, y=y, gate=False, reg=True))
        consumed.append(name)
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


def harvest_seats(
        registry: dict,
        seats: list[dict],
        params: np.ndarray,
        seat_amps: list[float],
        stamp: Stamp,
        *,
        band_key: str,
        tag: str = '',
) -> list[str]:
    """Merge one band's solved non-target seats into the registry.

    The target seat is never written: a target refit is field-specific
    by definition, and any other field sees this galaxy through its own
    catalog row. Re-harvesting a band replaces that band's component
    list, so a re-run cannot double an entry.

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
    stamp : Stamp
        This band's stamp (WCS and pixel scale).
    band_key : str
        Band label the components are stored under.
    tag : str
        Run-log prefix.

    Returns
    -------
    touched : list of str
        Registry entry names written.
    """
    wcs = stamp.wcs
    pix = stamp.pixscale
    fresh: dict[str, list[dict]] = {}
    anchors: dict[str, tuple[float, float]] = {}
    for seat, sl, amp in zip(seats, seat_slices(seats), seat_amps):
        if seat['owner'] == 'target':
            continue
        q = np.asarray(params[sl], float)
        name = registry_name(seat['ra'], seat['dec'])
        x, y = [float(v) for v in wcs.world_to_pixel(
            SkyCoord(seat['ra'], seat['dec'], unit='deg'))]
        center = wcs.pixel_to_world(x + q[4], y + q[5])
        record = dict(kind=seat['kind'], ra=float(center.ra.deg),
                      dec=float(center.dec.deg), ellip=float(q[2]),
                      pa=float(q[3]), flux_ref=max(float(amp), 0.0))
        if seat['kind'] == 'sersic':
            record.update(reff_as=float(q[0] * pix), n=float(q[1]))
        else:
            record.update(rb_as=float(q[0] * pix), beta=float(q[1]))
        fresh.setdefault(name, []).append(record)
        anchors[name] = (seat['ra'], seat['dec'])
    for name, records in fresh.items():
        entry = registry.setdefault(name, dict(
            ra=anchors[name][0], dec=anchors[name][1], components={}))
        entry['components'][band_key] = records
    if fresh:
        print(f"    {tag}registry: harvested {sorted(fresh)} [{band_key}]")
    return sorted(fresh)


def load_registry(path: str | Path | None) -> dict:
    """Read a registry file; {} when no path is given or none exists."""
    if path is None:
        return {}
    if not Path(path).exists():
        print(f"  registry: {path} does not exist; starting empty")
        return {}
    with open(path) as handle:
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
    with open(tmp, 'w') as handle:
        json.dump(registry, handle, indent=1, sort_keys=True)
    tmp.replace(path)
