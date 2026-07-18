"""
solve.py

Stage 6: The Joint Fit, with One Background Owner
---------------------------------------------------------
The background never sits in a design matrix next to component
amplitudes. It is only ever estimated by background.bin_plane on a
scene-subtracted image, alternating with the amplitude solve until its
constant converges (block coordinate descent). Gated systems get a
shape solve: variable projection -- every fixed amplitude solved
exactly at every trial of the shape parameters -- with the Gram block
of the constant fixed columns precomputed and shared across the
alternation's warm re-solves.

Reference bands solve seat shapes; transfer bands freeze the TARGET
seat at the reference shape (the measurement definition never
re-negotiates per band) and re-solve NEIGHBOR seat shapes warm, with
seat fluxes leashed to color-scaled reference values (subtraction wants
per-band fidelity: chromatic morphology is real, especially for large
envelopes).

Requirements:
    numpy, scipy, astropy

Notes:
    Amplitudes are MICROJANSKYS: every design column is normalized to
    unit in-stamp flux, so a fitted amplitude IS that component's
    in-stamp flux through the band -- one unit system for catalog
    components and seat columns on every instrument.
"""
from __future__ import annotations

import time

import numpy as np
from astropy.coordinates import SkyCoord
from scipy.optimize import least_squares, lsq_linear

from . import recipe
from .background import bin_plane
from .render import pa_map, render_nuker, render_sersic_boxed
from .seats import seat_slices
from .stamp import Stamp


# ------------------------------------
# Linear amplitudes (in microjanskys)
# ------------------------------------
def _design(bases, good, fluxes, bounds=None):
    """Normalized design matrix + bounds, built once per column set.

    Every column is divided by its in-stamp flux (uJy), so every fitted
    amplitude IS that component's in-stamp flux in uJy through this
    band. Default bounds: (0, AMP_MAX_X_CAT x the column's reference
    flux); explicit bounds (uJy) are honored verbatim.
    """
    G = np.column_stack([b[good] / f for b, f in zip(bases, fluxes)])
    lb = np.zeros(G.shape[1])
    ub = recipe.AMP_MAX_X_CAT * np.asarray(fluxes, float)
    if bounds is not None:
        for j, (blo, bhi) in enumerate(bounds):
            if blo is not None:
                lb[j] = blo
            if bhi is not None:
                ub[j] = bhi
    ub = np.maximum(ub, lb + 1e-12)   # the solver demands lb < ub
    norms = np.sqrt((G * G).sum(0))
    norms[norms == 0] = 1.0
    return G / norms, norms, lb, ub


def _amp_solve(Gn, norms, lb, ub, rhs):
    result = lsq_linear(Gn, rhs, bounds=(lb * norms, ub * norms),
                        tol=1e-10)
    return result.x / norms


# ------------------------------------
# Seat rescaling across grids
# ------------------------------------
# Pixel-valued entries of each seat kind: (size, dx, dy).
_KIND_RADIAL = {'sersic': (0, 4, 5), 'nuker': (0, 4, 5)}


def _scale_seat(seat: dict, s_px: float) -> dict:
    """Band-local copy of a seat: radial p0/lo/hi entries rescaled."""
    if s_px == 1.0:
        return seat
    out = dict(seat)
    for key in ('p0', 'lo', 'hi'):
        values = list(seat[key])
        for i in _KIND_RADIAL[seat['kind']]:
            values[i] = values[i] * s_px
        out[key] = values
    return out


def _scale_params(seats: list[dict], p, s_px: float) -> np.ndarray:
    """Rescale a full parameter vector's radial entries to a new grid."""
    p = np.array(p, float)
    if s_px == 1.0:
        return p
    for seat, sl in zip(seats, seat_slices(seats)):
        for k in _KIND_RADIAL[seat['kind']]:
            p[sl.start + k] *= s_px
    return p


# ------------------------------------
# Seat rendering
# ------------------------------------
def render_seats(
        seats: list[dict],
        p,
        stamp: Stamp,
        psf: np.ndarray,
        s_px: float = 1.0,
) -> tuple[list[np.ndarray], list[str]]:
    """All seat columns on this band's grid.

    Radial parameters are in reference-band pixels, rescaled
    arcsec-invariantly by s_px; centers resolve from sky coordinates
    through this band's WCS.
    """
    cols, owners = [], []
    shape_2d = stamp.shape
    for seat, sl in zip(seats, seat_slices(seats)):
        q = p[sl]
        x, y = [float(v) for v in stamp.wcs.world_to_pixel(
            SkyCoord(seat['ra'], seat['dec'], unit='deg'))]
        t0, slope = pa_map(stamp.wcs, x, y)
        if seat['kind'] == 'sersic':
            reff, n, ellip, pa, dx, dy = q
            cols.append(render_sersic_boxed(
                reff * s_px, n, ellip, t0 + slope * pa,
                x + dx * s_px, y + dy * s_px, shape_2d, psf))
        else:
            rb, beta, ellip, pa, dx, dy = q
            cols.append(render_nuker(
                rb * s_px, beta, ellip, t0 + slope * pa,
                x + dx * s_px, y + dy * s_px, shape_2d, psf,
                stamp.pixscale))
        owners.append(seat['owner'])
    return cols, owners


# ------------------------------------
# The shape solve (variable projection)
# ------------------------------------
def _fixed_gram(fixed_bases, good, extra_cols):
    """Normalized fixed-column block and its Gram matrix.

    The fixed columns are identical across the alternation's warm
    re-solves (only the background, hence only the right-hand side,
    changes), so the Gram block is computed once and shared. A scene
    whose every component is seated has an EMPTY fixed block; the
    zero-width matrices keep the algebra valid.
    """
    cols = [b[good] for b in fixed_bases]
    cols += [c[good] for c in extra_cols]
    if not cols:
        empty = np.zeros((int(good.sum()), 0))
        return empty, np.zeros(0), np.zeros((0, 0))
    Fb = np.column_stack(cols)
    norms = np.sqrt((Fb * Fb).sum(0))
    norms[norms == 0] = 1.0
    Fn = Fb / norms
    return Fn, norms, Fn.T @ Fn


def solve_shapes(
        image: np.ndarray,
        good: np.ndarray,
        comps: list[dict],
        bg_img: np.ndarray,
        stamp: Stamp,
        psf: np.ndarray,
        seats: list[dict],
        drops: set[str],
        *,
        p_seed=None,
        extra_fixed_cols=None,
        gram=None,
) -> dict:
    """Joint nonlinear solve of the given seats' shape parameters.

    The background is frozen at bg_img; every fixed amplitude is solved
    exactly at every trial (variable projection). p_seed warm-starts
    from a previous iterate. extra_fixed_cols are pre-rendered columns
    held fixed in the solve -- e.g. the frozen target seat on transfer
    bands, where only subtractive neighbor seats re-solve.

    Returns
    -------
    solve_info : dict
        p (the solved vector), params (as floats), seats (labels), nfev,
        status, cost, seconds, at_bound (names of parameters pinned at
        their box -- a parameter at its bound is a flag, not a
        measurement), pix_ref (the grid the radial parameters live in).
    """
    y = (image - bg_img)[good]
    if gram is None:
        fixed = [c for c in comps if c['name'] not in drops]
        gram = _fixed_gram([c['base'] for c in fixed], good,
                           extra_fixed_cols or [])
    Fn, _, FtF = gram
    Fty = Fn.T @ y
    kF = FtF.shape[0]
    sigma = stamp.sigma

    def inner(p):
        cols, _ = render_seats(seats, p, stamp, psf)
        E = np.column_stack([c[good] for c in cols])
        nE = np.sqrt((E * E).sum(0))
        nE[nE == 0] = 1.0
        En = E / nE
        FtE = Fn.T @ En
        nt = kF + En.shape[1]
        Mn = np.empty((nt, nt))
        Mn[:kF, :kF] = FtF
        Mn[:kF, kF:] = FtE
        Mn[kF:, :kF] = FtE.T
        Mn[kF:, kF:] = En.T @ En
        Mn.flat[::nt + 1] += 1e-10 * np.trace(Mn) / nt
        sol = np.linalg.solve(Mn, np.concatenate([Fty, En.T @ y]))
        return y - (Fn @ sol[:kF] + En @ sol[kF:])

    # Ownership penalty: a halo displaced beyond its own break radius
    # is not that galaxy's halo.
    nuker_starts = [sl.start for seat, sl in zip(seats, seat_slices(seats))
                    if seat['kind'] == 'nuker']

    def fun(p):
        resid = inner(p) / sigma
        pens = [100.0 * max(0.0, np.hypot(p[i0 + 4], p[i0 + 5]) - p[i0])
                for i0 in nuker_starts]
        return np.append(resid, pens)

    lo = np.concatenate([s['lo'] for s in seats])
    hi = np.concatenate([s['hi'] for s in seats])
    p0 = (np.asarray(p_seed, float) if p_seed is not None
          else np.concatenate([s['p0'] for s in seats]))
    t0 = time.time()
    nfev_stage1 = 0
    if p_seed is None:
        # Two-stage cold start: with center offsets in the vector, a
        # cold solve can converge into a nearby local minimum before
        # the geometry organizes. Stage 1 solves the centers-frozen
        # problem; stage 2 releases the Sersic centers warm from that
        # basin. Warm re-solves skip the staging.
        lo1, hi1 = lo.copy(), hi.copy()
        for seat, sl in zip(seats, seat_slices(seats)):
            if seat['kind'] == 'sersic':
                lo1[sl.start + 4:sl.start + 6] = -1e-6
                hi1[sl.start + 4:sl.start + 6] = 1e-6
        stage1 = least_squares(fun, np.clip(p0, lo1, hi1),
                               bounds=(lo1, hi1), loss='soft_l1',
                               f_scale=recipe.SOLVE_FSCALE,
                               x_scale='jac', max_nfev=recipe.SOLVE_NFEV)
        p0 = stage1.x
        nfev_stage1 = int(stage1.nfev)
    result = least_squares(fun, np.clip(p0, lo, hi), bounds=(lo, hi),
                           loss='soft_l1', f_scale=recipe.SOLVE_FSCALE,
                           x_scale='jac', max_nfev=recipe.SOLVE_NFEV)
    seconds = time.time() - t0

    param_names = []
    for seat in seats:
        keys = (('rb', 'beta', 'e', 'pa', 'dx', 'dy')
                if seat['kind'] == 'nuker'
                else ('reff', 'n', 'e', 'pa', 'dx', 'dy'))
        param_names += [f"{seat['owner']}.{seat['kind']}.{k}" for k in keys]
    at_bound = [param_names[i] for i, (v, l, h) in
                enumerate(zip(result.x, lo, hi))
                if (v - l < 1e-6 * (h - l) or h - v < 1e-6 * (h - l))]
    return dict(seats=[f"{s['owner']}:{s['kind']}" for s in seats],
                p=result.x, params=[float(v) for v in result.x],
                nfev=int(result.nfev) + nfev_stage1,
                status=int(result.status),
                cost=float(result.cost), seconds=round(seconds, 1),
                at_bound=at_bound, pix_ref=stamp.pixscale)


# ------------------------------------
# Transfer-band plumbing
# ------------------------------------
def _transfer_setup(seats, ref, stamp, psf):
    """Band-local seat machinery for a transfer band.

    Scales the reference seats and solved parameters onto this band's
    grid, renders the frozen target columns once, and splits the seat
    indices into frozen (target) and free (neighbor) sets.
    """
    s_px = ref['pix'] / stamp.pixscale
    seats_local = [_scale_seat(s, s_px) for s in seats]
    p_local = _scale_params(seats, ref['p'], s_px)
    slices = seat_slices(seats)
    free_idx = [i for i, s in enumerate(seats) if s['owner'] != 'target']
    frozen_idx = [i for i, s in enumerate(seats) if s['owner'] == 'target']
    frozen_cols = (render_seats(
        [seats_local[i] for i in frozen_idx],
        np.concatenate([p_local[slices[i]] for i in frozen_idx]),
        stamp, psf)[0] if frozen_idx else [])
    p_free = (np.concatenate([p_local[slices[i]] for i in free_idx])
              if free_idx else None)
    colors = ref.get('col_color') or [1.0] * len(seats)
    return dict(seats_local=seats_local, p_local=p_local, slices=slices,
                free_idx=free_idx, frozen_idx=frozen_idx,
                frozen_cols=frozen_cols, p_free=p_free, colors=colors)


def _transfer_columns(setup, seats, ref, free_cols):
    """Seat columns, owners, and flux bounds in original seat order."""
    cols, owners, bounds = [], [], []
    it_free = iter(free_cols)
    it_frozen = iter(setup['frozen_cols'])
    for j, seat in enumerate(seats):
        col = (next(it_frozen) if j in setup['frozen_idx']
               else next(it_free))
        cols.append(col)
        owners.append(seat['owner'])
        expected = setup['colors'][j] * max(ref['col_flux'][j], 0.0)
        bounds.append((recipe.TRANSFER_AMP_BAND[0] * expected,
                       max(recipe.TRANSFER_AMP_BAND[1] * expected, 1e-12)))
    return cols, owners, bounds


def _transfer_params(setup):
    """Full band-local parameter vector with the re-solved free slices."""
    p = np.array(setup['p_local'], float)
    if setup['free_idx']:
        offset = 0
        for i in setup['free_idx']:
            sl = setup['slices'][i]
            width = sl.stop - sl.start
            p[sl] = setup['p_free'][offset:offset + width]
            offset += width
    return p


# ------------------------------------
# The alternation
# ------------------------------------
def joint_fit(
        image: np.ndarray,
        good: np.ndarray,
        stamp: Stamp,
        psf: np.ndarray,
        comps: list[dict],
        seats: list[dict],
        drops: set[str],
        *,
        ref: dict | None = None,
) -> dict:
    """The whole fit: {shapes + amplitudes} <-> background, block
    coordinate descent to a fixed point.

    Shapes re-solve INSIDE the alternation (warm-started from the
    previous iterate): on halo-dominated stamps the first background is
    contaminated, and shapes solved once against it would inherit that
    bias frozen. Runaway (halo grows -> background drops -> halo grows)
    is contained by the profile bounds and truncation, the ownership
    penalty, and the background's bin-level rejection; the track
    witnesses whether a fixed point was reached.

    Parameters
    ----------
    image : np.ndarray
        Star-subtracted image (counts, finite everywhere).
    good : np.ndarray
        Usable-pixel map.
    stamp : Stamp
        This band's stamp.
    psf : np.ndarray
        This band's PSF kernel.
    comps : list of dict
        Scene components.
    seats, drops : list of dict, set of str
        Seat definitions and the component names they replace
        (seats.build_seats). On transfer bands, pass the reference
        band's seats verbatim.
    ref : dict, optional
        Transfer-band reference: seats' solved parameters (p), the
        reference pixel scale (pix), per-seat reference fluxes
        (col_flux), and per-seat color factors (col_color). None on a
        reference band -- seats solve their own shapes.

    Returns
    -------
    fit : dict
        amps and mults (aligned with fixed components then seat
        columns), bg (the converged background), track (the constant's
        path), solve_info, cols and owners (seat columns), fixed (the
        fixed component list), col_flux, and -- for the registry --
        seats_local, seat_params (band-local solved vector), seat_amps.
    """
    rr, pix, sigma = stamp.rr, stamp.pixscale, stamp.sigma
    cf = stamp.cf
    fixed = [c for c in comps if c['name'] not in drops]
    fixed_bases = [c['base'] for c in fixed]
    fixed_flux = [c['flux0'] for c in fixed]
    bg = bin_plane(image, good, rr, pix)
    track = [bg['const']]
    solve_info, nfev_hist = None, []
    cols, owners, col_flux, bounds = [], [], [], None

    solving = bool(seats) and ref is None
    transfer = (_transfer_setup(seats, ref, stamp, psf)
                if seats and ref is not None else None)

    p = None
    amps = np.zeros(len(fixed_bases))
    mults = amps
    bases = fixed_bases
    design = None   # rebuilt only when the seat columns change
    gram = None     # fixed-Gram block, shared by all warm re-solves
    done = False

    for _ in range(recipe.ALT_MAX_ITER):
        if solving:
            if gram is None:
                gram = _fixed_gram(fixed_bases, good, [])
            solve_info = solve_shapes(image, good, comps, bg['img'],
                                      stamp, psf, seats, drops,
                                      p_seed=p, gram=gram)
            p = solve_info['p']
            nfev_hist.append(solve_info['nfev'])
            cols, owners = render_seats(seats, p, stamp, psf)
            bounds = None
            design = None
        elif transfer is not None:
            if transfer['free_idx']:
                if gram is None:
                    gram = _fixed_gram(fixed_bases, good,
                                       transfer['frozen_cols'])
                solve_info = solve_shapes(
                    image, good, comps, bg['img'], stamp, psf,
                    [transfer['seats_local'][i]
                     for i in transfer['free_idx']],
                    drops, p_seed=transfer['p_free'],
                    extra_fixed_cols=transfer['frozen_cols'],
                    gram=gram)
                transfer['p_free'] = solve_info['p']
                nfev_hist.append(solve_info['nfev'])
                free_cols, _ = render_seats(
                    [transfer['seats_local'][i]
                     for i in transfer['free_idx']],
                    transfer['p_free'], stamp, psf)
            else:
                free_cols = []
            cols, owners, bounds = _transfer_columns(transfer, seats,
                                                     ref, free_cols)
            design = None
        if design is None and (fixed_bases or cols):
            col_flux = [max(float(c.sum()) * cf, 1e-9) for c in cols]
            bases = fixed_bases + cols
            # Registry components carry their solved per-band flux as a
            # tight leash (amp_lohi); everything else gets the default
            # catalog-multiple ceiling. Reference-band seats take the
            # same sanity ceiling as fixed components -- a seat stands
            # in for its catalog row, and an unbounded degenerate
            # column can solve to an absurd amplitude on a near-zero
            # render and poison every sibling band's leash.
            fixed_bounds = [c.get('amp_lohi', (None, None)) for c in fixed]
            if bounds is None:
                cat_by = {c['name']: c['cat'] for c in comps}
                seat_bounds = [(0.0, recipe.AMP_MAX_X_CAT
                                * max(cat_by.get(o, 1.0), 1.0))
                               for o in owners]
            else:
                seat_bounds = bounds
            design = _design(bases, good, fixed_flux + col_flux,
                             fixed_bounds + seat_bounds)
        scene = np.zeros_like(image)
        if design is not None:
            amps = _amp_solve(*design, (image - bg['img'])[good])
            mults = amps / np.asarray(fixed_flux + col_flux)
            for m, b in zip(mults, bases):
                scene += max(m, 0.0) * b
        bg_new = bin_plane(image - scene, good, rr, pix)
        track.append(bg_new['const'])
        done = abs(bg_new['const'] - bg['const']) < recipe.ALT_TOL_SIGMA \
            * sigma
        bg = bg_new
        if done:
            break
    if not done and design is not None:
        # The loop hit its cap with the background still moving:
        # refresh the amplitudes against the final background. (When
        # converged, the last in-loop solve is already consistent.)
        amps = _amp_solve(*design, (image - bg['img'])[good])
        mults = amps / np.asarray(fixed_flux + col_flux)
    if solve_info is not None:
        solve_info['nfev_track'] = nfev_hist

    # Band-local seat state for the registry harvest.
    if solving and p is not None:
        seats_local, seat_params = seats, np.asarray(p, float)
    elif transfer is not None:
        seats_local = transfer['seats_local']
        seat_params = _transfer_params(transfer)
    else:
        seats_local, seat_params = [], None
    seat_amps = [float(a) for a in amps[len(fixed_bases):]]

    return dict(amps=amps, mults=mults, bg=bg, track=track,
                solve_info=solve_info, cols=cols, owners=owners,
                fixed=fixed, col_flux=col_flux, seats_local=seats_local,
                seat_params=seat_params, seat_amps=seat_amps)
