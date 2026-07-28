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

Reference bands solve their seats' shapes. A transfer band takes those
shapes and solves in one of three modes:

    default            every TARGET-SYSTEM seat frozen at the reference
                       shape, neighbor seats re-solved warm
    free_target        every seat free -- the per-band shape a gating
                       target harvests to the registry
    freeze_neighbors   every seat frozen: the target at the reference
                       shape, the neighbors at a vector from an earlier
                       free solve, so no shape solve runs at all

The target system freezes because its shape is part of the measurement
definition and its light is the target's own. Neighbor shapes re-solve
per band because subtraction wants per-band fidelity: chromatic
morphology is real, especially for large envelopes.

Requirements:
    numpy, scipy, astropy

Notes:
    Amplitudes are MICROJANSKYS: every design column is normalized to
    unit in-stamp flux, so a fitted amplitude IS that component's
    in-stamp flux through the band -- one unit system for catalog
    components and seat columns on every instrument.

    An amplitude is LEASHED when its solve is bounded to a window around
    an expected flux rather than left free to the sanity ceiling: the
    catalog value for a star or fixed component, a color-scaled
    reference-band solution for a transfer-band seat, a registry entry's
    stored per-band flux for a consumed component.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque

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
    band. Default bounds: (0, recipe.AMP_MAX_X_CAT x the column's own
    in-stamp flux -- the value it is normalized by); explicit bounds
    (uJy) are honored verbatim.
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
# Parameter indices whose values are in pixels, per seat kind:
# (size, dx, dy).
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
def seat_anchors(seats: list[dict], stamp: Stamp) -> list[tuple]:
    """Per-seat (x, y, t0, slope): the WCS-derived constants of a seat.

    A seat's anchor pixel and its local sky-PA -> pixel-theta map are
    fixed for the whole solve, but a naive render recomputes them per
    seat per evaluation -- thousands of WCS round trips per solve.
    Compute once, thread through every render.
    """
    out = []
    for seat in seats:
        x, y = [float(v) for v in stamp.wcs.world_to_pixel(
            SkyCoord(seat['ra'], seat['dec'], unit='deg'))]
        t0, slope = pa_map(stamp.wcs, x, y)
        out.append((x, y, t0, slope))
    return out


def _render_one(seat: dict, q, anchor: tuple, stamp: Stamp,
                psf: np.ndarray, s_px: float) -> np.ndarray:
    x, y, t0, slope = anchor
    if seat['kind'] == 'sersic':
        reff, n, ellip, pa, dx, dy = q
        return render_sersic_boxed(
            reff * s_px, n, ellip, t0 + slope * pa,
            x + dx * s_px, y + dy * s_px, stamp.shape, psf)
    rb, beta, ellip, pa, dx, dy = q
    return render_nuker(
        rb * s_px, beta, ellip, t0 + slope * pa,
        x + dx * s_px, y + dy * s_px, stamp.shape, psf,
        stamp.pixscale)


def render_seats(
        seats: list[dict],
        p,
        stamp: Stamp,
        psf: np.ndarray,
        s_px: float = 1.0,
        anchors: list[tuple] | None = None,
) -> tuple[list[np.ndarray], list[str]]:
    """All seat columns on this band's grid.

    Radial parameters are in the pixels of whatever grid p was solved
    on; s_px rescales them arcsec-invariantly onto this band's grid
    (1.0 when p is already band-local). Centers resolve from sky
    coordinates through this band's WCS (precomputed anchors when
    given).
    """
    if anchors is None:
        anchors = seat_anchors(seats, stamp)
    cols, owners = [], []
    for seat, sl, anchor in zip(seats, seat_slices(seats), anchors):
        cols.append(_render_one(seat, p[sl], anchor, stamp, psf, s_px))
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
        stage_warm=False,
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
    slices = seat_slices(seats)
    anchors = seat_anchors(seats, stamp)

    def inner_cols(cols):
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

    def render_all(p):
        return [_render_one(seat, p[sl], anchor, stamp, psf, 1.0)
                for seat, sl, anchor in zip(seats, slices, anchors)]

    # Ownership penalty: a halo displaced beyond its own break radius
    # is not that galaxy's halo.
    nuker_starts = [sl.start for seat, sl in zip(seats, slices)
                    if seat['kind'] == 'nuker']

    # Per-pixel loss scale: sky rms plus a fractional model-error
    # floor on the source counts (recipe.SOLVE_MODEL_ERR_FRAC).
    scale = sigma + recipe.SOLVE_MODEL_ERR_FRAC * np.abs(y)

    def pens(p):
        return [100.0 * max(0.0, np.hypot(p[i0 + 4], p[i0 + 5]) - p[i0])
                for i0 in nuker_starts]

    def fun(p):
        return np.append(inner_cols(render_all(p)) / scale, pens(p))

    def make_jac(lo, hi):
        # Per-seat finite differences: perturbing one parameter
        # re-renders ONE seat's column against the cached others,
        # where a generic FD Jacobian re-renders every seat for every
        # parameter -- the render count drops from (6k+1) x k to 7k.
        def jac(p):
            cols = render_all(p)
            base = np.append(inner_cols(cols) / scale, pens(p))
            J = np.empty((base.size, p.size))
            sqrt_eps = np.sqrt(np.finfo(float).eps)
            for j, (seat, sl, anchor) in enumerate(
                    zip(seats, slices, anchors)):
                for k_par in range(sl.stop - sl.start):
                    idx = sl.start + k_par
                    h = sqrt_eps * max(1.0, abs(p[idx]))
                    # Step inward from whichever bound the parameter sits
                    # on. Both bounds need the check: the staged center
                    # freeze is a 2e-6-wide box, narrower than the step
                    # itself, so a one-sided flip would still render an
                    # out-of-bounds trial.
                    if p[idx] + h > hi[idx]:
                        h = -h
                    if p[idx] + h < lo[idx]:
                        h = -h
                    p2 = p.copy()
                    p2[idx] += h
                    col2 = _render_one(seat, p2[sl], anchor, stamp,
                                       psf, 1.0)
                    trial = cols[:j] + [col2] + cols[j + 1:]
                    resid2 = np.append(inner_cols(trial) / scale,
                                       pens(p2))
                    J[:, idx] = (resid2 - base) / h
            return J
        return jac

    lo = np.concatenate([s['lo'] for s in seats])
    hi = np.concatenate([s['hi'] for s in seats])
    p0 = (np.asarray(p_seed, float) if p_seed is not None
          else np.concatenate([s['p0'] for s in seats]))
    t0 = time.time()
    max_nfev = (recipe.SOLVE_NFEV if p_seed is None
                else recipe.SOLVE_NFEV_WARM)
    # Two-stage start: with the center offsets free from the start, the
    # center<->shape degeneracy can send a solve wandering before the
    # geometry organizes. Stage 1 freezes the Sersic centers; stage 2
    # releases them from that basin. Applied to COLD solves and to
    # CROSS-BAND warm re-solves (stage_warm): a seed scaled from another
    # band lands off-center on a bright envelope and can burn the whole
    # evaluation budget crawling back to center. A same-band warm re-solve
    # inside the alternation already sits in its basin and skips the
    # staging.
    lo1, hi1 = lo.copy(), hi.copy()
    staged = False
    for seat, sl in zip(seats, slices):
        if seat['kind'] == 'sersic':
            lo1[sl.start + 4:sl.start + 6] = -1e-6
            hi1[sl.start + 4:sl.start + 6] = 1e-6
            staged = True
    nfev_stage1 = 0
    if staged and (p_seed is None or stage_warm):
        stage1 = least_squares(fun, np.clip(p0, lo1, hi1),
                               jac=make_jac(lo1, hi1),
                               bounds=(lo1, hi1), loss='soft_l1',
                               f_scale=recipe.SOLVE_FSCALE,
                               x_scale='jac', max_nfev=max_nfev)
        p0 = stage1.x
        nfev_stage1 = int(stage1.nfev)
    result = least_squares(fun, np.clip(p0, lo, hi), jac=make_jac(lo, hi),
                           bounds=(lo, hi), loss='soft_l1',
                           f_scale=recipe.SOLVE_FSCALE,
                           x_scale='jac', max_nfev=max_nfev)
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
def _target_side(seat: dict) -> bool:
    """Whether a seat's light belongs to the target system.

    A declared target_system member -- a dumbbell's second nucleus, a bound
    companion -- is the target's OWN light: integrated into the aperture,
    never subtracted, never masked. Its shape is therefore part of the
    measurement definition, so a transfer band freezes it exactly as it
    freezes the target, and the system's total shape is not re-negotiated
    band by band through one of its components.

    Seats built without the tag fall back to the name, which is the same
    answer wherever no member was declared.
    """
    if 'system' in seat:
        return bool(seat['system'])
    return seat['owner'] == 'target'


def _transfer_setup(seats, ref, stamp, psf, free_target=False,
                    freeze_neighbors=None):
    """Band-local seat machinery for a transfer band.

    Scales the reference seats and solved parameters onto this band's
    grid, renders the frozen target columns once, and splits the seat
    indices into frozen (target) and free (neighbor) sets. When
    free_target, the target seats join the free set too: the per-band
    free shape a gating target harvests for the registry, versus the
    science pass that freezes it at the reference shape.

    freeze_neighbors adopts a full band-local vector's NEIGHBOR slices and
    freezes every seat, so no shape solve runs at all: the target holds the
    reference shape, the neighbors hold the shapes that vector carries, and
    only amplitudes and background are left to solve. The merged vector is
    what gets rendered AND what the fit reports, so the shapes used and the
    shapes recorded are one object.
    """
    s_px = ref['pix'] / stamp.pixscale
    seats_local = [_scale_seat(s, s_px) for s in seats]
    p_local = _scale_params(seats, ref['p'], s_px)
    slices = seat_slices(seats)
    if freeze_neighbors is not None:
        adopted = np.asarray(freeze_neighbors, float)
        for i, seat in enumerate(seats):
            if not _target_side(seat):
                p_local[slices[i]] = adopted[slices[i]]
        free_idx = []
        frozen_idx = list(range(len(seats)))
    elif free_target:
        free_idx = list(range(len(seats)))
        frozen_idx = []
    else:
        free_idx = [i for i, s in enumerate(seats) if not _target_side(s)]
        frozen_idx = [i for i, s in enumerate(seats) if _target_side(s)]
    frozen_cols = (render_seats(
        [seats_local[i] for i in frozen_idx],
        np.concatenate([p_local[slices[i]] for i in frozen_idx]),
        stamp, psf)[0] if frozen_idx else [])
    p_free = (np.concatenate([p_local[slices[i]] for i in free_idx])
              if free_idx else None)
    free_anchors = seat_anchors([seats_local[i] for i in free_idx],
                                stamp) if free_idx else []
    colors = ref.get('col_color') or [1.0] * len(seats)
    return dict(seats_local=seats_local, p_local=p_local, slices=slices,
                free_idx=free_idx, frozen_idx=frozen_idx,
                frozen_cols=frozen_cols, p_free=p_free,
                free_anchors=free_anchors, colors=colors)


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
        free_target: bool = False,
        freeze_neighbors=None,
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
    free_target : bool
        Solve the target's shape too (the registry-harvest pass), instead
        of freezing it at the reference shape.
    freeze_neighbors : array-like, optional
        A full band-local seat vector whose NEIGHBOR shapes this fit adopts
        and holds fixed. With it no shape solves at all -- every seat is a
        fixed column and only amplitudes and background are solved, so
        solve_info is None. Used by the free-solve-first ordering
        (recipe.NEIGHBOR_SHAPE_FROM_FREE_SOLVE).

    Returns
    -------
    fit : dict
        amps and mults (aligned with fixed components then seat
        columns), bg (the converged background), track (the constant's
        path), solve_info, cols and owners (seat columns), fixed (the
        fixed component list), col_flux, amp_bounds (per-column
        (lo, hi) in uJy, aligned with amps; absent from pinned_fit's
        result), and -- for the registry -- seats_local, seat_params
        (band-local solved vector), seat_amps.
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
    amp_bounds: list = []

    solving = bool(seats) and ref is None
    transfer = (_transfer_setup(seats, ref, stamp, psf,
                                free_target=free_target,
                                freeze_neighbors=freeze_neighbors)
                if seats and ref is not None else None)

    p = None
    amps = np.zeros(len(fixed_bases))
    mults = amps
    bases = fixed_bases
    design = None   # rebuilt only when the seat columns change
    gram = None     # fixed-Gram block, shared by all warm re-solves
    anchors = seat_anchors(seats, stamp) if (solving and seats) else None
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
            cols, owners = render_seats(seats, p, stamp, psf,
                                        anchors=anchors)
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
                    gram=gram, stage_warm=True)
                transfer['p_free'] = solve_info['p']
                nfev_hist.append(solve_info['nfev'])
                free_cols, _ = render_seats(
                    [transfer['seats_local'][i]
                     for i in transfer['free_idx']],
                    transfer['p_free'], stamp, psf,
                    anchors=transfer['free_anchors'])
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
            amp_bounds = fixed_bounds + seat_bounds
            design = _design(bases, good, fixed_flux + col_flux, amp_bounds)
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
                seat_params=seat_params, seat_amps=seat_amps,
                amp_bounds=amp_bounds)


def pinned_fit(
        image: np.ndarray,
        good: np.ndarray,
        stamp: Stamp,
        psf: np.ndarray,
        comps: list[dict],
        seats: list[dict],
        drops: set[str],
        *,
        pin: dict,
) -> dict:
    """joint_fit's result, rebuilt from a stored fit -- no solve at all.

    The pinned reconstruction: seats render at the stored shape vector
    (pin['seat_params'], whose radial entries live in pin['seat_pix']
    arcsec/px and are rescaled onto this band's grid), amplitudes come
    from the sidecar in the order it recorded them, one queue per owner
    (pin['amps']), and the plane is re-evaluated from its stored
    coefficients (pin['bg_coefs']). Nothing is optimized, so the
    expensive shape solve, the amplitude solve, and the background fit
    are all skipped. The band is treated as self-contained -- its own
    stored shape, its own grid -- so no reference/transfer state is
    threaded.

    An owner absent from pin['amps'] (a catalog source the stored fit did
    not carry) falls back to its own catalog flux -- amplitude one, the
    scene's default prediction -- rather than vanishing from the scene.

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
        Seat definitions and the component names they replace, as the
        stored fit carried them.
    pin : dict
        The stored fit: seat_params, seat_pix, amps ([[owner, uJy], ...]
        in the order the fit recorded them), bg_coefs.

    Returns
    -------
    fit : dict
        What joint_fit returns, so the measurement flow downstream is
        identical -- minus amp_bounds, and with solve_info None and a
        one-element track (nothing was solved or alternated).
    """
    from .background import eval_plane

    cf = stamp.cf
    fixed = [c for c in comps if c['name'] not in drops]
    fixed_bases = [c['base'] for c in fixed]
    fixed_flux = [c['flux0'] for c in fixed]
    p = (np.asarray(pin['seat_params'], float)
         if pin.get('seat_params') is not None else None)
    # Radial parameters are grid-relative, so a vector solved on one band
    # renders at the wrong physical size on a band with a different pixel
    # scale. Rescale arcsec-invariantly, exactly as the live transfer path
    # does (_transfer_setup). A record with no seat_pix falls back to this
    # band's own grid -- right for every single-pixel-scale instrument, and
    # all such a record supports.
    seat_pix = pin.get('seat_pix')
    s_px = float(seat_pix) / stamp.pixscale if seat_pix else 1.0
    if seats and p is not None:
        cols, owners = render_seats(seats, p, stamp, psf, s_px,
                                    anchors=seat_anchors(seats, stamp))
    else:
        cols, owners = [], []
    col_flux = [max(float(c.sum()) * cf, 1e-9) for c in cols]
    bases = fixed_bases + cols
    base_owner = [c['name'] for c in fixed] + owners
    # One owner can hold SEVERAL columns: a target_halo target owns both a
    # core Sersic and a Nuker halo seat. Keying the stored amplitudes by name
    # collapses those into one, so the core would render at the halo's
    # amplitude. Queue per owner and consume in the recorded order, which is
    # the order base_owner is built in on both sides.
    queued: dict = defaultdict(deque)
    for owner, amp in pin['amps']:
        queued[owner].append(float(amp))
    flux = np.asarray(fixed_flux + col_flux) if bases else np.zeros(0)
    amps = np.array([queued[o].popleft() if queued[o] else f
                     for o, f in zip(base_owner, flux)])
    mults = amps / flux if flux.size else amps
    coefs = [float(v) for v in pin['bg_coefs']]
    bg = dict(img=eval_plane(coefs, image.shape), const=coefs[0],
              coefs=coefs, n_rej=0, n_bins=0,
              keep_px=np.zeros(image.shape, bool))
    return dict(amps=amps, mults=mults, bg=bg, track=[coefs[0]],
                solve_info=None, cols=cols, owners=owners, fixed=fixed,
                col_flux=col_flux, seats_local=seats, seat_params=p,
                seat_amps=[float(a) for a in amps[len(fixed_bases):]])
