"""
background.py

Scene Background: One Owner per Spatial Scale
---------------------------------------------------------
The scene engine's background is estimated here and nowhere else: a
plane through sigma-clipped bin MEANS with bin-level MAD rejection.
The plane owns light varying at cutout scale -- a level and a tilt,
nothing sharper. Bins coherently elevated beyond the rejection
threshold (halo skirts, tidal light) are source structure and lose
their vote. Ownership of light is positional, not statistical: a bin
is background because of where it sits and what survives rejection
there, never because a fit found it convenient to call it background.
The plane never sits in a design matrix next to component amplitudes
-- it alternates with the amplitude solve, each estimated on the
other's residual.

Three estimators own three scales and none may take another's. The
plane owns level and tilt across the cutout. residual_mesh owns smooth
structure below that, built on the post-fit residual, zeroed of DC over
the plane's own accepted bins, subtracted only inside the curve of
growth and never fed back into a fit. The fitted source model owns
compact light. eval_plane and eval_mesh rebuild either surface from the
provenance sidecar's stored record, so a reconstruction reproduces the
background without re-binning.

The bin statistic is a clipped mean, not a median, because photometry
SUMS pixels: the background must estimate the mean sky under the
aperture, and a median silently diverges from it on any skewed or
spiked pixel distribution. Survey frames carry such spikes (heavily
repeated values from upstream integerization), and a spiked histogram
mode-locks bin medians: their bin-to-bin scatter collapses far below
noise, the MAD rejection threshold collapses with it and thrashes,
and the background inherits a (mean - median) x area flux systematic.
The clip supplies the robustness the median was doing duty for.

ambient_surface shares the same bin grid: a smoothed bin-mean
surface that downstream consumers (the flood mask channel, the twin
fill's asymmetry localization) read as the local ambient level.

Requirements:
    numpy, scipy, astropy

Notes:
    work is in native image counts, so the plane, the mesh, and the
    ambient surface are in counts too. rr is the stamp's radius map
    about the target (arcsec). The bin size derives from recipe.BIN_AS
    at the band's pixel scale, floored at 4 px.
"""
from __future__ import annotations

import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.stats import sigma_clipped_stats
from scipy.interpolate import RegularGridInterpolator

from . import recipe


# ------------------------------------
# The shared bin grid
# ------------------------------------
def bin_grid(
        work: np.ndarray,
        usable: np.ndarray,
        pixscale: float,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Sigma-clipped MEAN of the image in each bin of a regular grid.

    The shared front end of bin_plane, ambient_surface, and
    residual_mesh, so every consumer sees the same bins. A bin votes
    only when at least BIN_MIN_FRAC of its pixels are usable -- a bin
    dominated by masked or missing pixels has no honest level to
    offer. The statistic is a clipped mean, never a median: photometry
    sums pixels, and on the spiked pixel histograms real survey frames
    carry, bin medians mode-lock (see the module docstring).

    Parameters
    ----------
    work : np.ndarray
        Stamp-shaped image (counts).
    usable : np.ndarray
        Boolean map of pixels allowed to vote in the bin levels.
    pixscale : float
        Pixel scale (arcsec/px); sets the bin size from recipe.BIN_AS.

    Returns
    -------
    row_starts : np.ndarray
        First stamp row of each bin row.
    col_starts : np.ndarray
        First stamp column of each bin column.
    bin_px : int
        Bin size (px).
    levels : np.ndarray
        Clipped mean per bin, shape (len(row_starts),
        len(col_starts)), NaN where a bin does not vote.
    """
    bin_px = max(int(round(recipe.BIN_AS / pixscale)), 4)
    ny, nx = work.shape
    row_starts = np.arange(0, ny - bin_px + 1, bin_px)
    col_starts = np.arange(0, nx - bin_px + 1, bin_px)
    levels = np.full((len(row_starts), len(col_starts)), np.nan)
    # A vectorized variant of this loop measures slower at identical
    # output -- the per-bin blocks are small and cheap. Measure before
    # optimizing.
    for i, y0 in enumerate(row_starts):
        for j, x0 in enumerate(col_starts):
            block = (slice(y0, y0 + bin_px), slice(x0, x0 + bin_px))
            voters = usable[block]
            if voters.sum() < recipe.BIN_MIN_FRAC * bin_px * bin_px:
                continue
            mean, _, _ = sigma_clipped_stats(work[block][voters],
                                             sigma=3.0, maxiters=5)
            levels[i, j] = mean
    return row_starts, col_starts, bin_px, levels


# ------------------------------------
# The background plane
# ------------------------------------
def bin_plane(
        work: np.ndarray,
        good: np.ndarray,
        rr: np.ndarray,
        pixscale: float,
) -> dict:
    """THE background: a plane through the voting bin levels.

    Within-bin clipping cannot catch a bin that is uniformly bright,
    so rejection also happens at the bin level: bins elevated beyond
    BG_REJ_SIGMA x the robust bin-to-bin scatter are source structure
    and lose their vote. Pixels inside BG_RMIN_AS of the target never
    vote at all -- target light is excluded by position, not left to
    rejection.

    Parameters
    ----------
    work : np.ndarray
        Image to fit (counts) -- the raw stamp, or the scene-
        subtracted stamp inside the fit alternation.
    good : np.ndarray
        Boolean map of usable pixels.
    rr : np.ndarray
        Radius map about the target (arcsec).
    pixscale : float
        Pixel scale (arcsec/px).

    Returns
    -------
    plane : dict
        img : np.ndarray, the plane evaluated over the stamp (counts).
        const : float, the plane level at the stamp center.
        coefs : list of 3 floats, [level, x tilt, y tilt] in the
            centered / normalized parametrization below.
        n_rej : int, bins that lost their vote to rejection.
        n_bins : int, bins that voted before rejection.
        keep_px : np.ndarray, boolean map of the ACCEPTED bins'
            territory -- the pixels whose level the plane claims.
            Consumers that must not re-adjudicate the level (the
            residual mesh) zero themselves over this region.
    """
    usable = good & (rr > recipe.BG_RMIN_AS)
    row_starts, col_starts, bin_px, levels = bin_grid(work, usable,
                                                      pixscale)
    ny, nx = work.shape
    ii, jj = np.where(np.isfinite(levels))
    pts = levels[ii, jj]

    # Centered / normalized design at the bin centers: the constant
    # column reads directly as the level at the stamp center, and the
    # tilt columns stay order-unity for any stamp size.
    x_centers = col_starts[jj] + bin_px / 2.0
    y_centers = row_starts[ii] + bin_px / 2.0
    design = np.column_stack([np.ones(len(pts)),
                              (x_centers - nx / 2) / nx,
                              (y_centers - ny / 2) / ny])

    # Bin-level MAD rejection, re-fit until the vote is stable.
    # 1.4826 x the median absolute deviation estimates a Gaussian
    # sigma robustly; the guard keeps an all-equal residual set from
    # zeroing the threshold. The keep decision is recomputed against
    # ALL bins each pass, so a bin rejected by an early, still-biased
    # fit can win its vote back.
    keep = np.ones(len(pts), bool)
    coef = np.zeros(3)
    for _ in range(6):
        coef, *_ = np.linalg.lstsq(design[keep], pts[keep], rcond=None)
        res = pts - design @ coef
        kept_res = res[keep]
        sig = 1.4826 * np.median(np.abs(kept_res - np.median(kept_res)))
        new_keep = np.abs(res) < recipe.BG_REJ_SIGMA * max(sig, 1e-12)
        if (new_keep == keep).all():
            break
        keep = new_keep

    # Evaluated through eval_plane, the same function the sidecar
    # reconstruction uses, so the parametrization has exactly one home.
    img = eval_plane(coef, work.shape)
    keep_px = np.zeros(work.shape, bool)
    for i, j, kept in zip(ii, jj, keep):
        if kept:
            keep_px[row_starts[i]:row_starts[i] + bin_px,
                    col_starts[j]:col_starts[j] + bin_px] = True
    return dict(img=img, const=float(coef[0]),
                coefs=[float(v) for v in coef],
                n_rej=int((~keep).sum()), n_bins=int(len(pts)),
                keep_px=keep_px)


# ------------------------------------
# The ambient surface
# ------------------------------------
def ambient_surface(
        work: np.ndarray,
        good: np.ndarray,
        mask: np.ndarray,
        rr: np.ndarray,
        pixscale: float,
) -> np.ndarray | None:
    """Smoothed bin-level surface: the local ambient reference.

    Same bins as bin_plane, with masked pixels also barred from
    voting and no plane fit at all: the consumers compare each pixel
    to the ambient level HERE, not to the global plane. Non-voting
    bins are filled from their neighbors by the NaN-interpolating
    smoothing; the smoothed grid then interpolates back to full
    stamp resolution.

    Parameters
    ----------
    work : np.ndarray
        Image to bin (counts).
    good : np.ndarray
        Boolean map of usable pixels.
    mask : np.ndarray
        Boolean source mask; masked pixels never vote.
    rr : np.ndarray
        Radius map about the target (arcsec).
    pixscale : float
        Pixel scale (arcsec/px).

    Returns
    -------
    ambient : np.ndarray or None
        Ambient level per stamp pixel (counts), NaN outside the hull
        of the bin centers; None when eight or fewer bins vote --
        too few for a surface worth the name.
    """
    usable = good & ~mask & (rr > recipe.BG_RMIN_AS)
    row_starts, col_starts, bin_px, levels = bin_grid(work, usable,
                                                      pixscale)
    if np.isfinite(levels).sum() <= 8:
        return None
    smoothed = convolve(levels, Gaussian2DKernel(1.0),
                        boundary='extend', nan_treatment='interpolate',
                        preserve_nan=False)
    interp = RegularGridInterpolator(
        (row_starts + bin_px / 2.0, col_starts + bin_px / 2.0),
        smoothed, bounds_error=False, fill_value=np.nan)
    ny, nx = work.shape
    yy, xx = np.indices((ny, nx))
    return interp(np.stack([yy.ravel(), xx.ravel()],
                           axis=1)).reshape((ny, nx))


# ------------------------------------
# The measurement-side residual mesh
# ------------------------------------
def residual_mesh(
        resid: np.ndarray,
        vote: np.ndarray,
        pixscale: float,
        *,
        level_px: np.ndarray | None = None,
        state: dict | None = None,
) -> np.ndarray:
    """Post-fit background surface: the bin-level mesh of the residual.

    Built on light no model claimed -- image minus fitted scene minus
    the plane -- and subtracted only inside the curve of growth, never
    fed back into any fit. The construction bounds its resolution to
    background scales: recipe.BIN_AS bin levels, one-bin Gaussian
    smoothing (NaN-interpolating, so non-voting bins fill from
    neighbors), and a bilinear return to pixels with queries CLAMPED to
    the bin-center hull (no extrapolation growth at the stamp edge).
    Structure sharper than about two bins -- cores, PSF residuals, fit
    dipoles -- is invisible to it by construction, so the mesh can only
    ever own background-scale light, and a well-fit source leaves it
    nothing to take.

    The mesh carries STRUCTURE, never level: it is zeroed over the
    plane's own accepted-bin territory (level_px), so the level keeps
    exactly one owner. Without the zero point, coherent structure the
    plane's rejection excluded (a stack seam, an envelope-misfit patch)
    drags the mesh's field mean off zero, and subtracting it re-levies
    the whole field by that mean -- a DC the plane already set.

    Parameters
    ----------
    resid : np.ndarray
        Fit residual (counts): image - fitted scene - background plane.
    vote : np.ndarray
        Pixels allowed to vote (usable and not neighbor-masked).
    pixscale : float
        Pixel scale (arcsec/px).
    level_px : np.ndarray, optional
        The plane's accepted-bin pixel map (bin_plane's keep_px); the
        mesh is zeroed over vote & level_px. None zeroes over vote.
    state : dict, optional
        Filled with everything needed to re-evaluate this exact mesh
        without the pixels (row_starts, col_starts, bin_px, the
        smoothed grid, and the zero offset) -- the sidecar's
        reconstruction record.

    Returns
    -------
    mesh : np.ndarray
        The residual background surface per pixel (counts); zeros when
        fewer than nine bins vote.
    """
    row_starts, col_starts, bin_px, levels = bin_grid(resid, vote,
                                                      pixscale)
    if np.isfinite(levels).sum() <= 8:
        return np.zeros_like(resid)
    smoothed = convolve(levels, Gaussian2DKernel(1.0), boundary='extend',
                        nan_treatment='interpolate', preserve_nan=False)
    mesh = _bin_surface(row_starts, col_starts, bin_px, smoothed,
                        resid.shape)
    zero_over = vote if level_px is None else (vote & level_px)
    if not zero_over.any():
        zero_over = vote
    zero = float(mesh[zero_over].mean()) if zero_over.any() else 0.0
    mesh = mesh - zero
    if state is not None:
        state.update(row_starts=row_starts.tolist(),
                     col_starts=col_starts.tolist(), bin_px=int(bin_px),
                     smoothed=np.round(smoothed, 7).tolist(),
                     zero=round(zero, 7))
    return mesh


# ------------------------------------
# Sidecar reconstruction: re-evaluate a stored background
# ------------------------------------
def eval_plane(coefs, shape: tuple[int, int]) -> np.ndarray:
    """Re-evaluate a stored background plane over a stamp of `shape`.

    The inverse of bin_plane's `coefs` -- [level, x-tilt, y-tilt] in the
    centered/normalized parametrization -- so a pinned reconstruction
    reproduces bin_plane's `img` from the sidecar alone, with no bins and
    no fit. bin_plane BUILDS its own plane through this function, so the
    two cannot drift apart: there is one parametrization, in one place.
    """
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    c0, c1, c2 = (float(v) for v in coefs)
    return c0 + c1 * (xx - nx / 2) / nx + c2 * (yy - ny / 2) / ny


def _bin_surface(row_starts, col_starts, bin_px: float,
                 smoothed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Bilinear bin-grid surface at pixel resolution, clamped to the hull.

    The shared core of residual_mesh and eval_mesh: queries are clamped to
    the bin-center hull so the surface never extrapolates outward at the
    stamp edge. Both the producer and the sidecar reconstruction call it,
    so a mesh always re-evaluates to what it was.
    """
    rs = np.asarray(row_starts, float)
    cs = np.asarray(col_starts, float)
    interp = RegularGridInterpolator((rs + bin_px / 2.0, cs + bin_px / 2.0),
                                     smoothed, bounds_error=False,
                                     fill_value=None)
    ny, nx = shape
    yy, xx = np.indices((ny, nx))
    ys = np.clip(yy.ravel(), rs[0] + bin_px / 2.0, rs[-1] + bin_px / 2.0)
    xs = np.clip(xx.ravel(), cs[0] + bin_px / 2.0, cs[-1] + bin_px / 2.0)
    return interp(np.stack([ys, xs], axis=1)).reshape(ny, nx)


def eval_mesh(state: dict | None, shape: tuple[int, int]) -> np.ndarray:
    """Re-evaluate a stored residual mesh over a stamp of `shape`.

    The inverse of residual_mesh's `state` record: the same clamped
    bilinear surface over the same bin-center grid, minus the same DC
    zero. Returns zeros when the fit stored no mesh (too few voting bins,
    or a band that never built one).
    """
    if not state or not state.get('smoothed'):
        return np.zeros(shape)
    mesh = _bin_surface(state['row_starts'], state['col_starts'],
                        float(state['bin_px']),
                        np.asarray(state['smoothed'], float), shape)
    return mesh - float(state.get('zero', 0.0))
