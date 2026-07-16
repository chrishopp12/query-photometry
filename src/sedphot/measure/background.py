"""
background.py

Scene Background: One Owner, One Estimator
---------------------------------------------------------
The scene engine's background is estimated here and nowhere else: a
plane through sigma-clipped bin medians with bin-level MAD rejection.
The plane owns light varying at cutout scale -- a level and a tilt,
nothing sharper. Bins coherently elevated beyond the rejection
threshold (halo skirts, tidal light) are source structure and lose
their vote. Ownership of light is positional, not statistical: a bin
is background because of where it sits and what survives rejection
there, never because a fit found it convenient to call it background.
The plane never sits in a design matrix next to component amplitudes
-- it alternates with the amplitude solve, each estimated on the
other's residual.

ambient_surface shares the same bin grid: a smoothed bin-median
surface that downstream consumers (the flood mask channel, the twin
fill's asymmetry localization) read as the local ambient level.

Requirements:
    numpy, scipy, astropy

Notes:
    work is in native image counts, so the plane and the ambient
    surface are in counts too. rr is the stamp's radius map about the
    target (arcsec). The bin size derives from recipe.BIN_AS at the
    band's pixel scale, floored at 4 px.
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
    """Sigma-clipped median of the image in each bin of a regular grid.

    The shared front end of bin_plane and ambient_surface, so both
    consumers see the same bins. A bin votes only when at least
    BIN_MIN_FRAC of its pixels are usable -- a bin dominated by masked
    or missing pixels has no honest median to offer.

    Parameters
    ----------
    work : np.ndarray
        Stamp-shaped image (counts).
    usable : np.ndarray
        Boolean map of pixels allowed to vote in the bin medians.
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
    medians : np.ndarray
        Clipped median per bin, shape (len(row_starts),
        len(col_starts)), NaN where a bin does not vote.
    """
    bin_px = max(int(round(recipe.BIN_AS / pixscale)), 4)
    ny, nx = work.shape
    row_starts = np.arange(0, ny - bin_px + 1, bin_px)
    col_starts = np.arange(0, nx - bin_px + 1, bin_px)
    medians = np.full((len(row_starts), len(col_starts)), np.nan)
    # A vectorized variant of this loop was measured slower at
    # identical output -- the per-bin blocks are small and cheap.
    # Measure before optimizing.
    for i, y0 in enumerate(row_starts):
        for j, x0 in enumerate(col_starts):
            block = (slice(y0, y0 + bin_px), slice(x0, x0 + bin_px))
            voters = usable[block]
            if voters.sum() < recipe.BIN_MIN_FRAC * bin_px * bin_px:
                continue
            _, median, _ = sigma_clipped_stats(work[block][voters],
                                               sigma=3.0, maxiters=5)
            medians[i, j] = median
    return row_starts, col_starts, bin_px, medians


# ------------------------------------
# The background plane
# ------------------------------------
def bin_plane(
        work: np.ndarray,
        good: np.ndarray,
        rr: np.ndarray,
        pixscale: float,
) -> dict:
    """THE background: a plane through the voting bin medians.

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
    """
    usable = good & (rr > recipe.BG_RMIN_AS)
    row_starts, col_starts, bin_px, medians = bin_grid(work, usable,
                                                       pixscale)
    ny, nx = work.shape
    ii, jj = np.where(np.isfinite(medians))
    pts = medians[ii, jj]

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

    yy, xx = np.indices(work.shape)
    img = (coef[0] + coef[1] * (xx - nx / 2) / nx
           + coef[2] * (yy - ny / 2) / ny)
    return dict(img=img, const=float(coef[0]),
                coefs=[float(v) for v in coef],
                n_rej=int((~keep).sum()), n_bins=int(len(pts)))


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
    """Smoothed bin-median surface: the local ambient reference.

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
    row_starts, col_starts, bin_px, medians = bin_grid(work, usable,
                                                       pixscale)
    if np.isfinite(medians).sum() <= 8:
        return None
    smoothed = convolve(medians, Gaussian2DKernel(1.0),
                        boundary='extend', nan_treatment='interpolate',
                        preserve_nan=False)
    interp = RegularGridInterpolator(
        (row_starts + bin_px / 2.0, col_starts + bin_px / 2.0),
        smoothed, bounds_error=False, fill_value=np.nan)
    ny, nx = work.shape
    yy, xx = np.indices((ny, nx))
    return interp(np.stack([yy.ravel(), xx.ravel()],
                           axis=1)).reshape((ny, nx))
