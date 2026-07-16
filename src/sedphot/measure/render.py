"""
render.py

Scene-Engine Rendering Primitives
---------------------------------------------------------
Every image model the scene engine draws: FFT convolution with a cached
kernel transform, Sersic profiles (unconvolved, convolved, and boxed
for speed), the Gaussian-truncated Nuker halo, the analytic full-wing
Moffat point source, the Sersic total-flux <-> amplitude conversions,
and the local sky-PA -> pixel-theta map.

Requirements:
    numpy, scipy, astropy

Notes:
    Positions and radii are stamp pixels; theta is pixel-frame radians.
    render_nuker alone also takes the pixel scale, to place its fixed
    angular truncation. Rendered images are in image counts.
    render_sersic_boxed and render_nuker return unit-amplitude shape
    columns; the amplitude solve scales them.
    conv_same caches kernel transforms by PSF object identity: build
    one PSF array per band and reuse it for every render.
"""
from __future__ import annotations

import numpy as np
from astropy.wcs import WCS
from scipy.fft import irfft2, next_fast_len, rfft2
from scipy.special import gamma, gammaincinv

from . import recipe
from .sersic import MOFFAT_BETA, theta_from_pa


# ------------------------------------
# PSF convolution
# ------------------------------------
# Kernel transforms cached per (PSF object identity, padded FFT shape).
# Each entry pins its kernel array and conv_same identity-checks it on
# every hit, so a stale transform can never serve a different kernel
# that inherited the same object id.
_CONV_CACHE = {}


def conv_same(img: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """fftconvolve(img, psf, mode='same') with a cached kernel transform.

    Every render in a band convolves with the same PSF, and fftconvolve
    re-transforms the kernel on every call; here the kernel transform
    is computed once per (PSF identity, padded shape) and reused.
    """
    ih, iw = img.shape
    kh, kw = psf.shape
    fh = next_fast_len(ih + kh - 1)
    fw = next_fast_len(iw + kw - 1)
    key = (id(psf), fh, fw)
    ent = _CONV_CACHE.get(key)
    if ent is None or ent[0] is not psf:
        ent = (psf, rfft2(psf, s=(fh, fw)))
        _CONV_CACHE[key] = ent
    out = irfft2(rfft2(img, s=(fh, fw)) * ent[1], s=(fh, fw))
    y0, x0 = (kh - 1) // 2, (kw - 1) // 2
    return out[y0:y0 + ih, x0:x0 + iw]


# ------------------------------------
# Sersic profiles
# ------------------------------------
def sersic_extent_px(reff_px: float, n: float, frac: float = 0.9999) -> float:
    """Radius (px) enclosing `frac` of a Sersic's total flux (analytic)."""
    bn = gammaincinv(2 * n, 0.5)
    return float(reff_px * (gammaincinv(2 * n, frac) / bn) ** n)


def sersic_profile(
        params: list | np.ndarray,
        shape_2d: tuple[int, int],
) -> np.ndarray:
    """Unconvolved Sersic image by direct evaluation of the Sersic2D formula.

    Matches astropy.modeling.models.Sersic2D to 1e-12; the direct numpy
    evaluation exists because the model-class call overhead dominates
    the render cost inside a shape solve. The effective radius is
    floored at 0.3 px so a collapsing trial cannot produce a delta
    function.

    Parameters
    ----------
    params : list or np.ndarray
        [ampl, reff_px, n, ellip, theta, x0, y0]: amplitude at the
        effective radius (counts/px), effective radius (px), Sersic
        index, ellipticity, position angle (rad, pixel frame), and
        center (stamp px).
    shape_2d : tuple
        (ny, nx) of the output image.

    Returns
    -------
    profile : np.ndarray
        Unconvolved Sersic image (counts).
    """
    ampl, reff_px, n, ellip, theta, x0, y0 = params
    reff_px = max(reff_px, 0.3)
    yy, xx = np.indices(shape_2d, dtype=float)
    ct, st = np.cos(theta), np.sin(theta)
    xmaj = ((xx - x0) * ct + (yy - y0) * st) / reff_px
    xmin = (-(xx - x0) * st + (yy - y0) * ct) / ((1.0 - ellip) * reff_px)
    z = np.sqrt(xmaj * xmaj + xmin * xmin)
    bn = gammaincinv(2 * n, 0.5)
    return ampl * np.exp(-bn * (z ** (1.0 / n) - 1.0))


def render_sersic(
        params: list | np.ndarray,
        shape_2d: tuple[int, int],
        psf: np.ndarray,
) -> np.ndarray:
    """PSF-convolved Sersic image; params as in sersic_profile."""
    return conv_same(sersic_profile(params, shape_2d), psf)


def render_sersic_boxed(
        reff_px: float,
        n: float,
        ellip: float,
        theta: float,
        x0: float,
        y0: float,
        shape_2d: tuple[int, int],
        psf: np.ndarray,
) -> np.ndarray:
    """Unit-amplitude Sersic rendered on an adaptive box in a zero frame.

    The box spans the 99.9%-flux radius for this (reff_px, n) plus the
    kernel footprint, quantized up to a multiple of 32 px so solver
    trials reuse a few FFT sizes instead of paying for a new one per
    trial. The result matches the full-frame render to <= 1e-4 of the
    total flux. Rare high-n trials get big boxes, the common low-n case
    gets the speedup, and when the box would not help the render falls
    back to the full frame.
    """
    ih, iw = shape_2d
    # n is clamped to the seat index range for the extent estimate
    # only; the profile itself renders at the requested n.
    half = (sersic_extent_px(reff_px, min(max(n, 0.4), 6.0), frac=0.999)
            + max(psf.shape))
    if half >= 0.45 * max(ih, iw):
        return render_sersic([1.0, reff_px, n, ellip, theta, x0, y0],
                             shape_2d, psf)
    half = (int(half) // 32 + 1) * 32
    xlo, xhi = int(round(x0)) - half, int(round(x0)) + half + 1
    ylo, yhi = int(round(y0)) - half, int(round(y0)) + half + 1
    cxlo, cxhi = max(xlo, 0), min(xhi, iw)
    cylo, cyhi = max(ylo, 0), min(yhi, ih)
    out = np.zeros(shape_2d)
    if cxlo >= cxhi or cylo >= cyhi:
        return out
    sub = sersic_profile([1.0, reff_px, n, ellip, theta,
                          x0 - cxlo, y0 - cylo],
                         (cyhi - cylo, cxhi - cxlo))
    out[cylo:cyhi, cxlo:cxhi] = conv_same(sub, psf)
    return out


# ------------------------------------
# Nuker halo
# ------------------------------------
def render_nuker(
        rb_px: float,
        beta: float,
        ellip: float,
        theta: float,
        x0: float,
        y0: float,
        shape_2d: tuple[int, int],
        psf: np.ndarray,
        pixscale: float,
) -> np.ndarray:
    """Unit-amplitude, PSF-convolved Nuker halo with a Gaussian truncation.

    The outer profile family for solved envelopes: Sersic-family outers
    refuse cD envelopes -- the data want shallow-with-an-edge. The
    inner slope (recipe.NUKER_GAMMA) and break sharpness
    (recipe.NUKER_ALPHA) are frozen; only the outer slope beta and the
    break radius rb_px solve. The Gaussian truncation at
    recipe.NUKER_TRUNC_AS keeps the shallow outer power law from
    extending forever, and a +0.3 px softening keeps the inner cusp
    finite on the pixel grid.
    """
    yy, xx = np.indices(shape_2d)
    ct, st = np.cos(theta), np.sin(theta)
    u = (xx - x0) * ct + (yy - y0) * st
    v = -(xx - x0) * st + (yy - y0) * ct
    r = np.sqrt(u * u + (v / (1.0 - ellip + 1e-9)) ** 2) + 0.3
    z = (r / rb_px) ** recipe.NUKER_ALPHA
    rtrunc = recipe.NUKER_TRUNC_AS / pixscale
    img = ((r / rb_px) ** -recipe.NUKER_GAMMA
           * (1.0 + z) ** ((recipe.NUKER_GAMMA - beta) / recipe.NUKER_ALPHA)
           * np.exp(-(r / rtrunc) ** 2))
    return conv_same(img, psf)


# ------------------------------------
# Moffat point-source wings
# ------------------------------------
def moffat_wings(
        counts: float,
        fwhm_px: float,
        x0: float,
        y0: float,
        shape_2d: tuple[int, int],
) -> np.ndarray:
    """Analytic full-wing Moffat image of a point source.

    For a bright point source just off the stamp, a rendered kernel
    stamp truncates exactly the wings that still cross the edge; the
    analytic profile keeps them. beta is the standard PSF MOFFAT_BETA,
    gamma follows from the FWHM, and the normalization makes the
    infinite-plane integral equal `counts` -- any finite frame sums to
    less.
    """
    gam = fwhm_px / (2 * np.sqrt(2 ** (1 / MOFFAT_BETA) - 1))
    yy, xx = np.indices(shape_2d)
    rr2 = (xx - x0) ** 2 + (yy - y0) ** 2
    return (counts * (MOFFAT_BETA - 1) / (np.pi * gam ** 2)
            * (1 + rr2 / gam ** 2) ** -MOFFAT_BETA)


# ------------------------------------
# Sersic flux <-> amplitude
# ------------------------------------
def sersic_total(
        ampl: float,
        reff_px: float,
        n: float,
        ellip: float,
        cf: float,
) -> float:
    """Total flux (uJy) of an unconvolved Sersic with these parameters.

    The closed-form Sersic integral over the infinite plane, times the
    counts -> microjansky calibration factor cf (pass cf=1 for counts).
    """
    bn = gammaincinv(2 * n, 0.5)
    return float(ampl * 2 * np.pi * n * np.exp(bn) * bn ** (-2 * n)
                 * gamma(2 * n) * reff_px ** 2 * (1 - ellip) * cf)


def ampl_from_total(
        counts: float,
        reff_px: float,
        n: float,
        ellip: float,
) -> float:
    """Sersic amplitude (counts/px at the effective radius) whose total
    flux is `counts` -- the inverse of sersic_total at cf=1."""
    bn = gammaincinv(2 * n, 0.5)
    return counts / (2 * np.pi * n * np.exp(bn) * bn ** (-2 * n)
                     * gamma(2 * n) * reff_px ** 2 * (1 - ellip))


# ------------------------------------
# Position angles
# ------------------------------------
def pa_map(wcs: WCS, x: float, y: float) -> tuple[float, float]:
    """Local affine sky-PA -> pixel-theta map: theta = t0 + slope * pa.

    Two theta_from_pa evaluations (PA 0 and 90 deg E of N) anchor the
    map, so a solver can vary a position angle without a WCS round trip
    per trial. The profiles the thetas feed are 180-degree symmetric,
    so a branch-cut wrap between the two anchors is harmless.

    Returns
    -------
    t0, slope : float
        Pixel-frame theta (rad) at PA = 0, and rad per degree of PA.
    """
    t0 = theta_from_pa(wcs, x, y, 0.0)
    return t0, (theta_from_pa(wcs, x, y, 90.0) - t0) / 90.0
