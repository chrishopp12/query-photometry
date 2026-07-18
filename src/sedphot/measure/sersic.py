"""
sersic.py

Single-Sersic Shape Fit
---------------------------------------------------------
Fit one Sersic profile's shape on a chosen band, or accept explicit
parameters -- the shape source for the SPHEREx forced model and for
pinning the scene engine's target profile. The Moffat PSF and the
WCS position-angle transfer helpers live here because every consumer
of a fitted shape needs them.

Requirements:
    numpy, scipy, astropy

Notes:
    Position angles cross the module boundary as degrees east of north
    and convert to/from pixel-frame theta through each image's WCS, so a
    shape fit on one instrument transfers correctly to any other
    orientation.
    Fitted n and r_eff are PSF-sensitive -- errors in the assumed seeing
    map directly into them -- so explicit, trusted shape parameters are
    the precision path.
"""
from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.modeling.models import Moffat2D, Sersic2D
from astropy.wcs import WCS
from scipy.optimize import least_squares
from scipy.signal import fftconvolve

# ------------------------------------
# Constants
# ------------------------------------
MOFFAT_BETA = 3.0
SERSIC_N_MAX = 8.0        # fit bound; the SPHEREx tool caps n at 6


# ------------------------------------
# PSF and basis
# ------------------------------------
def moffat_psf(
        fwhm_arcsec: float,
        pixscale: float,
        *,
        beta: float = MOFFAT_BETA,
        size: int = 25,
) -> np.ndarray:
    """Unit-sum Moffat PSF stamp at the image pixel scale."""
    fwhm_pix = fwhm_arcsec / pixscale
    gamma = fwhm_pix / (2 * np.sqrt(2 ** (1 / beta) - 1))
    yy, xx = np.mgrid[0:size, 0:size]
    psf = Moffat2D(1.0, size // 2, size // 2, gamma, beta)(xx, yy)
    return psf / psf.sum()


def sersic_basis(
        shape: dict,
        fwhm_arcsec: float,
        pixscale: float,
        stamp_shape: tuple,
        *,
        oversample: int = 3,
) -> np.ndarray:
    """Unit-flux, PSF-convolved Sersic basis image (matched aperture).

    Renders the fixed-shape Sersic with pixel-area integration (oversample +
    bin), convolves with the band Moffat, and normalizes to unit sum -- so a
    fitted amplitude equals the total flux of the component.

    Parameters
    ----------
    shape : dict
        Sersic shape with keys xc, yc, reff (px), n, ellip, theta (rad,
        pixel frame).
    fwhm_arcsec, pixscale : float
        Band PSF FWHM and arcsec/pixel.
    stamp_shape : tuple
        (ny, nx) of the stamp.

    Returns
    -------
    basis : np.ndarray
        Unit-sum PSF-convolved model image.
    """
    ny, nx = stamp_shape
    idx = (np.arange(max(ny, nx) * oversample) + 0.5) / oversample - 0.5
    grid_x, grid_y = np.meshgrid(idx[:nx * oversample], idx[:ny * oversample])
    fine = Sersic2D(
        1.0, r_eff=shape["reff"], n=shape["n"],
        x_0=shape["xc"], y_0=shape["yc"], ellip=shape["ellip"], theta=shape["theta"],
    )(grid_x, grid_y)
    model = fine.reshape(ny, oversample, nx, oversample).mean(axis=(1, 3))
    convolved = fftconvolve(model, moffat_psf(fwhm_arcsec, pixscale), mode="same")
    return convolved / convolved.sum()


# ------------------------------------
# Position-angle transfer through the WCS
# ------------------------------------
def pa_east_of_north(stamp_wcs: WCS, cx: float, cy: float, theta_rad: float) -> float:
    """Sky position angle (deg E of N) of a pixel-frame major axis."""
    here = stamp_wcs.pixel_to_world(cx, cy)
    there = stamp_wcs.pixel_to_world(cx + 10 * np.cos(theta_rad),
                                     cy + 10 * np.sin(theta_rad))
    return float(here.position_angle(there).to(u.deg).value % 180.0)


def theta_from_pa(stamp_wcs: WCS, cx: float, cy: float, pa_deg: float) -> float:
    """Pixel-frame theta (rad) whose sky position angle is pa_deg E of N."""
    here = stamp_wcs.pixel_to_world(cx, cy)
    there = here.directional_offset_by(pa_deg * u.deg, 10 * u.arcsec)
    px, py = stamp_wcs.world_to_pixel(there)
    return float(np.arctan2(float(py) - cy, float(px) - cx))


# ------------------------------------
# Shape fit
# ------------------------------------
def fit_sersic_shape(
        stamp: np.ndarray,
        sky_std: float,
        cx: float,
        cy: float,
        pixscale: float,
        seeing_arcsec: float,
        *,
        mask: np.ndarray | None = None,
        fit_radius_arcsec: float = 12.0,
) -> dict:
    """Least-squares single-Sersic shape fit on one band.

    Fits (amplitude, xc, yc, r_eff, n, ellip, theta) on a sub-stamp around
    the target; the amplitude is discarded (the forced solve re-fits it per
    band) and the shape is what transfers. The fitted n and r_eff are
    PSF-sensitive: an error in seeing_arcsec maps directly into them.

    Parameters
    ----------
    stamp : np.ndarray
        Sky-subtracted stamp.
    sky_std : float
        Per-pixel background rms (residual weighting).
    cx, cy : float
        Target position in stamp pixels.
    pixscale, seeing_arcsec : float
        Pixel scale and band PSF FWHM.
    mask : np.ndarray, optional
        Neighbor mask (True = exclude).
    fit_radius_arcsec : float
        Sub-stamp half-size for the fit. [default: 12]

    Returns
    -------
    shape : dict
        n, reff_arcsec, ellip, theta (rad, THIS stamp's pixel frame),
        xc/yc (fitted center, full-stamp pixels), redchi2, success.
    """
    half = int(round(fit_radius_arcsec / pixscale))
    x0, y0 = int(round(cx)), int(round(cy))
    ys = slice(max(y0 - half, 0), y0 + half + 1)
    xs = slice(max(x0 - half, 0), x0 + half + 1)
    sub = stamp[ys, xs]
    sub_mask = np.zeros(sub.shape, bool) if mask is None else mask[ys, xs]
    scx, scy = cx - xs.start, cy - ys.start
    ok = np.isfinite(sub) & ~sub_mask

    amp0 = max(float(sub[ok].sum()), 10.0 * sky_std)
    p0 = [np.log10(amp0), scx, scy, np.log10(3.0 / pixscale), np.log10(2.5), 0.2, 0.5]
    bounds = ([np.log10(amp0) - 3, scx - 5, scy - 5, np.log10(0.5), np.log10(0.5),
               0.0, -np.pi],
              [np.log10(amp0) + 3, scx + 5, scy + 5, np.log10(2.0 * half),
               np.log10(SERSIC_N_MAX), 0.85, np.pi])

    def residual(params):
        log_amp, x, y, log_reff, log_n, ellip, theta = params
        shape = dict(xc=x, yc=y, reff=10 ** log_reff, n=10 ** log_n,
                     ellip=ellip, theta=theta)
        model = 10 ** log_amp * sersic_basis(shape, seeing_arcsec, pixscale, sub.shape)
        return ((model - sub)[ok] / sky_std).ravel()

    fit = least_squares(residual, p0, bounds=bounds, x_scale='jac', max_nfev=300)
    log_amp, x, y, log_reff, log_n, ellip, theta = fit.x
    ndof = max(int(ok.sum()) - 7, 1)
    return dict(
        n=float(10 ** log_n),
        reff_arcsec=float(10 ** log_reff * pixscale),
        ellip=float(ellip),
        theta=float(theta),
        xc=float(x + xs.start), yc=float(y + ys.start),
        redchi2=float(2 * fit.cost / ndof),
        success=bool(fit.success),
    )
