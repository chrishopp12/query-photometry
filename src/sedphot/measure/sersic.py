"""
sersic.py

Single-Sersic Shape Fit and Forced Photometry
---------------------------------------------------------

The forced-Sersic measurement mode: fit one Sersic profile's shape on a
chosen band (or accept explicit parameters), then force that fixed shape
at the target position in every band and solve only the amplitude -- so
every band shares one consistent profile-weighted aperture.

Requirements:
    numpy, scipy, astropy

Notes:
    The forced flux is profile-matched photometry: the total flux of the
    fixed-shape model. For a band whose light distribution differs from
    the shape band (color gradients), it is a profile-weighted amplitude,
    NOT that band's total flux.
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


def forced_photometry_single(
        stamp: np.ndarray,
        weight: np.ndarray,
        shape: dict,
        center_px: tuple,
        fwhm: float,
        pixscale: float,
        *,
        fit_mask: np.ndarray | None = None,
):
    """Single-component forced flux: fixed shape and position, amplitude only.

    The amplitude is the weighted least-squares scale of the unit-flux
    basis, so the returned flux is the model total -- profile-matched
    photometry, not the band's own total where its light distribution
    differs from the shape.

    Returns
    -------
    flux : float
        Total flux of the effective-Sersic model (image units).
    err : float
        Statistical error, 1 / sqrt(sum(w * basis^2)).
    model : np.ndarray
        Rendered model (flux * basis).
    redchi2 : float
        Reduced chi2 over the fitted pixels.
    """
    basis = sersic_basis({**shape, "xc": center_px[0], "yc": center_px[1]},
                         fwhm, pixscale, stamp.shape)
    ok = np.isfinite(stamp) & (weight > 0)
    if fit_mask is not None:
        ok &= ~fit_mask
    data = np.where(ok, stamp, 0.0)   # 0-weight NaNs would still poison sums
    weights = weight * ok
    denom = np.sum(weights * basis * basis)
    flux = float(np.sum(weights * basis * data) / denom)
    err = float(1.0 / np.sqrt(denom))
    model = flux * basis
    ndof = max(int(ok.sum()) - 1, 1)
    redchi2 = float(np.sum((weights * (data - model) ** 2)[ok]) / ndof)
    return flux, err, model, redchi2


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


# ------------------------------------
# Forced measurement (one band, fixed sky shape)
# ------------------------------------
def measure_forced(
        product,
        coord,
        shape_sky: dict,
        *,
        sky_in: float,
        sky_out: float,
        cutout_half_arcsec: float,
        user_mask: tuple | None = None,
        protect_radius: float = 4.0,
        rgrid=None,
) -> dict:
    """Forced single-Sersic flux for one band.

    Parameters
    ----------
    product : ImageProduct
        The band to measure.
    coord : SkyCoord
        Forced center.
    shape_sky : dict
        Instrument-independent shape: n, reff_arcsec, ellip, pa_deg
        (position angle E of N).
    sky_in, sky_out, cutout_half_arcsec, user_mask, protect_radius
        As in measure_aperture.
    rgrid : array, optional
        Radii (arcsec) for the model curve of growth.

    Returns
    -------
    measurement : dict
        Flux/err in uJy, the model and residual stamps, and QA curves.
    """
    from ..bands import wave_um as band_wave
    from .aperture import (COVERAGE_MIN, ApertureCoverageError,
                           DEFAULT_RGRID, prepare_stamp)

    prep = prepare_stamp(product, coord, cutout_half_arcsec=cutout_half_arcsec,
                         sky_in=sky_in, sky_out=sky_out, user_mask=user_mask,
                         protect_radius=protect_radius)
    sub, mask = prep['stamp'], prep['mask']
    nodata = prep['nodata']
    cx, cy = prep['cx'], prep['cy']
    pixscale, cf = prep['pixscale'], prep['cf']
    sky_std = prep['sky_std']

    # Coverage gate over the model-dominated region: missing pixels are
    # excluded from the fit (statistically clean), but when a big piece of
    # the galaxy itself is gone the amplitude rides on unconstrained wings.
    cov_radius = float(np.clip(3.0 * float(shape_sky['reff_arcsec']),
                               2.0, cutout_half_arcsec))
    core = prep['rr'] < cov_radius
    coverage = 1.0 - float((nodata & core).sum()) / max(int(core.sum()), 1)
    if coverage < COVERAGE_MIN:
        raise ApertureCoverageError(
            f"coverage {coverage:.2f} < {COVERAGE_MIN:g} within "
            f"{cov_radius:.1f}\" of the target (off footprint / blank)",
            coverage)

    theta_px = theta_from_pa(prep['stamp_wcs'], cx, cy, shape_sky['pa_deg'])
    shape_px = dict(reff=shape_sky['reff_arcsec'] / pixscale, n=shape_sky['n'],
                    ellip=shape_sky['ellip'], theta=theta_px)

    if product.invvar_path is not None:
        from astropy.nddata import Cutout2D
        from .calibrate import load_image
        invvar_image, _, _ = load_image(product.invvar_path)
        weight = Cutout2D(invvar_image, (prep['px'], prep['py']),
                          2 * prep['half_px'] + 1,
                          mode='partial', fill_value=0.0).data.astype(float)
        weight[~np.isfinite(weight)] = 0.0
        err_model = "ivm"
    else:
        weight = np.full(sub.shape, 1.0 / sky_std ** 2)
        err_model = "skyrms"
    weight[nodata] = 0.0

    flux, err, model, redchi2 = forced_photometry_single(
        sub, weight, shape_px, (cx, cy), product.seeing_arcsec, pixscale,
        fit_mask=mask | nodata)

    rgrid = np.asarray(rgrid if rgrid is not None else DEFAULT_RGRID, dtype=float)
    rr = prep['rr']
    model_cog = np.array([float(model[rr < radius].sum()) * cf for radius in rgrid])

    return dict(
        instrument=product.instrument, band=product.band,
        wave_um=product.wave_um if np.isfinite(product.wave_um)
        else band_wave(f"{product.instrument}_{product.band}"),
        pixscale=pixscale, cf=cf,
        flux_ujy=flux * cf, flux_err_ujy=err * cf, err_model=err_model,
        redchi2=redchi2,
        sky_level_ujy=prep['sky_level'] * cf, sky_std_ujy=sky_std * cf,
        rgrid=rgrid, enclosed_ujy=model_cog,
        stamp=sub, model=model, rr=rr, mask=mask, mask_mode=prep['mask_mode'],
        nodata=nodata,
        cx=cx, cy=cy,
        shape_sky=shape_sky, sky_in=sky_in, sky_out=sky_out,
        aperture_arcsec=float('nan'),
        n_masked_in_aperture=int(mask.sum()),
        aperture_coverage=coverage,
        masked_fraction=float((mask & core).sum()) / max(int(core.sum()), 1),
        cog_slope=float('nan'),   # the model curve is monotonic by construction
        target_ra=float(coord.ra.deg), target_dec=float(coord.dec.deg),
    )


def forced_to_row(measurement: dict) -> dict:
    """Convert a forced measurement to a schema table row."""
    from ..schema import make_row
    from ..units import flux_err_to_mag_err, ujy_to_mag
    from .aperture import qa_flags

    flux = measurement['flux_ujy']
    err = measurement['flux_err_ujy']
    shape = measurement['shape_sky']
    return make_row(
        band=f"{measurement['instrument']}_{measurement['band']}",
        flux_ujy=flux,
        flux_err_ujy=err,
        mag=ujy_to_mag(flux),
        mag_err=flux_err_to_mag_err(flux, err),
        target_ra=measurement['target_ra'],
        target_dec=measurement['target_dec'],
        match_ra=measurement['target_ra'],
        match_dec=measurement['target_dec'],
        sep_arcsec=0.0,
        flags=qa_flags(measurement),
        source=(f"sedphot_sersic_n{shape['n']:.2f}_re{shape['reff_arcsec']:.2f}as_"
                f"{measurement['mask_mode']}mask_{measurement['err_model']}"),
    )
