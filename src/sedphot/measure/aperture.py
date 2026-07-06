"""
aperture.py

Curve-of-Growth Aperture Photometry
---------------------------------------------------------

The one aperture recipe, applied identically to every instrument:

    1. cut a stamp at the target position;
    2. build the neighbor mask (auto, or a user mask reprojected onto this
       band's pixel grid);
    3. subtract a sigma-clipped annulus sky, with bright annulus sources
       matched-filter-detected and masked;
    4. integrate to the aperture -- data where unmasked, azimuthal-profile
       fill where masked;
    5. return the enclosed-flux curve of growth and the aperture flux, in
       microjanskys.

Requirements:
    numpy, astropy

Notes:
    Two error models, chosen by data availability:
      'skyrms'  sky_std * sqrt(N_aper) * cf -- used when no
                inverse-variance map exists.
      'ivm'     sqrt(sum 1/wht + N_aper^2 * sky_std^2 / N_sky) * cf -- adds
                the per-pixel noise where the archive serves real weights
                (Legacy bricks, HST); the second term is the sky-level
                uncertainty.
    Masked aperture pixels contribute their azimuthal-fill values; their
    noise is not separately inflated.
"""
from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

from ..bands import wave_um
from ..results import ImageProduct
from ..schema import make_row
from ..units import flux_err_to_mag_err, ujy_to_mag
from .calibrate import calib_factor, load_image, pixel_scale_arcsec
from .masks import neighbor_mask, radii_arcsec, reproject_mask
from .sky import annulus_sky

# ------------------------------------
# Constants
# ------------------------------------
# Curve-of-growth radius grid (arcsec).
DEFAULT_RGRID = np.arange(2.0, 30.0, 1.0)


# ------------------------------------
# Shared stamp preparation
# ------------------------------------
def prepare_stamp(
        product: ImageProduct,
        coord: SkyCoord,
        *,
        cutout_half_arcsec: float,
        sky_in: float,
        sky_out: float,
        user_mask: tuple | None = None,
        protect_radius: float = 4.0,
) -> dict:
    """Load, cut, sky-subtract, and mask one band -- the shared front half
    of both measurement modes.

    Returns
    -------
    prep : dict
        stamp (sky-subtracted), stamp_wcs, cx/cy, pixscale, cf, sky_level,
        sky_std, mask, mask_mode, annulus_srcmask, rr, px/py, half_px.
    """
    image, image_wcs, header = load_image(product.path)
    cf = calib_factor(product.calib, header)
    pixscale = pixel_scale_arcsec(image_wcs)
    px, py = [float(v) for v in image_wcs.world_to_pixel(coord)]
    half_px = int(round(cutout_half_arcsec / pixscale))
    cut = Cutout2D(image, (px, py), 2 * half_px + 1, wcs=image_wcs)
    stamp = cut.data.astype(float)
    stamp_wcs = cut.wcs
    cx, cy = [float(v) for v in stamp_wcs.world_to_pixel(coord)]
    rr = radii_arcsec(stamp.shape, cx, cy, pixscale)

    sky_level, sky_std, annulus_srcmask = annulus_sky(
        stamp, cx, cy, pixscale, sky_in=sky_in, sky_out=sky_out,
        seeing_arcsec=product.seeing_arcsec)
    sub = stamp - sky_level

    if user_mask is not None:
        mask, mask_wcs = user_mask
        if mask_wcs is not None:
            mask = reproject_mask(stamp_wcs, sub.shape, mask_wcs, mask)
        elif mask.shape != sub.shape:
            raise ValueError(
                f"user mask shape {mask.shape} != stamp shape {sub.shape} and "
                f"carries no WCS to reproject with (pass a FITS mask, or match grids)")
        mask_mode = "user"
    else:
        mask = neighbor_mask(sub, sky_std, cx, cy, pixscale,
                             protect_radius=protect_radius)
        mask_mode = "auto"

    return dict(stamp=sub, stamp_wcs=stamp_wcs, cx=cx, cy=cy, pixscale=pixscale,
                cf=cf, sky_level=sky_level, sky_std=sky_std, mask=mask,
                mask_mode=mask_mode, annulus_srcmask=annulus_srcmask, rr=rr,
                px=px, py=py, half_px=half_px)


# ------------------------------------
# Measurement
# ------------------------------------
def measure_aperture(
        product: ImageProduct,
        coord: SkyCoord,
        *,
        aperture_arcsec: float,
        sky_in: float,
        sky_out: float,
        cutout_half_arcsec: float,
        rgrid: np.ndarray | None = None,
        user_mask: tuple | None = None,
        protect_radius: float = 4.0,
) -> dict:
    """Uniform aperture measurement for one band.

    Parameters
    ----------
    product : ImageProduct
        The image to measure (path + calibration + seeing).
    coord : SkyCoord
        Aperture center.
    aperture_arcsec : float
        Aperture radius.
    sky_in, sky_out : float
        Background annulus radii (arcsec); must clear the source envelope.
    cutout_half_arcsec : float
        Stamp half-size; must contain the annulus.
    rgrid : np.ndarray, optional
        Curve-of-growth radii. [default: 2..29 arcsec, 1 arcsec steps]
    user_mask : (mask, wcs) tuple, optional
        From masks.load_user_mask; reprojected onto this band. When the
        mask carries no WCS it must already match this band's stamp grid.
    protect_radius : float
        Auto-mask protection radius around the target (arcsec). [default: 4.0]

    Returns
    -------
    measurement : dict
        Fluxes in uJy plus the stamps/masks/curves the QA figures draw.
    """
    if rgrid is None:
        rgrid = DEFAULT_RGRID
    rgrid = np.asarray(rgrid, dtype=float)

    prep = prepare_stamp(product, coord, cutout_half_arcsec=cutout_half_arcsec,
                         sky_in=sky_in, sky_out=sky_out, user_mask=user_mask,
                         protect_radius=protect_radius)
    sub = prep['stamp']
    cx, cy = prep['cx'], prep['cy']
    pixscale, cf = prep['pixscale'], prep['cf']
    sky_level, sky_std = prep['sky_level'], prep['sky_std']
    mask, mask_mode = prep['mask'], prep['mask_mode']
    rr = prep['rr']
    px, py, half_px = prep['px'], prep['py'], prep['half_px']

    # Azimuthal-profile fill of masked aperture pixels.
    edges = np.arange(0, rgrid.max() + 1, 1.0)
    profile = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        sel = (rr >= edges[i]) & (rr < edges[i + 1]) & ~mask
        if sel.sum():
            profile[i] = np.median(sub[sel])
    bin_index = np.clip(np.digitize(rr, edges) - 1, 0, len(profile) - 1)
    filled = sub.copy()
    filled[mask] = profile[bin_index[mask]]

    enclosed = np.array([float(filled[rr < radius].sum()) * cf for radius in rgrid])
    in_aperture = rr < aperture_arcsec
    flux_ujy = float(filled[in_aperture].sum()) * cf

    masked_fraction = float((mask & in_aperture).sum()) / max(int(in_aperture.sum()), 1)
    if masked_fraction > 0.2:
        print(f"  WARNING {product.instrument} {product.band}: "
              f"{100 * masked_fraction:.0f}% of the aperture is masked -- for a "
              f"bright/asymmetric target the auto-mask can eat real light; "
              f"inspect the QA figure and consider --mask")

    # Error model: inverse variance when the archive serves it, sky rms else.
    n_aper = int(in_aperture.sum())
    if product.invvar_path is not None:
        invvar_image, _, _ = load_image(product.invvar_path)
        invvar = Cutout2D(invvar_image, (px, py), 2 * half_px + 1).data.astype(float)
        ok = in_aperture & (invvar > 0)
        var_raw = float(np.sum(1.0 / invvar[ok]))
        n_sky_est = max(int(((rr > sky_in) & (rr < sky_out)).sum()), 1)
        var_sky = (n_aper * sky_std) ** 2 / n_sky_est
        flux_err_ujy = float(np.sqrt(var_raw + var_sky)) * cf
        err_model = "ivm"
    else:
        flux_err_ujy = float(sky_std * np.sqrt(n_aper)) * cf
        err_model = "skyrms"

    return dict(
        instrument=product.instrument, band=product.band,
        wave_um=product.wave_um if np.isfinite(product.wave_um)
        else wave_um(f"{product.instrument}_{product.band}"),
        pixscale=pixscale, cf=cf,
        flux_ujy=flux_ujy, flux_err_ujy=flux_err_ujy, err_model=err_model,
        sky_level_ujy=sky_level * cf, sky_std_ujy=sky_std * cf,
        rgrid=rgrid, enclosed_ujy=enclosed,
        stamp=sub, rr=rr, mask=mask, mask_mode=mask_mode,
        annulus_srcmask=prep['annulus_srcmask'],
        cx=cx, cy=cy,
        aperture_arcsec=aperture_arcsec, sky_in=sky_in, sky_out=sky_out,
        n_masked_in_aperture=int((mask & in_aperture).sum()),
        target_ra=float(coord.ra.deg), target_dec=float(coord.dec.deg),
    )


def measurement_to_row(measurement: dict) -> dict:
    """Convert a measurement dict to a schema table row."""
    flux = measurement['flux_ujy']
    err = measurement['flux_err_ujy']
    mag = ujy_to_mag(flux)
    return make_row(
        band=f"{measurement['instrument']}_{measurement['band']}",
        flux_ujy=flux,
        flux_err_ujy=err,
        mag=mag,
        mag_err=flux_err_to_mag_err(flux, err),
        target_ra=measurement['target_ra'],
        target_dec=measurement['target_dec'],
        match_ra=measurement['target_ra'],     # forced at the target position
        match_dec=measurement['target_dec'],
        sep_arcsec=0.0,
        flags='',
        source=(f"sedphot_aperture_r{measurement['aperture_arcsec']:g}as_"
                f"{measurement['mask_mode']}mask_{measurement['err_model']}"),
    )
