"""
stamp.py

Stage 1: Stamp Preparation and Data-Sufficiency Gates
---------------------------------------------------------
Cut the science stamp, calibrate it, flag unusable pixels, and measure
the global noise scale and the far-field level. Everything downstream
sees only the Stamp built here. No background is estimated at this
stage -- the background is owned entirely by background.bin_plane.

Requirements:
    numpy, astropy

Notes:
    sigma is a global clipped scatter of the outer stamp: an upper bound
    on the pixel noise used for thresholds and solver scales, not a
    background estimate. nodata marks non-finite pixels, exact archive
    zeros (off-footprint fill), and deeply negative outliers.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from . import recipe
from .calibrate import calib_factor, load_image, pixel_scale_arcsec


class ApertureCoverageError(RuntimeError):
    """The photometry aperture lands on too many missing pixels."""

    def __init__(self, message: str, coverage: float):
        super().__init__(message)
        self.coverage = coverage


def radii_arcsec(shape: tuple, cx: float, cy: float, pixscale: float) -> np.ndarray:
    """Radius map (arcsec) about a stamp-pixel center."""
    yy, xx = np.indices(shape)
    return np.hypot(xx - cx, yy - cy) * pixscale


# ------------------------------------
# The stamp
# ------------------------------------
@dataclass
class Stamp:
    """One calibrated cutout, ready for scene fitting and measurement.

    Attributes
    ----------
    data : np.ndarray
        Cutout pixels in native image counts (NaN off the footprint).
        Multiply by cf for microjanskys.
    wcs : astropy.wcs.WCS
        Cutout WCS.
    header : astropy.io.fits.Header
        Parent image header (calibration keywords, seeing keywords).
    cx, cy : float
        Target position in stamp pixels.
    pixscale : float
        Pixel scale (arcsec/px).
    cf : float
        Counts -> microjansky calibration factor.
    rr : np.ndarray
        Radius map about the target (arcsec).
    nodata : np.ndarray
        Boolean map of unusable pixels.
    sigma : float
        Global clipped pixel scatter (counts) of the outer stamp.
    farfield_sb : float or None
        Robust far-field level (uJy/arcsec^2), None when the stamp has
        too little far field to measure.
    invvar : np.ndarray or None
        Inverse-variance cutout on the same geometry, when the archive
        serves one.
    """
    data: np.ndarray
    wcs: WCS
    header: fits.Header
    cx: float
    cy: float
    pixscale: float
    cf: float
    rr: np.ndarray
    nodata: np.ndarray
    sigma: float
    farfield_sb: float | None
    invvar: np.ndarray | None = None

    @property
    def good(self) -> np.ndarray:
        """Boolean map of usable pixels."""
        return ~self.nodata

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def sb(self) -> float:
        """Counts -> uJy/arcsec^2 conversion for surface brightness."""
        return self.cf / self.pixscale ** 2


def load_stamp(
        path: str,
        calib: str,
        coord: SkyCoord,
        *,
        cutout_half_arcsec: float,
        invvar_path: str | None = None,
) -> Stamp:
    """Cut and characterize one band's stamp at the target position.

    Parameters
    ----------
    path : str
        Science FITS image.
    calib : str
        Calibration key for calibrate.calib_factor ('nmgy', 'photzp', ...).
    coord : SkyCoord
        Target position; the stamp centers here.
    cutout_half_arcsec : float
        Stamp half-size (arcsec).
    invvar_path : str, optional
        Inverse-variance map, cut on the same geometry for the error model.

    Returns
    -------
    stamp : Stamp
        The prepared stamp.
    """
    image, image_wcs, header = load_image(path)
    cf = calib_factor(calib, header)
    pixscale = pixel_scale_arcsec(image_wcs)
    px, py = [float(v) for v in image_wcs.world_to_pixel(coord)]
    size = 2 * int(round(cutout_half_arcsec / pixscale)) + 1
    cut = Cutout2D(image, (px, py), size, wcs=image_wcs,
                   mode='partial', fill_value=np.nan)
    data = cut.data.astype(float)
    stamp_wcs = cut.wcs
    cx, cy = [float(v) for v in stamp_wcs.world_to_pixel(coord)]
    rr = radii_arcsec(data.shape, cx, cy, pixscale)

    # Unusable pixels: off-footprint fill, exact archive zeros, then --
    # once the outer level and scatter are known -- deeply negative
    # outliers (dead pixels, cosmic-ray holes).
    nodata = ~np.isfinite(data) | (data == 0.0)
    outer = ~nodata & (rr > recipe.BG_RMIN_AS)
    if outer.sum() < 100:
        outer = ~nodata
    level, _, sigma = sigma_clipped_stats(data[outer], sigma=3.0, maxiters=6)
    nodata |= (data - level) < -10.0 * max(sigma, 1e-30)

    # Far-field witness: the stamp's own robust zero, measured where
    # target and halo light are weakest. Recorded per band and never fed
    # back into the fit; a strongly positive value flags archive-level
    # sky structure around the field.
    farfield_sb = None
    far = ~nodata & (rr > recipe.FARFIELD_RMIN_AS)
    if far.sum() >= recipe.FARFIELD_MIN_PX:
        _, far_level, _ = sigma_clipped_stats(data[far], sigma=3.0, maxiters=6)
        farfield_sb = float(far_level) * cf / pixscale ** 2

    invvar = None
    if invvar_path is not None:
        invvar_image, _, _ = load_image(invvar_path)
        invvar = Cutout2D(invvar_image, (px, py), size,
                          mode='partial', fill_value=0.0).data.astype(float)

    return Stamp(data=data, wcs=stamp_wcs, header=header, cx=cx, cy=cy,
                 pixscale=pixscale, cf=cf, rr=rr, nodata=nodata,
                 sigma=float(sigma), farfield_sb=farfield_sb, invvar=invvar)


# ------------------------------------
# Coverage gate
# ------------------------------------
def check_coverage(
        stamp: Stamp,
        *,
        aperture_arcsec: float,
        seeing_arcsec: float,
) -> float:
    """Data-sufficiency gate for the science aperture.

    Missing data inside the aperture is fill-corrected downstream, but
    only up to a point: past COVERAGE_MIN there is no honest profile to
    fill from, and the band demotes rather than ship a silently biased
    flux. The seeing-scale core is gated absolutely -- its peak carries
    an outsized flux share that no fill can reconstruct.

    Parameters
    ----------
    stamp : Stamp
        The prepared stamp.
    aperture_arcsec : float
        Science aperture radius.
    seeing_arcsec : float
        Band PSF FWHM; sets the protected core radius.

    Returns
    -------
    coverage : float
        Fraction of aperture pixels with real data.

    Raises
    ------
    ApertureCoverageError
        When the aperture cannot be honestly measured.
    """
    in_aperture = stamp.rr < aperture_arcsec
    n_aper = int(in_aperture.sum())
    coverage = 1.0 - float((stamp.nodata & in_aperture).sum()) / max(n_aper, 1)
    if coverage < recipe.COVERAGE_MIN:
        raise ApertureCoverageError(
            f"aperture coverage {coverage:.2f} < {recipe.COVERAGE_MIN:g} "
            f"(off footprint / blank pixels)", coverage)
    core_radius = max(3.0, 2.0 * seeing_arcsec)
    if (stamp.nodata & (stamp.rr < core_radius)).any():
        raise ApertureCoverageError(
            f"blank pixels inside the {core_radius:g}\" core (aperture "
            f"coverage {coverage:.2f}) -- no fill can reconstruct a "
            f"clipped peak", coverage)
    return coverage
