"""
calibrate.py

Image Loading and Flux Calibration
---------------------------------------------------------
FITS loading (plain or .fz) and the per-instrument calibration factors that
put every image on the common microjansky scale:

    'nmgy'    Legacy bricks/cutouts, SDSS frames:  uJy = ADU_nmgy * 3.631
    'photzp'  CFHT MegaPipe (PHOTZP header, AB):   uJy = ADU * 10^(-(ZP-23.9)/2.5)
    'ps1'     PanSTARRS stacks (ZP=25 for DN/s):   uJy = ADU * 10^((23.9-zp_dn)/2.5)
    'hst'     drizzled e/s with PHOTFLAM/PHOTPLAM: uJy = ADU * 10^((23.9-zp_ab)/2.5)

Ported from uniform_phot.py (_load, _calib_factor) and
hst_aperture_photometry.py (the AB zeropoint chain).

Requirements:
    numpy, astropy
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from ..units import NANOMAGGY_TO_UJY, AB_ZP_UJY


# ------------------------------------
# Loading
# ------------------------------------
def load_image(path: str) -> tuple[np.ndarray, WCS, fits.Header]:
    """Load a science image: (data, wcs, header).

    Uses the first HDU carrying a 2D image -- covers plain FITS (primary),
    fpack .fz (extension 1), and MEF cutouts from SODA services.
    """
    with fits.open(path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and getattr(hdu.data, "ndim", 0) == 2:
                data = hdu.data.astype(float)
                header = hdu.header
                return data, WCS(header), header
    raise ValueError(f"no 2D image HDU in {path}")


def pixel_scale_arcsec(wcs: WCS) -> float:
    """Pixel scale in arcsec/pixel from the WCS projection plane."""
    return float(np.abs(wcs.proj_plane_pixel_scales()[0].to("arcsec").value))


# ------------------------------------
# Calibration
# ------------------------------------
def hst_ab_zeropoint(photflam: float, photplam: float) -> float:
    """AB zeropoint for a drizzled HST image from PHOTFLAM/PHOTPLAM."""
    return -2.5 * np.log10(photflam) - 5.0 * np.log10(photplam) - 2.408


def calib_factor(calib: str, header: fits.Header) -> float:
    """ADU -> uJy factor for one image.

    Parameters
    ----------
    calib : str
        Calibration key: 'nmgy' | 'photzp' | 'ps1' | 'hst'.
    header : fits.Header
        Science-image header (supplies PHOTZP / EXPTIME / PHOTFLAM as
        needed by the key).

    Returns
    -------
    factor : float
        Multiply image counts by this to get microjanskys.
    """
    if calib == "nmgy":                                     # Legacy, SDSS frames
        return NANOMAGGY_TO_UJY
    if calib == "photzp":                                   # CFHT MegaPipe (AB)
        return 10 ** (-(header["PHOTZP"] - AB_ZP_UJY) / 2.5)
    if calib == "ps1":                                      # PS1 stack: ZP=25 for DN/s
        zp_dn = 25.0 + 2.5 * np.log10(header["EXPTIME"])
        return 10 ** ((AB_ZP_UJY - zp_dn) / 2.5)
    if calib == "hst":                                      # drizzled e/s
        zp_ab = hst_ab_zeropoint(header["PHOTFLAM"], header["PHOTPLAM"])
        return 10 ** ((AB_ZP_UJY - zp_ab) / 2.5)
    raise ValueError(f"unknown calib key {calib!r}")
