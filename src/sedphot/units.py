"""
units.py

Photometric Unit Conversions
---------------------------------------------------------
Pure conversion helpers: nanomaggies and AB magnitudes to microjanskys and
back, with error propagation. All magnitudes are AB; the AB zeropoint in
microjanskys is 23.9 (3631 Jy). These conversions are the package's numeric
core; every provider and the measurement engine share them.

Requirements:
    numpy

Notes:
    Non-finite inputs return NaN rather than raising, so providers can pass
    catalog columns through without per-value guards.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------
# Constants
# ------------------------------------
# 1 nanomaggy = 3.631 uJy (AB system, zeropoint 3631 Jy).
NANOMAGGY_TO_UJY = 3.631

# AB magnitude zeropoint expressed in microjanskys: m = 23.9 - 2.5 log10(f_uJy).
AB_ZP_UJY = 23.9


# ------------------------------------
# Conversions
# ------------------------------------
def nanomaggy_to_ujy(flux_nm: float) -> float:
    """Convert Legacy Survey nanomaggies to microjanskys."""
    return flux_nm * NANOMAGGY_TO_UJY


def mag_to_ujy(mag: float) -> float:
    """Convert AB magnitude to microjanskys. Returns NaN for non-finite input."""
    if not np.isfinite(mag):
        return np.nan
    return 10 ** ((AB_ZP_UJY - mag) / 2.5)


def mag_err_to_flux_err(mag: float, mag_err: float) -> float:
    """Convert magnitude uncertainty to flux uncertainty in microjanskys.

    Derived from error propagation on f = 10^((23.9 - m)/2.5):
        sigma_f = f * (ln10 / 2.5) * sigma_m
    """
    if not np.isfinite(mag) or not np.isfinite(mag_err):
        return np.nan
    flux = mag_to_ujy(mag)
    return flux * (np.log(10) / 2.5) * mag_err


def flux_err_to_mag_err(flux: float, flux_err: float) -> float:
    """Convert flux uncertainty (any linear unit) to magnitude uncertainty."""
    if not np.isfinite(flux) or flux <= 0 or not np.isfinite(flux_err):
        return np.nan
    return (2.5 / np.log(10)) * (flux_err / flux)


def ujy_to_mag(flux_ujy: float) -> float:
    """Convert microjanskys to AB magnitude. Non-positive flux returns NaN."""
    if not np.isfinite(flux_ujy) or flux_ujy <= 0:
        return np.nan
    return AB_ZP_UJY - 2.5 * np.log10(flux_ujy)
