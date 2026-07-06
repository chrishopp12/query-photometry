"""
test_units.py

Unit-Conversion Invariants
---------------------------------------------------------
AB zeropoint identity, magnitude/flux round trips, error propagation, and
NaN guards for sedphot.units.
"""
from __future__ import annotations

import numpy as np
import pytest

from sedphot.units import (
    flux_err_to_mag_err,
    mag_err_to_flux_err,
    mag_to_ujy,
    nanomaggy_to_ujy,
    ujy_to_mag,
)


def test_ab_zeropoint():
    # m = 23.9 is 1 uJy by definition of the uJy AB zeropoint.
    assert mag_to_ujy(23.9) == pytest.approx(1.0)
    assert ujy_to_mag(1.0) == pytest.approx(23.9)


def test_nanomaggy():
    assert nanomaggy_to_ujy(1.0) == pytest.approx(3.631)
    # 22.5 - 2.5log10(f_nmgy) == 23.9 - 2.5log10(f_uJy), to the precision of
    # the rounded 3.631 convention (exact factor is 10^0.56 = 3.63078...).
    assert ujy_to_mag(nanomaggy_to_ujy(1.0)) == pytest.approx(22.5, abs=1e-3)


def test_mag_flux_round_trip():
    for mag in (12.0, 18.5, 23.9, 27.3):
        assert ujy_to_mag(mag_to_ujy(mag)) == pytest.approx(mag)


def test_error_propagation_round_trip():
    mag, mag_err = 20.0, 0.05
    flux = mag_to_ujy(mag)
    flux_err = mag_err_to_flux_err(mag, mag_err)
    assert flux_err_to_mag_err(flux, flux_err) == pytest.approx(mag_err)


def test_nan_guards():
    assert np.isnan(mag_to_ujy(np.nan))
    assert np.isnan(ujy_to_mag(-5.0))
    assert np.isnan(ujy_to_mag(0.0))
    assert np.isnan(mag_err_to_flux_err(np.nan, 0.1))
    assert np.isnan(flux_err_to_mag_err(-1.0, 0.1))
