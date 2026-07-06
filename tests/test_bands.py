"""
test_bands.py

Band Wavelength Lookup
---------------------------------------------------------
Tabulated effective wavelengths and the HST filter-name parser in
sedphot.bands.
"""
from __future__ import annotations

import math

import pytest

from sedphot.bands import wave_um


def test_tabulated_bands():
    assert wave_um('Legacy_g') == pytest.approx(0.475)
    assert wave_um('GALEX_FUV') == pytest.approx(0.1528)
    assert wave_um('JPLUS_J0430') == pytest.approx(0.430)


def test_hst_filter_parsing():
    assert wave_um('HST_F475W') == pytest.approx(0.475)
    assert wave_um('HST_F850LP') == pytest.approx(0.850)
    assert wave_um('HST_F160W') == pytest.approx(1.60)   # WFC3/IR: <200 -> um/100
    assert wave_um('HST_F098M') == pytest.approx(0.98)
    assert wave_um('HST_F1042M') == pytest.approx(1.042)  # WFPC2 4-digit


def test_unknown_band_is_nan():
    assert math.isnan(wave_um('NOT_A_BAND'))
