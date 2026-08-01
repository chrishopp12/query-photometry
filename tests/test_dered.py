"""MW dereddening: the three tiers, and the E(B-V) flavor tier 2 pairs with.

EXT_COEFF is normalized per uncorrected SFD E(B-V), so the coefficient
path must read IRSA's SFD column; the recalibrated SandF column would
apply the ~0.86 twice. No network.
"""
from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from sedphot import dered
from sedphot.schema import ALL_COLS, make_row

COORD = SkyCoord(217.16, 56.86, unit='deg')

# IRSA serves both flavors of the same quantity; the ratio is the
# Schlafly & Finkbeiner recalibration.
EBV_SFD = 0.0134
EBV_SANDF = 0.0115


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Never sleep through the transport backoff."""
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)


def row(band, flux=100.0, err=2.0, transmission=np.nan):
    """One schema row, as-measured."""
    return make_row(band=band, flux_ujy=flux, flux_err_ujy=err,
                    mag=20.0, mag_err=0.02,
                    target_ra=float(COORD.ra.deg),
                    target_dec=float(COORD.dec.deg),
                    match_ra=float(COORD.ra.deg),
                    match_dec=float(COORD.dec.deg),
                    sep_arcsec=0.1, flags='', source='test',
                    mw_transmission=transmission)


def frame(*rows):
    return pd.DataFrame(list(rows), columns=ALL_COLS)


def serve_irsa(monkeypatch, table=None, boom=None):
    """Stand in for the astroquery module fetch_ebv_sfd imports at call time."""
    def get_query_table(coord, section=None):
        if boom is not None:
            raise boom
        return table
    monkeypatch.setitem(
        sys.modules, 'astroquery.ipac.irsa.irsa_dust',
        SimpleNamespace(IrsaDust=SimpleNamespace(
            get_query_table=get_query_table)))


# ------------------------------------
# The E(B-V) flavor
# ------------------------------------

def test_the_coefficient_path_reads_the_uncorrected_sfd_column(monkeypatch):
    """SF11 coefficients carry the ~0.86 recalibration already.

    Pairing them with the recalibrated SandF column applies it twice and
    leaves every tier-2 A_band low by that factor, so the flavor served
    here is the whole contract of the function.
    """
    serve_irsa(monkeypatch, Table({'ext SFD mean': [EBV_SFD],
                                   'ext SandF mean': [EBV_SANDF]}))
    assert dered.fetch_ebv_sfd(COORD) == pytest.approx(EBV_SFD)


def test_the_tier_two_scale_matches_what_the_surveys_publish(monkeypatch):
    """Tier 2 must land on tier 1's scale, since one table carries both.

    A survey ships A_band = SF11 coefficient x SFD E(B-V) in its own
    transmission column. Reproducing that from the coefficient path is
    what makes a native row and a computed row comparable.
    """
    serve_irsa(monkeypatch, Table({'ext SFD mean': [EBV_SFD],
                                   'ext SandF mean': [EBV_SANDF]}))
    out, meta = dered.apply_dereddening(frame(row('SDSS_u')), COORD)

    a_band = -2.5 * np.log10(out.at[0, 'mw_transmission'])
    assert a_band / EBV_SFD == pytest.approx(dered.EXT_COEFF['SDSS_u'],
                                             rel=1e-3)
    # The same figure against the recalibrated flavor would be low by
    # that factor -- the failure this pins.
    assert a_band / EBV_SANDF != pytest.approx(dered.EXT_COEFF['SDSS_u'],
                                               rel=1e-2)
    assert meta['ebv_sfd'] == pytest.approx(EBV_SFD)


# ------------------------------------
# The three tiers
# ------------------------------------

def test_a_coefficient_row_is_corrected_and_marked(monkeypatch):
    serve_irsa(monkeypatch, Table({'ext SFD mean': [EBV_SFD],
                                   'ext SandF mean': [EBV_SANDF]}))
    out, meta = dered.apply_dereddening(frame(row('PS1_g')), COORD)

    trans = 10 ** (-dered.EXT_COEFF['PS1_g'] * EBV_SFD / 2.5)
    assert out.at[0, 'mw_transmission'] == pytest.approx(trans, rel=1e-5)
    assert out.at[0, 'flux_uJy'] == pytest.approx(100.0 / trans, rel=1e-5)
    assert bool(out.at[0, 'dered_applied']) is True
    assert meta['bands_coefficient_transmission'] == ['PS1_g']


def test_a_native_row_keeps_its_providers_transmission(monkeypatch):
    """Tier 1 owns the value; the coefficient table must not overwrite it."""
    called = []
    monkeypatch.setattr(dered, 'fetch_ebv_sfd',
                        lambda coord: called.append(1) or EBV_SFD)
    out, meta = dered.apply_dereddening(
        frame(row('SDSS_u', transmission=0.9)), COORD)

    assert out.at[0, 'mw_transmission'] == 0.9
    assert out.at[0, 'flux_uJy'] == pytest.approx(100.0 / 0.9)
    assert meta['bands_native_transmission'] == ['SDSS_u']
    # A band that already carries transmission is not worth a query.
    assert called == []


def test_a_band_with_no_coefficient_is_left_as_measured(monkeypatch):
    """Tier 3 is a warning, never a crash and never a guess."""
    monkeypatch.setattr(dered, 'fetch_ebv_sfd', lambda coord: EBV_SFD)
    out, meta = dered.apply_dereddening(frame(row('JPLUS_J0378')), COORD)

    assert out.at[0, 'flux_uJy'] == 100.0
    assert bool(out.at[0, 'dered_applied']) is False
    assert not np.isfinite(out.at[0, 'mw_transmission'])
    assert meta['bands_left_as_measured'] == ['JPLUS_J0378']


def test_a_failed_lookup_demotes_tier_two_to_as_measured(monkeypatch):
    """An IRSA outage costs the correction, not the run."""
    serve_irsa(monkeypatch, boom=RuntimeError("service down"))
    out, meta = dered.apply_dereddening(frame(row('PS1_g')), COORD)

    assert out.at[0, 'flux_uJy'] == 100.0
    assert bool(out.at[0, 'dered_applied']) is False
    assert meta['ebv_sfd'] is None
    assert meta['bands_left_as_measured'] == ['PS1_g']


# ------------------------------------
# What the correction does to a row
# ------------------------------------

def test_the_correction_is_multiplicative_so_snr_is_unchanged(monkeypatch):
    """Flux and error scale together; only the scale moves, not the weight."""
    monkeypatch.setattr(dered, 'fetch_ebv_sfd', lambda coord: EBV_SFD)
    out, _ = dered.apply_dereddening(
        frame(row('PS1_g', flux=100.0, err=2.0)), COORD)

    assert (out.at[0, 'flux_err_uJy'] / out.at[0, 'flux_uJy']
            == pytest.approx(2.0 / 100.0, rel=1e-4))
    assert out.at[0, 'mag_AB'] < 20.0        # brighter once corrected


def test_a_non_detection_survives_without_a_magnitude(monkeypatch):
    """Negative catalog fluxes are legitimate; they must not raise."""
    monkeypatch.setattr(dered, 'fetch_ebv_sfd', lambda coord: EBV_SFD)
    out, _ = dered.apply_dereddening(
        frame(row('PS1_g', flux=-5.0, err=2.0)), COORD)

    assert out.at[0, 'flux_uJy'] < 0
    assert not np.isfinite(out.at[0, 'mag_AB'])
    assert bool(out.at[0, 'dered_applied']) is True


def test_mixed_tiers_are_reported_separately(monkeypatch):
    """One table can hold all three; the record has to say which is which."""
    monkeypatch.setattr(dered, 'fetch_ebv_sfd', lambda coord: EBV_SFD)
    out, meta = dered.apply_dereddening(
        frame(row('Legacy_g', transmission=0.96),
              row('PS1_g'),
              row('JPLUS_J0378')), COORD)

    assert meta['bands_native_transmission'] == ['Legacy_g']
    assert meta['bands_coefficient_transmission'] == ['PS1_g']
    assert meta['bands_left_as_measured'] == ['JPLUS_J0378']
    assert [bool(v) for v in out['dered_applied']] == [True, True, False]


def test_the_input_frame_is_not_mutated(monkeypatch):
    """apply_dereddening returns a corrected copy; as-measured stays put."""
    monkeypatch.setattr(dered, 'fetch_ebv_sfd', lambda coord: EBV_SFD)
    original = frame(row('PS1_g'))
    out, _ = dered.apply_dereddening(original, COORD)

    assert original.at[0, 'flux_uJy'] == 100.0
    assert bool(original.at[0, 'dered_applied']) is False
    assert out.at[0, 'flux_uJy'] != 100.0
