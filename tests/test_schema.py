"""
test_schema.py

Output Schema Contract
---------------------------------------------------------
Column order stability, NaN guards, defaults, and frame assembly for
sedphot.schema.
"""
from __future__ import annotations

import numpy as np

from sedphot.schema import ALL_COLS, BASE_COLS, EXTRA_COLS, make_row, rows_to_frame


def _row(**overrides):
    base = dict(
        band='Legacy_g', flux_ujy=679.3, flux_err_ujy=1.2, mag=16.82, mag_err=0.002,
        target_ra=216.988087, target_dec=56.9878, match_ra=216.98809,
        match_dec=56.98781, sep_arcsec=0.02, flags='', source='Legacy_DR9',
    )
    base.update(overrides)
    return make_row(**base)


def test_base_cols_are_frozen_contract():
    # The first twelve columns are the legacy retrieval-script column set in
    # their original order -- existing consumers key on this. Never reorder;
    # append only.
    assert BASE_COLS == [
        'band', 'flux_uJy', 'flux_err_uJy', 'mag_AB', 'mag_err',
        'target_ra', 'target_dec', 'match_ra', 'match_dec',
        'sep_arcsec', 'flags', 'source',
    ]
    assert ALL_COLS[:len(BASE_COLS)] == BASE_COLS
    assert ALL_COLS[len(BASE_COLS):] == EXTRA_COLS


def test_make_row_covers_schema():
    row = _row()
    assert list(row) == ALL_COLS


def test_make_row_nan_guards():
    row = _row(flux_ujy=np.nan, mag=np.nan)
    assert np.isnan(row['flux_uJy'])
    assert np.isnan(row['mag_AB'])
    # Negative fluxes are preserved, not NaN-ed (they are non-detections).
    row = _row(flux_ujy=-3.2)
    assert row['flux_uJy'] == -3.2


def test_make_row_defaults():
    row = _row()
    assert row['dered_applied'] is False
    assert np.isnan(row['mw_transmission'])
    assert len(row['retrieved']) == 10  # ISO date


def test_rows_to_frame_empty_and_order():
    assert list(rows_to_frame([]).columns) == ALL_COLS
    frame = rows_to_frame([_row(), _row(band='Legacy_r')])
    assert list(frame.columns) == ALL_COLS
    assert len(frame) == 2
