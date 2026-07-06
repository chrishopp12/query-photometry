"""
schema.py

Output Table Schema and Row Builder
---------------------------------------------------------
The one table contract every catalog provider and measurement writes. The
first twelve columns are the legacy retrieval-script column set, kept in
order so existing consumers (plot_hst_image overlays, analysis notebooks)
read new tables unchanged; the appended columns carry retrieval provenance.

Column conventions:
    band            <Instrument>_<filter>  (Legacy_g, PS1_g, GALEX_FUV, HST_F475W)
    flux_uJy        flux in microjanskys; negative values are legitimate
                    non-detections and are preserved
    flux_err_uJy    statistical error only -- floors/inflation belong to the
                    SED fitter, never to this table
    mag_AB, mag_err AB magnitude view of the same measurement (NaN if flux <= 0)
    source          measurement provenance string (stable API: overlay styles
                    and downstream tools key on it)
    retrieved       ISO date the value was pulled from the archive
    mw_transmission per-band Milky Way transmission where the provider supplies
                    it (Legacy Tractor natively); NaN when unknown
    dered_applied   True when flux/mag have been corrected for MW extinction

Requirements:
    numpy, pandas

Notes:
    Missing values are NaN (empty CSV fields). make_row NaN-guards every
    numeric so providers can pass catalog values straight through.
"""
from __future__ import annotations

import datetime

import numpy as np
import pandas as pd


# ------------------------------------
# Column schema
# ------------------------------------
# Legacy retrieval-script column set -- order is a compatibility contract,
# do not reorder.
BASE_COLS = [
    'band',
    'flux_uJy',
    'flux_err_uJy',
    'mag_AB',
    'mag_err',
    'target_ra',
    'target_dec',
    'match_ra',
    'match_dec',
    'sep_arcsec',
    'flags',
    'source',
]

# Appended by sedphot (append-only; new columns go at the end).
EXTRA_COLS = [
    'retrieved',
    'mw_transmission',
    'dered_applied',
]

ALL_COLS = BASE_COLS + EXTRA_COLS


# ------------------------------------
# Row builder
# ------------------------------------
def make_row(
        band: str,
        flux_ujy: float,
        flux_err_ujy: float,
        mag: float,
        mag_err: float,
        target_ra: float,
        target_dec: float,
        match_ra: float,
        match_dec: float,
        sep_arcsec: float,
        flags,
        source: str,
        *,
        retrieved: str | None = None,
        mw_transmission: float = np.nan,
        dered_applied: bool = False,
) -> dict:
    """Build a single output row dict with all schema columns.

    Parameters
    ----------
    band : str
        Band label, <Instrument>_<filter>.
    flux_ujy, flux_err_ujy : float
        Flux and statistical error in microjanskys.
    mag, mag_err : float
        AB magnitude view of the same measurement.
    target_ra, target_dec : float
        Requested position (deg).
    match_ra, match_dec : float
        Matched catalog source position (deg).
    sep_arcsec : float
        Target-to-match separation.
    flags
        Passed through from the source catalog ('' when none).
    source : str
        Measurement provenance string.
    retrieved : str, optional
        ISO retrieval date. [default: today]
    mw_transmission : float
        Per-band MW transmission if the provider supplies it. [default: NaN]
    dered_applied : bool
        Whether MW dereddening has been applied to flux/mag. [default: False]

    Returns
    -------
    row : dict
        One table row keyed by ALL_COLS.
    """
    return {
        'band':            band,
        'flux_uJy':        round(float(flux_ujy),     6) if np.isfinite(flux_ujy)     else np.nan,
        'flux_err_uJy':    round(float(flux_err_ujy), 6) if np.isfinite(flux_err_ujy) else np.nan,
        'mag_AB':          round(float(mag),           4) if np.isfinite(mag)          else np.nan,
        'mag_err':         round(float(mag_err),       4) if np.isfinite(mag_err)      else np.nan,
        'target_ra':       round(float(target_ra),     8),
        'target_dec':      round(float(target_dec),    8),
        'match_ra':        round(float(match_ra),      8),
        'match_dec':       round(float(match_dec),     8),
        'sep_arcsec':      round(float(sep_arcsec),    4),
        'flags':           flags,
        'source':          source,
        'retrieved':       retrieved or datetime.date.today().isoformat(),
        'mw_transmission': round(float(mw_transmission), 6) if np.isfinite(mw_transmission) else np.nan,
        'dered_applied':   bool(dered_applied),
    }


def rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    """Assemble row dicts into a DataFrame with the full column schema."""
    if not rows:
        return pd.DataFrame(columns=ALL_COLS)
    return pd.DataFrame(rows, columns=ALL_COLS)
