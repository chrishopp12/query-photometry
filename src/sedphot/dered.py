"""
dered.py

Milky Way Dereddening (opt-in)
---------------------------------------------------------
Applies Galactic extinction corrections to an assembled photometry table
when --dered is requested. Default package behavior is AS-MEASURED; this
module is the one place the correction happens, recorded per row.

Three tiers, per row:
    1. Native: the row already carries mw_transmission from its provider
       (Legacy Tractor, SDSS extinction_*, GALEX GUVcat E(B-V)) -- divide.
    2. Coefficient: the band has an A_band/E(B-V) coefficient in
       EXT_COEFF and E(B-V)_SFD is fetched once per target from IRSA --
       compute the transmission, fill the column, divide.
    3. Neither: the row is left as-measured with a printed warning
       (dered_applied stays False) -- never a crash.

Requirements:
    numpy, pandas, astropy; astroquery (IRSA dust, tier 2 only)

Notes:
    Coefficients are Schlafly & Finkbeiner 2011 (R_V = 3.1) for the optical
    sets, WISE per Fitzpatrick-based IR values, GALEX per Bianchi+2017.
    J-PLUS, CFHT, and HST bands currently have no entries and fall to
    tier 3 -- add coefficients here when a dereddened fit needs them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .units import ujy_to_mag


# ------------------------------------
# A_band / E(B-V) coefficients (tier 2)
# ------------------------------------
EXT_COEFF = {
    # SDSS ugriz (SF11 Table 6, R_V=3.1)
    'SDSS_u': 4.239, 'SDSS_g': 3.303, 'SDSS_r': 2.285, 'SDSS_i': 1.698, 'SDSS_z': 1.263,
    # Pan-STARRS grizy (SF11)
    'PS1_g': 3.172, 'PS1_r': 2.271, 'PS1_i': 1.682, 'PS1_z': 1.322, 'PS1_y': 1.087,
    # GALEX (Bianchi+2017) -- backup; GUVcat rows carry native transmission
    'GALEX_FUV': 8.06, 'GALEX_NUV': 7.95,
    # WISE (near-zero in the mid-IR)
    'WISE_W1': 0.189, 'WISE_W2': 0.146, 'WISE_W3': 0.0, 'WISE_W4': 0.0,
    # 2MASS (SF11)
    '2MASS_J': 0.723, '2MASS_H': 0.460, '2MASS_Ks': 0.310,
}


# ------------------------------------
# E(B-V) lookup (one query per target)
# ------------------------------------
def fetch_ebv_sfd(coord: SkyCoord) -> float:
    """Schlafly & Finkbeiner-recalibrated SFD E(B-V) at the target position.

    Queries the IRSA dust service once; returns NaN on failure (tier-2
    rows then fall through to tier 3).
    """
    try:
        from astroquery.ipac.irsa.irsa_dust import IrsaDust
        table = IrsaDust.get_query_table(coord, section='ebv')
        return float(table['ext SandF mean'][0])
    except Exception as e:
        print(f"  [dered] IRSA dust query failed: {e}")
        return np.nan


# ------------------------------------
# Application
# ------------------------------------
def apply_dereddening(df: pd.DataFrame, coord: SkyCoord) -> tuple[pd.DataFrame, dict]:
    """Apply MW dereddening to a schema table, returning a corrected copy.

    Parameters
    ----------
    df : pd.DataFrame
        Assembled schema table (as-measured).
    coord : SkyCoord
        Target position, for the single E(B-V) lookup.

    Returns
    -------
    dered_df : pd.DataFrame
        Copy with corrected flux/err/mag on every row that had a
        transmission (native or coefficient); dered_applied marks which.
    meta : dict
        Sidecar record: E(B-V) used, coefficient source, per-tier band lists.
    """
    out = df.copy()
    ebv = np.nan
    needs_ebv = [
        band for band, trans in zip(out['band'], out['mw_transmission'])
        if not np.isfinite(trans) and band in EXT_COEFF
    ]
    if needs_ebv:
        ebv = fetch_ebv_sfd(coord)

    native, coeff, skipped = [], [], []
    for i in out.index:
        band = out.at[i, 'band']
        trans = out.at[i, 'mw_transmission']

        if not np.isfinite(trans):
            if band in EXT_COEFF and np.isfinite(ebv):
                trans = 10 ** (-EXT_COEFF[band] * ebv / 2.5)
                out.at[i, 'mw_transmission'] = round(float(trans), 6)
                coeff.append(band)
            else:
                skipped.append(band)
                continue
        else:
            native.append(band)

        out.at[i, 'flux_uJy'] = round(float(out.at[i, 'flux_uJy'] / trans), 6)
        out.at[i, 'flux_err_uJy'] = round(float(out.at[i, 'flux_err_uJy'] / trans), 6)
        flux = out.at[i, 'flux_uJy']
        out.at[i, 'mag_AB'] = round(ujy_to_mag(flux), 4) if flux > 0 else np.nan
        out.at[i, 'dered_applied'] = True

    if skipped:
        print(f"  [dered] left as-measured (no transmission or coefficient): "
              f"{', '.join(skipped)}")

    meta = {
        'ebv_sfd': None if not np.isfinite(ebv) else round(float(ebv), 5),
        'coefficient_source': 'SF11 optical; Bianchi+2017 GALEX; WISE IR values',
        'bands_native_transmission': native,
        'bands_coefficient_transmission': coeff,
        'bands_left_as_measured': skipped,
    }
    return out, meta
