"""
bands.py

Band Effective Wavelengths
---------------------------------------------------------
Effective (pivot-ish) wavelengths per band label, for QA figure coloring and
the combined SED plot ONLY -- these numbers never enter a measurement. SED
fitters own their bandpass physics (sedpy curves in the sed_fitting package);
duplicating full filter curves here would create a second authority.

HST filters are parsed from the name instead of tabulated: the flight-filter
convention encodes the wavelength (F475W -> 0.475 um, F160W -> 1.60 um).

Requirements:
    (stdlib only)

Notes:
    Milky Way extinction coefficients for --dered live in dered.py
    (EXT_COEFF), keyed by these same band labels.
"""
from __future__ import annotations

import re


# ------------------------------------
# Effective wavelengths (micron) per band label
# ------------------------------------
WAVE_UM = {
    # GALEX
    'GALEX_FUV': 0.1528, 'GALEX_NUV': 0.2310,
    # SDSS
    'SDSS_u': 0.355, 'SDSS_g': 0.475, 'SDSS_r': 0.622, 'SDSS_i': 0.763, 'SDSS_z': 0.905,
    # CFHT MegaCam
    'CFHT_u': 0.355, 'CFHT_g': 0.475, 'CFHT_r': 0.640, 'CFHT_i': 0.776, 'CFHT_z': 0.925,
    # Legacy Surveys (BASS/MzLS/DECam) + unWISE forced photometry
    'Legacy_g': 0.475, 'Legacy_r': 0.625, 'Legacy_i': 0.755, 'Legacy_z': 0.920,
    'Legacy_W1': 3.368, 'Legacy_W2': 4.618, 'Legacy_W3': 12.082, 'Legacy_W4': 22.194,
    # Pan-STARRS
    'PS1_g': 0.481, 'PS1_r': 0.617, 'PS1_i': 0.752, 'PS1_z': 0.866, 'PS1_y': 0.962,
    # WISE (AllWISE)
    'WISE_W1': 3.368, 'WISE_W2': 4.618, 'WISE_W3': 12.082, 'WISE_W4': 22.194,
    # 2MASS
    '2MASS_J': 1.235, '2MASS_H': 1.662, '2MASS_Ks': 2.159,
    # J-PLUS DR3 (12 filters)
    'JPLUS_uJAVA': 0.3485, 'JPLUS_J0378': 0.3785, 'JPLUS_J0395': 0.3950,
    'JPLUS_J0410': 0.4100, 'JPLUS_J0430': 0.4300, 'JPLUS_gSDSS': 0.4803,
    'JPLUS_J0515': 0.5150, 'JPLUS_rSDSS': 0.6254, 'JPLUS_J0660': 0.6600,
    'JPLUS_iSDSS': 0.7668, 'JPLUS_J0861': 0.8610, 'JPLUS_zSDSS': 0.9114,
}

_HST_FILTER = re.compile(r"F(\d{3,4})(W|LP|M|N)$", re.IGNORECASE)


# ------------------------------------
# Lookup
# ------------------------------------
def wave_um(band: str) -> float:
    """Effective wavelength in micron for a band label; NaN when unknown.

    Parameters
    ----------
    band : str
        Schema band label (<Instrument>_<filter>).

    Returns
    -------
    wave : float
        Effective wavelength in micron, or NaN if the band is not tabulated
        and not an HST filter name.
    """
    if band in WAVE_UM:
        return WAVE_UM[band]
    # HST flight-filter names encode the wavelength: 3 digits are nm/10
    # (F475W -> 0.475 um) except the IR channel where <200 means um/100
    # (F160W -> 1.60 um); 4 digits are nm (WFPC2 F1042M -> 1.042 um).
    match = _HST_FILTER.search(band.rsplit("_", 1)[-1])
    if match:
        number = int(match.group(1))
        if len(match.group(1)) == 4:
            return number / 1000.0
        return number / 100.0 if number < 200 else number / 1000.0
    return float("nan")
