"""
jplus.py

J-PLUS DR3 Catalog Provider
---------------------------------------------------------
Closest-source 12-band photometry from the J-PLUS DR3 dual-mode catalog via
the CEFCA TAP service, using the PSF-corrected magnitudes (MAG_PSFCOR) --
the catalog's PSF-homogenized magnitude set.

Service facts:
  - TAP sync endpoint: TAP_SYNC_URL below; FORMAT=csv. The astroquery
    TapPlus wrappers choke on this service's table metadata and on its
    error responses, so the module POSTs the sync endpoint directly.
  - Table jplus.MagABDualObj; positions are alpha_j2000/delta_j2000.
  - Magnitude columns are 12-element arrays indexed by the CEFCA filter
    macro (mag_psfcor[jplus::rSDSS]). NUMERIC indices do not follow filter
    order -- never use them.

Requirements:
    numpy, pandas, requests, astropy

Notes:
    Non-detections carry the SExtractor mag=99 sentinel and are skipped.
    Fluxes are emitted AS-MEASURED (no MW dereddening).
    The uJAVA and bluest medium bands are shallow; expect low S/N and
    possible background over-subtraction systematics for faint sources.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import requests
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky

from ..results import STATUS_NO_MATCH, STATUS_OK, ProviderResult
from ..retry import retry_transient, with_expanding_radius
from ..schema import make_row
from ..units import mag_err_to_flux_err, mag_to_ujy


# ------------------------------------
# Constants
# ------------------------------------
TAP_SYNC_URL = "https://archive.cefca.es/catalogues/vo/tap/jplus-dr3/sync"
TABLE = "jplus.MagABDualObj"

# Band label -> CEFCA filter macro (order = wavelength order).
JPLUS_FILTERS = {
    'uJAVA': 'uJAVA',
    'J0378': 'J0378',
    'J0395': 'J0395',
    'J0410': 'J0410',
    'J0430': 'J0430',
    'gSDSS': 'gSDSS',
    'J0515': 'J0515',
    'rSDSS': 'rSDSS',
    'J0660': 'J0660',
    'iSDSS': 'iSDSS',
    'J0861': 'J0861',
    'zSDSS': 'zSDSS',
}

MAG_SENTINEL = 50.0    # SExtractor 99-style non-detection guard


# ------------------------------------
# Query
# ------------------------------------
def _adql(coord: SkyCoord, radius_arcsec: float) -> str:
    """The cone query with one aliased array element per filter."""
    per_filter = []
    for label, macro in JPLUS_FILTERS.items():
        per_filter.append(f"mag_psfcor[jplus::{macro}] AS mag_{label}")
        per_filter.append(f"mag_err_psfcor[jplus::{macro}] AS err_{label}")
        per_filter.append(f"flags[jplus::{macro}] AS flags_{label}")
        per_filter.append(f"mask_flags[jplus::{macro}] AS mask_{label}")
    return (
        "SELECT alpha_j2000, delta_j2000, tile_id, number, "
        + ", ".join(per_filter)
        + f" FROM {TABLE}"
        + " WHERE 1=CONTAINS(POINT('ICRS', alpha_j2000, delta_j2000), "
        + f"CIRCLE('ICRS', {float(coord.ra.deg):.8f}, {float(coord.dec.deg):.8f}, "
        + f"{radius_arcsec / 3600.0:.8f}))"
    )


def _query_once(coord: SkyCoord, radius_arcsec: float, *, holder: dict) -> list[dict]:
    """One sync TAP query; closest source; one row per detected band."""
    def _post():
        response = requests.post(
            TAP_SYNC_URL,
            data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv",
                  "QUERY": _adql(coord, radius_arcsec)},
            timeout=120,
        )
        response.raise_for_status()
        if "QUERY_STATUS" in response.text and "ERROR" in response.text:
            raise RuntimeError(response.text[:400])
        return response.text

    try:
        text = retry_transient(_post, "J-PLUS")
    except Exception as e:
        print(f"  [J-PLUS] Query error: {e}")
        return []

    df = pd.read_csv(io.StringIO(text), comment="#")
    if df.empty:
        return []

    src_coords = SkyCoord(df['ALPHA_J2000'].values, df['DELTA_J2000'].values, unit=u.deg)
    idx, sep, _ = match_coordinates_sky(coord, src_coords)
    sep_arcsec = float(sep.arcsec[0])
    src = df.iloc[int(idx)]
    holder['tile_id'] = int(src['TILE_ID'])
    holder['number'] = int(src['NUMBER'])
    holder['radius_used'] = radius_arcsec

    rows = []
    for label in JPLUS_FILTERS:
        mag = float(src.get(f'mag_{label}', np.nan))
        mag_err = float(src.get(f'err_{label}', np.nan))

        if not np.isfinite(mag) or mag > MAG_SENTINEL or mag < 0:
            # SExtractor 99 sentinel: not detected in this band.
            continue

        flags = int(src.get(f'flags_{label}', 0))
        mask_flags = int(src.get(f'mask_{label}', 0))
        flag_str = f"{flags}|{mask_flags}" if (flags or mask_flags) else ''

        rows.append(make_row(
            band=f'JPLUS_{label}',
            flux_ujy=mag_to_ujy(mag),
            flux_err_ujy=mag_err_to_flux_err(mag, mag_err),
            mag=mag,
            mag_err=mag_err,
            target_ra=float(coord.ra.deg),
            target_dec=float(coord.dec.deg),
            match_ra=float(src['ALPHA_J2000']),
            match_dec=float(src['DELTA_J2000']),
            sep_arcsec=sep_arcsec,
            flags=flag_str,
            source='JPLUS_DR3_PSFCOR',
        ))

    return rows


def query(coord: SkyCoord, radius_arcsec: float) -> ProviderResult:
    """Query J-PLUS DR3 PSFCOR photometry for the closest source.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    radius_arcsec : float
        Starting search radius; expands on no-match.

    Returns
    -------
    result : ProviderResult
        One row per detected band on success; meta carries tile_id/number.
    """
    holder: dict = {}
    rows = with_expanding_radius(
        lambda c, r: _query_once(c, r, holder=holder),
        coord, radius_arcsec, "J-PLUS DR3",
    )
    meta = {'endpoint': TAP_SYNC_URL, 'table': TABLE, 'mag_type': 'PSFCOR'}
    if rows:
        meta.update(tile_id=holder.get('tile_id'), number=holder.get('number'))
        return ProviderResult(provider='jplus', status=STATUS_OK, rows=rows,
                              radius_used=holder.get('radius_used'), meta=meta)
    return ProviderResult(
        provider='jplus', status=STATUS_NO_MATCH,
        message="no J-PLUS DR3 source found (footprint is the northern sky, ~3000 deg2)",
        meta=meta,
    )
