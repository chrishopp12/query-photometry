"""
allwise.py

AllWISE Catalog Provider
---------------------------------------------------------
Closest-source W1-W4 photometry from the AllWISE source catalog
(allwise_p3as_psd) via the IRSA service. Profile-fit magnitudes (w*mpro),
converted from the catalog's Vega system to AB before the common uJy
conversion.

Band labels are WISE_Wn -- the same filters as the unWISE forced photometry
the Legacy provider returns -- with 'AllWISE' in the source column carrying
the provenance: band identity is the filter, measurement provenance lives
in the source column.

Requirements:
    numpy, astropy, astroquery

Notes:
    AllWISE profile-fit photometry treats sources as point-like; for faint
    or extended galaxies it UNDER-COUNTS relative to unWISE forced
    photometry, sometimes severalfold. Prefer the Legacy provider's unWISE
    values for galaxies inside the Legacy footprint; this provider covers
    everything else (all-sky).
    A null w*sigmpro marks an upper limit, not a measurement; such bands
    are skipped.
"""
from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astroquery.ipac.irsa import Irsa

from ..results import STATUS_NO_MATCH, STATUS_OK, ProviderResult
from ..retry import with_expanding_radius
from ..schema import make_row
from ..units import mag_err_to_flux_err, mag_to_ujy


# ------------------------------------
# Constants
# ------------------------------------
ALLWISE_CAT = "allwise_p3as_psd"

# Vega -> AB offsets per WISE band (Jarrett et al. 2011 / WISE docs).
VEGA_TO_AB = {'W1': 2.699, 'W2': 3.339, 'W3': 5.174, 'W4': 6.620}

ALLWISE_BANDS = {
    'W1': ('w1mpro', 'w1sigmpro'),
    'W2': ('w2mpro', 'w2sigmpro'),
    'W3': ('w3mpro', 'w3sigmpro'),
    'W4': ('w4mpro', 'w4sigmpro'),
}


# ------------------------------------
# Query
# ------------------------------------
def _query_once(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    """One IRSA cone query; closest source; one row per measured band."""
    mag_cols = [c for pair in ALLWISE_BANDS.values() for c in pair]
    try:
        result = Irsa.query_region(
            coord,
            catalog=ALLWISE_CAT,
            spatial="Cone",
            radius=radius_arcsec * u.arcsec,
            columns=",".join(['ra', 'dec', 'cc_flags', 'ext_flg'] + mag_cols),
        )
    except Exception as e:
        print(f"  [AllWISE] Query error: {e}")
        return []

    if result is None or len(result) == 0:
        return []

    df = result.to_pandas()
    src_coords = SkyCoord(df['ra'].values, df['dec'].values, unit=u.deg)
    idx, sep, _ = match_coordinates_sky(coord, src_coords)
    sep_arcsec = float(sep.arcsec[0])
    src = df.iloc[int(idx)]

    rows = []
    for band, (mag_col, err_col) in ALLWISE_BANDS.items():
        mag_vega = float(src.get(mag_col, np.nan))
        mag_err = float(src.get(err_col, np.nan))

        if not np.isfinite(mag_vega) or not np.isfinite(mag_err):
            # Null w*mpro, or null w*sigmpro: the catalog convention for an
            # upper limit (the mag is a 95% bound, not a measurement). Skip
            # rather than emit a bound dressed up as a flux.
            continue

        mag_ab = mag_vega + VEGA_TO_AB[band]

        rows.append(make_row(
            band=f'WISE_{band}',
            flux_ujy=mag_to_ujy(mag_ab),
            flux_err_ujy=mag_err_to_flux_err(mag_ab, mag_err),
            mag=mag_ab,
            mag_err=mag_err,
            target_ra=float(coord.ra.deg),
            target_dec=float(coord.dec.deg),
            match_ra=float(src['ra']),
            match_dec=float(src['dec']),
            sep_arcsec=sep_arcsec,
            flags=f"cc={str(src.get('cc_flags', '')).strip()}|ext={src.get('ext_flg', '')}",
            source='AllWISE',
        ))

    return rows


def query(coord: SkyCoord, radius_arcsec: float) -> ProviderResult:
    """Query the AllWISE source catalog for the closest source.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    radius_arcsec : float
        Starting search radius; expands on no-match (the W1 PSF is ~6",
        so a few-arcsec radius is already generous for isolated sources).

    Returns
    -------
    result : ProviderResult
        One row per measured WISE band on success (AB magnitudes).
    """
    rows = with_expanding_radius(_query_once, coord, radius_arcsec, "AllWISE")
    meta = {'catalog': ALLWISE_CAT, 'service': 'IRSA', 'mag_type': 'w*mpro (Vega->AB)'}
    if rows:
        return ProviderResult(provider='allwise', status=STATUS_OK, rows=rows, meta=meta)
    return ProviderResult(provider='allwise', status=STATUS_NO_MATCH,
                          message="no AllWISE source found",
                          meta=meta)
