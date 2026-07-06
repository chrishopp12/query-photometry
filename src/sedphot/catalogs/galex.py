"""
galex.py

GALEX GUVcat Catalog Provider
---------------------------------------------------------
Closest-source FUV/NUV photometry from GUVcat_AIS (Bianchi et al. 2017, the
All-sky Imaging Survey tier) via VizieR catalog II/335/galex_ais. The
catalog is queried through VizieR rather than the MAST GALEX endpoint,
which can error where VizieR answers.

The catalog's own E(B-V) (SFD at the source position) is converted to a
per-band MW transmission carried in the mw_transmission column, using
A_FUV = 8.06 x E(B-V), A_NUV = 7.95 x E(B-V) (Bianchi+2017). Fluxes are
emitted AS-MEASURED; --dered applies the correction downstream.

Requirements:
    numpy, astropy, astroquery

Notes:
    GUVcat magnitudes are native AB. Non-detections (sentinel/NaN mags) are
    skipped per band. AIS is shallow (~100 s); a no_match is common outside
    deeper GI fields. VizieR outages can present as empty results -- see
    panstarrs.py.
"""
from __future__ import annotations

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astroquery.vizier import Vizier

from ..results import STATUS_NO_MATCH, STATUS_OK, ProviderResult
from ..retry import query_vizier_mirrors, with_expanding_radius
from ..schema import make_row
from ..units import mag_err_to_flux_err, mag_to_ujy


# ------------------------------------
# Constants
# ------------------------------------
GALEX_CAT = "II/335/galex_ais"

# A_band / E(B-V) (Bianchi+2017).
EXT_PER_EBV = {'FUV': 8.06, 'NUV': 7.95}

GALEX_BANDS = {
    'FUV': ('FUVmag', 'e_FUVmag'),
    'NUV': ('NUVmag', 'e_NUVmag'),
}


# ------------------------------------
# Query
# ------------------------------------
def _query_once(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    """One VizieR cone query; closest source; one row per detected band."""
    mag_cols = [c for pair in GALEX_BANDS.values() for c in pair]
    vizier = Vizier(
        columns=['RAJ2000', 'DEJ2000', 'E(B-V)'] + mag_cols,
        row_limit=-1,
    )
    result = query_vizier_mirrors(
        lambda: vizier.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
            catalog=GALEX_CAT,
        ),
        "GALEX",
    )
    if not result:
        return []

    df = result[0].to_pandas()
    if df.empty:
        return []

    src_coords = SkyCoord(df['RAJ2000'].values, df['DEJ2000'].values, unit=u.deg)
    idx, sep, _ = match_coordinates_sky(coord, src_coords)
    sep_arcsec = float(sep.arcsec[0])
    src = df.iloc[int(idx)]
    ebv = float(src.get('E_B-V_', src.get('E(B-V)', np.nan)))

    rows = []
    for band, (mag_col, err_col) in GALEX_BANDS.items():
        mag = float(src.get(mag_col, np.nan))
        mag_err = float(src.get(err_col, np.nan))

        if not np.isfinite(mag) or mag <= 0 or mag > 40:
            # GUVcat sentinel / NaN: not detected in this band.
            continue

        if np.isfinite(ebv):
            mw_transmission = 10 ** (-EXT_PER_EBV[band] * ebv / 2.5)
        else:
            mw_transmission = np.nan

        rows.append(make_row(
            band=f'GALEX_{band}',
            flux_ujy=mag_to_ujy(mag),
            flux_err_ujy=mag_err_to_flux_err(mag, mag_err),
            mag=mag,
            mag_err=mag_err,
            target_ra=float(coord.ra.deg),
            target_dec=float(coord.dec.deg),
            match_ra=float(src['RAJ2000']),
            match_dec=float(src['DEJ2000']),
            sep_arcsec=sep_arcsec,
            flags='',
            source='GALEX_GUVcat_AIS',
            mw_transmission=mw_transmission,
        ))

    return rows


def query(coord: SkyCoord, radius_arcsec: float) -> ProviderResult:
    """Query GUVcat_AIS for the closest GALEX source.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    radius_arcsec : float
        Starting search radius; expands on no-match (GALEX PSF is ~5",
        so expansion matters more here than for the optical catalogs).

    Returns
    -------
    result : ProviderResult
        FUV/NUV rows where detected.
    """
    rows = with_expanding_radius(_query_once, coord, radius_arcsec, "GALEX GUVcat")
    meta = {'catalog': GALEX_CAT, 'service': 'VizieR'}
    if rows:
        return ProviderResult(provider='galex', status=STATUS_OK, rows=rows, meta=meta)
    return ProviderResult(provider='galex', status=STATUS_NO_MATCH,
                          message="no GUVcat AIS source found (AIS is ~100 s deep; "
                                  "faint galaxies are often absent)",
                          meta=meta)
