"""
legacy.py

Legacy Surveys Tractor Catalog Provider
---------------------------------------------------------
Closest-source photometry from the Legacy Surveys Tractor catalog via the
NOIRLab Datalab TAP service: optical broadbands plus the unWISE forced
photometry carried in the same table (the WISE measurement matched to the
Legacy source model -- distinct provenance from AllWISE).

Ported from phot_coord_search.py with three additions: the Milky Way
transmission columns are queried and carried per row, W3/W4 join W1/W2, and
the data release is selectable (DR10 has i-band and the southern sky; DR9
covers the BASS/MzLS north).

One deliberate deviation from v1: WISE rows are labeled WISE_Wn (not
Legacy_Wn), with the unWISE provenance in the source column -- band identity
is the filter, measurement provenance is the source (the A1925 convention
and the sed_fitting registry rule). This keeps unWISE-forced and AllWISE
values under one band name, distinguished where provenance belongs.

Column conventions:
    Tractor flux_* are nanomaggies; flux_ivar_* are 1/nanomaggy^2.
    mw_transmission_* are the per-band MW transmission factors (<=1);
    fluxes are emitted AS-MEASURED (not dereddened) with the factor carried
    in the mw_transmission column.

Requirements:
    numpy, astropy, astroquery

Notes:
    Negative Tractor fluxes are legitimate non-detections and are preserved.
    A position outside the release footprint exhausts the radius expansion
    and reports no_match with a hint to try the other release.
"""
from __future__ import annotations

import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astroquery.utils.tap.core import TapPlus

from ..results import STATUS_NO_MATCH, STATUS_OK, ProviderResult
from ..retry import with_expanding_radius
from ..schema import make_row
from ..units import flux_err_to_mag_err, nanomaggy_to_ujy, ujy_to_mag


# ------------------------------------
# Constants
# ------------------------------------
LEGACY_URL = "https://datalab.noirlab.edu/tap"

# Datalab table and band set per data release. DR10 adds i-band (DECam south);
# DR9 is the release covering the BASS/MzLS north.
LEGACY_TABLES = {
    'dr10': 'ls_dr10.tractor',
    'dr9': 'ls_dr9.tractor',
}
LEGACY_BANDS = {
    'dr10': ('g', 'r', 'i', 'z', 'W1', 'W2', 'W3', 'W4'),
    'dr9': ('g', 'r', 'z', 'W1', 'W2', 'W3', 'W4'),
}


def _columns(band: str) -> tuple[str, str, str]:
    """Tractor flux, inverse-variance, and MW-transmission columns for a band."""
    suffix = band.lower()
    return f'flux_{suffix}', f'flux_ivar_{suffix}', f'mw_transmission_{suffix}'


# ------------------------------------
# Query
# ------------------------------------
def _query_once(coord: SkyCoord, radius_arcsec: float, *, dr: str, holder: dict) -> list[dict]:
    """One TAP cone query; closest source; one row per band. [] on no result."""
    table = LEGACY_TABLES[dr]
    bands = LEGACY_BANDS[dr]
    ra = float(coord.ra.deg)
    dec = float(coord.dec.deg)
    radius_deg = radius_arcsec / 3600.0

    band_cols = []
    for band in bands:
        band_cols.extend(_columns(band))
    query = f"""
    SELECT ra, dec, brickname, release,
           {', '.join(band_cols)}
    FROM {table}
    WHERE brick_primary = 1
      AND 't' = q3c_radial_query(ra, dec, {ra:.8f}, {dec:.8f}, {radius_deg:.8f})
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tap = TapPlus(url=LEGACY_URL)
            job = tap.launch_job(query)
            result = job.get_results().to_pandas()
    except Exception as e:
        print(f"  [Legacy] Query error: {e}")
        return []

    if result.empty:
        return []

    # Among all returned sources, pick the one closest to the target.
    src_coords = SkyCoord(result['ra'].values, result['dec'].values, unit=u.deg)
    idx, sep, _ = match_coordinates_sky(coord, src_coords)
    sep_arcsec = float(sep.arcsec[0])
    src = result.iloc[int(idx)]
    holder['brickname'] = str(src['brickname'])
    holder['release'] = int(src['release'])
    holder['radius_used'] = radius_arcsec

    rows = []
    for band in bands:
        flux_col, ivar_col, mwt_col = _columns(band)
        flux_nm = float(src.get(flux_col, np.nan))
        ivar_nm = float(src.get(ivar_col, np.nan))
        mw_transmission = float(src.get(mwt_col, np.nan))

        # Tractor stores negative fluxes for non-detections; keep them
        # (SED fitters handle flux PDFs that straddle zero).
        flux_ujy = nanomaggy_to_ujy(flux_nm) if np.isfinite(flux_nm) else np.nan

        if np.isfinite(ivar_nm) and ivar_nm > 0:
            flux_err_ujy = nanomaggy_to_ujy(1.0 / np.sqrt(ivar_nm))
        else:
            flux_err_ujy = np.nan

        mag = ujy_to_mag(flux_ujy)
        mag_err = flux_err_to_mag_err(flux_ujy, flux_err_ujy)

        # WISE bands: filter identity in the label, unWISE provenance in the
        # source string (see module notes).
        if band.startswith('W'):
            band_label = f'WISE_{band}'
            source = f'unWISE_Legacy_{dr.upper()}'
        else:
            band_label = f'Legacy_{band}'
            source = f'Legacy_{dr.upper()}'

        rows.append(make_row(
            band=band_label,
            flux_ujy=flux_ujy,
            flux_err_ujy=flux_err_ujy,
            mag=mag,
            mag_err=mag_err,
            target_ra=float(coord.ra.deg),
            target_dec=float(coord.dec.deg),
            match_ra=float(src['ra']),
            match_dec=float(src['dec']),
            sep_arcsec=sep_arcsec,
            flags='',
            source=source,
            mw_transmission=mw_transmission,
        ))

    return rows


def query(coord: SkyCoord, radius_arcsec: float, *, dr: str = 'dr10') -> ProviderResult:
    """Query the Legacy Tractor catalog for the closest source.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    radius_arcsec : float
        Starting search radius; expands on no-match.
    dr : str
        Data release, 'dr10' or 'dr9'. [default: 'dr10']

    Returns
    -------
    result : ProviderResult
        One row per band on success; meta carries brickname/release.
    """
    if dr not in LEGACY_TABLES:
        raise ValueError(f"unknown Legacy release {dr!r}; known: {sorted(LEGACY_TABLES)}")

    holder: dict = {}
    rows = with_expanding_radius(
        lambda c, r: _query_once(c, r, dr=dr, holder=holder),
        coord, radius_arcsec, f"Legacy {dr.upper()}",
    )
    if rows:
        return ProviderResult(
            provider='legacy', status=STATUS_OK, rows=rows,
            radius_used=holder.get('radius_used'),
            meta={'endpoint': LEGACY_URL, 'table': LEGACY_TABLES[dr],
                  'brickname': holder.get('brickname'), 'release': holder.get('release')},
        )
    other = 'dr9' if dr == 'dr10' else 'dr10'
    return ProviderResult(
        provider='legacy', status=STATUS_NO_MATCH,
        message=f"no {LEGACY_TABLES[dr]} source found; if outside the {dr.upper()} "
                f"footprint try --legacy-dr {other}",
        meta={'endpoint': LEGACY_URL, 'table': LEGACY_TABLES[dr]},
    )
