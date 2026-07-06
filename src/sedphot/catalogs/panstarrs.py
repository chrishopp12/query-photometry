"""
panstarrs.py

Pan-STARRS DR1 Catalog Provider
---------------------------------------------------------
Closest-source grizy photometry from the Pan-STARRS DR1 mean-object catalog
via VizieR (II/349/ps1).

Requirements:
    numpy, astropy, astroquery

Notes:
    These are PSF magnitudes; for extended sources the Kron magnitudes
    (gKmag etc.) may be more appropriate -- inspect before trusting bright
    galaxy totals. Bands with NaN magnitudes (non-detections) are skipped
    rather than propagated.

    VizieR outages can present as EMPTY results rather than errors, on the
    mirrors as well as the primary host (so the mirror fallback does not
    catch them). A no_match during such an outage is indistinguishable from
    a true no-match -- if a PS1-covered target reports no_match
    unexpectedly, re-run later.
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
PANSTARRS_CAT = "II/349/ps1"

# Column names in VizieR II/349/ps1.
PANSTARRS_BANDS = {
    'g': ('gmag', 'e_gmag'),
    'r': ('rmag', 'e_rmag'),
    'i': ('imag', 'e_imag'),
    'z': ('zmag', 'e_zmag'),
    'y': ('ymag', 'e_ymag'),
}


# ------------------------------------
# Query
# ------------------------------------
def _query_once(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    """One VizieR cone query; closest source; one row per detected band."""
    mag_cols = [c for pair in PANSTARRS_BANDS.values() for c in pair]
    vizier = Vizier(
        columns=['RAJ2000', 'DEJ2000'] + mag_cols,
        column_filters={},
        row_limit=-1,
    )

    result = query_vizier_mirrors(
        lambda: vizier.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
            catalog=PANSTARRS_CAT,
        ),
        "Pan-STARRS",
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

    rows = []
    for band, (mag_col, err_col) in PANSTARRS_BANDS.items():
        mag = float(src.get(mag_col, np.nan))
        mag_err = float(src.get(err_col, np.nan))

        if not np.isfinite(mag):
            # Pan-STARRS returns NaN for non-detections; skip rather than
            # propagate garbage.
            continue

        rows.append(make_row(
            band=f'PS1_{band}',
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
            source='PanSTARRS_DR1',
        ))

    return rows


def query(coord: SkyCoord, radius_arcsec: float) -> ProviderResult:
    """Query Pan-STARRS DR1 via VizieR for the closest source.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    radius_arcsec : float
        Starting search radius; expands on no-match.

    Returns
    -------
    result : ProviderResult
        One row per detected grizy band on success.
    """
    rows = with_expanding_radius(_query_once, coord, radius_arcsec, "Pan-STARRS")
    if rows:
        return ProviderResult(provider='panstarrs', status=STATUS_OK, rows=rows,
                              meta={'catalog': PANSTARRS_CAT, 'service': 'VizieR'})
    message = "no PS1 source found within the expanded radius"
    if coord.dec.deg < -30:
        message += " (Dec < -30 is outside the PS1 footprint)"
    return ProviderResult(provider='panstarrs', status=STATUS_NO_MATCH,
                          message=message,
                          meta={'catalog': PANSTARRS_CAT, 'service': 'VizieR'})
