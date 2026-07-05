"""
sdss.py

SDSS Photometric Catalog Provider
---------------------------------------------------------
Closest primary-source ugriz photometry from the SDSS photometric catalog
via astroquery.sdss. Uses cModelMag -- the composite-model magnitude that
approximates a galaxy total (the right default for a galaxy SED package;
point sources would prefer PSF magnitudes).

The catalog's per-band extinction_* columns (SFD) are converted to MW
transmission and carried per row; fluxes are emitted AS-MEASURED.

Requirements:
    numpy, astropy, astroquery

Notes:
    SDSS photometry deblends aggressively; bright extended galaxies can be
    shredded into pieces (the reason catalog SDSS was dropped from the A1925
    BCG work in favor of frame-level measurement). Inspect sep_arcsec and
    compare against another catalog before trusting a bright-galaxy total.
    Sentinel magnitudes (-9999) are skipped per band.
"""
from __future__ import annotations

import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astroquery.sdss import SDSS

from ..results import STATUS_NO_MATCH, STATUS_OK, ProviderResult
from ..retry import with_expanding_radius
from ..schema import make_row
from ..units import mag_err_to_flux_err, mag_to_ujy

# ------------------------------------
# Constants
# ------------------------------------
DATA_RELEASE = 17
SDSS_BANDS = ('u', 'g', 'r', 'i', 'z')

_PHOTOOBJ_FIELDS = (
    ['ra', 'dec', 'mode', 'type']
    + [f'cModelMag_{b}' for b in SDSS_BANDS]
    + [f'cModelMagErr_{b}' for b in SDSS_BANDS]
    + [f'extinction_{b}' for b in SDSS_BANDS]
)


# ------------------------------------
# Query
# ------------------------------------
def _query_once(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    """One SDSS region query; closest primary source; one row per band."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = SDSS.query_region(
                coord,
                radius=radius_arcsec * u.arcsec,
                photoobj_fields=_PHOTOOBJ_FIELDS,
                data_release=DATA_RELEASE,
            )
    except Exception as e:
        print(f"  [SDSS] Query error: {e}")
        return []

    if result is None or len(result) == 0:
        return []

    df = result.to_pandas()
    # mode == 1 keeps primary detections (drops duplicates from overlaps).
    df = df[df['mode'] == 1]
    if df.empty:
        return []

    src_coords = SkyCoord(df['ra'].values, df['dec'].values, unit=u.deg)
    idx, sep, _ = match_coordinates_sky(coord, src_coords)
    sep_arcsec = float(sep.arcsec[0])
    src = df.iloc[int(idx)]

    rows = []
    for band in SDSS_BANDS:
        mag = float(src.get(f'cModelMag_{band}', np.nan))
        mag_err = float(src.get(f'cModelMagErr_{band}', np.nan))
        extinction = float(src.get(f'extinction_{band}', np.nan))

        if not np.isfinite(mag) or mag < 0 or mag > 40:
            # -9999 sentinel: no measurement in this band.
            continue

        mw_transmission = 10 ** (-extinction / 2.5) if np.isfinite(extinction) else np.nan

        rows.append(make_row(
            band=f'SDSS_{band}',
            flux_ujy=mag_to_ujy(mag),
            flux_err_ujy=mag_err_to_flux_err(mag, mag_err),
            mag=mag,
            mag_err=mag_err,
            target_ra=float(coord.ra.deg),
            target_dec=float(coord.dec.deg),
            match_ra=float(src['ra']),
            match_dec=float(src['dec']),
            sep_arcsec=sep_arcsec,
            flags=int(src.get('type', 0)),
            source=f'SDSS_DR{DATA_RELEASE}_cModel',
            mw_transmission=mw_transmission,
        ))

    return rows


def query(coord: SkyCoord, radius_arcsec: float) -> ProviderResult:
    """Query SDSS for the closest primary photometric source.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    radius_arcsec : float
        Starting search radius; expands on no-match.

    Returns
    -------
    result : ProviderResult
        One row per measured ugriz band on success.
    """
    rows = with_expanding_radius(_query_once, coord, radius_arcsec, "SDSS")
    meta = {'service': 'astroquery.sdss', 'data_release': DATA_RELEASE,
            'mag_type': 'cModelMag'}
    if rows:
        return ProviderResult(provider='sdss', status=STATUS_OK, rows=rows, meta=meta)
    return ProviderResult(provider='sdss', status=STATUS_NO_MATCH,
                          message="no SDSS primary source found (footprint is mostly "
                                  "the northern galactic cap)",
                          meta=meta)
