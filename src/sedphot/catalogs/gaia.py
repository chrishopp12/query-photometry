"""
gaia.py

Gaia DR3 Scene-Input Catalog
---------------------------------------------------------
Cone queries against gaia_dr3.gaia_source through the NOIRLab Datalab TAP
service (the same endpoint that serves the Legacy Surveys tables). This is
a scene-input catalog module, not a photometry provider: it returns the
raw catalog DataFrame for scene construction and does not join
catalogs.CATALOG_PROVIDERS.

Column conventions:
    GAIA_COLS carries position, the G-band magnitude, the five-parameter
    astrometric solution (parallax and proper motions, with errors), and
    the RUWE fit-quality statistic. Values are Gaia-native; nothing is
    converted here.

Requirements:
    pandas, astropy, astroquery

Notes:
    Cache-first: when cache_path names an existing file the frame is read
    from disk and the network is never touched. Gaia membership alone does
    not confirm a star -- compact galaxy nuclei appear in Gaia -- so star
    confirmation thresholds are applied by the caller, not here.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord
from astroquery.utils.tap.core import TapPlus

from ..retry import retry_transient
from .legacy import LEGACY_URL


# ------------------------------------
# Constants
# ------------------------------------
GAIA_TABLE = 'gaia_dr3.gaia_source'

# Everything the scene engine needs to confirm a star and rank PSF-star
# candidates: position, G magnitude, the five-parameter astrometric
# solution with errors, and RUWE.
GAIA_COLS = (
    'ra', 'dec', 'phot_g_mean_mag',
    'parallax', 'parallax_error',
    'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
    'ruwe',
)


# ------------------------------------
# Query
# ------------------------------------
def query_cone(
        coord: SkyCoord,
        radius_arcsec: float,
        *,
        cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """Every Gaia DR3 source in a cone, cache-first.

    Parameters
    ----------
    coord : SkyCoord
        Cone center.
    radius_arcsec : float
        Cone radius.
    cache_path : str or Path, optional
        CSV cache. When the file exists it is read and returned with no
        network call; otherwise the query result is written there.

    Returns
    -------
    gaia_df : pd.DataFrame
        One row per source, columns GAIA_COLS. The frame is identical
        whether it came from the cache or the network.
    """
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        return pd.read_csv(cache)

    ra = float(coord.ra.deg)
    dec = float(coord.dec.deg)
    radius_deg = radius_arcsec / 3600.0
    query = f"""
    SELECT {', '.join(GAIA_COLS)}
    FROM {GAIA_TABLE}
    WHERE 't' = q3c_radial_query(ra, dec, {ra:.8f}, {dec:.8f}, {radius_deg:.8f})
    """

    def _run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tap = TapPlus(url=LEGACY_URL)
            job = tap.launch_job(query)
            return job.get_results().to_pandas()

    gaia_df = retry_transient(_run, "Gaia DR3 TAP")
    if cache is not None:
        # Write, then read back: the network path returns exactly what a
        # later cache hit will return (the CSV round trip normalizes dtypes).
        gaia_df.to_csv(cache, index=False)
        gaia_df = pd.read_csv(cache)
    return gaia_df
