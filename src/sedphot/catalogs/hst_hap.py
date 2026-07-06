"""
hst_hap.py

HST Hubble Advanced Products Catalog Provider
---------------------------------------------------------
Per-filter photometry from the HAP point/segment catalogs discovered at the
target position via MAST. Prefers the segment catalog's MagSegment
(isophotal, integrated over the full detected footprint -- right for
extended sources such as lensed arcs); falls back to the point catalog's
MagAp2 (0.15" aperture) with a warning only when no segment catalog exists.

Requirements:
    numpy, pandas, astropy, astroquery

Notes:
    Filter discovery is automatic from the MAST observation list (calib
    level 3, detection products skipped). The closest catalog source is
    culled if it lies beyond the search radius -- unlike the cone-search
    providers, HAP catalogs cover the whole visit footprint.
"""
from __future__ import annotations

import os
import tempfile
import urllib.request

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
from astroquery.mast import Observations

from ..results import STATUS_NO_MATCH, STATUS_OK, ProviderResult
from ..retry import with_expanding_radius
from ..schema import make_row
from ..units import flux_err_to_mag_err, mag_err_to_flux_err, mag_to_ujy


# ------------------------------------
# Constants
# ------------------------------------
MAST_FILE_URL = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/{filename}"


# ------------------------------------
# Catalog discovery and download
# ------------------------------------
def _fetch_hap_catalog(filename: str) -> pd.DataFrame | None:
    """Download a HAP catalog ECSV from MAST; None on failure."""
    url = MAST_FILE_URL.format(filename=filename)
    try:
        content = urllib.request.urlopen(url, timeout=60).read()
    except Exception as e:
        print(f"  [HST] Failed to fetch {filename}: {e}")
        return None

    with tempfile.NamedTemporaryFile(suffix='.ecsv', delete=False) as f:
        f.write(content)
        tmppath = f.name

    try:
        cat = Table.read(tmppath, format='ascii.ecsv').to_pandas()
    except Exception as e:
        print(f"  [HST] Failed to parse {filename}: {e}")
        cat = None
    finally:
        os.unlink(tmppath)

    return cat


def _discover_hst_catalogs(coord: SkyCoord, radius_arcsec: float) -> dict[str, dict[str, str | None]]:
    """Map {filter: {'point': filename, 'segment': filename}} for HAP products here.

    Either filename may be None when that catalog type does not exist for a
    filter.
    """
    try:
        obs = Observations.query_region(coord, radius=radius_arcsec * u.arcsec)
    except Exception as e:
        print(f"  [HST] MAST observation query failed: {e}")
        return {}

    hst_obs = obs[obs['obs_collection'] == 'HST']
    if len(hst_obs) == 0:
        return {}

    # Keep only calib_level=3 science products, skip detection images.
    mask = [
        (int(row['calib_level']) == 3)
        and (str(row['filters']).lower() not in ('', 'detection', '--'))
        for row in hst_obs
    ]
    science_obs = hst_obs[mask]

    filter_to_cats: dict[str, dict[str, str]] = {}
    for row in science_obs:
        filt = str(row['filters']).upper()
        obsid = int(row['obsid'])

        if filt in filter_to_cats:
            continue

        try:
            products = Observations.get_product_list(str(obsid))
        except Exception as e:
            print(f"  [HST] get_product_list failed for obsid {obsid}: {e}")
            continue

        point_cat = None
        segment_cat = None
        for prod in products:
            fname = str(prod['productFilename'])
            if 'point-cat.ecsv' in fname:
                point_cat = fname
            elif 'segment-cat.ecsv' in fname:
                segment_cat = fname

        if point_cat or segment_cat:
            filter_to_cats[filt] = {'point': point_cat, 'segment': segment_cat}

    return filter_to_cats


# ------------------------------------
# Query
# ------------------------------------
def _query_once(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    """Discover HAP catalogs, download per filter, return closest-match rows."""
    filter_cats = _discover_hst_catalogs(coord, radius_arcsec)
    if not filter_cats:
        print("  [HST] No HAP catalogs found at this position.")
        return []

    print(f"  [HST] Found catalogs for filters: {list(filter_cats.keys())}")

    rows = []
    for filt, catfiles in filter_cats.items():
        # Prefer segment catalog; fall back to point catalog.
        use_segment = catfiles.get('segment') is not None
        catfile = catfiles['segment'] if use_segment else catfiles.get('point')
        cat_type = 'segment' if use_segment else 'point'

        if catfile is None:
            print(f"  [HST] {filt}: no catalog file found, skipping.")
            continue

        print(f"  [HST] {filt}: using {cat_type} catalog ({catfile})")
        cat = _fetch_hap_catalog(catfile)
        if cat is None or cat.empty:
            continue

        src_coords = SkyCoord(cat['RA'].values, cat['DEC'].values, unit=u.deg)
        idx, sep, _ = match_coordinates_sky(coord, src_coords)
        sep_arcsec = float(sep.arcsec[0])

        if sep_arcsec > radius_arcsec:
            print(f"  [HST] {filt}: nearest source is {sep_arcsec:.2f}\" away, "
                  f"outside search radius.")
            continue

        src = cat.iloc[int(idx)]
        flags = int(src['Flags'])

        if use_segment:
            # MagSegment: isophotal AB magnitude integrated over the full
            # source footprint from the pipeline segmentation map; the catalog
            # zeropoints are calibrated to an infinite aperture, so no further
            # aperture correction applies.
            mag = float(src['MagSegment'])
            flux = float(src['FluxSegment'])
            flux_err = float(src['FluxSegmentErr'])
            mag_err = flux_err_to_mag_err(flux, flux_err) if flux > 0 else np.nan
        else:
            # Point-source aperture photometry (MagAp2, 0.15" radius)
            # undercounts flux for extended sources -- warn accordingly.
            print(f"  [HST] {filt}: WARNING using point catalog MagAp2 -- "
                  f"flux likely underestimated for extended sources.")
            mag = float(src['MagAp2'])
            mag_err = float(src['MagErrAp2'])

        rows.append(make_row(
            band=f'HST_{filt}',
            flux_ujy=mag_to_ujy(mag),
            flux_err_ujy=mag_err_to_flux_err(mag, mag_err),
            mag=mag,
            mag_err=mag_err,
            target_ra=float(coord.ra.deg),
            target_dec=float(coord.dec.deg),
            match_ra=float(src['RA']),
            match_dec=float(src['DEC']),
            sep_arcsec=sep_arcsec,
            flags=flags,
            source=f'HST_HAP_{cat_type}',
        ))

    return rows


def query(coord: SkyCoord, radius_arcsec: float) -> ProviderResult:
    """Query HST HAP catalogs at the target position.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    radius_arcsec : float
        Starting search radius; expands in case the target sits near the
        edge of coverage (blind expansion cannot conjure coverage, but it
        rescues edge cases).

    Returns
    -------
    result : ProviderResult
        One row per filter with a match on success.
    """
    rows = with_expanding_radius(_query_once, coord, radius_arcsec, "HST HAP")
    if rows:
        return ProviderResult(provider='hst', status=STATUS_OK, rows=rows,
                              meta={'service': 'MAST HAP'})
    return ProviderResult(provider='hst', status=STATUS_NO_MATCH,
                          message="no HAP catalogs (or no source within radius) at this position",
                          meta={'service': 'MAST HAP'})
