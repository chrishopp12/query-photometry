#!/usr/bin/env python3
"""
phot_coord_search.py

Single-Object Photometry Retrieval Pipeline
------------------------------------------------------------

Queries archival photometric data for a single sky position from:
  - Legacy Survey DR10 (Tractor catalog via NOIRLab TAP)
      grz broadband + forced-photometry WISE W1/W2 fluxes
  - Pan-STARRS DR1 (via Vizier)
      grizy broadband magnitudes
  - HST HAP (Hubble Advanced Products via MAST)
      Per-band point-source aperture photometry (AB mag)
      Automatically discovers which HST bands cover the position.

Output is a single CSV intended as input for SED fitting (e.g., Prospector).
Each row is one photometric measurement: band, flux_uJy, flux_err_uJy,
magnitude, mag_err, source catalog, angular separation from target, and flags.

Usage:
    python phot_coord_search.py --ra <deg> --dec <deg> [options]

Example:
    python phot_coord_search.py --ra 150.0 --dec 2.2 --radius 2.0 --out target.csv

Options:
    --ra        <float>   Target RA in decimal degrees       [required]
    --dec       <float>   Target Dec in decimal degrees      [required]
    --radius    <float>   Search radius in arcseconds        [default: 2.0]
    --out       <str>     Output CSV filename                [default: target_photometry.csv]
    --no-legacy           Skip Legacy DR10 query
    --no-panstarrs        Skip Pan-STARRS query
    --no-hst              Skip HST HAP query

Notes:
    - The iterative radius search expands by EXPAND_FACTOR (2x) up to
      MAX_RETRIES (5) attempts before giving up on a catalog.
    - HST query discovers available bands automatically from MAST observations.
    - Legacy fluxes (including WISE) are in nanomaggies in the Tractor catalog.
      This script converts them to uJy for a common flux unit.
    - Magnitudes reported are AB system throughout.
    - HST flux uses the HAP segment catalog MagSegment (isophotal, integrated
      over the full source footprint), falling back to the point catalog MagAp2
      (0.15" aperture) only when no segment catalog exists.
    - Pan-STARRS reports PSF magnitudes; for extended sources the Kron
      magnitudes (gKmag etc.) may be more appropriate.
    - Flags column is passed through from source catalogs; inspect before use.
"""
from __future__ import annotations

import os
import io
import argparse
import tempfile
import urllib.request
import warnings

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table
import astropy.units as u

from astroquery.vizier import Vizier
from astroquery.mast import Observations
from astroquery.utils.tap.core import TapPlus


# ------------------------------------
# Constants
# ------------------------------------

LEGACY_URL      = "https://datalab.noirlab.edu/tap"
LEGACY_CATALOG  = "ls_dr10.tractor"
PANSTARRS_CAT   = "II/349/ps1"
MAST_FILE_URL   = "https://mast.stsci.edu/api/v0.1/Download/file?uri=mast:HST/product/{filename}"

DEFAULT_RADIUS_ARCSEC = 2.0
EXPAND_FACTOR         = 2.0   # multiply radius by this on each retry
MAX_RETRIES           = 5     # max expansions before giving up on a catalog

# Nanomaggy -> uJy conversion factor.
# 1 nanomaggy = 3.631 uJy  (AB system, zero-point 3631 Jy)
NANOMAGGY_TO_UJY = 3.631

# Output column schema
OUT_COLS = [
    'band',
    'flux_uJy',
    'flux_err_uJy',
    'mag_AB',
    'mag_err',
    'target_ra',
    'target_dec',
    'match_ra',
    'match_dec',
    'sep_arcsec',
    'flags',
    'source',
]


# ------------------------------------
# Unit Conversion Helpers
# ------------------------------------

def nanomaggy_to_ujy(flux_nm: float) -> float:
    """Convert Legacy Survey nanomaggies to microjanskys."""
    return flux_nm * NANOMAGGY_TO_UJY


def mag_to_ujy(mag: float) -> float:
    """Convert AB magnitude to microjanskys. Returns NaN for non-finite input."""
    if not np.isfinite(mag):
        return np.nan
    return 10 ** ((23.9 - mag) / 2.5)


def mag_err_to_flux_err(mag: float, mag_err: float) -> float:
    """
    Convert magnitude uncertainty to flux uncertainty in the same flux units.

    Derived from error propagation on f = 10^((23.9 - m)/2.5):
        sigma_f = f * (ln10 / 2.5) * sigma_m
    """
    if not np.isfinite(mag) or not np.isfinite(mag_err):
        return np.nan
    flux = mag_to_ujy(mag)
    return flux * (np.log(10) / 2.5) * mag_err


def flux_err_to_mag_err(flux: float, flux_err: float) -> float:
    """Convert flux uncertainty (any linear unit) to magnitude uncertainty."""
    if not np.isfinite(flux) or flux <= 0 or not np.isfinite(flux_err):
        return np.nan
    return (2.5 / np.log(10)) * (flux_err / flux)


def ujy_to_mag(flux_ujy: float) -> float:
    """Convert microjanskys to AB magnitude."""
    if not np.isfinite(flux_ujy) or flux_ujy <= 0:
        return np.nan
    return 23.9 - 2.5 * np.log10(flux_ujy)


# ------------------------------------
# Result Row Builder
# ------------------------------------

def make_row(
        band: str,
        flux_ujy: float,
        flux_err_ujy: float,
        mag: float,
        mag_err: float,
        target_ra: float,
        target_dec: float,
        match_ra: float,
        match_dec: float,
        sep_arcsec: float,
        flags,
        source: str,
) -> dict:
    """Build a single output row dict with all required columns."""
    return {
        'band':         band,
        'flux_uJy':     round(float(flux_ujy),     6) if np.isfinite(flux_ujy)     else np.nan,
        'flux_err_uJy': round(float(flux_err_ujy), 6) if np.isfinite(flux_err_ujy) else np.nan,
        'mag_AB':       round(float(mag),           4) if np.isfinite(mag)          else np.nan,
        'mag_err':      round(float(mag_err),       4) if np.isfinite(mag_err)      else np.nan,
        'target_ra':    round(float(target_ra),     8),
        'target_dec':   round(float(target_dec),    8),
        'match_ra':     round(float(match_ra),      8),
        'match_dec':    round(float(match_dec),     8),
        'sep_arcsec':   round(float(sep_arcsec),    4),
        'flags':        flags,
        'source':       source,
    }


# ------------------------------------
# Iterative Radius Wrapper
# ------------------------------------

def with_expanding_radius(query_fn, coord: SkyCoord, radius_arcsec: float, label: str) -> list[dict]:
    """
    Call query_fn(coord, radius_arcsec) up to MAX_RETRIES times,
    doubling the radius each time if nothing is returned.

    Parameters
    ----------
    query_fn : callable
        Function with signature (coord, radius_arcsec) -> list[dict].
        Should return [] on no results (not raise).
    coord : SkyCoord
    radius_arcsec : float
        Starting search radius.
    label : str
        Catalog name for logging.

    Returns
    -------
    list[dict] -- rows from the first successful attempt, or [] if all fail.
    """
    r = radius_arcsec
    for attempt in range(1, MAX_RETRIES + 1):
        print(f"  [{label}] Attempt {attempt}/{MAX_RETRIES}, radius={r:.1f}\"")
        rows = query_fn(coord, r)
        if rows:
            print(f"  [{label}] Found {len(rows)} match(es) at radius={r:.1f}\"")
            return rows
        print(f"  [{label}] No results. Expanding radius.")
        r *= EXPAND_FACTOR
    print(f"  [{label}] No results after {MAX_RETRIES} attempts.")
    return []


# ------------------------------------
# Legacy DR10 Query
# ------------------------------------

# Bands and their Tractor column names.
# flux_* columns are in nanomaggies; flux_ivar_* are inverse variance in nanomaggies^-2.
LEGACY_BANDS = {
    'g': ('flux_g',  'flux_ivar_g'),
    'r': ('flux_r',  'flux_ivar_r'),
    'i': ('flux_i',  'flux_ivar_i'),
    'z': ('flux_z',  'flux_ivar_z'),
    'W1': ('flux_w1', 'flux_ivar_w1'),
    'W2': ('flux_w2', 'flux_ivar_w2'),
}


def _query_legacy_once(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    """
    Query Legacy DR10 Tractor catalog for the closest source within radius.

    Returns a list of row dicts (one per band with a valid flux), or [].
    """
    tap = TapPlus(url=LEGACY_URL)
    ra  = float(coord.ra.deg)
    dec = float(coord.dec.deg)
    radius_deg = radius_arcsec / 3600.0

    flux_cols  = ', '.join(fc for fc, _ in LEGACY_BANDS.values())
    ivar_cols  = ', '.join(ic for _, ic in LEGACY_BANDS.values())

    query = f"""
    SELECT ra, dec,
           {flux_cols},
           {ivar_cols}
    FROM {LEGACY_CATALOG}
    WHERE brick_primary = 1
      AND 't' = q3c_radial_query(ra, dec, {ra:.8f}, {dec:.8f}, {radius_deg:.8f})
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            job    = tap.launch_job(query)
            result = job.get_results().to_pandas()
    except Exception as e:
        print(f"  [Legacy] Query error: {e}")
        return []

    if result.empty:
        return []

    # Among all returned sources, pick the one closest to the target
    src_coords = SkyCoord(result['ra'].values, result['dec'].values, unit=u.deg)
    idx, sep, _ = match_coordinates_sky(coord, src_coords)
    sep_arcsec  = sep.arcsec[0]
    row_df      = result.iloc[int(idx)]
    m_ra        = float(row_df['ra'])
    m_dec       = float(row_df['dec'])

    rows = []
    for band_label, (flux_col, ivar_col) in LEGACY_BANDS.items():
        flux_nm = float(row_df.get(flux_col, np.nan))
        ivar_nm = float(row_df.get(ivar_col, np.nan))

        # Tractor can store negative fluxes for non-detections; keep them
        # (Prospector can handle flux PDFs that straddle zero).
        flux_ujy = nanomaggy_to_ujy(flux_nm) if np.isfinite(flux_nm) else np.nan

        if np.isfinite(ivar_nm) and ivar_nm > 0:
            flux_err_nm  = 1.0 / np.sqrt(ivar_nm)
            flux_err_ujy = nanomaggy_to_ujy(flux_err_nm)
        else:
            flux_err_ujy = np.nan

        mag     = ujy_to_mag(flux_ujy)
        mag_err = flux_err_to_mag_err(flux_ujy, flux_err_ujy)

        rows.append(make_row(
            band        = f'Legacy_{band_label}',
            flux_ujy    = flux_ujy,
            flux_err_ujy= flux_err_ujy,
            mag         = mag,
            mag_err     = mag_err,
            target_ra   = float(coord.ra.deg),
            target_dec  = float(coord.dec.deg),
            match_ra    = m_ra,
            match_dec   = m_dec,
            sep_arcsec  = sep_arcsec,
            flags       = '',
            source      = 'Legacy_DR10',
        ))

    return rows


def query_legacy(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    return with_expanding_radius(_query_legacy_once, coord, radius_arcsec, "Legacy DR10")


# ------------------------------------
# Pan-STARRS DR1 Query
# ------------------------------------

# Column names in Vizier II/349/ps1
PANSTARRS_BANDS = {
    'g': ('gmag', 'e_gmag'),
    'r': ('rmag', 'e_rmag'),
    'i': ('imag', 'e_imag'),
    'z': ('zmag', 'e_zmag'),
    'y': ('ymag', 'e_ymag'),
}


def _query_panstarrs_once(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    """
    Query Pan-STARRS DR1 via Vizier for the closest source within radius.

    Returns a list of row dicts (one per detected band), or [].
    """
    mag_cols = [c for pair in PANSTARRS_BANDS.values() for c in pair]
    vizier = Vizier(
        columns=['RAJ2000', 'DEJ2000'] + mag_cols,
        column_filters={},
        row_limit=-1,
    )

    try:
        result = vizier.query_region(
            coord,
            radius=radius_arcsec * u.arcsec,
            catalog=PANSTARRS_CAT,
        )
    except Exception as e:
        print(f"  [Pan-STARRS] Query error: {e}")
        return []

    if not result:
        return []

    df = result[0].to_pandas()
    if df.empty:
        return []

    src_coords = SkyCoord(df['RAJ2000'].values, df['DEJ2000'].values, unit=u.deg)
    idx, sep, _ = match_coordinates_sky(coord, src_coords)
    sep_arcsec  = sep.arcsec[0]
    row_df      = df.iloc[int(idx)]
    m_ra        = float(row_df['RAJ2000'])
    m_dec       = float(row_df['DEJ2000'])

    rows = []
    for band_label, (mag_col, err_col) in PANSTARRS_BANDS.items():
        mag     = float(row_df.get(mag_col, np.nan))
        mag_err = float(row_df.get(err_col, np.nan))

        if not np.isfinite(mag):
            # Pan-STARRS returns NaN for non-detections; skip rather than propagate garbage
            continue

        flux_ujy     = mag_to_ujy(mag)
        flux_err_ujy = mag_err_to_flux_err(mag, mag_err)

        rows.append(make_row(
            band         = f'PS1_{band_label}',
            flux_ujy     = flux_ujy,
            flux_err_ujy = flux_err_ujy,
            mag          = mag,
            mag_err      = mag_err,
            target_ra    = float(coord.ra.deg),
            target_dec   = float(coord.dec.deg),
            match_ra     = m_ra,
            match_dec    = m_dec,
            sep_arcsec   = sep_arcsec,
            flags        = '',
            source       = 'PanSTARRS_DR1',
        ))

    return rows


def query_panstarrs(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    return with_expanding_radius(_query_panstarrs_once, coord, radius_arcsec, "Pan-STARRS")


# ------------------------------------
# HST HAP Query
# ------------------------------------

def _fetch_hap_catalog(filename: str) -> pd.DataFrame | None:
    """
    Download a HAP catalog ECSV from MAST and return as DataFrame.
    Returns None on failure.
    """
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


def _discover_hst_catalogs(coord: SkyCoord, radius_arcsec: float) -> dict[str, dict[str, str]]:
    """
    Query MAST for HST observations overlapping this position and return a dict
    mapping {filter_name: {'point': filename, 'segment': filename}} for HAP products.

    Both the point-source and segment catalog filenames are returned so the caller
    can choose which to use.  Either value may be None if that catalog type was not
    found for a given filter.
    """
    try:
        obs = Observations.query_region(coord, radius=radius_arcsec * u.arcsec)
    except Exception as e:
        print(f"  [HST] MAST observation query failed: {e}")
        return {}

    hst_obs = obs[obs['obs_collection'] == 'HST']
    if len(hst_obs) == 0:
        return {}

    # Keep only calib_level=3, skip detection products
    mask = [
        (int(row['calib_level']) == 3)
        and (str(row['filters']).lower() not in ('', 'detection', '--'))
        for row in hst_obs
    ]
    science_obs = hst_obs[mask]

    filter_to_cats: dict[str, dict[str, str]] = {}

    for row in science_obs:
        filt  = str(row['filters']).upper()
        obsid = int(row['obsid'])

        if filt in filter_to_cats:
            continue

        try:
            products = Observations.get_product_list(str(obsid))
        except Exception as e:
            print(f"  [HST] get_product_list failed for obsid {obsid}: {e}")
            continue

        point_cat   = None
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


def _query_hst_once(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    """
    Discover HST HAP catalogs at this position, download the segment catalog for
    each filter, and return the closest source match per filter using MagSegment
    (isophotal flux over the full detected source footprint).

    MagSegment is the right choice for extended sources such as lensed arcs because
    it integrates flux over the entire source segment rather than a small fixed
    aperture.  The point-source MagAp2 (0.15" radius) severely undercounts flux
    for sources with half-light radii of even a few tenths of an arcsecond.

    Falls back to the point catalog / MagAp2 only when no segment catalog exists.

    Returns a list of row dicts (one per filter with a match), or [].
    """
    filter_cats = _discover_hst_catalogs(coord, radius_arcsec)
    if not filter_cats:
        print("  [HST] No HAP catalogs found at this position.")
        return []

    print(f"  [HST] Found catalogs for filters: {list(filter_cats.keys())}")

    rows = []
    for filt, catfiles in filter_cats.items():

        # Prefer segment catalog; fall back to point catalog
        use_segment = catfiles.get('segment') is not None
        catfile     = catfiles['segment'] if use_segment else catfiles.get('point')
        cat_type    = 'segment' if use_segment else 'point'

        if catfile is None:
            print(f"  [HST] {filt}: no catalog file found, skipping.")
            continue

        print(f"  [HST] {filt}: using {cat_type} catalog ({catfile})")
        cat = _fetch_hap_catalog(catfile)
        if cat is None or cat.empty:
            continue

        src_coords = SkyCoord(cat['RA'].values, cat['DEC'].values, unit=u.deg)
        idx, sep, _ = match_coordinates_sky(coord, src_coords)
        sep_arcsec  = float(sep.arcsec[0])

        if sep_arcsec > radius_arcsec:
            print(f"  [HST] {filt}: nearest source is {sep_arcsec:.2f}\" away, outside search radius.")
            continue

        row_df = cat.iloc[int(idx)]
        m_ra   = float(row_df['RA'])
        m_dec  = float(row_df['DEC'])
        flags  = int(row_df['Flags'])

        if use_segment:
            # MagSegment: isophotal AB magnitude integrated over the full source
            # footprint as detected by the pipeline segmentation map.
            # FluxSegment is in electrons/s; the catalog zeropoints are calibrated
            # to an infinite aperture, so MagSegment is already on an absolute scale
            # (no additional aperture correction needed beyond what the pipeline applied).
            mag     = float(row_df['MagSegment'])
            # FluxSegmentErr is the propagated uncertainty from the error array.
            # Convert to a magnitude error via standard error propagation.
            flux    = float(row_df['FluxSegment'])
            flux_err= float(row_df['FluxSegmentErr'])
            mag_err = flux_err_to_mag_err(flux, flux_err) if flux > 0 else np.nan
        else:
            # Fallback: point-source aperture photometry (MagAp2, 0.15" radius).
            # This will undercount flux for extended sources -- flag accordingly.
            print(f"  [HST] {filt}: WARNING using point catalog MagAp2 -- flux likely underestimated for extended sources.")
            mag     = float(row_df['MagAp2'])
            mag_err = float(row_df['MagErrAp2'])

        flux_ujy     = mag_to_ujy(mag)
        flux_err_ujy = mag_err_to_flux_err(mag, mag_err)

        rows.append(make_row(
            band         = f'HST_{filt}',
            flux_ujy     = flux_ujy,
            flux_err_ujy = flux_err_ujy,
            mag          = mag,
            mag_err      = mag_err,
            target_ra    = float(coord.ra.deg),
            target_dec   = float(coord.dec.deg),
            match_ra     = m_ra,
            match_dec    = m_dec,
            sep_arcsec   = sep_arcsec,
            flags        = flags,
            source       = f'HST_HAP_{cat_type}',
        ))

    return rows



def query_hst(coord: SkyCoord, radius_arcsec: float) -> list[dict]:
    # HST discovery doesn't benefit from blind radius expansion the same way;
    # if no observations cover the field, a larger radius won't help.
    # But we still try expanding in case the target is near the edge of coverage.
    return with_expanding_radius(_query_hst_once, coord, radius_arcsec, "HST HAP")


# ------------------------------------
# Main Pipeline
# ------------------------------------

def run_query(
        ra: float,
        dec: float,
        radius_arcsec: float = DEFAULT_RADIUS_ARCSEC,
        out_file: str = "target_photometry.csv",
        do_legacy: bool = True,
        do_panstarrs: bool = True,
        do_hst: bool = True,
) -> pd.DataFrame:
    """
    Query all enabled catalogs and write a combined photometry CSV.

    Parameters
    ----------
    ra, dec : float
        Target position in decimal degrees (ICRS).
    radius_arcsec : float
        Initial search radius; will expand up to MAX_RETRIES times if needed.
    out_file : str
        Output CSV path.
    do_legacy, do_panstarrs, do_hst : bool
        Toggle individual catalogs.

    Returns
    -------
    pd.DataFrame  -- the combined photometry table (also written to out_file).
    """
    coord = SkyCoord(ra, dec, unit=u.deg)
    print(f"\nTarget: RA={ra:.6f}, Dec={dec:.6f}  (search radius={radius_arcsec:.1f}\")\n")

    all_rows: list[dict] = []

    if do_legacy:
        print("=== Legacy DR10 ===")
        all_rows.extend(query_legacy(coord, radius_arcsec))

    if do_panstarrs:
        print("\n=== Pan-STARRS DR1 ===")
        all_rows.extend(query_panstarrs(coord, radius_arcsec))

    if do_hst:
        print("\n=== HST HAP ===")
        all_rows.extend(query_hst(coord, radius_arcsec))

    if not all_rows:
        print("\nNo photometry retrieved from any catalog.")
        return pd.DataFrame(columns=OUT_COLS)

    df = pd.DataFrame(all_rows, columns=OUT_COLS)

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    df.to_csv(out_file, index=False)
    print(f"\nSaved {len(df)} photometric points to: {out_file}")
    print(df.to_string(index=False))

    return df


# ------------------------------------
# CLI
# ------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Query archival photometry for a single sky position."
    )
    parser.add_argument('--ra', type=float, required=True,
                        help="RA in decimal degrees")
    parser.add_argument('--dec', type=float, required=True,
                        help="Dec in decimal degrees")
    parser.add_argument('--radius', type=float, default=DEFAULT_RADIUS_ARCSEC,
                        help=f"Search radius in arcseconds (default: {DEFAULT_RADIUS_ARCSEC})")
    parser.add_argument('--out', type=str, default="target_photometry.csv",
                        help="Output CSV filename (default: target_photometry.csv)")
    parser.add_argument('--no-legacy', action='store_true', help="Skip Legacy DR10")
    parser.add_argument('--no-panstarrs', action='store_true', help="Skip Pan-STARRS")
    parser.add_argument('--no-hst', action='store_true', help="Skip HST HAP")
    args = parser.parse_args()

    run_query(
        ra=args.ra,
        dec=args.dec,
        radius_arcsec=args.radius,
        out_file=args.out,
        do_legacy=not args.no_legacy,
        do_panstarrs=not args.no_panstarrs,
        do_hst=not args.no_hst,
    )


if __name__ == "__main__":
    main()