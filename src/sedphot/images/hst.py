"""
hst.py

HST HAP Image Provider
---------------------------------------------------------
Drizzled HAP single-visit mosaics from MAST, generalized from
hst_aperture_photometry.py (the RMJ0019 pipeline): the visit-group ranking
(closest pointing, then most filters, then deepest) is ported intact, and
three ACS-only assumptions are removed --

    - any HST imaging instrument (ACS/WFC, WFC3/UVIS, WFC3/IR, ...) unless
      restricted with instruments=;
    - DRC and DRZ products both accepted (WFC3/IR mosaics are DRZ);
    - SCI/WHT extensions located by EXTNAME, not fixed indices.

Each mosaic is split into plain sci/wht FITS files in the cache so the
measurement engine's ImageProduct contract (separate image + inverse
variance paths) applies unchanged. The WHT extension of a drizzled HAP
product is an inverse-variance (IVM) map; drizzle correlates neighboring
pixels, so IVM-summed errors underestimate the true noise by ~1.5-2x (the
original pipeline's documented caveat -- it rides along here).

Requirements:
    numpy, astropy, astroquery

Notes:
    HST pixel scales are 0.03-0.13 arcsec; the measure defaults (10 arcsec
    aperture, 30-45 arcsec sky) are galaxy-survey-sized. For compact HST
    targets pass e.g. --aperture 1 --sky-in 3 --sky-out 4.5 --cutout-size 20
    (the RMJ0019 conventions).
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from ..bands import wave_um
from ..results import STATUS_ERROR, STATUS_NO_COVERAGE, ImageProduct, ProviderResult

SEEING = 0.1                       # arcsec; detection-kernel scale for HST
_FILTER_IN_NAME = re.compile(r"_(f\d{3,4}(?:w|lp|m|n))[_.]", re.IGNORECASE)


# ------------------------------------
# Observation selection (ported group ranking)
# ------------------------------------
def _select_group(coord: SkyCoord, *, proposal_id=None, instruments=None,
                  search_radius_arcmin=5.0, max_sep_arcsec=10.0):
    """Pick the best HAP-SVM visit group covering the position.

    Ranking: closest pointing center, then most filters, then deepest
    total exposure (ported from hst_aperture_photometry.query_mast_hap).
    Returns the observation table for the selected group, or None when
    nothing covers the position.
    """
    from astroquery.mast import Observations

    if proposal_id:
        obs = Observations.query_criteria(
            proposal_id=str(proposal_id),
            dataproduct_type="image",
            provenance_name="HAP-SVM",
        )
    else:
        all_obs = Observations.query_region(
            coord, radius=search_radius_arcmin * u.arcmin)
        keep = ((all_obs["provenance_name"] == "HAP-SVM")
                & (all_obs["dataproduct_type"] == "image"))
        obs = all_obs[keep]

    if instruments:
        wanted = {inst.upper() for inst in instruments}
        obs = obs[[str(row["instrument_name"]).upper() in wanted for row in obs]]

    obs = obs[[str(f).lower() != "detection" for f in obs["filters"]]]
    if len(obs) == 0:
        return None

    if proposal_id:
        obs_coords = SkyCoord(ra=obs["s_ra"], dec=obs["s_dec"], unit="deg")
        obs = obs[coord.separation(obs_coords) < search_radius_arcmin * u.arcmin]
        if len(obs) == 0:
            return None

    groups = defaultdict(list)
    for row in obs:
        groups[(str(row["proposal_id"]), str(row["target_name"]))].append(row)

    group_scores = []
    for (pid, target_name), rows in groups.items():
        center = SkyCoord(ra=np.mean([float(r["s_ra"]) for r in rows]),
                          dec=np.mean([float(r["s_dec"]) for r in rows]),
                          unit="deg")
        sep = coord.separation(center).arcsec
        n_filters = len(set(str(r["filters"]) for r in rows))
        total_exp = sum(float(r["t_exptime"]) for r in rows)
        group_scores.append((sep, -n_filters, -total_exp, pid, target_name, rows))
    group_scores.sort(key=lambda g: g[:3])

    print(f"  [HST] {len(group_scores)} visit group(s):")
    for sep, neg_nf, neg_texp, pid, target_name, rows in group_scores:
        filters = sorted(set(str(r["filters"]) for r in rows))
        print(f"    proposal {pid} '{target_name}': {filters}, "
              f"pointing {sep:.1f}\" away, {-neg_texp:.0f}s total")
    best = group_scores[0]
    ties = [g for g in group_scores if g[0] < best[0] + max_sep_arcsec
            and g is not best]
    if ties:
        print(f"  [HST] note: {len(ties)} group(s) at similar distance; "
              f"pass proposal_id to pick one explicitly")
    print(f"  [HST] selected proposal {best[3]} '{best[4]}'")

    selected = set(str(r["obs_id"]) for r in best[5])
    return obs[[str(row["obs_id"]) in selected for row in obs]]


# ------------------------------------
# Mosaic download + MEF split
# ------------------------------------
def _band_of(filename: str) -> str | None:
    match = _FILTER_IN_NAME.search(str(filename).lower())
    return match.group(1).upper() if match else None


def _split_mosaic(mosaic_path: Path, sci_path: Path, wht_path: Path) -> bool:
    """Split a drizzled MEF into plain sci/wht files (EXTNAME-based)."""
    with fits.open(mosaic_path) as hdul:
        sci = next((h for h in hdul
                    if str(h.header.get("EXTNAME", "")).upper() == "SCI"), None)
        wht = next((h for h in hdul
                    if str(h.header.get("EXTNAME", "")).upper() == "WHT"), None)
        if sci is None and len(hdul) > 1 and hdul[1].data is not None:
            sci = hdul[1]
        if sci is None or sci.data is None:
            return False
        # Calibration keywords can live in either header; merge primary in.
        header = sci.header.copy()
        for key in ("PHOTFLAM", "PHOTPLAM", "EXPTIME", "FILTER", "INSTRUME",
                    "DETECTOR", "BUNIT"):
            if key not in header and key in hdul[0].header:
                header[key] = hdul[0].header[key]
        fits.writeto(sci_path, sci.data.astype("f4"), header, overwrite=True)
        if wht is not None and wht.data is not None:
            fits.writeto(wht_path, wht.data.astype("f4"), wht.header,
                         overwrite=True)
            return True
    return True


def fetch(coord: SkyCoord, *, bands: tuple | None = None, size_arcsec: float = 120.0,
          cache_dir: str | Path, proposal_id=None,
          instruments: tuple | None = None) -> list[ImageProduct] | ProviderResult:
    """Fetch HAP drizzled mosaics covering the target.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    bands : tuple, optional
        Filter subset ('F475W', ...). [default: every filter in the group]
    size_arcsec : float
        Unused (full mosaics are downloaded); kept for interface uniformity.
    cache_dir : str or Path
        Photometry/HST/ directory.
    proposal_id : str, optional
        Restrict to one HST program.
    instruments : tuple, optional
        Instrument restriction ('ACS/WFC', 'WFC3/UVIS', 'WFC3/IR').

    Returns
    -------
    products or result : list[ImageProduct] | ProviderResult
    """
    from astroquery.mast import Observations

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    wanted = {b.upper() for b in bands} if bands else None

    try:
        obs = _select_group(coord, proposal_id=proposal_id,
                            instruments=instruments)
        if obs is None:
            return ProviderResult(provider='hst', status=STATUS_NO_COVERAGE,
                                  message="no HAP-SVM imaging at this position")

        products_table = Observations.get_product_list(obs)
        drizzled = products_table[[
            str(row["productSubGroupDescription"]) in ("DRC", "DRZ")
            for row in products_table]]

        # Combined per-filter mosaics have the shortest obs_id per filter
        # (individual exposures carry a trailing suffix) -- ported heuristic.
        per_filter: dict[str, tuple] = {}
        for row in drizzled:
            band = _band_of(row["productFilename"])
            if band is None or (wanted and band not in wanted):
                continue
            obs_id = str(row["obs_id"])
            if band not in per_filter or len(obs_id) < len(per_filter[band][0]):
                per_filter[band] = (obs_id, row)
        if not per_filter:
            return ProviderResult(provider='hst', status=STATUS_NO_COVERAGE,
                                  message="HAP group found but no drizzled "
                                          "mosaics match the band selection")

        products: list[ImageProduct] = []
        for band, (obs_id, row) in sorted(per_filter.items()):
            sci_path = cache_dir / f"hst_{band}_sci.fits"
            wht_path = cache_dir / f"hst_{band}_wht.fits"
            if not sci_path.exists():
                from astropy.table import Table
                manifest = Observations.download_products(
                    Table(rows=[row], names=drizzled.colnames),
                    download_dir=str(cache_dir))
                local = str(manifest["Local Path"][0])
                if not _split_mosaic(Path(local), sci_path, wht_path):
                    print(f"  [HST] {band}: no SCI extension, skipping")
                    continue
            products.append(ImageProduct(
                provider='hst', instrument='HST', band=band,
                path=str(sci_path), calib='hst',
                invvar_path=str(wht_path) if wht_path.exists() else None,
                seeing_arcsec=SEEING, wave_um=wave_um(f"HST_{band}")))
    except Exception as e:
        return ProviderResult(provider='hst', status=STATUS_ERROR,
                              message=f"{type(e).__name__}: {e}")

    if not products:
        return ProviderResult(provider='hst', status=STATUS_NO_COVERAGE,
                              message="no usable HAP mosaics here")
    return products
