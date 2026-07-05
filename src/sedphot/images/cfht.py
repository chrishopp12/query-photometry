"""
cfht.py

CFHT MegaPipe Image Provider
---------------------------------------------------------
MegaPipe deep-stack cutouts via the CADC SODA service, following the recipe
validated for A1925 (archival inventory 2026-06-01 sections 11.2-11.5):

    Cadc().query_region(coord, collection='CFHTMEGAPIPE')
        -> filter to calibrated image planes
        -> get_image_list(...) SODA cutout URLs
        -> download per band

Pinned pitfalls (all observed):
    - The deep optical stacks are collection CFHTMEGAPIPE. 'CFHTLS' returns
      ZERO planes at positions the MegaPipe stacks cover.
    - astroquery.cadc resolves its endpoints through the CADC registry
      (argus); a registry 503 blocks everything -- transient, retry later.
    - The data host flaps occasionally (404/503/timeout on valid URLs);
      transport retries wrap every call.
    - ugriz stacks carry PHOTZP = 30.0 (AB) -- the 'photzp' calib key reads
      it from the header.

Requirements:
    numpy, requests, astropy, astroquery

Notes:
    Band identity is parsed from the MegaCam filter name in the plane
    metadata (u.MP9302 -> u). WIRCam products are not handled here.
"""
from __future__ import annotations

import re
from pathlib import Path

import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits

from ..results import STATUS_ERROR, STATUS_NO_COVERAGE, ImageProduct, ProviderResult
from ..retry import retry_transient

# ------------------------------------
# Constants
# ------------------------------------
COLLECTION = "CFHTMEGAPIPE"
SEEING = 0.8
WAVE_UM = {'u': 0.355, 'g': 0.475, 'r': 0.640, 'i': 0.776, 'z': 0.925}

# MegaCam filter names look like 'u.MP9301', 'r.MP9601', 'gri.MP9605' (the
# last is a chihuahua-rare combined filter -- skipped by the single-letter rule).
_FILTER_RE = re.compile(r"^([ugriz])\.MP\d+$", re.IGNORECASE)


def _band_of(filter_name: str) -> str | None:
    match = _FILTER_RE.match(str(filter_name).strip())
    return match.group(1).lower() if match else None


# ------------------------------------
# Provider entry
# ------------------------------------
def fetch(coord: SkyCoord, *, bands: tuple | None = None, size_arcsec: float = 120.0,
          cache_dir: str | Path) -> list[ImageProduct] | ProviderResult:
    """Fetch MegaPipe stack cutouts at the target via CADC SODA.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    bands : tuple, optional
        Subset of ugriz. [default: every band with a stack here]
    size_arcsec : float
        Cutout width. [default: 120]
    cache_dir : str or Path
        Photometry/CFHT/ directory; downloads are cached here.

    Returns
    -------
    products or result : list[ImageProduct] | ProviderResult
    """
    import astropy.units as u
    from astroquery.cadc import Cadc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    radius = (size_arcsec / 2.0) * u.arcsec

    try:
        cadc = Cadc()
        result = retry_transient(
            lambda: cadc.query_region(coord, radius=radius, collection=COLLECTION),
            "CADC query")
        if result is None or len(result) == 0:
            return ProviderResult(provider='cfht', status=STATUS_NO_COVERAGE,
                                  message=f"no {COLLECTION} planes at this position")

        # Keep calibrated science images; newest stack per band.
        keep = [i for i, row in enumerate(result)
                if str(row['dataProductType']) == 'image'
                and int(row['calibrationLevel']) >= 3
                and _band_of(row['energy_bandpassName'])]
        result = result[keep]
        if len(result) == 0:
            return ProviderResult(provider='cfht', status=STATUS_NO_COVERAGE,
                                  message="MegaPipe planes found but none are "
                                          "calibrated single-band images")

        per_band: dict[str, list] = {}
        for row in result:
            per_band.setdefault(_band_of(row['energy_bandpassName']), []).append(row)
        wanted = tuple(bands) if bands else tuple(sorted(per_band))

        products: list[ImageProduct] = []
        for band in wanted:
            rows = per_band.get(band)
            if not rows:
                print(f"  [CFHT] no {band} stack here")
                continue
            path = cache_dir / f"cfht_megapipe_{band}.fits"
            if not path.exists():
                subset = result[[i for i, row in enumerate(result)
                                 if _band_of(row['energy_bandpassName']) == band]]
                urls = retry_transient(
                    lambda: cadc.get_image_list(subset, coord, radius),
                    f"CADC SODA {band}")
                if not urls:
                    print(f"  [CFHT] no cutout URL for {band}")
                    continue
                response = retry_transient(
                    lambda: requests.get(urls[0], timeout=600), f"CADC download {band}")
                response.raise_for_status()
                path.write_bytes(response.content)
                # Sanity: PHOTZP present for photometric use.
                with fits.open(path) as hdul:
                    hdu = hdul[1] if len(hdul) > 1 and hdul[0].data is None else hdul[0]
                    if 'PHOTZP' not in hdu.header:
                        print(f"  [CFHT] WARNING {band}: no PHOTZP in header -- "
                              f"morphology-grade only, skipping photometry")
                        path.rename(path.with_suffix(".nophotzp.fits"))
                        continue
            products.append(ImageProduct(
                provider='cfht', instrument='CFHT', band=band,
                path=str(path), calib='photzp', seeing_arcsec=SEEING,
                wave_um=WAVE_UM.get(band, float('nan'))))
    except Exception as e:
        return ProviderResult(provider='cfht', status=STATUS_ERROR,
                              message=f"{type(e).__name__}: {e} (CADC registry/host "
                                      f"outages are transient -- retry later)")

    if not products:
        return ProviderResult(provider='cfht', status=STATUS_NO_COVERAGE,
                              message=f"no usable {COLLECTION} stacks here")
    return products
