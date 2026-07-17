"""
panstarrs.py

Pan-STARRS Image Provider
---------------------------------------------------------
PS1 stack cutouts via the ps1filenames.py listing service plus fitscut.cgi.
Stack pixels are linear DN/s with ZP = 25 + 2.5 log10(EXPTIME) -- the 'ps1'
calib key; EXPTIME rides in the returned header.

Data products (cached in cache_dir, the target's Photometry/PanSTARRS/):
    ps1_<band>.fits    stack cutout, one file per band

Requirements:
    requests, astropy

Notes:
    Coverage is Dec > -30. fitscut occasionally serves a cutout with NaN
    padding at survey edges; the measurement engine's finite-pixel masks
    handle that.
"""
from __future__ import annotations

import io
from pathlib import Path

import requests
from astropy.coordinates import SkyCoord
from astropy.io import fits

from ..results import STATUS_ERROR, STATUS_NO_COVERAGE, ImageProduct, ProviderResult
from ..retry import retry_transient
from .common import warn_undersized_cache

# ------------------------------------
# Constants
# ------------------------------------
PS1_FILENAMES = "https://ps1images.stsci.edu/cgi-bin/ps1filenames.py"
PS1_FITSCUT = "https://ps1images.stsci.edu/cgi-bin/fitscut.cgi"

PIXSCALE = 0.25
SEEING = 1.1
DEFAULT_BANDS = ('g', 'r', 'i', 'z', 'y')
WAVE_UM = {'g': 0.481, 'r': 0.617, 'i': 0.752, 'z': 0.866, 'y': 0.962}


# ------------------------------------
# Provider entry
# ------------------------------------
def fetch(coord: SkyCoord, *, bands: tuple | None = None, size_arcsec: float = 120.0,
          cache_dir: str | Path) -> list[ImageProduct] | ProviderResult:
    """Fetch PS1 stack cutouts at the target.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    bands : tuple, optional
        Subset of grizy. [default: all five]
    size_arcsec : float
        Cutout width. [default: 120]
    cache_dir : str or Path
        Photometry/PanSTARRS/ directory; downloads are cached here.

    Returns
    -------
    products or result : list[ImageProduct] | ProviderResult
        Image products on success; a no_coverage/error result otherwise.
    """
    bands = tuple(bands) if bands else DEFAULT_BANDS
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    size_px = int(round(size_arcsec / PIXSCALE))

    try:
        listing = retry_transient(
            lambda: requests.get(
                PS1_FILENAMES,
                params=dict(ra=f"{coord.ra.deg:.8f}", dec=f"{coord.dec.deg:.8f}",
                            filters=''.join(bands), type="stack"),
                timeout=90,
            ),
            "PS1 listing",
        )
        listing.raise_for_status()
        lines = listing.text.strip().split("\n")
        if len(lines) < 2:
            return ProviderResult(provider='panstarrs', status=STATUS_NO_COVERAGE,
                                  message="no PS1 stack at this position "
                                          "(Dec < -30 is outside the footprint)")
        header_cols = lines[0].split()
        fcol, bcol = header_cols.index("filename"), header_cols.index("filter")
        files = {row.split()[bcol]: row.split()[fcol] for row in lines[1:]}

        products: list[ImageProduct] = []
        for band in bands:
            if band not in files:
                print(f"  [PS1] no {band} stack here")
                continue
            path = cache_dir / f"ps1_{band}.fits"
            if path.exists():
                warn_undersized_cache(path, size_arcsec, 'PS1')
            else:
                url = (f"{PS1_FITSCUT}?ra={coord.ra.deg:.8f}&dec={coord.dec.deg:.8f}"
                       f"&size={size_px}&format=fits&red={files[band]}")
                response = retry_transient(
                    lambda: requests.get(url, timeout=300), f"PS1 fitscut {band}")
                response.raise_for_status()
                hdu = fits.open(io.BytesIO(response.content))[0]
                hdu.header["SURVEY"] = "PS1"
                hdu.header["FILTER"] = band
                fits.writeto(path, hdu.data.astype("f4"), hdu.header, overwrite=True)
            products.append(ImageProduct(
                provider='panstarrs', instrument='PS1', band=band,
                path=str(path), calib='ps1', seeing_arcsec=SEEING,
                wave_um=WAVE_UM.get(band, float('nan'))))
    except Exception as e:
        return ProviderResult(provider='panstarrs', status=STATUS_ERROR,
                              message=f"{type(e).__name__}: {e}")

    if not products:
        return ProviderResult(provider='panstarrs', status=STATUS_NO_COVERAGE,
                              message="no PS1 stacks at this position")
    return products
