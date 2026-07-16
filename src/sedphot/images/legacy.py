"""
legacy.py

Legacy Surveys Image Provider
---------------------------------------------------------
Two routes to Legacy Surveys pixels:

    cutout (default)  the legacysurvey.org fits-cutout service -- fast,
                      sized to the request, nanomaggy units, but serves NO
                      inverse variance (errors fall back to sky rms).
    bricks            the NERSC brick coadds -- full 0.25 deg bricks,
                      image + invvar per band, tens of MB per file. The
                      brick is resolved from the Tractor catalog at the
                      target position. Because the cutout service serves
                      no inverse variance, bricks are the only route to
                      real per-pixel noise.

Data products (cached in cache_dir, the target's Photometry/Legacy/):
    legacy_<layer>_<band>.fits                   viewer cutout plane (nmgy)
    legacysurvey-<brick>-image-<band>.fits.fz    brick coadd image
    legacysurvey-<brick>-invvar-<band>.fits.fz   brick inverse variance

Requirements:
    requests, astropy; astroquery (brick route only)

Notes:
    The viewer layer and coadd hemisphere follow the data release and the
    Dec 32.375 deg north/south boundary; on a miss the other hemisphere is
    tried (the overlap strip exists in both). A cutout that comes back
    blank (all-zero) is treated as no coverage for that layer.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import requests
import astropy.units as u  # noqa: F401  (kept for SkyCoord callers)
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from ..results import STATUS_ERROR, STATUS_NO_COVERAGE, ImageProduct, ProviderResult
from ..retry import retry_transient

# ------------------------------------
# Constants
# ------------------------------------
VIEWER_URL = "https://www.legacysurvey.org/viewer/fits-cutout"
COADD_BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey"
TAP_URL = "https://datalab.noirlab.edu/tap"

PIXSCALE = 0.262           # native arcsec/pixel
SEEING = 1.2               # typical arcsec, for detection kernels
NORTH_SOUTH_DEC = 32.375   # survey hemisphere boundary

VIEWER_LAYERS = {
    ('dr9', 'north'): 'ls-dr9-north',
    ('dr9', 'south'): 'ls-dr9',
    ('dr10', 'south'): 'ls-dr10',
}
DEFAULT_BANDS = {'dr9': ('g', 'r', 'z'), 'dr10': ('g', 'r', 'i', 'z')}

WAVE_UM = {'g': 0.475, 'r': 0.625, 'i': 0.755, 'z': 0.920}


def _hemisphere(coord: SkyCoord, dr: str) -> str:
    if dr == 'dr10':
        return 'south'
    return 'north' if coord.dec.deg >= NORTH_SOUTH_DEC else 'south'


# ------------------------------------
# Cutout route
# ------------------------------------
def _fetch_cutouts(coord: SkyCoord, bands: tuple, size_arcsec: float,
                   cache_dir: Path, dr: str) -> list[ImageProduct]:
    """Viewer fits-cutout: one multi-band cube request, split per band."""
    size_px = int(round(size_arcsec / PIXSCALE))
    hemis = _hemisphere(coord, dr)
    layers = [VIEWER_LAYERS.get((dr, hemis))]
    other = 'south' if hemis == 'north' else 'north'
    if VIEWER_LAYERS.get((dr, other)):
        layers.append(VIEWER_LAYERS[(dr, other)])

    products: list[ImageProduct] = []
    for layer in layers:
        paths = [cache_dir / f"legacy_{layer}_{band}.fits" for band in bands]
        if all(p.exists() for p in paths):
            cube_data = None      # cached; skip the request
        else:
            url = (f"{VIEWER_URL}?ra={coord.ra.deg:.8f}&dec={coord.dec.deg:.8f}"
                   f"&layer={layer}&pixscale={PIXSCALE}&bands={''.join(bands)}"
                   f"&size={size_px}")
            response = retry_transient(
                lambda: requests.get(url, timeout=300), f"Legacy cutout {layer}")
            response.raise_for_status()
            cube = fits.open(io.BytesIO(response.content))[0]
            if cube.data is None or not float(abs(cube.data).sum()) > 0:
                print(f"  [Legacy] {layer}: blank cutout (outside coverage)")
                continue
            cube_data = cube.data
            wcs2d = WCS(cube.header).celestial
            for i, band in enumerate(bands):
                header = wcs2d.to_header()
                header["BUNIT"] = "nanomaggy"
                header["SURVEY"] = f"Legacy_{layer}"
                header["FILTER"] = band
                plane = cube_data[i] if cube_data.ndim == 3 else cube_data
                fits.writeto(paths[bands.index(band)], plane.astype("f4"),
                             header, overwrite=True)
        for band, path in zip(bands, paths):
            if path.exists():
                products.append(ImageProduct(
                    provider='legacy', instrument='Legacy', band=band,
                    path=str(path), calib='nmgy', seeing_arcsec=SEEING,
                    wave_um=WAVE_UM.get(band, float('nan'))))
        if products:
            break
    return products


# ------------------------------------
# Brick route
# ------------------------------------
def _resolve_brick(coord: SkyCoord, dr: str) -> tuple[str, str] | None:
    """(brickname, hemisphere) at the target position, from the Tractor catalog."""
    from astroquery.utils.tap.core import TapPlus
    table = {'dr9': 'ls_dr9.tractor', 'dr10': 'ls_dr10.tractor'}[dr]
    query = f"""
    SELECT TOP 1 brickname FROM {table}
    WHERE 't' = q3c_radial_query(ra, dec, {coord.ra.deg:.8f}, {coord.dec.deg:.8f},
                                 {60.0 / 3600.0:.8f})
    """
    from ..retry import retry_transient

    def _run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return TapPlus(url=TAP_URL).launch_job(query).get_results()

    try:
        result = retry_transient(_run, "Legacy brick TAP")
    except Exception as e:
        print(f"  [Legacy] brick resolution failed after retries: {e}")
        return None
    if len(result) == 0:
        return None
    brickname = str(result[0]['brickname'])
    return brickname, _hemisphere(coord, dr)


def _fetch_bricks(coord: SkyCoord, bands: tuple, cache_dir: Path,
                  dr: str) -> list[ImageProduct]:
    """NERSC brick coadds: image + invvar per band, cached, hemisphere fallback."""
    resolved = _resolve_brick(coord, dr)
    if resolved is None:
        return []
    brick, hemis = resolved
    products: list[ImageProduct] = []
    for band in bands:
        band_paths = {}
        for kind in ("image", "invvar"):
            # The release belongs in the cache name: brick names are
            # shared across releases, and an untagged cache would let a
            # dr switch silently reuse the other release's pixels.
            path = cache_dir / f"legacysurvey-{dr}-{brick}-{kind}-{band}.fits.fz"
            untagged = cache_dir / f"legacysurvey-{brick}-{kind}-{band}.fits.fz"
            if not path.exists() and untagged.exists():
                print(f"  [Legacy] using untagged brick cache "
                      f"{untagged.name} (assumed {dr})")
                path = untagged
            if not path.exists():
                fetched = False
                for hemi_try in (hemis, 'south' if hemis == 'north' else 'north'):
                    url = (f"{COADD_BASE}/{dr}/{hemi_try}/coadd/{brick[:3]}/{brick}/"
                           f"legacysurvey-{brick}-{kind}-{band}.fits.fz")
                    print(f"  [Legacy] downloading {kind}-{band} ({brick}, {hemi_try})")
                    response = requests.get(url, timeout=600)
                    if response.status_code == 200:
                        path.write_bytes(response.content)
                        fetched = True
                        break
                if not fetched:
                    print(f"  [Legacy] no {kind}-{band} coadd for {brick}")
                    break
            band_paths[kind] = path
        if len(band_paths) == 2:
            products.append(ImageProduct(
                provider='legacy', instrument='Legacy', band=band,
                path=str(band_paths['image']), calib='nmgy',
                invvar_path=str(band_paths['invvar']),
                seeing_arcsec=SEEING, wave_um=WAVE_UM.get(band, float('nan'))))
    return products


# ------------------------------------
# Provider entry
# ------------------------------------
def fetch(coord: SkyCoord, *, bands: tuple | None = None, size_arcsec: float = 120.0,
          cache_dir: str | Path, dr: str = 'dr9',
          use_bricks: bool = False) -> list[ImageProduct] | ProviderResult:
    """Fetch Legacy Surveys images at the target.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    bands : tuple, optional
        Bands to fetch. [default: grz for dr9, griz for dr10]
    size_arcsec : float
        Cutout width (cutout route only). [default: 120]
    cache_dir : str or Path
        Photometry/Legacy/ directory; downloads are cached here.
    dr : str
        'dr9' or 'dr10'. [default: 'dr9']
    use_bricks : bool
        Fetch NERSC brick coadds (image + inverse variance) instead of
        viewer cutouts -- real per-pixel errors at ~40 MB per file.
        [default: False]

    Returns
    -------
    products or result : list[ImageProduct] | ProviderResult
        Image products on success; a no_coverage/error result otherwise.
    """
    bands = tuple(bands) if bands else DEFAULT_BANDS[dr]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        if use_bricks:
            products = _fetch_bricks(coord, bands, cache_dir, dr)
        else:
            products = _fetch_cutouts(coord, bands, size_arcsec, cache_dir, dr)
    except Exception as e:
        return ProviderResult(provider='legacy', status=STATUS_ERROR,
                              message=f"{type(e).__name__}: {e}")
    if not products:
        return ProviderResult(provider='legacy', status=STATUS_NO_COVERAGE,
                              message=f"no Legacy {dr} imaging at this position")
    return products
