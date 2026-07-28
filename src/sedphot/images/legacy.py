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
                      target position.

Data products (cached in cache_dir, the target's Photometry/Legacy/):
    legacy_<layer>_<band>.fits                   viewer cutout plane (nmgy)
    legacysurvey-<brick>-image-<band>.fits.fz    brick coadd image
    legacysurvey-<brick>-invvar-<band>.fits.fz   brick inverse variance

Requirements:
    requests, numpy, astropy, astroquery

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
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from ..catalogs.legacy import LEGACY_DR_DEFAULT
from ..results import STATUS_ERROR, STATUS_NO_COVERAGE, ImageProduct, ProviderResult
from ..retry import retry_transient, try_services
from .common import warn_undersized_cache

# ------------------------------------
# Constants
# ------------------------------------
VIEWER_URL = "https://www.legacysurvey.org/viewer/fits-cutout"
COADD_BASE = "https://portal.nersc.gov/cfs/cosmo/data/legacysurvey"
TAP_URL = "https://datalab.noirlab.edu/tap"
# NOIRLab Data Lab cutout service: serves the SAME DR coadd pixels as the
# viewer, from a host independent of NERSC -- the cutout fallback when
# legacysurvey.org and portal.nersc.gov are down. The brick is resolved from
# the Data Lab TAP catalog (also NERSC-independent), then each band is cut
# from legacysurvey-<brick>-image-<band>.fits.fz.
NOIRLAB_CUTOUT_URL = "https://datalab.noirlab.edu/svc/cutout"

# A dead host hangs rather than refuses, so the viewer request fails fast on
# connect (letting the service rotation reach NOIRLab) but stays patient on
# read for a slow-but-live viewer.
CUTOUT_TIMEOUT = (6, 180)

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
            for path in paths:
                warn_undersized_cache(path, size_arcsec, 'Legacy')
        else:
            url = (f"{VIEWER_URL}?ra={coord.ra.deg:.8f}&dec={coord.dec.deg:.8f}"
                   f"&layer={layer}&pixscale={PIXSCALE}&bands={''.join(bands)}"
                   f"&size={size_px}")
            # One attempt, fail fast: try_services rotates to the NOIRLab
            # fallback on failure, so a dead viewer must not burn a backoff
            # cycle here first.
            response = requests.get(url, timeout=CUTOUT_TIMEOUT)
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
# NOIRLab SIA route (NERSC-independent cutout fallback)
# ------------------------------------
def _fetch_noirlab(coord: SkyCoord, bands: tuple, size_arcsec: float,
                   cache_dir: Path, dr: str) -> list[ImageProduct]:
    """NOIRLab Data Lab cutouts: the same DR coadd pixels as the viewer.

    Served from datalab.noirlab.edu, so a drop-in fallback when NERSC is down.
    Same nanomaggy calibration and 0.262"/pixel scale, no inverse variance
    (errors fall back to sky rms, exactly as the viewer route). Files take the
    viewer-route name so downstream filename provenance stays uniform; the real
    source is stamped in the header (FETCHSRC = noirlab-cutout).

    The brick is resolved from the Data Lab TAP catalog (strict, so a TAP
    outage raises and the whole rotation reports an error rather than a silent
    no-coverage), then each band is cut directly from its coadd. Cutting
    directly -- rather than trusting the SIA product list -- also recovers the
    MzLS z coadd, which the north SIA collection does not index.
    """
    hemis = _hemisphere(coord, dr)
    layer = VIEWER_LAYERS.get((dr, hemis)) or f"ls-{dr}"
    size_deg = size_arcsec / 3600.0
    paths = {band: cache_dir / f"legacy_{layer}_{band}.fits" for band in bands}

    brick = None
    if any(not p.exists() for p in paths.values()):
        resolved = _resolve_brick(coord, dr, strict=True)
        if resolved is None:
            return []      # no brick here -> genuine no coverage
        brick, _ = resolved

    products: list[ImageProduct] = []
    for band in bands:
        path = paths[band]
        if not path.exists():
            # A RAW comma in POS is essential: urlencoding it to %2C makes the
            # cutout service ignore POS and return the whole ~50 MB brick.
            url = (f"{NOIRLAB_CUTOUT_URL}?col=ls_{dr}"
                   f"&siaRef=legacysurvey-{brick}-image-{band}.fits.fz&extn=1"
                   f"&POS={coord.ra.deg:.8f},{coord.dec.deg:.8f}"
                   f"&SIZE={size_deg:.6f}")
            try:
                response = requests.get(url, timeout=CUTOUT_TIMEOUT)
                response.raise_for_status()
                hdul = fits.open(io.BytesIO(response.content))
                hdu = next((h for h in hdul if h.data is not None), None)
            except Exception as e:
                # A single absent/failed band must not drop the others; the
                # brick already resolved, so this is not a full outage.
                print(f"  [Legacy] NOIRLab {band}: {type(e).__name__}: {e}")
                continue
            if hdu is None or not float(abs(hdu.data).sum()) > 0:
                print(f"  [Legacy] NOIRLab {band}: blank cutout")
                continue
            header = WCS(hdu.header).celestial.to_header()
            header["BUNIT"] = "nanomaggy"
            header["SURVEY"] = f"Legacy_{layer}"
            header["FILTER"] = band
            header["FETCHSRC"] = "noirlab-cutout"
            fits.writeto(path, hdu.data.astype("f4"), header, overwrite=True)
        if path.exists():
            products.append(ImageProduct(
                provider='legacy', instrument='Legacy', band=band,
                path=str(path), calib='nmgy', seeing_arcsec=SEEING,
                wave_um=WAVE_UM.get(band, float('nan'))))
    return products


# ------------------------------------
# Brick route
# ------------------------------------
def _resolve_brick(coord: SkyCoord, dr: str,
                   strict: bool = False) -> tuple[str, str] | None:
    """(brickname, hemisphere) at the target position, from the Tractor catalog.

    The CLOSEST brick-primary source names the brick: bricks overlap by
    ~20 arcsec, so near a tile boundary an unordered match can return a
    neighboring source whose primary brick does not even contain the
    target's pixels. The closest source is (essentially) the target
    itself, and its primary brick is the tile that owns this sky
    position. Selection happens CLIENT-SIDE: the Datalab TAP service
    rejects ORDER BY q3c_dist, so the query returns the cone's rows and
    the closest is picked here.
    """
    import numpy as np
    from astroquery.utils.tap.core import TapPlus
    table = {'dr9': 'ls_dr9.tractor', 'dr10': 'ls_dr10.tractor'}[dr]
    query = f"""
    SELECT brickname, ra, dec FROM {table}
    WHERE brick_primary = 1
      AND 't' = q3c_radial_query(ra, dec, {coord.ra.deg:.8f}, {coord.dec.deg:.8f},
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
        if strict:
            raise      # let a TAP outage surface as an error, not no-coverage
        return None
    if len(result) == 0:
        return None
    sources = SkyCoord(np.asarray(result['ra'], dtype=float),
                       np.asarray(result['dec'], dtype=float), unit='deg')
    closest = int(coord.separation(sources).argmin())
    brickname = str(result[closest]['brickname'])
    return brickname, _hemisphere(coord, dr)


def _fetch_bricks(coord: SkyCoord, bands: tuple, cache_dir: Path,
                  dr: str) -> list[ImageProduct]:
    """NERSC brick coadds: image + invvar per band, cached, hemisphere fallback."""
    # Cache-first brick identity: the brickname is already encoded in
    # any cached coadd's filename, so once the files exist a re-measure
    # skips the TAP resolution entirely and is unaffected by its outages.
    cached = sorted(cache_dir.glob(f'legacysurvey-{dr}-*-image-*.fits.fz')) \
        or sorted(cache_dir.glob('legacysurvey-*-image-*.fits.fz'))
    if cached:
        brick = cached[0].name.replace(f'legacysurvey-{dr}-', '') \
            .replace('legacysurvey-', '').split('-image-')[0]
        hemis = _hemisphere(coord, dr)
        print(f"  [Legacy] brick {brick} from cached coadds; "
              f"skipping the TAP resolution")
    else:
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
          cache_dir: str | Path, dr: str = LEGACY_DR_DEFAULT,
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
        'dr9' or 'dr10'. [default: LEGACY_DR_DEFAULT]
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
            # Rotate cutout services: the NERSC viewer first, then the
            # NOIRLab SIA (same DR pixels, independent host). Alternating on
            # failure reaches the live fallback without waiting out a dead
            # primary's backoff.
            products = try_services([
                ('NERSC viewer', lambda: _fetch_cutouts(
                    coord, bands, size_arcsec, cache_dir, dr)),
                ('NOIRLab cutout', lambda: _fetch_noirlab(
                    coord, bands, size_arcsec, cache_dir, dr)),
            ], 'Legacy cutout')
    except Exception as e:
        return ProviderResult(provider='legacy', status=STATUS_ERROR,
                              message=f"{type(e).__name__}: {e}")
    if not products:
        return ProviderResult(provider='legacy', status=STATUS_NO_COVERAGE,
                              message=f"no Legacy {dr} imaging at this position")
    return products
