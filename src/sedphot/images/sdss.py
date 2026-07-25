"""
sdss.py

SDSS Frame Image Provider
---------------------------------------------------------
Calibrated SDSS frames via astroquery.sdss.get_images. Frame pixels are
nanomaggies (the 'nmgy' calib key). The frame overlapping the target is
saved per band; frames are 13.5 x 9.9 arcmin, so a target near a frame edge
may sit closer to the boundary than the requested stamp half-size -- the
Cutout2D in the measurement engine handles partial stamps.

Data products (cached in cache_dir, the target's Photometry/SDSS/):
    sdss_<band>_frame.fits    full calibrated frame, one file per band

Requirements:
    astropy, astroquery

Notes:
    SDSS frames are shallow relative to the other imaging this package
    fetches; expect larger sky-limited errors, especially in u and z.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

from ..results import STATUS_ERROR, STATUS_NO_COVERAGE, ImageProduct, ProviderResult
from ..retry import retry_transient

# ------------------------------------
# Constants
# ------------------------------------
SEEING = 1.4
DEFAULT_BANDS = ('u', 'g', 'r', 'i', 'z')
WAVE_UM = {'u': 0.355, 'g': 0.475, 'r': 0.622, 'i': 0.763, 'z': 0.905}


# ------------------------------------
# Provider entry
# ------------------------------------
def fetch(coord: SkyCoord, *, bands: tuple | None = None, size_arcsec: float = 120.0,
          cache_dir: str | Path) -> list[ImageProduct] | ProviderResult:
    """Fetch SDSS frames covering the target.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    bands : tuple, optional
        Subset of ugriz. [default: all five]
    size_arcsec : float
        Unused (full frames are saved); kept for interface uniformity.
    cache_dir : str or Path
        Photometry/SDSS/ directory; downloads are cached here.

    Returns
    -------
    products or result : list[ImageProduct] | ProviderResult
        Image products on success; a no_coverage/error result otherwise.
    """
    from astroquery.sdss import SDSS

    bands = tuple(bands) if bands else DEFAULT_BANDS
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the frame covering the target ONCE via a region query, then
    # fetch each band from it by (run, rerun, camcol, field). Fetching by
    # coordinates -- SDSS.get_images(coordinates=...) -- is unreliable: its
    # internal field lookup can return a malformed or short table for some
    # positions, surfacing as InconsistentTableError or KeyError 'run' and
    # failing EVERY band even where imaging plainly exists (query_region at
    # the same spot returns objects fine). query_region hands back the frame
    # identifiers directly, and matches= fetches from them without that
    # lookup. Retry the transport itself for genuine service flaps.
    frame = None
    need = [b for b in bands
            if not (cache_dir / f"sdss_{b}_frame.fits").exists()]
    if need:
        try:
            xid = retry_transient(
                lambda: SDSS.query_region(coord, radius=30 * u.arcsec,
                                          spectro=False),
                "SDSS region")
        except Exception as e:
            return ProviderResult(provider='sdss', status=STATUS_ERROR,
                                  message=f"region query: "
                                          f"{type(e).__name__}: {e}")
        if xid is None or not len(xid):
            return ProviderResult(provider='sdss', status=STATUS_NO_COVERAGE,
                                  message="no SDSS objects at this position")
        # The object nearest the target sits on the frame that contains it
        # (frames are 13.5'x9.9', far larger than the 30" search cone).
        seps = coord.separation(SkyCoord(xid['ra'], xid['dec'], unit='deg'))
        i = int(seps.argmin())
        frame = xid['run', 'rerun', 'camcol', 'field'][i:i + 1]

    products: list[ImageProduct] = []
    failures: list[str] = []
    for band in bands:
        path = cache_dir / f"sdss_{band}_frame.fits"
        # Per-band containment: one bad band must not take the others down.
        try:
            if not path.exists():
                def _get_frame(band=band, frame=frame):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return SDSS.get_images(matches=frame, band=band)
                images = retry_transient(_get_frame, f"SDSS {band}")
                if not images:
                    print(f"  [SDSS] no {band} frame here")
                    continue
                hdu = images[0][0]
                fits.writeto(path, hdu.data, hdu.header, overwrite=True)
        except Exception as e:
            print(f"  [SDSS] {band} fetch failed: {type(e).__name__}: {e}")
            failures.append(f"{band}: {type(e).__name__}: {e}")
            continue
        products.append(ImageProduct(
            provider='sdss', instrument='SDSS', band=band,
            path=str(path), calib='nmgy', seeing_arcsec=SEEING,
            wave_um=WAVE_UM.get(band, float('nan'))))

    if not products:
        if failures:
            return ProviderResult(provider='sdss', status=STATUS_ERROR,
                                  message="; ".join(failures))
        return ProviderResult(provider='sdss', status=STATUS_NO_COVERAGE,
                              message="no SDSS frames at this position")
    return products
