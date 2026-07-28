"""
cfht.py

CFHT MegaPipe Image Provider
---------------------------------------------------------
MegaPipe deep-stack cutouts via the CADC SODA service:

    Cadc().query_region(coord, collection='CFHTMEGAPIPE')
        -> filter to calibrated image planes
        -> get_image_list(...) SODA cutout URLs
        -> download per band

Data products (cached in cache_dir, the target's Photometry/CFHT/):
    cfht_megapipe_<band>.fits    SODA stack cutout (renamed *.nophotzp.fits
                                 and skipped when PHOTZP is absent; the
                                 renamed file stays part of the cache, so a
                                 rejected stack is not fetched again)

Requirements:
    requests, numpy, astropy, astroquery; reproject for the tile-edge mosaic

Notes:
    - The deep optical stacks are collection CFHTMEGAPIPE; 'CFHTLS' returns
      ZERO planes at positions the MegaPipe stacks cover.
    - astroquery.cadc resolves its endpoints through the CADC registry
      (argus), so a registry 503 blocks everything -- transient, retry later.
    - The data host has transient failures (404/503/timeout on valid URLs);
      transport retries wrap every call.
    - ugriz stacks carry PHOTZP = 30.0 (AB); the 'photzp' calib key reads
      it from the header.
    - HiPS pixels are display-normalized and unusable for photometry; only
      SODA cutouts of the calibrated stacks are fetched.
    - Band identity is parsed from the MegaCam filter name in the plane
      metadata (u.MP9302 -> u). WIRCam products are not handled here.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import requests
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from ..results import STATUS_ERROR, STATUS_NO_COVERAGE, ImageProduct, ProviderResult
from ..retry import retry_transient
from .common import mosaic_first_valid, warn_undersized_cache

# ------------------------------------
# Constants
# ------------------------------------
COLLECTION = "CFHTMEGAPIPE"
SEEING = 0.8
DEFAULT_BANDS = ('u', 'g', 'r', 'i', 'z')
WAVE_UM = {'u': 0.355, 'g': 0.475, 'r': 0.640, 'i': 0.776, 'z': 0.925}

# MegaCam filter names look like 'u.MP9301', 'r.MP9601', 'gri.MP9605' (the
# last is a rare combined filter -- excluded by the single-letter rule).
_FILTER_RE = re.compile(r"^([ugriz])\.MP\d+$", re.IGNORECASE)


def _band_of(filter_name: str) -> str | None:
    match = _FILTER_RE.match(str(filter_name).strip())
    return match.group(1).lower() if match else None


def _footprint_center_offset(row, coord: SkyCoord) -> float:
    """Angular distance (deg) from a plane's footprint centroid to the target.

    A target near a stack's edge yields a half-blank SODA cutout, so ordering
    candidate stacks by this offset puts the stack that best centers the target
    (largest margin to its boundary) first. Returns a large value on parse
    failure so unparseable planes sort last.
    """
    try:
        samples = np.asarray(row['position_bounds_samples'], dtype=float).ravel()
        samples = samples[np.isfinite(samples)]
        if samples.size < 2:
            return 1e9
        ra = samples[0::2].mean()
        dec = samples[1::2].mean()
        return float(SkyCoord(ra * u.deg, dec * u.deg).separation(coord).deg)
    except Exception:
        return 1e9


def _soda_nodata(data: np.ndarray) -> np.ndarray:
    """SODA off-footprint fill as a nodata mask, matching the measurement gate.

    SODA fills a clipped region with exact zeros AND deeply-negative values --
    the latter finite and non-zero, so an isfinite/non-zero test counts them as
    real data (and the mosaic's first-valid combine would keep them over an
    adjacent tile's good pixels). Same RULE as the measurement gate
    (stamp.load_stamp): non-finite, exact zero, or more than 10 robust sigma
    below the outer level.

    The level here is a median + MAD where the gate uses a clipped mean and
    its clipped sigma. Deliberate, and the one place that difference is safe:
    this is a DETECTION threshold at -10 sigma, where the two estimators agree
    far more closely than the threshold's own margin, and no flux is summed
    from it. Where a level feeds photometry the clipped mean is mandatory --
    see background.py on why a median biases a sum.
    """
    nodata = ~np.isfinite(data) | (data == 0.0)
    good = data[~nodata]
    if good.size >= 100:
        level = float(np.median(good))
        sigma = 1.4826 * float(np.median(np.abs(good - level))) or 1e-30
        nodata |= (data - level) < -10.0 * sigma
    return nodata


def _covers_target(content: bytes, coord: SkyCoord,
                   aperture_arcsec: float = 12.0) -> bool:
    """Does the cutout cover the science aperture with real data?

    Aligned with the measurement coverage gate (COVERAGE_MIN): at least 95% of
    the aperture-radius disk around the target must be finite and non-zero. A
    stack that clips the aperture -- one the gate would demote -- fails here
    too, so the mosaic fallback fires rather than the clip being accepted as a
    single stack only to fail downstream. A target whose aperture runs off the
    cutout edge (true survey boundary) also fails, and CFHT is skipped.
    """
    try:
        with fits.open(io.BytesIO(content)) as hdul:
            hdu = next(h for h in hdul
                       if h.data is not None and h.data.ndim == 2)
            data = hdu.data
            wcs = WCS(hdu.header)
        x, y = (float(v) for v in wcs.world_to_pixel(coord))
        scale = float(np.mean(proj_plane_pixel_scales(wcs))) * 3600.0
        r = aperture_arcsec / scale
        ny, nx = data.shape
        if not (r <= x < nx - r and r <= y < ny - r):
            return False       # aperture runs off the cutout edge
        yy, xx = np.ogrid[:ny, :nx]
        aper = (xx - x) ** 2 + (yy - y) ** 2 < r ** 2
        good = ~_soda_nodata(data) & aper
        return bool(good.sum() / max(int(aper.sum()), 1) > 0.95)
    except Exception:
        return False


def _mosaic_clipping_stacks(cutouts: list[bytes], coord: SkyCoord,
                            size_arcsec: float) -> bytes | None:
    """First-valid mosaic of SODA cutouts that each clip the target.

    A target on a MegaPipe tile boundary is covered by no single stack but by
    several overlapping ones; combine them so an adjacent tile fills the
    clipped side. PHOTZP is carried from a source stack (all MegaPipe stacks
    share it) so the mosaic stays photometric.
    """
    planes, photzp = [], None
    for content in cutouts:
        try:
            with fits.open(io.BytesIO(content)) as hdul:
                hdu = next(h for h in hdul
                           if h.data is not None and h.data.ndim == 2)
                plane = np.where(_soda_nodata(hdu.data), np.nan,
                                 hdu.data).astype('f4')
                planes.append((plane, WCS(hdu.header)))
                if photzp is None and 'PHOTZP' in hdu.header:
                    photzp = hdu.header['PHOTZP']
        except Exception:
            continue
    if len(planes) < 2 or photzp is None:
        return None
    pixscale = float(np.mean(proj_plane_pixel_scales(planes[0][1]))) * 3600.0
    data, wcs = mosaic_first_valid(planes, coord, size_arcsec, pixscale)
    if data is None:
        return None
    header = wcs.to_header()
    header['PHOTZP'] = photzp
    buf = io.BytesIO()
    fits.PrimaryHDU(data=data, header=header).writeto(buf)
    return buf.getvalue()


# ------------------------------------
# Provider entry
# ------------------------------------
def fetch(coord: SkyCoord, *, bands: tuple | None = None, size_arcsec: float = 120.0,
          cache_dir: str | Path,
          aperture_arcsec: float = 12.0) -> list[ImageProduct] | ProviderResult:
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
    aperture_arcsec : float
        Science aperture the stack coverage is judged against; a stack that
        clips inside it fails and drives the tile-edge mosaic, matching the
        measurement coverage gate at the same aperture. The pipeline passes
        the run's own value; the default matches the CLI's. [default: 12]

    Returns
    -------
    products or result : list[ImageProduct] | ProviderResult
        Image products on success; a no_coverage/error result otherwise.
    """
    import astropy.units as u
    from astroquery.cadc import Cadc

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Cache-complete short circuit: when every requested band's stack
    # is already on disk (with no request, every band in DEFAULT_BANDS),
    # the CADC round trip's only product would be a band list the
    # cached filenames already encode -- and a CADC outage must not
    # stall a fully offline re-measure. The unrequested case is judged
    # against DEFAULT_BANDS rather than the cache itself: a list read
    # off the cache satisfies the test by construction, so a band that
    # failed on an earlier run would never be fetched again.
    cached = {p.name.split('_')[-1].split('.')[0]: p
              for p in sorted(cache_dir.glob('cfht_megapipe_?.fits'))}
    # A stack rejected for a missing PHOTZP is settled as well: refetching
    # it would only reject it again, so it counts as answered here -- and,
    # being uncalibrated, it yields no product.
    rejected = {p.name.split('_')[-1].split('.')[0]: p for p in
                sorted(cache_dir.glob('cfht_megapipe_?.nophotzp.fits'))}
    wanted_now = tuple(bands) if bands else DEFAULT_BANDS
    usable = [band for band in wanted_now if band in cached]

    def _cached_products() -> list[ImageProduct]:
        """The already-cached bands as products, no service call."""
        return [ImageProduct(
            provider='cfht', instrument='CFHT', band=band,
            path=str(cached[band]), calib='photzp',
            seeing_arcsec=SEEING,
            wave_um=WAVE_UM.get(band, float('nan')))
            for band in usable]

    if usable and all(band in cached or band in rejected
                      for band in wanted_now):
        for band in usable:
            warn_undersized_cache(cached[band], size_arcsec, 'CFHT')
        print(f"  [CFHT] cached stacks cover "
              f"{''.join(usable)}; skipping the CADC query")
        return _cached_products()

    radius = (size_arcsec / 2.0) * u.arcsec
    products: list[ImageProduct] = []

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

        for band in wanted:
            rows = per_band.get(band)
            if not rows:
                print(f"  [CFHT] no {band} stack here")
                continue
            path = cache_dir / f"cfht_megapipe_{band}.fits"
            no_photzp = path.with_suffix(".nophotzp.fits")
            if path.exists():
                warn_undersized_cache(path, size_arcsec, 'CFHT')
            elif no_photzp.exists():
                # An absent PHOTZP is a property of the data product, not a
                # transient failure: the renamed file is the record that this
                # band was already tried and cannot be calibrated, so it is
                # skipped rather than downloaded and rejected again.
                print(f"  [CFHT] {band}: cached stack has no PHOTZP "
                      f"({no_photzp.name}) -- skipping photometry")
                continue
            else:
                # Order candidate stacks by how well their footprint centers the
                # target, then take the first that actually covers the stamp
                # center. SODA returns overlapping stacks in a non-deterministic
                # order, so a fixed urls[0] can grab a stack that clips the target
                # at its edge (half-blank cutout) even when covering stacks exist.
                band_idx = sorted(
                    (i for i, row in enumerate(result)
                     if _band_of(row['energy_bandpassName']) == band),
                    key=lambda i: _footprint_center_offset(result[i], coord))
                subset = result[band_idx]
                urls = retry_transient(
                    lambda: cadc.get_image_list(subset, coord, radius),
                    f"CADC SODA {band}")
                if not urls:
                    print(f"  [CFHT] no cutout URL for {band}")
                    continue
                content = None
                clipped = []
                for url in urls:
                    # One dead candidate must not kill the band (let alone
                    # the provider) -- log it and try the next stack.
                    try:
                        resp = retry_transient(
                            lambda: requests.get(url, timeout=600),
                            f"CADC download {band}")
                        resp.raise_for_status()
                    except Exception as e:
                        print(f"  [CFHT] {band}: candidate stack failed "
                              f"({type(e).__name__}: {e}); trying the next")
                        continue
                    if _covers_target(resp.content, coord, aperture_arcsec):
                        content = resp.content
                        break
                    clipped.append(resp.content)
                if content is None and len(clipped) >= 2:
                    # No single stack covers the aperture -- the target sits
                    # on a tile boundary. Mosaic the overlapping stacks so an
                    # adjacent tile fills the clipped side.
                    mosaic = _mosaic_clipping_stacks(clipped, coord, size_arcsec)
                    if mosaic is not None and _covers_target(mosaic, coord,
                                                             aperture_arcsec):
                        content = mosaic
                        print(f"  [CFHT] {band}: mosaicked {len(clipped)} "
                              f"tile-edge stacks")
                if content is None:
                    print(f"  [CFHT] {band}: all {len(urls)} stack(s) clip the "
                          f"target at their edge (near survey boundary)")
                    continue
                path.write_bytes(content)
                # Sanity: PHOTZP present for photometric use.
                with fits.open(path) as hdul:
                    hdu = hdul[1] if len(hdul) > 1 and hdul[0].data is None else hdul[0]
                    if 'PHOTZP' not in hdu.header:
                        print(f"  [CFHT] WARNING {band}: no PHOTZP in header -- "
                              f"morphology-grade only, skipping photometry; "
                              f"kept as {no_photzp.name} so it is not "
                              f"fetched again")
                        path.rename(no_photzp)
                        continue
            products.append(ImageProduct(
                provider='cfht', instrument='CFHT', band=band,
                path=str(path), calib='photzp', seeing_arcsec=SEEING,
                wave_um=WAVE_UM.get(band, float('nan'))))
    except Exception as e:
        # A position the archive covers only partially never satisfies the
        # short circuit, so it reaches CADC on every run. An outage must
        # still not cost it the stacks already on disk: report those and
        # keep the re-measure offline.
        fallback = products or _cached_products()
        if fallback:
            print(f"  [CFHT] CADC unreachable ({type(e).__name__}: {e}); "
                  f"reporting {len(fallback)} cached stack(s)")
            return fallback
        return ProviderResult(provider='cfht', status=STATUS_ERROR,
                              message=f"{type(e).__name__}: {e} (CADC registry/host "
                                      f"outages are transient -- retry later)")

    if not products:
        return ProviderResult(provider='cfht', status=STATUS_NO_COVERAGE,
                              message=f"no usable {COLLECTION} stacks here")
    return products
