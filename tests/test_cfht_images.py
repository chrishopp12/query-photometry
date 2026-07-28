"""CFHT stack selection: footprint-centroid ordering, coverage, and caching."""
import io

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from sedphot.images import cfht as cfht_images
from sedphot.images.cfht import (COLLECTION, DEFAULT_BANDS, _band_of,
                                 _covers_target, _footprint_center_offset,
                                 _mosaic_clipping_stacks, fetch)

TARGET = SkyCoord(150.0, 30.0, unit='deg')
SCALE_DEG = 0.187 / 3600.0  # MegaPipe pixel scale


def _fits_bytes(data, crval=(150.0, 30.0), crpix=None):
    """Serialize a TAN-projected image the way a SODA cutout arrives."""
    ny, nx = data.shape
    header = fits.Header()
    header['CTYPE1'], header['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
    header['CRVAL1'], header['CRVAL2'] = crval
    # FITS CRPIX is 1-based: these put CRVAL at 0-based pixel (nx//2, ny//2).
    header['CRPIX1'], header['CRPIX2'] = crpix or (nx // 2 + 1, ny // 2 + 1)
    header['CD1_1'], header['CD1_2'] = -SCALE_DEG, 0.0
    header['CD2_1'], header['CD2_2'] = 0.0, SCALE_DEG
    buf = io.BytesIO()
    fits.PrimaryHDU(data=data, header=header).writeto(buf)
    return buf.getvalue()


# ------------------------------------
# _covers_target
# ------------------------------------
def test_full_coverage_passes():
    assert _covers_target(_fits_bytes(np.ones((400, 400), np.float32)), TARGET)


def test_half_blank_cutout_fails():
    # The stack-edge failure mode: SODA returns a full-size array whose
    # far side of the stack boundary is zero-filled, target in the blank.
    data = np.ones((400, 400), np.float32)
    data[:, 150:] = 0.0
    assert not _covers_target(_fits_bytes(data), TARGET)


def test_nan_blank_cutout_fails():
    data = np.ones((400, 400), np.float32)
    data[:, 150:] = np.nan
    assert not _covers_target(_fits_bytes(data), TARGET)


def test_truncated_strip_fails():
    # A stack-boundary cutout arrives as a thin strip: the target sits on
    # real data but closer than pad_arcsec to the array edge, so the
    # aperture cannot fit.
    assert not _covers_target(_fits_bytes(np.ones((60, 400), np.float32)), TARGET)


def test_partial_edge_through_aperture_fails():
    # Blank edge slicing through the pad box but not the target pixel
    # itself -- the failure mode that reported fluxes ~30x low.
    data = np.ones((400, 400), np.float32)
    data[:, 230:] = 0.0  # target at x=200, pad box reaches x=280
    assert not _covers_target(_fits_bytes(data), TARGET)


def test_covers_target_respects_the_requested_aperture():
    # A blank column ~13" right of the target sits inside a 20" aperture but
    # clear of a 12" one: coverage must track the aperture it is asked about,
    # not a hardcoded 12". This is what lets the fetch/mosaic decision match
    # the measurement gate at whatever science aperture the run uses.
    data = np.ones((400, 400), np.float32)
    data[:, 270:] = 0.0                       # target at x=200; 70 px ~= 13"
    assert _covers_target(_fits_bytes(data), TARGET, 12.0)
    assert not _covers_target(_fits_bytes(data), TARGET, 20.0)


def test_garbage_bytes_fail():
    assert not _covers_target(b"this is not a FITS file", TARGET)


# ------------------------------------
# _footprint_center_offset
# ------------------------------------
def _row(center_ra, center_dec, half=0.3):
    """Fake CADC plane row: interleaved footprint corner samples."""
    return {'position_bounds_samples': [
        center_ra - half, center_dec - half,
        center_ra + half, center_dec - half,
        center_ra + half, center_dec + half,
        center_ra - half, center_dec + half,
    ]}


def test_centered_footprint_sorts_first():
    near = _footprint_center_offset(_row(150.0, 30.0), TARGET)
    far = _footprint_center_offset(_row(150.5, 30.0), TARGET)
    assert near < 0.01
    assert near < far


def test_nested_samples_are_flattened():
    row = {'position_bounds_samples': [[149.7, 29.7], [150.3, 29.7],
                                       [150.3, 30.3], [149.7, 30.3]]}
    assert _footprint_center_offset(row, TARGET) < 0.01


def test_unparseable_rows_sort_last():
    for row in ({'position_bounds_samples': []},
                {'position_bounds_samples': None},
                {'position_bounds_samples': [np.nan, np.nan]},
                {}):
        assert _footprint_center_offset(row, TARGET) == 1e9


# ------------------------------------
# _mosaic_clipping_stacks
# ------------------------------------
def _stack_bytes(data, *, photzp=30.0, crval=(150.0, 30.0)):
    """A SODA cutout carrying MegaPipe's photometric zeropoint."""
    ny, nx = data.shape
    header = fits.Header()
    header['CTYPE1'], header['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
    header['CRVAL1'], header['CRVAL2'] = crval
    header['CRPIX1'], header['CRPIX2'] = nx // 2 + 1, ny // 2 + 1
    header['CD1_1'], header['CD1_2'] = -SCALE_DEG, 0.0
    header['CD2_1'], header['CD2_2'] = 0.0, SCALE_DEG
    if photzp is not None:
        header['PHOTZP'] = photzp
    buf = io.BytesIO()
    fits.PrimaryHDU(data=data, header=header).writeto(buf)
    return buf.getvalue()


def _clipped_pair():
    """Two stacks that clip the target from OPPOSITE sides.

    The tile-boundary failure mode: neither stack covers the aperture on its
    own, but together they do.
    """
    left, right = (np.ones((400, 400), np.float32) for _ in range(2))
    left[:, 200:] = 0.0     # SODA zero-fills past the stack boundary
    right[:, :200] = 0.0
    return [_stack_bytes(left), _stack_bytes(right)]


def test_mosaic_fills_each_stacks_clipped_side():
    got = _mosaic_clipping_stacks(_clipped_pair(), TARGET, 60.0)
    assert got is not None
    with fits.open(io.BytesIO(got)) as hdul:
        hdu = next(h for h in hdul if h.data is not None and h.data.ndim == 2)
        # Neither input covered the target's aperture; the mosaic does.
        assert np.isfinite(hdu.data).mean() > 0.95
        assert hdu.header['PHOTZP'] == 30.0    # carried, so it stays photometric
    # ... and the inputs really were each insufficient alone.
    assert not any(_covers_target(c, TARGET) for c in _clipped_pair())


def test_mosaic_needs_two_usable_planes():
    one = [_stack_bytes(np.ones((400, 400), np.float32))]
    assert _mosaic_clipping_stacks(one, TARGET, 60.0) is None
    # An unparseable member does not count toward the two.
    assert _mosaic_clipping_stacks(one + [b'not a fits file'],
                                   TARGET, 60.0) is None


def test_mosaic_refuses_without_a_zeropoint():
    """No PHOTZP anywhere means the mosaic could not be calibrated, and an
    uncalibrated stack must not reach the measurement."""
    planes = [_stack_bytes(d, photzp=None) for d in
              (np.ones((400, 400), np.float32),
               np.ones((400, 400), np.float32))]
    assert _mosaic_clipping_stacks(planes, TARGET, 60.0) is None


# ------------------------------------
# fetch: what the cache is allowed to skip
# ------------------------------------
class _FakeResponse:
    """The two attributes the provider touches on a requests Response."""

    def __init__(self, content):
        self.content = content

    def raise_for_status(self):
        pass


def _plane_table(bands):
    """A CADC query result holding one calibrated MegaPipe plane per band."""
    return Table({
        'dataProductType': ['image'] * len(bands),
        'calibrationLevel': [3] * len(bands),
        'energy_bandpassName': [f"{band}.MP9301" for band in bands],
        'position_bounds_samples': [[149.7, 29.7, 150.3, 29.7,
                                     150.3, 30.3, 149.7, 30.3]] * len(bands),
    })


def _cache_stack(cache_dir, band, **kw):
    """Write a cached stack the way an earlier successful fetch left it."""
    path = cache_dir / f"cfht_megapipe_{band}.fits"
    path.write_bytes(_stack_bytes(np.ones((400, 400), np.float32), **kw))
    return path


def _patch_cadc(monkeypatch, archive_bands, *, queried=None, downloaded=None):
    """Patch the two CADC calls and the stack download the provider makes.

    archive_bands : bands the fake archive holds a plane for.
    queried, downloaded : lists the fakes append to, so a test can assert a
        round trip did or did not happen -- the whole point of the cache.
    """
    from astroquery.cadc import CadcClass

    def fake_query_region(self, coordinates, *, radius=None, collection=None,
                          **kw):
        if queried is not None:
            queried.append(collection)
        return _plane_table(archive_bands)

    def fake_get_image_list(self, query_result, coordinates, radius):
        band = _band_of(query_result['energy_bandpassName'][0])
        return [f"https://cadc.invalid/soda/{band}"]

    def fake_get(url, timeout=None):
        band = url.rsplit('/', 1)[-1]
        if downloaded is not None:
            downloaded.append(band)
        return _FakeResponse(_stack_bytes(np.ones((400, 400), np.float32)))

    monkeypatch.setattr(CadcClass, 'query_region', fake_query_region)
    monkeypatch.setattr(CadcClass, 'get_image_list', fake_get_image_list)
    monkeypatch.setattr(cfht_images.requests, 'get', fake_get)


def test_complete_cache_skips_the_cadc_query(monkeypatch, tmp_path):
    # Every band on disk: the query would only re-derive the band list the
    # filenames already encode, so a CADC outage cannot stall a re-measure.
    for band in DEFAULT_BANDS:
        _cache_stack(tmp_path, band)
    queried, downloaded = [], []
    _patch_cadc(monkeypatch, DEFAULT_BANDS, queried=queried,
                downloaded=downloaded)
    products = fetch(TARGET, cache_dir=tmp_path, size_arcsec=60.0)
    assert queried == [] and downloaded == []
    assert sorted(p.band for p in products) == sorted(DEFAULT_BANDS)


def test_partial_cache_retries_the_missing_bands(monkeypatch, tmp_path):
    # A band that failed on an earlier run leaves only the bands that
    # worked on disk. Judging completeness on the cache itself would pass
    # by construction and strand the failed bands forever, so an
    # unrequested fetch is judged against DEFAULT_BANDS.
    for band in ('g', 'r', 'i'):
        _cache_stack(tmp_path, band)
    queried, downloaded = [], []
    _patch_cadc(monkeypatch, DEFAULT_BANDS, queried=queried,
                downloaded=downloaded)
    products = fetch(TARGET, cache_dir=tmp_path, size_arcsec=60.0)
    assert queried == [COLLECTION]                  # the round trip happened
    assert sorted(downloaded) == ['u', 'z']         # only what was missing
    assert sorted(p.band for p in products) == ['g', 'i', 'r', 'u', 'z']


def test_requested_bands_are_judged_on_what_was_asked_for(monkeypatch,
                                                          tmp_path):
    # An explicit request is complete once ITS bands are cached -- the
    # bands the caller never asked for must not force a query.
    _cache_stack(tmp_path, 'g')
    queried, downloaded = [], []
    _patch_cadc(monkeypatch, DEFAULT_BANDS, queried=queried,
                downloaded=downloaded)
    products = fetch(TARGET, bands=('g',), cache_dir=tmp_path,
                     size_arcsec=60.0)
    assert queried == [] and downloaded == []
    assert [p.band for p in products] == ['g']
