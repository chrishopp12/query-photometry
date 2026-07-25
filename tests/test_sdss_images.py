"""SDSS image provider: frame resolution + per-band failure containment."""
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from sedphot.images import sdss as sdss_images
from sedphot.results import STATUS_ERROR, STATUS_NO_COVERAGE, ProviderResult

COORD = SkyCoord(217.0, 56.9, unit='deg')


def _fake_frame():
    return [fits.HDUList([fits.PrimaryHDU(data=np.ones((16, 16), np.float32))])]


def _fake_xid():
    """A one-frame region-query result (the columns matches= needs)."""
    return Table({'ra': [217.0], 'dec': [56.9], 'run': [1000],
                  'rerun': [301], 'camcol': [1], 'field': [100]})


def _patch_sdss(monkeypatch, responses, xid=None):
    """Patch the two SDSS calls the provider makes.

    responses : band -> list of HDULists, or an Exception to raise (from the
        per-band matches= fetch).
    xid : the query_region result. None -> one frame; an Exception -> raise
        it; an empty Table -> the no-objects path.
    """
    from astroquery.sdss import SDSS

    def fake_region(coordinates, radius=None, spectro=False, **kw):
        if isinstance(xid, Exception):
            raise xid
        return _fake_xid() if xid is None else xid

    def fake_images(*, matches, band):
        value = responses[band]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(SDSS, 'query_region', staticmethod(fake_region))
    monkeypatch.setattr(SDSS, 'get_images', staticmethod(fake_images))


def test_one_bad_band_keeps_the_others(monkeypatch, tmp_path):
    # A KeyError 'run' from deep inside astroquery for one
    # band must not take down the bands that resolve fine.
    _patch_sdss(monkeypatch, {'g': _fake_frame(), 'r': KeyError('run')})
    products = sdss_images.fetch(COORD, bands=('g', 'r'), cache_dir=tmp_path)
    assert not isinstance(products, ProviderResult)
    assert [p.band for p in products] == ['g']


def test_all_bands_failing_reports_error(monkeypatch, tmp_path):
    _patch_sdss(monkeypatch, {'g': KeyError('run'), 'r': KeyError('run')})
    result = sdss_images.fetch(COORD, bands=('g', 'r'), cache_dir=tmp_path)
    assert isinstance(result, ProviderResult)
    assert result.status == STATUS_ERROR
    assert 'KeyError' in result.message


def test_no_frames_is_no_coverage(monkeypatch, tmp_path):
    # Region query finds objects, but no frame comes back for any band.
    _patch_sdss(monkeypatch, {'g': [], 'r': []})
    result = sdss_images.fetch(COORD, bands=('g', 'r'), cache_dir=tmp_path)
    assert isinstance(result, ProviderResult)
    assert result.status == STATUS_NO_COVERAGE


def test_no_objects_is_no_coverage(monkeypatch, tmp_path):
    # An empty region query -> the position is off the SDSS footprint.
    _patch_sdss(monkeypatch, {}, xid=Table({'ra': [], 'dec': [], 'run': [],
                                            'rerun': [], 'camcol': [],
                                            'field': []}))
    result = sdss_images.fetch(COORD, bands=('g', 'r'), cache_dir=tmp_path)
    assert isinstance(result, ProviderResult)
    assert result.status == STATUS_NO_COVERAGE


def test_region_query_error_reports_error(monkeypatch, tmp_path):
    # The region query itself failing after retries is an error, not
    # silent no-coverage.
    _patch_sdss(monkeypatch, {}, xid=ConnectionError("TAP down"))
    result = sdss_images.fetch(COORD, bands=('g', 'r'), cache_dir=tmp_path)
    assert isinstance(result, ProviderResult)
    assert result.status == STATUS_ERROR
    assert 'ConnectionError' in result.message


def test_cached_frame_skips_network(monkeypatch, tmp_path):
    # A cached file must not touch astroquery at all -- not even the
    # region query (need is empty, so no frame lookup happens).
    fits.PrimaryHDU(data=np.ones((16, 16), np.float32)).writeto(
        tmp_path / "sdss_g_frame.fits")
    _patch_sdss(monkeypatch, {'g': ConnectionError("no network")},
                xid=ConnectionError("no network"))
    products = sdss_images.fetch(COORD, bands=('g',), cache_dir=tmp_path)
    assert [p.band for p in products] == ['g']
