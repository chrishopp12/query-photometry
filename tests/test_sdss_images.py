"""SDSS image provider: per-band failure containment."""
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from sedphot.images import sdss as sdss_images
from sedphot.results import STATUS_ERROR, STATUS_NO_COVERAGE, ProviderResult

COORD = SkyCoord(217.0, 56.9, unit='deg')


def _fake_frame():
    return [fits.HDUList([fits.PrimaryHDU(data=np.ones((16, 16), np.float32))])]


def _patch_get_images(monkeypatch, responses):
    """responses: band -> list of HDULists, or an Exception to raise."""
    from astroquery.sdss import SDSS

    def fake(*, coordinates, radius, band):
        value = responses[band]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(SDSS, 'get_images', staticmethod(fake))


def test_one_bad_band_keeps_the_others(monkeypatch, tmp_path):
    # A KeyError 'run' from deep inside astroquery for one
    # band must not take down the bands that resolve fine.
    _patch_get_images(monkeypatch, {'g': _fake_frame(), 'r': KeyError('run')})
    products = sdss_images.fetch(COORD, bands=('g', 'r'), cache_dir=tmp_path)
    assert not isinstance(products, ProviderResult)
    assert [p.band for p in products] == ['g']


def test_all_bands_failing_reports_error(monkeypatch, tmp_path):
    _patch_get_images(monkeypatch, {'g': KeyError('run'), 'r': KeyError('run')})
    result = sdss_images.fetch(COORD, bands=('g', 'r'), cache_dir=tmp_path)
    assert isinstance(result, ProviderResult)
    assert result.status == STATUS_ERROR
    assert 'KeyError' in result.message


def test_no_frames_is_no_coverage(monkeypatch, tmp_path):
    _patch_get_images(monkeypatch, {'g': [], 'r': []})
    result = sdss_images.fetch(COORD, bands=('g', 'r'), cache_dir=tmp_path)
    assert isinstance(result, ProviderResult)
    assert result.status == STATUS_NO_COVERAGE


def test_cached_frame_skips_network(monkeypatch, tmp_path):
    # A cached file must not touch astroquery at all.
    fits.PrimaryHDU(data=np.ones((16, 16), np.float32)).writeto(
        tmp_path / "sdss_g_frame.fits")
    _patch_get_images(monkeypatch, {'g': ConnectionError("no network")})
    products = sdss_images.fetch(COORD, bands=('g',), cache_dir=tmp_path)
    assert [p.band for p in products] == ['g']
