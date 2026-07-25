"""Offline tests for the shared image-cache size guard."""
from __future__ import annotations

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.images.common import warn_undersized_cache


def write_cutout(path, *, npix, pixscale_arcsec):
    """A blank square cutout with a real celestial WCS."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [(npix + 1) / 2.0] * 2
    wcs.wcs.cd = np.array([[-pixscale_arcsec / 3600.0, 0.0],
                           [0.0, pixscale_arcsec / 3600.0]])
    fits.writeto(path, np.zeros((npix, npix), dtype='f4'), wcs.to_header())


def test_full_size_cache_is_quiet(tmp_path, capsys):
    path = tmp_path / 'cut.fits'
    write_cutout(path, npix=480, pixscale_arcsec=0.25)   # 120 arcsec
    assert warn_undersized_cache(path, 120.0, 'PS1') is False
    assert capsys.readouterr().out == ''


def test_undersized_cache_warns(tmp_path, capsys):
    path = tmp_path / 'cut.fits'
    write_cutout(path, npix=480, pixscale_arcsec=0.25)   # 120 arcsec
    assert warn_undersized_cache(path, 240.0, 'PS1') is True
    out = capsys.readouterr().out
    assert 'WARNING' in out
    assert '120' in out and '240' in out


def test_unreadable_cache_is_silent(tmp_path, capsys):
    path = tmp_path / 'junk.fits'
    path.write_bytes(b'not a fits file')
    assert warn_undersized_cache(path, 120.0, 'CFHT') is False
    assert capsys.readouterr().out == ''


def test_mosaic_fills_from_complementary_tiles():
    """Two tiles that each clip half the target combine to full coverage."""
    from astropy.coordinates import SkyCoord

    from sedphot.images.common import mosaic_first_valid
    ra, dec, pix, n = 150.0, 2.0, 0.2, 100

    def plane(nan_rows):
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        wcs.wcs.crval = [ra, dec]
        wcs.wcs.crpix = [(n + 1) / 2.0] * 2
        wcs.wcs.cd = np.array([[-pix / 3600.0, 0.0], [0.0, pix / 3600.0]])
        d = np.ones((n, n), dtype='f4')
        d[nan_rows] = np.nan          # this tile clips these rows
        return d, wcs

    a = plane(np.s_[:3 * n // 10, :])   # clips the top 30% (tiles overlap
    b = plane(np.s_[7 * n // 10:, :])   # in the middle, as real ones do)
    coord = SkyCoord(ra, dec, unit='deg')
    data, wcs = mosaic_first_valid([a, b], coord,
                                   size_arcsec=n * pix, pixscale=pix)
    assert data is not None and data.shape == (n, n)
    assert np.isfinite(data[5:-5, 5:-5]).all()   # halves cover the interior


def test_mosaic_single_plane_gap_stays_nan():
    """A pixel no tile covers stays NaN -- coverage is honest, not fabricated."""
    from astropy.coordinates import SkyCoord

    from sedphot.images.common import mosaic_first_valid
    ra, dec, pix, n = 150.0, 2.0, 0.2, 100
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(n + 1) / 2.0] * 2
    wcs.wcs.cd = np.array([[-pix / 3600.0, 0.0], [0.0, pix / 3600.0]])
    d = np.ones((n, n), dtype='f4')
    d[:n // 2, :] = np.nan            # only one tile, covering half
    data, _ = mosaic_first_valid([(d, wcs)], SkyCoord(ra, dec, unit='deg'),
                                 size_arcsec=n * pix, pixscale=pix)
    assert not np.isfinite(data[5, 50])          # uncovered pixel stays NaN
    assert np.isfinite(data[-5, 50])             # covered pixel is filled
