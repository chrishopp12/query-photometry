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
