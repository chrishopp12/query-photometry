"""Offline tests for stamp preparation and the coverage gates."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.measure import recipe
from sedphot.measure.stamp import (ApertureCoverageError, check_coverage,
                                   load_stamp, radii_arcsec)

PIX = 0.5          # synthetic pixel scale (arcsec/px)
RA, DEC = 150.0, 2.0


# Parent-frame pixel of the target (0-indexed, an exact pixel center so
# cutout indices map to parent indices by a constant offset) and the
# offset into a 201-px cutout (cutout_half_arcsec=50 at PIX=0.5).
CENTER_PX = 119
CUTOUT_OFF = CENTER_PX - 100


def write_image(path, data):
    """Write a synthetic TAN-projection FITS image; return its center."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [RA, DEC]
    wcs.wcs.crpix = [CENTER_PX + 1.0, CENTER_PX + 1.0]
    wcs.wcs.cd = np.array([[-PIX / 3600.0, 0.0], [0.0, PIX / 3600.0]])
    header = wcs.to_header()
    fits.PrimaryHDU(data=data.astype(np.float32),
                    header=header).writeto(path, overwrite=True)
    return SkyCoord(RA, DEC, unit='deg')


def noise_image(shape=(240, 240), sigma=0.1, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, sigma, size=shape)


def test_load_stamp_geometry_and_noise(tmp_path):
    path = tmp_path / 'band.fits'
    coord = write_image(path, noise_image())
    stamp = load_stamp(str(path), 'nmgy', coord, cutout_half_arcsec=50.0)
    assert stamp.pixscale == pytest.approx(PIX, rel=1e-6)
    assert stamp.cf == pytest.approx(3.631)
    ny, nx = stamp.shape
    assert stamp.cx == pytest.approx((nx - 1) / 2.0, abs=0.01)
    assert stamp.cy == pytest.approx((ny - 1) / 2.0, abs=0.01)
    assert stamp.sigma == pytest.approx(0.1, rel=0.1)
    assert stamp.rr[int(stamp.cy), int(stamp.cx)] < PIX
    assert not stamp.nodata.any()
    assert stamp.good.all()


def test_radii_arcsec_scale():
    rr = radii_arcsec((11, 11), 5.0, 5.0, 2.0)
    assert rr[5, 5] == 0.0
    assert rr[5, 10] == pytest.approx(10.0)


def test_nodata_flags_nan_zero_and_deep_negative(tmp_path):
    data = noise_image()
    off = CUTOUT_OFF
    data[:off + 20, :] = np.nan               # off-footprint edge
    data[off + 30:off + 34, off + 30:off + 34] = 0.0   # archive zero fill
    data[off + 150, off + 40] = -100.0        # dead pixel
    path = tmp_path / 'band.fits'
    coord = write_image(path, data)
    stamp = load_stamp(str(path), 'nmgy', coord, cutout_half_arcsec=50.0)
    assert stamp.nodata[:20, :].all()
    assert stamp.nodata[30:34, 30:34].all()
    assert stamp.nodata[150, 40]
    assert not stamp.nodata[150, 60]


def test_farfield_measures_offset(tmp_path):
    data = noise_image() + 0.05    # uniform pedestal
    path = tmp_path / 'band.fits'
    coord = write_image(path, data)
    stamp = load_stamp(str(path), 'nmgy', coord, cutout_half_arcsec=60.0)
    assert stamp.farfield_sb is not None
    expected_sb = 0.05 * 3.631 / PIX ** 2
    assert stamp.farfield_sb == pytest.approx(expected_sb, rel=0.05)


def test_farfield_none_on_small_stamp(tmp_path):
    path = tmp_path / 'band.fits'
    coord = write_image(path, noise_image((120, 120)))
    stamp = load_stamp(str(path), 'nmgy', coord, cutout_half_arcsec=30.0)
    assert stamp.farfield_sb is None


def test_invvar_cut_on_same_geometry(tmp_path):
    path = tmp_path / 'band.fits'
    ivpath = tmp_path / 'invvar.fits'
    coord = write_image(path, noise_image())
    write_image(ivpath, np.full((240, 240), 25.0))
    stamp = load_stamp(str(path), 'nmgy', coord, cutout_half_arcsec=50.0,
                       invvar_path=str(ivpath))
    assert stamp.invvar is not None
    assert stamp.invvar.shape == stamp.shape
    assert stamp.invvar[10, 10] == pytest.approx(25.0)


def test_coverage_passes_clean_aperture(tmp_path):
    path = tmp_path / 'band.fits'
    coord = write_image(path, noise_image())
    stamp = load_stamp(str(path), 'nmgy', coord, cutout_half_arcsec=50.0)
    coverage = check_coverage(stamp, aperture_arcsec=12.0, seeing_arcsec=1.3)
    assert coverage == 1.0


def test_coverage_demotes_blank_wedge(tmp_path):
    data = noise_image()
    data[CENTER_PX + 1:, CENTER_PX + 1:] = np.nan   # quadrant hole
    path = tmp_path / 'band.fits'
    coord = write_image(path, data)
    stamp = load_stamp(str(path), 'nmgy', coord, cutout_half_arcsec=50.0)
    with pytest.raises(ApertureCoverageError) as err:
        check_coverage(stamp, aperture_arcsec=12.0, seeing_arcsec=1.3)
    assert err.value.coverage < recipe.COVERAGE_MIN


def test_core_gate_demotes_at_any_fraction(tmp_path):
    data = noise_image()
    data[CENTER_PX, CENTER_PX + 1] = np.nan   # one blank pixel on the peak
    path = tmp_path / 'band.fits'
    coord = write_image(path, data)
    stamp = load_stamp(str(path), 'nmgy', coord, cutout_half_arcsec=50.0)
    with pytest.raises(ApertureCoverageError, match='core'):
        check_coverage(stamp, aperture_arcsec=12.0, seeing_arcsec=1.3)
