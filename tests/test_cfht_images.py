"""CFHT stack selection: footprint-centroid ordering and cutout coverage."""
import io

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits

from sedphot.images.cfht import _covers_target, _footprint_center_offset

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
    # aperture cannot fit (the c35 failure).
    assert not _covers_target(_fits_bytes(np.ones((60, 400), np.float32)), TARGET)


def test_partial_edge_through_aperture_fails():
    # Blank edge slicing through the pad box but not the target pixel
    # itself (the ~30x-low c12 failure).
    data = np.ones((400, 400), np.float32)
    data[:, 230:] = 0.0  # target at x=200, pad box reaches x=280
    assert not _covers_target(_fits_bytes(data), TARGET)


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
