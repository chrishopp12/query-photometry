"""Offline tests for star confirmation and the measured-star stage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.measure import recipe
from sedphot.measure.stamp import Stamp, radii_arcsec
from sedphot.measure.stars import (confirm_stars, measure_star_profile,
                                   subtract_stars)

# Synthetic pixel scale (arcsec/px), fine enough that even the
# innermost 1-arcsec profile ring clears STAR_RING_MIN_PX.
PIX = 0.25
MOFFAT_BETA = 3.0


def make_stamp(data, pixscale=0.5):
    """Wrap a bare array in a Stamp (unit calibration, TAN WCS)."""
    ny, nx = data.shape
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs.wcs.cd = np.array([[-pixscale / 3600.0, 0.0],
                           [0.0, pixscale / 3600.0]])
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    nodata = ~np.isfinite(data)
    rr = radii_arcsec(data.shape, cx, cy, pixscale)
    sigma = float(np.nanstd(data[rr > 15.0]))
    return Stamp(data=data, wcs=wcs, header=fits.Header(), cx=cx, cy=cy,
                 pixscale=pixscale, cf=1.0, rr=rr, nodata=nodata,
                 sigma=sigma, farfield_sb=None)


def moffat_blob(shape, x, y, total, fwhm_as, pixscale):
    """Analytic circular Moffat star with a known total flux (counts)."""
    gamma_px = (fwhm_as / pixscale) / (
        2.0 * np.sqrt(2.0 ** (1.0 / MOFFAT_BETA) - 1.0))
    yy, xx = np.indices(shape, dtype=float)
    rr2 = (xx - x) ** 2 + (yy - y) ** 2
    peak = total * (MOFFAT_BETA - 1.0) / (np.pi * gamma_px ** 2)
    return peak * (1.0 + rr2 / gamma_px ** 2) ** -MOFFAT_BETA


def make_comp(name, cat, x, y, base):
    """One scene-component dict with the keys the star stage reads."""
    return dict(name=name, cat=float(cat), x=float(x), y=float(y), base=base)


def star_rows(stamp, positions):
    """Confirmed-star rows (ra, dec, G) at the given stamp-pixel spots."""
    rows = []
    for x, y, gmag in positions:
        sky = stamp.wcs.pixel_to_world(x, y)
        rows.append(dict(ra=float(sky.ra.deg), dec=float(sky.dec.deg),
                         phot_g_mean_mag=gmag))
    return pd.DataFrame(rows)


def test_confirm_stars_truth_table():
    gaia = pd.DataFrame({
        'phot_g_mean_mag': [16.0, 17.0, 18.0, 19.0],
        'parallax':        [5.0, 0.1, np.nan, 0.2],
        'parallax_error':  [0.5, 0.5, np.nan, 0.5],
        'pmra':            [0.5, 30.0, 1.0, 0.5],
        'pmra_error':      [1.0, 1.0, np.nan, 1.0],
        'pmdec':           [0.5, 40.0, 1.0, 0.5],
        'pmdec_error':     [1.0, 1.0, np.nan, 1.0],
    })
    out = confirm_stars(gaia)
    # parallax star and proper-motion star pass; the solution-less row
    # (NaN errors) and the low-significance row fail
    assert out['phot_g_mean_mag'].tolist() == [16.0, 17.0]
    assert list(out.index) == [0, 1]


def test_profile_recovers_synthetic_star():
    rng = np.random.default_rng(1)
    shape = (480, 480)
    cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    sx, sy = cx + 120.0, cy      # 30 arcsec east of the frame center
    blob = moffat_blob(shape, sx, sy, 2000.0, 6.0, PIX)
    data = rng.normal(0.0, 0.05, size=shape) + blob
    stamp = make_stamp(data, pixscale=PIX)

    profile_img = measure_star_profile(
        stamp.data, stamp.good, np.zeros(shape), 0.0, sx, sy,
        PIX, stamp.rr, 0.05)

    # Integrates to the injected flux
    assert profile_img.sum() == pytest.approx(blob.sum(), rel=0.05)

    # Monotone decreasing along radius
    yy, xx = np.indices(shape)
    r_star = np.hypot(yy - sy, xx - sx) * PIX
    order = np.argsort(r_star.ravel())
    assert np.all(np.diff(profile_img.ravel()[order]) <= 1e-9)

    # Zero beyond the terminus
    assert np.all(profile_img[r_star > recipe.STAR_PROF_MAX_AS] == 0.0)


def test_subtract_stars_brightest_first_and_gates():
    rng = np.random.default_rng(2)
    shape = (480, 480)
    cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    x1, y1 = cx + 120.0, cy           # 30" east: bright star
    x2, y2 = cx, cy + 112.0           # 28" north: faint star
    x3, y3 = cx - 100.0, cy - 60.0    # 29" southwest: below STAR_MIN_UJY
    target_blob = moffat_blob(shape, cx, cy, 500.0, 4.0, PIX)
    blob1 = moffat_blob(shape, x1, y1, 3000.0, 6.0, PIX)
    blob2 = moffat_blob(shape, x2, y2, 400.0, 6.0, PIX)
    blob3 = moffat_blob(shape, x3, y3, 50.0, 6.0, PIX)
    data = (rng.normal(0.0, 0.05, size=shape)
            + target_blob + blob1 + blob2 + blob3)
    stamp = make_stamp(data, pixscale=PIX)

    comps = [make_comp('target', 500.0, cx, cy, target_blob),
             make_comp('src1', 3000.0, x1, y1, blob1),
             make_comp('src2', 400.0, x2, y2, blob2),
             make_comp('src3', 50.0, x3, y3, blob3)]
    # Faint star's Gaia row first: treatment order must come from the
    # component catalog flux, not the row order. Rows also land on the
    # too-faint component and on the target -- both must be left alone.
    stars = star_rows(stamp, [(x2, y2, 17.5),
                              (x1, y1, 15.5),
                              (x3, y3, 18.5),
                              (cx, cy, 16.5)])

    star_img, star_masks, pruned, star_log = subtract_stars(
        stamp, stamp.data, stamp.good, comps, stars, 0.0)

    # Brightest first; target and the faint component keep their seats
    assert [rec['comp'] for rec in star_log] == ['src1', 'src2']
    assert [rec['gmag'] for rec in star_log] == [15.5, 17.5]
    assert [name for name, _ in star_masks] == ['src1', 'src2']
    assert [c['name'] for c in pruned] == ['target', 'src3']

    # The measured profiles carry the injected star flux...
    assert star_log[0]['profile_uJy'] == pytest.approx(3000.0, rel=0.1)
    assert star_log[1]['profile_uJy'] == pytest.approx(400.0, rel=0.1)
    assert star_img.sum() * stamp.cf == pytest.approx(3400.0, rel=0.1)

    # ...and the subtraction removes most of each star from the frame
    residual = stamp.data - star_img
    yy, xx = np.indices(shape)
    near1 = np.hypot(yy - y1, xx - x1) * PIX < 10.0
    near2 = np.hypot(yy - y2, xx - x2) * PIX < 10.0
    assert abs(residual[near1].sum()) < 0.1 * blob1[near1].sum()
    assert abs(residual[near2].sum()) < 0.1 * blob2[near2].sum()


def test_two_gaia_rows_same_component_treated_once():
    rng = np.random.default_rng(3)
    shape = (240, 240)
    cx, cy = (shape[1] - 1) / 2.0, (shape[0] - 1) / 2.0
    sx, sy = cx + 80.0, cy       # 20 arcsec east
    blob = moffat_blob(shape, sx, sy, 1000.0, 6.0, PIX)
    data = rng.normal(0.0, 0.05, size=shape) + blob
    stamp = make_stamp(data, pixscale=PIX)

    comps = [make_comp('target', 300.0, cx, cy, np.zeros(shape)),
             make_comp('src1', 1000.0, sx, sy, blob)]
    # Two Gaia rows within the match radius of the same component
    stars = star_rows(stamp, [(sx, sy, 16.0), (sx + 2.0, sy, 16.2)])

    star_img, star_masks, pruned, star_log = subtract_stars(
        stamp, stamp.data, stamp.good, comps, stars, 0.0)

    assert len(star_log) == 1
    assert len(star_masks) == 1
    assert star_log[0]['comp'] == 'src1'
    assert star_log[0]['gmag'] == pytest.approx(16.0)   # first row claims
    assert [c['name'] for c in pruned] == ['target']
    assert star_img.sum() > 0.0
