"""Offline tests for the scene-engine rendering primitives."""
from __future__ import annotations

import numpy as np
import pytest
from astropy.modeling.models import Sersic2D
from astropy.wcs import WCS
from scipy.signal import fftconvolve

from sedphot.measure import recipe, render
from sedphot.measure.render import (ampl_from_total, conv_same, moffat_wings,
                                    pa_map, render_nuker, render_sersic,
                                    render_sersic_boxed, sersic_profile,
                                    sersic_total)
from sedphot.measure.sersic import moffat_psf, theta_from_pa

PIX = 0.5          # synthetic pixel scale (arcsec/px)

# A 1x1 unit kernel: convolution with it returns the image to FFT
# round-off, so profile shapes can be checked directly.
DELTA_PSF = np.ones((1, 1))


def tan_wcs():
    """Plain TAN WCS (north +y, east -x) centered mid-frame."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [65.0, 65.0]
    wcs.wcs.cd = np.array([[-PIX / 3600.0, 0.0], [0.0, PIX / 3600.0]])
    return wcs


def test_sersic_profile_matches_astropy():
    shape = (128, 128)
    yy, xx = np.indices(shape, dtype=float)
    cases = [
        [2.0, 6.0, 1.5, 0.0, 0.0, 64.0, 64.0],      # round, centered
        [0.7, 10.0, 4.0, 0.45, 0.9, 60.3, 70.2],    # elliptical, rotated
    ]
    for ampl, reff, n, ellip, theta, x0, y0 in cases:
        ours = sersic_profile([ampl, reff, n, ellip, theta, x0, y0], shape)
        astro = Sersic2D(amplitude=ampl, r_eff=reff, n=n, x_0=x0, y_0=y0,
                         ellip=ellip, theta=theta)(xx, yy)
        np.testing.assert_allclose(ours, astro, rtol=1e-10, atol=0.0)


def test_sersic_profile_reff_floor():
    shape = (64, 64)
    tiny = sersic_profile([1.0, 0.01, 2.0, 0.1, 0.3, 32.0, 32.0], shape)
    floored = sersic_profile([1.0, 0.3, 2.0, 0.1, 0.3, 32.0, 32.0], shape)
    assert np.array_equal(tiny, floored)


def test_conv_same_matches_fftconvolve_and_caches():
    rng = np.random.default_rng(42)
    img = rng.normal(size=(128, 128))
    psf = moffat_psf(1.3, PIX)
    first = conv_same(img, psf)
    np.testing.assert_allclose(first, fftconvolve(img, psf, mode='same'),
                               rtol=0.0, atol=1e-12)
    n_entries = len(render._CONV_CACHE)
    second = conv_same(img, psf)    # cache hit: same kernel, same shape
    assert len(render._CONV_CACHE) == n_entries
    assert np.array_equal(first, second)


def test_boxed_render_matches_full_frame():
    shape = (256, 256)
    psf = moffat_psf(1.3, PIX)
    args = (4.0, 2.0, 0.3, 0.7, 130.2, 120.8)
    boxed = render_sersic_boxed(*args, shape, psf)
    full = render_sersic([1.0, *args], shape, psf)
    # The box was actually used: the frame corner is exactly zero in
    # the pasted render, never in the full-frame convolution.
    assert boxed[0, 0] == 0.0
    assert full[0, 0] != 0.0
    assert np.abs(boxed - full).sum() <= 1e-4 * full.sum()


def test_boxed_render_full_frame_fallback():
    shape = (256, 256)
    psf = moffat_psf(1.3, PIX)
    args = (60.0, 4.0, 0.2, 0.3, 128.0, 128.0)   # extent far beyond frame
    boxed = render_sersic_boxed(*args, shape, psf)
    full = render_sersic([1.0, *args], shape, psf)
    assert np.array_equal(boxed, full)


def test_flux_amplitude_round_trip():
    counts = 1234.5
    for reff, n, ellip in [(5.0, 3.0, 0.25), (2.0, 0.7, 0.0)]:
        ampl = ampl_from_total(counts, reff, n, ellip)
        assert sersic_total(ampl, reff, n, ellip, 1.0) == pytest.approx(
            counts, rel=1e-12)
    ampl = ampl_from_total(counts, 5.0, 3.0, 0.25)
    assert sersic_total(ampl, 5.0, 3.0, 0.25, 2.5) == pytest.approx(
        2.5 * counts, rel=1e-12)


def test_profile_sum_approaches_total():
    shape = (256, 256)
    counts = 1000.0
    reff, n, ellip = 5.0, 1.0, 0.2
    ampl = ampl_from_total(counts, reff, n, ellip)
    img = sersic_profile([ampl, reff, n, ellip, 0.4, 127.5, 127.5], shape)
    assert float(img.sum()) == pytest.approx(counts, rel=0.01)


def test_nuker_monotone_and_truncated():
    shape = (256, 256)
    pixscale = 2.0
    rb = 10.0
    img = render_nuker(rb, 3.0, 0.0, 0.0, 128.0, 128.0, shape,
                       DELTA_PSF, pixscale)
    ray = img[128, 129:250]                    # radial cut along +x
    assert np.all(np.diff(ray) < 0)
    rtrunc_px = recipe.NUKER_TRUNC_AS / pixscale
    far = img[128, 128 + int(round(1.5 * rtrunc_px))]
    at_break = img[128, 128 + int(rb)]
    assert far < 1e-3 * at_break


def test_nuker_ellipticity_and_theta():
    shape = (256, 256)
    pixscale = 2.0
    along_x = render_nuker(10.0, 3.0, 0.4, 0.0, 128.0, 128.0, shape,
                           DELTA_PSF, pixscale)
    assert along_x[128, 158] > along_x[158, 128]   # major axis along +x
    along_y = render_nuker(10.0, 3.0, 0.4, np.pi / 2.0, 128.0, 128.0, shape,
                           DELTA_PSF, pixscale)
    assert along_y[158, 128] > along_y[128, 158]   # rotated to +y
    assert along_y[158, 128] == pytest.approx(along_x[128, 158], rel=1e-6)


def test_moffat_wings_analytic_normalization():
    counts, fwhm_px = 1000.0, 6.0
    small_sum = float(moffat_wings(counts, fwhm_px, 32.0, 32.0,
                                   (64, 64)).sum())
    big_sum = float(moffat_wings(counts, fwhm_px, 256.0, 256.0,
                                 (512, 512)).sum())
    assert small_sum < counts
    assert abs(counts - big_sum) < abs(counts - small_sum)
    assert big_sum == pytest.approx(counts, rel=1e-3)


def test_pa_map_quarter_turn():
    wcs = tan_wcs()
    t0, slope = pa_map(wcs, 64.0, 64.0)
    assert t0 == pytest.approx(np.pi / 2.0, abs=1e-3)   # north is +y here
    # theta(90) - theta(0) is a quarter turn (modulo a full turn).
    assert (slope * 90.0) % (2.0 * np.pi) == pytest.approx(np.pi / 2.0,
                                                           abs=1e-3)
    # The map is affine in PA: check 45 deg against theta_from_pa,
    # compared mod pi (the profiles it feeds are 180-deg symmetric).
    t45 = theta_from_pa(wcs, 64.0, 64.0, 45.0)
    wrap = (t0 + slope * 45.0 - t45) % np.pi
    assert min(wrap, np.pi - wrap) < 1e-3
