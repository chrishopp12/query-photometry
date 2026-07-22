"""Offline tests for PSF resolution: seeing chain, kernels, star loop."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.measure.psf import (empirical_psf, moffat_kernel, resolve_psf,
                                 resolve_seeing)
from sedphot.measure.stamp import Stamp, radii_arcsec

PIX = 0.25         # synthetic pixel scale for star fields (arcsec/px)
FWHM_TRUE = 2.4    # synthetic star FWHM (arcsec)


def make_stamp(data, pixscale=0.5):
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


def star_field(amp=10.0, offset_px=(0.0, 120.0), shape=(321, 321), seed=42):
    """Noise stamp with one beta=3 Moffat star; return (stamp, star coord).

    Offsets and shape are PIXELS at PIX = 0.25"/px: the default puts
    the star 30" north on the 80" stamp -- outside the 25" kernel
    exclusion zone (still a candidate), 10" inside the edge margin."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 0.005, size=shape)
    sx = (shape[1] - 1) / 2.0 + offset_px[0]
    sy = (shape[0] - 1) / 2.0 + offset_px[1]
    yy, xx = np.indices(shape)
    r_as = np.hypot(yy - sy, xx - sx) * PIX
    gamma = FWHM_TRUE / (2.0 * np.sqrt(2.0 ** (1.0 / 3.0) - 1.0))
    data += amp * (1.0 + (r_as / gamma) ** 2) ** -3.0
    stamp = make_stamp(data, pixscale=PIX)
    return stamp, stamp.wcs.pixel_to_world(sx, sy)


def star_table(coord, gmag=17.0):
    return pd.DataFrame({'ra': [coord.ra.deg], 'dec': [coord.dec.deg],
                         'phot_g_mean_mag': [gmag]})


def test_empirical_psf_recovers_star_fwhm():
    stamp, star = star_field()
    result = empirical_psf(stamp, star_table(star))
    assert result is not None
    kernel, fwhm, provenance = result
    assert provenance.startswith('empirical star (G=17.0')
    assert fwhm == pytest.approx(FWHM_TRUE, rel=0.15)
    assert kernel.sum() == pytest.approx(1.0)
    assert kernel.shape[0] == kernel.shape[1]
    assert kernel.shape[0] % 2 == 1
    assert kernel.shape[0] >= 25


@pytest.mark.parametrize('gmag', [15.7, 19.6])
def test_gmag_outside_window_is_skipped(gmag):
    stamp, star = star_field()
    assert empirical_psf(stamp, star_table(star, gmag=gmag)) is None


def test_star_near_edge_is_skipped():
    # the star's center is 2 arcsec from the stamp edge: on the image,
    # but inside the 3-arcsec core margin, so it never becomes a candidate
    stamp, star = star_field(offset_px=(152.0, 0.0))
    assert empirical_psf(stamp, star_table(star)) is None


def test_star_inside_target_zone_is_not_a_kernel_candidate():
    # a perfectly good star 15 arcsec from the target: its rings would
    # measure the target's structure, so it never builds a kernel
    stamp, star = star_field(offset_px=(60.0, 0.0))
    assert empirical_psf(stamp, star_table(star)) is None


def test_contaminated_wings_force_the_early_graft():
    # High-S/N contamination never trips the noise graft; the
    # wing-fraction ceiling must catch it and force the early graft.
    stamp, star = star_field()
    sx, sy = [float(v) for v in stamp.wcs.world_to_pixel(star)]
    yy, xx = np.indices(stamp.data.shape)
    # a broad companion 8" from the star: its gradient across the
    # rings does NOT cancel against the star's own background annulus
    # (centered smooth structure does -- the medians circularize it
    # away, which is the builder's own robustness)
    r_b = np.hypot(yy - sy, xx - (sx + 8.0 / PIX)) * PIX
    gamma_b = 10.0 / 1.0196
    stamp.data[...] = stamp.data + 3.0 * (1 + (r_b / gamma_b) ** 2) ** -3.0
    result = empirical_psf(stamp, star_table(star))
    assert result is not None
    kernel, fwhm, provenance = result
    assert 'forced moffat wings' in provenance
    c = kernel.shape[0] // 2
    kyy, kxx = np.indices(kernel.shape)
    kr = np.hypot(kyy - c, kxx - c) * PIX
    assert float(kernel[kr > 2 * fwhm].sum()) <= 0.10 + 1e-6


def test_faint_star_gets_moffat_wing_graft():
    stamp, star = star_field(amp=0.5)
    result = empirical_psf(stamp, star_table(star))
    assert result is not None
    kernel, fwhm, provenance = result
    assert '+moffat wings' in provenance
    assert fwhm == pytest.approx(FWHM_TRUE, rel=0.15)
    assert kernel.sum() == pytest.approx(1.0)


def test_resolve_psf_prefers_empirical():
    stamp, star = star_field()
    _, _, provenance = resolve_psf(stamp, None, star_table(star))
    assert provenance.startswith('empirical star')
    assert 'moffat fallback' not in provenance


def test_resolve_psf_moffat_fallback():
    stamp = make_stamp(np.random.default_rng(1).normal(0.0, 0.01, (121, 121)))
    no_stars = pd.DataFrame(columns=['ra', 'dec', 'phot_g_mean_mag'])
    kernel, fwhm, provenance = resolve_psf(stamp, None, no_stars)
    assert provenance == 'provider default (moffat fallback)'
    assert fwhm == 1.0
    assert kernel.shape == (33, 33)     # 16 x (1.0 / 0.5) px -> odd 33
    assert kernel.sum() == pytest.approx(1.0)

    stamp.header['SEEING'] = 1.7        # a sane header keyword wins next
    _, fwhm, provenance = resolve_psf(stamp, None, no_stars)
    assert fwhm == 1.7
    assert provenance == 'header SEEING (moffat fallback)'


def test_resolve_seeing_catalog_beats_header():
    cat = pd.DataFrame({'psfsize_r': [1.2, 1.4, 1.3]})
    header = fits.Header()
    header['SEEING'] = 2.0
    seeing, provenance = resolve_seeing(cat, header, psfsize_col='psfsize_r')
    assert seeing == pytest.approx(1.3)
    assert provenance == 'catalog median psfsize_r'
    # the column participates only when the caller names it
    seeing, provenance = resolve_seeing(cat, header)
    assert seeing == 2.0
    assert provenance == 'header SEEING'


def test_resolve_seeing_header_guards_and_order():
    header = fits.Header()
    header['SEEING'] = 0.05             # implausibly small: skipped
    assert resolve_seeing(None, header) == (1.0, 'provider default')
    header['SEEING'] = 'junk'           # non-numeric: skipped
    assert resolve_seeing(None, header) == (1.0, 'provider default')
    header['SEEING'] = 2.0
    header['FINALIQ'] = 0.9             # FINALIQ outranks SEEING
    assert resolve_seeing(None, header) == (0.9, 'header FINALIQ')


def test_resolve_seeing_falls_back():
    cat = pd.DataFrame({'psfsize_r': [-1.0, -1.0]})   # not positive: skipped
    seeing, provenance = resolve_seeing(cat, None, psfsize_col='psfsize_r',
                                        fallback_arcsec=0.8,
                                        fallback_label='survey default')
    assert seeing == 0.8
    assert provenance == 'survey default'


def test_moffat_kernel_sizing_taper_and_normalization():
    small = moffat_kernel(0.8, 0.5)     # 16 x 1.6 px -> 26 -> odd 27
    assert small.shape == (27, 27)
    assert small.sum() == pytest.approx(1.0)
    big = moffat_kernel(3.0, 0.5)       # 16 x 6 px -> 96 -> odd 97
    assert big.shape == (97, 97)
    assert big.sum() == pytest.approx(1.0)
    # the edge taper reaches exactly zero on the whole boundary: no
    # square step can survive into a rendered scene
    assert float(big[0, :].max()) == 0.0
    assert float(big[-1, :].max()) == 0.0
    assert float(big[:, 0].max()) == 0.0
    assert float(big[:, -1].max()) == 0.0
