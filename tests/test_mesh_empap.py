"""Offline tests for the post-fit residual mesh and the empty-aperture
error: unclaimed background-scale light leaves the curve of growth, a
well-fit target is untouched, and the reported error is the measured
scatter of source-free aperture sums."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.measure import recipe
from sedphot.measure.aperture import empap_error
from sedphot.measure.background import residual_mesh
from sedphot.measure.psf import moffat_kernel
from sedphot.measure.render import ampl_from_total, render_sersic
from sedphot.measure.stamp import Stamp, radii_arcsec

PIX = 0.5
RA, DEC = 150.0, 2.0
NOISE = 0.05


def _make_wcs(nx, ny):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [RA, DEC]
    wcs.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs.wcs.cd = np.array([[-PIX / 3600.0, 0.0], [0.0, PIX / 3600.0]])
    return wcs


def _make_stamp(data):
    ny, nx = data.shape
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    return Stamp(data=data, wcs=_make_wcs(nx, ny), header=fits.Header(),
                 cx=cx, cy=cy, pixscale=PIX, cf=1.0,
                 rr=radii_arcsec(data.shape, cx, cy, PIX),
                 nodata=np.zeros(data.shape, bool), sigma=NOISE,
                 farfield_sb=None)


# ------------------------------------
# The mesh itself
# ------------------------------------
def test_mesh_follows_broad_structure_not_compact_light():
    rng = np.random.default_rng(15)
    shape = (241, 241)
    yy, xx = np.indices(shape)
    broad = 0.04 * np.exp(-((xx - 60.0) ** 2 + (yy - 170.0) ** 2)
                          / (2 * (24.0) ** 2))          # sigma 12"
    psf = moffat_kernel(1.3, PIX)
    compact = render_sersic([ampl_from_total(50.0, 2.0, 1.0, 0.0),
                             2.0, 1.0, 0.0, 0.0, 170.0, 60.0],
                            shape, psf)
    resid = broad + compact + rng.normal(0.0, NOISE, shape)
    mesh = residual_mesh(resid, np.ones(shape, bool), PIX)
    # the broad blob is captured near its peak...
    assert mesh[170, 60] == pytest.approx(0.04, rel=0.25)
    # ...while the compact source's peak is diluted far below itself
    assert mesh[60, 170] < 0.25 * compact[60, 170]


def test_mesh_of_pure_noise_is_flat():
    rng = np.random.default_rng(16)
    shape = (241, 241)
    mesh = residual_mesh(rng.normal(0.0, NOISE, shape),
                         np.ones(shape, bool), PIX)
    assert float(np.abs(mesh).max()) < 3.0 * NOISE / np.sqrt(
        (recipe.BIN_AS / PIX) ** 2)


# ------------------------------------
# Empty-aperture error
# ------------------------------------
def test_empap_matches_white_noise_and_sees_correlation():
    rng = np.random.default_rng(17)
    shape = (241, 241)
    white = rng.normal(0.0, NOISE, shape)
    stamp = _make_stamp(white)
    vote = np.ones(shape, bool)
    err_w, n_w = empap_error(white, vote, stamp, aperture_arcsec=12.0)
    assert n_w >= recipe.EMPAP_MIN_APS
    n_ap = np.pi * (12.0 / PIX) ** 2
    expected = NOISE * np.sqrt(n_ap)
    assert err_w == pytest.approx(expected, rel=0.4)

    # correlate the same noise field (5-px boxcar): per-pixel sigma
    # drops but the APERTURE error must not drop with it
    from scipy.ndimage import uniform_filter
    corr = uniform_filter(white, size=5)
    stamp_c = _make_stamp(corr)
    err_c, n_c = empap_error(corr, vote, stamp_c, aperture_arcsec=12.0)
    naive_c = float(corr.std()) * np.sqrt(n_ap)
    assert n_c >= recipe.EMPAP_MIN_APS
    assert err_c > 2.0 * naive_c


def test_empap_too_small_stamp_reports_zero():
    rng = np.random.default_rng(18)
    shape = (61, 61)          # 30" stamp: no room outside the exclusion
    stamp = _make_stamp(rng.normal(0.0, NOISE, shape))
    err, n = empap_error(stamp.data, np.ones(shape, bool), stamp,
                         aperture_arcsec=12.0)
    assert err == 0.0 and n < recipe.EMPAP_MIN_APS


# ------------------------------------
# Engine chain: the mesh in the curve of growth
# ------------------------------------
def test_unclaimed_blob_is_meshed_out_of_the_flux(tmp_path):
    """A broad unclaimed glow across the aperture: the mesh takes it,
    the flux lands on the target's own value, and the row's error is
    the measured empty-aperture scatter."""
    from sedphot.measure.aperture import measurement_to_row
    from sedphot.measure.engine import measure_band
    from sedphot.results import ImageProduct

    rng = np.random.default_rng(19)
    nx = ny = 241
    psf = moffat_kernel(1.3, PIX)
    cx = cy = (nx - 1) / 2.0
    f_target = 300.0
    # A SYMMETRIC pair of broad glows: no net tilt for the plane to
    # absorb, centers far outside any component's reach, skirts
    # crossing the aperture from both sides -- pure mesh food. The
    # target refit is disabled so the target's claim is exactly its
    # (exact) catalog model and the glow is wholly unclaimed.
    yy, xx = np.indices((ny, nx))
    blob = 0.06 * (np.exp(-((xx - cx - 70.0) ** 2 + (yy - cy) ** 2)
                          / (2 * 36.0 ** 2))
                   + np.exp(-((xx - cx + 70.0) ** 2 + (yy - cy) ** 2)
                            / (2 * 36.0 ** 2)))
    image = (render_sersic([ampl_from_total(f_target / 3.631, 2.0 / PIX,
                                            2.0, 0.0),
                            2.0 / PIX, 2.0, 0.0, 0.0, cx, cy],
                           (ny, nx), psf)
             + blob + 0.02 + rng.normal(0.0, NOISE, (ny, nx)))
    wcs = _make_wcs(nx, ny)
    sky = wcs.pixel_to_world(cx, cy)
    cat = pd.DataFrame([dict(
        ra=float(sky.ra.deg), dec=float(sky.dec.deg), type='SER',
        sersic=2.0, shape_r=2.0, shape_e1=0.0, shape_e2=0.0,
        flux_g=f_target / 3.631, flux_r=f_target / 3.631,
        flux_z=f_target / 3.631, psfsize_g=1.3, psfsize_r=1.3,
        psfsize_z=1.3, rchisq_g=1.0, rchisq_r=1.0, rchisq_z=1.0,
        fracflux_r=0.0, fracin_r=1.0, uJy=f_target)])
    path = tmp_path / 'legacy_r.fits'
    fits.PrimaryHDU(data=image.astype(np.float32),
                    header=wcs.to_header()).writeto(path)
    stars = pd.DataFrame(columns=['ra', 'dec', 'phot_g_mean_mag',
                                  'parallax', 'parallax_error', 'pmra',
                                  'pmra_error', 'pmdec', 'pmdec_error',
                                  'ruwe'])
    scene = dict(cat=cat, stars=stars, patches={'target_refit': False},
                 registry={}, registry_path=None)
    product = ImageProduct(provider='legacy', instrument='Legacy',
                           band='r', path=str(path), calib='nmgy',
                           invvar_path=None, seeing_arcsec=1.3,
                           wave_um=0.64)
    m, _ = measure_band(
        product, SkyCoord(RA, DEC, unit='deg'), scene, None, {},
        aperture_arcsec=12.0, cutout_half_arcsec=55.0,
        rgrid=np.arange(2.0, 26.0, 1.0))
    witness = m['witness']
    assert witness['mesh_ap_uJy'] > 15.0          # the skirts left the flux
    assert m['flux_ujy'] == pytest.approx(f_target, rel=0.10)
    assert 'mesh=' in measurement_to_row(m)['flags']
    assert m['flux_err_ujy'] > 0.0
