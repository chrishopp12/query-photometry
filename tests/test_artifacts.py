"""Offline tests for the catastrophic-pixel artifact mask: detection
geometry, the claimed-pixel ratio guard, and the engine chain on a
synthetic bleed trail (mask, void, witness token, coverage demotion)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.measure import recipe
from sedphot.measure.artifacts import find_artifacts
from sedphot.measure.psf import moffat_kernel
from sedphot.measure.render import ampl_from_total, render_sersic
from sedphot.measure.stamp import ApertureCoverageError, radii_arcsec

PIX = 0.5
RA, DEC = 150.0, 2.0
NOISE = 0.05


# ------------------------------------
# Detection geometry
# ------------------------------------
def test_streak_detected_small_blob_left_to_flood():
    rng = np.random.default_rng(3)
    shape = (201, 201)
    raw = rng.normal(0.0, NOISE, shape)
    raw[20:26, 30:170] += 5.0            # 100 sigma, ~210 as2
    raw[150:152, 40:42] += 5.0           # bright but tiny: flood territory
    good = np.ones(shape, bool)
    rr = radii_arcsec(shape, 100.0, 100.0, PIX)
    mask, area = find_artifacts(raw, good, np.zeros(shape),
                                np.zeros(shape), rr, NOISE, PIX)
    assert mask[22, 100] and mask[22, 35]
    assert not mask[151, 41]
    assert not mask[100, 100]
    assert area > 200.0


def test_claimed_bright_source_is_protected():
    """A real source the catalog under-predicts fails the ratio test;
    an unclaimed region of the same brightness is masked."""
    rng = np.random.default_rng(4)
    shape = (201, 201)
    raw = rng.normal(0.0, NOISE, shape)
    raw[55:65, 55:65] += 5.0             # claimed at 30 sigma: ratio < 5
    raw[135:147, 135:147] += 5.0         # unclaimed: masked
    pred = np.zeros(shape)
    pred[55:65, 55:65] = 1.5
    good = np.ones(shape, bool)
    rr = radii_arcsec(shape, 100.0, 100.0, PIX)
    mask, _ = find_artifacts(raw, good, pred, np.zeros(shape), rr,
                             NOISE, PIX)
    assert not mask[60, 60]
    assert mask[140, 140]


def test_target_zone_is_never_eligible():
    rng = np.random.default_rng(5)
    shape = (201, 201)
    raw = rng.normal(0.0, NOISE, shape)
    raw[95:107, 95:107] += 5.0
    protect = np.zeros(shape)
    protect[90:112, 90:112] = 2.0 * NOISE
    good = np.ones(shape, bool)
    rr = radii_arcsec(shape, 100.0, 100.0, PIX)
    mask, area = find_artifacts(raw, good, np.zeros(shape), protect, rr,
                                NOISE, PIX)
    assert area == 0.0 and not mask.any()


# ------------------------------------
# Engine chain on a synthetic bleed
# ------------------------------------
def _make_wcs(nx, ny):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [RA, DEC]
    wcs.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs.wcs.cd = np.array([[-PIX / 3600.0, 0.0], [0.0, PIX / 3600.0]])
    return wcs


def _catalog_row(wcs, x, y, *, flux_nmgy, shape_r=2.0, rchisq=1.0,
                 type_='SER'):
    sky = wcs.pixel_to_world(x, y)
    return dict(ra=float(sky.ra.deg), dec=float(sky.dec.deg), type=type_,
                sersic=2.0, shape_r=shape_r, shape_e1=0.0, shape_e2=0.0,
                flux_g=flux_nmgy, flux_r=flux_nmgy, flux_z=flux_nmgy,
                psfsize_g=1.3, psfsize_r=1.3, psfsize_z=1.3,
                rchisq_g=rchisq, rchisq_r=rchisq, rchisq_z=rchisq,
                fracflux_r=0.0, fracin_r=1.0, uJy=flux_nmgy * 3.631)


def _measure(tmp_path, image, rows):
    from sedphot.measure.engine import measure_band
    from sedphot.results import ImageProduct

    ny, nx = image.shape
    wcs = _make_wcs(nx, ny)
    path = tmp_path / 'legacy_r.fits'
    fits.PrimaryHDU(data=image.astype(np.float32),
                    header=wcs.to_header()).writeto(path)
    cat = pd.DataFrame(rows).sort_values(
        'flux_r', ascending=False).reset_index(drop=True)
    stars = pd.DataFrame(columns=['ra', 'dec', 'phot_g_mean_mag',
                                  'parallax', 'parallax_error', 'pmra',
                                  'pmra_error', 'pmdec', 'pmdec_error',
                                  'ruwe'])
    scene = dict(cat=cat, stars=stars, patches={}, registry={},
                 registry_path=None)
    product = ImageProduct(provider='legacy', instrument='Legacy',
                           band='r', path=str(path), calib='nmgy',
                           invvar_path=None, seeing_arcsec=1.3,
                           wave_um=0.64)
    measurement, _ = measure_band(
        product, SkyCoord(RA, DEC, unit='deg'), scene, None, {},
        aperture_arcsec=12.0, cutout_half_arcsec=55.0,
        rgrid=np.arange(2.0, 26.0, 1.0))
    return measurement


def test_bleed_is_masked_its_row_voided_and_the_flux_clean(tmp_path):
    """A bleed trail with a catalog wreck on it: the pixels leave good,
    the wreck row leaves the scene (no seats), the witness carries the
    area, and the target flux is unharmed."""
    from sedphot.measure.aperture import measurement_to_row

    rng = np.random.default_rng(11)
    nx = ny = 241
    psf = moffat_kernel(1.3, PIX)
    cx = cy = (nx - 1) / 2.0
    f_target = 300.0
    ampl = ampl_from_total(f_target / 3.631, 2.0 / PIX, 2.0, 0.0)
    image = (render_sersic([ampl, 2.0 / PIX, 2.0, 0.0, 0.0, cx, cy],
                           (ny, nx), psf)
             + 0.02 + rng.normal(0.0, NOISE, (ny, nx)))
    image[38:43, 20:220] += 30.0         # the bleed: ~600 sigma, 41" south
    wcs = _make_wcs(nx, ny)
    rows = [
        _catalog_row(wcs, cx, cy, flux_nmgy=f_target / 3.631),
        # the wreck: the catalog's echo of the bleed -- bright, absurd
        # reduced chi-square, sitting on the trail
        _catalog_row(wcs, 110.0, 40.0, flux_nmgy=200.0, shape_r=1.5,
                     rchisq=5000.0),
    ]
    m = _measure(tmp_path, image, rows)
    witness = m['witness']
    assert witness['artifact_as2'] > 100.0
    assert witness['n_comps'] == 1                    # the wreck is gone
    assert witness['solve']['seats'] == ['target:sersic']
    assert m['flux_ujy'] == pytest.approx(f_target, rel=0.10)
    assert 'art=' in measurement_to_row(m)['flags']
    # the trail's pixels left the usable map (stamp frame: 10 px offset)
    assert not m['good'][30, 100]


def test_bleed_through_the_aperture_demotes(tmp_path):
    """An artifact crossing the aperture is nodata the fill cannot
    honestly repair past the coverage floor: the band demotes."""
    rng = np.random.default_rng(12)
    nx = ny = 241
    psf = moffat_kernel(1.3, PIX)
    cx = cy = (nx - 1) / 2.0
    ampl = ampl_from_total(300.0 / 3.631, 2.0 / PIX, 2.0, 0.0)
    image = (render_sersic([ampl, 2.0 / PIX, 2.0, 0.0, 0.0, cx, cy],
                           (ny, nx), psf)
             + 0.02 + rng.normal(0.0, NOISE, (ny, nx)))
    image[int(cy) - 20:int(cy) - 16, :] += 30.0   # crosses the aperture
    wcs = _make_wcs(nx, ny)
    rows = [_catalog_row(wcs, cx, cy, flux_nmgy=300.0 / 3.631)]
    with pytest.raises(ApertureCoverageError):
        _measure(tmp_path, image, rows)
