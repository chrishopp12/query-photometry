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
    mask, area, _ = find_artifacts(raw, good, np.zeros(shape), rr, NOISE, PIX)
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
    mask, _, _ = find_artifacts(raw, good, pred, rr, NOISE, PIX)
    assert not mask[60, 60]
    assert mask[140, 140]


def test_core_damage_beyond_the_claim_is_masked():
    """The target guards its core through its claim, not by fiat: light
    far past the ratio ON the target is instrument damage (and the
    coverage gate downstream demotes the band)."""
    rng = np.random.default_rng(5)
    shape = (201, 201)
    raw = rng.normal(0.0, NOISE, shape)
    raw[95:107, 95:107] += 5.0
    pred = np.zeros(shape)
    pred[90:112, 90:112] = 2.0 * NOISE       # a real-but-faint claim
    good = np.ones(shape, bool)
    rr = radii_arcsec(shape, 100.0, 100.0, PIX)
    mask, area, _ = find_artifacts(raw, good, pred, rr, NOISE, PIX)
    assert mask[100, 100] and area > recipe.ARTIFACT_AREA_MIN


def test_bright_core_is_model_business_not_artifact():
    """Data far above a LOUD claim (a cD cusp over its smooth model, a
    saturated star over its clipped catalog flux) is shape business:
    the detector operates only where the scene is quiet."""
    rng = np.random.default_rng(8)
    shape = (201, 201)
    raw = rng.normal(0.0, NOISE, shape)
    pred = np.zeros(shape)
    pred[95:107, 95:107] = 30.0 * NOISE      # loud claim
    raw += 5.0 * pred                        # data 5x past it
    good = np.ones(shape, bool)
    rr = radii_arcsec(shape, 100.0, 100.0, PIX)
    mask, area, _ = find_artifacts(raw, good, pred, rr, NOISE, PIX)
    assert area == 0.0 and not mask.any()


def test_soft_skirt_floods_from_the_hard_core():
    """A soft-edged artifact's sub-threshold skirt is taken by the
    seeded flood; equally bright unseeded structure elsewhere is not."""
    rng = np.random.default_rng(6)
    shape = (201, 201)
    raw = rng.normal(0.0, NOISE, shape)
    raw[20:25, 30:170] += 5.0            # hard core, 100 sigma
    raw[17:28, 30:170] += 0.15           # 3-sigma skirt hugging the core
    raw[150:156, 30:80] += 0.15          # 3-sigma blob, no seed: untouched
    good = np.ones(shape, bool)
    rr = radii_arcsec(shape, 100.0, 100.0, PIX)
    mask, area, flood_area = find_artifacts(raw, good, np.zeros(shape),
                                            rr, NOISE, PIX)
    assert flood_area > 0.0
    assert mask[17, 100] and mask[27, 100]     # skirt rows taken
    assert not mask[152, 50]                   # unseeded blob left alone
    assert area > 300.0


def test_skirt_flood_stops_at_a_claim():
    """The flood cannot grow into pixels a real source claims: the
    boundary stays surgical exactly where the field is crowded."""
    rng = np.random.default_rng(7)
    shape = (201, 201)
    raw = rng.normal(0.0, NOISE, shape)
    raw[20:25, 30:170] += 5.0            # hard core
    raw[25:34, 30:170] += 0.2            # skirt spreading north
    pred = np.zeros(shape)
    pred[29:60, 30:170] = 0.2            # a claimed source's light there
    good = np.ones(shape, bool)
    rr = radii_arcsec(shape, 100.0, 100.0, PIX)
    mask, _, flood_area = find_artifacts(raw, good, pred, rr, NOISE, PIX)
    assert flood_area > 0.0
    assert mask[27, 100]                       # unclaimed skirt flooded
    assert not mask[31, 100]                   # claimed rows respected


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
    image[36:47, 20:220] += 30.0         # the bleed: ~600 sigma, 41" south
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


def test_broad_source_nicked_by_a_narrow_artifact_is_kept(tmp_path):
    """A narrow strip through a broad neighbor's center must not void
    it: the mask owns only a sliver of its claim, so the component
    stays and its amplitude solves from the clean pixels alone."""
    from sedphot.measure.engine import measure_band
    from sedphot.results import ImageProduct

    rng = np.random.default_rng(14)
    nx = ny = 241
    psf = moffat_kernel(1.3, PIX)
    cx = cy = (nx - 1) / 2.0
    f_target, f_nb = 300.0, 400.0
    dx = 40.0                              # neighbor 20" east
    image = (render_sersic([ampl_from_total(f_target / 3.631, 2.0 / PIX,
                                            2.0, 0.0),
                            2.0 / PIX, 2.0, 0.0, 0.0, cx, cy],
                           (ny, nx), psf)
             + render_sersic([ampl_from_total(f_nb / 3.631, 4.0 / PIX,
                                              1.0, 0.0),
                              4.0 / PIX, 1.0, 0.0, 0.0, cx + dx, cy],
                             (ny, nx), psf)
             + 0.02 + rng.normal(0.0, NOISE, (ny, nx)))
    # a 3-px vertical strip straight through the neighbor's center
    image[:, int(cx + dx) - 1:int(cx + dx) + 2] += 30.0
    wcs = _make_wcs(nx, ny)
    rows = [_catalog_row(wcs, cx, cy, flux_nmgy=f_target / 3.631),
            _catalog_row(wcs, cx + dx, cy, flux_nmgy=f_nb / 3.631,
                         shape_r=4.0)]
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
    m, _ = measure_band(
        product, SkyCoord(RA, DEC, unit='deg'), scene, None, {},
        aperture_arcsec=12.0, cutout_half_arcsec=55.0,
        rgrid=np.arange(2.0, 26.0, 1.0), dump_dir=str(tmp_path))
    witness = m['witness']
    assert witness['artifact_as2'] > 50.0
    assert witness['n_comps'] == 2            # the broad neighbor stays
    z = np.load(tmp_path / 'Legacy_r_arrays.npz')
    nb_amp = [a for a, o in zip(z['amps'], z['owners']) if o != 'target']
    # its amplitude comes from the clean pixels, not the strip
    assert nb_amp[0] == pytest.approx(f_nb, rel=0.3)
    assert m['flux_ujy'] == pytest.approx(f_target, rel=0.10)


def test_artifact_nicking_the_aperture_is_twin_filled(tmp_path):
    """An artifact clipping the aperture edge below the coverage budget
    is reconstructed by the same twin fill as a neighbor mask, and the
    flux survives."""
    rng = np.random.default_rng(13)
    nx = ny = 241
    psf = moffat_kernel(1.3, PIX)
    cx = cy = (nx - 1) / 2.0
    f_target = 300.0
    ampl = ampl_from_total(f_target / 3.631, 2.0 / PIX, 2.0, 0.0)
    image = (render_sersic([ampl, 2.0 / PIX, 2.0, 0.0, 0.0, cx, cy],
                           (ny, nx), psf)
             + 0.02 + rng.normal(0.0, NOISE, (ny, nx)))
    # a streak segment whose chord clips the aperture rim from one side
    image[int(cy) - 24:int(cy) - 22, int(cx) - 5:nx] += 30.0
    wcs = _make_wcs(nx, ny)
    rows = [_catalog_row(wcs, cx, cy, flux_nmgy=f_target / 3.631)]
    m = _measure(tmp_path, image, rows)
    witness = m['witness']
    assert witness['artifact_as2'] > 30.0
    in_ap_holes = (~m['good']) & (np.hypot(
        *np.meshgrid(np.arange(m['good'].shape[1]) - m['cx'],
                     np.arange(m['good'].shape[0]) - m['cy'])) * PIX < 12.0)
    assert in_ap_holes.any()                  # the aperture was nicked
    assert witness['twinfrac'] > 0.9          # and mirror-reconstructed
    assert m['flux_ujy'] == pytest.approx(f_target, rel=0.10)


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


def test_registry_component_claims_its_own_light():
    """A consumed frozen component's base is at physical amplitude, so
    its real light is claimed and the detector never masks it."""
    from sedphot.measure.seats import apply_registry
    from sedphot.measure.stamp import Stamp
    from astropy.io import fits as afits

    ny = nx = 201
    wcs = _make_wcs(nx, ny)
    cx = cy = (nx - 1) / 2.0
    rng = np.random.default_rng(21)
    sky = wcs.pixel_to_world(cx + 50.0, cy)
    entry = {'J0': dict(ra=float(sky.ra.deg), dec=float(sky.dec.deg),
                        components={'Legacy_r': [dict(
                            kind='sersic', ra=float(sky.ra.deg),
                            dec=float(sky.dec.deg), ellip=0.1, pa=0.0,
                            flux_ref=500.0, reff_as=2.0, n=2.0)]})}
    data = rng.normal(0.0, NOISE, (ny, nx))
    stamp = Stamp(data=data, wcs=wcs, header=afits.Header(), cx=cx, cy=cy,
                  pixscale=PIX, cf=1.0,
                  rr=radii_arcsec((ny, nx), cx, cy, PIX),
                  nodata=np.zeros((ny, nx), bool), sigma=NOISE,
                  farfield_sb=None)
    psf = moffat_kernel(1.3, PIX)
    comps, consumed = apply_registry([], entry, stamp, psf, 'Legacy_r',
                                     'Legacy')
    assert consumed == ['J0']
    frozen = comps[0]
    # the base carries the solved flux, not a unit shape
    assert float(frozen['base'].sum()) * stamp.cf == pytest.approx(500.0,
                                                                   rel=0.05)
    # and light at that amplitude is claimed: the detector stays quiet
    raw = data + frozen['base']
    mask, area, _ = find_artifacts(raw, ~stamp.nodata, frozen['base'],
                                   stamp.rr, NOISE, PIX)
    assert area == 0.0
