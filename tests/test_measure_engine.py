"""Measurement engine on synthetic galaxies: ownership, sky, coverage."""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling.models import Sersic2D
from astropy.wcs import WCS

from sedphot.measure.aperture import (ApertureCoverageError, cog_slope,
                                      cog_step, measure_aperture,
                                      measurement_to_row)
from sedphot.results import ImageProduct
from sedphot.units import NANOMAGGY_TO_UJY

# One synthetic sky for the whole module: 0.5"/px, 241 px = 120" stamps.
PIXSCALE = 0.5
SIZE = 241
CENTER = SkyCoord(150.0, 30.0, unit='deg')
APERTURE = 12.0
SKY_IN, SKY_OUT = 30.0, 45.0
SEEING = 1.2
NOISE = 0.02


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [CENTER.ra.deg, CENTER.dec.deg]
    w.wcs.crpix = [SIZE // 2 + 1, SIZE // 2 + 1]
    w.wcs.cd = np.array([[-PIXSCALE / 3600.0, 0.0], [0.0, PIXSCALE / 3600.0]])
    return w


def _render(sources):
    """Noiseless image from (dx_as, dy_as, amplitude, reff_as, ellip) tuples."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    c = SIZE // 2
    image = np.zeros((SIZE, SIZE))
    for dx, dy, ampl, reff, ellip in sources:
        model = Sersic2D(amplitude=ampl, r_eff=reff / PIXSCALE, n=2.0,
                         x_0=c + dx / PIXSCALE, y_0=c + dy / PIXSCALE,
                         ellip=ellip, theta=0.6)
        image += model(xx, yy)
    return image


GALAXY = (0.0, 0.0, 1.0, 4.0, 0.4)   # the target every test measures


def _truth_flux(image):
    """Aperture-summed flux of a noiseless rendering, in uJy."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    c = SIZE // 2
    rr = np.hypot(xx - c, yy - c) * PIXSCALE
    return float(image[rr < APERTURE].sum()) * NANOMAGGY_TO_UJY


def _product(tmp_path, image, name="syn"):
    path = tmp_path / f"{name}.fits"
    header = _wcs().to_header()
    fits.PrimaryHDU(data=image.astype(np.float32), header=header).writeto(path)
    return ImageProduct(provider='syn', instrument='SYN', band='r',
                        path=str(path), calib='nmgy',
                        seeing_arcsec=SEEING, wave_um=0.62)


def _measure(tmp_path, image, name="syn", **kwargs):
    return measure_aperture(
        _product(tmp_path, image, name), CENTER,
        aperture_arcsec=APERTURE, sky_in=SKY_IN, sky_out=SKY_OUT,
        cutout_half_arcsec=60.0, **kwargs)


TRUTH = _truth_flux(_render([GALAXY]))
RNG_SKY = np.random.default_rng(7)


def _noisy(image, seed=7):
    return image + np.random.default_rng(seed).normal(0.0, NOISE, image.shape)


# ------------------------------------
# Ownership
# ------------------------------------
def test_isolated_target_is_never_self_eaten(tmp_path):
    m = _measure(tmp_path, _noisy(_render([GALAXY])))
    assert m['masked_fraction'] < 0.03          # envelope not gouged
    assert m['aperture_coverage'] == pytest.approx(1.0)
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.03)


def test_neighbor_inside_aperture_is_deblended(tmp_path):
    # A companion inside the aperture radius. The apportioned
    # symmetric deblend removes its share -- wings included,
    # not just a masked footprint -- before the sky or the curve see it.
    neighbor = (8.0, 3.0, 0.8, 1.5, 0.1)
    m = _measure(tmp_path, _noisy(_render([GALAXY, neighbor])))
    assert m['n_deblended'] == 1
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.06)


def test_neighbor_wings_are_deblended(tmp_path):
    # A bright neighbor's wings spread over the aperture. Masking
    # any isophote keeps what lies below it (+15%
    # residual at best); the apportioned deblend removes the neighbor's
    # share everywhere, far-side wings included.
    neighbor = (8.0, 0.0, 2.5, 3.0, 0.15)
    m = _measure(tmp_path, _noisy(_render([GALAXY, neighbor])))
    assert m['n_deblended'] == 1
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.06)


def test_wings_flood_mask_backstop_without_deblend(tmp_path):
    # The flood-mask channel is the safety net when deblending is off
    # (or a core evades detection): footprint masked to its excess
    # boundary, twin-filled, and the residual is the min-template floor
    # (~+15%), never the naked +80%.
    neighbor = (8.0, 0.0, 2.5, 3.0, 0.15)
    m = _measure(tmp_path, _noisy(_render([GALAXY, neighbor])),
                 deblend=False)
    c = SIZE // 2
    wing = c + int(round(5.5 / PIXSCALE))     # between target and neighbor
    assert m['mask'][c, wing], "neighbor wing inside the aperture must be masked"
    assert m['masked_fraction'] > 0.25, "flood must extend well past the core"
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.20)


def test_overlapping_blend_is_apportioned(tmp_path):
    # An elongated neighbor touching the target. Shared light in
    # the overlap zone splits by symmetric-template ratio
    # instead of being claimed whole by either side.
    neighbor = (6.0, 1.0, 1.5, 2.5, 0.6)
    m = _measure(tmp_path, _noisy(_render([GALAXY, neighbor])))
    assert m['n_deblended'] == 1
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.10)


def test_neighbor_envelope_is_deblended_and_flagged(tmp_path):
    # A giant's diffuse envelope floods the aperture, seeded for
    # deblending by its cuspy nucleus. The symmetric-about-
    # target part of the envelope is structurally unremovable, so this
    # class must BOTH improve (was +193% naked, +76% masked) AND stay
    # loudly flagged: the residual reads as a pedestal far above any
    # clean field's.
    giant = [(20.0, 5.0, 3.0, 10.0, 0.2), (20.0, 5.0, 25.0, 0.6, 0.05)]
    m = _measure(tmp_path, _noisy(_render([GALAXY] + giant)))
    assert m['n_deblended'] >= 1
    assert m['flux_ujy'] < 1.55 * TRUTH
    assert m['cog_pedestal'] > 0.6, \
        "the unremovable envelope residual must scream in cogped"


def test_symmetric_neighbor_pair_is_deblended(tmp_path):
    # Two equal neighbors placed point-symmetrically about the target
    # cancel in the target-centered excess map -- the old flood's blind
    # spot. Neighbor-centered templates do NOT cancel, so the deblend
    # closes the hole (+25% under masking alone).
    pair = [(9.0, 0.0, 1.2, 2.0, 0.1), (-9.0, 0.0, 1.2, 2.0, 0.1)]
    m = _measure(tmp_path, _noisy(_render([GALAXY] + pair)))
    assert m['n_deblended'] == 2
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.12)


def test_shallow_band_borrows_deep_band_neighbor_models(tmp_path):
    # The depth problem: a neighbor's wings are sub-threshold per pixel
    # in a noisy band, so per-band deblending cannot reach them and the
    # contamination stays. With the deep band's contained templates as
    # fixed shapes -- reprojected, one amplitude fit in the shallow
    # band -- the wings are subtracted below that band's own detection
    # floor (forced photometry of the contaminants).
    from sedphot.measure.deblend import reference_component_templates
    neighbor = (8.0, 0.0, 2.5, 3.0, 0.15)
    scene = _render([GALAXY, neighbor])
    c = SIZE // 2
    deep_noise, shallow_noise, band_scale = 0.004, 0.3, 0.6
    deep = scene + np.random.default_rng(11).normal(0, deep_noise, scene.shape)
    shallow = band_scale * scene \
        + np.random.default_rng(12).normal(0, shallow_noise, scene.shape)
    templates, ref_target = reference_component_templates(
        deep, deep_noise, float(c), float(c), PIXSCALE,
        seeing_arcsec=SEEING)
    assert templates, "the deep band must model the neighbor"
    truth = band_scale * TRUTH
    m_self = _measure(tmp_path, shallow, name="shal_self")
    m_tpl = _measure(tmp_path, shallow, name="shal_tpl",
                     deblend_templates=(templates, ref_target,
                                        _wcs(), SEEING))
    err_self = abs(m_self['flux_ujy'] - truth) / truth
    err_tpl = abs(m_tpl['flux_ujy'] - truth) / truth
    assert m_tpl['flux_ujy'] == pytest.approx(truth, rel=0.10)
    assert err_tpl < err_self, \
        "borrowed deep models must beat the band's own blind deblend"


def test_lumpy_envelope_is_not_masked(tmp_path):
    # LSB envelope substructure -- brightness patchiness around the
    # smooth profile -- must NOT be eaten. An
    # interloper is masked only when it DOMINATES the local profile;
    # +/-35% patchiness sits at the profile level and stays in the flux.
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(3)
    base = _render([GALAXY])
    patch = gaussian_filter(rng.normal(0.0, 1.0, base.shape), 3.0 / PIXSCALE)
    patch *= 0.35 / patch.std()
    image = base * (1.0 + patch)
    truth = _truth_flux(image)
    m = _measure(tmp_path, _noisy(image, seed=3))
    assert m['masked_fraction'] < 0.05, "envelope patchiness must stay in the flux"
    assert m['flux_ujy'] == pytest.approx(truth, rel=0.04)


def test_background_gradient_is_absorbed(tmp_path):
    # A bright halo lays a smooth gradient across the field; a scalar annulus median tilts the curve
    # of growth. The plane fit must absorb it.
    image = _noisy(_render([GALAXY]))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    image += 0.004 * (xx - SIZE / 2) * PIXSCALE   # 0.004 counts/arcsec ramp
    m = _measure(tmp_path, image)
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.04)
    assert abs(m['cog_slope']) < 0.006, "gradient must not tilt the curve"


def test_far_neighbors_do_not_bias_the_sky(tmp_path):
    # The declining-growth-curve failure: a deep field crowds the
    # annulus with faint sources the 4-sigma peak rejection cannot see;
    # the second sky pass masks their segments before the clip.
    rng = np.random.default_rng(11)
    crowd = []
    for _ in range(140):
        radius = rng.uniform(24.0, 55.0)
        angle = rng.uniform(0, 2 * np.pi)
        crowd.append((radius * np.cos(angle), radius * np.sin(angle),
                      rng.uniform(0.05, 0.25), rng.uniform(0.8, 1.6), 0.2))
    m = _measure(tmp_path, _noisy(_render([GALAXY] + crowd)))
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.04)
    assert m['cog_slope'] > -0.008, "sky over-subtraction signature"


# ------------------------------------
# Coverage
# ------------------------------------
def test_small_blank_wedge_is_fill_corrected(tmp_path):
    # A stack edge clipping the aperture RIM: the azimuthal fill corrects
    # the missing area instead of silently under-counting it.
    image = _noisy(_render([GALAXY]))
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    c = SIZE // 2
    rr = np.hypot(xx - c, yy - c) * PIXSCALE
    angle = np.degrees(np.arctan2(yy - c, xx - c)) % 360
    image[(angle < 12) & (rr > 5) & (rr < 40)] = 0.0
    m = _measure(tmp_path, image)
    assert 0.95 <= m['aperture_coverage'] < 1.0
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.04)
    row = measurement_to_row(m)
    assert f"cov={m['aperture_coverage']:.3f}" in row['flags']


def test_core_clipping_sliver_demotes_at_any_coverage(tmp_path):
    # An edge through the seeing-scale core is unrecoverable -- the annulus
    # median cannot rebuild a clipped peak -- so even a thin sliver with
    # high area coverage must demote, not fill.
    image = _noisy(_render([GALAXY]))
    c = SIZE // 2
    image[c - 1:c + 2, :] = 0.0   # 3-px blank stripe through the center
    with pytest.raises(ApertureCoverageError):
        _measure(tmp_path, image)


def test_half_blank_stamp_demotes_to_no_coverage(tmp_path):
    # The stack-edge failure: half the cutout is fill zeros through the
    # target. 0.0-uJy-with-status-ok must be impossible.
    image = _noisy(_render([GALAXY]))
    image[:, SIZE // 2 - 4:] = 0.0
    with pytest.raises(ApertureCoverageError) as excinfo:
        _measure(tmp_path, image)
    assert excinfo.value.coverage < 0.6


def test_blank_annulus_chunk_leaves_flux_alone(tmp_path):
    # Blank pixels confined to the sky annulus: exact zeros sit at the
    # expected sky level of a subtracted stack, so left in they drag the
    # median and fake flux; excluded, the flux stands.
    image = _noisy(_render([GALAXY]))
    image[:, :50] = 0.0          # x < 50 px: 35"+ from the target
    m = _measure(tmp_path, image)
    assert m['aperture_coverage'] == pytest.approx(1.0)
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.04)


def test_dead_column_sentinel_is_nodata(tmp_path):
    # A dead CCD column in a sky-inclusive stack is not exact zero -- it
    # reads as a deeply negative outlier after sky subtraction. Unflagged,
    # one such column craters the enclosed curve the moment the growing
    # radius reaches it (seen on a real MegaPipe stack).
    sky = 5.0                    # sky-inclusive stack: counts sit high
    image = _noisy(_render([GALAXY])) + sky
    c = SIZE // 2
    column = c + int(round(14.0 / PIXSCALE))   # 14" out: outside aperture
    image[:, column:column + 3] = 0.02          # dead, but not exactly 0
    m = _measure(tmp_path, image)
    assert m['aperture_coverage'] == pytest.approx(1.0)
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.04)
    assert m['cog_slope'] > -0.01, "curve must not crater at the column"


# ------------------------------------
# Metrics
# ------------------------------------
def test_cog_slope_flat_and_declining():
    rgrid = np.arange(2.0, 30.0, 1.0)
    flat = np.full(rgrid.size, 400.0)
    assert cog_slope(rgrid, flat, 400.0) == pytest.approx(0.0, abs=1e-9)
    declining = 400.0 - 5.0 * (rgrid - rgrid.min())
    assert cog_slope(rgrid, declining, 400.0) == pytest.approx(-5.0 / 400.0,
                                                               rel=0.01)


def test_cog_step_refuses_a_plateau_that_does_not_hold():
    # The converged-then-stepped signature: quiet by 6", steps +80 uJy across 10-12"
    # (neighbor wings entering), flat again beyond. The outer slope --
    # fitted at 21-29" -- reads benign, and a quiet-increments-only
    # criterion would call 6" converged; the hold test must refuse:
    # the flux at the aperture is NOT the flux the plateau promised.
    rgrid = np.arange(2.0, 30.0, 1.0)
    enclosed = np.where(rgrid < 6, 400.0 * (rgrid / 6.0), 400.0)
    enclosed = np.where(rgrid > 10, enclosed + 40.0 * np.clip(rgrid - 10, 0, 2),
                        enclosed)
    conv, step = cog_step(rgrid, enclosed, 480.0, 12.0)
    assert not np.isfinite(conv) and not np.isfinite(step)
    assert abs(cog_slope(rgrid, enclosed, 480.0)) < 0.005, \
        "the outer window must be blind here -- that blindness is the point"


def test_cog_step_rejects_subthreshold_linear_drift():
    # 0.7%/arcsec forever: every increment clears a 1%/arcsec bar, but
    # the cumulative drift is 7% by the aperture edge -- linear is not
    # flat, and the metric must not call it converged anywhere.
    rgrid = np.arange(2.0, 30.0, 1.0)
    enclosed = np.minimum(400.0 * (rgrid / 5.0), 400.0) + 2.8 * rgrid
    conv, step = cog_step(rgrid, enclosed, 430.0, 12.0)
    assert not np.isfinite(conv)


def test_cog_step_converged_curve_reads_flat():
    rgrid = np.arange(2.0, 30.0, 1.0)
    enclosed = 400.0 * (1.0 - np.exp(-((rgrid / 3.0) ** 2)))
    conv, step = cog_step(rgrid, enclosed, 400.0, 12.0)
    assert np.isfinite(conv) and conv <= 9.0
    assert abs(step) < 0.02


def test_cog_step_never_converged_returns_nan():
    rgrid = np.arange(2.0, 30.0, 1.0)
    climbing = 50.0 * rgrid                     # blend: no plateau anywhere
    conv, step = cog_step(rgrid, climbing, 600.0, 12.0)
    assert not np.isfinite(conv) and not np.isfinite(step)


def test_cog_step_noise_widens_the_plateau_tolerance():
    # A faint band's noise wander must not read as never-converged.
    rng = np.random.default_rng(5)
    rgrid = np.arange(2.0, 30.0, 1.0)
    enclosed = np.where(rgrid < 5, 30.0 * (rgrid / 5.0), 30.0) \
        + np.cumsum(rng.normal(0.0, 2.0, rgrid.size))
    noise = np.full(rgrid.size - 1, 2.0)
    conv, _ = cog_step(rgrid, enclosed, 30.0, 12.0, step_noise=noise)
    assert np.isfinite(conv)


def test_flags_tokens_are_machine_parsable(tmp_path):
    m = _measure(tmp_path, _noisy(_render([GALAXY])))
    row = measurement_to_row(m)
    tokens = dict(t.split('=') for t in row['flags'].split(';'))
    assert set(tokens) >= {'cov', 'maskfrac', 'cogslope', 'cogconv',
                           'cogped', 'cogrms', 'nbsub'}
    assert float(tokens['cov']) == pytest.approx(1.0)
    assert tokens['nbsub'] == '0'          # isolated target: deblend no-op
    if tokens['cogconv'] != 'none':
        float(tokens['cogconv'])
        float(tokens['cogstep'])
