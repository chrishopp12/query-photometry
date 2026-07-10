"""Measurement engine on synthetic galaxies: ownership, sky, coverage."""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling.models import Sersic2D
from astropy.wcs import WCS

from sedphot.measure.aperture import (ApertureCoverageError, cog_slope,
                                      measure_aperture, measurement_to_row)
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


def test_neighbor_inside_aperture_is_masked_and_filled(tmp_path):
    # The c34 failure: a companion inside the aperture radius. Structural
    # ownership masks it regardless of radius; the azimuthal fill replaces
    # its pixels with the target's own profile.
    neighbor = (8.0, 3.0, 0.8, 1.5, 0.1)
    m = _measure(tmp_path, _noisy(_render([GALAXY, neighbor])))
    c = SIZE // 2
    ny = int(round(c + neighbor[1] / PIXSCALE))
    nx = int(round(c + neighbor[0] / PIXSCALE))
    assert m['mask'][ny, nx], "companion center must be masked"
    assert m['flux_ujy'] == pytest.approx(TRUTH, rel=0.05)


def test_far_neighbors_do_not_bias_the_sky(tmp_path):
    # The declining-growth-curve failure (c48): a deep field crowds the
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


def test_flags_tokens_are_machine_parsable(tmp_path):
    m = _measure(tmp_path, _noisy(_render([GALAXY])))
    row = measurement_to_row(m)
    tokens = dict(t.split('=') for t in row['flags'].split(';'))
    assert set(tokens) == {'cov', 'maskfrac', 'cogslope'}
    assert float(tokens['cov']) == pytest.approx(1.0)
