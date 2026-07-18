"""Offline tests for the scene background: bin grid, plane, ambient surface."""
from __future__ import annotations

import numpy as np
import pytest

from sedphot.measure import recipe
from sedphot.measure.background import ambient_surface, bin_grid, bin_plane
from sedphot.measure.stamp import radii_arcsec

PIX = 0.5              # synthetic pixel scale (arcsec/px)
SHAPE = (240, 240)     # 10 px bins at PIX: a 24 x 24 bin grid
NOISE = 0.1            # pixel noise sigma (counts)


def center_radii():
    """Radius map (arcsec) about the frame center."""
    ny, nx = SHAPE
    return radii_arcsec(SHAPE, (nx - 1) / 2.0, (ny - 1) / 2.0, PIX)


def noise_image(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, NOISE, size=SHAPE)


def tilted_plane(const, x_slope, y_slope):
    """Plane in bin_plane's parametrization: const at the stamp
    center, slopes in counts per pixel."""
    ny, nx = SHAPE
    yy, xx = np.indices(SHAPE)
    return const + x_slope * (xx - nx / 2) + y_slope * (yy - ny / 2)


def test_bin_plane_recovers_injected_plane():
    rr = center_radii()
    good = np.ones(SHAPE, bool)
    truth = tilted_plane(0.05, 2e-4, -1e-4)
    work = truth + noise_image(11)
    plane = bin_plane(work, good, rr, PIX)
    assert plane['const'] == pytest.approx(0.05, abs=0.005)
    assert len(plane['coefs']) == 3
    assert plane['n_bins'] > 500
    rms = np.sqrt(((plane['img'] - truth) ** 2).mean())
    assert rms < NOISE / 10


def test_bin_rejection_defends_the_plane_against_a_blob():
    rr = center_radii()
    good = np.ones(SHAPE, bool)
    work = noise_image(22)
    work[30:90, 150:210] += 0.5    # extended blob covering 36 whole bins
    plane = bin_plane(work, good, rr, PIX)
    assert plane['n_rej'] >= 30
    assert plane['const'] == pytest.approx(0.0, abs=0.005)
    # without bin rejection the blob owns the level: the plain mean of
    # the same voting pixels is biased by several times the tolerance
    biased = work[good & (rr > recipe.BG_RMIN_AS)].mean()
    assert abs(biased) > 0.02


def test_bin_votes_require_half_the_pixels_usable():
    work = noise_image(33)
    usable = np.ones(SHAPE, bool)
    usable[0:10, 0:6] = False      # bin (0, 0): only 40 of 100 px usable
    usable[0:10, 10:15] = False    # bin (0, 1): exactly half usable
    row_starts, col_starts, bin_px, medians = bin_grid(work, usable, PIX)
    assert bin_px == 10
    assert len(row_starts) == 24 and len(col_starts) == 24
    assert np.isnan(medians[0, 0])        # over half unusable: no vote
    assert np.isfinite(medians[0, 1])     # exactly half usable still votes
    assert np.isfinite(medians).sum() == medians.size - 1


def test_target_light_inside_rmin_never_votes():
    rr = center_radii()
    good = np.ones(SHAPE, bool)
    work = noise_image(44)
    work[rr < recipe.BG_RMIN_AS] += 1000.0    # violent target light
    plane = bin_plane(work, good, rr, PIX)
    assert plane['const'] == pytest.approx(0.0, abs=0.005)
    assert plane['n_rej'] <= 10    # excluded by position, not by rejection
    # bins wholly inside the exclusion radius hold no vote at all
    usable = good & (rr > recipe.BG_RMIN_AS)
    _, _, _, medians = bin_grid(work, usable, PIX)
    assert np.isnan(medians[11, 11]) and np.isnan(medians[12, 12])


def test_ambient_surface_none_when_nearly_all_masked():
    rr = center_radii()
    good = np.ones(SHAPE, bool)
    work = noise_image(55)
    mask = np.ones(SHAPE, bool)
    mask[0:20, 0:20] = False    # leaves four voting bins, under the floor
    assert ambient_surface(work, good, mask, rr, PIX) is None


def test_ambient_surface_tracks_a_smooth_gradient():
    rr = center_radii()
    good = np.ones(SHAPE, bool)
    ny, nx = SHAPE
    yy, xx = np.indices(SHAPE)
    truth = 0.02 + 3e-4 * (xx - nx / 2) + 0.05 * np.sin(2.0 * np.pi * yy / ny)
    work = truth + noise_image(66)
    ambient = ambient_surface(work, good, np.zeros(SHAPE, bool), rr, PIX)
    assert ambient is not None
    # compare inside the bin-center hull (the surface is NaN outside
    # it by construction) and outside the excluded center
    inner = np.zeros(SHAPE, bool)
    inner[10:230, 10:230] = True
    compare = inner & (rr > 20.0)
    resid = (ambient - truth)[compare]
    assert np.isfinite(resid).all()
    assert np.sqrt((resid ** 2).mean()) < NOISE / 5


def test_ambient_surface_excludes_masked_pixels():
    rr = center_radii()
    good = np.ones(SHAPE, bool)
    work = noise_image(77)
    work[35:73, 35:73] += 5.0    # bright source, deliberately bin-misaligned
    mask = np.zeros(SHAPE, bool)
    mask[35:73, 35:73] = True
    masked_ambient = ambient_surface(work, good, mask, rr, PIX)
    naive_ambient = ambient_surface(work, good, np.zeros(SHAPE, bool),
                                    rr, PIX)
    over_blob = (slice(45, 65), slice(45, 65))
    assert np.abs(masked_ambient[over_blob]).max() < 0.05
    assert naive_ambient[over_blob].mean() > 1.0
    # a bin exactly half covered by the mask still votes, and with its
    # clean pixels only: were masked pixels counted, this median would
    # sit near +2.5, not at zero
    usable = good & ~mask & (rr > recipe.BG_RMIN_AS)
    _, _, _, medians = bin_grid(work, usable, PIX)
    assert np.isfinite(medians[3, 4])
    assert abs(medians[3, 4]) < 0.05
