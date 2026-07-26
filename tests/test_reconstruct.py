"""Sidecar reconstruction: the stored-background evaluators must reproduce
their producers exactly, the pin record must name the grid its shapes were
solved on, and a record that cannot describe the seat list must be refused
rather than rendered (the pinned path stands on all three)."""
import json

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.measure import recipe
from sedphot.measure.aperture import witness_row
from sedphot.measure.background import (bin_plane, residual_mesh,
                                        eval_plane, eval_mesh)
from sedphot.measure.components import build_components
from sedphot.measure.psf import moffat_kernel
from sedphot.measure.render import ampl_from_total, render_sersic
from sedphot.measure.seats import build_seats
from sedphot.measure.solve import joint_fit, pinned_fit
from sedphot.measure.stamp import Stamp, radii_arcsec
from sedphot.remeasure import _build_pin_by_band


def _sersic(stamp, psf, flux, reff_px, n, dx, dy):
    """One PSF-convolved Sersic at (dx, dy) pixels from the stamp center."""
    ampl = ampl_from_total(flux, reff_px, n, 0.0)
    return render_sersic([ampl, reff_px, n, 0.0, 0.0,
                          stamp.cx + dx, stamp.cy + dy],
                         stamp.data.shape, psf)


def _cat(stamp):
    """Scene catalog: the target plus one gate-qualifying neighbor."""
    rows = []
    for dx, dy, nmgy, rchisq, reff in ((0.0, 0.0, 400.0 / 3.631, 1.0, 3.0),
                                       (30, 8, 150.0 / 3.631, 9.0, 2.0)):
        sky = stamp.wcs.pixel_to_world(stamp.cx + dx, stamp.cy + dy)
        rows.append(dict(ra=float(sky.ra.deg), dec=float(sky.dec.deg),
                         type='SER', sersic=2.0, shape_r=reff, shape_e1=0.0,
                         shape_e2=0.0, flux_g=nmgy, flux_r=nmgy, flux_z=nmgy,
                         psfsize_g=1.3, psfsize_r=1.3, psfsize_z=1.3,
                         rchisq_g=rchisq, rchisq_r=rchisq, rchisq_z=rchisq,
                         fracflux_r=0.0, fracin_r=1.0, uJy=nmgy * 3.631))
    return pd.DataFrame(rows).sort_values(
        'flux_r', ascending=False).reset_index(drop=True)


def _geom(shape=(120, 120), pixscale=0.262):
    ny, nx = shape
    yy, xx = np.indices(shape)
    rr = np.hypot(xx - nx / 2, yy - ny / 2) * pixscale
    return rr, np.ones(shape, bool), pixscale


def test_eval_plane_recovers_an_injected_gradient():
    """bin_plane BUILDS its plane through eval_plane, so comparing the two is
    an identity and proves nothing. Assert the real property instead: the
    stored coefficients describe the gradient that was actually there, in the
    units eval_plane reads them back in.

    The tilts come back exact. The LEVEL carries a known, tiny offset: the
    design places each bin at col_starts + bin_px/2, while the clipped mean
    of a linear ramp over that bin sits at col_starts + (bin_px - 1)/2 -- a
    half-pixel gap, so the constant absorbs tilt x 0.5/n per axis. Asserted
    here rather than fixed: the correction is ~1e-4 of a level whose tilt is
    itself a small term, and moving the convention would shift the background
    of every measurement already on disk.
    """
    shape = (460, 460)              # a realistic 120" stamp at 0.262"/px
    ny, nx = shape
    rr, good, pix = _geom(shape)
    yy, xx = np.indices(shape)
    level, tilt_x, tilt_y = 0.5, 0.12, -0.24
    truth = (level + tilt_x * (xx - nx / 2) / nx
             + tilt_y * (yy - ny / 2) / ny)
    bg = bin_plane(truth, good, rr, pix)

    assert bg['n_bins'] > 100        # a real fit, not a near-degenerate one
    assert bg['coefs'][1] == pytest.approx(tilt_x, abs=1e-12)
    assert bg['coefs'][2] == pytest.approx(tilt_y, abs=1e-12)
    half_pixel = tilt_x * 0.5 / nx + tilt_y * 0.5 / ny
    assert bg['const'] == pytest.approx(level - half_pixel, abs=1e-9)
    # ... and the sidecar's coefficients alone rebuild that same surface.
    assert np.allclose(eval_plane(bg['coefs'], shape), truth,
                       atol=2 * abs(half_pixel))


def test_eval_mesh_roundtrips_residual_mesh():
    shape = (120, 120)
    rr, good, pix = _geom(shape)
    yy, xx = np.indices(shape)
    resid = 0.3 * np.sin(xx / 25.0) * np.cos(yy / 30.0)   # smooth structure
    state: dict = {}
    mesh = residual_mesh(resid, good, pix, state=state)
    got = eval_mesh(state, shape)
    assert np.allclose(got, mesh, atol=1e-5)


def test_eval_mesh_empty_state_is_zero():
    shape = (60, 60)
    assert np.array_equal(eval_mesh(None, shape), np.zeros(shape))
    assert np.array_equal(eval_mesh({}, shape), np.zeros(shape))
    assert np.array_equal(eval_mesh({'smoothed': []}, shape), np.zeros(shape))


# ------------------------------------
# The pin record's grid
# ------------------------------------
def _stamp(shape=(240, 240), pixscale=0.5):
    ny, nx = shape
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [150.0, 2.0]
    wcs.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs.wcs.cd = np.array([[-pixscale / 3600.0, 0.0],
                           [0.0, pixscale / 3600.0]])
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    return Stamp(data=np.zeros(shape), wcs=wcs, header=fits.Header(),
                 cx=cx, cy=cy, pixscale=pixscale, cf=1.0,
                 rr=radii_arcsec(shape, cx, cy, pixscale),
                 nodata=np.zeros(shape, bool), sigma=0.05, farfield_sb=None)


def _pinned_seat_column(stamp, psf, params, seat_pix):
    """The single seat column pinned_fit renders for one stored vector."""
    seat = [dict(kind='sersic', owner='n1',
                 ra=float(stamp.wcs.wcs.crval[0]),
                 dec=float(stamp.wcs.wcs.crval[1]))]
    pin = dict(seat_params=params, seat_pix=seat_pix,
               amps=[['n1', 100.0]], bg_coefs=[0.0, 0.0, 0.0])
    fit = pinned_fit(np.zeros(stamp.data.shape),
                     np.ones(stamp.data.shape, bool), stamp, psf, [],
                     seat, set(), pin=pin)
    return fit['cols'][0]


def test_pinned_fit_rescales_shapes_onto_the_bands_own_grid():
    """A vector solved on a COARSER grid renders at the same ANGULAR size.

    Radial seat parameters are grid-relative, so rendering a reference-band
    vector verbatim on a finer-pixel sibling shrinks the source by the scale
    ratio -- silent, and invisible on any instrument with one pixel scale.
    Doubling seat_pix must equal doubling the radial entries by hand (size
    and center offsets; ellipticity and PA do not scale).
    """
    stamp = _stamp()
    psf = moffat_kernel(0.9, stamp.pixscale)
    coarse = _pinned_seat_column(
        stamp, psf, [4.0, 2.0, 0.15, 20.0, 1.0, -0.5],
        seat_pix=2 * stamp.pixscale)
    by_hand = _pinned_seat_column(
        stamp, psf, [8.0, 2.0, 0.15, 20.0, 2.0, -1.0], seat_pix=None)
    assert np.allclose(coarse, by_hand, atol=1e-9)

    # ... and the guard: ignoring seat_pix is a DIFFERENT render, so this
    # test fails if the rescale is dropped.
    unscaled = _pinned_seat_column(
        stamp, psf, [4.0, 2.0, 0.15, 20.0, 1.0, -0.5], seat_pix=None)
    assert not np.allclose(unscaled, by_hand, atol=1e-6)


def test_pinned_fit_without_a_stored_grid_assumes_the_bands_own():
    """A sidecar predating pix_ref renders on this band's grid unchanged."""
    stamp = _stamp()
    psf = moffat_kernel(0.9, stamp.pixscale)
    params = [4.0, 2.0, 0.15, 20.0, 1.0, -0.5]
    assert np.allclose(_pinned_seat_column(stamp, psf, params, None),
                       _pinned_seat_column(stamp, psf, params,
                                           stamp.pixscale), atol=1e-9)


# ------------------------------------
# What the witness records
# ------------------------------------
def _witness(solve_info, solve_free=None, shapes=None):
    stamp = _stamp((40, 40))
    zeros = np.zeros(stamp.data.shape)
    return witness_row(np.zeros(3), np.zeros(3), None, stamp,
                       np.ones(stamp.data.shape, bool),
                       np.zeros(stamp.data.shape, bool), 1.0, zeros, zeros,
                       dict(const=0.0, coefs=[0.0, 0.0, 0.0], n_rej=0,
                            n_bins=4),
                       [0.0], 0.0, 1.0, 'test',
                       rgrid=np.array([2.0, 5.0, 10.0]),
                       aperture_arcsec=5.0, solve_info=solve_info,
                       solve_free=solve_free, shapes=shapes)


def _solve(params, pix_ref=0.262):
    return dict(seats=['target:sersic'], params=params, nfev=10,
                seconds=0.1, at_bound=[], pix_ref=pix_ref)


def test_witness_records_the_solve_grid_and_the_free_solve():
    """Both facts a stored vector needs to be interpretable: the grid its
    radial entries live in, and -- separately from the frozen-target science
    solve -- the free-target vector, which otherwise exists only in the
    mutable cross-field registry."""
    row = _witness(_solve([1.0] * 6, pix_ref=0.262),
                   solve_free=_solve([2.0] * 12, pix_ref=0.5))
    assert row['solve']['pix_ref'] == 0.262
    assert len(row['solve']['params']) == 6        # neighbor seats only
    assert len(row['solve_free']['params']) == 12  # the full seat vector
    assert row['solve_free']['pix_ref'] == 0.5


def test_witness_omits_the_free_solve_where_none_ran():
    assert 'solve_free' not in _witness(_solve([1.0] * 6))


# ------------------------------------
# The pin record's seat vector
# ------------------------------------
def _band(params, *, pix_ref=0.262, free=None, shapes=None, shapes_pix=None):
    band = {'fit_state': {'amps': [['target', 1.0]],
                          'bg_coefs': [0.1, 0.0, 0.0]}}
    if params is not None:
        band['solve'] = {'params': params, 'pix_ref': pix_ref}
    if free is not None:
        band['solve_free'] = {'params': free, 'pix_ref': pix_ref}
    if shapes is not None:
        band['shapes'] = {'params': shapes,
                          'pix': shapes_pix if shapes_pix else pix_ref,
                          'seats': ['t:s'] * (len(shapes) // 6)}
    return band


def test_witness_records_the_shapes_it_measured_with():
    """`shapes` is the whole seat list on the band's own grid, whatever put
    each shape there. `solve` reports only what its solve varied."""
    row = _witness(_solve([1.0] * 6, pix_ref=0.262),
                   shapes=dict(seats=['n:s', 't:s'], params=[2.0] * 12,
                               pix=0.262))
    assert len(row['solve']['params']) == 6      # the free subset
    assert len(row['shapes']['params']) == 12    # every seat
    assert row['shapes']['pix'] == 0.262


def test_pin_prefers_the_measured_shapes_over_the_reference_fallback():
    """A band that records what it measured with needs no fallback, and it
    brings its OWN grid -- which is what makes a differing pixel scale safe."""
    prov = {'per_band': {
        'HST_F606W': _band(list(range(12)), pix_ref=0.04),
        'HST_F160W': _band(list(range(6)), pix_ref=0.04,
                           shapes=[9.0] * 12, shapes_pix=0.09)}}
    pin = _build_pin_by_band(prov, 'forced')
    assert pin['HST_F160W']['seat_params'] == [9.0] * 12
    assert pin['HST_F160W']['seat_pix'] == 0.09     # its own grid, not 0.04
    # The reference band still supplies the fallback for bands without it.
    assert pin['HST_F606W']['seat_params'] == list(range(12))
    assert pin['HST_F606W']['seat_pix'] == 0.04


def test_pin_falls_back_for_sidecars_predating_the_shapes_record():
    """Older sidecars carry only the reference band's solve; that IS the
    shape a transfer band's target was frozen at, so forced is still right."""
    prov = {'per_band': {'Legacy_r': _band(list(range(12))),
                         'Legacy_z': _band(list(range(6)))}}
    pin = _build_pin_by_band(prov, 'forced')
    assert pin['Legacy_z']['seat_params'] == list(range(12))
    assert pin['Legacy_z']['seat_pix'] == 0.262


def test_pin_refuses_a_partial_transfer_vector(capsys):
    """A transfer band's `solve` record covers only the NEIGHBOR seats it
    re-solved. Rendered as a full seat vector it truncates the target's seats
    away, and the band leaves the report with no trace. 'fitted' must fall
    back to forced, and name the bands it demoted."""
    prov = {'per_band': {'Legacy_r': _band(list(range(12))),
                         'Legacy_z': _band(list(range(6)))}}
    pin = _build_pin_by_band(prov, 'fitted')
    assert pin['Legacy_z']['seat_params'] == list(range(12))
    assert 'Legacy_z' in capsys.readouterr().out


def test_pin_uses_a_stored_free_vector_when_present():
    """A gating target's transfer band records its own free-target vector;
    'fitted' is exactly the request to use it, 'forced' to ignore it."""
    free = [9.0] * 12
    prov = {'per_band': {'Legacy_r': _band(list(range(12))),
                         'Legacy_z': _band(list(range(6)), free=free)}}
    assert _build_pin_by_band(prov, 'fitted')['Legacy_z']['seat_params'] == free
    assert _build_pin_by_band(prov, 'forced')['Legacy_z']['seat_params'] \
        == list(range(12))


def test_pin_reference_is_the_first_band_not_the_r_band():
    """per_band preserves order_bands order, so the first band of each
    instrument is its reference -- position in the record is the rule, not
    the band name."""
    prov = {'per_band': {'CFHT_i': _band(list(range(12))),
                         'CFHT_r': _band(list(range(6)))}}
    pin = _build_pin_by_band(prov, 'forced')
    assert pin['CFHT_r']['seat_params'] == list(range(12))
    assert pin['CFHT_i']['seat_params'] == list(range(12))


def test_pin_carries_the_solve_grid():
    prov = {'per_band': {'HST_F606W': _band(list(range(12)), pix_ref=0.04),
                         'HST_F160W': _band(list(range(6)), pix_ref=0.09)}}
    pin = _build_pin_by_band(prov, 'forced')
    # Both pin to the reference vector, so both carry the REFERENCE grid --
    # which is what lets pinned_fit rescale onto the finer/coarser sibling.
    assert pin['HST_F606W']['seat_pix'] == 0.04
    assert pin['HST_F160W']['seat_pix'] == 0.04


# ------------------------------------
# The whole seam, on real solver output
# ------------------------------------
def test_transfer_band_free_shape_survives_to_a_pinned_render():
    """Solve -> witness -> JSON -> pin -> pinned render, on a transfer band.

    Unit tests on either side of this seam can both pass while the seam
    itself is wrong: the engine records the frozen-target science solve
    (neighbor seats only), and the pin builder needs a full seat vector.
    Exercised end to end, through a real JSON round trip.
    """
    rng = np.random.default_rng(7)
    stamp = _stamp((240, 240))
    psf = moffat_kernel(1.3, stamp.pixscale)
    image = (_sersic(stamp, psf, 400.0, 3.0 / stamp.pixscale, 2.5, 0.0, 0.0)
             + _sersic(stamp, psf, 150.0, 2.0 / stamp.pixscale, 1.5, 30, 8)
             + rng.normal(0.0, stamp.sigma, stamp.data.shape))
    comps = build_components(_cat(stamp), stamp, psf, 1.3)
    good = np.ones(stamp.data.shape, bool)
    seats, drops = build_seats(comps, {}, stamp, image)
    need = len(seats) * recipe.SEAT_NPARAMS
    assert ('target', 'sersic') in [(s['owner'], s['kind']) for s in seats]

    ref_fit = joint_fit(image, good, stamp, psf, comps, seats, drops)
    n_fixed = len(ref_fit['fixed'])
    ref = dict(seats=seats, drops=sorted(drops), p=ref_fit['solve_info']['p'],
               pix=stamp.pixscale,
               col_flux=[max(float(a), 0.0)
                         for a in ref_fit['amps'][n_fixed:]])
    science = joint_fit(image, good, stamp, psf, comps, seats, drops, ref=ref)
    free = joint_fit(image, good, stamp, psf, comps, seats, drops, ref=ref,
                     free_target=True)

    # The science solve on a transfer band is NOT a full seat vector.
    assert len(science['solve_info']['params']) < need
    assert len(free['solve_info']['params']) == need

    def band(fit, solve_free=None):
        owners = [c['name'] for c in fit['fixed']] + fit['owners']
        row = _witness(fit['solve_info'], solve_free=solve_free)
        row['fit_state'] = dict(
            amps=[[o, float(a)] for o, a in zip(owners, fit['amps'])],
            bg_coefs=fit['bg']['coefs'])
        return row

    prov = json.loads(json.dumps(
        {'per_band': {'Legacy_r': band(ref_fit),
                      'Legacy_z': band(science,
                                       solve_free=free['solve_info'])}},
        default=float))

    for shape in ('forced', 'fitted'):
        pin = _build_pin_by_band(prov, shape)['Legacy_z']
        assert len(pin['seat_params']) == need, shape
        # The render is the assertion: a short vector raises here.
        rebuilt = pinned_fit(image, good, stamp, psf, comps, seats, drops,
                             pin=pin)
        assert len(rebuilt['cols']) == len(seats)
    # 'fitted' really used the free vector, not the reference one.
    assert _build_pin_by_band(prov, 'fitted')['Legacy_z']['seat_params'] \
        != _build_pin_by_band(prov, 'forced')['Legacy_z']['seat_params']
