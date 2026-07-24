"""Offline tests for the scene engine core: components, seats, the
joint solve, the measurement witnesses, and one synthetic band
end-to-end through the driver."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.measure import recipe
from sedphot.measure.aperture import (build_mask, flux_error,
                                      measurement_to_row, ped_fit,
                                      plateau_hold, qa_flags, twin_fill,
                                      witness_row)
from sedphot.measure.components import build_components, gated_row
from sedphot.measure.psf import moffat_kernel
from sedphot.measure.render import (ampl_from_total, render_sersic,
                                    sersic_profile)
from sedphot.measure.seats import (apply_registry, build_seats,
                                   harvest_seats, load_registry,
                                   registry_name, save_registry,
                                   seat_slices)
from sedphot.measure.solve import joint_fit
from sedphot.measure.stamp import Stamp, radii_arcsec

PIX = 0.5
RA, DEC = 150.0, 2.0
NOISE = 0.05


def make_wcs(nx, ny, *, pixscale=PIX):
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    wcs.wcs.crval = [RA, DEC]
    wcs.wcs.crpix = [(nx + 1) / 2.0, (ny + 1) / 2.0]
    wcs.wcs.cd = np.array([[-pixscale / 3600.0, 0.0],
                           [0.0, pixscale / 3600.0]])
    return wcs


def make_stamp(data, *, pixscale=PIX, sigma=NOISE):
    ny, nx = data.shape
    wcs = make_wcs(nx, ny, pixscale=pixscale)
    cx, cy = (nx - 1) / 2.0, (ny - 1) / 2.0
    return Stamp(data=data, wcs=wcs, header=fits.Header(), cx=cx, cy=cy,
                 pixscale=pixscale, cf=1.0,
                 rr=radii_arcsec(data.shape, cx, cy, pixscale),
                 nodata=~np.isfinite(data), sigma=sigma, farfield_sb=None)


def catalog_row(wcs, x, y, *, flux_nmgy=50.0, type_='SER', sersic=2.0,
                shape_r=2.0, rchisq=1.0):
    """One scene-catalog row at stamp pixel (x, y)."""
    sky = wcs.pixel_to_world(x, y)
    return dict(ra=float(sky.ra.deg), dec=float(sky.dec.deg), type=type_,
                sersic=sersic, shape_r=shape_r, shape_e1=0.0, shape_e2=0.0,
                flux_g=flux_nmgy, flux_r=flux_nmgy, flux_z=flux_nmgy,
                psfsize_g=1.3, psfsize_r=1.3, psfsize_z=1.3,
                rchisq_g=rchisq, rchisq_r=rchisq, rchisq_z=rchisq,
                fracflux_r=0.0, fracin_r=1.0, uJy=flux_nmgy * 3.631)


def make_catalog(rows):
    cat = pd.DataFrame(rows)
    return cat.sort_values('flux_r', ascending=False).reset_index(drop=True)


def inject_sersic(shape_2d, psf, *, flux, reff_px, n, x, y):
    ampl = ampl_from_total(flux, reff_px, n, 0.0)
    return render_sersic([ampl, reff_px, n, 0.0, 0.0, x, y], shape_2d, psf)


# ------------------------------------
# Components
# ------------------------------------
def test_gated_row_truth_table():
    def row(**kw):
        base = dict(type='SER', flux_g=0.1, flux_r=0.1, flux_z=0.1,
                    rchisq_g=1.0, rchisq_r=1.0, rchisq_z=1.0)
        base.update(kw)
        return pd.Series(base)

    bright = 500.0 / 3.631
    assert gated_row(row(flux_r=bright, rchisq_r=8.0), 20.0)
    assert not gated_row(row(flux_r=bright, rchisq_r=8.0), 0.5)  # target
    assert not gated_row(row(type='PSF', flux_r=bright, rchisq_r=8.0),
                         20.0)
    assert not gated_row(row(flux_r=50 / 3.631, rchisq_r=8.0), 20.0)
    assert not gated_row(row(flux_r=bright, rchisq_r=1.0), 20.0)
    # the pair test is PER BAND: a red member faint in r gates on z
    assert gated_row(row(flux_z=bright, rchisq_z=8.0), 20.0)
    # ... and a bright-in-r, misfit-only-in-z row does NOT gate
    assert not gated_row(row(flux_r=bright, rchisq_z=8.0), 20.0)
    # past the ceiling in ANY band vetoes the row outright
    assert not gated_row(row(flux_r=bright, rchisq_r=8.0, rchisq_z=5e4),
                         20.0)


def test_red_row_with_no_scene_band_flux_still_renders():
    """A row admitted on its z flux alone (flux_r <= 0, a legitimate
    non-detection) must not become a dead zero column: the render
    floors at a tenth of the brightest band, so the column exists and
    the ceiling (100x the render) clears the true band flux."""
    stamp = make_stamp(np.zeros((200, 200)))
    psf = moffat_kernel(1.3, PIX)
    red = catalog_row(stamp.wcs, stamp.cx + 30, stamp.cy, flux_nmgy=-0.05,
                      type_='PSF', sersic=0.0, shape_r=0.0)
    red['flux_z'] = 30.0 / 3.631          # bright in z only
    red['uJy'] = red['flux_r'] * 3.631
    cat = make_catalog([
        catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=100.0), red])
    comps = build_components(cat, stamp, psf, 1.3)
    comp = next(c for c in comps if c['name'] != 'target')
    assert comp['flux0'] > 2.0            # ~0.1 x 30 uJy, alive
    assert float(comp['base'].sum()) > 0.0
    # ceiling headroom: 100 x flux0 comfortably covers the z flux
    assert 100.0 * comp['flux0'] > 30.0


def test_phantom_component_amplitude_drives_to_zero():
    """A cataloged component with NO light in the band solves to
    amplitude zero cleanly (bounded nonnegative solve) and subtracts
    nothing -- absence is the easy direction."""
    rng = np.random.default_rng(23)
    stamp = make_stamp(np.zeros((200, 200)))
    psf = moffat_kernel(1.3, PIX)
    f_target = 300.0
    # cf = 1 on a bare make_stamp: counts ARE microjanskys here
    image = (inject_sersic((200, 200), psf, flux=f_target,
                           reff_px=2.0 / PIX, n=2.0, x=stamp.cx, y=stamp.cy)
             + rng.normal(0.0, NOISE, (200, 200)))
    cat = make_catalog([
        catalog_row(stamp.wcs, stamp.cx, stamp.cy,
                    flux_nmgy=f_target / 3.631, shape_r=2.0),
        # cataloged 40" east at 100 uJy -- absent from the pixels
        catalog_row(stamp.wcs, stamp.cx + 40.0 / PIX, stamp.cy,
                    flux_nmgy=100.0 / 3.631, shape_r=1.5),
    ])
    comps = build_components(cat, stamp, psf, 1.3)
    from sedphot.measure.solve import joint_fit
    fit = joint_fit(image, np.ones((200, 200), bool), stamp, psf, comps,
                    [], set())
    by_name = dict(zip([c['name'] for c in fit['fixed']], fit['amps']))
    phantom = next(n for n in by_name if n != 'target')
    assert by_name[phantom] == pytest.approx(0.0, abs=1.0)
    assert by_name['target'] == pytest.approx(f_target, rel=0.05)


def test_point_source_paste_matches_convolution():
    """A point source's base is the kernel pasted at its bilinear
    corners -- identical to convolving the delta image, at a fraction
    of the cost."""
    from sedphot.measure.components import _paste_kernel
    from sedphot.measure.render import conv_same
    psf = moffat_kernel(1.3, PIX)
    shape = (120, 130)
    rng = np.random.default_rng(2)
    for x, y in ((30.3, 40.7), (5.1, 3.2), (127.6, 116.9)):
        img = np.zeros(shape)
        canvas = np.zeros(shape)
        ix, iy = int(np.floor(x)), int(np.floor(y))
        wx, wy = x - ix, y - iy
        for px, py, w in ((ix, iy, (1 - wx) * (1 - wy)),
                          (ix + 1, iy, wx * (1 - wy)),
                          (ix, iy + 1, (1 - wx) * wy),
                          (ix + 1, iy + 1, wx * wy)):
            if 0 <= px < shape[1] and 0 <= py < shape[0]:
                img[py, px] = w
                _paste_kernel(canvas, psf, px, py, w)
        ref = conv_same(img, psf)
        assert float(np.abs(canvas - ref).max()) < 1e-10


def test_build_components_names_normalization_and_margin():
    stamp = make_stamp(np.zeros((200, 200)))
    psf = moffat_kernel(1.3, PIX)
    rows = [
        catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=100.0),
        catalog_row(stamp.wcs, stamp.cx + 40, stamp.cy, flux_nmgy=50.0),
        # compact source just off-stamp: its render cannot reach -> pruned
        catalog_row(stamp.wcs, -10.0, stamp.cy, flux_nmgy=10.0,
                    shape_r=0.5),
        # bright point source just off-stamp: analytic Moffat wings stay
        catalog_row(stamp.wcs, -10.0, stamp.cy + 30, flux_nmgy=100.0,
                    type_='PSF', sersic=0.0, shape_r=0.0),
    ]
    cat = make_catalog(rows)
    comps = build_components(cat, stamp, psf, 1.3)
    names = {c['name'] for c in comps}
    assert 'target' in names
    target = next(c for c in comps if c['name'] == 'target')
    # uJy normalization: the rendered base carries the catalog flux
    assert target['flux0'] == pytest.approx(target['cat'], rel=0.02)
    # irow points back at the catalog row, no name parsing needed
    assert cat.iloc[target['irow']]['uJy'] == target['cat']
    # the off-stamp compact source was pruned; the bright PSF stayed
    assert len(comps) == 3
    wings = [c for c in comps if c['shape'] is None]
    assert len(wings) == 1 and wings[0]['cat'] >= recipe.BRIGHT_PSF_UJY


def test_drop_target_shreds_scope_and_pinning():
    from sedphot.measure.components import drop_target_shreds

    stamp = make_stamp(np.zeros((200, 200)))
    coord = stamp.wcs.pixel_to_world(stamp.cx, stamp.cy)
    hot = dict(rchisq=1.0)
    rows = [
        catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=200.0, **hot),
        catalog_row(stamp.wcs, stamp.cx + 8, stamp.cy, flux_nmgy=4.0, **hot),
        catalog_row(stamp.wcs, stamp.cx + 16, stamp.cy, flux_nmgy=4.0, **hot),
        catalog_row(stamp.wcs, stamp.cx + 50, stamp.cy, flux_nmgy=4.0, **hot),
    ]
    cat = make_catalog(rows)
    # all rows hot: even a hot TARGET row never drops
    cat['fracflux_r'] = [5.0, 5.0, 5.0, 5.0]
    pinned = stamp.wcs.pixel_to_world(stamp.cx + 16, stamp.cy)
    patches = {'free_seats': [dict(ra=float(pinned.ra.deg),
                                   dec=float(pinned.dec.deg))]}
    kept = drop_target_shreds(cat, coord, aperture_arcsec=12.0,
                              patches=patches)
    # dropped: only the unpinned in-aperture shred (at +8 px = 4")
    assert len(kept) == 3
    dists = np.hypot(kept.ra - coord.ra.deg, 0)   # crude: check by flux
    assert (kept['flux_r'] == 200.0).any()        # target stays
    assert (kept['flux_r'] == 4.0).sum() == 2     # pinned + far stay


def test_gate_radius_is_radial():
    stamp = make_stamp(np.zeros((240, 240)))     # 120" stamp, half 60"
    psf = moffat_kernel(1.3, PIX)
    rows = [
        catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=40.0),
        # corner source: inside the square stamp at 40" diagonal reach,
        # gate-eligible by flux and misfit
        catalog_row(stamp.wcs, stamp.cx + 57, stamp.cy + 57,
                    flux_nmgy=300.0, rchisq=9.0, shape_r=2.5),
    ]
    cat = make_catalog(rows)
    corner = lambda comps: next(c for c in comps if c['x'] > stamp.cx + 30)
    with_reach = build_components(cat, stamp, psf, 1.3,
                                  gate_radius_arcsec=35.0)
    assert not corner(with_reach)['gate']   # 40" > 35" reach: no gate
    without = build_components(cat, stamp, psf, 1.3,
                               gate_radius_arcsec=60.0)
    assert corner(without)['gate']          # inside 60" reach: gates


def test_off_stamp_rows_never_gate():
    stamp = make_stamp(np.zeros((200, 200)))
    psf = moffat_kernel(1.3, PIX)
    rows = [
        catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=40.0),
        # bright + misfit + ON-stamp: gates
        catalog_row(stamp.wcs, stamp.cx + 40, stamp.cy, flux_nmgy=300.0,
                    rchisq=9.0, sersic=5.0, shape_r=2.5),
        # the same source pushed past the edge: its cuspy wings still
        # clear the margin, but a shape solve has no pixels to stand on
        catalog_row(stamp.wcs, -18.0, stamp.cy, flux_nmgy=290.0,
                    rchisq=9.0, sersic=5.0, shape_r=2.5),
    ]
    cat = make_catalog(rows)
    comps = build_components(cat, stamp, psf, 1.3)
    on_stamp = next(c for c in comps if c['x'] > stamp.cx + 10)
    off_stamp = next(c for c in comps if c['x'] < 0)
    assert on_stamp['gate']
    assert not off_stamp['gate']          # present, fixed profile only
    assert off_stamp['flux0'] >= recipe.MARGIN_MIN_UJY


def test_build_components_profile_cache_reuse():
    stamp = make_stamp(np.zeros((160, 160)))
    psf = moffat_kernel(1.3, PIX)
    cat = make_catalog([catalog_row(stamp.wcs, stamp.cx, stamp.cy)])
    cache = {}
    first = build_components(cat, stamp, psf, 1.3, profile_cache=cache)
    second = build_components(cat, stamp, psf, 1.3, profile_cache=cache)
    assert cache and len(first) == len(second) == 1
    np.testing.assert_array_equal(first[0]['base'], second[0]['base'])


# ------------------------------------
# Seats
# ------------------------------------
def _components_with_gated(stamp, psf):
    rows = [
        catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=40.0),
        catalog_row(stamp.wcs, stamp.cx + 30, stamp.cy, flux_nmgy=200.0,
                    rchisq=9.0, shape_r=3.0),
    ]
    cat = make_catalog(rows)
    return cat, build_components(cat, stamp, psf, 1.3)


def test_build_seats_standard_set():
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    _, comps = _components_with_gated(stamp, psf)
    seats, drops = build_seats(comps, {}, stamp, stamp.data)
    kinds = [(s['owner'], s['kind']) for s in seats]
    gated = next(c['name'] for c in comps if c['gate'])
    assert (gated, 'sersic') in kinds and (gated, 'nuker') in kinds
    assert ('target', 'sersic') in kinds     # the refit is standard
    assert drops == {gated, 'target'}
    assert all(len(s['p0']) == recipe.SEAT_NPARAMS for s in seats)


def test_point_source_renders_at_subpixel_position():
    """The delta split bilinearly: the rendered base's centroid lands
    at the requested fractional position, not the nearest pixel."""
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    rows = [catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=50.0),
            catalog_row(stamp.wcs, stamp.cx + 30.37, stamp.cy + 10.61,
                        flux_nmgy=100.0, type_='PSF', shape_r=0.0)]
    comps = build_components(make_catalog(rows), stamp, psf, 1.3)
    point = next(c for c in comps if c['shape'] is None)
    yy, xx = np.indices(point['base'].shape)
    total = point['base'].sum()
    cx_m = float((point['base'] * xx).sum() / total)
    cy_m = float((point['base'] * yy).sum() / total)
    assert cx_m == pytest.approx(point['x'], abs=0.05)
    assert cy_m == pytest.approx(point['y'], abs=0.05)


def test_flags_carry_the_farfield_witness():
    witness = dict(cov=1.0, maskfrac_ap=0.01, twinfrac=1.0,
                   nbsub_ap_uJy=2.0, excess_growth_uJy=1.0,
                   ped_b_sb=0.001, r_conv_as=15.0, bg_sb=0.002,
                   farfield_sb=0.1234)
    flags = qa_flags(witness, n_comps=3, consumed=[])
    assert 'far=+0.1234' in flags
    witness['farfield_sb'] = None
    assert 'far=' not in qa_flags(witness, n_comps=3, consumed=[])


def test_target_is_the_closest_row_not_every_row_in_match():
    """A double nucleus straddling the requested position: the closest
    row is the target; the companion keeps its own component identity
    (it must stay eligible for a free seat, and must not be dropped by
    the target-seat replacement)."""
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    rows = [
        catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=40.0),
        catalog_row(stamp.wcs, stamp.cx + 2.4, stamp.cy,
                    flux_nmgy=200.0),      # brighter, 1.2" off-center
    ]
    comps = build_components(make_catalog(rows), stamp, psf, 1.3)
    names = sorted(c['name'] for c in comps)
    assert names.count('target') == 1
    target = next(c for c in comps if c['name'] == 'target')
    assert target['cat'] == pytest.approx(40.0 * 3.631, rel=1e-3)
    assert any(c['name'].startswith('src') for c in comps)


def test_free_seat_snap_never_lands_on_the_target():
    """A blended pair on a blurry grid: the free seat's snap finds the
    target's peak; the seat must keep its requested position instead of
    seating itself on top of the target."""
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    rows = [
        catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=200.0,
                    shape_r=2.0),
        catalog_row(stamp.wcs, stamp.cx + 4.0, stamp.cy, flux_nmgy=50.0,
                    shape_r=1.5),      # companion 2" east
    ]
    cat = make_catalog(rows)
    comps = build_components(cat, stamp, psf, 1.3)
    # a bright peak AT the target: the snap box (2") reaches it
    image = inject_sersic(stamp.data.shape, psf, flux=200.0,
                          reff_px=2.0 / PIX, n=2.0,
                          x=stamp.cx, y=stamp.cy)
    comp = next(c for c in comps if c['name'] != 'target')
    ra, dec = [float(v) for v in stamp.wcs.pixel_to_world_values(
        comp['x'], comp['y'])]
    seats, _ = build_seats(
        comps, {'free_seats': [{'ra': ra, 'dec': dec, 'snap': True}],
                'target_refit': False}, stamp, image)
    seat = next(s for s in seats if s['kind'] == 'sersic')
    sep = np.hypot(*(np.array(stamp.wcs.world_to_pixel_values(
        seat['ra'], seat['dec']), dtype=float)
        - np.array([stamp.cx, stamp.cy]))) * PIX
    assert sep > recipe.TARGET_MATCH_AS   # did not seat on the target


def test_build_seats_target_halo_grants_the_gated_pair():
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    _, comps = _components_with_gated(stamp, psf)
    seats, drops = build_seats(comps, {'target_halo': True}, stamp,
                               stamp.data)
    target_kinds = sorted(s['kind'] for s in seats
                          if s['owner'] == 'target')
    assert target_kinds == ['nuker', 'sersic']
    assert 'target' in drops


def test_harvest_include_target_writes_the_target_entry():
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    _, comps = _components_with_gated(stamp, psf)
    seats, _ = build_seats(comps, {'target_halo': True}, stamp,
                           stamp.data)
    params = np.concatenate([s['p0'] for s in seats])
    amps = [300.0] * len(seats)
    withheld: dict = {}
    harvest_seats(withheld, seats, params, amps, stamp,
                  band_key='Legacy_r')
    written: dict = {}
    harvest_seats(written, seats, params, amps, stamp,
                  band_key='Legacy_r', include_target=True)
    target = next(c for c in comps if c['name'] == 'target')
    target_name = registry_name(*[float(v) for v in
                                  stamp.wcs.pixel_to_world_values(
                                      target['x'], target['y'])])
    assert target_name not in withheld
    assert target_name in written
    assert len(written[target_name]['components']['Legacy_r']) == 2


def test_apply_registry_never_consumes_the_target(capsys):
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    _, comps = _components_with_gated(stamp, psf)
    target = next(c for c in comps if c['name'] == 'target')
    ra, dec = [float(v) for v in stamp.wcs.pixel_to_world_values(
        target['x'], target['y'])]
    registry = {'self': dict(ra=ra, dec=dec, components={
        'Legacy_r': [dict(kind='sersic', ra=ra, dec=dec, ellip=0.1,
                          pa=0.0, reff_as=1.5, n=2.0, flux_ref=300.0)]})}
    out, consumed = apply_registry(comps, registry, stamp, psf,
                                   'Legacy_r', 'Legacy')
    assert consumed == []
    assert [c['name'] for c in out] == [c['name'] for c in comps]
    assert "this field's target" in capsys.readouterr().out


def test_build_seats_patch_disables_refit_and_no_target_survives():
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    _, comps = _components_with_gated(stamp, psf)
    seats, drops = build_seats(comps, {'target_refit': False}, stamp,
                               stamp.data)
    assert 'target' not in drops
    # a catalog with no target row must not crash the refit block
    no_target = [c for c in comps if c['name'] != 'target']
    seats2, drops2 = build_seats(no_target, {}, stamp, stamp.data)
    assert 'target' not in drops2


# ------------------------------------
# Registry round trip
# ------------------------------------
def test_registry_harvest_then_consume_round_trip():
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    cat, comps = _components_with_gated(stamp, psf)
    seats, drops = build_seats(comps, {}, stamp, stamp.data)
    params = np.concatenate([s['p0'] for s in seats])
    amps = [300.0] * len(seats)
    registry: dict = {}
    touched = harvest_seats(registry, seats, params, amps, stamp,
                            band_key='Legacy_r')
    # the target seat is never written
    assert len(touched) == 1
    entry = registry[touched[0]]
    assert set(entry['components']) == {'Legacy_r'}
    assert len(entry['components']['Legacy_r']) == 2   # core + halo
    # re-harvest replaces, never doubles
    harvest_seats(registry, seats, params, amps, stamp,
                  band_key='Legacy_r')
    assert len(entry['components']['Legacy_r']) == 2

    # a TARGET-vantage entry consumes frozen: the matched catalog row
    # drops and the frozen comps take its place
    for rc in entry['components']['Legacy_r']:
        rc['vantage'] = 'target'
    fresh = build_components(cat, stamp, psf, 1.3)
    out, consumed = apply_registry(fresh, registry, stamp, psf,
                                   'Legacy_r', 'Legacy')
    assert consumed == touched
    names = [c['name'] for c in out]
    gated = next(c['name'] for c in fresh if c['gate'])
    assert gated not in names
    frozen = [c for c in out if c.get('reg')]
    assert len(frozen) == 2
    lo, hi = recipe.REGISTRY_AMP_BAND
    for c in frozen:
        assert c['amp_lohi'] == (lo * 300.0, hi * 300.0)


def test_registry_v2_vantage_tombstones_and_zero_amp():
    """Every record consumes FROZEN; a capped solve tombstones the band
    (gate dies, no repeat); a rail alone is a witness (harvest
    proceeds); a zero-amplitude seat is never stored; and a
    target-vantage record is never downgraded by a neighbor rewrite."""
    stamp = make_stamp(np.zeros((240, 240)))
    psf = moffat_kernel(1.3, PIX)
    cat, comps = _components_with_gated(stamp, psf)
    seats, _ = build_seats(comps, {}, stamp, stamp.data)
    params = np.concatenate([s['p0'] for s in seats])
    amps = [300.0] * len(seats)
    reg: dict = {}
    harvest_seats(reg, seats, params, amps, stamp, band_key='Legacy_r')
    name = next(iter(reg))
    assert reg[name]['components']['Legacy_r'][0]['vantage'] == 'neighbor'

    # neighbor-vantage entries still consume FROZEN: matched row drops,
    # frozen comps take its place, gate dies
    fresh = build_components(cat, stamp, psf, 1.3)
    gated = next(c['name'] for c in fresh if c['gate'])
    out, consumed = apply_registry(fresh, reg, stamp, psf,
                                   'Legacy_r', 'Legacy')
    assert consumed == [name]
    assert gated not in [c['name'] for c in out]
    assert sum(1 for c in out if c.get('reg')) == 2

    # a capped solve tombstones the band; consumption kills the gate
    harvest_seats(reg, seats, params, amps, stamp, band_key='Legacy_r',
                  solve_health=dict(capped=True))
    assert 'Legacy_r' in reg[name]['tombstones']
    assert 'Legacy_r' not in reg[name]['components']
    fresh2 = build_components(cat, stamp, psf, 1.3)
    out2, consumed2 = apply_registry(fresh2, reg, stamp, psf,
                                     'Legacy_r', 'Legacy')
    assert consumed2 == []
    assert not any(c['gate'] for c in out2 if c['name'] == gated)

    # a rail alone is a WITNESS, not a failure: validated envelope
    # fits carry beta rails throughout the sample -- harvest proceeds
    reg2: dict = {}
    harvest_seats(reg2, seats, params, amps, stamp, band_key='Legacy_r',
                  solve_health=dict(capped=False,
                                    at_bound=[f'{gated}.sersic.reff']))
    assert 'Legacy_r' in reg2[name]['components']
    assert 'Legacy_r' not in (reg2[name].get('tombstones') or {})

    # a zero-amplitude seat is a verdict, not a record: never stored.
    # src0's entry holds its two seats -- zero the sersic, the nuker
    # survives, and the dropped shape is nowhere in the record.
    reg3: dict = {}
    zero_amps = [0.0] + [300.0] * (len(seats) - 1)
    harvest_seats(reg3, seats, params, zero_amps, stamp,
                  band_key='Legacy_r')
    stored = reg3[name]['components'].get('Legacy_r', [])
    assert [rc['kind'] for rc in stored] == ['nuker']
    assert all(rc['flux_ref'] >= recipe.MARGIN_MIN_UJY for rc in stored)

    # a healthy re-harvest clears the tombstone...
    harvest_seats(reg, seats, params, amps, stamp, band_key='Legacy_r')
    assert 'Legacy_r' not in (reg[name].get('tombstones') or {})
    # ... and a target-vantage record outranks any neighbor rewrite
    for rc in reg[name]['components']['Legacy_r']:
        rc['vantage'] = 'target'
    marker = reg[name]['components']['Legacy_r'][0]['flux_ref']
    harvest_seats(reg, seats, params, [111.0] * len(seats), stamp,
                  band_key='Legacy_r')
    assert reg[name]['components']['Legacy_r'][0]['flux_ref'] == marker


def test_registry_save_is_atomic_and_round_trips(tmp_path):
    registry = {'J100000.00+020000.0': {'ra': 150.0, 'dec': 2.0,
                                        'components': {}}}
    path = tmp_path / 'registry.json'
    save_registry(registry, path)
    assert load_registry(path) == registry
    # write-then-replace leaves no sibling temp file behind
    assert list(tmp_path.iterdir()) == [path]


def test_registry_name_deterministic():
    assert registry_name(150.0, 2.0) == registry_name(150.0, 2.0)
    assert registry_name(150.0, 2.0) != registry_name(150.1, 2.0)


# ------------------------------------
# The joint fit
# ------------------------------------
def test_joint_fit_recovers_fluxes_and_background():
    rng = np.random.default_rng(3)
    shape_2d = (200, 200)
    stamp = make_stamp(np.zeros(shape_2d))
    psf = moffat_kernel(1.3, PIX)
    rows = [catalog_row(stamp.wcs, stamp.cx, stamp.cy, flux_nmgy=80.0),
            catalog_row(stamp.wcs, stamp.cx + 24, stamp.cy,
                        flux_nmgy=40.0)]
    cat = make_catalog(rows)
    truth = {}
    image = np.full(shape_2d, 0.02) + rng.normal(0.0, NOISE, shape_2d)
    for _, row in cat.iterrows():
        x, y = [float(v) for v in stamp.wcs.world_to_pixel(
            SkyCoord(row['ra'], row['dec'], unit='deg'))]
        image += inject_sersic(shape_2d, psf, flux=row['uJy'],
                               reff_px=row['shape_r'] / PIX, n=2.0,
                               x=x, y=y)
        truth[(round(x), round(y))] = row['uJy']
    stamp = make_stamp(image)
    comps = build_components(cat, stamp, psf, 1.3)
    # amplitude-only fit: no seats (refit exercised separately)
    fit = joint_fit(image, stamp.good, stamp, psf, comps, [], set())
    for comp, amp in zip(fit['fixed'], fit['amps']):
        assert amp == pytest.approx(comp['cat'], rel=0.05)
    assert fit['bg']['const'] == pytest.approx(0.02, abs=0.005)
    assert len(fit['track']) >= 2


def test_joint_fit_blind_scene_converges():
    rng = np.random.default_rng(4)
    image = np.full((160, 160), 0.05) + rng.normal(0.0, NOISE, (160, 160))
    stamp = make_stamp(image)
    fit = joint_fit(image, stamp.good, stamp, moffat_kernel(1.3, PIX),
                    [], [], set())
    assert len(fit['amps']) == 0
    assert fit['seat_amps'] == []
    assert fit['bg']['const'] == pytest.approx(0.05, abs=0.005)


def test_joint_fit_target_refit_recovers_shape():
    rng = np.random.default_rng(5)
    shape_2d = (200, 200)
    base_stamp = make_stamp(np.zeros(shape_2d))
    psf = moffat_kernel(1.3, PIX)
    cat = make_catalog([catalog_row(base_stamp.wcs, base_stamp.cx,
                                    base_stamp.cy, flux_nmgy=150.0,
                                    shape_r=1.5)])
    # inject a DIFFERENT reff than the catalog claims (2.5" vs 1.5")
    image = (inject_sersic(shape_2d, psf, flux=cat.iloc[0]['uJy'],
                           reff_px=2.5 / PIX, n=2.0,
                           x=base_stamp.cx, y=base_stamp.cy)
             + rng.normal(0.0, NOISE, shape_2d))
    stamp = make_stamp(image)
    comps = build_components(cat, stamp, psf, 1.3)
    seats, drops = build_seats(comps, {}, stamp, image)
    fit = joint_fit(image, stamp.good, stamp, psf, comps, seats, drops)
    info = fit['solve_info']
    assert info is not None and info['status'] > 0
    reff = info['params'][0] * PIX      # target seat leads the vector
    assert reff == pytest.approx(2.5, rel=0.15)
    assert fit['seat_amps'][0] == pytest.approx(cat.iloc[0]['uJy'],
                                                rel=0.1)


# ------------------------------------
# Measurement pieces
# ------------------------------------
def test_ped_fit_recovers_pedestal():
    rgrid = np.arange(2.0, 30.0, 1.0)
    b_true = 0.02
    enc = 120.0 + b_true * np.pi * rgrid ** 2
    F, b, rms = ped_fit(enc, rgrid)
    assert b == pytest.approx(b_true, rel=1e-6)
    assert rms < 1e-9


def test_plateau_hold_certifies_and_refuses():
    rgrid = np.arange(2.0, 30.0, 1.0)
    flat = np.full(len(rgrid), 100.0)
    assert plateau_hold(flat, 100.0, rgrid) == rgrid[0]
    drifting = 100.0 + 0.8 * (rgrid - rgrid[0])   # quiet but never holds
    assert plateau_hold(drifting, 100.0, rgrid) == -1.0


def test_twin_fill_reconstructs_masked_wedge():
    shape_2d = (160, 160)
    psf = moffat_kernel(1.3, PIX)
    stamp = make_stamp(np.zeros(shape_2d))
    image = inject_sersic(shape_2d, psf, flux=500.0, reff_px=6.0, n=1.0,
                          x=stamp.cx, y=stamp.cy)
    stamp = make_stamp(image)
    mask = np.zeros(shape_2d, bool)
    mask[int(stamp.cy) + 4:int(stamp.cy) + 14,
         int(stamp.cx) - 5:int(stamp.cx) + 6] = True
    fill = twin_fill(image, np.zeros_like(image), mask, stamp.good,
                     stamp, np.zeros_like(image), aperture_arcsec=12.0)
    # a symmetric profile's mirror restores the masked light almost
    # exactly (the model fill offered here is zero, so agreement with
    # the pre-mask image is the twin's doing -- within the clamp)
    lost = image[mask].sum()
    restored = fill['filled'][mask].sum()
    assert restored == pytest.approx(lost, rel=0.05)
    assert fill['twin_frac'] == 1.0


def test_build_mask_flood_catches_escaped_glow():
    shape_2d = (200, 200)
    psf = moffat_kernel(1.3, PIX)
    stamp = make_stamp(np.zeros(shape_2d))
    nx, ny = int(stamp.cx) + 36, int(stamp.cy)
    neighbor = inject_sersic(shape_2d, psf, flux=800.0, reff_px=4.0,
                             n=1.0, x=nx, y=ny)
    rng = np.random.default_rng(6)
    image = neighbor + rng.normal(0.0, NOISE, shape_2d)
    stamp = make_stamp(image)
    comps = [dict(name='src1', irow=1, cat=800.0, x=float(nx), y=float(ny),
                  gate=False, base=neighbor, flux0=800.0,
                  shape=dict(reff_px=4.0, n=1.0, ellip=0.0, theta=0.0,
                             pa=0.0))]
    # the fitted model under-claims (60%): escaped glow must flood
    fitted = {'src1': 0.6 * neighbor}
    neighbors = fitted['src1']
    mask, flood_ujy = build_mask(comps, fitted, [], stamp, 1.3,
                                 neighbors, neighbors, image, stamp.good)
    assert mask.any()
    assert flood_ujy > 0.0


def test_flux_error_models():
    stamp = make_stamp(np.random.default_rng(0).normal(0, NOISE, (160, 160)))
    err, model = flux_error(stamp, stamp.good, aperture_arcsec=12.0)
    n_aper = int((stamp.rr < 12.0).sum())
    assert model == 'skyrms'
    assert err == pytest.approx(NOISE * np.sqrt(n_aper), rel=0.2)
    stamp.invvar = np.full((160, 160), 1.0 / NOISE ** 2)
    err_ivm, model_ivm = flux_error(stamp, stamp.good,
                                    aperture_arcsec=12.0)
    assert model_ivm == 'ivm'
    assert err_ivm == pytest.approx(err, rel=0.2)


def test_witness_row_and_flags_tokens():
    rgrid = np.arange(2.0, 30.0, 1.0)
    stamp = make_stamp(np.zeros((200, 200)))
    enc = np.full(len(rgrid), 100.0)
    model = np.full(len(rgrid), 95.0)
    bg = dict(img=np.zeros((200, 200)), const=0.01,
              coefs=[0.01, 0.001, -0.001], n_rej=3, n_bins=500)
    witness = witness_row(enc, model, 98.0, stamp, stamp.good,
                          np.zeros((200, 200), bool), 0.5,
                          np.zeros((200, 200)), np.zeros((200, 200)),
                          bg, [0.012, 0.01], 1.5, 1.3, 'test',
                          rgrid=rgrid, aperture_arcsec=12.0)
    assert witness['f_ap_uJy'] == 100.0
    assert witness['excess_growth_uJy'] == 0.0
    assert witness['r_conv_as'] == rgrid[0]
    flags = qa_flags(witness, n_comps=0, consumed=[])
    assert 'cov=1.000' in flags and 'scene=none' in flags
    tokens = dict(t.split('=') for t in flags.split(';'))
    assert float(tokens['conv']) == rgrid[0]


def test_measurement_to_row_modes():
    witness = dict(cov=1.0, maskfrac_ap=0.0, twinfrac=0.0,
                   nbsub_ap_uJy=0.0, excess_growth_uJy=1.0,
                   ped_b_sb=0.0, r_conv_as=5.0, bg_sb=0.01,
                   target_model_uJy=90.0)
    measurement = dict(instrument='Legacy', band='r', flux_ujy=100.0,
                       flux_err_ujy=2.0, err_model='skyrms',
                       target_ra=RA, target_dec=DEC, witness=witness,
                       n_comps=2, registry_consumed=[])
    row = measurement_to_row(measurement)
    assert row['flux_uJy'] == 100.0
    assert row['source'] == 'sedphot_aperture_scene_skyrms'
    forced = measurement_to_row(measurement, mode='sersic')
    assert forced['flux_uJy'] == 90.0
    assert forced['source'] == 'sedphot_sersic_scene_skyrms'


# ------------------------------------
# End to end: one synthetic band through the driver
# ------------------------------------
def test_measure_band_end_to_end(tmp_path):
    from sedphot.measure.engine import measure_band
    from sedphot.results import ImageProduct

    rng = np.random.default_rng(7)
    nx = ny = 241
    wcs = make_wcs(nx, ny)
    psf = moffat_kernel(1.3, PIX)
    cx = cy = (nx - 1) / 2.0
    # the image is calibrated as nanomaggies (cf = 3.631), so inject the
    # truth flux in nmgy counts and expect it back in uJy
    flux_true_ujy = 300.0
    image = (inject_sersic((ny, nx), psf, flux=flux_true_ujy / 3.631,
                           reff_px=2.0 / PIX, n=2.0, x=cx, y=cy)
             + 0.03 + rng.normal(0.0, NOISE, (ny, nx)))
    path = tmp_path / 'legacy_r.fits'
    fits.PrimaryHDU(data=image.astype(np.float32),
                    header=wcs.to_header()).writeto(path)

    coord = SkyCoord(RA, DEC, unit='deg')
    cat = make_catalog([catalog_row(wcs, cx, cy,
                                    flux_nmgy=flux_true_ujy / 3.631,
                                    shape_r=2.0)])
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
    rgrid = np.arange(2.0, 26.0, 1.0)
    measurement, new_ref = measure_band(
        product, coord, scene, None, {}, aperture_arcsec=12.0,
        cutout_half_arcsec=55.0, rgrid=rgrid)
    assert measurement['flux_ujy'] == pytest.approx(flux_true_ujy,
                                                    rel=0.05)
    assert new_ref is not None        # the reference band solved seats
    witness = measurement['witness']
    assert witness['n_comps'] == 1
    assert witness['target_refit_x_cat'] == pytest.approx(1.0, abs=0.1)
    row = measurement_to_row(measurement)
    assert row['band'] == 'Legacy_r'
    assert 'cov=' in row['flags']


def test_masked_star_light_is_excluded_from_the_solve(tmp_path):
    """A bright star in the aperture zone whose profile starves: no
    column, footprint masked -- and the solve must not fit its pixels.
    The target refit stays true and the aperture flux stays the
    target's (the star reconstructed away by mask + twin fill)."""
    from sedphot.measure.engine import measure_band
    from sedphot.results import ImageProduct

    rng = np.random.default_rng(21)
    nx = ny = 241
    wcs = make_wcs(nx, ny)
    psf = moffat_kernel(1.3, PIX)
    cx = cy = (nx - 1) / 2.0
    dx_px = 8.0 / PIX
    f_target, f_star = 300.0, 900.0
    image = (inject_sersic((ny, nx), psf, flux=f_target / 3.631,
                           reff_px=2.0 / PIX, n=2.0, x=cx, y=cy)
             + inject_sersic((ny, nx), psf, flux=f_star / 3.631,
                             reff_px=0.8, n=0.6, x=cx + dx_px, y=cy)
             + 0.03 + rng.normal(0.0, NOISE, (ny, nx)))
    path = tmp_path / 'legacy_r.fits'
    fits.PrimaryHDU(data=image.astype(np.float32),
                    header=wcs.to_header()).writeto(path)

    cat = make_catalog([
        catalog_row(wcs, cx, cy, flux_nmgy=f_target / 3.631, shape_r=2.0),
        catalog_row(wcs, cx + dx_px, cy, flux_nmgy=f_star / 3.631,
                    type_='PSF', shape_r=0.0),
    ])
    star_sky = wcs.pixel_to_world(cx + dx_px, cy)
    stars = pd.DataFrame([dict(ra=float(star_sky.ra.deg),
                               dec=float(star_sky.dec.deg),
                               phot_g_mean_mag=16.0)])
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

    witness = measurement['witness']
    assert witness['stars'][0]['mode'] == 'masked'
    # the solve never saw the star: the refit is the target's own
    assert witness['target_refit_x_cat'] == pytest.approx(1.0, abs=0.15)
    # and the aperture flux is the target's, the star filled away
    # (masked-star fill geometry is estimator-sensitive at ~1.5%; see
    # the deblend test)
    assert measurement['flux_ujy'] == pytest.approx(f_target, rel=0.12)


def test_target_system_member_is_kept_in_the_aperture(tmp_path):
    """A declared system member (second nucleus) is neither subtracted
    nor masked: the aperture flux carries the pair, while an undeclared
    companion is deblended away."""
    from sedphot.measure.engine import measure_band
    from sedphot.results import ImageProduct

    rng = np.random.default_rng(11)
    nx = ny = 241
    wcs = make_wcs(nx, ny)
    psf = moffat_kernel(1.3, PIX)
    cx = cy = (nx - 1) / 2.0
    dx_px = 6.0 / PIX                      # companion 6" east, in-aperture
    f_main, f_comp = 300.0, 150.0
    image = (inject_sersic((ny, nx), psf, flux=f_main / 3.631,
                           reff_px=2.0 / PIX, n=2.0, x=cx, y=cy)
             + inject_sersic((ny, nx), psf, flux=f_comp / 3.631,
                             reff_px=1.5 / PIX, n=1.5, x=cx + dx_px, y=cy)
             + 0.03 + rng.normal(0.0, NOISE, (ny, nx)))
    path = tmp_path / 'legacy_r.fits'
    fits.PrimaryHDU(data=image.astype(np.float32),
                    header=wcs.to_header()).writeto(path)

    comp_ra, comp_dec = [float(v) for v in
                         wcs.pixel_to_world_values(cx + dx_px, cy)]
    cat = make_catalog([
        catalog_row(wcs, cx, cy, flux_nmgy=f_main / 3.631, shape_r=2.0),
        catalog_row(wcs, cx + dx_px, cy, flux_nmgy=f_comp / 3.631,
                    shape_r=1.5),
    ])
    stars = pd.DataFrame(columns=['ra', 'dec', 'phot_g_mean_mag',
                                  'parallax', 'parallax_error', 'pmra',
                                  'pmra_error', 'pmdec', 'pmdec_error',
                                  'ruwe'])
    product = ImageProduct(provider='legacy', instrument='Legacy',
                           band='r', path=str(path), calib='nmgy',
                           invvar_path=None, seeing_arcsec=1.3,
                           wave_um=0.64)
    coord = SkyCoord(RA, DEC, unit='deg')
    rgrid = np.arange(2.0, 26.0, 1.0)

    def run(patches):
        scene = dict(cat=cat.copy(), stars=stars, patches=patches,
                     registry={}, registry_path=None)
        measurement, _ = measure_band(
            product, coord, scene, None, {}, aperture_arcsec=12.0,
            cutout_half_arcsec=55.0, rgrid=rgrid)
        return measurement

    deblended = run({})
    system = run({'target_system': [{'ra': comp_ra, 'dec': comp_dec}]})
    # a 6" pair at 1.3" seeing carries real deblend systematics, and
    # the residual-skirt handling is estimator-sensitive at the ~1.5%
    # level (measured mean-vs-median bin statistic A/B); the tolerance
    # reflects the geometry, not the machinery
    assert deblended['flux_ujy'] == pytest.approx(f_main, rel=0.09)
    assert system['flux_ujy'] == pytest.approx(f_main + f_comp, rel=0.06)
    # the member is no longer masked, so the mask shrinks
    assert system['mask'].sum() < deblended['mask'].sum()


# ------------------------------------
# Transfer band: reference shapes carry, fluxes re-solve at color
# ------------------------------------
def test_order_bands_puts_the_reference_first():
    from sedphot.measure.engine import order_bands
    from sedphot.results import ImageProduct

    def product(band):
        return ImageProduct(provider='legacy', instrument='Legacy',
                            band=band, path='', calib='nmgy')

    ordered = order_bands([product('g'), product('z'), product('r')])
    assert [p.band for p in ordered] == ['r', 'z', 'g']


def test_transfer_band_recovers_sibling_fluxes(tmp_path):
    from sedphot.measure.engine import measure_band
    from sedphot.results import ImageProduct

    nx = ny = 241
    wcs = make_wcs(nx, ny)
    psf = moffat_kernel(1.3, PIX)
    cx = cy = (nx - 1) / 2.0
    cf = 3.631
    color = 0.5              # g = color * r, in catalog and in the sky

    # target + one ungated neighbor outside the aperture (15") whose
    # wings still contaminate it; truths are measured from the rendered
    # injections (the center-sampled render makes them differ from the
    # nominal flux at this reff)
    tgt = inject_sersic((ny, nx), psf, flux=300.0 / cf,
                        reff_px=2.0 / PIX, n=2.0, x=cx, y=cy)
    nbr = inject_sersic((ny, nx), psf, flux=150.0 / cf,
                        reff_px=1.5 / PIX, n=2.0, x=cx + 30, y=cy)
    yy, xx = np.mgrid[0:ny, 0:nx]
    in_ap = np.hypot(xx - cx, yy - cy) * PIX < 12.0
    tgt_total = tgt.sum() * cf
    tgt_ap = (tgt * in_ap).sum() * cf
    nbr_ap = (nbr * in_ap).sum() * cf

    # g catalog fluxes carry the color the transfer leash must scale by
    row_t = catalog_row(wcs, cx, cy, flux_nmgy=300.0 / cf, shape_r=2.0)
    row_n = catalog_row(wcs, cx + 30, cy, flux_nmgy=150.0 / cf,
                        shape_r=1.5)
    for row in (row_t, row_n):
        row['flux_g'] = row['flux_r'] * color
    cat = make_catalog([row_t, row_n])

    def write_band(band, scale, seed):
        rng = np.random.default_rng(seed)
        image = (scale * (tgt + nbr) + 0.03
                 + rng.normal(0.0, NOISE, (ny, nx)))
        path = tmp_path / f'legacy_{band}.fits'
        fits.PrimaryHDU(data=image.astype(np.float32),
                        header=wcs.to_header()).writeto(path)
        return ImageProduct(provider='legacy', instrument='Legacy',
                            band=band, path=str(path), calib='nmgy',
                            invvar_path=None, seeing_arcsec=1.3,
                            wave_um=0.64 if band == 'r' else 0.48)

    product_r = write_band('r', 1.0, 11)
    product_g = write_band('g', color, 12)
    coord = SkyCoord(RA, DEC, unit='deg')
    stars = pd.DataFrame(columns=['ra', 'dec', 'phot_g_mean_mag',
                                  'parallax', 'parallax_error', 'pmra',
                                  'pmra_error', 'pmdec', 'pmdec_error',
                                  'ruwe'])
    scene = dict(cat=cat, stars=stars, patches={}, registry={},
                 registry_path=None)
    rgrid = np.arange(2.0, 26.0, 1.0)
    caches = {}

    measurement_r, ref = measure_band(
        product_r, coord, scene, None, caches, aperture_arcsec=12.0,
        cutout_half_arcsec=55.0, rgrid=rgrid)
    assert measurement_r['flux_ujy'] == pytest.approx(tgt_ap, rel=0.05)
    assert measurement_r['witness']['nbsub_ap_uJy'] == pytest.approx(
        nbr_ap, rel=0.2)
    assert ref is not None and 'target' in ref['drops']
    assert ref['col_flux'][0] == pytest.approx(tgt_total, rel=0.1)

    measurement_g, ref_g = measure_band(
        product_g, coord, scene, ref, caches, aperture_arcsec=12.0,
        cutout_half_arcsec=55.0, rgrid=rgrid)
    assert ref_g is None      # only the reference band returns one
    assert measurement_g['witness']['seat_owners'] == ['target']
    # the sibling flux re-solves at its own color inside the leash
    assert measurement_g['flux_ujy'] == pytest.approx(color * tgt_ap,
                                                      rel=0.05)
    # neighbor subtraction lands at the sibling color too
    assert measurement_g['witness']['nbsub_ap_uJy'] == pytest.approx(
        color * nbr_ap, rel=0.2)
    # the frozen-shape target model re-solves at the sibling color
    assert measurement_g['witness']['target_model_uJy'] == pytest.approx(
        color * tgt_total, rel=0.1)


def test_prepare_scene_cone_scales_with_the_stamp(tmp_path, monkeypatch):
    """The query cone floors at the recipe radius for the default stamp
    and grows past the corners of a larger one, with the radius keyed
    into the cache names only when it exceeds the floor."""
    from sedphot.measure import engine

    calls = {}

    def fake_scene(coord, radius, *, dr, min_flux_nmgy, cache_path):
        calls['scene'] = (radius, Path(cache_path).name)
        return pd.DataFrame()

    def fake_cone(coord, radius, *, cache_path):
        calls['gaia'] = (radius, Path(cache_path).name)
        return pd.DataFrame()

    monkeypatch.setattr(engine, 'query_scene', fake_scene)
    monkeypatch.setattr(engine.gaia, 'query_cone', fake_cone)
    coord = SkyCoord(RA, DEC, unit='deg')

    engine.prepare_scene(coord, phot_dir=tmp_path, out_dir=tmp_path,
                         aperture_arcsec=12.0, cutout_half_arcsec=60.0)
    assert calls['scene'] == (recipe.QUERY_RADIUS_AS, 'tractor_scene_dr9.csv')
    assert calls['gaia'] == (recipe.QUERY_RADIUS_AS, 'gaia_scene.csv')

    engine.prepare_scene(coord, phot_dir=tmp_path, out_dir=tmp_path,
                         aperture_arcsec=12.0, cutout_half_arcsec=120.0)
    grown = 120.0 * np.sqrt(2.0) + recipe.QUERY_PAD_AS
    tag = int(round(grown))
    assert calls['scene'][0] == pytest.approx(grown)
    assert calls['scene'][1] == f'tractor_scene_dr9_r{tag}.csv'
    assert calls['gaia'][1] == f'gaia_scene_r{tag}.csv'
