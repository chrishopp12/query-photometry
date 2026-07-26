"""Sidecar reconstruction: the stored-background evaluators must reproduce
their producers exactly, the pin record must name the grid its shapes were
solved on, and a record that cannot describe the seat list must be refused
rather than rendered (the pinned path stands on all three)."""
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.measure.background import (bin_plane, residual_mesh,
                                        eval_plane, eval_mesh)
from sedphot.measure.psf import moffat_kernel
from sedphot.measure.solve import pinned_fit
from sedphot.measure.stamp import Stamp, radii_arcsec


def _geom(shape=(120, 120), pixscale=0.262):
    ny, nx = shape
    yy, xx = np.indices(shape)
    rr = np.hypot(xx - nx / 2, yy - ny / 2) * pixscale
    return rr, np.ones(shape, bool), pixscale


def test_eval_plane_roundtrips_bin_plane():
    shape = (120, 120)
    rr, good, pix = _geom(shape)
    yy, xx = np.indices(shape)
    image = 0.5 + 0.001 * (xx - shape[1] / 2) - 0.002 * (yy - shape[0] / 2)
    bg = bin_plane(image, good, rr, pix)
    got = eval_plane(bg['coefs'], shape)
    assert np.allclose(got, bg['img'], atol=1e-9)


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
    Doubling seat_pix must be identical to doubling the radial entries by
    hand (size and center offsets; ellipticity and PA do not scale).
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
    # test fails if the rescale is ever dropped again.
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
