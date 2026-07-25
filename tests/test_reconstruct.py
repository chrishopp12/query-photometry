"""Sidecar reconstruction: the stored-background evaluators must reproduce
their producers exactly (the pinned path stands on this)."""
import numpy as np

from sedphot.measure.background import (bin_plane, residual_mesh,
                                        eval_plane, eval_mesh)


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
