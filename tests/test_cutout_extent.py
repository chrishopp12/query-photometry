"""Short-cutout detection and the fetch-source stamp."""
from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from sedphot.images.legacy import PIXSCALE, fetch, stamp_spans_request
from sedphot.pipeline import image_fetch_sources

RA0, DEC0 = 210.0, 30.0
COORD = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
SIZE = 120.0


def _stamp(ny, nx, *, dec_shift_arcsec=0.0):
    """A TAN grid centered on the target, optionally shifted in Dec."""
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    w.wcs.cdelt = [-PIXSCALE / 3600.0, PIXSCALE / 3600.0]
    w.wcs.crval = [RA0, DEC0 + dec_shift_arcsec / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return np.zeros((ny, nx), dtype="f4"), w


def test_a_full_stamp_passes() -> None:
    n = int(round(SIZE / PIXSCALE))
    data, w = _stamp(n, n)
    assert stamp_spans_request(data, w, COORD, SIZE)


def test_the_clipped_stamp_is_caught() -> None:
    # The shape actually returned by the single-brick cutout route for a
    # 120" request: 103 rows short, center pulled off in Dec.
    data, w = _stamp(355, 459, dec_shift_arcsec=13.7)
    assert not stamp_spans_request(data, w, COORD, SIZE)


def test_a_centered_but_short_stamp_is_caught() -> None:
    # Short on BOTH sides rather than one: still not the requested box.
    data, w = _stamp(300, 300)
    assert not stamp_spans_request(data, w, COORD, SIZE)


def test_service_rounding_is_tolerated() -> None:
    # A pixel or two either way is the service's own rounding, not a clip.
    n = int(round(SIZE / PIXSCALE))
    for delta in (-2, -1, 0, 1, 2):
        data, w = _stamp(n + delta, n + delta)
        assert stamp_spans_request(data, w, COORD, SIZE), delta


def test_an_off_center_full_size_stamp_is_caught() -> None:
    # Right pixel count, wrong placement: the far side is still missing.
    n = int(round(SIZE / PIXSCALE))
    data, w = _stamp(n, n, dec_shift_arcsec=30.0)
    assert not stamp_spans_request(data, w, COORD, SIZE)


def test_the_tolerance_is_not_a_free_pass() -> None:
    n = int(round(SIZE / PIXSCALE))
    data, w = _stamp(n, n, dec_shift_arcsec=6.0)
    assert not stamp_spans_request(data, w, COORD, SIZE, tol_arcsec=2.0)
    # Widening the slack past the shortfall accepts it, so the tolerance
    # is doing the work it claims and not silently swallowing more.
    assert stamp_spans_request(data, w, COORD, SIZE, tol_arcsec=8.0)


# ------------------------------------
# Route selection
# ------------------------------------

def _stub_routes(monkeypatch, *, viewer, noirlab):
    """Record which routes are attempted; each returns a marker or raises."""
    tried = []

    def _viewer(*a, **k):
        tried.append('viewer')
        if viewer is None:
            raise ConnectionError("NERSC viewer down")
        return viewer

    def _noirlab(*a, **k):
        tried.append('noirlab')
        if noirlab is None:
            raise ConnectionError("NOIRLab down")
        return noirlab

    monkeypatch.setattr('sedphot.images.legacy._fetch_cutouts', _viewer)
    monkeypatch.setattr('sedphot.images.legacy._fetch_noirlab', _noirlab)
    return tried


def test_auto_falls_back_to_the_second_service(tmp_path, monkeypatch) -> None:
    tried = _stub_routes(monkeypatch, viewer=None, noirlab=['product'])
    out = fetch(COORD, cache_dir=tmp_path, route='auto')
    assert out == ['product']
    assert tried == ['viewer', 'noirlab']


def test_a_pinned_route_does_not_substitute_the_other(tmp_path,
                                                      monkeypatch) -> None:
    # The whole point of pinning: an outage must surface, not silently
    # re-grid the sample onto the other service.
    tried = _stub_routes(monkeypatch, viewer=None, noirlab=['product'])
    result = fetch(COORD, cache_dir=tmp_path, route='viewer')
    assert 'noirlab' not in tried
    assert getattr(result, 'status', None) in ('error', 'no_coverage')


def test_a_pinned_route_uses_only_the_one_asked_for(tmp_path,
                                                    monkeypatch) -> None:
    tried = _stub_routes(monkeypatch, viewer=['viewer_product'],
                         noirlab=['noirlab_product'])
    assert fetch(COORD, cache_dir=tmp_path,
                 route='noirlab') == ['noirlab_product']
    assert tried == ['noirlab']


def test_an_unknown_route_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown Legacy route"):
        fetch(COORD, cache_dir=tmp_path, route='nersc')
