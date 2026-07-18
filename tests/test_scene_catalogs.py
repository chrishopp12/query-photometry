"""Offline tests for the scene-input catalogs: Gaia cone and Tractor scene."""
from __future__ import annotations

import time

import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

from sedphot.catalogs import gaia, legacy
from sedphot.units import NANOMAGGY_TO_UJY

COORD = SkyCoord(150.0, 2.0, unit='deg')


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Keep a regression from sleeping through the retry backoff."""
    monkeypatch.setattr(time, 'sleep', lambda s: None)


# ------------------------------------
# Fakes and synthetic frames
# ------------------------------------
class NoNetworkTap:
    """TapPlus stand-in for cache-hit tests: any contact is a failure."""

    def __init__(self, url):
        raise AssertionError("cache hit must not touch the network")


def serve_once_tap(frame, queries):
    """Build a TapPlus stand-in serving `frame` once; a second contact raises."""

    class FakeJob:
        def get_results(self):
            class Result:
                def to_pandas(self):
                    return frame.copy()
            return Result()

    class FakeTap:
        def __init__(self, url):
            pass

        def launch_job(self, query):
            queries.append(query)
            if len(queries) > 1:
                raise AssertionError("the cache should answer the second call")
            return FakeJob()

    return FakeTap


def scene_frame():
    """Three Tractor rows in dr9 scene_cols order, deliberately not flux-sorted."""
    return pd.DataFrame({
        'ra': [150.0010, 150.0020, 150.0030],
        'dec': [2.0010, 2.0020, 2.0030],
        'type': ['SER', 'PSF', 'DEV'],
        'sersic': [3.0, 0.0, 4.0],
        'shape_r': [4.8, 0.0, 2.0],
        'shape_e1': [0.1, 0.0, 0.05],
        'shape_e2': [0.0, 0.0, 0.02],
        'flux_g': [5.0, 20.0, 2.0],
        'flux_r': [10.0, 40.0, 4.0],
        'flux_z': [15.0, 60.0, 6.0],
        'psfsize_g': [1.5, 1.5, 1.5],
        'psfsize_r': [1.3, 1.3, 1.3],
        'psfsize_z': [1.2, 1.2, 1.2],
        'rchisq_g': [1.0, 1.1, 0.9],
        'rchisq_r': [1.2, 0.8, 5.0],
        'rchisq_z': [1.1, 1.0, 1.0],
        'fracflux_r': [0.05, 0.01, 0.30],
        'fracin_r': [0.98, 1.00, 0.90],
    })


def gaia_frame():
    """Two Gaia rows in GAIA_COLS order."""
    return pd.DataFrame({
        'ra': [150.0040, 150.0050],
        'dec': [2.0040, 2.0050],
        'phot_g_mean_mag': [17.2, 19.8],
        'parallax': [2.5, 0.1],
        'parallax_error': [0.1, 0.4],
        'pmra': [12.0, 0.3],
        'pmra_error': [0.2, 0.5],
        'pmdec': [-4.0, 0.1],
        'pmdec_error': [0.2, 0.5],
        'ruwe': [1.0, 1.4],
    })


# ------------------------------------
# Column contracts
# ------------------------------------
def test_gaia_cols_are_the_expected_set():
    assert gaia.GAIA_COLS == (
        'ra', 'dec', 'phot_g_mean_mag',
        'parallax', 'parallax_error',
        'pmra', 'pmra_error', 'pmdec', 'pmdec_error',
        'ruwe',
    )
    assert tuple(gaia_frame().columns) == gaia.GAIA_COLS


def test_scene_cols_are_the_expected_set():
    assert legacy.scene_cols('dr9') == (
        'ra', 'dec', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
        'flux_g', 'flux_r', 'flux_z',
        'psfsize_g', 'psfsize_r', 'psfsize_z',
        'rchisq_g', 'rchisq_r', 'rchisq_z',
        'fracflux_r', 'fracin_r',
    )
    assert legacy.scene_cols('dr10') == (
        'ra', 'dec', 'type', 'sersic', 'shape_r', 'shape_e1', 'shape_e2',
        'flux_g', 'flux_r', 'flux_i', 'flux_z',
        'psfsize_g', 'psfsize_r', 'psfsize_i', 'psfsize_z',
        'rchisq_g', 'rchisq_r', 'rchisq_i', 'rchisq_z',
        'fracflux_r', 'fracin_r',
    )
    assert tuple(scene_frame().columns) == legacy.scene_cols('dr9')


# ------------------------------------
# Cache-first: an existing CSV answers with no network contact
# ------------------------------------
def test_query_cone_cache_hit_skips_network(tmp_path, monkeypatch):
    monkeypatch.setattr(gaia, 'TapPlus', NoNetworkTap)
    cache = tmp_path / 'gaia.csv'
    gaia_frame().to_csv(cache, index=False)
    out = gaia.query_cone(COORD, 100.0, cache_path=str(cache))
    pd.testing.assert_frame_equal(out, gaia_frame())


def test_query_scene_cache_hit_skips_network(tmp_path, monkeypatch):
    monkeypatch.setattr(legacy, 'TapPlus', NoNetworkTap)
    cache = tmp_path / 'scene.csv'
    scene_frame().to_csv(cache, index=False)
    out = legacy.query_scene(COORD, 100.0, cache_path=cache)
    assert list(out.columns) == list(legacy.scene_cols('dr9')) + ['uJy']
    assert out['flux_r'].tolist() == [40.0, 10.0, 4.0]      # brightest-first
    assert out['type'].tolist() == ['PSF', 'SER', 'DEV']    # rows move together
    assert out.index.tolist() == [0, 1, 2]                  # fresh index
    assert out['uJy'].tolist() == pytest.approx(
        [flux * NANOMAGGY_TO_UJY for flux in (40.0, 10.0, 4.0)])


# ------------------------------------
# Network path: query once, cache, reuse
# ------------------------------------
def test_query_scene_network_writes_cache_then_reuses(tmp_path, monkeypatch):
    queries = []
    monkeypatch.setattr(legacy, 'TapPlus', serve_once_tap(scene_frame(), queries))
    cache = tmp_path / 'scene.csv'
    first = legacy.query_scene(COORD, 100.0, cache_path=cache)
    assert cache.exists()
    assert len(queries) == 1
    assert 'ls_dr9.tractor' in queries[0]
    assert 'brick_primary = 1' in queries[0]
    assert 'flux_r > 0.5' in queries[0]
    second = legacy.query_scene(COORD, 100.0, cache_path=cache)
    assert len(queries) == 1
    pd.testing.assert_frame_equal(first, second)
    assert first['flux_r'].tolist() == [40.0, 10.0, 4.0]


def test_query_cone_network_writes_cache_then_reuses(tmp_path, monkeypatch):
    queries = []
    monkeypatch.setattr(gaia, 'TapPlus', serve_once_tap(gaia_frame(), queries))
    cache = tmp_path / 'gaia.csv'
    first = gaia.query_cone(COORD, 100.0, cache_path=cache)
    assert cache.exists()
    assert len(queries) == 1
    assert 'gaia_dr3.gaia_source' in queries[0]
    second = gaia.query_cone(COORD, 100.0, cache_path=cache)
    assert len(queries) == 1
    pd.testing.assert_frame_equal(first, second)


def test_query_scene_without_cache_returns_processed_frame(monkeypatch):
    queries = []
    monkeypatch.setattr(legacy, 'TapPlus', serve_once_tap(scene_frame(), queries))
    out = legacy.query_scene(COORD, 100.0)
    assert len(queries) == 1
    assert out['flux_r'].tolist() == [40.0, 10.0, 4.0]
    assert 'uJy' in out.columns


def test_query_scene_selects_release_table_and_rejects_unknown(monkeypatch):
    queries = []
    monkeypatch.setattr(legacy, 'TapPlus', serve_once_tap(scene_frame(), queries))
    legacy.query_scene(COORD, 100.0, dr='dr10')
    assert 'ls_dr10.tractor' in queries[0]
    assert 'psfsize_i' in queries[0]       # dr10 carries the i-band set
    with pytest.raises(ValueError, match='unknown Legacy release'):
        legacy.query_scene(COORD, 100.0, dr='dr8')
