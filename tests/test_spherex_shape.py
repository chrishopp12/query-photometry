"""SPHEREx shape resolution: retry the Tractor lookup, abort on failure."""
import time

import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

import sedphot.catalogs.legacy as legacy
import sedphot.spherex as spherex_mod
from sedphot import pipeline
from sedphot.results import STATUS_ERROR

COORD = SkyCoord(217.0, 56.9, unit='deg')


# ------------------------------------
# query_shape transport behavior
# ------------------------------------
def test_query_shape_retries_then_raises(monkeypatch):
    monkeypatch.setattr(time, 'sleep', lambda s: None)
    calls = []

    class FakeTap:
        def __init__(self, url):
            pass

        def launch_job(self, query):
            calls.append(1)
            raise ConnectionError("HTTP 504")

    monkeypatch.setattr(legacy, 'TapPlus', FakeTap)
    with pytest.raises(RuntimeError, match="after retries"):
        legacy.query_shape(COORD)
    assert len(calls) == 3  # the transient policy actually retried


def test_query_shape_empty_is_none(monkeypatch):
    class FakeJob:
        def get_results(self):
            class Result:
                def to_pandas(self):
                    return pd.DataFrame()
            return Result()

    class FakeTap:
        def __init__(self, url):
            pass

        def launch_job(self, query):
            return FakeJob()

    monkeypatch.setattr(legacy, 'TapPlus', FakeTap)
    assert legacy.query_shape(COORD) is None


# ------------------------------------
# run_spherex abort behavior
# ------------------------------------
def _forbid_fetch(monkeypatch):
    submitted = []
    monkeypatch.setattr(spherex_mod, 'fetch',
                        lambda *a, **k: submitted.append(1))
    return submitted


def test_lookup_failure_aborts_before_submitting(monkeypatch, tmp_path):
    monkeypatch.setattr(legacy, 'query_shape',
                        lambda coord, **kw: (_ for _ in ()).throw(
                            RuntimeError("shape query failed after retries")))
    submitted = _forbid_fetch(monkeypatch)
    result = pipeline.run_spherex(COORD, 'x', tmp_path, model='sersic')
    assert result.status == STATUS_ERROR
    assert 'aborting' in result.message
    assert not submitted


def test_no_usable_shape_aborts_before_submitting(monkeypatch, tmp_path):
    monkeypatch.setattr(legacy, 'query_shape', lambda coord, **kw: None)
    submitted = _forbid_fetch(monkeypatch)
    result = pipeline.run_spherex(COORD, 'x', tmp_path, model='sersic')
    assert result.status == STATUS_ERROR
    assert '--model psf' in result.message
    assert not submitted
