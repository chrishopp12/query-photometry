"""AllWISE provider: the IRSA transport rides out a blip, and a failure that
outlives the retries is reported rather than raised. No network."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from sedphot.catalogs import allwise

COORD = SkyCoord(150.0, 2.0, unit='deg')


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """Never sleep through the backoff."""
    monkeypatch.setattr(time, 'sleep', lambda seconds: None)


def allwise_table():
    """One source: W1/W2 measured, W3/W4 null-sigma upper limits."""
    return Table({
        'ra': [150.00001], 'dec': [2.00001],
        'cc_flags': ['0000'], 'ext_flg': [0],
        'w1mpro': [12.0], 'w1sigmpro': [0.02],
        'w2mpro': [11.8], 'w2sigmpro': [0.03],
        'w3mpro': [8.0], 'w3sigmpro': [np.nan],
        'w4mpro': [np.nan], 'w4sigmpro': [np.nan],
    })


def _serve(responder):
    """An Irsa stand-in whose query_region is the given callable."""
    return SimpleNamespace(query_region=responder)


def test_a_transient_irsa_failure_is_retried(monkeypatch):
    """One service blip must not cost the whole provider its match."""
    calls = []

    def flaky(coord, **kwargs):
        calls.append(kwargs['catalog'])
        if len(calls) < 3:
            raise ConnectionError("IRSA blip")
        return allwise_table()

    monkeypatch.setattr(allwise, 'Irsa', _serve(flaky))
    rows = allwise._query_once(COORD, 2.0)
    assert calls == [allwise.ALLWISE_CAT] * 3
    # Only the bands with a real sigma; a null w*sigmpro is an upper limit.
    assert [row['band'] for row in rows] == ['WISE_W1', 'WISE_W2']


def test_a_failure_outliving_the_retries_reports_no_rows(monkeypatch, capsys):
    calls = []

    def dead(coord, **kwargs):
        calls.append(1)
        raise ConnectionError("IRSA is down")

    monkeypatch.setattr(allwise, 'Irsa', _serve(dead))
    assert allwise._query_once(COORD, 2.0) == []
    assert len(calls) > 1                       # it did retry first
    assert "Query error" in capsys.readouterr().out


def test_a_clean_query_is_asked_once(monkeypatch):
    calls = []

    def healthy(coord, **kwargs):
        calls.append(1)
        return allwise_table()

    monkeypatch.setattr(allwise, 'Irsa', _serve(healthy))
    assert allwise._query_once(COORD, 2.0)
    assert len(calls) == 1
