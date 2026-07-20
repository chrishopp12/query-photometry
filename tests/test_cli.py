"""CLI exit codes: single-provider verbs must fail loudly."""
import sys

import pytest

from sedphot import cli
from sedphot.results import STATUS_ERROR, STATUS_OK, ProviderResult


def _fake_run_spherex(status):
    def fake(*args, **kwargs):
        return ProviderResult(provider='spherex', status=status, message='')
    return fake


def test_spherex_error_exits_nonzero(monkeypatch):
    monkeypatch.setattr(cli, 'run_spherex', _fake_run_spherex(STATUS_ERROR))
    monkeypatch.setattr(sys, 'argv',
                        ['sedphot', 'spherex', '--ra', '10.0', '--dec', '20.0'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_spherex_ok_exits_zero(monkeypatch):
    monkeypatch.setattr(cli, 'run_spherex', _fake_run_spherex(STATUS_OK))
    monkeypatch.setattr(sys, 'argv',
                        ['sedphot', 'spherex', '--ra', '10.0', '--dec', '20.0'])
    assert cli.main() is None


def test_run_stage_failure_exits_nonzero(monkeypatch):
    monkeypatch.setattr(cli, 'run_all',
                        lambda *a, **k: {'measure': 'RuntimeError: boom'})
    monkeypatch.setattr(sys, 'argv',
                        ['sedphot', 'run', '--ra', '10.0', '--dec', '20.0'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_run_clean_exits_zero(monkeypatch):
    monkeypatch.setattr(cli, 'run_all', lambda *a, **k: {})
    monkeypatch.setattr(sys, 'argv',
                        ['sedphot', 'run', '--ra', '10.0', '--dec', '20.0'])
    assert cli.main() is None
