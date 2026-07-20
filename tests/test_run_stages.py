"""run_all stage isolation: one dead stage cannot cost the others."""
import json

from astropy.coordinates import SkyCoord

from sedphot import pipeline
from sedphot.results import STATUS_ERROR, ProviderResult

COORD = SkyCoord(150.0, 2.0, unit='deg')


def _quiet(calls, name):
    def stage(*args, **kwargs):
        calls.append(name)
    return stage


def test_measure_failure_recorded_and_run_continues(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'run_catalogs', _quiet(calls, 'catalogs'))

    def boom(*args, **kwargs):
        raise RuntimeError('scene catalog outage')

    monkeypatch.setattr(pipeline, 'run_measure', boom)
    monkeypatch.setattr(pipeline, 'run_sed', _quiet(calls, 'sed'))
    failures = pipeline.run_all(COORD, 'tgt', tmp_path)
    assert calls == ['catalogs', 'sed']
    assert 'RuntimeError' in failures['measure']
    report = json.loads(
        (tmp_path / 'Photometry' / 'coverage_measure.json').read_text())
    assert report['measure']['status'] == STATUS_ERROR


def test_spherex_error_status_is_a_failure(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(pipeline, 'run_catalogs', _quiet(calls, 'catalogs'))
    monkeypatch.setattr(pipeline, 'run_measure', _quiet(calls, 'measure'))
    monkeypatch.setattr(pipeline, 'run_sed', _quiet(calls, 'sed'))
    monkeypatch.setattr(pipeline, 'run_spherex', lambda *a, **k: ProviderResult(
        provider='spherex', status=STATUS_ERROR, message='no usable shape'))
    failures = pipeline.run_all(COORD, 'tgt', tmp_path,
                                spherex_model='sersic')
    assert failures == {'spherex': 'no usable shape'}
    assert calls == ['catalogs', 'measure', 'sed']


def test_clean_run_returns_no_failures(tmp_path, monkeypatch):
    calls = []
    for name in ('run_catalogs', 'run_measure', 'run_sed'):
        monkeypatch.setattr(pipeline, name, _quiet(calls, name))
    assert pipeline.run_all(COORD, 'tgt', tmp_path) == {}
    assert not (tmp_path / 'Photometry' / 'coverage_measure.json').exists()


def test_measure_own_coverage_report_is_never_overwritten(tmp_path,
                                                          monkeypatch):
    phot = tmp_path / 'Photometry'
    phot.mkdir()
    own_report = {'legacy': {'status': 'ok'}}

    def writes_then_dies(*args, **kwargs):
        (phot / 'coverage_measure.json').write_text(json.dumps(own_report))
        raise RuntimeError('died after reporting')

    monkeypatch.setattr(pipeline, 'run_catalogs', lambda *a, **k: None)
    monkeypatch.setattr(pipeline, 'run_measure', writes_then_dies)
    monkeypatch.setattr(pipeline, 'run_sed', lambda *a, **k: None)
    failures = pipeline.run_all(COORD, 'tgt', tmp_path)
    assert 'measure' in failures
    report = json.loads((phot / 'coverage_measure.json').read_text())
    assert report == own_report
