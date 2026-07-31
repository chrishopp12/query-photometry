"""Pipeline drivers: run_all stage isolation (one dead stage cannot cost the
others) and run_sed's extinction-scale guard."""
import json

import pytest
from astropy.coordinates import SkyCoord

from sedphot import pipeline, qa
from sedphot.results import STATUS_ERROR, ProviderResult
from sedphot.schema import make_row, rows_to_frame

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


# ------------------------------------
# run_sed: the two tables on one axis
# ------------------------------------
def _sed_tables(tmp_path, monkeypatch, *, catalog_dered, measured_dered):
    """Write a catalog + measured pair at the given dered states; no figure."""
    phot = tmp_path / 'Photometry'
    phot.mkdir(parents=True)
    for kind, dered in (('catalog', catalog_dered),
                        ('measured', measured_dered)):
        rows_to_frame([make_row(
            band='SDSS_r', flux_ujy=10.0, flux_err_ujy=1.0, mag=20.0,
            mag_err=0.1, target_ra=150.0, target_dec=2.0, match_ra=150.0,
            match_dec=2.0, sep_arcsec=0.1, flags='', source=kind,
            dered_applied=dered)]).to_csv(phot / f"t_{kind}.csv", index=False)
    monkeypatch.setattr(qa, 'plot_sed',
                        lambda frames, outpath, title='': outpath)
    return phot


def test_run_sed_warns_when_the_tables_disagree_on_dered(tmp_path, monkeypatch,
                                                         capsys):
    """Only the catalog table is ever dereddened, so a --dered run draws
    corrected and uncorrected fluxes on one axis."""
    _sed_tables(tmp_path, monkeypatch, catalog_dered=True,
                measured_dered=False)
    pipeline.run_sed('t', tmp_path)
    out = capsys.readouterr().out
    assert 'WARNING' in out
    assert 't_catalog.csv dered_applied=True' in out
    assert 't_measured.csv dered_applied=False' in out
    assert 'common extinction scale' in out


def test_run_sed_is_quiet_when_both_tables_are_as_measured(tmp_path,
                                                           monkeypatch,
                                                           capsys):
    _sed_tables(tmp_path, monkeypatch, catalog_dered=False,
                measured_dered=False)
    pipeline.run_sed('t', tmp_path)
    assert 'extinction' not in capsys.readouterr().out


# ------------------------------------
# Stage and provider selection
# ------------------------------------

def test_selection_accepts_any_combination() -> None:
    from sedphot.catalogs import CATALOG_PROVIDERS
    from sedphot.images import IMAGE_PROVIDERS
    from sedphot.pipeline import select_providers

    # Unstated means every provider: an unstated selection must not
    # silently drop a stage.
    assert select_providers(CATALOG_PROVIDERS, None) == list(CATALOG_PROVIDERS)
    assert select_providers(CATALOG_PROVIDERS, ['all']) == \
        list(CATALOG_PROVIDERS)
    assert select_providers(CATALOG_PROVIDERS, ['none']) == []
    assert select_providers(IMAGE_PROVIDERS, ['legacy']) == ['legacy']

    # Registry order, not the order they were typed, so a run's provider
    # sequence never depends on how the command line was written.
    typed = select_providers(IMAGE_PROVIDERS, ['sdss', 'legacy', 'cfht'])
    assert typed == [n for n in IMAGE_PROVIDERS if n in typed]

    # skip subtracts from either form.
    assert 'hst' not in select_providers(CATALOG_PROVIDERS, None, skip=['hst'])
    assert select_providers(CATALOG_PROVIDERS, ['legacy'],
                            skip=['legacy']) == []

    with pytest.raises(ValueError, match="unknown provider"):
        select_providers(CATALOG_PROVIDERS, ['nope'])


def test_a_stage_with_no_providers_is_skipped_not_run(tmp_path,
                                                      monkeypatch) -> None:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from sedphot import pipeline

    called = []
    monkeypatch.setattr(pipeline, 'run_catalogs',
                        lambda *a, **k: called.append('catalogs'))
    monkeypatch.setattr(pipeline, 'run_measure',
                        lambda *a, **k: called.append('measure'))
    monkeypatch.setattr(pipeline, 'run_sed', lambda *a, **k: None)
    coord = SkyCoord(210.0 * u.deg, 30.0 * u.deg)

    # Catalog-only: the measurement stage must not run at all, rather
    # than run with an empty provider list and fetch nothing.
    failures = pipeline.run_all(coord, 'target_a', tmp_path, images=['none'])
    assert called == ['catalogs']
    assert failures == {}

    called.clear()
    pipeline.run_all(coord, 'target_a', tmp_path, catalogs=['none'])
    assert called == ['measure']

    called.clear()
    pipeline.run_all(coord, 'target_a', tmp_path,
                     catalogs=['none'], images=['none'])
    assert called == []
