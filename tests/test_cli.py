"""CLI exit codes: single-provider verbs must fail loudly."""
import sys

import pytest

from sedphot import cli
from sedphot.results import STATUS_ERROR, STATUS_OK, ProviderResult

# The target spec every writing verb needs. --out-dir is required, so a
# test that omits it exits 2 on the argparse error rather than reaching
# the code under test.
TARGET = ['--ra', '10.0', '--dec', '20.0', '--out-dir', 'out']


def _fake_run_spherex(status):
    def fake(*args, **kwargs):
        return ProviderResult(provider='spherex', status=status, message='')
    return fake


def test_spherex_error_exits_nonzero(monkeypatch):
    monkeypatch.setattr(cli, 'run_spherex', _fake_run_spherex(STATUS_ERROR))
    monkeypatch.setattr(sys, 'argv', ['sedphot', 'spherex'] + TARGET)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_spherex_ok_exits_zero(monkeypatch):
    monkeypatch.setattr(cli, 'run_spherex', _fake_run_spherex(STATUS_OK))
    monkeypatch.setattr(sys, 'argv', ['sedphot', 'spherex'] + TARGET)
    assert cli.main() is None


def test_run_stage_failure_exits_nonzero(monkeypatch):
    monkeypatch.setattr(cli, 'run_all',
                        lambda *a, **k: {'measure': 'RuntimeError: boom'})
    monkeypatch.setattr(sys, 'argv', ['sedphot', 'run'] + TARGET)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 1


def test_run_clean_exits_zero(monkeypatch):
    monkeypatch.setattr(cli, 'run_all', lambda *a, **k: {})
    monkeypatch.setattr(sys, 'argv', ['sedphot', 'run'] + TARGET)
    assert cli.main() is None


# ------------------------------------
# Where products land
# ------------------------------------
@pytest.mark.parametrize('verb', ['catalogs', 'measure', 'spherex', 'run',
                                  'sed', 'overlay'])
def test_writing_verbs_require_an_output_directory(verb, monkeypatch):
    """A verb that writes must be told where, never default to the cwd.

    Products, image caches and the scene cache all land under --out-dir,
    so a defaulted one builds a galaxy tree wherever the shell stands.
    """
    argv = ['sedphot', verb]
    if verb not in ('sed', 'overlay'):
        argv += ['--ra', '10.0', '--dec', '20.0']
    if verb in ('catalogs', 'measure'):
        argv += ['--all']
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_resolve_takes_no_output_directory(monkeypatch):
    """resolve only prints, so it has no output directory to give."""
    monkeypatch.setattr(sys, 'argv',
                        ['sedphot', 'resolve', '--ra', '10.0', '--dec', '20.0',
                         '--out-dir', 'out'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


# ------------------------------------
# The measurement option surface
# ------------------------------------
MEASURE_OPTS = ('mode', 'bands', 'aperture', 'radii', 'cutout_size',
                'sky_rmin', 'registry', 'registry_update', 'sersic_from',
                'sersic_params', 'sersic_seeing', 'legacy_dr',
                'legacy_bricks', 'hst_proposal_id')


@pytest.mark.parametrize('verb', ['measure', 'run'])
def test_measure_options_reach_both_verbs(verb):
    """The run verb must accept every option the verb it drives accepts.

    A reduced set on run_all leaves a measurement option given to `run`
    simply absent -- discoverable only from a wrong answer.
    """
    args = cli.build_parser().parse_args(
        [verb] + TARGET + (['--all'] if verb == 'measure' else []))
    missing = [o for o in MEASURE_OPTS if not hasattr(args, o)]
    assert not missing, f"{verb} is missing {missing}"


def test_run_forwards_every_measure_option(monkeypatch):
    """...and forwards them, rather than accepting and dropping them."""
    seen = {}
    monkeypatch.setattr(cli, 'run_all',
                        lambda *a, **k: (seen.update(k), {})[1])
    monkeypatch.setattr(sys, 'argv', ['sedphot', 'run'] + TARGET + [
        '--mode', 'sersic', '--bands', 'r', 'z', '--sky-rmin', '3.0',
        '--radii', '2', '4', '8', '--sersic-from', 'z',
        '--sersic-seeing', '0.9', '--hst-proposal-id', '12345'])
    cli.main()
    assert seen['mode'] == 'sersic'
    assert seen['bands'] == ['r', 'z']
    assert seen['sky_rmin_arcsec'] == 3.0
    assert seen['rgrid'] == [2.0, 4.0, 8.0]
    assert seen['sersic_from'] == 'z'
    assert seen['sersic_seeing'] == 0.9
    assert seen['hst_proposal_id'] == '12345'


def test_measure_and_remeasure_share_an_aperture_default():
    """A bare remeasure must re-report at the radius measure used.

    Two defaults would re-report a stored fit at a radius the measurement
    never ran, and the flux would look entirely ordinary.
    """
    measure = cli.build_parser().parse_args(['measure'] + TARGET + ['--all'])
    remeasure = cli.build_parser().parse_args(
        ['remeasure', 'x.provenance.json'])
    assert measure.aperture == remeasure.aperture


def test_measure_and_remeasure_share_a_mode_default():
    """Same argument as the aperture default: a bare remeasure must report
    the same quantity a bare measure produced, not a different estimator."""
    measure = cli.build_parser().parse_args(['measure'] + TARGET + ['--all'])
    remeasure = cli.build_parser().parse_args(
        ['remeasure', 'x.provenance.json'])
    assert measure.mode == remeasure.mode == 'aperture'


@pytest.mark.parametrize('flag', [['--sersic-from', 'z'],
                                  ['--sersic-seeing', '0.9'],
                                  ['--sersic-params', '2', '1.5', '30', '3']])
def test_run_spherex_sersic_accepts_every_shape_flag(flag, monkeypatch):
    """run_all forwards all three shape flags to run_spherex, so refusing
    any of them under --mode aperture refuses a flag that IS honored."""
    monkeypatch.setattr(cli, 'run_all', lambda *a, **k: {})
    monkeypatch.setattr(sys, 'argv', ['sedphot', 'run'] + TARGET + [
        '--mode', 'aperture', '--spherex', 'sersic'] + flag)
    assert cli.main() is None


@pytest.mark.parametrize('params,message', [
    (['7', '1.5', '30', '3'], 'outside the fitted range'),
    (['0.2', '1.5', '30', '3'], 'outside the fitted range'),
    (['2', '0.5', '30', '3'], 'a/b >= 1'),
    (['2', '1.5', '30', '0'], 'reff_arcsec must be positive'),
])
def test_explicit_sersic_params_are_bounded(params, message):
    """The help promised n<=6 and nothing enforced it."""
    from sedphot.pipeline import _resolve_shape
    with pytest.raises(ValueError, match=message):
        _resolve_shape(None, None, sersic_from=None,
                       sersic_params=[float(v) for v in params],
                       cutout_half_arcsec=60.0, sersic_seeing=None)


def test_an_in_range_sersic_shape_is_accepted():
    from sedphot.pipeline import _resolve_shape
    shape, origin = _resolve_shape(None, None, sersic_from=None,
                                   sersic_params=[2.0, 1.5, 30.0, 3.0],
                                   cutout_half_arcsec=60.0, sersic_seeing=None)
    assert shape['n'] == 2.0
    assert origin['source'] == 'explicit parameters'


@pytest.mark.parametrize('verb', ['measure', 'run'])
@pytest.mark.parametrize('flag', [['--sersic-from', 'z'],
                                  ['--sersic-seeing', '0.9'],
                                  ['--sersic-params', '2', '1.5', '30', '3']])
def test_shape_flags_refused_under_aperture_mode(verb, flag, monkeypatch,
                                                 capsys):
    """A shape flag with --mode aperture must be refused, never dropped.

    A contradictory flag is a usage error, so it exits 2 like argparse's
    own and names itself on stderr.
    """
    monkeypatch.setattr(sys, 'argv',
                        ['sedphot', verb] + TARGET
                        + (['--all'] if verb == 'measure' else [])
                        + ['--mode', 'aperture'] + flag)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert 'only applies to --mode sersic' in capsys.readouterr().err


def test_spherex_sersic_keeps_its_shape_flag_under_aperture_mode(monkeypatch):
    """--sersic-params also declares the SPHEREx extraction shape, so under
    `run --spherex sersic` it is meaningful with an aperture measurement."""
    monkeypatch.setattr(cli, 'run_all', lambda *a, **k: {})
    monkeypatch.setattr(sys, 'argv', ['sedphot', 'run'] + TARGET + [
        '--mode', 'aperture', '--spherex', 'sersic',
        '--sersic-params', '2', '1.5', '30', '3'])
    assert cli.main() is None


# ------------------------------------
# SPHEREx extraction config reaches run_all
# ------------------------------------
def _capture_run_all(seen):
    def fake(*args, **kwargs):
        seen.update(kwargs)
        return {}
    return fake


def test_run_forwards_the_spherex_extraction_config(monkeypatch):
    """run_all owns the SPHEREx stage, so every knob that defines the
    extraction has to reach it -- an MJD window most of all, since epochs
    with broken metadata kill jobs server-side."""
    seen = {}
    monkeypatch.setattr(cli, 'run_all', _capture_run_all(seen))
    monkeypatch.setattr(sys, 'argv', ['sedphot', 'run'] + TARGET + [
        '--spherex', 'sersic',
        '--spherex-mjd-range', '60676.0001273', '61174.5063773',
        '--spherex-bkg-size', '27',
        '--spherex-timeout', '7200'])
    assert cli.main() is None
    assert seen['spherex_model'] == 'sersic'
    assert seen['spherex_mjd_range'] == [60676.0001273, 61174.5063773]
    assert seen['spherex_bkg_size'] == 27.0
    assert seen['spherex_timeout'] == 7200.0


def test_run_spherex_config_defaults_are_explicit(monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, 'run_all', _capture_run_all(seen))
    monkeypatch.setattr(sys, 'argv', ['sedphot', 'run'] + TARGET)
    assert cli.main() is None
    assert seen['spherex_mjd_range'] is None
    assert seen['spherex_bkg_size'] == 15.0
    assert seen['spherex_timeout'] == 10800.0


def test_batch_refuses_more_workers_than_irsa_admits(monkeypatch, tmp_path):
    """IRSA runs two spectrophotometry jobs at a time. Whether a third
    queues or is refused is unverified, and a sweep is the wrong place to
    find out: each job is a 20-60 minute server-side extraction."""
    plan = tmp_path / 'plan.json'
    plan.write_text('{"harvest": [], "parallel": []}')
    monkeypatch.setattr(sys, 'argv', [
        'sedphot', 'batch', '--plan', str(plan),
        '--registry-dir', str(tmp_path / 'reg'),
        '--report', str(tmp_path / 'report.json'),
        '--spherex', 'sersic', '--workers', '4'])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert 'concurrent IRSA extractions' in str(exc.value.code)


def test_batch_worker_cap_only_applies_when_spherex_is_on(monkeypatch,
                                                          tmp_path):
    """The cap is about IRSA, not about the measurement stage: with the
    extraction off, worker count is the archive's business alone."""
    import sedphot.batch as batch_mod
    monkeypatch.setattr(batch_mod, 'run_sweep',
                        lambda *a, **k: {'merge_problems': [],
                                         'violations': [],
                                         'aborted': False, 'n_failed': 0})
    plan = tmp_path / 'plan.json'
    plan.write_text('{"harvest": [], "parallel": []}')
    monkeypatch.setattr(sys, 'argv', [
        'sedphot', 'batch', '--plan', str(plan),
        '--registry-dir', str(tmp_path / 'reg'),
        '--report', str(tmp_path / 'report.json'),
        '--spherex', 'off', '--workers', '8'])
    cli.main()          # no SystemExit: 8 workers is fine without SPHEREx
