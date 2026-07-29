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
