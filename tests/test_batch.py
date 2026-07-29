"""Sweep execution: passes, per-group registries, the merge and its audit."""
from __future__ import annotations

import csv
import json

import pytest

from sedphot import batch as batch_module
from sedphot.batch import (
    MERGED_REGISTRY_NAME,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_SKIPPED,
    audit_merged,
    is_complete,
    measured_products,
    merge_registries,
    run_sweep,
)
from sedphot.schedule import PASS_HARVEST, PASS_PARALLEL

RA0, DEC0 = 210.0, 30.0
CUTOUT = 120.0
SCENE = 100.0


def _east(arcsec):
    import numpy as np
    return RA0 + arcsec / 3600.0 / np.cos(np.radians(DEC0))


def _target(name, *, sep=0.0, which=PASS_HARVEST, group=0, order=0,
            tmp_path=None):
    return {'name': name, 'label': name,
            'dir': str((tmp_path / name) if tmp_path else f"/tmp/{name}"),
            'ra_deg': _east(sep), 'dec_deg': DEC0,
            'pass': which, 'group': group if which == PASS_HARVEST else None,
            'order': order if which == PASS_HARVEST else None,
            'gates': True, 'matched': True, 'n_consumers': 1, 'n_gated': 0,
            'priority': None, 'entries': []}


def _plan(targets, **extra):
    plan = {'cutout_arcsec': CUTOUT, 'scene_radius_arcsec': SCENE,
            'gate_reach_arcsec': 45.0, 'link_radius_arcsec': 265.0,
            'link_margin_arcsec': 120.0, 'n_violations': 0, 'violations': [],
            'targets': targets}
    plan.update(extra)
    return plan


def _entry(sep, band_flux=1.0):
    return {'ra': _east(sep), 'dec': DEC0,
            'components': {'r': [{'vantage': 'target', 'flux': band_flux}]}}


def _run_options(**extra):
    options = dict(skip=None, radius_arcsec=2.0, dered=False, mode='aperture',
                   bands=None, aperture_arcsec=12.0, cutout_arcsec=CUTOUT,
                   sky_rmin_arcsec=None, rgrid=None, sersic_from=None,
                   sersic_seeing=None, spherex_model='off',
                   sersic_params=None, legacy_dr='dr9', legacy_bricks=False,
                   hst_proposal_id=None)
    options.update(extra)
    return options


# ------------------------------------
# Resume
# ------------------------------------

def test_completion_needs_a_table_and_a_readable_sidecar(tmp_path) -> None:
    target = _target('a', tmp_path=tmp_path)
    table, sidecar = measured_products(target)
    assert not is_complete(target)

    table.parent.mkdir(parents=True)
    table.write_text("band,flux_uJy\n", encoding='utf-8')
    assert not is_complete(target)          # no sidecar

    sidecar.write_text("{ not json", encoding='utf-8')
    assert not is_complete(target)          # sidecar unreadable

    sidecar.write_text(json.dumps({'label': 'a'}), encoding='utf-8')
    assert is_complete(target)


# ------------------------------------
# The merge
# ------------------------------------

def test_merge_unions_disjoint_group_registries(tmp_path) -> None:
    (tmp_path / "group_0.json").write_text(
        json.dumps({'J1': _entry(0.0), 'J2': _entry(10.0)}), encoding='utf-8')
    (tmp_path / "group_1.json").write_text(
        json.dumps({'J3': _entry(9000.0)}), encoding='utf-8')

    merged, problems = merge_registries(sorted(tmp_path.glob("group_*.json")))
    assert sorted(merged) == ['J1', 'J2', 'J3']
    assert problems == []


def test_merge_reports_a_key_two_groups_claim(tmp_path) -> None:
    (tmp_path / "group_0.json").write_text(
        json.dumps({'J1': _entry(0.0)}), encoding='utf-8')
    (tmp_path / "group_1.json").write_text(
        json.dumps({'J1': _entry(0.0)}), encoding='utf-8')

    merged, problems = merge_registries(sorted(tmp_path.glob("group_*.json")))
    assert len(problems) == 1
    assert 'J1' in problems[0]
    # The first writer stands; nothing is invented to resolve the clash.
    assert list(merged) == ['J1']


def test_merge_is_order_independent(tmp_path) -> None:
    (tmp_path / "group_0.json").write_text(
        json.dumps({'J1': _entry(0.0)}), encoding='utf-8')
    (tmp_path / "group_1.json").write_text(
        json.dumps({'J2': _entry(9000.0)}), encoding='utf-8')
    paths = sorted(tmp_path.glob("group_*.json"))
    assert merge_registries(paths)[0] == merge_registries(paths[::-1])[0]


# ------------------------------------
# The post-harvest audit
# ------------------------------------

def test_audit_passes_records_that_stay_in_their_group() -> None:
    rows = [_target('a', sep=0.0, group=0),
            _target('b', sep=9000.0, group=1)]
    registries = {0: {'J1': _entry(20.0)}, 1: {'J2': _entry(9020.0)}}
    assert audit_merged(registries, rows, scene_arcsec=SCENE) == []


def test_audit_catches_a_record_the_other_group_would_read() -> None:
    rows = [_target('a', sep=0.0, group=0),
            _target('b', sep=80.0, group=1)]
    # Written by group 0, but 20" from group 1's field.
    registries = {0: {'J1': _entry(60.0)}, 1: {}}
    violations = audit_merged(registries, rows, scene_arcsec=SCENE)
    assert len(violations) == 1
    assert violations[0]['read_by'] == 'b'
    assert violations[0]['written_group'] == 0


def test_audit_skips_entries_with_no_position() -> None:
    rows = [_target('a', sep=0.0, group=0), _target('b', sep=80.0, group=1)]
    registries = {0: {'J1': {'components': {}}}, 1: {}}
    assert audit_merged(registries, rows, scene_arcsec=SCENE) == []


# ------------------------------------
# The sweep
# ------------------------------------

@pytest.fixture
def fake_run_all(monkeypatch):
    """Record every call and write the products a real run would."""
    calls = []

    def _fake(coord, label, out_dir, **kwargs):
        calls.append({'label': label, 'out_dir': str(out_dir),
                      'registry_path': kwargs.get('registry_path'),
                      'registry_update': kwargs.get('registry_update')})
        phot = batch_module.Path(out_dir) / "Photometry"
        phot.mkdir(parents=True, exist_ok=True)
        (phot / f"{label}_measured.csv").write_text("band\n", encoding='utf-8')
        (phot / f"{label}_measured.provenance.json").write_text(
            json.dumps({'label': label}), encoding='utf-8')
        registry_path = kwargs.get('registry_path')
        if kwargs.get('registry_update') and registry_path:
            path = batch_module.Path(registry_path)
            existing = json.loads(path.read_text()) if path.exists() else {}
            existing[f"J_{label}"] = _entry(0.0)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(existing), encoding='utf-8')
        return {}

    monkeypatch.setattr("sedphot.pipeline.run_all", _fake)
    return calls


def _sweep(tmp_path, targets, **extra):
    kwargs = dict(registry_dir=tmp_path / "registry",
                  run_options=_run_options(),
                  report_path=tmp_path / "report.csv",
                  workers=1, progress=lambda line: None)
    kwargs.update(extra)
    return run_sweep(_plan(targets), **kwargs)


def _report(path):
    with open(path, newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def test_each_pass_gets_the_registry_its_role_requires(tmp_path,
                                                       fake_run_all) -> None:
    targets = [_target('h0', sep=0.0, group=0, order=0, tmp_path=tmp_path),
               _target('h1', sep=50.0, group=0, order=1, tmp_path=tmp_path),
               _target('p0', sep=9000.0, which=PASS_PARALLEL,
                       tmp_path=tmp_path)]
    summary = _sweep(tmp_path, targets)

    assert summary['n_ok'] == 3
    assert summary['merge_problems'] == []
    assert summary['violations'] == []

    by_label = {c['label']: c for c in fake_run_all}
    # The harvest pass writes, into its own group's registry.
    for label in ('h0', 'h1'):
        assert by_label[label]['registry_update'] is True
        assert by_label[label]['registry_path'].endswith("group_0.json")
    # The parallel pass reads the frozen union and writes nothing.
    assert by_label['p0']['registry_update'] is False
    assert by_label['p0']['registry_path'].endswith(MERGED_REGISTRY_NAME)

    merged = json.loads(
        (tmp_path / "registry" / MERGED_REGISTRY_NAME).read_text())
    assert sorted(merged) == ['J_h0', 'J_h1']


def test_harvest_groups_get_separate_registries(tmp_path,
                                                fake_run_all) -> None:
    targets = [_target('a', sep=0.0, group=0, tmp_path=tmp_path),
               _target('b', sep=9000.0, group=1, tmp_path=tmp_path)]
    _sweep(tmp_path, targets)

    registry_dir = tmp_path / "registry"
    assert sorted(p.name for p in registry_dir.glob("group_*.json")) == \
        ['group_0.json', 'group_1.json']
    paths = {c['label']: c['registry_path'] for c in fake_run_all
             if c['registry_update']}
    assert paths['a'] != paths['b']


def test_harvest_runs_its_group_in_plan_order(tmp_path,
                                              fake_run_all) -> None:
    targets = [_target('second', sep=50.0, group=0, order=1,
                       tmp_path=tmp_path),
               _target('first', sep=0.0, group=0, order=0, tmp_path=tmp_path)]
    _sweep(tmp_path, targets)
    assert [c['label'] for c in fake_run_all] == ['first', 'second']


def test_grouping_off_runs_one_sequence_against_one_registry(
        tmp_path, fake_run_all) -> None:
    targets = [_target('a', sep=0.0, group=0, order=0, tmp_path=tmp_path),
               _target('b', sep=9000.0, group=1, order=0, tmp_path=tmp_path)]
    _sweep(tmp_path, targets, groups=False)

    registry_dir = tmp_path / "registry"
    assert list(registry_dir.glob("group_*.json")) == []
    assert all(c['registry_path'].endswith(MERGED_REGISTRY_NAME)
               for c in fake_run_all)
    merged = json.loads((registry_dir / MERGED_REGISTRY_NAME).read_text())
    assert sorted(merged) == ['J_a', 'J_b']


def test_resume_skips_a_measured_target(tmp_path, fake_run_all) -> None:
    targets = [_target('a', sep=0.0, group=0, tmp_path=tmp_path)]
    first = _sweep(tmp_path, targets)
    assert first['n_ok'] == 1

    second = _sweep(tmp_path, targets)
    assert second['n_ok'] == 0
    assert second['n_skipped'] == 1
    assert len(fake_run_all) == 1

    third = _sweep(tmp_path, targets, resume=False)
    assert third['n_ok'] == 1
    assert len(fake_run_all) == 2


def test_a_stage_failure_is_recorded_and_the_sweep_continues(
        tmp_path, monkeypatch) -> None:
    def _fake(coord, label, out_dir, **kwargs):
        if label == 'bad':
            return {'measure': 'RuntimeError: no coverage'}
        return {}

    monkeypatch.setattr("sedphot.pipeline.run_all", _fake)
    targets = [_target('bad', sep=0.0, which=PASS_PARALLEL,
                       tmp_path=tmp_path),
               _target('good', sep=9000.0, which=PASS_PARALLEL,
                       tmp_path=tmp_path)]
    summary = _sweep(tmp_path, targets)

    assert summary['n_ok'] == 1
    assert summary['n_failed'] == 1
    rows = {r['name']: r for r in _report(summary['report'])}
    assert rows['bad']['status'] == STATUS_FAILED
    assert 'no coverage' in rows['bad']['failures']
    assert rows['good']['status'] == STATUS_OK


def test_a_raising_target_does_not_abort_the_sweep(tmp_path,
                                                   monkeypatch) -> None:
    def _fake(coord, label, out_dir, **kwargs):
        if label == 'boom':
            raise RuntimeError("archive down")
        return {}

    monkeypatch.setattr("sedphot.pipeline.run_all", _fake)
    targets = [_target('boom', sep=0.0, which=PASS_PARALLEL,
                       tmp_path=tmp_path),
               _target('fine', sep=9000.0, which=PASS_PARALLEL,
                       tmp_path=tmp_path)]
    summary = _sweep(tmp_path, targets)

    assert summary['n_failed'] == 1
    rows = {r['name']: r for r in _report(summary['report'])}
    assert 'archive down' in rows['boom']['error']
    assert rows['boom']['stage'] == 'run'


def test_a_mismatched_stamp_width_is_refused(tmp_path) -> None:
    targets = [_target('a', tmp_path=tmp_path)]
    with pytest.raises(ValueError, match="differs from the plan"):
        _sweep(tmp_path, targets,
               run_options=_run_options(cutout_arcsec=200.0))


def test_a_plan_with_crossings_is_refused(tmp_path) -> None:
    plan = _plan([_target('a', tmp_path=tmp_path)], n_violations=1)
    with pytest.raises(ValueError, match="boundary crossing"):
        run_sweep(plan, registry_dir=tmp_path / "registry",
                  run_options=_run_options(),
                  report_path=tmp_path / "report.csv",
                  workers=1, progress=lambda line: None)


def test_a_merge_clash_stops_the_parallel_pass(tmp_path,
                                               monkeypatch) -> None:
    # Both groups harvest the SAME key, which the grouping says cannot
    # happen: the sweep must refuse rather than freeze a merged registry.
    def _fake(coord, label, out_dir, **kwargs):
        path = batch_module.Path(kwargs['registry_path'])
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(path.read_text()) if path.exists() else {}
        existing['J_shared'] = _entry(0.0)
        path.write_text(json.dumps(existing), encoding='utf-8')
        return {}

    monkeypatch.setattr("sedphot.pipeline.run_all", _fake)
    targets = [_target('a', sep=0.0, group=0, tmp_path=tmp_path),
               _target('b', sep=9000.0, group=1, tmp_path=tmp_path),
               _target('p', sep=18000.0, which=PASS_PARALLEL,
                       tmp_path=tmp_path)]
    summary = _sweep(tmp_path, targets)

    assert summary['aborted']
    assert len(summary['merge_problems']) == 1
    assert not (tmp_path / "registry" / MERGED_REGISTRY_NAME).exists()
    # The parallel pass never ran against an unmerged registry.
    assert all(r['name'] != 'p' for r in _report(summary['report']))


def test_a_fully_resumed_harvest_keeps_the_registry_it_found(
        tmp_path, fake_run_all) -> None:
    targets = [_target('a', sep=0.0, group=0, tmp_path=tmp_path),
               _target('p', sep=9000.0, which=PASS_PARALLEL,
                       tmp_path=tmp_path)]
    first = _sweep(tmp_path, targets)
    assert first['n_ok'] == 2
    merged_path = tmp_path / "registry" / MERGED_REGISTRY_NAME
    before = json.loads(merged_path.read_text())
    assert before

    # Nothing to harvest the second time, so no group registry is
    # written; merging that empty set must not blank the frozen one.
    for path in (tmp_path / "registry").glob("group_*.json"):
        path.unlink()
    second = _sweep(tmp_path, targets)
    assert second['n_skipped'] == 2
    assert json.loads(merged_path.read_text()) == before


def test_pass_selection_runs_only_what_was_asked(tmp_path,
                                                 fake_run_all) -> None:
    targets = [_target('h', sep=0.0, group=0, tmp_path=tmp_path),
               _target('p', sep=9000.0, which=PASS_PARALLEL,
                       tmp_path=tmp_path)]
    _sweep(tmp_path, targets, which_pass='harvest')
    assert [c['label'] for c in fake_run_all] == ['h']

    fake_run_all.clear()
    _sweep(tmp_path, targets, which_pass='parallel')
    assert [c['label'] for c in fake_run_all] == ['p']


def test_report_is_written_in_pass_then_plan_order(tmp_path,
                                                   fake_run_all) -> None:
    targets = [_target('p', sep=9000.0, which=PASS_PARALLEL,
                       tmp_path=tmp_path),
               _target('h1', sep=50.0, group=0, order=1, tmp_path=tmp_path),
               _target('h0', sep=0.0, group=0, order=0, tmp_path=tmp_path)]
    summary = _sweep(tmp_path, targets)
    rows = _report(summary['report'])
    assert [r['name'] for r in rows] == ['h0', 'h1', 'p']
