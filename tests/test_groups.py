"""Group configs: validation, planning, and the sweep."""
import json

import astropy.units as u
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord

import sedphot.groups as groups_mod
from sedphot.groups import (DOCUMENTED_CONCURRENCY, link_groups, load_config,
                            plan_groups, run_config, sources_for,
                            write_config)
from sedphot.results import ProviderResult, STATUS_ERROR, STATUS_OK
from sedphot.spherex import GROUP_SOURCES_MAX, QUEUE_TIMEOUT

SHAPE = {"n": 4.06, "axis_ratio": 1.10, "pa_deg": 5.08, "reff_arcsec": 5.19}
MJD = [60676.0001273, 61174.5063773]


def _member(label, ra, dec, role="science", shape=SHAPE, **extra):
    member = {"label": label, "ra_deg": ra, "dec_deg": dec, "role": role}
    if shape is not None:
        member["shape"] = dict(shape)
    if role == "science":
        member["dir"] = extra.pop("dir", f"Galaxies/{label}")
    member.update(extra)
    return member


def _config(groups=None, **top):
    config = {
        "schema_version": 1,
        "data_root": ".",
        "defaults": {"model": "sersic", "bkg_size": 15, "mjd_range": MJD},
        "groups": groups if groups is not None else [
            {"id": "g001", "members": [_member("a", 217.0, 57.0),
                                       _member("b", 217.005, 57.003)]}],
    }
    config.update(top)
    return config


def _write(tmp_path, config, name="groups.json"):
    path = tmp_path / name
    path.write_text(json.dumps(config))
    return path


# ------------------------------------
# Linking
# ------------------------------------
def test_linking_is_transitive():
    # A-B and B-C are within reach; A-C are not. Blending is transitive
    # at the PSF scale, so all three belong in one job.
    coords = SkyCoord([217.0, 217.008, 217.016] * u.deg,
                      [57.0, 57.0, 57.0] * u.deg)
    assert link_groups(coords, 40.0) == [[0, 1, 2]]
    assert link_groups(coords, 10.0) == [[0], [1], [2]]


def test_linking_handles_an_empty_list():
    assert link_groups(SkyCoord([] * u.deg, [] * u.deg), 45.0) == []


# ------------------------------------
# Config validation
# ------------------------------------
def test_a_valid_config_loads(tmp_path):
    config = load_config(_write(tmp_path, _config()))
    assert len(config["groups"]) == 1
    assert config["defaults"]["timeout"] == 10800.0     # filled in
    assert config["data_root"] == str(tmp_path.resolve())


def test_member_dirs_resolve_against_data_root(tmp_path):
    config = load_config(_write(tmp_path, _config()))
    sources = sources_for(config["groups"][0], config)
    assert sources[0].out_dir == str(tmp_path.resolve() / "Galaxies/a")


def test_an_absolute_member_dir_wins(tmp_path):
    absolute = str(tmp_path / "elsewhere")
    config = _config([{"id": "g001", "members": [
        _member("a", 217.0, 57.0, dir=absolute)]}])
    loaded = load_config(_write(tmp_path, config))
    assert sources_for(loaded["groups"][0], loaded)[0].out_dir == absolute


@pytest.mark.parametrize("mutate,message", [
    (lambda c: c.pop("groups"), "missing required"),
    (lambda c: c.update(nonsense=1), "unknown key"),
    (lambda c: c.update(schema_version=99), "schema_version"),
    (lambda c: c.update(groups=[]), "non-empty list"),
    (lambda c: c["defaults"].update(model="gaussian"), "not one of"),
    (lambda c: c["defaults"].update(bkgsize=15), "unknown key"),
])
def test_a_malformed_config_is_refused(tmp_path, mutate, message):
    config = _config()
    mutate(config)
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, config))


def test_duplicate_group_ids_are_refused(tmp_path):
    config = _config([{"id": "g001", "members": [_member("a", 217.0, 57.0)]},
                      {"id": "g001", "members": [_member("b", 218.0, 57.0)]}])
    with pytest.raises(ValueError, match="duplicate group id"):
        load_config(_write(tmp_path, config))


@pytest.mark.parametrize("members,message", [
    ([_member("a", 217.0, 57.0, role="neighbour")], "role"),
    ([_member("a", 217.0, 57.0, role="ancillary", shape=SHAPE)],
     "no science member"),
    ([_member("a", 217.0, 57.0), _member("a", 217.01, 57.0)],
     "duplicate member label"),
    ([{"label": "a", "ra_deg": 217.0, "dec_deg": 57.0, "shape": SHAPE}],
     "science member needs a dir"),
    ([_member("a", 217.0, 57.0, shape=None)], "every member needs a shape"),
    ([_member("a", 217.0, 57.0, shape={**SHAPE, "reff_arcsec": 25.0})],
     "ceiling"),
    ([_member("a", 217.0, 57.0, shape={**SHAPE, "n": 9.0})], "tested range"),
    ([_member("a", 217.0, 57.0, typo=1)], "unknown key"),
])
def test_a_malformed_member_is_refused(tmp_path, members, message):
    with pytest.raises(ValueError, match=message):
        load_config(_write(tmp_path, _config([{"id": "g001",
                                               "members": members}])))


def test_a_group_over_the_seat_cap_is_refused(tmp_path):
    members = [_member(f"s{i}", 217.0 + i * 0.01, 57.0)
               for i in range(GROUP_SOURCES_MAX + 1)]
    with pytest.raises(ValueError, match="exceeds"):
        load_config(_write(tmp_path, _config([{"id": "g001",
                                               "members": members}])))


def test_a_shape_under_a_psf_model_is_refused(tmp_path):
    # Silently ignoring it would run a point-source job while the config
    # on disk claims a shape was used.
    config = _config()
    config["defaults"]["model"] = "psf"
    with pytest.raises(ValueError, match="silently ignored"):
        load_config(_write(tmp_path, config))


def test_a_psf_config_needs_no_shapes(tmp_path):
    config = _config([{"id": "g001", "members": [
        _member("a", 217.0, 57.0, shape=None),
        _member("b", 217.005, 57.003, shape=None)]}])
    config["defaults"]["model"] = "psf"
    loaded = load_config(_write(tmp_path, config))
    assert all(s.model is None
               for s in sources_for(loaded["groups"][0], loaded))


# ------------------------------------
# Planning
# ------------------------------------
def _scene(rows):
    return pd.DataFrame(rows, columns=["ra", "dec", "type", "sersic",
                                       "shape_r", "shape_e1", "shape_e2",
                                       "flux_r"])


def _fake_catalog(monkeypatch, frame):
    import sedphot.schedule as schedule_mod
    monkeypatch.setattr(schedule_mod, 'fetch_catalog',
                        lambda coord, d, **k: frame)


def test_planning_groups_blended_targets_and_seats_companions(monkeypatch):
    targets = [
        {"name": "a", "label": "a", "ra_deg": 217.0, "dec_deg": 57.0,
         "dir": "Galaxies/a"},
        {"name": "b", "label": "b", "ra_deg": 217.005, "dec_deg": 57.0,
         "dir": "Galaxies/b"},
        {"name": "far", "label": "far", "ra_deg": 218.0, "dec_deg": 57.0,
         "dir": "Galaxies/far"},
    ]
    scene = _scene([
        [217.0, 57.0, "SER", 4.0, 3.0, 0.1, 0.0, 100.0],       # a itself
        [217.005, 57.0, "SER", 2.0, 2.0, 0.0, 0.1, 80.0],      # b itself
        [217.002, 57.002, "DEV", 4.0, 1.5, 0.0, 0.0, 40.0],    # companion
        [217.003, 57.001, "PSF", 0.0, 0.0, 0.0, 0.0, 30.0],    # point-like
        [217.001, 57.001, "REX", 1.0, 0.5, 0.0, 0.0, 0.5],     # too faint
        [218.0, 57.0, "SER", 3.0, 2.0, 0.0, 0.0, 90.0],        # far itself
    ])
    _fake_catalog(monkeypatch, scene)
    config = plan_groups(targets, blend_radius_as=45.0,
                         companion_flux_ratio=0.1, progress=lambda m: None)

    assert len(config["groups"]) == 2                  # a+b together, far apart
    first = config["groups"][0]
    science = [m for m in first["members"] if m["role"] == "science"]
    ancillary = [m for m in first["members"] if m["role"] == "ancillary"]
    assert sorted(m["label"] for m in science) == ["a", "b"]
    assert len(ancillary) == 2                         # the faint one is out
    # A point-like neighbor rides a sersic job at a sub-threshold radius.
    point_like = [m for m in ancillary if m["shape"]["reff_arcsec"] < 1.0]
    assert len(point_like) == 1
    assert "point-source threshold" in point_like[0]["shape_origin"]


def test_a_planned_config_loads_back(tmp_path, monkeypatch):
    # The planner's own output is the config most likely to be run, so
    # its round trip is the one that must not break.
    targets = [{"name": "a", "label": "a", "ra_deg": 217.0, "dec_deg": 57.0,
                "dir": "Galaxies/a"}]
    _fake_catalog(monkeypatch, _scene(
        [[217.0, 57.0, "SER", 4.0, 3.0, 0.1, 0.0, 100.0]]))
    config = plan_groups(targets, progress=lambda m: None)
    path, csv_path = write_config(config, tmp_path / "groups.json")
    loaded = load_config(path)
    assert len(loaded["groups"]) == 1
    assert csv_path.is_file()
    assert "label" in csv_path.read_text().splitlines()[0]


def test_planning_reports_a_target_it_cannot_shape(monkeypatch):
    targets = [{"name": "a", "label": "a", "ra_deg": 217.0, "dec_deg": 57.0,
                "dir": "Galaxies/a"}]
    # A PSF source at the target position: no extended shape to freeze.
    _fake_catalog(monkeypatch, _scene(
        [[217.0, 57.0, "PSF", 0.0, 0.0, 0.0, 0.0, 100.0]]))
    config = plan_groups(targets, progress=lambda m: None)
    assert "shape" not in config["groups"][0]["members"][0]
    assert any("no usable Tractor shape" in n
               for n in config["planning_notes"])


def test_the_seat_cap_is_reported_not_silent(monkeypatch):
    targets = [{"name": "a", "label": "a", "ra_deg": 217.0, "dec_deg": 57.0,
                "dir": "Galaxies/a"}]
    rows = [[217.0, 57.0, "SER", 4.0, 3.0, 0.1, 0.0, 100.0]]
    rows += [[217.0 + 0.001 * (i + 1), 57.0, "SER", 2.0, 2.0, 0.0, 0.0, 50.0]
             for i in range(GROUP_SOURCES_MAX + 5)]
    _fake_catalog(monkeypatch, _scene(rows))
    config = plan_groups(targets, progress=lambda m: None)
    assert len(config["groups"][0]["members"]) == GROUP_SOURCES_MAX
    assert any("did not fit" in n for n in config["planning_notes"])


# ------------------------------------
# The sweep
# ------------------------------------
def _fake_run(monkeypatch, status=STATUS_OK, message="done"):
    calls = []

    def fake(sources, *, group_id, **kwargs):
        calls.append(group_id)
        if kwargs.get('on_job_url'):
            kwargs['on_job_url'](f"https://irsa/{group_id}")
        return ProviderResult(provider='spherex', status=status,
                              message=message, meta={'tag': 'joint-sersic-aaa',
                                                     'group_id': group_id})
    monkeypatch.setattr(groups_mod, 'fetch_group', fake)
    return calls


def test_the_sweep_runs_each_group_and_reports(tmp_path, monkeypatch):
    config = _config([
        {"id": "g001", "members": [_member("a", 217.0, 57.0)]},
        {"id": "g002", "members": [_member("b", 218.0, 57.0)]}])
    loaded = load_config(_write(tmp_path, config))
    calls = _fake_run(monkeypatch)
    summary = run_config(loaded, report_path=tmp_path / "report.csv",
                         progress=lambda m: None)
    assert sorted(calls) == ["g001", "g002"]
    assert summary["counts"][STATUS_OK] == 2
    report = (tmp_path / "report.csv").read_text()
    # The job URL is recorded, so a crashed sweep can poll a job that
    # outlived it.
    assert "https://irsa/g001" in report
    assert "a:S" in report


def test_only_runs_the_named_groups(tmp_path, monkeypatch):
    config = _config([
        {"id": "g001", "members": [_member("a", 217.0, 57.0)]},
        {"id": "g002", "members": [_member("b", 218.0, 57.0)]}])
    loaded = load_config(_write(tmp_path, config))
    calls = _fake_run(monkeypatch)
    run_config(loaded, report_path=tmp_path / "report.csv", only=["g002"],
               progress=lambda m: None)
    assert calls == ["g002"]
    with pytest.raises(ValueError, match="no such group"):
        run_config(loaded, report_path=tmp_path / "r2.csv", only=["g009"],
                   progress=lambda m: None)


def test_resume_skips_a_group_already_on_disk(tmp_path, monkeypatch):
    loaded = load_config(_write(tmp_path, _config()))
    monkeypatch.setattr(groups_mod, 'group_is_complete',
                        lambda s, p, t: (True, []))
    calls = _fake_run(monkeypatch)
    summary = run_config(loaded, report_path=tmp_path / "report.csv",
                         progress=lambda m: None)
    assert calls == []
    assert summary["counts"]["skipped"] == 1


def test_a_failing_group_does_not_stop_the_sweep(tmp_path, monkeypatch):
    config = _config([
        {"id": "g001", "members": [_member("a", 217.0, 57.0)]},
        {"id": "g002", "members": [_member("b", 218.0, 57.0)]}])
    loaded = load_config(_write(tmp_path, config))

    def fake(sources, *, group_id, **kwargs):
        if group_id == "g001":
            raise RuntimeError("service outage")
        return ProviderResult(provider='spherex', status=STATUS_OK,
                              message="done", meta={'tag': 't'})
    monkeypatch.setattr(groups_mod, 'fetch_group', fake)
    summary = run_config(loaded, report_path=tmp_path / "report.csv",
                         progress=lambda m: None)
    assert summary["counts"][STATUS_OK] == 1
    assert summary["counts"]["failed"] == 1
    assert "service outage" in (tmp_path / "report.csv").read_text()


def test_more_workers_than_the_service_runs_is_allowed(tmp_path, monkeypatch):
    # The service QUEUES the extras rather than refusing them, so this is
    # a notice. Refusing would forfeit throughput the service will give.
    loaded = load_config(_write(tmp_path, _config()))
    calls = _fake_run(monkeypatch)
    said = []
    run_config(loaded, report_path=tmp_path / "report.csv",
               workers=DOCUMENTED_CONCURRENCY + 4, progress=said.append)
    assert calls == ["g001"]
    assert any("queue" in m for m in said)


def test_zero_workers_is_refused(tmp_path, monkeypatch):
    loaded = load_config(_write(tmp_path, _config()))
    _fake_run(monkeypatch)
    with pytest.raises(ValueError, match="at least 1"):
        run_config(loaded, report_path=tmp_path / "report.csv", workers=0,
                   progress=lambda m: None)


def test_the_queue_budget_reaches_the_job(tmp_path, monkeypatch):
    loaded = load_config(_write(tmp_path, _config()))
    seen = {}

    def fake(sources, *, group_id, **kwargs):
        seen.update(kwargs)
        return ProviderResult(provider='spherex', status=STATUS_OK,
                              message="done", meta={'tag': 't'})
    monkeypatch.setattr(groups_mod, 'fetch_group', fake)
    run_config(loaded, report_path=tmp_path / "report.csv",
               progress=lambda m: None)
    assert seen['queue_timeout'] == QUEUE_TIMEOUT


def test_per_group_logs_are_written(tmp_path, monkeypatch):
    loaded = load_config(_write(tmp_path, _config()))

    def fake(sources, *, group_id, **kwargs):
        print("inside the job")
        return ProviderResult(provider='spherex', status=STATUS_OK,
                              message="done", meta={'tag': 't'})
    monkeypatch.setattr(groups_mod, 'fetch_group', fake)
    run_config(loaded, report_path=tmp_path / "report.csv",
               log_dir=tmp_path / "logs", progress=lambda m: None)
    assert "inside the job" in (tmp_path / "logs" / "g001.log").read_text()
