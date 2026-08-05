"""Joint SPHEREx extractions: the upload, the split, and group identity."""
import json

import pandas as pd
import pytest

import sedphot.spherex as spherex_mod
from sedphot.results import STATUS_ERROR, STATUS_OK
from sedphot.spherex import (FF_MODE_KEY, FF_MODE_UPLOAD, GROUP_SOURCES_MAX,
                             GroupSource, Sersic, _sidecar_payload,
                             _votable_to_flat, build_group_request,
                             build_upload_frame, check_shape, fetch_group,
                             group_config_payload, group_extraction_tag,
                             split_group_table, upload_table,
                             verify_group_shapes)

SHAPE = Sersic(n=4.06, axis_ratio=1.10, pa_deg=5.08, reff_arcsec=5.19)
OTHER = Sersic(n=1.20, axis_ratio=2.00, pa_deg=90.0, reff_arcsec=2.50)
MJD = (60676.0001273, 61174.5063773)


def _source(label, ra, dec, model=SHAPE, role="science", out_dir=None):
    return GroupSource(label=label, ra_deg=ra, dec_deg=dec, model=model,
                       role=role, out_dir=out_dir)


def _pair(tmp_path=None):
    a = _source("a", 217.00000, 57.00000,
                out_dir=str(tmp_path / "a") if tmp_path else "a")
    b = _source("b", 217.00500, 57.00300, model=OTHER,
                out_dir=str(tmp_path / "b") if tmp_path else "b")
    return [a, b]


# ------------------------------------
# The uploaded target table
# ------------------------------------
def test_upload_frame_carries_every_named_column():
    frame = build_upload_frame(_pair())
    assert list(frame.columns) == ["label", "ra", "dec", "sersic_n",
                                   "axis_ratio", "pa_deg", "reff_arcsec"]
    assert list(frame["label"]) == ["a", "b"]
    # The request names these columns; a rename here silently breaks it.
    request = build_group_request("srv", sersic=True)
    for key in ("uploadCenterLonColumns", "uploadCenterLatColumns",
                "nameColumn", "sersicIdxColumn", "axisRatioColumn",
                "positionAngleColumn", "effectiveRadiusColumn"):
        assert request[key] in frame.columns


def test_upload_frame_writes_radius_in_arcsec():
    # The single-position path sends DEGREES; the column path must not,
    # or every radius is off by 3600 with no error anywhere.
    frame = build_upload_frame(_pair())
    assert frame["reff_arcsec"].iloc[0] == pytest.approx(SHAPE.reff_arcsec)


def test_point_source_group_omits_the_shape_columns():
    frame = build_upload_frame([_source("a", 217.0, 57.0, model=None)])
    assert list(frame.columns) == ["label", "ra", "dec"]
    request = build_group_request("srv", sersic=False)
    assert request["shapeFit"] == "false"
    assert "sersicIdxColumn" not in request


def test_upload_frame_refuses_a_group_it_cannot_represent():
    with pytest.raises(ValueError, match="at least one source"):
        build_upload_frame([])
    too_many = [_source(f"s{i}", 217.0 + i * 0.01, 57.0)
                for i in range(GROUP_SOURCES_MAX + 1)]
    with pytest.raises(ValueError, match="exceeds"):
        build_upload_frame(too_many)
    with pytest.raises(ValueError, match="duplicate labels"):
        build_upload_frame([_source("a", 217.0, 57.0),
                            _source("a", 217.1, 57.1)])
    with pytest.raises(ValueError, match="point-source or elliptical"):
        build_upload_frame([_source("a", 217.0, 57.0),
                            _source("b", 217.1, 57.1, model=None)])


# ------------------------------------
# The request
# ------------------------------------
def test_group_request_switches_mode_and_caps_rows():
    request = build_group_request("srv/file.csv", sersic=True,
                                  bkg_region_size=21, mjd_range=MJD)
    assert request[FF_MODE_KEY] == FF_MODE_UPLOAD
    assert "UserTargetWorldPt" not in request
    assert request["uploadFile"] == "srv/file.csv"
    assert request["uploadRowsMax"] == GROUP_SOURCES_MAX
    # Shared with the single-position path, and still an integer string.
    assert request["bgEstimationRegion"] == "21"
    assert request["exposureTimeMode"] == "mjd"
    assert request["startTime"].startswith("60676.")


def test_single_position_request_is_unchanged():
    request = spherex_mod.build_server_request(217.0, 57.0, model=SHAPE)
    assert request[FF_MODE_KEY] == "CONE"
    assert request["UserTargetWorldPt"] == "217.0;57.0;EQ_J2000"
    assert float(request["effectiveRadius"]) == pytest.approx(
        SHAPE.reff_arcsec / 3600.0)


# ------------------------------------
# Upload transport
# ------------------------------------
class _Response:
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class _Session:
    def __init__(self, text):
        self.text = text
        self.posted = None

    def post(self, url, **kwargs):
        self.posted = (url, kwargs)
        return _Response(self.text)


TOKEN = "${upload-dir}/0F5/upload_1421928165089579124_sedphot_targets.csv"


@pytest.mark.parametrize("body", [
    # The observed live reply: positional, message empty on success, and
    # a token carrying a placeholder the server expands.
    f"200::::{TOKEN}",
    f"200::uploaded::{TOKEN}",
    f'{{"cacheKey": "{TOKEN}"}}',
    f'[{{"success": true, "serverFile": "{TOKEN}"}}]',
])
def test_upload_finds_the_token_in_every_known_shape(body):
    session = _Session(body)
    assert upload_table(session, build_upload_frame(_pair())) == TOKEN
    url, kwargs = session.posted
    # The command rides in the query string; a body cmd is answered 500.
    assert url.endswith("/CmdSrv/sync?cmd=upload")
    assert "data" not in kwargs
    assert "file" in kwargs["files"]


@pytest.mark.parametrize("body", [
    "<html>maintenance</html>",
    '{"success":false,"error":{}}',
    "500::something broke::",
])
def test_upload_refuses_an_unrecognized_response(body):
    with pytest.raises(RuntimeError, match="no server file token"):
        upload_table(_Session(body), build_upload_frame(_pair()))


# ------------------------------------
# Parsing a multi-source result
# ------------------------------------
def _votable(rows):
    fields = "".join(
        f'<FIELD name="{n}" datatype="double"/>'
        for n in ("ra", "dec", "sersic", "ab_ratio", "phi", "shape_r"))
    fields = ('<FIELD name="source_id" datatype="int"/>' + fields
              + '<FIELD name="lambda" datatype="double" arraysize="*"/>'
              + '<FIELD name="flux" datatype="double" arraysize="*"/>')
    body = ""
    for row in rows:
        cells = "".join(f"<TD>{v}</TD>" for v in row)
        body += f"<TR>{cells}</TR>"
    return (
        '<?xml version="1.0"?>'
        '<VOTABLE version="1.3" xmlns="http://www.ivoa.net/xml/VOTable/v1.3">'
        f"<RESOURCE><TABLE>{fields}"
        f"<DATA><TABLEDATA>{body}</TABLEDATA></DATA>"
        "</TABLE></RESOURCE></VOTABLE>").encode()


def _row(source_id, source, n_visits):
    model = source.model
    waves = " ".join(str(1.0 + i) for i in range(n_visits))
    fluxes = " ".join(str(10.0 * (i + 1)) for i in range(n_visits))
    return (source_id, source.ra_deg, source.dec_deg, model.n,
            model.axis_ratio, model.pa_deg, model.reff_arcsec, waves, fluxes)


def test_every_source_survives_the_parse():
    a, b = _pair()
    frame = _votable_to_flat(_votable([_row(1, a, 3), _row(2, b, 2)]))
    # Reading only the first row would give 3 rows and look entirely
    # well-formed -- this is the check that catches it.
    assert len(frame) == 5
    assert sorted(frame["source_id"].unique()) == [1, 2]
    assert list(frame[frame["source_id"] == 1]["flux"]) == [10.0, 20.0, 30.0]


def test_a_single_source_parse_is_unchanged():
    a, _ = _pair()
    frame = _votable_to_flat(_votable([_row(1, a, 4)]))
    assert len(frame) == 4
    assert set(frame["ra"]) == {a.ra_deg}


# ------------------------------------
# Splitting a joint result
# ------------------------------------
def test_split_attributes_rows_by_position():
    sources = _pair()
    frame = _votable_to_flat(
        _votable([_row(1, sources[1], 2), _row(2, sources[0], 3)]))
    blocks, complaints = split_group_table(frame, sources)
    assert complaints == []
    # Matched on position, so the tool's own row order is immaterial.
    assert len(blocks["a"]) == 3
    assert len(blocks["b"]) == 2


def test_split_refuses_a_member_the_job_did_not_return():
    sources = _pair()
    frame = _votable_to_flat(_votable([_row(1, sources[0], 3)]))
    with pytest.raises(RuntimeError, match=r"b: no source within"):
        split_group_table(frame, sources)


def test_split_refuses_a_source_no_member_claims():
    sources = _pair()
    stray = _source("stray", 217.05, 57.05)
    frame = _votable_to_flat(_votable([_row(1, sources[0], 2),
                                       _row(2, sources[1], 2),
                                       _row(3, stray, 2)]))
    with pytest.raises(RuntimeError, match="no member claims"):
        split_group_table(frame, sources)


def test_split_refuses_members_it_cannot_tell_apart():
    # Two members closer together than the match radius: every member
    # matches both spectra, so no attribution is defensible.
    sources = [_source("a", 217.0, 57.0),
               _source("b", 217.0 + 1e-4, 57.0)]
    frame = _votable_to_flat(_votable([_row(1, sources[0], 2),
                                       _row(2, sources[1], 2)]))
    with pytest.raises(RuntimeError, match="cannot be told apart"):
        split_group_table(frame, sources, tol_arcsec=5.0)
    # At a radius that separates them, the same result splits cleanly.
    blocks, _ = split_group_table(frame, sources, tol_arcsec=0.1)
    assert set(blocks) == {"a", "b"}


def test_split_refuses_two_members_at_one_position():
    sources = [_source("a", 217.0, 57.0), _source("b", 217.0, 57.0)]
    frame = _votable_to_flat(_votable([_row(1, sources[0], 2)]))
    with pytest.raises(RuntimeError, match="already claimed"):
        split_group_table(frame, sources)


def test_the_shape_echo_catches_a_unit_error():
    source = _source("a", 217.0, 57.0)
    # What a radius uploaded in degrees and read as arcsec looks like.
    block = pd.DataFrame({"sersic": [SHAPE.n], "ab_ratio": [SHAPE.axis_ratio],
                          "phi": [SHAPE.pa_deg],
                          "shape_r": [SHAPE.reff_arcsec / 3600.0]})
    complaints = verify_group_shapes(block, source)
    assert len(complaints) == 1
    assert "shape_r" in complaints[0]


def test_the_shape_echo_passes_a_faithful_round_trip():
    source = _source("a", 217.0, 57.0)
    block = pd.DataFrame({"sersic": [SHAPE.n], "ab_ratio": [SHAPE.axis_ratio],
                          "phi": [SHAPE.pa_deg + 360.0],
                          "shape_r": [SHAPE.reff_arcsec]})
    assert verify_group_shapes(block, source) == []


# ------------------------------------
# Group identity
# ------------------------------------
def test_membership_is_part_of_the_extraction():
    solo = group_extraction_tag([_pair()[0]], 15, MJD)
    joint = group_extraction_tag(_pair(), 15, MJD)
    assert solo != joint


def test_upload_order_is_immaterial():
    sources = _pair()
    assert (group_extraction_tag(sources, 15, MJD)
            == group_extraction_tag(sources[::-1], 15, MJD))


def test_role_and_bookkeeping_do_not_change_the_extraction():
    sources = _pair()
    relabeled = [GroupSource(label="renamed", ra_deg=sources[0].ra_deg,
                              dec_deg=sources[0].dec_deg,
                              model=sources[0].model, role="ancillary"),
                  sources[1]]
    assert (group_extraction_tag(sources, 15, MJD)
            == group_extraction_tag(relabeled, 15, MJD))


def test_a_joint_tag_cannot_match_a_solo_glob():
    from fnmatch import fnmatch
    joint = f"table_photometry.{group_extraction_tag(_pair(), 15, MJD)}.csv"
    assert joint.startswith("table_photometry.joint-sersic-")
    assert not fnmatch(joint, "table_photometry.sersic-*.csv")
    assert not fnmatch(joint, "table_photometry.psf-*.csv")


# ------------------------------------
# Shape bounds the tool does not enforce on an upload
# ------------------------------------
def test_check_shape_names_every_stated_bound():
    assert check_shape(SHAPE) == []
    assert check_shape(Sersic(0.2, 1.1, 5.0, 5.0))          # n too low
    assert check_shape(Sersic(9.0, 1.1, 5.0, 5.0))          # n too high
    assert check_shape(Sersic(4.0, 0.5, 5.0, 5.0))          # a/b < 1
    assert check_shape(Sersic(4.0, 1.1, 5.0, 25.0))         # reff ceiling


# ------------------------------------
# fetch_group products
# ------------------------------------
def _fake_job(monkeypatch, sources, n_visits=3):
    def fake(srcs, **kwargs):
        frame = _votable_to_flat(
            _votable([_row(i + 1, s, n_visits) for i, s in enumerate(srcs)]))
        blocks, complaints = split_group_table(frame, srcs)
        return blocks, frame, complaints
    monkeypatch.setattr(spherex_mod, 'fetch_group_spectrophotometry', fake)


def _no_job(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("the job must not be submitted")
    monkeypatch.setattr(spherex_mod, 'fetch_group_spectrophotometry', explode)


def test_each_science_member_gets_its_own_table(tmp_path, monkeypatch):
    sources = _pair(tmp_path)
    _fake_job(monkeypatch, sources)
    result = fetch_group(sources, group_id="g001",
                         group_dir=tmp_path / "groups", mjd_range=MJD)
    assert result.status == STATUS_OK
    tag = result.meta['tag']
    for source in sources:
        path = tmp_path / source.label / "Photometry" / "SPHEREx" / \
            f"table_photometry.{tag}.csv"
        assert path.is_file()
        # One table, one source: what every downstream consumer reads.
        assert pd.read_csv(path)["source_id"].nunique() == 1
    assert (tmp_path / "groups" / f"joint_photometry.{tag}.csv").is_file()


def test_the_sidecar_round_trips_the_group(tmp_path, monkeypatch):
    sources = _pair(tmp_path)
    _fake_job(monkeypatch, sources)
    result = fetch_group(sources, group_id="g001", mjd_range=MJD)
    tag = result.meta['tag']
    sidecar = json.loads(
        (tmp_path / "a" / "Photometry" / "SPHEREx"
         / f"table_photometry.{tag}.provenance.json").read_text())
    assert sidecar['joint']['group_id'] == "g001"
    assert [m['label'] for m in sidecar['joint']['members']] == ["a", "b"]
    # The reuse check reads the group back out of the record.
    assert _sidecar_payload(sidecar) == group_config_payload(sources, 15, MJD)


def test_rerunning_a_group_reuses_it(tmp_path, monkeypatch):
    sources = _pair(tmp_path)
    _fake_job(monkeypatch, sources)
    first = fetch_group(sources, group_id="g001", mjd_range=MJD)
    _no_job(monkeypatch)
    again = fetch_group(sources, group_id="g001", mjd_range=MJD)
    assert again.status == STATUS_OK
    assert again.meta['reused'] is True
    assert again.meta['tag'] == first.meta['tag']


def test_a_different_membership_is_a_different_extraction(tmp_path,
                                                          monkeypatch):
    sources = _pair(tmp_path)
    _fake_job(monkeypatch, sources)
    joint = fetch_group(sources, group_id="g001", mjd_range=MJD)
    _fake_job(monkeypatch, sources[:1])
    solo = fetch_group(sources[:1], group_id="g002", mjd_range=MJD)
    assert solo.meta['tag'] != joint.meta['tag']
    # Both coexist; neither was reused for the other.
    directory = tmp_path / "a" / "Photometry" / "SPHEREx"
    assert len(list(directory.glob("table_photometry.*.csv"))) == 2


def test_an_unphysical_member_never_reaches_the_service(tmp_path,
                                                        monkeypatch):
    _no_job(monkeypatch)
    sources = _pair(tmp_path)
    sources[1].model = Sersic(4.0, 1.1, 5.0, 40.0)
    result = fetch_group(sources, group_id="g001")
    assert result.status == STATUS_ERROR
    assert "ceiling" in result.message


def test_a_disagreeing_shape_echo_writes_no_product(tmp_path, monkeypatch):
    sources = _pair(tmp_path)

    def fake(srcs, **kwargs):
        frame = _votable_to_flat(
            _votable([_row(i + 1, s, 3) for i, s in enumerate(srcs)]))
        blocks, _ = split_group_table(frame, srcs)
        return blocks, frame, ["a: the tool used shape_r=0.0014"]
    monkeypatch.setattr(spherex_mod, 'fetch_group_spectrophotometry', fake)

    result = fetch_group(sources, group_id="g001",
                         group_dir=tmp_path / "groups")
    assert result.status == STATUS_ERROR
    assert not list((tmp_path / "a").rglob("table_photometry.*.csv"))
    # The hour of compute is kept for diagnosis, but not as a product.
    quarantined = list((tmp_path / "groups").glob("*.unverified.csv"))
    assert len(quarantined) == 1
    assert not quarantined[0].with_suffix(".provenance.json").exists()


def test_an_unvouched_table_is_refused_not_overwritten(tmp_path, monkeypatch):
    sources = _pair(tmp_path)
    _fake_job(monkeypatch, sources)
    tag = group_extraction_tag(sources, 15, None)
    squatter = (tmp_path / "a" / "Photometry" / "SPHEREx"
                / f"table_photometry.{tag}.csv")
    squatter.parent.mkdir(parents=True)
    squatter.write_text("hand,downloaded\n1,2\n")

    result = fetch_group(sources, group_id="g001")
    assert result.status == STATUS_ERROR
    assert squatter.read_text() == "hand,downloaded\n1,2\n"
    # The members it could write, it wrote.
    assert (tmp_path / "b" / "Photometry" / "SPHEREx"
            / f"table_photometry.{tag}.csv").is_file()


def test_a_science_member_needs_somewhere_to_write(tmp_path, monkeypatch):
    _no_job(monkeypatch)
    sources = [_source("a", 217.0, 57.0)]
    sources[0].out_dir = None
    result = fetch_group(sources, group_id="g001")
    assert result.status == STATUS_ERROR
    assert "out_dir" in result.message


# ------------------------------------
# Queue time is budgeted apart from run time
# ------------------------------------
def _phases(sequence):
    """A poll function returning each phase in turn, then holding."""
    remaining = list(sequence)

    def poll():
        phase = remaining.pop(0) if remaining else sequence[-1]
        return phase, []
    return poll


def test_a_queued_job_does_not_spend_the_run_budget(monkeypatch):
    # A job that queues far longer than the run timeout, then runs
    # briefly, must complete: it had not started, so it had consumed no
    # extraction time. A single wall-clock budget would kill it -- and
    # would do so more readily the more jobs the caller submitted.
    clock = [0.0]
    monkeypatch.setattr(spherex_mod.time, 'time', lambda: clock[0])
    monkeypatch.setattr(spherex_mod.time, 'sleep',
                        lambda s: clock.__setitem__(0, clock[0] + 600))
    poll = _phases(["QUEUED"] * 20 + ["EXECUTING", "COMPLETED"])
    phase, _ = spherex_mod._wait(poll, interval=1, timeout=1800,
                                 queue_timeout=36000)
    assert phase == "COMPLETED"
    assert clock[0] > 1800          # far past the run budget, still fine


def test_a_job_that_never_starts_still_gives_up(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(spherex_mod.time, 'time', lambda: clock[0])
    monkeypatch.setattr(spherex_mod.time, 'sleep',
                        lambda s: clock.__setitem__(0, clock[0] + 600))
    with pytest.raises(TimeoutError, match="never started"):
        spherex_mod._wait(_phases(["QUEUED"]), interval=1, timeout=1800,
                          queue_timeout=3600)


def test_a_job_that_runs_too_long_still_times_out(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(spherex_mod.time, 'time', lambda: clock[0])
    monkeypatch.setattr(spherex_mod.time, 'sleep',
                        lambda s: clock.__setitem__(0, clock[0] + 600))
    with pytest.raises(TimeoutError, match="still 'EXECUTING'"):
        spherex_mod._wait(_phases(["EXECUTING"]), interval=1, timeout=1800,
                          queue_timeout=36000)
