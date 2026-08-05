"""SPHEREx extraction configs: tagged tables coexist, reuse is idempotent."""
import json

import pandas as pd
from astropy.coordinates import SkyCoord

import sedphot.spherex as spherex_mod
from sedphot.spherex import (PRETAG_TABLE_NAME, Sersic, config_payload,
                             extraction_tag, fetch, find_table)
from sedphot.results import STATUS_ERROR, STATUS_OK

COORD = SkyCoord(217.0, 56.9, unit='deg')
SHAPE = Sersic(n=4.48, axis_ratio=1.31, pa_deg=16.7, reff_arcsec=1.15)
MJD = (60676.0001273, 61174.5063773)


# ------------------------------------
# Tag identity
# ------------------------------------
def test_tag_is_deterministic():
    a = extraction_tag(SHAPE, 15, MJD)
    b = extraction_tag(Sersic(4.48, 1.31, 16.7, 1.15), 15.0, list(MJD))
    assert a == b
    assert a.startswith("sersic-")


def test_tag_separates_configurations():
    base = extraction_tag(SHAPE, 15, MJD)
    assert extraction_tag(None, 15, MJD) != base            # psf vs sersic
    assert extraction_tag(None, 15, MJD).startswith("psf-")
    assert extraction_tag(SHAPE, 20, MJD) != base           # bkg region
    assert extraction_tag(SHAPE, 15, None) != base          # visit window
    other = Sersic(2.0, 1.31, 16.7, 1.15)
    assert extraction_tag(other, 15, MJD) != base           # shape


def test_mjd_order_is_immaterial():
    assert (extraction_tag(SHAPE, 15, MJD)
            == extraction_tag(SHAPE, 15, MJD[::-1]))


# ------------------------------------
# fetch() reuse and coexistence
# ------------------------------------
def _no_network(monkeypatch):
    def explode(*a, **k):
        raise AssertionError("fetch_spectrophotometry must not be called")
    monkeypatch.setattr(spherex_mod, 'fetch_spectrophotometry', explode)


def _fake_network(monkeypatch, n_rows=5):
    def fake(ra, dec, *, model=None, bkg_region_size=15, mjd_range=None,
             out_csv=None, poll=5, timeout=3600):
        df = pd.DataFrame({'flux': range(n_rows)})
        if out_csv:
            df.to_csv(out_csv, index=False)
        return df
    monkeypatch.setattr(spherex_mod, 'fetch_spectrophotometry', fake)


def _write_tagged(spherex_dir, tag, sidecar=True):
    """A tagged table and, by default, the sidecar that vouches for it."""
    table = spherex_dir / f"table_photometry.{tag}.csv"
    table.write_text("flux\n1\n")
    if sidecar:
        table.with_suffix(".provenance.json").write_text(json.dumps({
            "model": {"type": "sersic", "n": 4.48, "axis_ratio": 1.31,
                      "pa_deg": 16.7, "reff_arcsec": 1.15},
            "bkg_region_size_px": 15, "mjd_range": list(MJD)}))
    return table


def test_existing_tag_is_reused_without_network(monkeypatch, tmp_path):
    tag = extraction_tag(SHAPE, 15, MJD)
    spherex_dir = tmp_path / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    _write_tagged(spherex_dir, tag)
    _no_network(monkeypatch)

    result = fetch(COORD, out_dir=tmp_path, model=SHAPE, mjd_range=MJD)
    assert result.status == STATUS_OK
    assert result.meta['reused'] is True
    assert result.meta['tag'] == tag


def test_tagged_table_without_a_sidecar_is_not_reused(monkeypatch, tmp_path):
    """write_sidecar runs after the CSV, so a tagged table missing one was
    never verified and never reached the manifest. Reusing it on the
    strength of its filename would make that permanent -- and the raw table
    must not be overwritten to fix it either."""
    tag = extraction_tag(SHAPE, 15, MJD)
    spherex_dir = tmp_path / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    table = _write_tagged(spherex_dir, tag, sidecar=False)
    _no_network(monkeypatch)          # nothing may be fetched over it either

    result = fetch(COORD, out_dir=tmp_path, model=SHAPE, mjd_range=MJD)
    assert result.status == STATUS_ERROR
    assert table.read_text() == "flux\n1\n"
    assert not (spherex_dir / "extractions.json").exists()


def test_tagged_table_whose_sidecar_disagrees_is_not_reused(monkeypatch,
                                                            tmp_path):
    """A sidecar recording a different configuration is evidence AGAINST
    the filename, so the table cannot stand in for this extraction."""
    tag = extraction_tag(SHAPE, 15, MJD)
    spherex_dir = tmp_path / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    table = _write_tagged(spherex_dir, tag)
    table.with_suffix(".provenance.json").write_text(json.dumps({
        "model": "point", "bkg_region_size_px": 15, "mjd_range": list(MJD)}))
    _no_network(monkeypatch)

    result = fetch(COORD, out_dir=tmp_path, model=SHAPE, mjd_range=MJD)
    assert result.status == STATUS_ERROR
    assert table.read_text() == "flux\n1\n"


def test_pretag_table_with_matching_sidecar_is_reused(monkeypatch, tmp_path):
    # The 58 archive tables predate tags: same requested config -> reuse
    # in place (names are baked into roster provenance; never renamed).
    spherex_dir = tmp_path / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    (spherex_dir / PRETAG_TABLE_NAME).write_text("flux\n1\n")
    sidecar = {
        "model": {"type": "sersic", "n": 4.48, "axis_ratio": 1.31,
                  "pa_deg": 16.7, "reff_arcsec": 1.15,
                  "shape_origin": "ls_dr9.tractor SER, sep 0.11\""},
        "bkg_region_size_px": 15,
        "mjd_range": list(MJD),
        "n_rows": 306,
    }
    (spherex_dir / "table_photometry.provenance.json").write_text(
        json.dumps(sidecar))
    _no_network(monkeypatch)

    result = fetch(COORD, out_dir=tmp_path, model=SHAPE, mjd_range=MJD)
    assert result.status == STATUS_OK
    assert result.meta['reused'] is True
    assert result.meta['path'].endswith(PRETAG_TABLE_NAME)
    # The reuse indexes the pre-tag table under its tag, origin preserved.
    manifest = json.loads((spherex_dir / "extractions.json").read_text())
    entry = manifest["entries"][result.meta['tag']]
    assert entry["file"] == PRETAG_TABLE_NAME
    assert entry["n_rows"] == 306
    assert "tractor" in entry["shape_origin"]


def test_different_config_coexists_with_pretag_table(monkeypatch, tmp_path):
    # A psf extraction alongside an existing sersic table: new tagged
    # file, nothing touched.
    spherex_dir = tmp_path / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    (spherex_dir / PRETAG_TABLE_NAME).write_text("flux\n1\n")
    (spherex_dir / "table_photometry.provenance.json").write_text(json.dumps({
        "model": {"type": "sersic", "n": 4.48, "axis_ratio": 1.31,
                  "pa_deg": 16.7, "reff_arcsec": 1.15},
        "bkg_region_size_px": 15, "mjd_range": list(MJD)}))
    _fake_network(monkeypatch)

    result = fetch(COORD, out_dir=tmp_path, model=None, mjd_range=MJD)
    assert result.status == STATUS_OK
    assert 'reused' not in result.meta
    tag = result.meta['tag']
    assert tag.startswith("psf-")
    assert (spherex_dir / f"table_photometry.{tag}.csv").exists()
    assert (spherex_dir / PRETAG_TABLE_NAME).read_text() == "flux\n1\n"
    manifest = json.loads((spherex_dir / "extractions.json").read_text())
    assert tag in manifest["entries"]


def test_fresh_fetch_writes_sidecar_and_manifest(monkeypatch, tmp_path):
    _fake_network(monkeypatch, n_rows=7)
    result = fetch(COORD, out_dir=tmp_path, model=SHAPE, mjd_range=MJD,
                   shape_origin="explicit parameters")
    assert result.status == STATUS_OK
    tag = result.meta['tag']
    spherex_dir = tmp_path / "Photometry" / "SPHEREx"
    sidecar = json.loads(
        (spherex_dir / f"table_photometry.{tag}.provenance.json").read_text())
    assert sidecar["extraction_tag"] == tag
    assert sidecar["n_rows"] == 7
    manifest = json.loads((spherex_dir / "extractions.json").read_text())
    assert manifest["entries"][tag]["shape_origin"] == "explicit parameters"

    # Immediately re-requesting the same config reuses, no second fetch.
    _no_network(monkeypatch)
    again = fetch(COORD, out_dir=tmp_path, model=SHAPE, mjd_range=MJD)
    assert again.meta['reused'] is True


def test_pretag_table_without_sidecar_is_not_matched(monkeypatch, tmp_path):
    # No sidecar -> unknown config -> fetch fresh alongside, don't guess.
    spherex_dir = tmp_path / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    (spherex_dir / PRETAG_TABLE_NAME).write_text("flux\n1\n")
    _fake_network(monkeypatch)

    result = fetch(COORD, out_dir=tmp_path, model=SHAPE, mjd_range=MJD)
    assert result.status == STATUS_OK
    assert 'reused' not in result.meta


# ------------------------------------
# find_table: the canonical table, config unknown
# ------------------------------------
def test_find_table_takes_the_vouched_tagged_table(tmp_path):
    spherex_dir = tmp_path / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    table = _write_tagged(spherex_dir, "sersic-abc123")
    assert find_table(spherex_dir) == table


def test_find_table_ignores_tables_no_sidecar_vouches_for(tmp_path):
    # The real shape of a directory carrying hand-downloaded tables from
    # before this package: they match the glob, nothing vouches for them.
    spherex_dir = tmp_path / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    for orphan in ("table_photometry-this.csv",
                   "table_photometry-this_sersic.csv",
                   "table_photometry-this_sersic_27.csv"):
        (spherex_dir / orphan).write_text("flux\n1\n")
    assert find_table(spherex_dir) is None

    table = _write_tagged(spherex_dir, "sersic-abc123")
    assert find_table(spherex_dir) == table


def test_find_table_prefers_a_tagged_table_over_the_pretag_name(tmp_path):
    spherex_dir = tmp_path / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    pretag = spherex_dir / PRETAG_TABLE_NAME
    pretag.write_text("flux\n1\n")
    pretag.with_suffix(".provenance.json").write_text("{}")
    tagged = _write_tagged(spherex_dir, "sersic-abc123")
    assert find_table(spherex_dir) == tagged


def test_find_table_falls_back_to_the_pretag_name(tmp_path):
    spherex_dir = tmp_path / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    pretag = spherex_dir / PRETAG_TABLE_NAME
    pretag.write_text("flux\n1\n")
    pretag.with_suffix(".provenance.json").write_text("{}")
    assert find_table(spherex_dir) == pretag


def test_find_table_is_deterministic_across_several_tagged_tables(tmp_path):
    spherex_dir = tmp_path / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    for tag in ("sersic-ffffff", "psf-000000", "sersic-aaaaaa"):
        _write_tagged(spherex_dir, tag)
    # Sorted filename, not directory order.
    assert find_table(spherex_dir).name == "table_photometry.psf-000000.csv"


def test_find_table_is_none_without_a_directory(tmp_path):
    assert find_table(tmp_path / "nope" / "SPHEREx") is None


# ------------------------------------
# Poll resilience
# ------------------------------------
def test_wait_survives_transient_poll_failures(monkeypatch):
    import requests as requests_mod
    from sedphot.spherex import _wait
    monkeypatch.setattr('time.sleep', lambda s: None)
    calls = []

    def flaky_poll():
        calls.append(1)
        if len(calls) < 3:
            raise requests_mod.exceptions.ReadTimeout("dropped read")
        return "COMPLETED", ["result"]

    phase, payload = _wait(flaky_poll, interval=0)
    assert phase == "COMPLETED"
    assert len(calls) == 3


def test_wait_gives_up_after_persistent_failures(monkeypatch):
    import pytest as pytest_mod
    import requests as requests_mod
    from sedphot.spherex import _wait
    monkeypatch.setattr('time.sleep', lambda s: None)

    def dead_poll():
        raise requests_mod.exceptions.ReadTimeout("dead service")

    with pytest_mod.raises(requests_mod.exceptions.ReadTimeout):
        _wait(dead_poll, interval=0, max_poll_failures=3)


def test_sub_threshold_sersic_flagged_as_point_source(monkeypatch, tmp_path, capsys):
    # The tool point-sources anything with reff < 1"; the fetch warns and
    # the sidecar records it so downstream knows the shape was cosmetic.
    _fake_network(monkeypatch)
    tiny = Sersic(n=1.0, axis_ratio=1.0, pa_deg=0.0, reff_arcsec=0.39)
    result = fetch(COORD, out_dir=tmp_path, model=tiny, mjd_range=MJD)
    assert result.status == STATUS_OK
    assert "point-source threshold" in capsys.readouterr().out
    spherex_dir = tmp_path / "Photometry" / "SPHEREx"
    tag = result.meta['tag']
    sidecar = json.loads(
        (spherex_dir / f"table_photometry.{tag}.provenance.json").read_text())
    assert sidecar["model"]["effectively_point_source"] is True


def test_manifest_write_is_atomic_and_accumulates(tmp_path):
    spherex_mod._index_extraction(
        tmp_path, "sersic-abc123", "table_photometry.sersic-abc123.csv",
        config_payload(SHAPE, 15, MJD), shape_origin="test", n_rows=3)
    spherex_mod._index_extraction(
        tmp_path, "psf-def456", "table_photometry.psf-def456.csv",
        config_payload(None, 15, MJD), n_rows=5)
    manifest = json.loads((tmp_path / "extractions.json").read_text())
    assert set(manifest["entries"]) == {"sersic-abc123", "psf-def456"}
    # write-then-replace leaves no sibling temp file behind
    assert [p.name for p in tmp_path.iterdir()] == ["extractions.json"]


# ------------------------------------
# Failure reporting
# ------------------------------------
# Verbatim from IRSA on a job that failed 2026-07-30: the detail lives in
# the job document's errorSummary, and {job}/error is explicitly unused.
IRSA_ERRORED_JOB = """<?xml version="1.0" encoding="UTF-8"?>
<uws:job version="1.1" xmlns:uws="http://www.ivoa.net/xml/UWS/v1.0"
         xmlns:xlink="http://www.w3.org/1999/xlink">
<uws:jobId>j1</uws:jobId>
<uws:phase>ERROR</uws:phase>
<uws:results></uws:results>
<uws:errorSummary type="fatal" hasDetail="false">
<uws:message>Internal error; contact IRSA user support (Subprocess \
executing one of the pipelines failed)</uws:message>
</uws:errorSummary>
</uws:job>"""

IRSA_HEALTHY_JOB = """<?xml version="1.0" encoding="UTF-8"?>
<uws:job version="1.1" xmlns:uws="http://www.ivoa.net/xml/UWS/v1.0">
<uws:jobId>j1</uws:jobId>
<uws:phase>EXECUTING</uws:phase>
<uws:results></uws:results>
</uws:job>"""

IRSA_ERROR_ENDPOINT = ("Error details are included in the job's errorSummary "
                       "element. This endpoint is not used.")

UWS_URL = "https://irsa.ipac.caltech.edu/api/spherex/spectrophotometry/async/j1"


class _Reply:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


class _Session:
    """Answers the job document and /error separately, as UWS does."""

    def __init__(self, job=None, error=None):
        self._job = job
        self._error = error
        self.asked = []

    def get(self, url, timeout=None):
        self.asked.append(url)
        if url.endswith("/error"):
            return self._error or _Reply(404, IRSA_ERROR_ENDPOINT)
        return self._job or _Reply(404, "no job")


def test_uws_error_detail_reads_the_irsa_error_summary():
    from sedphot.spherex import uws_error_detail
    session = _Session(job=_Reply(200, IRSA_ERRORED_JOB))
    detail = uws_error_detail(session, UWS_URL)
    assert "Subprocess executing one of the pipelines failed" in detail
    assert "[fatal]" in detail
    # the job document answers it; /error is IRSA-unused and not needed
    assert session.asked[0] == UWS_URL


def test_uws_error_detail_is_none_for_a_healthy_job():
    # No errorSummary, and IRSA's /error 404s -- neither may read as a
    # message, or every healthy job would look like a failure.
    from sedphot.spherex import uws_error_detail
    session = _Session(job=_Reply(200, IRSA_HEALTHY_JOB))
    assert uws_error_detail(session, UWS_URL) is None


def test_uws_error_detail_falls_back_to_the_error_endpoint():
    # Other UWS services do populate {job}/error; the fallback keeps this
    # useful against them without assuming IRSA's convention.
    from sedphot.spherex import uws_error_detail
    session = _Session(job=_Reply(200, IRSA_HEALTHY_JOB),
                       error=_Reply(200, json.dumps({"detail": "bad shape"})))
    assert uws_error_detail(session, UWS_URL) == "bad shape"


def test_uws_error_detail_survives_an_unreachable_service():
    import requests as requests_mod
    from sedphot.spherex import uws_error_detail

    class _Dead:
        def get(self, url, timeout=None):
            raise requests_mod.exceptions.ReadTimeout("no route")

    assert uws_error_detail(_Dead(), UWS_URL) is None
