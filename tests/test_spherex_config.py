"""SPHEREx extraction configs: tagged tables coexist, reuse is idempotent."""
import json

import pandas as pd
from astropy.coordinates import SkyCoord

import sedphot.spherex as spherex_mod
from sedphot.spherex import (PRETAG_TABLE_NAME, Sersic, config_payload,
                             extraction_tag, fetch)
from sedphot.results import STATUS_OK

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


def test_existing_tag_is_reused_without_network(monkeypatch, tmp_path):
    tag = extraction_tag(SHAPE, 15, MJD)
    spherex_dir = tmp_path / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True)
    (spherex_dir / f"table_photometry.{tag}.csv").write_text("flux\n1\n")
    _no_network(monkeypatch)

    result = fetch(COORD, out_dir=tmp_path, model=SHAPE, mjd_range=MJD)
    assert result.status == STATUS_OK
    assert result.meta['reused'] is True
    assert result.meta['tag'] == tag


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
