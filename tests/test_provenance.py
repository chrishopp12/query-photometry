"""write_sidecar: the automatic fields are guaranteed, and a caller key that
names one is reported rather than silently swallowed. No network."""
from __future__ import annotations

import json

from sedphot import __version__
from sedphot.provenance import write_sidecar


def _product(tmp_path):
    """A written product for the sidecar to describe."""
    path = tmp_path / "t_catalog.csv"
    path.write_text("band,flux_uJy\nSDSS_r,10.0\n", encoding='utf-8')
    return path


def test_sidecar_carries_the_automatic_fields(tmp_path):
    path = _product(tmp_path)
    sidecar = write_sidecar(path, {"kind": "catalog_photometry"})
    assert sidecar.name == "t_catalog.provenance.json"
    record = json.loads(sidecar.read_text())
    assert record['product'] == path.name
    assert record['package'] == 'sedphot'
    assert record['package_version'] == __version__
    assert record['kind'] == 'catalog_photometry'
    assert {'written', 'sha256_16', 'git_rev', 'git_dirty'} <= set(record)


def test_a_caller_key_cannot_replace_an_automatic_field(tmp_path):
    """A caller-supplied hash or version would leave a record that reads
    authoritative and is not, so the automatic value wins."""
    path = _product(tmp_path)
    honest = json.loads(write_sidecar(path, {}).read_text())
    record = json.loads(write_sidecar(path, {
        'product': 'a_different_product.csv',
        'sha256_16': '0' * 16,
        'package_version': '0.0.0',
        'kind': 'catalog_photometry',
    }).read_text())
    assert record['product'] == path.name
    assert record['sha256_16'] == honest['sha256_16']
    assert record['package_version'] == __version__
    assert record['kind'] == 'catalog_photometry'   # other keys still land


def test_a_collision_is_named_rather_than_silent(tmp_path, capsys):
    write_sidecar(_product(tmp_path), {'sha256_16': '0' * 16, 'kind': 'x'})
    out = capsys.readouterr().out
    assert 'sha256_16' in out
    assert 'kind' not in out
