"""write_sidecar: the automatic fields are guaranteed, and a caller key that
names one is reported rather than silently swallowed. Plus the measured
sidecar payload, whose keys remeasure.reconstruct replays. No network."""
from __future__ import annotations

import json

import pytest
from astropy.coordinates import SkyCoord

from sedphot import __version__
from sedphot.pipeline import measure_sidecar_payload
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


# ------------------------------------
# The measured sidecar payload
# ------------------------------------

# Every sidecar key remeasure.reconstruct reads back, minus the automatic
# fields write_sidecar supplies. hst_proposal_id sat on this list and was
# never written: reconstruct replayed None and refetched HST without the
# program restriction the fit had used. Extend this list whenever
# reconstruct learns to read another key.
RECONSTRUCT_READS = (
    'target', 'instruments', 'cutout_arcsec', 'aperture_arcsec',
    'scene', 'legacy', 'hst_proposal_id', 'per_band',
)


def _payload(**overrides):
    """A measured payload with the scene-engine shape, minus a measurement."""
    kwargs = dict(
        coord=SkyCoord(217.48948, 57.04403, unit='deg'),
        label='control_20', target_name=None,
        instruments=['legacy', 'sdss'], mode='aperture',
        aperture_arcsec=12.0, cutout_arcsec=120.0,
        shape_sky=None, shape_origin=None,
        scene={'cat': [0] * 116, 'stars': [0] * 4, 'patches': {}},
        registry_path=None, registry_update=False,
        recipe_snapshot={'BG_RMIN_AS': 15.0},
        legacy_dr='dr9', legacy_bricks=False,
        hst_proposal_id=None,
        measurements=[{'instrument': 'Legacy', 'band': 'z',
                       'witness': {'flux_uJy': 1099.6}}],
    )
    kwargs.update(overrides)
    return measure_sidecar_payload(**kwargs)


@pytest.mark.parametrize('key', RECONSTRUCT_READS)
def test_the_payload_supplies_every_key_reconstruct_reads(key):
    assert key in _payload()


def test_the_payload_records_the_hst_program_when_hst_ran():
    """The regression: an HST rebuild must replay the program restriction."""
    payload = _payload(instruments=['hst'], hst_proposal_id='16729')
    assert payload['hst_proposal_id'] == '16729'


def test_the_hst_program_is_null_when_hst_did_not_run():
    assert _payload(hst_proposal_id='16729')['hst_proposal_id'] is None


def test_the_legacy_block_tracks_the_instrument_list():
    assert _payload()['legacy'] == {'dr': 'dr9', 'bricks': False}
    assert _payload(instruments=['sdss'])['legacy'] is None


def test_the_scene_block_records_what_the_solve_saw():
    scene = {'cat': [0] * 116, 'stars': [0] * 4, 'patches': {'J1428': {}}}
    block = _payload(scene=scene, registry_path='/r.json',
                     registry_update=True)['scene']
    assert block['n_catalog_rows'] == 116
    assert block['n_confirmed_stars'] == 4
    assert block['patches'] == ['J1428']
    assert block['registry_path'] == '/r.json'
    assert block['registry_updated'] is True
    assert block['recipe']['BG_RMIN_AS'] == 15.0


def test_per_band_is_keyed_instrument_band():
    assert list(_payload()['per_band']) == ['Legacy_z']


def test_the_kind_follows_the_mode():
    assert _payload()['kind'] == 'aperture_photometry'
    assert _payload(mode='sersic')['kind'] == 'sersic_photometry'


def test_a_sersic_run_records_where_its_shape_came_from():
    """Without a pinned shape the refit is named, not left null."""
    assert _payload(mode='sersic')['sersic_shape'] == {
        'source': 'reference-band refit'}
    pinned = _payload(mode='sersic', shape_sky={'n': 2.0, 'reff_arcsec': 3.0},
                      shape_origin={'source': 'explicit parameters'})
    assert pinned['sersic_shape'] == {'n': 2.0, 'reff_arcsec': 3.0,
                                      'source': 'explicit parameters'}


def test_the_payload_survives_the_round_trip_through_write_sidecar(tmp_path):
    """The seam is only worth having if what it builds is what lands."""
    path = _product(tmp_path)
    record = json.loads(write_sidecar(path, _payload()).read_text())
    for key in RECONSTRUCT_READS:
        assert key in record
    assert record['package'] == 'sedphot'      # automatic fields still land
    assert record['aperture_arcsec'] == 12.0   # caller fields survive
