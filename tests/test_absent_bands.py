"""A band that is requested and does not reach the table must say so: in the
coverage report as data, on the console, and in the measure verb's exit code.
Plus the two silent-degrade sites that now refuse. No network."""
from __future__ import annotations

import json

import pytest

from sedphot.measure.aperture import measurement_to_row
from sedphot.results import (ProviderResult, STATUS_OK, absent_bands,
                             write_coverage_report)


def _result(**meta):
    return ProviderResult(provider='cfht', status=STATUS_OK,
                          rows=[{'band': 'CFHT_r'}], message='',
                          meta={"measured_bands": ['r'],
                                "demoted_bands": [],
                                "failed_bands": [], **meta})


# ------------------------------------
# absent_bands
# ------------------------------------
def test_a_clean_provider_reports_nothing_absent():
    assert absent_bands([_result()]) == {'demoted': [], 'failed': []}


def test_a_demoted_band_is_reported():
    absent = absent_bands([_result(demoted_bands=['u (coverage 0.31)'])])
    assert absent['demoted'] == ['cfht u (coverage 0.31)']
    assert absent['failed'] == []


def test_a_failed_band_is_reported_with_its_error():
    absent = absent_bands([_result(
        failed_bands=[{'band': 'i', 'error': 'ValueError', 'message': 'x'}])])
    assert absent['failed'] == ['cfht i (ValueError)']


def test_a_provider_without_the_meta_keys_is_tolerated():
    """Catalog providers build ProviderResult without band bookkeeping."""
    bare = ProviderResult(provider='galex', status=STATUS_OK)
    assert absent_bands([bare]) == {'demoted': [], 'failed': []}


# ------------------------------------
# The coverage report carries it as data
# ------------------------------------
def test_the_report_records_absent_bands_as_data_not_prose(tmp_path):
    """The console log scrolls away; coverage_measure.json is the record."""
    path = write_coverage_report(
        [_result(demoted_bands=['u (coverage 0.31)'],
                 failed_bands=[{'band': 'i', 'error': 'ValueError',
                                'message': 'boom'}])],
        tmp_path / 'coverage_measure.json')
    report = json.loads(path.read_text())['cfht']
    assert report['status'] == STATUS_OK          # the provider still ran
    assert report['measured_bands'] == ['r']
    assert report['demoted_bands'] == ['u (coverage 0.31)']
    assert report['failed_bands'][0]['band'] == 'i'
    assert report['failed_bands'][0]['message'] == 'boom'


def test_a_report_without_band_bookkeeping_keeps_its_old_shape(tmp_path):
    path = write_coverage_report(
        [ProviderResult(provider='galex', status=STATUS_OK, message='hi')],
        tmp_path / 'coverage_catalogs.json')
    entry = json.loads(path.read_text())['galex']
    assert set(entry) == {'status', 'n_rows', 'message', 'radius_used_arcsec'}


# ------------------------------------
# A4: sersic mode with no target model
# ------------------------------------
# The witness keys qa_flags reads unconditionally.
WITNESS = {'cov': 1.0, 'maskfrac_ap': 0.0, 'twinfrac': 0.0,
           'nbsub_ap_uJy': 0.0, 'excess_growth_uJy': 0.0, 'ped_b_sb': 0.0,
           'r_conv_as': 20.0, 'bg_sb': 0.0}


def _measurement(witness):
    return {'witness': {**WITNESS, **witness},
            'instrument': 'CFHT', 'band': 'r',
            'flux_ujy': 100.0, 'flux_err_ujy': 1.0,
            'target_ra': 150.0, 'target_dec': 2.0,
            'n_comps': 3, 'registry_consumed': [], 'err_model': 'skyrms'}


def test_sersic_mode_refuses_when_no_target_model_was_fitted():
    """NaN under a sersic-tagged filename claims a flux that never existed."""
    with pytest.raises(ValueError, match='no target model'):
        measurement_to_row(_measurement({'flux_uJy': 100.0}), mode='sersic')


def test_sersic_mode_reports_the_model_flux_when_there_is_one():
    row = measurement_to_row(
        _measurement({'target_model_uJy': 321.0}), mode='sersic')
    assert row['flux_uJy'] == 321.0
    assert row['source'].startswith('sedphot_sersic_')


def test_aperture_mode_is_unaffected_by_a_missing_target_model():
    row = measurement_to_row(_measurement({'flux_uJy': 100.0}))
    assert row['flux_uJy'] == 100.0
