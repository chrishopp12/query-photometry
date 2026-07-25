"""Offline tests for the Sersic-model remeasure (no images, no scene, no fit)."""
import json

from sedphot.remeasure import model_flux_within, remeasure_sersic


def test_model_flux_within_interpolates_and_caps():
    rgrid = [2.0, 6.0, 12.0, 20.0]
    cog = [100.0, 300.0, 500.0, 560.0]
    total = 570.0
    # within the grid: linear interpolation
    assert model_flux_within(6.0, rgrid, cog, total) == 300.0
    assert model_flux_within(9.0, rgrid, cog, total) == 400.0    # midway 6->12
    # below the first grid radius: from the pinned origin (0, 0)
    assert model_flux_within(1.0, rgrid, cog, total) == 50.0     # half of cog[0]
    # at or past the last radius: the converged total
    assert model_flux_within(20.0, rgrid, cog, total) == total
    assert model_flux_within(50.0, rgrid, cog, total) == total
    # None or <= 0 requests the integrated total
    assert model_flux_within(None, rgrid, cog, total) == total
    assert model_flux_within(0.0, rgrid, cog, total) == total


def test_remeasure_sersic_table(tmp_path):
    prov = {
        'git_rev': 'abc123',
        'target': {'label': 'g1'},
        'per_band': {
            'Legacy_r': {
                'target_model_uJy': 570.0,
                'fit_state': {'rgrid': [2.0, 6.0, 12.0, 20.0],
                              'model_cog_uJy': [100.0, 300.0, 500.0, 560.0]}},
            'SDSS_u': {'target_model_uJy': 40.0, 'fit_state': {}},  # no COG
        },
    }
    p = tmp_path / 'g1.provenance.json'
    p.write_text(json.dumps(prov))

    at12 = remeasure_sersic(p, 12.0)
    # the band with a stored COG is reported; the demoted band is skipped
    assert list(at12['band']) == ['Legacy_r']
    assert at12.iloc[0]['flux_uJy'] == 500.0
    assert at12.iloc[0]['aperture_as'] == 12.0
    assert 'abc123' in at12.iloc[0]['source']       # git_rev-pinned provenance

    integ = remeasure_sersic(p, None)               # integrated total
    assert integ.iloc[0]['flux_uJy'] == 570.0
    assert integ.iloc[0]['aperture_as'] == float('inf')
