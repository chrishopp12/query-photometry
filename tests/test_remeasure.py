"""Offline tests for --remeasure (no images, no scene, no fit)."""
import json

import pytest

from sedphot.remeasure import model_flux_within, remeasure


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


def test_remeasure_both_modes_and_skips(tmp_path):
    prov = {
        'git_rev': 'abc123',
        'per_band': {
            'Legacy_r': {
                'target_model_uJy': 570.0,
                'fit_state': {'rgrid': [2.0, 6.0, 12.0, 20.0],
                              'model_cog_uJy': [100.0, 300.0, 500.0, 560.0],
                              'enclosed_uJy': [90.0, 290.0, 480.0, 545.0]}},
            'SDSS_u': {'target_model_uJy': 40.0, 'fit_state': {}},   # no COG
        },
    }
    p = tmp_path / 'g1.provenance.json'
    p.write_text(json.dumps(prov))

    # sersic: model COG; the demoted band is skipped; integrated -> model total
    s12 = remeasure(p, 12.0, mode='sersic')
    assert list(s12['band']) == ['Legacy_r']
    assert s12.iloc[0]['flux_uJy'] == 500.0
    assert 'abc123' in s12.iloc[0]['source']       # git_rev-pinned provenance
    assert remeasure(p, None, mode='sersic').iloc[0]['flux_uJy'] == 570.0

    # aperture: empirical COG; integrated -> its outermost measured value
    a12 = remeasure(p, 12.0, mode='aperture')
    assert a12.iloc[0]['flux_uJy'] == 480.0
    assert a12.iloc[0]['mode'] == 'aperture'
    assert remeasure(p, None, mode='aperture').iloc[0]['flux_uJy'] == 545.0

    with pytest.raises(ValueError):
        remeasure(p, 12.0, mode='bogus')
