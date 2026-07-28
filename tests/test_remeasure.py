"""Offline tests for --remeasure (no images, no scene, no fit)."""
import json

import pytest

from sedphot.remeasure import model_flux_within, reconstruct, remeasure


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


def _shape_prov():
    """One gating band (a free-target curve stored) and one without."""
    return {
        'git_rev': 'abc123',
        'per_band': {
            'Legacy_r': {
                'target_model_uJy': 570.0,
                'target_model_free_uJy': 610.0,
                'fit_state': {
                    'rgrid': [2.0, 6.0, 12.0, 20.0],
                    'model_cog_uJy': [100.0, 300.0, 500.0, 560.0],
                    'model_cog_free_uJy': [110.0, 330.0, 550.0, 600.0],
                    'enclosed_uJy': [90.0, 290.0, 480.0, 545.0]}},
            'Legacy_z': {
                'target_model_uJy': 400.0,
                'fit_state': {
                    'rgrid': [2.0, 6.0, 12.0, 20.0],
                    'model_cog_uJy': [80.0, 240.0, 360.0, 395.0],
                    'enclosed_uJy': [70.0, 230.0, 350.0, 390.0]}},
        },
    }


def test_sersic_shape_selects_the_per_band_free_curve(tmp_path, capsys):
    """--shape fitted reads the free-target curve where one was stored, and
    demotes to forced -- loudly, and visibly in `source` -- where none was."""
    p = tmp_path / 'g2.provenance.json'
    p.write_text(json.dumps(_shape_prov()))

    forced = remeasure(p, 12.0, mode='sersic', shape='forced')
    fitted = remeasure(p, 12.0, mode='sersic', shape='fitted')
    by_band = {r['band']: r for _, r in fitted.iterrows()}

    # The gating band moves onto its own free-target curve...
    assert dict(zip(forced['band'], forced['flux_uJy']))['Legacy_r'] == 500.0
    assert by_band['Legacy_r']['flux_uJy'] == 550.0
    assert 'fitted' in by_band['Legacy_r']['source']
    # ... the band with no free record stays on the forced one, and says so.
    assert by_band['Legacy_z']['flux_uJy'] == 360.0
    assert 'forced' in by_band['Legacy_z']['source']
    assert 'Legacy_z' in capsys.readouterr().out

    # Integrated reads the matching total, not the forced one.
    assert remeasure(p, None, mode='sersic',
                     shape='fitted').iloc[0]['flux_uJy'] == 610.0


def test_reconstruct_reports_bands_the_rebuild_dropped(tmp_path, capsys,
                                                       monkeypatch):
    """A band that dies inside the pinned rebuild must not vanish silently.

    run_measure reports a dead band by printing and continuing, and
    reconstruct runs it under a stdout redirect to keep the re-report clean
    -- so that message lands in a discarded buffer and the band leaves the
    table with nothing said.
    """
    import pandas as pd

    from sedphot import pipeline

    prov = {'git_rev': 'abc123',
            'target': {'ra_deg': 150.0, 'dec_deg': 2.0, 'label': 'g4'},
            'instruments': ['legacy'], 'cutout_arcsec': 120.0,
            'per_band': {
                'Legacy_r': {'solve': {'params': [1.0] * 6, 'pix_ref': 0.262},
                             'fit_state': {'amps': [['target', 1.0]],
                                           'bg_coefs': [0.0, 0.0, 0.0],
                                           'rgrid': [2.0, 20.0]}},
                'Legacy_z': {'solve': {'params': [1.0] * 6, 'pix_ref': 0.262},
                             'fit_state': {'amps': [['target', 1.0]],
                                           'bg_coefs': [0.0, 0.0, 0.0],
                                           'rgrid': [2.0, 20.0]}}}}
    p = tmp_path / 'Photometry' / 'g4_measured.provenance.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(prov))

    def fake_run_measure(*a, **kw):
        # Legacy_z dies the way the pipeline reports it: printed, then skipped.
        print("  Legacy z FAILED: ValueError: not enough values to unpack")
        return pd.DataFrame([{'band': 'Legacy_r', 'flux_uJy': 700.0}])

    monkeypatch.setattr(pipeline, 'run_measure', fake_run_measure)
    got = reconstruct(p, 30.0)
    assert got == {'Legacy_r': 700.0}
    printed = capsys.readouterr().out
    assert 'Legacy_z' in printed              # named as dropped
    assert 'not enough values to unpack' in printed   # and the reason replayed


def test_reconstruct_replays_the_fetch_options(tmp_path, monkeypatch):
    """The fetch options decide which PIXELS arrive, so defaulting them
    rebuilds on different data: `bricks` swaps Legacy brick coadds (with real
    inverse variance) for viewer cutouts on a different pixel grid. A
    reconstruction that re-fetches differently is not reproducing the fit."""
    import pandas as pd

    from sedphot import pipeline

    prov = {'git_rev': 'abc123',
            'target': {'ra_deg': 150.0, 'dec_deg': 2.0, 'label': 'g5'},
            'instruments': ['legacy'], 'cutout_arcsec': 120.0,
            'legacy': {'dr': 'dr10', 'bricks': True},
            'hst_proposal_id': '15307',
            'per_band': {'Legacy_r': {
                'solve': {'params': [1.0] * 6, 'pix_ref': 0.262},
                'fit_state': {'amps': [['target', 1.0]],
                              'bg_coefs': [0.0, 0.0, 0.0],
                              'rgrid': [2.0, 20.0]}}}}
    p = tmp_path / 'Photometry' / 'g5_measured.provenance.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(prov))

    seen = {}

    def fake_run_measure(*a, **k):
        seen.update(k)
        return pd.DataFrame([{'band': 'Legacy_r', 'flux_uJy': 1.0}])

    monkeypatch.setattr(pipeline, 'run_measure', fake_run_measure)
    reconstruct(p, 30.0)
    assert seen['legacy_bricks'] is True
    assert seen['legacy_dr'] == 'dr10'
    assert seen['hst_proposal_id'] == '15307'
    # ... and the sky boundary the fit was built on rides along too.
    assert 'sky_rmin_arcsec' in seen


def test_beyond_grid_threshold_uses_the_shortest_stored_grid(tmp_path):
    """A band whose grid stops earlier has no value to read past its own end;
    reading it anyway silently returns its capped outermost point. So the
    SHORTEST grid decides when to rebuild, not the longest."""
    prov = {'git_rev': 'abc',
            'target': {'ra_deg': 150.0, 'dec_deg': 2.0},
            'per_band': {
                'Legacy_r': {'target_model_uJy': 1.0,
                             'fit_state': {'rgrid': [2.0, 40.0],
                                           'enclosed_uJy': [1.0, 9.0]}},
                'Legacy_z': {'target_model_uJy': 1.0,
                             'fit_state': {'rgrid': [2.0, 20.0],
                                           'enclosed_uJy': [1.0, 5.0]}}}}
    p = tmp_path / 'g6.provenance.json'
    p.write_text(json.dumps(prov))
    # 30" is inside Legacy_r's grid but PAST Legacy_z's, so it must rebuild
    # rather than hand back Legacy_z's capped r=20 value.
    with pytest.raises(ValueError):          # no pinnable fit in this fixture
        remeasure(p, 30.0, mode='aperture')
    # ... and inside both grids it stays an interpolation.
    assert 'remeasure' in remeasure(p, 15.0, mode='aperture').iloc[0]['source']


def test_remeasure_announces_crossing_the_stored_grid(tmp_path, capsys,
                                                      monkeypatch):
    """Inside the grid this is an interpolation of values already on disk;
    past it the images are re-read and the whole scene rebuilt. The two
    cost wildly different amounts, so the boundary must be reported."""
    import pandas as pd

    from sedphot import pipeline

    prov = {'git_rev': 'abc123',
            'target': {'ra_deg': 150.0, 'dec_deg': 2.0, 'label': 'g8'},
            'instruments': ['legacy'], 'cutout_arcsec': 120.0,
            'per_band': {'Legacy_r': {
                'solve': {'params': [1.0] * 6, 'pix_ref': 0.262},
                'fit_state': {'amps': [['target', 1.0]],
                              'bg_coefs': [0.0, 0.0, 0.0],
                              'rgrid': [2.0, 20.0],
                              'enclosed_uJy': [90.0, 545.0]}}}}
    p = tmp_path / 'Photometry' / 'g8_measured.provenance.json'
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(prov))

    def fake_run_measure(*a, **k):
        return pd.DataFrame([{'band': 'Legacy_r', 'flux_uJy': 700.0}])

    monkeypatch.setattr(pipeline, 'run_measure', fake_run_measure)
    remeasure(p, 30.0, mode='aperture')
    printed = capsys.readouterr().out
    assert '30"' in printed          # what was asked for
    assert '20"' in printed          # and where the stored curve ends

    # Inside the grid nothing is rebuilt, so nothing is announced.
    remeasure(p, 15.0, mode='aperture')
    assert 'past the stored curve' not in capsys.readouterr().out


def test_seatless_band_is_pinnable_without_a_shape_vector():
    """A scene with NO seats needs no shape vector: the fixed components at
    their stored amplitudes on the stored plane ARE the whole fit."""
    from sedphot.remeasure import _build_pin_by_band
    prov = {'per_band': {'Legacy_r': {
        'seat_owners': [],
        'fit_state': {'amps': [['src1', 5.0]], 'bg_coefs': [0.1, 0.0, 0.0]}}}}
    pin = _build_pin_by_band(prov, 'forced')
    assert 'Legacy_r' in pin
    assert pin['Legacy_r']['seat_params'] is None


def test_band_with_seats_but_no_vector_is_not_pinned():
    """Seats existed and nothing covers them, so the band cannot be rebuilt.
    It must stay out of the pin rather than be rendered from a wrong vector."""
    from sedphot.remeasure import _build_pin_by_band
    prov = {'per_band': {'Legacy_r': {
        'seat_owners': ['target'],
        'fit_state': {'amps': [['target', 5.0]], 'bg_coefs': [0.1, 0.0, 0.0]}}}}
    assert _build_pin_by_band(prov, 'forced') == {}


def test_aperture_mode_is_shape_independent_within_the_grid(tmp_path):
    """The empirical curve is a measurement of the pixels, so shape cannot
    change it inside the grid -- only the beyond-grid rebuild uses shape."""
    p = tmp_path / 'g3.provenance.json'
    p.write_text(json.dumps(_shape_prov()))
    a = remeasure(p, 12.0, mode='aperture', shape='forced')
    b = remeasure(p, 12.0, mode='aperture', shape='fitted')
    assert list(a['flux_uJy']) == list(b['flux_uJy']) == [480.0, 350.0]
