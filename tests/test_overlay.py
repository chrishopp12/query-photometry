"""
test_overlay.py

Catalog-Position Overlay
---------------------------------------------------------
Offline tests for sedphot.overlay: style resolution against the schema's
source vocabulary, the grid-mismatch refusal, marker deduplication, and
the figure assembled from a synthetic composite. No network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS

from sedphot import overlay
from sedphot.schema import SOURCE_PREFIXES, make_row, rows_to_frame

TARGET_RA, TARGET_DEC = 150.10155, 2.20482
PIX_DEG = 0.05 / 3600.0        # 0.05"/px, the ACS/WFC drizzle scale
SHAPE = (400, 400)             # 20" x 20", enough for both panels


# ------------------------------------
# Fixtures
# ------------------------------------
def synthetic_wcs(shape=SHAPE):
    """A TAN WCS centered on the target."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    wcs.wcs.crval = [TARGET_RA, TARGET_DEC]
    wcs.wcs.cdelt = [-PIX_DEG, PIX_DEG]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


def synthetic_rgb(shape=SHAPE):
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(*shape, 3), dtype=np.uint8)


def write_pair(tmp_path, rgb_shape=SHAPE, fits_shape=SHAPE):
    """Write a composite JPG and a WCS-bearing FITS to tmp_path."""
    from PIL import Image

    color = tmp_path / "hst_total_drc_color.jpg"
    Image.fromarray(synthetic_rgb(rgb_shape)).save(color)
    wcs_path = tmp_path / "hst_total_drc.fits"
    fits.writeto(wcs_path, np.zeros(fits_shape, dtype="f4"),
                 synthetic_wcs(fits_shape).to_header(), overwrite=True)
    return color, wcs_path


def phot_table(sources, offsets_arcsec=None):
    """A schema table: one row per source, optionally offset from target."""
    offsets = offsets_arcsec or {}
    rows = []
    for i, source in enumerate(sources):
        d_ra, d_dec = offsets.get(source, (0.0, 0.0))
        rows.append(make_row(
            band=f'Band_{i}', flux_ujy=10.0, flux_err_ujy=1.0,
            mag=20.0, mag_err=0.1,
            target_ra=TARGET_RA, target_dec=TARGET_DEC,
            match_ra=TARGET_RA + d_ra / 3600.0,
            match_dec=TARGET_DEC + d_dec / 3600.0,
            sep_arcsec=float(np.hypot(d_ra, d_dec)), flags='', source=source))
    return rows_to_frame(rows)


# ------------------------------------
# Style resolution
# ------------------------------------
def test_every_schema_source_prefix_has_a_style():
    """A provider the schema knows must never fall through to 'other' --
    that was the drift the standalone script carried."""
    for token, prefix in SOURCE_PREFIXES.items():
        style = overlay.style_for(f"{prefix}SOMETHING")
        assert style is not overlay.DEFAULT_STYLE, f"{token} unstyled"
        assert overlay._token_of(f"{prefix}SOMETHING") == token


def test_data_release_does_not_change_the_marker():
    assert overlay.style_for('Legacy_DR9') is overlay.style_for('Legacy_DR10')


def test_unwise_is_not_swallowed_by_the_legacy_prefix():
    assert (overlay.style_for('unWISE_Legacy_DR9')
            is not overlay.style_for('Legacy_DR9'))


def test_hap_segment_and_point_keep_distinct_styles():
    seg = overlay.style_for('HST_HAP_segment')
    pt = overlay.style_for('HST_HAP_point')
    assert seg['marker'] != pt['marker']
    assert seg['label'] != pt['label']


def test_unknown_source_falls_back():
    assert overlay.style_for('SomethingNew_DR1') is overlay.DEFAULT_STYLE


def test_source_prefixes_stay_prefix_free():
    """style_for returns the first hit, so no prefix may contain another."""
    prefixes = list(SOURCE_PREFIXES.values())
    for a in prefixes:
        for b in prefixes:
            if a is not b:
                assert not a.startswith(b), f"{a!r} starts with {b!r}"


# ------------------------------------
# Composite loading
# ------------------------------------
def test_load_color_image_and_wcs_round_trip(tmp_path):
    color, wcs_path = write_pair(tmp_path)
    loaded = overlay.load_color_image_and_wcs(color, wcs_path)
    assert loaded is not None
    rgb, wcs = loaded
    assert rgb.shape == (*SHAPE, 3)
    # the target must land at the center of the flipped array
    x, y = wcs.world_to_pixel(SkyCoord(TARGET_RA, TARGET_DEC, unit='deg'))
    assert x == pytest.approx(SHAPE[1] / 2, abs=1.5)


def test_grid_mismatch_refuses_rather_than_misplacing_markers(tmp_path):
    """The composite carries no WCS; a wrong grid would place every marker
    at a plausible-looking wrong position."""
    color, wcs_path = write_pair(tmp_path, fits_shape=(300, 300))
    assert overlay.load_color_image_and_wcs(color, wcs_path) is None


# ------------------------------------
# Markers
# ------------------------------------
def _axis_with_wcs():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection=synthetic_wcs())
    return fig, ax


def test_one_marker_per_provider_not_per_band():
    """A table carries a row per band; every band of a catalog matched the
    same object, so the panel must not stack ten identical markers."""
    import matplotlib.pyplot as plt

    table = phot_table(['Legacy_DR9'] * 6 + ['PanSTARRS_DR1'] * 5)
    fig, ax = _axis_with_wcs()
    overlay.overlay_markers(ax, synthetic_wcs(), table,
                            SkyCoord(TARGET_RA, TARGET_DEC, unit='deg'))
    assert len(ax.collections) == 2
    plt.close(fig)


def test_same_provider_at_two_positions_draws_twice():
    import matplotlib.pyplot as plt

    table = phot_table(['HST_HAP_segment', 'HST_HAP_point'],
                       offsets_arcsec={'HST_HAP_point': (0.4, 0.0)})
    fig, ax = _axis_with_wcs()
    overlay.overlay_markers(ax, synthetic_wcs(), table,
                            SkyCoord(TARGET_RA, TARGET_DEC, unit='deg'))
    assert len(ax.collections) == 2
    plt.close(fig)


def test_nonfinite_match_positions_are_skipped():
    import matplotlib.pyplot as plt

    table = phot_table(['Legacy_DR9'])
    table.loc[0, 'match_ra'] = np.nan
    fig, ax = _axis_with_wcs()
    overlay.overlay_markers(ax, synthetic_wcs(), table,
                            SkyCoord(TARGET_RA, TARGET_DEC, unit='deg'))
    assert len(ax.collections) == 0
    plt.close(fig)


def test_legend_lists_the_target_and_each_provider_once():
    handles = overlay.legend_handles(
        {'Legacy_DR9', 'Legacy_DR10', 'PanSTARRS_DR1', 'HST_HAP_segment'})
    labels = [h.get_label() for h in handles]
    assert labels[0] == 'target'
    assert labels.count('Legacy') == 1
    assert 'Pan-STARRS' in labels and 'HST HAP (segment)' in labels


# ------------------------------------
# Figure
# ------------------------------------
def test_plot_overlay_writes_a_figure(tmp_path):
    color, wcs_path = write_pair(tmp_path)
    rgb, wcs = overlay.load_color_image_and_wcs(color, wcs_path)
    coord = SkyCoord(TARGET_RA, TARGET_DEC, unit='deg')
    panel = {
        'label': 'ngc_4889',
        'coord': coord,
        'table': phot_table(['Legacy_DR9', 'PanSTARRS_DR1']),
        'ctx': overlay.rgb_cutout(rgb, wcs, coord, 8.0),
        'zoom': overlay.rgb_cutout(rgb, wcs, coord, 3.0),
    }
    assert panel['ctx'] is not None and panel['zoom'] is not None
    out = overlay.plot_overlay([panel], tmp_path / "overlay.png",
                               zoom_arcsec=3.0, context_arcsec=8.0, dpi=60)
    assert out.exists() and out.stat().st_size > 0


def test_rgb_cutout_off_the_mosaic_returns_none():
    rgb, wcs = synthetic_rgb(), synthetic_wcs()
    far = SkyCoord(TARGET_RA + 1.0, TARGET_DEC + 1.0, unit='deg')
    assert overlay.rgb_cutout(rgb, wcs, far, 5.0) is None


# ------------------------------------
# Driver
# ------------------------------------
def test_build_returns_none_when_no_table_has_rows(tmp_path):
    empty = pd.DataFrame(columns=['target_ra', 'target_dec', 'source'])
    assert overlay.build({'a': empty}, tmp_path / "x.png", tmp_path) is None


def test_run_overlay_reports_missing_tables(tmp_path, capsys):
    from sedphot.pipeline import run_overlay

    (tmp_path / "Photometry").mkdir(parents=True)
    assert run_overlay('nothing_here', tmp_path) is None
    assert "no tables" in capsys.readouterr().out


def test_run_overlay_joins_catalog_and_measured_into_one_panel(tmp_path,
                                                              monkeypatch):
    """Both tables describe one target, so they share one panel pair."""
    phot_dir = tmp_path / "Photometry"
    phot_dir.mkdir(parents=True)
    phot_table(['Legacy_DR9']).to_csv(phot_dir / "t_catalog.csv", index=False)
    phot_table(['sedphot_aperture_Legacy_z']).to_csv(
        phot_dir / "t_measured.csv", index=False)

    seen = {}

    def fake_build(tables, outpath, cache_dir, **kwargs):
        seen['tables'] = tables
        seen['cache_dir'] = cache_dir
        return outpath

    monkeypatch.setattr(overlay, 'build', fake_build)
    from sedphot.pipeline import run_overlay
    run_overlay('t', tmp_path)

    assert list(seen['tables']) == ['t']
    joined = seen['tables']['t']
    assert len(joined) == 2
    assert set(joined['source']) == {'Legacy_DR9', 'sedphot_aperture_Legacy_z'}
    assert seen['cache_dir'].name == 'HST'
