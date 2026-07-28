"""
overlay.py

Catalog-Position Overlay on the HAP Color Composite
---------------------------------------------------------
Where every catalog thinks the target is. Fetches the Hubble Advanced
Products (HAP) pipeline's pre-made color composite for the field and plots
each table row's matched position on it, so a table can be checked for the
failure the flux columns cannot show: two providers confidently reporting
photometry for two different objects.

This answers a positional question, not a photometric one -- the QA scene
figures cover the latter. It reads only the schema columns match_ra,
match_dec, and source, which is why the first twelve columns of the table
are a frozen contract (see schema.py).

Data products (written under <out-dir>/Photometry/):
    <label>_overlay.png        context + zoom panels per target
    HST/*_drc_color.jpg        the cached HAP composite
    HST/*total*_drc.fits       the cached detection mosaic (WCS only)

Requirements:
    numpy, pandas, matplotlib, astropy, astroquery, pillow

Notes:
    The composite carries no WCS of its own, so the detection mosaic it
    was rendered from supplies one -- a ~380 MB download the first time a
    field is used, cached thereafter. Pass wcs_from= to point at a local
    file already on that drizzle grid instead; a path that names no file
    refuses rather than reinstating the download, and the pixel dimensions
    are checked against the JPG either way, so a grid mismatch refuses
    rather than silently misplacing every marker.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Headless backend: figures are only ever written to disk.
matplotlib.use("Agg")
import astropy.units as u
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .schema import SOURCE_PREFIXES

# ------------------------------------
# Constants
# ------------------------------------
# Marker style per source -- open symbols, so a marker never hides the
# object underneath it. An exact source string wins; otherwise the key is
# the SOURCE_PREFIXES provider token, so a data release carried in the
# string (Legacy_DR9 vs Legacy_DR10) never changes how a point is drawn.
PROVIDER_STYLES = {
    # Exact source strings first, for the one case where a single provider
    # reports genuinely different positions: HAP segment centroids are
    # isophotal, point centroids are PSF-fit, and on an extended source
    # they part by tenths of an arcsec -- visible on a 5" zoom panel.
    'HST_HAP_segment': dict(marker='o', s=70, edgecolors='tab:green', label='HST HAP (segment)'),
    'HST_HAP_point':   dict(marker='D', s=55, edgecolors='tab:olive', label='HST HAP (point)'),

    'legacy':    dict(marker='s', s=70, edgecolors='tab:orange',  label='Legacy'),
    'unwise':    dict(marker='P', s=70, edgecolors='tab:brown',   label='unWISE (Legacy-forced)'),
    'allwise':   dict(marker='X', s=70, edgecolors='tab:red',     label='AllWISE'),
    'galex':     dict(marker='v', s=80, edgecolors='tab:purple',  label='GALEX'),
    'sdss':      dict(marker='p', s=80, edgecolors='tab:cyan',    label='SDSS'),
    'panstarrs': dict(marker='^', s=80, edgecolors='tab:blue',    label='Pan-STARRS'),
    'jplus':     dict(marker='*', s=110, edgecolors='tab:pink',   label='J-PLUS'),
    # Fallback for a HAP catalog type not listed above. Deliberately NOT the
    # segment style: an unrecognized type drawn as segment would be
    # indistinguishable from the real thing on the figure.
    'hst':       dict(marker='8', s=70, edgecolors='tab:gray',    label='HST HAP (other)'),
    'aperture':  dict(marker='D', s=55, edgecolors='white',       label='sedphot aperture'),
    'sersic':    dict(marker='h', s=60, edgecolors='0.8',         label='sedphot sersic'),
}
DEFAULT_STYLE = dict(marker='D', s=55, edgecolors='yellow', label='other')

# Shared to every scatter call; kept out of the table so a style entry
# only ever declares what makes it different.
_MARKER_COMMON = dict(facecolors='none', linewidths=1.4)


# ------------------------------------
# Source -> style
# ------------------------------------
def style_for(source: str) -> dict:
    """Marker style for a schema `source` string.

    An exact source string wins; otherwise the style comes from the
    SOURCE_PREFIXES vocabulary rather than the whole string. The prefix
    set is prefix-free by contract, so the first hit is the only hit, and
    a provider that revs its data release keeps its marker.
    """
    source = str(source)
    if source in PROVIDER_STYLES:
        return PROVIDER_STYLES[source]
    return PROVIDER_STYLES.get(_token_of(source), DEFAULT_STYLE)


def _token_of(source: str) -> str:
    """Provider token for a source string ('other' when unrecognized)."""
    for token, prefix in SOURCE_PREFIXES.items():
        if str(source).startswith(prefix):
            return token
    return 'other'


# ------------------------------------
# HAP composite discovery and download
# ------------------------------------
def _calib_level(row) -> int | None:
    """calib_level as an int, or None when the cell carries no usable one.

    MAST returns a masked table, and int() on a masked cell raises. Such a
    row simply cannot be identified as the level-3 total product; skipping
    it is the answer, not aborting the whole verb.
    """
    value = row['calib_level']
    if value is None or np.ma.is_masked(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def discover_hap_total(coord: SkyCoord,
                       radius_arcsec: float = 60.0) -> dict | None:
    """Locate the HAP total (detection) product covering the position.

    Overlapping HAP visits can each cover one position, and MAST does not
    specify the order it returns them in, so the candidates are ranked by
    obs_id and the lowest wins: the same field yields the same composite on
    every run. A row whose calib_level is masked is skipped.

    Returns {'color_uri', 'color_file', 'fits_uri', 'fits_file'}, or None
    when no HAP total product covers the position.
    """
    from astroquery.mast import Observations

    try:
        obs = Observations.query_region(coord, radius=radius_arcsec * u.arcsec)
    except Exception as e:
        print(f"  [overlay] MAST query failed: {e}")
        return None

    hst_obs = obs[obs['obs_collection'] == 'HST']
    if len(hst_obs) == 0:
        return None

    candidates = [row for row in hst_obs
                  if str(row['filters']).lower() == 'detection'
                  and _calib_level(row) == 3]
    if not candidates:
        print("  [overlay] no HAP total/detection product at this position")
        return None
    total = min(candidates, key=lambda row: str(row['obs_id']))
    if len(candidates) > 1:
        print(f"  [overlay] {len(candidates)} HAP total products cover this "
              f"position; taking the lowest obs_id")
    print(f"  [overlay] HAP total product: {total['obs_id']}")

    try:
        products = Observations.get_product_list(str(int(total['obsid'])))
    except Exception as e:
        print(f"  [overlay] get_product_list failed: {e}")
        return None

    found: dict[str, str] = {}
    for prod in products:
        fname = str(prod['productFilename'])
        uri = str(prod.get('dataURI', ''))
        if fname.endswith('_drc_color.jpg'):
            found.update(color_uri=uri, color_file=fname)
        elif fname.endswith('_drc.fits') and 'total' in fname:
            found.update(fits_uri=uri, fits_file=fname)

    if 'color_uri' not in found or 'fits_uri' not in found:
        print("  [overlay] HAP total product carries no color JPG "
              "+ detection FITS pair")
        return None
    return found


def download_file(uri: str, filename: str,
                  cache_dir: str | Path) -> Path | None:
    """Download one MAST product, cache-first. None on failure.

    The detection mosaic is ~380 MB; the color JPG is small.
    """
    from astroquery.mast import Observations

    cache_dir = Path(cache_dir)
    cache_path = cache_dir / filename
    if cache_path.exists():
        print(f"  [overlay] cached: {filename}")
        return cache_path

    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [overlay] downloading {filename} ...")
    try:
        Observations.download_file(uri, local_path=str(cache_path))
    except Exception as e:
        print(f"  [overlay] download failed: {e}")
        return None
    if not cache_path.exists():
        return None
    print(f"  [overlay] done ({os.path.getsize(cache_path) / 1e6:.1f} MB)")
    return cache_path


def load_color_image_and_wcs(color_path: str | Path,
                             wcs_path: str | Path,
                             ) -> tuple[np.ndarray, WCS] | None:
    """Load the composite JPG and the WCS of the grid it was drawn on.

    The pixel dimensions must agree: the composite has no WCS of its own,
    so a grid mismatch would place every marker at a plausible-looking
    wrong position. Refuse instead. Returns (rgb, wcs) with rgb already
    flipped to matplotlib's origin='lower' convention.
    """
    from PIL import Image

    rgb = np.array(Image.open(color_path))
    with fits.open(wcs_path) as hdul:
        for hdu in hdul:
            if hdu.data is None or getattr(hdu.data, "ndim", 0) != 2:
                continue
            if hdu.data.shape != rgb.shape[:2]:
                print(f"  [overlay] grid mismatch: WCS image "
                      f"{hdu.data.shape} vs composite {rgb.shape[:2]}; "
                      f"refusing to place markers")
                return None
            # The JPG's origin is top-left, matplotlib's is bottom-left.
            return rgb[::-1, :, :], WCS(hdu.header, naxis=2)
    print(f"  [overlay] no 2D image HDU in {wcs_path}")
    return None


# ------------------------------------
# Cutouts and panel furniture
# ------------------------------------
def rgb_cutout(rgb: np.ndarray, wcs: WCS, coord: SkyCoord,
               half_arcsec: float) -> tuple[np.ndarray, WCS] | None:
    """WCS-aware cutout of an RGB image, or None when off the mosaic."""
    # Cutout2D needs a 2D array, and the slices it computes are the same for
    # every channel, so one channel is enough to derive them.
    try:
        cut = Cutout2D(rgb[:, :, 0], coord, 2 * half_arcsec * u.arcsec,
                       wcs=wcs)
    except Exception as e:
        print(f"  [overlay] cutout failed: {e}")
        return None
    ys, xs = cut.slices_original
    return rgb[ys, xs, :], cut.wcs


def _arcsec_per_pixel(wcs: WCS) -> float:
    scale = wcs.proj_plane_pixel_scales()[0] * 3600
    return float(getattr(scale, "value", scale))


def overlay_markers(ax, wcs: WCS, table: pd.DataFrame,
                    target: SkyCoord) -> None:
    """Target crosshair plus one marker per distinct catalog match.

    The crosshair sits UNDER the markers: a provider that matched exactly
    on target is the one worth seeing, and it would otherwise be the one
    the crosshair hides. Thin arms keep the reference readable anyway.
    """
    tx, ty = wcs.world_to_pixel(target)
    ax.plot(tx, ty, '+', color='red', markersize=14, markeredgewidth=2.0,
            zorder=8)

    drawn = set()
    for _, row in table.iterrows():
        ra, dec = row['match_ra'], row['match_dec']
        if not (np.isfinite(ra) and np.isfinite(dec)):
            continue
        # One marker per position per style: a table carries a row per
        # band, and every band of a catalog matched the same object. Keyed
        # on the style, not the provider, so HAP segment and point still
        # draw separately when their centroids disagree.
        style = style_for(row['source'])
        key = (style['label'], round(float(ra), 8), round(float(dec), 8))
        if key in drawn:
            continue
        drawn.add(key)
        style = {k: v for k, v in style.items() if k != 'label'}
        mx, my = wcs.world_to_pixel(SkyCoord(ra, dec, unit=u.deg))
        ax.scatter(mx, my, zorder=9, **style, **_MARKER_COMMON)


def add_zoom_box(ax, wcs: WCS, center: SkyCoord, half_arcsec: float) -> None:
    """Dashed rectangle marking the zoom panel's footprint."""
    half_pix = half_arcsec / _arcsec_per_pixel(wcs)
    cx, cy = wcs.world_to_pixel(center)
    ax.add_patch(Rectangle((cx - half_pix, cy - half_pix),
                           2 * half_pix, 2 * half_pix, linewidth=1.5,
                           edgecolor='white', facecolor='none',
                           linestyle='--', zorder=8))


def add_scale_bar(ax, wcs: WCS, shape: tuple, bar_arcsec: float) -> None:
    """Scale bar in the lower-left corner."""
    bar_pix = bar_arcsec / _arcsec_per_pixel(wcs)
    ny, nx = shape[:2]
    x0, y0 = nx * 0.05, ny * 0.06
    ax.plot([x0, x0 + bar_pix], [y0, y0], '-', color='white', linewidth=2.5,
            solid_capstyle='butt')
    ax.text(x0 + bar_pix / 2, y0 + ny * 0.03, f'{bar_arcsec:g}"',
            color='white', fontsize=9, ha='center', va='bottom',
            fontweight='bold')


def style_wcs_axes(ax, label_size: int = 9) -> None:
    """Sexagesimal ticks, inward, publication styling."""
    ra_ax, dec_ax = ax.coords['ra'], ax.coords['dec']
    ra_ax.set_major_formatter('hh:mm:ss.s')
    dec_ax.set_major_formatter('dd:mm:ss')
    ra_ax.set_ticklabel(fontsize=label_size)
    dec_ax.set_ticklabel(fontsize=label_size)
    ra_ax.set_ticks_position('b')
    dec_ax.set_ticks_position('l')
    ra_ax.set_axislabel('R.A.', fontsize=label_size + 1)
    dec_ax.set_axislabel('Decl.', fontsize=label_size + 1, minpad=0.8)
    ax.tick_params(axis='both', direction='in', length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)


def legend_handles(sources: set[str]) -> list:
    """Legend entries for the target and every provider actually drawn."""
    handles = [Line2D([0], [0], marker='+', color='red', linestyle='None',
                      markersize=10, markeredgewidth=2, label='target')]
    seen: dict[str, dict] = {}
    for source in sorted(sources):
        style = style_for(source)
        seen.setdefault(style['label'], style)
    for label, style in sorted(seen.items()):
        handles.append(Line2D([0], [0], marker=style['marker'],
                              color=style['edgecolors'], linestyle='None',
                              markersize=8, markerfacecolor='none',
                              markeredgewidth=1.3, label=label))
    return handles


# ------------------------------------
# The figure
# ------------------------------------
def plot_overlay(
        panels: list[dict],
        outpath: str | Path,
        *,
        zoom_arcsec: float,
        context_arcsec: float,
        dpi: int = 200,
) -> Path:
    """Context + zoom panel pair per target, markers on the zoom.

    Parameters
    ----------
    panels : list[dict]
        One entry per target: 'label', 'coord', 'table', and the
        'ctx'/'zoom' (rgb, wcs) pairs from rgb_cutout.
    outpath : str or Path
        Output PNG.
    zoom_arcsec, context_arcsec : float
        Panel half-widths, for the zoom box and titles.
    dpi : int
        Figure resolution. [default: 200]

    Returns
    -------
    outpath : Path
    """
    # Scoped, not global: rcParams is process-wide state, and a caller that
    # plots something else afterwards must not silently inherit this
    # figure's serif fonts and black tick colors.
    style = {
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    }
    with plt.rc_context(style):
        n = len(panels)
        fig = plt.figure(figsize=(5.5 * 2 * n, 6.2))
        fig.patch.set_facecolor('white')
        sources: set[str] = set()
        first_ax = None

        for i, panel in enumerate(panels):
            name = panel['label'].replace('_', ' ').strip().title()
            coord = panel['coord']

            if panel['ctx'] is not None:
                rgb, wcs = panel['ctx']
                ax = fig.add_subplot(1, 2 * n, 2 * i + 1, projection=wcs)
                if first_ax is None:
                    first_ax = ax
                ax.set_facecolor('black')
                ax.imshow(rgb, origin='lower', interpolation='bilinear')
                tx, ty = wcs.world_to_pixel(coord)
                ax.plot(tx, ty, '+', color='red', markersize=12,
                        markeredgewidth=1.5, zorder=10)
                add_zoom_box(ax, wcs, coord, zoom_arcsec)
                add_scale_bar(ax, wcs, rgb.shape, bar_arcsec=5.0)
                style_wcs_axes(ax)
                ax.set_title(f'{name} ({2 * context_arcsec:g}")', fontsize=12,
                             pad=8)

            if panel['zoom'] is not None:
                rgb, wcs = panel['zoom']
                ax = fig.add_subplot(1, 2 * n, 2 * i + 2, projection=wcs)
                ax.set_facecolor('black')
                ax.imshow(rgb, origin='lower', interpolation='bilinear')
                overlay_markers(ax, wcs, panel['table'], coord)
                add_scale_bar(ax, wcs, rgb.shape, bar_arcsec=1.0)
                style_wcs_axes(ax)
                ax.set_title(f'{name} zoom ({2 * zoom_arcsec:g}")',
                             fontsize=12, pad=8)

            sources.update(panel['table']['source'].dropna().unique())

        if first_ax is not None:
            first_ax.legend(handles=legend_handles(sources), loc='upper left',
                            fontsize=8, frameon=True, fancybox=True,
                            facecolor='white', edgecolor='black',
                            framealpha=0.85, borderpad=0.6,
                            handletextpad=0.4, labelspacing=0.35)

        outpath = Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(outpath, dpi=dpi, facecolor='white', bbox_inches='tight')
        plt.close(fig)
    return outpath


# ------------------------------------
# Driver
# ------------------------------------
def build(
        tables: dict[str, pd.DataFrame],
        outpath: str | Path,
        cache_dir: str | Path,
        *,
        zoom_arcsec: float = 5.0,
        context_arcsec: float = 15.0,
        wcs_from: str | Path | None = None,
        dpi: int = 200,
) -> Path | None:
    """Fetch the composite for the field and overlay every table on it.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        {panel label: schema table}. Each table supplies its own target
        position from target_ra/target_dec; the first one's position
        decides which HAP field is fetched.
    outpath : str or Path
        Output PNG.
    cache_dir : str or Path
        Where the composite and detection mosaic are cached.
    zoom_arcsec, context_arcsec : float
        Panel half-widths in arcsec. [default: 5 and 15]
    wcs_from : str or Path, optional
        Local FITS already on the composite's drizzle grid, used instead of
        downloading the ~380 MB detection mosaic for its WCS. Dimension-
        checked either way. A path that is not a readable file refuses:
        falling back to the download is the one outcome the caller asked
        for by name to avoid.
    dpi : int
        Figure resolution. [default: 200]

    Returns
    -------
    figure_path : Path or None
        The written PNG, or None when the field has no HAP composite or
        wcs_from names no file.
    """
    # Checked before anything is queried or downloaded: a mistyped path
    # must cost nothing, and must never silently become the ~380 MB fetch.
    if wcs_from and not Path(wcs_from).is_file():
        print(f"  [overlay] --wcs-from names no file: {wcs_from}; refusing "
              f"to fall back to the detection-mosaic download")
        return None

    targets = []
    for label, table in tables.items():
        if table is None or table.empty:
            continue
        targets.append({
            'label': label,
            'coord': SkyCoord(float(table['target_ra'].iloc[0]),
                              float(table['target_dec'].iloc[0]),
                              unit=u.deg),
            'table': table,
        })
    if not targets:
        print("  [overlay] no non-empty tables")
        return None

    # The composite always comes from MAST; only its WCS can come from a
    # local file, which is the whole point of wcs_from -- the detection
    # mosaic is the expensive half of the pair.
    found = discover_hap_total(targets[0]['coord'])
    if found is None:
        return None
    color_path = download_file(found['color_uri'], found['color_file'],
                               cache_dir)
    wcs_path = (Path(wcs_from) if wcs_from
                else download_file(found['fits_uri'], found['fits_file'],
                                   cache_dir))

    if color_path is None or wcs_path is None:
        print("  [overlay] could not obtain the composite + its WCS")
        return None

    loaded = load_color_image_and_wcs(color_path, wcs_path)
    if loaded is None:
        return None
    rgb, wcs = loaded
    print(f"  [overlay] composite {rgb.shape[1]}x{rgb.shape[0]} px")

    panels = []
    for target in targets:
        ctx = rgb_cutout(rgb, wcs, target['coord'], context_arcsec)
        zoom = rgb_cutout(rgb, wcs, target['coord'], zoom_arcsec)
        if ctx is None and zoom is None:
            print(f"  [overlay] {target['label']}: off the mosaic, skipping")
            continue
        panels.append({**target, 'ctx': ctx, 'zoom': zoom})

    if not panels:
        print("  [overlay] every target fell off the mosaic")
        return None

    return plot_overlay(panels, outpath, zoom_arcsec=zoom_arcsec,
                        context_arcsec=context_arcsec, dpi=dpi)
