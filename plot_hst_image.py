#!/usr/bin/env python3
"""
plot_hst_image.py

HST Photometry Verification Overlay
-------------------------------------

Downloads the pre-made HAP color composite image and the detection FITS
(for WCS) from MAST, then overlays matched photometric source positions
from CSVs produced by phot_coord_search.py.

For each target, produces two panels:
  - Wide context view (~15" half-width) showing surrounding field
  - Zoomed detail view (~5" half-width) with photometry markers

Usage:
    python plot_hst_image.py photometry.csv [more_photometry.csv ...]

Options:
    --zoom-size     <float>   Zoomed panel half-width in arcsec   [default: 5.0]
    --context-size  <float>   Context panel half-width in arcsec  [default: 15.0]
    --cache-dir     <str>     Directory for cached FITS files     [default: ./hst_cache]
    --out           <str>     Output figure filename              [default: phot_overlay.png]
    --dpi           <int>     Figure resolution                   [default: 200]

Notes:
    - First run downloads the HAP color JPG (~6 MB) and detection FITS
      (~380 MB, needed for WCS only); subsequent runs use the local cache.
    - The color image is the pipeline-produced composite from MAST's HAP
      processing, giving a properly stretched astronomical color image.
    - The script discovers the correct HAP total product automatically
      from MAST based on the target position.
"""
from __future__ import annotations

import os
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import astropy.units as u

from astroquery.mast import Observations


# -------------------------------------------------------
# Constants
# -------------------------------------------------------

# Marker styles per catalog source - open symbols for academic look
CATALOG_STYLES = {
    'Legacy_DR10':     dict(marker='s', facecolors='none', s=70, linewidths=1.4,
                            edgecolors='#FF6B35', label='Legacy DR10'),
    'PanSTARRS_DR1':   dict(marker='^', facecolors='none', s=80, linewidths=1.4,
                            edgecolors='#00BFFF', label='Pan-STARRS DR1'),
    'HST_HAP_segment': dict(marker='o', facecolors='none', s=70, linewidths=1.4,
                            edgecolors='#76FF03', label='HST HAP (segment)'),
    'HST_HAP_point':   dict(marker='D', facecolors='none', s=55, linewidths=1.4,
                            edgecolors='#AEEA00', label='HST HAP (point)'),
    'HST_HAP':         dict(marker='o', facecolors='none', s=70, linewidths=1.4,
                            edgecolors='#76FF03', label='HST HAP'),
}
DEFAULT_STYLE = dict(marker='D', facecolors='none', s=55, linewidths=1.2,
                     edgecolors='yellow', label='Other')


# -------------------------------------------------------
# MAST Image Discovery & Download
# -------------------------------------------------------

def discover_hap_total(coord: SkyCoord,
                       radius_arcsec: float = 60.0,
                       ) -> dict | None:
    """
    Find the HAP total (detection) observation at this position and return
    the URIs for the color JPG and detection DRC FITS.

    Returns {'color_uri': ..., 'color_file': ...,
             'fits_uri': ..., 'fits_file': ...} or None.
    """
    try:
        obs = Observations.query_region(coord, radius=radius_arcsec * u.arcsec)
    except Exception as e:
        print(f"  MAST query failed: {e}")
        return None

    hst_obs = obs[obs['obs_collection'] == 'HST']
    if len(hst_obs) == 0:
        return None

    # Find the "total" (detection) observation - filter name is "detection"
    total_obs = None
    for row in hst_obs:
        if (str(row['filters']).lower() == 'detection'
                and int(row['calib_level']) == 3):
            total_obs = row
            break

    if total_obs is None:
        print("  No HAP total/detection observation found.")
        return None

    obsid = str(int(total_obs['obsid']))
    obs_id = str(total_obs['obs_id'])
    print(f"  Found HAP total observation: {obs_id}")

    try:
        products = Observations.get_product_list(obsid)
    except Exception as e:
        print(f"  get_product_list failed: {e}")
        return None

    color_uri = color_file = fits_uri = fits_file = None
    for prod in products:
        fname = str(prod['productFilename'])
        uri   = str(prod.get('dataURI', ''))

        if fname.endswith('_drc_color.jpg'):
            color_uri  = uri
            color_file = fname
        elif fname.endswith('_drc.fits') and 'total' in fname:
            fits_uri  = uri
            fits_file = fname

    if not color_uri or not fits_uri:
        print("  Could not find color JPG or detection FITS in products.")
        return None

    return {
        'color_uri': color_uri, 'color_file': color_file,
        'fits_uri': fits_uri, 'fits_file': fits_file,
    }


def download_file(uri: str, filename: str, cache_dir: str) -> str | None:
    """Download a file from MAST with local caching."""
    cache_path = os.path.join(cache_dir, filename)
    if os.path.exists(cache_path):
        print(f"  Cached: {filename}")
        return cache_path

    os.makedirs(cache_dir, exist_ok=True)
    print(f"  Downloading {filename}...")

    try:
        Observations.download_file(uri, local_path=cache_path)
        if os.path.exists(cache_path):
            size_mb = os.path.getsize(cache_path) / 1e6
            print(f"  Done ({size_mb:.1f} MB)")
            return cache_path
    except Exception as e:
        print(f"  Download failed: {e}")

    return None


def load_color_image_and_wcs(color_path: str, fits_path: str,
                             ) -> tuple[np.ndarray, WCS] | None:
    """
    Load the pre-made color JPG and extract WCS from the matching FITS.

    The HAP pipeline generates the color JPG from the same drizzle grid
    as the detection DRC FITS, so their pixel coordinates are identical.

    Returns (rgb_array, wcs) where rgb_array is (ny, nx, 3) in [0,255].
    """
    # Load color image
    img = Image.open(color_path)
    rgb = np.array(img)  # (height, width, 3)

    # Load WCS from FITS
    with fits.open(fits_path) as hdul:
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                wcs = WCS(hdu.header, naxis=2)

                # Verify dimensions match
                fits_shape = hdu.data.shape  # (ny, nx)
                jpg_shape  = rgb.shape[:2]    # (ny, nx)
                if fits_shape != jpg_shape:
                    print(f"  WARNING: FITS {fits_shape} vs JPG {jpg_shape} "
                          f"- dimensions don't match!")
                    return None

                return rgb, wcs

    return None


# -------------------------------------------------------
# Cutout
# -------------------------------------------------------

def make_rgb_cutout(rgb: np.ndarray, wcs: WCS, coord: SkyCoord,
                    half_arcsec: float) -> tuple[np.ndarray, WCS] | None:
    """
    Extract a WCS-aware cutout from an RGB image.

    Uses Cutout2D on one channel to get the pixel slice, then applies
    the same slice to all 3 channels.
    """
    size = 2 * half_arcsec * u.arcsec
    try:
        # Use the first channel to get the cutout geometry
        cutout = Cutout2D(rgb[:, :, 0], coord, size, wcs=wcs)
        slices = cutout.slices_original

        rgb_cut = rgb[slices[0], slices[1], :]
        return rgb_cut, cutout.wcs
    except Exception as e:
        print(f"  Cutout failed: {e}")
        return None


# -------------------------------------------------------
# Plotting Helpers
# -------------------------------------------------------

def overlay_markers(ax, wcs: WCS, phot_df: pd.DataFrame,
                    target_ra: float, target_dec: float):
    """Overlay target crosshair and catalog match markers."""
    tgt_pix = wcs.world_to_pixel(SkyCoord(target_ra, target_dec, unit=u.deg))
    ax.plot(tgt_pix[0], tgt_pix[1], '+', color='red', markersize=14,
            markeredgewidth=2.0, zorder=10)

    plotted = set()
    for _, row in phot_df.iterrows():
        source = row['source']
        mra, mdec = row['match_ra'], row['match_dec']
        key = (source, round(mra, 8), round(mdec, 8))
        if key in plotted:
            continue
        plotted.add(key)

        style = CATALOG_STYLES.get(source, DEFAULT_STYLE).copy()
        style.pop('label')
        match_pix = wcs.world_to_pixel(SkyCoord(mra, mdec, unit=u.deg))
        ax.scatter(match_pix[0], match_pix[1], zorder=9, **style)


def add_zoom_box(ax, wcs: WCS, center_ra: float, center_dec: float,
                 half_arcsec: float):
    """Draw a dashed rectangle showing the zoom region."""
    pix_scales = wcs.proj_plane_pixel_scales()
    arcsec_per_pix = pix_scales[0] * 3600
    if hasattr(arcsec_per_pix, 'value'):
        arcsec_per_pix = arcsec_per_pix.value

    box_half_pix = half_arcsec / arcsec_per_pix
    cx, cy = wcs.world_to_pixel(
        SkyCoord(center_ra, center_dec, unit=u.deg))

    rect = Rectangle((cx - box_half_pix, cy - box_half_pix),
                      2 * box_half_pix, 2 * box_half_pix,
                      linewidth=1.5, edgecolor='white', facecolor='none',
                      linestyle='--', zorder=8)
    ax.add_patch(rect)


def add_scale_bar(ax, wcs: WCS, rgb_shape: tuple,
                  bar_arcsec: float = 1.0, label: str = '1"'):
    """Add a scale bar to the lower-left corner."""
    pix_scales = wcs.proj_plane_pixel_scales()
    arcsec_per_pix = pix_scales[0] * 3600
    if hasattr(arcsec_per_pix, 'value'):
        arcsec_per_pix = arcsec_per_pix.value
    bar_pix = bar_arcsec / arcsec_per_pix

    ny, nx = rgb_shape[:2]
    x0 = nx * 0.05
    y0 = ny * 0.06
    ax.plot([x0, x0 + bar_pix], [y0, y0], '-', color='white', linewidth=2.5,
            solid_capstyle='butt')
    ax.text(x0 + bar_pix / 2, y0 + ny * 0.03, label,
            color='white', fontsize=9, ha='center', va='bottom',
            fontweight='bold')


def style_wcs_axes(ax, wcs: WCS, label_size=9, show_ra_label=True,
                   show_dec_label=True):
    """Configure a WCSAxes for publication-quality sexagesimal display."""
    ra_ax = ax.coords['ra']
    dec_ax = ax.coords['dec']

    # Sexagesimal formatting
    ra_ax.set_major_formatter('hh:mm:ss.s')
    dec_ax.set_major_formatter('dd:mm:ss')

    ra_ax.set_ticklabel(fontsize=label_size)
    dec_ax.set_ticklabel(fontsize=label_size)

    ra_ax.set_ticks_position('b')
    dec_ax.set_ticks_position('l')

    if show_ra_label:
        ra_ax.set_axislabel('R.A.', fontsize=label_size + 1)
    else:
        ra_ax.set_axislabel('')

    if show_dec_label:
        dec_ax.set_axislabel('Decl.', fontsize=label_size + 1, minpad=0.8)
    else:
        dec_ax.set_axislabel('')

    ax.tick_params(axis='both', direction='in', length=4, width=0.8)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.8)


def build_legend_handles(sources_used: set[str]) -> list:
    """Build legend handles for plotted catalogs."""
    handles = [
        Line2D([0], [0], marker='+', color='red', linestyle='None',
               markersize=10, markeredgewidth=2, label='Target'),
    ]
    for source in sorted(sources_used):
        style = CATALOG_STYLES.get(source, DEFAULT_STYLE)
        handles.append(Line2D(
            [0], [0], marker=style['marker'],
            color=style.get('edgecolors', style.get('color', 'white')),
            linestyle='None', markersize=8,
            markerfacecolor='none',
            markeredgewidth=1.3,
            label=style['label'],
        ))
    return handles


def nice_name(raw_label: str) -> str:
    """Convert a CSV-derived label fragment into a display name."""
    pretty = raw_label.replace('_', ' ').strip().title()
    return pretty or raw_label


# -------------------------------------------------------
# Main
# -------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Overlay photometry positions on HST color images."
    )
    parser.add_argument('csvs', nargs='+',
                        help="Photometry CSV(s) from phot_coord_search.py")
    parser.add_argument('--zoom-size', type=float, default=5.0,
                        help="Zoomed panel half-width in arcsec (default: 5)")
    parser.add_argument('--context-size', type=float, default=15.0,
                        help="Context panel half-width in arcsec (default: 15)")
    parser.add_argument('--cache-dir', type=str, default='./hst_cache',
                        help="FITS cache directory (default: ./hst_cache)")
    parser.add_argument('--out', type=str, default='phot_overlay.png',
                        help="Output filename (default: phot_overlay.png)")
    parser.add_argument('--dpi', type=int, default=200,
                        help="Figure DPI (default: 200)")
    args = parser.parse_args()

    # ---- Load CSVs ----
    targets = []
    for csvfile in args.csvs:
        df = pd.read_csv(csvfile)
        tra  = df['target_ra'].iloc[0]
        tdec = df['target_dec'].iloc[0]
        label = (Path(csvfile).stem
                 .replace('target_photometry_', '')
                 .replace('target_', '')
                 .replace('_photometry', ''))
        targets.append({'label': label, 'ra': tra, 'dec': tdec, 'df': df})

    print(f"Loaded {len(targets)} target(s).")

    # ---- Discover and download the HAP color image ----
    ref_coord = SkyCoord(targets[0]['ra'], targets[0]['dec'], unit=u.deg)
    print("\nSearching MAST for HAP color image...")
    product_info = discover_hap_total(ref_coord)

    if product_info is None:
        print("Could not find HAP color image. Exiting.")
        return

    color_path = download_file(product_info['color_uri'],
                               product_info['color_file'],
                               args.cache_dir)
    fits_path  = download_file(product_info['fits_uri'],
                               product_info['fits_file'],
                               args.cache_dir)

    if color_path is None or fits_path is None:
        print("Download failed. Exiting.")
        return

    # ---- Load image and WCS ----
    result = load_color_image_and_wcs(color_path, fits_path)
    if result is None:
        print("Could not load image. Exiting.")
        return

    full_rgb, full_wcs = result
    print(f"  Loaded color image: {full_rgb.shape}")

    # The JPG has origin at top-left, but matplotlib imshow with
    # origin='lower' expects bottom-left.  Flip vertically.
    full_rgb = full_rgb[::-1, :, :]

    # ---- Build cutouts for each target ----
    panels = []
    for t in targets:
        coord = SkyCoord(t['ra'], t['dec'], unit=u.deg)

        ctx  = make_rgb_cutout(full_rgb, full_wcs, coord, args.context_size)
        zoom = make_rgb_cutout(full_rgb, full_wcs, coord, args.zoom_size)

        if ctx is None and zoom is None:
            print(f"  WARNING: cutouts failed for '{t['label']}', skipping.")
            continue

        panels.append({
            **t,
            'ctx_rgb':  ctx[0] if ctx else None,
            'ctx_wcs':  ctx[1] if ctx else None,
            'zoom_rgb': zoom[0] if zoom else None,
            'zoom_wcs': zoom[1] if zoom else None,
        })

    if not panels:
        print("Nothing to plot.")
        return

    # ---- Global style: serif font, white background ----
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'dejavuserif',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
    })

    # ---- Figure layout with WCS projections ----
    n = len(panels)
    fig = plt.figure(figsize=(5.5 * 2 * n, 6.2))
    fig.patch.set_facecolor('white')

    all_sources = set()
    first_ax = None  # for legend placement

    for i, panel in enumerate(panels):
        name = nice_name(panel['label'])

        # --- Context panel (WCS axes) ---
        if panel['ctx_rgb'] is not None:
            ax_ctx = fig.add_subplot(1, 2 * n, 2 * i + 1,
                                     projection=panel['ctx_wcs'])
            if first_ax is None:
                first_ax = ax_ctx
            ax_ctx.set_facecolor('black')
            ax_ctx.imshow(panel['ctx_rgb'], origin='lower',
                          interpolation='bilinear')

            # Target crosshair + zoom box on context
            tgt_pix = panel['ctx_wcs'].world_to_pixel(
                SkyCoord(panel['ra'], panel['dec'], unit=u.deg))
            ax_ctx.plot(tgt_pix[0], tgt_pix[1], '+', color='red',
                        markersize=12, markeredgewidth=1.5, zorder=10)
            add_zoom_box(ax_ctx, panel['ctx_wcs'],
                         panel['ra'], panel['dec'], args.zoom_size)

            # Scale bar on context
            add_scale_bar(ax_ctx, panel['ctx_wcs'], panel['ctx_rgb'].shape,
                          bar_arcsec=5.0, label='5"')

            style_wcs_axes(ax_ctx, panel['ctx_wcs'],
                           show_ra_label=True, show_dec_label=True)
            ax_ctx.set_title(name, fontsize=12, pad=8)

        # --- Zoom panel (WCS axes) with photometry markers ---
        if panel['zoom_rgb'] is not None:
            ax_zoom = fig.add_subplot(1, 2 * n, 2 * i + 2,
                                      projection=panel['zoom_wcs'])
            ax_zoom.set_facecolor('black')
            ax_zoom.imshow(panel['zoom_rgb'], origin='lower',
                           interpolation='bilinear')

            overlay_markers(ax_zoom, panel['zoom_wcs'], panel['df'],
                            panel['ra'], panel['dec'])

            # Scale bar on zoom
            add_scale_bar(ax_zoom, panel['zoom_wcs'],
                          panel['zoom_rgb'].shape,
                          bar_arcsec=1.0, label='1"')

            style_wcs_axes(ax_zoom, panel['zoom_wcs'],
                           show_ra_label=True, show_dec_label=True)
            ax_zoom.set_title(f'{name} Zoom', fontsize=12, pad=8)

        all_sources.update(panel['df']['source'].unique())

    # Legend - upper-left of the first axes, fancybox
    handles = build_legend_handles(all_sources)
    if first_ax is not None:
        first_ax.legend(handles=handles, loc='upper left',
                        fontsize=8, frameon=True, fancybox=True,
                        facecolor='white', edgecolor='black',
                        framealpha=0.85, borderpad=0.6,
                        handletextpad=0.4, labelspacing=0.35)

    plt.tight_layout()
    plt.savefig(args.out, dpi=args.dpi, facecolor='white',
                bbox_inches='tight')
    print(f"\nSaved: {args.out}")
    plt.close()


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
