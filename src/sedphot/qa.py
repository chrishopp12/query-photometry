"""
qa.py

QA and SED Figures
---------------------------------------------------------

The diagnostic figures every image extraction writes, and the combined SED
plot. Shared conventions across the figures: asinh/ZScale grayscale stamps,
tab:blue aperture markings, tab:green mask contours, wavelength-ordered point colors.

Data products:
    QA/<inst>_<band>.png          per-band scene panels: data | fitted scene |
                                  residual | masked + filled | curve of growth
    QA/growth_curves.png          enclosed flux vs radius, all bands
    <label>_sed.png               combined SED (catalog + measured points)

Requirements:
    numpy, pandas, matplotlib, astropy
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

# Headless backend: figures are only ever written to disk.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.visualization import AsinhStretch, ImageNormalize, ZScaleInterval
from matplotlib.patches import Circle

from .bands import wave_um

# ------------------------------------
# Constants
# ------------------------------------
# Line style per instrument in the growth-curve overlay.
INSTRUMENT_STYLE = {"Legacy": "-", "SDSS": "--", "CFHT": ":", "PS1": "-.",
                    "PanSTARRS": "-.", "HST": "-"}


# ------------------------------------
# Helpers
# ------------------------------------
def _wave_color(wave: float) -> tuple:
    """Wavelength -> color, blue at 0.15 um through red at 25 um (log scale)."""
    if not np.isfinite(wave):
        return (0.3, 0.3, 0.3, 1.0)
    log_position = (np.log10(wave) - np.log10(0.15)) / (np.log10(25.0) - np.log10(0.15))
    return plt.cm.turbo(float(np.clip(log_position, 0.0, 1.0)))


# ------------------------------------
# Per-band QA
# ------------------------------------
def qa_scene_figure(measurement: dict, out_dir: str | Path) -> Path:
    """Data | fitted scene | residual | masked + filled | curve of growth.

    One row per band: the star-subtracted data, the fitted scene
    (components + background), their residual, the measurement image
    (masked pixels twin-filled), and the curve of growth against the
    fitted target model's own curve.
    """
    image = measurement['image']
    scene = measurement['scene']
    filled = measurement['filled']
    mask = measurement['mask']
    good = measurement['good']
    witness = measurement['witness']
    cx, cy = measurement['cx'], measurement['cy']
    pixscale = measurement['pixscale']
    aperture = measurement['aperture_arcsec']

    fig, axes = plt.subplots(1, 5, figsize=(21.5, 4.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.3]))
    shown = np.where(good, image, np.nan)
    norm = ImageNormalize(shown, interval=ZScaleInterval(),
                          stretch=AsinhStretch())
    gray = plt.cm.gray.copy()
    gray.set_bad("0.15")

    panels = [
        (rf"data (bg {witness['bg_sb']:+.3f} $\mu$Jy/as$^2$, "
         rf"tilt {witness['bg_tilt_sb']:.3f})", shown),
        ("fitted scene + background", scene),
        ("residual", np.where(good, image - scene, np.nan)),
        (f"masked ({witness['maskfrac_ap']:.0%} of aperture) + filled",
         filled),
    ]
    window = float(measurement['rgrid'].max() + 8.0) / pixscale
    for i, (title, img) in enumerate(panels):
        ax = axes[i]
        ax.imshow(img, origin="lower", cmap=gray, norm=norm)
        ax.set_xlim(cx - window, cx + window)
        ax.set_ylim(cy - window, cy + window)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        ax.add_patch(Circle((cx, cy), aperture / pixscale, fill=False,
                            ec="tab:blue", lw=1.0))
        if i in (2, 3) and mask.any():
            ax.contour(mask, levels=[0.5], colors="tab:green",
                       linewidths=0.6)

    ax = axes[4]
    ax.plot(measurement['rgrid'], measurement['enclosed_ujy'], "o-",
            color="C3", ms=3, lw=1.4, label="CoG (data - bg)")
    ax.plot(measurement['rgrid'], measurement['model_cog'], "k--", lw=1.4,
            label="fitted target model")
    ax.axvline(aperture, color="tab:blue", lw=1.0)
    conv = witness['r_conv_as']
    if conv > 0:
        ax.axvline(conv, color="0.55", lw=0.9, ls=":")
    ax.set_xlabel("aperture radius (arcsec)")
    ax.set_ylabel(r"enclosed flux ($\mu$Jy)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    conv_label = f'conv@{conv:.0f}"' if conv > 0 else 'no plateau'
    ax.set_title(
        rf'{measurement["flux_ujy"]:.1f} $\pm$ '
        rf'{measurement["flux_err_ujy"]:.1f} $\mu$Jy ({aperture:g}", '
        f'{measurement["err_model"]})  excess '
        f'{witness["excess_growth_uJy"]:+.1f} '
        f'(own {witness["model_own_growth_uJy"]:+.1f})  {conv_label}',
        fontsize=9)

    fig.suptitle(f"{measurement['instrument']} {measurement['band']} -- "
                 f"scene fit", fontsize=12)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{measurement['instrument']}_{measurement['band']}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_growth_curves(measurements: list[dict], out_dir: str | Path) -> Path:
    """Enclosed flux vs radius for every measured band on one figure."""
    fig, ax = plt.subplots(figsize=(11, 7))
    for m in measurements:
        style = INSTRUMENT_STYLE.get(m['instrument'], "-")
        ax.plot(m['rgrid'], m['enclosed_ujy'], style,
                color=_wave_color(m['wave_um']), lw=1.6, alpha=0.85,
                label=f"{m['instrument']} {m['band']}")
    if measurements:
        ax.axvline(measurements[0]['aperture_arcsec'], color="tab:blue",
                   lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel("aperture radius (arcsec)")
    ax.set_ylabel(r"enclosed flux ($\mu$Jy)")
    ax.set_title("Curve of growth, all measured bands")
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    ax.grid(alpha=0.25, which="both")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "growth_curves.png"
    fig.tight_layout()
    fig.savefig(out, dpi=135)
    plt.close(fig)
    return out


# ------------------------------------
# Combined SED
# ------------------------------------
def plot_sed(
        frames: dict[str, pd.DataFrame],
        outpath: str | Path,
        *,
        title: str = "",
) -> Path:
    """Combined SED: flux vs wavelength for every table given.

    Parameters
    ----------
    frames : dict[str, pd.DataFrame]
        {legend label: schema table}; typically {'catalog': ..., 'measured': ...}.
    outpath : str or Path
        Output PNG.
    title : str
        Figure title (the target label).

    Returns
    -------
    outpath : Path
    """
    fig, ax = plt.subplots(figsize=(10, 6.5))
    markers = {"catalog": "o", "measured": "s"}
    plotted = 0
    for label, df in frames.items():
        if df is None or df.empty:
            continue
        waves = np.array([wave_um(b) for b in df['band']])
        flux = df['flux_uJy'].to_numpy(dtype=float)
        err = df['flux_err_uJy'].to_numpy(dtype=float)
        ok = np.isfinite(waves) & np.isfinite(flux) & (flux > 0)
        dropped = [str(b) for b, keep in zip(df['band'], ok) if not keep]
        if dropped:
            print(f"  [sed] not plotted ({label}; no wavelength or flux <= 0): "
                  f"{', '.join(dropped)}")
        for i in np.where(ok)[0]:
            color = _wave_color(waves[i])
            ax.errorbar(waves[i], flux[i], yerr=err[i] if np.isfinite(err[i]) else None,
                        fmt=markers.get(label, "D"), color=color, ms=6,
                        mec="k", mew=0.4, elinewidth=1.0, capsize=2)
            plotted += 1
    # Legend proxies: one entry per table, shape only.
    for label in frames:
        if frames[label] is not None and not frames[label].empty:
            ax.plot([], [], markers.get(label, "D"), color="0.4", mec="k", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"wavelength ($\mu$m)")
    ax.set_ylabel(r"flux ($\mu$Jy)")
    ax.set_title(title or "SED")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25, which="both")
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=140, bbox_inches="tight")
    plt.close(fig)
    if plotted == 0:
        print("  [sed] nothing plottable; wrote an empty frame")
    return outpath
