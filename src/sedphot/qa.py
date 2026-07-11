"""
qa.py

QA and SED Figures
---------------------------------------------------------

The diagnostic figures every image extraction writes, and the combined SED
plot. Shared conventions across the figures: asinh/ZScale grayscale stamps,
cyan aperture markings, gold sky annulus, wavelength-ordered point colors.

Data products:
    QA/<inst>_<band>.png          per-band: cutout | masked + regions | growth curve
    QA/<inst>_<band>_sersic.png   forced mode: data | model | residual | model growth
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
def qa_band_figure(measurement: dict, out_dir: str | Path) -> Path:
    """Cutout | masked stamp + extraction regions | curve of growth."""
    stamp = measurement['stamp']
    mask = measurement['mask']
    cx, cy = measurement['cx'], measurement['cy']
    pixscale = measurement['pixscale']
    aperture = measurement['aperture_arcsec']
    sky_in, sky_out = measurement['sky_in'], measurement['sky_out']

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    norm = ImageNormalize(stamp, interval=ZScaleInterval(), stretch=AsinhStretch())
    gray = plt.cm.gray.copy()
    gray.set_bad("0.15")

    nodata = measurement.get('nodata')
    n_deblended = measurement.get('n_deblended', 0)
    raw = measurement.get('stamp_raw')
    left = raw if (raw is not None and n_deblended) else stamp
    shown = left if nodata is None else np.where(nodata, np.nan, left)
    axes[0].imshow(shown, origin="lower", cmap=gray, norm=norm)
    axes[0].set_title("cutout (pre-deblend)" if n_deblended
                      else "cutout (sky-subtracted)", fontsize=10)

    hidden = mask if nodata is None else (mask | nodata)
    axes[1].imshow(np.where(hidden, np.nan, stamp), origin="lower", cmap=gray,
                   norm=norm)
    # The two invisible exclusion sets, tinted: the sky fit's source mask
    # (gold) and the diagnostic curve's outer fill (orange). Neither
    # touches the aperture flux; showing them keeps the panel honest about
    # what the sky and the curve actually saw.
    from matplotlib.colors import ListedColormap
    rr_px = np.hypot(*np.meshgrid(np.arange(stamp.shape[1]) - cx,
                                  np.arange(stamp.shape[0]) - cy)) * pixscale
    sky_excluded = measurement.get('annulus_srcmask')
    if sky_excluded is not None:
        show = sky_excluded & (rr_px > sky_in) & (rr_px < sky_out) & ~hidden
        axes[1].imshow(np.where(show, 1.0, np.nan), origin="lower",
                       cmap=ListedColormap(["#eda100"]), alpha=0.4,
                       interpolation="nearest")
    outer_fill = measurement.get('outer_fill')
    if outer_fill is not None:
        show = outer_fill & ~hidden & (rr_px >= aperture) & (rr_px <= sky_in)
        axes[1].imshow(np.where(show, 1.0, np.nan), origin="lower",
                       cmap=ListedColormap(["#eb6834"]), alpha=0.4,
                       interpolation="nearest")
    axes[1].add_patch(Circle((cx, cy), aperture / pixscale, fill=False,
                             ec="cyan", lw=1.2))
    for radius in (sky_in, sky_out):
        axes[1].add_patch(Circle((cx, cy), radius / pixscale, fill=False,
                                 ec="gold", lw=0.9, ls=(0, (4, 3))))
    mask_title = (f"{measurement['mask_mode']} mask (dark) | "
                  f"curve fill (orange) | sky-excluded (gold)")
    if n_deblended:
        mask_title = f"deblended ({n_deblended} nbr) | " + mask_title
    coverage = measurement.get('aperture_coverage')
    if coverage is not None and coverage < 1.0:
        mask_title += f" | coverage {coverage:.2f}"
    axes[1].set_title(mask_title, fontsize=9)
    for ax in axes[:2]:
        window = (sky_out + 6) / pixscale
        ax.set_xlim(cx - window, cx + window)
        ax.set_ylim(cy - window, cy + window)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[2].plot(measurement['rgrid'], measurement['enclosed_ujy'], "o-",
                 color="0.25", ms=3, lw=1.2)
    axes[2].axvline(aperture, color="cyan", lw=1.2)
    axes[2].axvspan(sky_in, sky_out, color="gold", alpha=0.15)
    conv = measurement.get('cog_conv_arcsec')
    if conv is not None and np.isfinite(conv):
        axes[2].axvline(conv, color="0.55", lw=0.9, ls=":")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("aperture radius (arcsec)")
    axes[2].set_ylabel(r"enclosed flux ($\mu$Jy)")
    axes[2].grid(alpha=0.25, which="both")
    parts = [
        rf"{measurement['flux_ujy']:.1f} $\pm$ "
        rf"{measurement['flux_err_ujy']:.1f} $\mu$Jy ({aperture:g}\")",
        measurement['err_model']]
    slope = measurement.get('cog_slope')
    if slope is not None and np.isfinite(slope):
        parts.append(f"slope {slope:+.3f}")
    if conv is not None:
        step = measurement.get('cog_step')
        if np.isfinite(conv) and step is not None and np.isfinite(step):
            parts.append(rf"step {step:+.2f} past {conv:g}\"")
        elif not np.isfinite(conv):
            end = measurement.get('cog_end_slope')
            parts.append(f"grow {end:+.1%}/as" if end is not None
                         and np.isfinite(end) else "not converged")
    # The wide-range fit decomposes the curve into flux + uniform
    # pedestal: report the pedestal's share OF the aperture flux (the
    # honest "how much uniform background is in this number"), never
    # just the fit residual -- a curve that fits well with b != 0 is
    # pedestal-laden, not flat.
    ped = measurement.get('cog_pedestal')
    flux = measurement.get('flux_ujy')
    if ped is not None and np.isfinite(ped) and flux:
        parts.append(
            f"ped {np.pi * ped * aperture ** 2 / abs(flux):+.1%} in ap")
    rms = measurement.get('cog_fit_rms')
    if rms is not None and np.isfinite(rms):
        parts.append(f"resid {rms:.1%}")
    axes[2].set_title(" | ".join(parts), fontsize=9)

    fig.suptitle(f"{measurement['instrument']} {measurement['band']}", fontsize=12)
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
        ax.axvline(measurements[0]['aperture_arcsec'], color="cyan", lw=1.2)
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


def qa_forced_figure(measurement: dict, out_dir: str | Path) -> Path:
    """Masked data | forced-Sersic model | residual, plus the model growth curve."""
    stamp = measurement['stamp']
    model = measurement['model']
    mask = measurement['mask']
    cx, cy = measurement['cx'], measurement['cy']
    pixscale = measurement['pixscale']
    sky_in, sky_out = measurement['sky_in'], measurement['sky_out']

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2),
                             gridspec_kw=dict(width_ratios=[1, 1, 1, 1.2]))
    norm = ImageNormalize(stamp, interval=ZScaleInterval(), stretch=AsinhStretch())
    gray = plt.cm.gray.copy()
    gray.set_bad("0.15")

    axes[0].imshow(np.where(mask, np.nan, stamp), origin="lower", cmap=gray, norm=norm)
    axes[0].set_title("data (masked)", fontsize=10)
    axes[1].imshow(model, origin="lower", cmap=gray, norm=norm)
    axes[1].set_title("forced-Sersic model", fontsize=10)
    residual = stamp - model
    axes[2].imshow(np.where(mask, np.nan, residual), origin="lower",
                   cmap="RdBu_r",
                   vmin=-5 * measurement['sky_std_ujy'] / measurement['cf'],
                   vmax=5 * measurement['sky_std_ujy'] / measurement['cf'])
    axes[2].set_title(rf"residual ($\chi^2_\nu$ = {measurement['redchi2']:.2f})",
                      fontsize=10)
    window = (sky_out + 6) / pixscale
    for ax in axes[:3]:
        ax.set_xlim(cx - window, cx + window)
        ax.set_ylim(cy - window, cy + window)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[3].plot(measurement['rgrid'], measurement['enclosed_ujy'], "o-",
                 color="0.25", ms=3, lw=1.2)
    axes[3].axhline(measurement['flux_ujy'], color="cyan", lw=1.0, ls="--")
    axes[3].set_xlabel("radius (arcsec)")
    axes[3].set_ylabel(r"enclosed model flux ($\mu$Jy)")
    axes[3].grid(alpha=0.25)
    axes[3].set_title(
        rf"{measurement['flux_ujy']:.1f} $\pm$ {measurement['flux_err_ujy']:.1f} "
        rf"$\mu$Jy total", fontsize=10)

    shape = measurement['shape_sky']
    fig.suptitle(
        f"{measurement['instrument']} {measurement['band']}  |  forced Sersic: "
        rf"n={shape['n']:.2f}, $r_e$={shape['reff_arcsec']:.2f}\", "
        rf"ellip={shape['ellip']:.2f}, PA={shape['pa_deg']:.1f}$^\circ$",
        fontsize=11)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{measurement['instrument']}_{measurement['band']}_sersic.png"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
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
