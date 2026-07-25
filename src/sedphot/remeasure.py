"""
remeasure.py

Re-report band fluxes from a stored fit
---------------------------------------------------------
Recompute a galaxy's band fluxes from the IMMUTABLE per-galaxy provenance
sidecar, with no re-fetch and no re-fit. The fit already stored the target
model's curve of growth (PSF-convolved, circular apertures, arcsec radii), so
a different aperture -- or the integrated model total -- is an interpolation on
values already on disk. Because the fitted model IS the deblended target, this
needs neither the images nor the scene, and the provenance is git_rev-pinned,
so a re-report is reproducible even after other galaxies rewrite the registry.

This is the model (Sersic/Nuker) pathway of --remeasure. The empirical-aperture
pathway, which rebuilds the cleaned scene, is separate.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# uJy AB zeropoint: mag = 23.9 - 2.5 log10(f / uJy).
UJY_AB_ZP = 23.9


def model_flux_within(aperture_arcsec: float | None, rgrid, cog,
                      total: float) -> float:
    """Fitted-model flux inside a circular aperture, from the stored COG.

    rgrid/cog are the model curve of growth (arcsec radii, enclosed uJy). The
    origin is pinned at (0, 0); past the last grid radius the model has
    converged, so the total is returned. aperture_arcsec None or <= 0 requests
    the integrated model total.
    """
    if aperture_arcsec is None or aperture_arcsec <= 0:
        return float(total)
    rg = np.asarray(rgrid, dtype=float)
    cg = np.asarray(cog, dtype=float)
    if rg.size == 0:
        return float(total)
    if aperture_arcsec >= rg[-1]:
        return float(total)
    xs = np.concatenate(([0.0], rg))
    ys = np.concatenate(([0.0], cg))
    return float(np.interp(aperture_arcsec, xs, ys))


def remeasure_sersic(provenance_path: str | Path,
                     aperture_arcsec: float | None = None) -> pd.DataFrame:
    """Per-band fitted-model fluxes at aperture_arcsec (None/<=0 = integrated).

    Reads the provenance sidecar and returns a DataFrame (band, flux_uJy,
    mag_AB, aperture_as, mode, source). Bands whose provenance lacks a stored
    model curve of growth (a demoted or unmeasured band) are skipped.
    """
    prov = json.loads(Path(provenance_path).read_text())
    rev = prov.get('git_rev', '?')
    integrated = aperture_arcsec is None or aperture_arcsec <= 0
    rows = []
    for band, b in (prov.get('per_band') or {}).items():
        fs = b.get('fit_state') or {}
        rgrid, cog = fs.get('rgrid'), fs.get('model_cog_uJy')
        total = b.get('target_model_uJy')
        if not rgrid or not cog or total is None:
            continue
        flux = model_flux_within(aperture_arcsec, rgrid, cog, total)
        rows.append(dict(
            band=band,
            flux_uJy=round(flux, 4),
            mag_AB=(round(UJY_AB_ZP - 2.5 * np.log10(flux), 4)
                    if flux > 0 else float('nan')),
            aperture_as=(float('inf') if integrated else float(aperture_arcsec)),
            mode='sersic',
            source=f"sersic_model_remeasure:{rev}"))
    return pd.DataFrame(rows, columns=['band', 'flux_uJy', 'mag_AB',
                                       'aperture_as', 'mode', 'source'])
