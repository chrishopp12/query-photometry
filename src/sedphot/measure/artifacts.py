"""
artifacts.py

Catastrophic-Pixel Artifact Mask
---------------------------------------------------------
Bleed trails, satellite streaks, and their kin: pixels far beyond both
the sky noise AND the catalog scene's own claim leave the usable-pixel
map exactly like nodata -- masked, never fit. The catalog cannot vouch
for light it never modeled, and a shape solve pointed at a bleed can
only rail its parameters trying to become one.

Only the catastrophic regime lives here. Broad low-surface-brightness
structure (scattered-light arcs, stacking bands at the noise scale) is
invisible to any per-pixel threshold and belongs to the background
plane's bin rejection and the far-field witness.

Requirements:
    numpy, scipy

Notes:
    Detection runs per band on its own pixels and needs a rendered
    catalog scene to define "unclaimed" -- a blind scene (no catalog)
    skips it. Deeply negative counterparts (dead columns, holes) are
    already nodata at stamp preparation.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, label

from . import recipe


def find_artifacts(
        raw: np.ndarray,
        good: np.ndarray,
        pred: np.ndarray,
        protect: np.ndarray,
        rr: np.ndarray,
        sigma: float,
        pixscale: float,
) -> tuple[np.ndarray, float]:
    """Mask of catastrophic unclaimed-bright pixels.

    A pixel is an artifact candidate when it sits ARTIFACT_SIG sigma
    above the outer-stamp level AND ARTIFACT_RATIO times above the
    catalog scene's claim there: an under-predicted real source fails
    the ratio test, a bleed trail is orders of magnitude past both.
    Candidates must form a connected region of at least
    ARTIFACT_AREA_MIN arcsec^2 -- smaller leftovers stay with the flood
    channel. Pixels where the target system's own render exceeds sigma
    are never eligible.

    Parameters
    ----------
    raw : np.ndarray
        Stamp data (counts, finite everywhere).
    good : np.ndarray
        Usable-pixel map.
    pred : np.ndarray
        Catalog scene at catalog amplitudes (counts): every component
        base whose catalog row is trusted to claim pixels.
    protect : np.ndarray
        The target system's own rendered light (counts).
    rr : np.ndarray
        Radius map about the target (arcsec).
    sigma : float
        Global pixel scatter (counts).
    pixscale : float
        Pixel scale (arcsec/px).

    Returns
    -------
    mask, area : np.ndarray, float
        Boolean artifact mask and its total area (arcsec^2).
    """
    outer = good & (rr > recipe.BG_RMIN_AS)
    if outer.sum() < 100:
        outer = good
    if not outer.any():
        return np.zeros_like(good), 0.0
    level = float(np.median(raw[outer]))
    resid = raw - level
    cand = (good
            & (resid > recipe.ARTIFACT_SIG * sigma)
            & (resid > recipe.ARTIFACT_RATIO * np.maximum(pred, 0.0))
            & ~(protect > sigma))
    if not cand.any():
        return np.zeros_like(good), 0.0
    cand = binary_dilation(cand, iterations=2)
    labels, n_regions = label(cand)
    mask = np.zeros_like(cand)
    px_min = recipe.ARTIFACT_AREA_MIN / pixscale ** 2
    for i in range(1, n_regions + 1):
        region = labels == i
        if region.sum() >= px_min:
            mask |= region
    return mask, float(mask.sum()) * pixscale ** 2
