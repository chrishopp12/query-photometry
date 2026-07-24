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
from .background import ambient_surface


def find_artifacts(
        raw: np.ndarray,
        good: np.ndarray,
        pred: np.ndarray,
        rr: np.ndarray,
        sigma: float,
        pixscale: float,
        sb: float = 0.0,
) -> tuple[np.ndarray, float]:
    """Mask of catastrophic unclaimed-bright pixels.

    A pixel is an artifact candidate when it sits above the brightness
    floor -- the LARGER of ARTIFACT_SIG sigma and the absolute
    ARTIFACT_SB_MIN (uJy/arcsec^2) -- AND ARTIFACT_RATIO times above
    the catalog scene's claim there. An under-predicted real source
    fails the ratio test; a galaxy outskirt or cD envelope rim fails
    the absolute floor (bright against the noise on a deep stack, but
    far below the sky); a bleed trail is orders of magnitude past all
    of them. Candidates must form a connected region of at least
    ARTIFACT_AREA_MIN arcsec^2 -- smaller leftovers stay with the flood
    channel. The target protects its own pixels the same way every
    source does, through its claim in pred: light beyond the ratio on
    the core is instrument damage, and the coverage gate demotes the
    band rather than measure it.

    Parameters
    ----------
    raw : np.ndarray
        Stamp data (counts, finite everywhere).
    good : np.ndarray
        Usable-pixel map.
    pred : np.ndarray
        The whole catalog scene at catalog amplitudes (counts), target
        system included: every component whose catalog row is trusted
        to claim pixels.
    rr : np.ndarray
        Radius map about the target (arcsec).
    sigma : float
        Global pixel scatter (counts).
    pixscale : float
        Pixel scale (arcsec/px).
    sb : float
        Counts -> uJy/arcsec^2 factor (stamp.sb); sets the absolute
        brightness floor. Zero disables it (the noise floor alone).

    Returns
    -------
    mask, area, flood_area : np.ndarray, float, float
        Boolean artifact mask, its total area, and the part of that
        area added by the skirt flood (arcsec^2).
    """
    outer = good & (rr > recipe.BG_RMIN_AS)
    if outer.sum() < 100:
        outer = good
    if not outer.any():
        return np.zeros_like(good), 0.0, 0.0
    level = float(np.median(raw[outer]))
    resid = raw - level
    # Brightness floor: the LARGER of the depth-relative noise floor
    # and the absolute surface-brightness floor. On a deep stack the
    # absolute floor dominates -- a real bleed is brighter than the
    # sky, not merely brighter than the noise -- so envelope skirts and
    # galaxy outskirts (bright in sigma, faint in uJy/as^2) are spared.
    bright_floor = recipe.ARTIFACT_SIG * sigma
    if sb > 0:
        bright_floor = max(bright_floor, recipe.ARTIFACT_SB_MIN / sb)
    # The detector operates only where the scene is QUIET (claim below
    # the same brightness floor): artifacts live on quiet sky, while a
    # bright real source's core -- a cD cusp above its smooth model, a
    # saturated star above its clipped catalog flux -- is the least
    # quiet place in the field, and disagreement there is model
    # business (shape misfit, the star stage), never instrument damage.
    cand = (good
            & (resid > bright_floor)
            & (resid > recipe.ARTIFACT_RATIO * np.maximum(pred, 0.0))
            & (pred < bright_floor))
    if not cand.any():
        return np.zeros_like(good), 0.0, 0.0
    cand = binary_dilation(cand, iterations=2)
    labels, n_regions = label(cand)
    mask = np.zeros_like(cand)
    px_min = recipe.ARTIFACT_AREA_MIN / pixscale ** 2
    for i in range(1, n_regions + 1):
        region = labels == i
        if region.sum() >= px_min:
            mask |= region

    # Skirt flood: a soft-edged artifact leaves light below the hard
    # threshold that a model would otherwise chase. Seeded ONLY by the
    # hard mask, growth follows contiguous pixels departing from the
    # local ambient surface that the scene does not claim -- claims
    # stop it, so the boundary stays surgical exactly where the field
    # is crowded. Same constants as the neighbor-mask flood: no new
    # knobs.
    flood_area = 0.0
    if mask.any():
        ambient = ambient_surface(raw, good, mask, rr, pixscale)
        if ambient is not None:
            skirt = (good & np.isfinite(ambient)
                     & (raw - ambient > recipe.K_ISO * sigma)
                     & (pred < 0.5 * sigma))
            flood = mask.copy()
            for _ in range(int(round(recipe.FLOOD_MAX_AS / pixscale))):
                grown = binary_dilation(flood) & skirt
                if not (grown & ~flood).any():
                    break
                flood = flood | grown
            new_px = flood & ~mask
            if new_px.any():
                flood_area = float(new_px.sum()) * pixscale ** 2
                mask = flood
    return mask, float(mask.sum()) * pixscale ** 2, flood_area
