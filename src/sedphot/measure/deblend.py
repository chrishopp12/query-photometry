"""
deblend.py

Apportioned Symmetric Deblending
---------------------------------------------------------

Model-free removal of neighbor light before aperture photometry, built
on one assumption only: a galaxy is approximately 180-degree symmetric
about its own center, and the SUM of two offset galaxies never is.

Every source -- the target and each interloper core -- gets a symmetric
template, the pixelwise minimum of the image and its point reflection
through that source's center (the SDSS photo deblender ansatz). Each
pixel's light is then apportioned by template ratio, and the summed
neighbor share is subtracted. Shared light in an overlap zone divides
proportionally instead of being claimed whole, which is what masking
(all target) or sequential min-subtraction (all neighbor) get wrong.

What this handles that masking cannot: neighbor light BELOW any masking
isophote -- wings and low-surface-brightness envelope fringes -- which
integrates to real contamination over an aperture and biases every
background estimate made outside it.

Requirements:
    numpy, scipy

Notes:
    Templates are built on the fine-smoothed map, so the share ratio is
    noise-stable; the share is applied to the unsmoothed pixels. Pixels
    claimed by nobody (below the support threshold) are left untouched:
    ambient background stays in both the aperture and the sky annulus,
    preserving the engine's ambient-cancellation doctrine.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.ndimage import map_coordinates

from .masks import _smooth, interloper_cores, radii_arcsec

# Template support (units of the raw background rms): below this summed
# template level a pixel is unclaimed background and is left alone.
SUPPORT_SIGMA = 0.3

# Per-template significance floor (units of the raw background rms). A
# template is min(image, reflection), and where the reflected side is
# empty sky that min is CLIPPED POSITIVE NOISE (~+0.3 sigma-smooth per
# template): summed over a dozen cores it builds a spurious neighbor
# pedestal on the target's own outskirts and the apportion ratio hands
# the neighbors a percent-level cut of the envelope on even a clean
# field. A template may only claim pixels where it is INDIVIDUALLY
# significant.
TEMPLATE_FLOOR_SIGMA = 0.5

# The most cores worth deblending; beyond this the remainder are faint
# specks whose templates are noise.
MAX_CORES = 12


def _template(fine: np.ndarray, good: np.ndarray, py: float, px: float,
              yy: np.ndarray, xx: np.ndarray,
              floor: float = 0.0) -> np.ndarray:
    """Symmetric template about (py, px): min(image, its reflection).

    Zero wherever the reflected sample falls off the stamp or on blank
    pixels -- symmetry proves nothing there -- and below the
    significance floor.
    """
    coords = np.array([2.0 * py - yy, 2.0 * px - xx])
    reflected = map_coordinates(fine, coords, order=1, mode='constant',
                                cval=0.0)
    ok = map_coordinates(good.astype(float), coords, order=1,
                         mode='constant', cval=0.0) > 0.99
    template = np.where(ok, np.clip(np.minimum(fine, reflected), 0.0, None),
                        0.0)
    if floor > 0.0:
        template[template <= floor] = 0.0
    return template


def neighbor_templates(
        stamp: np.ndarray,
        threshold_std: float,
        cx: float,
        cy: float,
        pixscale: float,
        *,
        centers: list[tuple[float, float]] | None = None,
        protect_radius: float = 4.0,
        seeing_arcsec: float = 1.0,
        nodata: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Contained symmetric template of every interloper, brightest first.

    Each array is one neighbor's floored, containment-restricted
    template on this stamp's grid -- the data-driven MODEL of that
    neighbor. Derived on a deep reference band, these carry the shape of
    each neighbor's light down into its sub-threshold wings, which a
    shallow band cannot see but can still fit an amplitude for.
    """
    if centers is None:
        centers = [(py, px) for _, py, px in interloper_cores(
            stamp, threshold_std, cx, cy, pixscale,
            seeing_arcsec=seeing_arcsec, nodata=nodata)]
    keep = []
    for py, px in centers[:MAX_CORES]:
        if (0 <= py < stamp.shape[0] and 0 <= px < stamp.shape[1]
                and np.hypot(px - cx, py - cy) * pixscale > protect_radius):
            keep.append((py, px))
    if not keep:
        return []
    good = np.isfinite(stamp) if nodata is None else (np.isfinite(stamp) & ~nodata)
    fine = _smooth(np.where(good, np.nan_to_num(stamp), 0.0), 1.5)
    yy, xx = np.indices(stamp.shape)
    floor = TEMPLATE_FLOOR_SIGMA * threshold_std
    templates = []
    for py, px in keep:
        template = _template(fine, good, py, px, yy, xx, floor=floor)
        labels, _ = ndimage.label(template > 0)
        own = labels[int(np.clip(round(py), 0, stamp.shape[0] - 1)),
                     int(np.clip(round(px), 0, stamp.shape[1] - 1))]
        if own == 0:
            continue
        templates.append(np.where(labels == own, template, 0.0))
    return templates


def target_template(
        stamp: np.ndarray,
        cx: float,
        cy: float,
        *,
        nodata: np.ndarray | None = None,
) -> np.ndarray:
    """The target's own defending template on this stamp (no floor)."""
    good = np.isfinite(stamp) if nodata is None else (np.isfinite(stamp) & ~nodata)
    fine = _smooth(np.where(good, np.nan_to_num(stamp), 0.0), 1.5)
    yy, xx = np.indices(stamp.shape)
    return _template(fine, good, cy, cx, yy, xx)


def reference_component_templates(
        stamp: np.ndarray,
        threshold_std: float,
        cx: float,
        cy: float,
        pixscale: float,
        *,
        centers: list[tuple[float, float]] | None = None,
        protect_radius: float = 4.0,
        seeing_arcsec: float = 1.0,
        nodata: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Mutually-cleaned component templates for the fitted-subtraction path.

    A min-template about one center contains the OTHER sources'
    symmetric-about-that-center parts -- an extended target's wing sits
    at both a neighbor-region pixel and its reflection, so a raw
    neighbor template steals it, and fitted subtraction (unlike
    per-pixel apportioning, where the target defends in the ratio)
    removes the stolen wing wholesale. One mutual round on the deep
    reference band cleans both sides: neighbor templates are built on
    the target-symmetric-subtracted image, the target's on the
    neighbor-subtracted one.
    """
    target_0 = target_template(stamp, cx, cy, nodata=nodata)
    templates = neighbor_templates(
        stamp - target_0, threshold_std, cx, cy, pixscale, centers=centers,
        protect_radius=protect_radius, seeing_arcsec=seeing_arcsec,
        nodata=nodata)
    cleaned = stamp.copy()
    for t in templates:
        cleaned = cleaned - t
    return templates, target_template(cleaned, cx, cy, nodata=nodata)


def reproject_template(
        template: np.ndarray,
        src_wcs,
        dst_wcs,
        dst_shape: tuple,
) -> np.ndarray:
    """Bilinear reprojection of a template onto another band's grid."""
    yy, xx = np.indices(dst_shape)
    sx, sy = src_wcs.world_to_pixel(dst_wcs.pixel_to_world(xx, yy))
    return map_coordinates(template, np.array([sy, sx]), order=1,
                           mode='constant', cval=0.0)


def subtract_fitted_templates(
        stamp: np.ndarray,
        templates: list[np.ndarray],
        t_target: np.ndarray,
        threshold_std: float,
        *,
        nodata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit one amplitude per reference template and subtract the models.

    Forced photometry of the CONTAMINANTS: the shape of each neighbor is
    fixed (deep reference band); this band contributes one linear
    amplitude per component. The solve is SIMULTANEOUS and includes the
    target's own template as a component -- component overlap is carried
    by the off-diagonal of the normal matrix, so target light inside a
    neighbor's support informs the target's amplitude instead of
    biasing the neighbor's (excluding
    target-dominated pixels instead leaves the blend zone to
    contaminate the fit). Only the NEIGHBOR components are subtracted.
    The fit absorbs zeropoint, pixel area, and color differences per
    component; a neighbor absent in this band fits ~0 and subtracts
    nothing.

    Returns
    -------
    residual, share, n_used : np.ndarray, np.ndarray, int
    """
    good = np.isfinite(stamp) if nodata is None else (np.isfinite(stamp) & ~nodata)
    templates = [t for t in templates if (t > 0).any()]
    if not templates:
        return stamp.copy(), np.zeros_like(stamp), 0
    support = t_target > 0
    for t in templates:
        support = support | (t > 0)
    fit_at = support & good
    if fit_at.sum() < 10 * (len(templates) + 1):
        return stamp.copy(), np.zeros_like(stamp), 0
    design = np.column_stack([t_target[fit_at]]
                             + [t[fit_at] for t in templates])
    amps, *_ = np.linalg.lstsq(design, np.nan_to_num(stamp[fit_at]),
                               rcond=None)
    amps = np.clip(amps, 0.0, None)
    share = np.zeros_like(stamp)
    for amp, t in zip(amps[1:], templates):
        share += amp * t
    if nodata is not None:
        share[nodata] = 0.0
    return stamp - share, share, int((amps[1:] > 0).sum())


def apportion_neighbors(
        stamp: np.ndarray,
        threshold_std: float,
        cx: float,
        cy: float,
        pixscale: float,
        *,
        centers: list[tuple[float, float]] | None = None,
        protect_radius: float = 4.0,
        seeing_arcsec: float = 1.0,
        nodata: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Subtract every interloper's apportioned share from the stamp.

    Parameters
    ----------
    stamp : np.ndarray
        Sky-subtracted stamp (a scalar sky estimate suffices; the
        templates only need the source light roughly zeroed).
    threshold_std : float
        Background rms; sets template support and core detection.
    cx, cy : float
        Stamp-pixel center of the target.
    pixscale : float
        Arcsec per pixel.
    centers : list of (py, px), optional
        Deblend centers in THIS stamp's pixel coordinates -- pass the
        reference band's cores transformed by WCS so every band removes
        the same physical sources; detected on this stamp when None.
    protect_radius : float
        Cores inside this radius (arcsec) are not deblended, matching
        the mask exemption: a same-protect companion stays in the flux
        with only the QA metrics to flag it.
    seeing_arcsec : float
        Band PSF FWHM, for core detection when centers is None.
    nodata : np.ndarray (bool), optional
        Off-footprint / blank pixels.

    Returns
    -------
    residual, share, n_cores : np.ndarray, np.ndarray, int
        The deblended stamp, the subtracted neighbor-share map, and how
        many cores were used.
    """
    # Containment lives in neighbor_templates: min(image, reflection) is
    # a model of THIS source only where this source actually is, so each
    # template is restricted to its own connected support patch. A true
    # blend stays connected into the overlap (a legitimate claim); a
    # coincidence across the stamp does not.
    templates = neighbor_templates(
        stamp, threshold_std, cx, cy, pixscale, centers=centers,
        protect_radius=protect_radius, seeing_arcsec=seeing_arcsec,
        nodata=nodata)
    if not templates:
        return stamp.copy(), np.zeros_like(stamp), 0

    good = np.isfinite(stamp) if nodata is None else (np.isfinite(stamp) & ~nodata)
    fine = _smooth(np.where(good, np.nan_to_num(stamp), 0.0), 1.5)
    yy, xx = np.indices(stamp.shape)
    # The target's template carries no floor and no containment: its job
    # is to DEFEND the target's light in the ratio, and zeroing its
    # faint outskirts would hand them to any overlapping neighbor.
    t_target = _template(fine, good, cy, cx, yy, xx)
    t_neighbors = np.zeros_like(fine)
    for template in templates:
        t_neighbors += template

    total = t_target + t_neighbors
    claimed = total > SUPPORT_SIGMA * threshold_std
    share = np.zeros_like(stamp)
    share[claimed] = (np.nan_to_num(stamp[claimed])
                      * (t_neighbors[claimed] / total[claimed]))
    if nodata is not None:
        share[nodata] = 0.0
    return stamp - share, share, len(templates)
