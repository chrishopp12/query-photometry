"""
aperture.py

Stage 7: Mask, Fill, Curve of Growth, Witnesses
---------------------------------------------------------
The measurement side of the scene engine. The fitted neighbors and the
converged background are subtracted, residual neighbor pixels are
masked through three channels and reconstructed by the twin fill, and
the aperture flux is the curve of growth of what remains -- the target
model itself is never integrated into the measurement. Every step
leaves a witness; the witnesses ride the output row's flags column and
the provenance sidecar, not a human's memory of the run.

Requirements:
    numpy, scipy

Notes:
    All fluxes microjansky. The reported flux is the aperture integral
    of (filled - background); the fitted target model appears only in
    witnesses (model curve, fill fallback) and comparisons.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation

from ..schema import make_row
from ..units import flux_err_to_mag_err, ujy_to_mag
from . import recipe
from .background import ambient_surface
from .stamp import Stamp


# ------------------------------------
# Masks: three channels
# ------------------------------------
def build_mask(
        comps: list[dict],
        fitted_by: dict,
        star_masks: list,
        stamp: Stamp,
        seeing_arcsec: float,
        scene: np.ndarray,
        neighbors: np.ndarray,
        image: np.ndarray,
        good: np.ndarray,
        *,
        tag: str = '',
) -> tuple[np.ndarray, float]:
    """Mask residual neighbor light through three channels.

    1. Intersection: a neighbor's own fitted model above the isophote
       threshold AND inside its catalog-geometry ellipse (seeing floor)
       -- faint shreds mask little, bright sources stay geometrically
       bounded.
    2. Stars: each measured profile above its own isophote, no
       geometric cap (a measured profile cannot claim light it does not
       see).
    3. Flood: existing mask islands grown into contiguous pixels whose
       data departs from the local ambient surface -- catches glow the
       catalog never admitted (a shredded catalog row can leak several
       times its cataloged flux). Symmetric: escaped glow AND
       over-subtraction holes both flood. Pixels the fitted target
       model claims are protected, so the protection perimeter tracks
       whatever the solve says the target is.

    Parameters
    ----------
    comps : list of dict
        Scene components.
    fitted_by : dict
        Per-component fitted neighbor images (owner name -> image).
    star_masks : list of (name, profile image)
        Measured star profiles (stars.subtract_stars).
    stamp : Stamp
        This band's stamp.
    seeing_arcsec : float
        Band PSF FWHM (the geometric floor).
    scene : np.ndarray
        Full fitted scene (target + neighbors), counts.
    neighbors : np.ndarray
        Fitted neighbor light only, counts.
    image : np.ndarray
        Star-subtracted data (counts).
    good : np.ndarray
        Usable-pixel map.
    tag : str
        Run-log prefix.

    Returns
    -------
    mask : np.ndarray
        Boolean neighbor mask.
    flood_ujy : float
        Escaped-glow flux claimed by the flood channel (uJy; witness).
    """
    pix, sigma = stamp.pixscale, stamp.sigma
    rr = stamp.rr
    seeing_px = seeing_arcsec / pix
    yy, xx = np.indices(image.shape)
    mask = np.zeros(image.shape, bool)
    for comp in comps:
        if comp['name'] == 'target' or comp['name'] not in fitted_by:
            continue
        dx, dy = xx - comp['x'], yy - comp['y']
        if comp['shape'] is None:
            geo = (dx * dx + dy * dy) < (recipe.GEO_SEEING_FLOOR
                                         * seeing_px) ** 2
        else:
            shape = comp['shape']
            ct, st = np.cos(shape['theta']), np.sin(shape['theta'])
            u = dx * ct + dy * st
            v = -dx * st + dy * ct
            z = np.sqrt(u * u + (v / (1.0 - shape['ellip'] + 1e-9)) ** 2)
            geo = z < max(recipe.GEO_REFF_FACTOR * shape['reff_px'],
                          recipe.GEO_SEEING_FLOOR * seeing_px)
        mask |= geo & (fitted_by[comp['name']] > recipe.K_ISO * sigma)
    for _, star_profile in star_masks:
        mask |= star_profile > recipe.K_ISO * sigma

    # Flood channel: seeded only -- it cannot invent masks. The
    # ambient-relative threshold keeps coherent large-scale light from
    # flooding or chaining; growth is bounded in radius.
    work = image - neighbors
    flood_ujy = 0.0
    ambient = ambient_surface(work, good, mask, rr, pix)
    if ambient is not None and mask.any():
        candidates = (good & np.isfinite(ambient)
                      & (np.abs(work - ambient) > recipe.K_ISO * sigma)
                      & ((scene - neighbors) < 0.5 * sigma))
        flood = mask.copy()
        for _ in range(int(round(recipe.FLOOD_MAX_AS / pix))):
            grown = binary_dilation(flood) & candidates
            if not (grown & ~flood).any():
                break
            flood = flood | grown
        new_px = flood & ~mask
        if new_px.any():
            flood_ujy = float((work - ambient)[new_px].sum() * stamp.cf)
            print(f"    {tag}flood: +{new_px.sum() * pix ** 2:.0f} as2 "
                  f"masked, {flood_ujy:+.1f} uJy of escaped glow")
            mask = flood

    # Optional mask-free core: with TARGET_MASK_FREE_AS > 0 the
    # target's inner radius is never masked by any channel; at 0 the
    # twin fill is trusted over a neighbor model's core subtraction.
    freed = mask & (rr < recipe.TARGET_MASK_FREE_AS)
    if freed.any():
        print(f"    {tag}mask-free core: restored "
              f"{freed.sum() * pix ** 2:.1f} as2 of data inside "
              f"{recipe.TARGET_MASK_FREE_AS:.0f}\"")
        mask = mask & ~freed
    return mask, flood_ujy


# ------------------------------------
# Twin fill
# ------------------------------------
def twin_fill(
        image: np.ndarray,
        neighbors: np.ndarray,
        mask: np.ndarray,
        good: np.ndarray,
        stamp: Stamp,
        model_fill: np.ndarray,
        *,
        aperture_arcsec: float,
        tag: str = '',
) -> dict:
    """Reconstruct masked pixels from their mirror through the target.

    Masked pixels take the neighbor-subtracted DATA at the point
    reflection through the target center, corrected by the odd part of
    the ambient surface (even components -- background plane, the
    target's own wings, centered diffuse light -- cancel identically in
    the mirror difference), CLAMPED between the mirror value and the
    model fill so holes are impossible by construction. The model fill
    (fitted target + background) is the fallback for invalid mirrors.
    Blank pixels inside the aperture fill the same way -- the coverage
    gate has already bounded how many there can be.

    Returns
    -------
    fill : dict
        filled (image with holes replaced), twin_frac (fraction of
        masked aperture pixels with a valid mirror), corr_ujy (ambient
        odd-part correction, uJy), fill_vs_model_ap (what the fill put
        in the aperture relative to the model's own expectation).
    """
    work = image - neighbors
    cx, cy = stamp.cx, stamp.cy
    rr, pix, cf = stamp.rr, stamp.pixscale, stamp.cf
    ny, nx = work.shape
    yy, xx = np.indices(work.shape)
    myy = np.round(2.0 * cy - yy).astype(int)
    mxx = np.round(2.0 * cx - xx).astype(int)
    in_bounds = (myy >= 0) & (myy < ny) & (mxx >= 0) & (mxx < nx)
    mirror_val = np.zeros_like(work)
    mirror_ok = np.zeros(work.shape, bool)
    donor_ok = good & ~mask
    mirror_val[in_bounds] = work[myy[in_bounds], mxx[in_bounds]]
    mirror_ok[in_bounds] = donor_ok[myy[in_bounds], mxx[in_bounds]]

    holes = mask | (~good & (rr < aperture_arcsec))
    delta_ambient = np.zeros_like(work)
    if holes.any():
        ambient = ambient_surface(work, good, mask, rr, pix)
        if ambient is not None:
            amb_mirror = np.full_like(work, np.nan)
            amb_mirror[in_bounds] = ambient[myy[in_bounds], mxx[in_bounds]]
            delta_ambient = np.where(
                np.isfinite(ambient) & np.isfinite(amb_mirror),
                ambient - amb_mirror, 0.0)
    fill_lo = np.minimum(mirror_val, model_fill)
    fill_hi = np.maximum(mirror_val, model_fill)
    fill_val = np.clip(mirror_val + delta_ambient, fill_lo, fill_hi)
    filled = np.where(holes,
                      np.where(mirror_ok, fill_val, model_fill),
                      work)

    ap = rr < aperture_arcsec
    ap_holes = holes & ap
    twin_frac = float(mirror_ok[ap_holes].mean()) if ap_holes.any() else 0.0
    corr_ujy = float((fill_val - mirror_val)[holes & mirror_ok].sum() * cf)
    # Attribution witness: what the fill put in the aperture RELATIVE
    # to the model's own expectation there (fallback pixels are the
    # model, so they contribute zero -- this is the twin pixels' vote).
    fill_vs_model = float((filled - model_fill)[ap_holes].sum() * cf)
    if holes.any():
        print(f"    {tag}fill: {ap_holes.sum() / max(ap.sum(), 1) * 100:.0f}%"
              f" of aperture masked, {twin_frac * 100:.0f}% twin-filled, "
              f"ambient corr {corr_ujy:+.1f} uJy, fill-vs-model "
              f"{fill_vs_model:+.1f} uJy")
    return dict(filled=filled, twin_frac=twin_frac, corr_ujy=corr_ujy,
                fill_vs_model_ap=fill_vs_model)


# ------------------------------------
# Curve of growth and its witnesses
# ------------------------------------
def curve(img: np.ndarray, rr: np.ndarray, cf: float,
          rgrid: np.ndarray) -> np.ndarray:
    """Enclosed flux (uJy) at each curve-of-growth radius."""
    return np.array([img[rr < r].sum() * cf for r in rgrid])


def enclosed_at(rgrid: np.ndarray, enc: np.ndarray, radius: float) -> float:
    """Enclosed flux interpolated at an arbitrary radius."""
    return float(np.interp(radius, rgrid, enc))


def ped_fit(enc: np.ndarray, rgrid: np.ndarray) -> tuple[float, float, float]:
    """Fit enclosed(r) = F + b*pi*r^2 over the pedestal window.

    b is the residual uniform-background term (uJy/arcsec^2): zero when
    the background is right, and the fit rms is the flatness proof no
    single-radius statistic gives.
    """
    lo, hi = recipe.PED_WINDOW_AS
    window = (rgrid >= lo) & (rgrid <= hi)
    design = np.column_stack([np.ones(int(window.sum())),
                              np.pi * rgrid[window] ** 2])
    (F, b), *_ = np.linalg.lstsq(design, enc[window], rcond=None)
    rms = float((enc[window] - design @ [F, b]).std())
    return float(F), float(b), rms


def plateau_hold(enc: np.ndarray, flux_ap: float,
                 rgrid: np.ndarray) -> float:
    """First radius where the curve plateaus AND holds to the grid end.

    Per-increment quietness alone cannot tell flat from a steady
    sub-threshold drift, hence the hold test. Returns -1 when the curve
    never certifies.
    """
    ref = max(abs(flux_ap), 1e-9)
    increments = np.abs(np.diff(enc) / np.diff(rgrid))
    converged = increments < recipe.PLATEAU_EPS * ref
    for i in range(len(converged) - recipe.PLATEAU_RUN + 1):
        if converged[i:i + recipe.PLATEAU_RUN].all():
            if abs(enc[-1] - enc[i]) < recipe.HOLD_MAX * ref:
                return float(rgrid[i])
    return -1.0


# ------------------------------------
# The witness row
# ------------------------------------
def witness_row(
        enc: np.ndarray,
        model_cog: np.ndarray,
        m_ap_cat: float | None,
        stamp: Stamp,
        good: np.ndarray,
        mask: np.ndarray,
        twin_frac: float,
        neighbors: np.ndarray,
        star_img: np.ndarray,
        bg: dict,
        track: list,
        flood_ujy: float,
        seeing_arcsec: float,
        seeing_src: str,
        *,
        rgrid: np.ndarray,
        aperture_arcsec: float,
        solve_info: dict | None = None,
) -> dict:
    """Assemble every per-band witness into one dict.

    The witnesses are the reproducibility mechanism: every quantity a
    reader would need to trust (or distrust) the flux is measured and
    recorded, never eyeballed.

    Parameters
    ----------
    enc : np.ndarray
        Measured curve of growth (uJy at each rgrid radius).
    model_cog : np.ndarray
        The fitted target model's own curve of growth.
    m_ap_cat : float or None
        Catalog-model flux in the aperture (native scene band only).
    stamp : Stamp
        This band's stamp.
    good, mask : np.ndarray
        Usable-pixel and neighbor-mask maps.
    twin_frac : float
        Mirror-filled fraction of the masked aperture area.
    neighbors, star_img : np.ndarray
        Subtracted neighbor and star light (counts).
    bg : dict
        The converged background (background.bin_plane).
    track : list of float
        The background constant's path through the alternation.
    flood_ujy : float
        Escaped-glow flux claimed by the flood channel.
    seeing_arcsec, seeing_src : float, str
        Band PSF FWHM and its provenance.
    rgrid : np.ndarray
        Curve-of-growth radii.
    aperture_arcsec : float
        Science aperture radius.
    solve_info : dict, optional
        Shape-solve diagnostics (solve.solve_shapes), when one ran.

    Returns
    -------
    witness : dict
        One JSON-ready dict of every per-band witness.
    """
    rr, cf = stamp.rr, stamp.cf
    sb = stamp.sb
    ap = rr < aperture_arcsec
    excess_out = min(recipe.EXCESS_OUT_AS, float(rgrid.max()))
    flux_ap = enclosed_at(rgrid, enc, aperture_arcsec)
    growth = enclosed_at(rgrid, enc, excess_out) - flux_ap
    model_ap = enclosed_at(rgrid, model_cog, aperture_arcsec)
    own = enclosed_at(rgrid, model_cog, excess_out) - model_ap
    F_ped, b_ped, rms_ped = ped_fit(enc, rgrid)
    row = dict(
        f_ap_uJy=round(flux_ap, 1),
        aperture_as=float(aperture_arcsec),
        excess_growth_uJy=round(growth - own, 1),
        model_own_growth_uJy=round(own, 1),
        m_ap_fit_uJy=round(model_ap, 1),
        m_ap_cat_uJy=round(m_ap_cat, 1) if m_ap_cat is not None else None,
        cov=round(float(good[ap].mean()), 3),
        maskfrac_ap=round(float(mask[ap].mean()), 3),
        twinfrac=round(twin_frac, 2),
        nbsub_ap_uJy=round(float(neighbors[ap].sum() * cf), 1),
        starsub_ap_uJy=round(float(star_img[ap].sum() * cf), 1),
        flood_uJy=round(flood_ujy, 1),
        bg_sb=round(bg['const'] * sb, 4),
        bg_tilt_sb=round(max(abs(bg['coefs'][1]), abs(bg['coefs'][2]))
                         * sb, 4),
        bg_rej_bins=f"{bg['n_rej']}/{bg['n_bins']}",
        farfield_sb=(round(stamp.farfield_sb, 4)
                     if stamp.farfield_sb is not None else None),
        alt_track_sb=[round(v * sb, 4) for v in track],
        ped_b_sb=round(b_ped, 4),
        ped_rms_uJy=round(rms_ped, 2),
        r_conv_as=plateau_hold(enc, flux_ap, rgrid),
        seeing_as=round(seeing_arcsec, 2),
        seeing_src=seeing_src,
    )
    if solve_info is not None:
        row['solve'] = dict(seats=solve_info['seats'],
                            nfev=solve_info['nfev'],
                            nfev_track=solve_info.get('nfev_track'),
                            seconds=solve_info['seconds'],
                            at_bound=solve_info['at_bound'],
                            params=solve_info['params'])
    return row


# ------------------------------------
# Error model
# ------------------------------------
def flux_error(
        stamp: Stamp,
        good: np.ndarray,
        *,
        aperture_arcsec: float,
) -> tuple[float, str]:
    """Statistical flux error: inverse variance when the archive serves
    it, global sky rms otherwise. Floors and inflation belong to the
    SED fitter, never to this table.

    Parameters
    ----------
    stamp : Stamp
        This band's stamp (carries the invvar cutout when one exists).
    good : np.ndarray
        Usable-pixel map.
    aperture_arcsec : float
        Science aperture radius.

    Returns
    -------
    error : tuple
        (flux_err_ujy, error-model label 'ivm' or 'skyrms').
    """
    in_aperture = stamp.rr < aperture_arcsec
    n_aper = int(in_aperture.sum())
    if stamp.invvar is not None:
        ok = in_aperture & good & (stamp.invvar > 0)
        var_raw = float(np.sum(1.0 / stamp.invvar[ok])) if ok.any() else 0.0
        n_bg = max(int((good & (stamp.rr > recipe.BG_RMIN_AS)).sum()), 1)
        var_sky = (n_aper * stamp.sigma) ** 2 / n_bg
        return float(np.sqrt(var_raw + var_sky)) * stamp.cf, 'ivm'
    return float(stamp.sigma * np.sqrt(n_aper)) * stamp.cf, 'skyrms'


# ------------------------------------
# Output row
# ------------------------------------
def qa_flags(witness: dict, *, n_comps: int, consumed: list[str]) -> str:
    """Machine-parsable QA tokens for the output row's flags column.

    The decisive per-row witnesses, as key=value tokens; the full
    witness dict rides the provenance sidecar. Downstream selection
    filters on these without opening a single QA figure.
    """
    tokens = [
        f"cov={witness['cov']:.3f}",
        f"maskfrac={witness['maskfrac_ap']:.3f}",
        f"twinfrac={witness['twinfrac']:.2f}",
        f"nbsub={witness['nbsub_ap_uJy']:.1f}",
        f"excess={witness['excess_growth_uJy']:+.1f}",
        f"pedb={witness['ped_b_sb']:+.4f}",
        f"conv={witness['r_conv_as']:.0f}",
        f"bg={witness['bg_sb']:+.4f}",
    ]
    if witness.get('target_refit_x_cat') is not None:
        tokens.append(f"refit={witness['target_refit_x_cat']:.2f}")
    solve = witness.get('solve')
    if solve and solve.get('at_bound'):
        tokens.append(f"atbound={len(solve['at_bound'])}")
    if consumed:
        tokens.append(f"reg={len(consumed)}")
    if n_comps == 0:
        tokens.append("scene=none")
    return ";".join(tokens)


def measurement_to_row(measurement: dict, *, mode: str = 'aperture') -> dict:
    """Schema row for one measured band (schema.make_row keywords).

    Aperture mode reports the curve-of-growth aperture flux; sersic
    mode reports the fitted target model's flux instead (forced
    photometry through the same scene fit). Both carry the same
    witnesses and the same statistical error model.
    """
    witness = measurement['witness']
    if mode == 'sersic':
        # NaN when no target model exists to report (blind scene, or
        # the refit disabled by patch without an explicit shape).
        flux = witness.get('target_model_uJy', float('nan'))
    else:
        flux = measurement['flux_ujy']
    err = measurement['flux_err_ujy']
    return make_row(
        band=f"{measurement['instrument']}_{measurement['band']}",
        flux_ujy=flux,
        flux_err_ujy=err,
        mag=ujy_to_mag(flux),
        mag_err=flux_err_to_mag_err(flux, err),
        target_ra=measurement['target_ra'],
        target_dec=measurement['target_dec'],
        match_ra=measurement['target_ra'],
        match_dec=measurement['target_dec'],
        sep_arcsec=0.0,
        flags=qa_flags(witness, n_comps=measurement['n_comps'],
                       consumed=measurement['registry_consumed']),
        source=f"sedphot_{mode}_scene_{measurement['err_model']}",
    )
