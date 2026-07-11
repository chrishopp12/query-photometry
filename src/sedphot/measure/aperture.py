"""
aperture.py

Curve-of-Growth Aperture Photometry
---------------------------------------------------------

The one aperture recipe, applied identically to every instrument:

    1. cut a stamp at the target position;
    2. build the neighbor mask (auto, or a user mask reprojected onto this
       band's pixel grid);
    3. subtract a sigma-clipped annulus sky, with bright annulus sources
       matched-filter-detected and masked;
    4. integrate to the aperture -- data where unmasked; masked pixels
       take their point-reflected twin through the target center, or the
       azimuthal profile where the twin is unusable;
    5. return the enclosed-flux curve of growth and the aperture flux, in
       microjanskys.

Requirements:
    numpy, astropy

Notes:
    Two error models, chosen by data availability:
      'skyrms'  sky_std * sqrt(N_aper) * cf -- used when no
                inverse-variance map exists.
      'ivm'     sqrt(sum 1/wht + N_aper^2 * sky_std^2 / N_sky) * cf -- adds
                the per-pixel noise where the archive serves real weights
                (Legacy bricks, HST); the second term is the sky-level
                uncertainty.
    Masked aperture pixels contribute their fill values (mirror twin or
    azimuthal profile); their noise is not separately inflated.
"""
from __future__ import annotations

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D

from ..bands import wave_um
from ..results import ImageProduct
from ..schema import make_row
from ..units import flux_err_to_mag_err, ujy_to_mag
from .calibrate import calib_factor, load_image, pixel_scale_arcsec
from .deblend import (apportion_neighbors, reproject_template,
                      subtract_fitted_templates, target_template)
from .masks import (neighbor_mask, nontarget_parents, radii_arcsec,
                    reproject_mask, sky_source_mask)
from .sky import annulus_sky, annulus_sky_plane

# ------------------------------------
# Constants
# ------------------------------------
# Curve-of-growth radius grid (arcsec).
DEFAULT_RGRID = np.arange(2.0, 30.0, 1.0)

# Minimum fraction of aperture pixels with real data; below this the band
# demotes to no_coverage (the azimuthal fill corrects small deficits, but
# there is no honest correction when a whole sector of the profile is gone).
COVERAGE_MIN = 0.95

# Outer window (arcsec) of the curve of growth used for the sky-bias slope.
COG_SLOPE_WINDOW = 8.0

# Convergence search: an increment of the curve of growth reads as
# converged below this fraction of the aperture flux per arcsec (or
# below twice its own shot noise, whichever is larger), and a plateau is
# this many consecutive converged increments. 1%/arcsec tolerates the
# slow legitimate growth of an extended profile approaching total while
# a real contamination step (several % per arcsec) still breaks the run.
PLATEAU_EPS = 0.01
PLATEAU_RUN = 3

# A plateau is only a plateau if it HOLDS: the cumulative drift from it
# to the aperture edge must stay within this fraction of the flux (or
# twice the accumulated noise). Per-increment quietness alone cannot
# tell flat-with-noise from a steady sub-threshold drift -- 0.7 uJy per
# arcsec for 18 arcsec is 8% of a 150 uJy flux and every single step of
# it clears a 1%-per-arcsec bar.
STEP_CONV_MAX = 0.02


class ApertureCoverageError(RuntimeError):
    """The photometry aperture lands on too many missing pixels."""

    def __init__(self, message: str, coverage: float):
        super().__init__(message)
        self.coverage = coverage


def _fill_from_mirror(
        sub: np.ndarray,
        fill: np.ndarray,
        cx: float,
        cy: float,
        profile: np.ndarray,
        bin_index: np.ndarray,
) -> np.ndarray:
    """Replace fill pixels by their point-reflected twin, profile else.

    The twin through the target center carries the target's own light at
    the same radius WITH its ellipticity and local ambient statistics --
    a circular-annulus median dilutes an elongated profile and, in a
    crowded field, averages unmasked neighbor wings into the very pixels
    the mask just removed. The azimuthal profile remains the fallback
    wherever the twin is itself masked, blank, or off the stamp
    (opposite-side contamination, chip edges).
    """
    out = sub.copy()
    yy, xx = np.indices(sub.shape)
    my = np.round(2.0 * cy - yy).astype(int)
    mx = np.round(2.0 * cx - xx).astype(int)
    inb = ((my >= 0) & (my < sub.shape[0])
           & (mx >= 0) & (mx < sub.shape[1]))
    twin_ok = fill & inb
    sel = np.where(twin_ok)
    twin_ok[sel] &= ~fill[my[sel], mx[sel]] \
        & np.isfinite(sub[my[sel], mx[sel]])
    out[twin_ok] = sub[my[twin_ok], mx[twin_ok]]
    rest = fill & ~twin_ok
    out[rest] = profile[bin_index[rest]]
    return out


def cog_slope(rgrid: np.ndarray, enclosed: np.ndarray, flux: float,
              window: float = COG_SLOPE_WINDOW) -> float:
    """Relative slope of the curve of growth over its outer window.

    Fraction of the aperture flux per arcsec of radius: ~0 for a converged
    curve, strongly negative when the sky is over-estimated (every annulus
    of area is debited too much and the curve turns over and falls).
    """
    outer = rgrid >= rgrid.max() - window
    if outer.sum() < 3 or not np.isfinite(flux) or flux == 0:
        return float('nan')
    slope = np.polyfit(rgrid[outer], enclosed[outer], 1)[0]
    return float(slope / abs(flux))


def cog_step(rgrid: np.ndarray, enclosed: np.ndarray, flux: float,
             aperture_arcsec: float,
             step_noise: np.ndarray | None = None,
             hold_to: float | None = None) -> tuple[float, float]:
    """Pre-aperture convergence radius and the step acquired past it.

    Walks the curve of growth outward to the FIRST plateau -- PLATEAU_RUN
    consecutive increments each below PLATEAU_EPS x |flux| per arcsec
    (or below twice its own noise when step_noise is given) -- and
    reports where the curve first converged and what fraction of the
    aperture flux arrived between there and the aperture edge. This is
    the mid-curve witness the outer slope cannot be: a curve that
    converges at 8", steps +20% at 10-12" (a neighbor's wings entering),
    and re-flattens shows cogslope ~ 0 but a loud step here.

    Parameters
    ----------
    rgrid, enclosed : np.ndarray
        The curve of growth (arcsec, uJy).
    flux : float
        Aperture flux (uJy); the step normalization.
    aperture_arcsec : float
        The aperture radius; the search stops there.
    step_noise : np.ndarray, optional
        1-sigma flux noise of each increment (len(rgrid) - 1); widens
        the plateau tolerance so a noisy faint curve is not read as
        never-converged.

    hold_to : float, optional
        Radius the plateau must HOLD to (default: the end of rgrid).
        Callers keep this short of the sky annulus, where the
        diagnostic curve loses meaning.

    Returns
    -------
    conv_arcsec, step : float, float
        Radius from which the enclosed flux HOLDS -- locally quiet
        increments AND cumulative drift bounded by STEP_CONV_MAX
        (noise-widened) both to the aperture edge and out to hold_to --
        and the drift to the aperture. Bounding only to the aperture
        would be a tautology (any drift "converges" just before the
        edge); a plateau that the curve walks away from beyond the
        aperture was never a plateau. (nan, nan) when no radius
        qualifies: the flux is still a function of radius (a blend, a
        residual background pedestal, or a target genuinely larger than
        the aperture; cogend and cogped say which way).
    """
    rgrid = np.asarray(rgrid, float)
    enclosed = np.asarray(enclosed, float)
    if not np.isfinite(flux) or flux == 0 or rgrid.size < PLATEAU_RUN + 1:
        return float('nan'), float('nan')
    if hold_to is None:
        hold_to = float(rgrid.max())
    hold_to = min(float(hold_to), float(rgrid.max()))
    increments = np.diff(enclosed)
    tolerance = PLATEAU_EPS * abs(flux) * np.diff(rgrid)
    if step_noise is not None:
        step_noise = np.asarray(step_noise, float)
        tolerance = np.maximum(tolerance, 2.0 * step_noise)
    # Quiet increments anywhere inside the hold horizon; the plateau run
    # may extend beyond the aperture -- with a small aperture there is
    # no room for the run before it -- but must START at least 1" inside
    # it: a plateau beginning AT the aperture edge certifies nothing
    # about a step that just landed inside (the flux would contain the
    # step while the metric read clean).
    quiet = (np.abs(increments) <= tolerance) & (rgrid[1:] <= hold_to + 1e-9)
    at_aperture = float(np.interp(aperture_arcsec, rgrid, enclosed))
    at_hold = float(np.interp(hold_to, rgrid, enclosed))
    end_i = int(np.searchsorted(rgrid, hold_to + 1e-9)) - 1
    for j in range(len(quiet) - PLATEAU_RUN + 1):
        if rgrid[j] > aperture_arcsec - 1.0 + 1e-9:
            break
        if not quiet[j:j + PLATEAU_RUN].all():
            continue
        plateau = float(np.median(enclosed[j:j + PLATEAU_RUN + 1]))
        step = (at_aperture - plateau) / abs(flux)
        drift = (at_hold - plateau) / abs(flux)
        drift_max = STEP_CONV_MAX
        if step_noise is not None:
            drift_max = max(drift_max, 2.0 * float(
                np.sqrt((step_noise[j:end_i] ** 2).sum())) / abs(flux))
        if abs(step) <= drift_max and abs(drift) <= drift_max:
            return float(rgrid[j]), step
    return float('nan'), float('nan')


# ------------------------------------
# Shared stamp preparation
# ------------------------------------
def prepare_stamp(
        product: ImageProduct,
        coord: SkyCoord,
        *,
        cutout_half_arcsec: float,
        sky_in: float,
        sky_out: float,
        user_mask: tuple | None = None,
        protect_radius: float = 4.0,
        deblend: bool = True,
        deblend_centers: SkyCoord | None = None,
        deblend_templates: tuple | None = None,
) -> dict:
    """Load, cut, deblend, sky-subtract, and mask one band -- the shared
    front half of both measurement modes.

    The cutout is padded (NaN) where it leaves the array, and blank pixels
    (NaN or exactly zero -- the fill value of every archive served here)
    are carried as a nodata mask: excluded from the sky, the moments, and
    the profile, and handed to the caller for fill and coverage
    accounting. Interloper light is then removed by apportioned symmetric
    deblending (see measure.deblend): masking alone keeps everything
    below its isophote, and a bright complex's sub-threshold fringe both
    contaminates the aperture and biases any annulus statistic. The sky
    is estimated twice AROUND that: a first pass with matched-filter peak
    rejection sets the detection threshold, then -- on the deblended
    stamp -- every detected segment is masked and a sigma-clipped PLANE
    is fit through the annulus.

    Parameters (beyond measure_aperture's)
    ----------
    deblend : bool
        Remove interloper light by apportioned symmetric deblending
        before the plane sky and the mask. [default: True]
    deblend_centers : SkyCoord, optional
        World positions of the deblend cores -- pass the reference
        band's detections so every band removes the same physical
        sources; detected on this band when None.
    deblend_templates : (templates, target_template, wcs, seeing), optional
        The reference band's contained symmetric neighbor templates,
        its target template, and their grid. When given, every
        component's MODEL is a fixed reference shape -- reprojected,
        PSF-matched, one amplitude fit per band -- so a shallow band
        subtracts wings it cannot itself detect. Takes precedence over
        per-band template building.

    Returns
    -------
    prep : dict
        stamp (deblended, sky-subtracted), stamp_raw (pre-deblend, same
        sky), stamp_wcs, cx/cy, pixscale, cf, sky_level, sky_std, mask,
        mask_mode, nodata, annulus_srcmask, rr, px/py, half_px,
        n_deblended.
    """
    image, image_wcs, header = load_image(product.path)
    cf = calib_factor(product.calib, header)
    pixscale = pixel_scale_arcsec(image_wcs)
    px, py = [float(v) for v in image_wcs.world_to_pixel(coord)]
    half_px = int(round(cutout_half_arcsec / pixscale))
    cut = Cutout2D(image, (px, py), 2 * half_px + 1, wcs=image_wcs,
                   mode='partial', fill_value=np.nan)
    stamp = cut.data.astype(float)
    stamp_wcs = cut.wcs
    cx, cy = [float(v) for v in stamp_wcs.world_to_pixel(coord)]
    rr = radii_arcsec(stamp.shape, cx, cy, pixscale)

    # Blank = NaN (partial-cutout pad, PS1 edges) or exact zero (CFHT/HST/
    # Legacy fill). Real float pixels are never exactly zero.
    nodata = ~np.isfinite(stamp) | (stamp == 0.0)

    # Sky pass 1: crude level for the detection threshold.
    sky_level, sky_std, annulus_srcmask = annulus_sky(
        stamp, cx, cy, pixscale, sky_in=sky_in, sky_out=sky_out,
        seeing_arcsec=product.seeing_arcsec, nodata=nodata)
    # Dead detector regions are not always exact zeros: a chip gap in a
    # sky-inclusive stack holds some low sentinel that reads as a pixel
    # tens of sigma BELOW sky -- physically impossible for data -- and one
    # such column through the integration region craters the curve of
    # growth. Anything deeper than 10 sigma below sky is nodata.
    nodata |= (stamp - sky_level) < -10.0 * max(sky_std, 1e-30)

    # Deblend: subtract every interloper's apportioned symmetric share
    # (pass-1 scalar zero is enough for the templates). Everything
    # downstream -- the plane sky, the mask, the fills, the curve --
    # then sees a stamp whose neighbor light is already gone, fringes
    # included, instead of masked down to some isophote.
    n_deblended = 0
    stamp_deblended = stamp
    if deblend:
        work = np.where(nodata, np.nan, stamp - sky_level)
        if deblend_templates is not None:
            # Forced photometry of the contaminants: the reference band
            # defines EVERY component's shape -- the target's included,
            # since a shallow band's own target template is noise-gutted
            # at low surface brightness and would lose the blend zone to
            # the neighbors in the solve. Reprojected, PSF-matched, one
            # fitted amplitude per component per band; the fit absorbs
            # zeropoint, pixel area, and color.
            ref_templates, ref_target, ref_wcs, ref_seeing = deblend_templates
            blur_px = (np.sqrt(max(product.seeing_arcsec ** 2
                                   - ref_seeing ** 2, 0.0))
                       / 2.355 / pixscale)
            from .masks import _smooth

            def _to_band(t):
                t_band = reproject_template(t, ref_wcs, stamp_wcs,
                                            stamp.shape)
                return _smooth(t_band, blur_px) if blur_px > 0.3 else t_band

            residual, share, n_deblended = subtract_fitted_templates(
                work, [_to_band(t) for t in ref_templates],
                _to_band(ref_target), sky_std, nodata=nodata)
        else:
            centers_px = None
            if deblend_centers is not None:
                mx, my = stamp_wcs.world_to_pixel(deblend_centers)
                centers_px = list(zip(np.atleast_1d(my).astype(float),
                                      np.atleast_1d(mx).astype(float)))
            residual, share, n_deblended = apportion_neighbors(
                work, sky_std, cx, cy, pixscale, centers=centers_px,
                protect_radius=protect_radius,
                seeing_arcsec=product.seeing_arcsec, nodata=nodata)
        stamp_deblended = stamp - share

    # Sky pass 2: a sigma-clipped PLANE through the annulus of the
    # DEBLENDED stamp, with sources excluded SYMMETRICALLY with the
    # aperture's own treatment (the target's segment, bright sources in
    # full, DoG cores of the faint -- the ambient faint-source
    # background stays in both aperture and sky so its mean cancels; see
    # sky_source_mask). The plane absorbs the large-scale gradient a
    # halo or ICL lays across the field. The scalar from pass 1 stands
    # in when the fit is starved or not believable.
    segmask = sky_source_mask(stamp_deblended - sky_level, sky_std, cx, cy,
                              pixscale, seeing_arcsec=product.seeing_arcsec,
                              nodata=nodata)
    sky_map: np.ndarray | float = sky_level
    try:
        sky_map, sky_level, sky_std, annulus_srcmask = annulus_sky_plane(
            stamp_deblended, cx, cy, pixscale, sky_in=sky_in, sky_out=sky_out,
            seeing_arcsec=product.seeing_arcsec, nodata=nodata,
            extra_mask=segmask)
    except ValueError as e:
        print(f"  {product.instrument} {product.band}: plane sky pass "
              f"fell back to the annulus median ({e})")
    sub = stamp_deblended - sky_map
    sub_raw = stamp - sky_map

    if user_mask is not None:
        mask, mask_wcs = user_mask
        if mask_wcs is not None:
            mask = reproject_mask(stamp_wcs, sub.shape, mask_wcs, mask)
        elif mask.shape != sub.shape:
            raise ValueError(
                f"user mask shape {mask.shape} != stamp shape {sub.shape} and "
                f"carries no WCS to reproject with (pass a FITS mask, or match grids)")
        mask_mode = "user"
    else:
        mask = neighbor_mask(sub, sky_std, cx, cy, pixscale,
                             protect_radius=protect_radius,
                             seeing_arcsec=product.seeing_arcsec,
                             nodata=nodata)
        mask_mode = "auto"

    return dict(stamp=sub, stamp_raw=sub_raw, stamp_wcs=stamp_wcs,
                cx=cx, cy=cy, pixscale=pixscale,
                cf=cf, sky_level=sky_level, sky_std=sky_std, mask=mask,
                mask_mode=mask_mode, nodata=nodata,
                annulus_srcmask=annulus_srcmask, rr=rr,
                px=px, py=py, half_px=half_px, n_deblended=n_deblended)


# ------------------------------------
# Measurement
# ------------------------------------
def measure_aperture(
        product: ImageProduct,
        coord: SkyCoord,
        *,
        aperture_arcsec: float,
        sky_in: float,
        sky_out: float,
        cutout_half_arcsec: float,
        rgrid: np.ndarray | None = None,
        user_mask: tuple | None = None,
        protect_radius: float = 4.0,
        mask_mode_label: str | None = None,
        deblend: bool = True,
        deblend_centers: SkyCoord | None = None,
        deblend_templates: tuple | None = None,
) -> dict:
    """Uniform aperture measurement for one band.

    Parameters
    ----------
    product : ImageProduct
        The image to measure (path + calibration + seeing).
    coord : SkyCoord
        Aperture center.
    aperture_arcsec : float
        Aperture radius.
    sky_in, sky_out : float
        Background annulus radii (arcsec); must clear the source envelope.
    cutout_half_arcsec : float
        Stamp half-size; must contain the annulus.
    rgrid : np.ndarray, optional
        Curve-of-growth radii. [default: 2..29 arcsec, 1 arcsec steps]
    user_mask : (mask, wcs) tuple, optional
        From masks.load_user_mask; reprojected onto this band. When the
        mask carries no WCS it must already match this band's stamp grid.
    protect_radius : float
        Auto-mask protection radius around the target (arcsec). [default: 4.0]
    mask_mode_label : str, optional
        Override for the recorded mask mode -- e.g. 'autoref' when
        user_mask carries the auto-mask derived once on a reference band
        and shared across instruments.

    Returns
    -------
    measurement : dict
        Fluxes in uJy plus the stamps/masks/curves the QA figures draw.
    """
    if rgrid is None:
        rgrid = DEFAULT_RGRID
    rgrid = np.asarray(rgrid, dtype=float)

    prep = prepare_stamp(product, coord, cutout_half_arcsec=cutout_half_arcsec,
                         sky_in=sky_in, sky_out=sky_out, user_mask=user_mask,
                         protect_radius=protect_radius, deblend=deblend,
                         deblend_centers=deblend_centers,
                         deblend_templates=deblend_templates)
    sub = prep['stamp']
    cx, cy = prep['cx'], prep['cy']
    pixscale, cf = prep['pixscale'], prep['cf']
    sky_level, sky_std = prep['sky_level'], prep['sky_std']
    mask, mask_mode = prep['mask'], (mask_mode_label or prep['mask_mode'])
    nodata = prep['nodata']
    rr = prep['rr']
    px, py, half_px = prep['px'], prep['py'], prep['half_px']

    # Coverage gate: nodata in the aperture is corrected by the azimuthal
    # fill below, but only up to a point -- past COVERAGE_MIN missing there
    # is no honest profile to fill from, and the band demotes rather than
    # ship a silently biased flux (the "0.0 uJy with status ok" class).
    # The core is gated absolutely: its peak carries an outsized flux share
    # that an annulus median cannot reconstruct, so an edge slicing the
    # inner seeing-scale disk demotes at ANY area fraction.
    in_aperture = rr < aperture_arcsec
    n_aper = int(in_aperture.sum())
    coverage = 1.0 - float((nodata & in_aperture).sum()) / max(n_aper, 1)
    if coverage < COVERAGE_MIN:
        raise ApertureCoverageError(
            f"aperture coverage {coverage:.2f} < {COVERAGE_MIN:g} "
            f"(off footprint / blank pixels)", coverage)
    core_radius = max(3.0, 2.0 * product.seeing_arcsec)
    if (nodata & (rr < core_radius)).any():
        raise ApertureCoverageError(
            f"blank pixels inside the {core_radius:g}\" core (aperture "
            f"coverage {coverage:.2f}) -- the azimuthal fill cannot "
            f"reconstruct a clipped peak", coverage)

    # Fill of masked and nodata aperture pixels -- the correction that
    # keeps a masked companion or a small blank wedge from simply
    # deleting aperture area. Each fill pixel takes its point-reflected
    # twin through the target center where the twin is clean, the
    # azimuthal profile where it is not (_fill_from_mirror). The
    # DIAGNOSTIC curve additionally fills every non-target segment
    # beyond the aperture: an unmasked neighbor out there is in no flux,
    # but it would step the curve and fake a sky alarm in the outer
    # slope.
    fill = mask | nodata
    outer_fill = fill | (nontarget_parents(sub, sky_std, cx, cy, nodata=nodata)
                         & (rr >= aperture_arcsec))
    edges = np.arange(0, rgrid.max() + 1, 1.0)
    profile = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        sel = (rr >= edges[i]) & (rr < edges[i + 1]) & ~outer_fill
        if sel.sum():
            profile[i] = np.median(sub[sel])
    bin_index = np.clip(np.digitize(rr, edges) - 1, 0, len(profile) - 1)
    filled = _fill_from_mirror(sub, fill, cx, cy, profile, bin_index)
    # The DIAGNOSTIC zone beyond the aperture fills from the azimuthal
    # profile, not the twin: out there a twin donor can sit on another
    # complex's low-surface-brightness fringe, and a sub-sigma-per-pixel
    # donor bias integrated over a filled segment re-injects the very
    # light the mask removed (a fake outer climb of the curve). Inside
    # the aperture the twin stands -- it preserves ellipticity and the
    # bias any single masked companion can carry is bounded and small.
    display = sub.copy()
    display[outer_fill] = profile[bin_index[outer_fill]]
    in_aperture_fill = fill & (rr < aperture_arcsec)
    display[in_aperture_fill] = filled[in_aperture_fill]

    enclosed = np.array([float(display[rr < radius].sum()) * cf for radius in rgrid])
    flux_ujy = float(filled[in_aperture].sum()) * cf

    masked_fraction = float((mask & in_aperture).sum()) / max(n_aper, 1)
    if masked_fraction > 0.2:
        print(f"  WARNING {product.instrument} {product.band}: "
              f"{100 * masked_fraction:.0f}% of the aperture is masked -- for a "
              f"bright/asymmetric target the auto-mask can eat real light; "
              f"inspect the QA figure and consider --mask")

    # Sky-bias witness: a converged curve of growth is flat past the
    # aperture; a negative outer slope means the sky was over-estimated.
    slope = cog_slope(rgrid, enclosed, flux_ujy)

    # Contamination witness: where the curve first converges, and how
    # much flux arrived between there and the aperture edge (the
    # mid-curve step the outer slope is blind to). Per-increment noise
    # keeps a faint band's noise wander from reading as never-converged.
    annulus_px = np.array([((rr >= lo) & (rr < hi)).sum()
                           for lo, hi in zip(rgrid[:-1], rgrid[1:])])
    step_noise = sky_std * np.sqrt(annulus_px) * cf
    # The certified neighborhood scales with the aperture: flat across
    # [plateau, aperture + 4"] says the measurement is insensitive to
    # the aperture choice; demanding flatness a fixed 13" further out
    # in a cluster core tests the cluster, not the measurement.
    conv_arcsec, step = cog_step(rgrid, enclosed, flux_ujy, aperture_arcsec,
                                 step_noise=step_noise,
                                 hold_to=min(aperture_arcsec + 4.0,
                                             sky_in - 5.0))
    # Growth rate at the aperture edge (mean of the last pre-aperture
    # increments, fraction of the flux per arcsec): quantifies a
    # non-converged curve -- mild ambient pickup reads ~+1-3%/arcsec, a
    # blend reads +5%+ -- so 'not converged' is a number, not a verdict.
    pre_edge = rgrid[1:] <= aperture_arcsec + 1e-9
    end_slope = float('nan')
    if pre_edge.sum() >= PLATEAU_RUN and np.isfinite(flux_ujy) and flux_ujy:
        last = np.where(pre_edge)[0][-PLATEAU_RUN:]
        end_slope = float(np.diff(enclosed)[last].mean()
                          / np.diff(rgrid)[last].mean() / abs(flux_ujy))

    # Wide-range flatness witness: enclosed(r) = F + pi r^2 b fit from
    # past the aperture core to short of the sky annulus. b is any
    # residual uniform background pedestal (uJy/arcsec^2); the fit rms,
    # as a fraction of the flux, is the PROOF of flatness no
    # single-radius metric gives -- a handled field fits this
    # two-parameter model, an unhandled one does not.
    pedestal = fit_rms = float('nan')
    ped_window = (rgrid >= 6.0) & (rgrid <= min(25.0, sky_in - 5.0))
    if ped_window.sum() >= 5 and np.isfinite(flux_ujy) and flux_ujy:
        design = np.column_stack([np.ones(int(ped_window.sum())),
                                  np.pi * rgrid[ped_window] ** 2])
        coef, *_ = np.linalg.lstsq(design, enclosed[ped_window], rcond=None)
        pedestal = float(coef[1])
        fit_rms = float((enclosed[ped_window] - design @ coef).std()
                        / abs(flux_ujy))

    # Error model: inverse variance when the archive serves it, sky rms else.
    if product.invvar_path is not None:
        invvar_image, _, _ = load_image(product.invvar_path)
        invvar = Cutout2D(invvar_image, (px, py), 2 * half_px + 1,
                          mode='partial', fill_value=0.0).data.astype(float)
        ok = in_aperture & (invvar > 0) & ~nodata
        var_raw = float(np.sum(1.0 / invvar[ok]))
        n_sky_est = max(int(((rr > sky_in) & (rr < sky_out)).sum()), 1)
        var_sky = (n_aper * sky_std) ** 2 / n_sky_est
        flux_err_ujy = float(np.sqrt(var_raw + var_sky)) * cf
        err_model = "ivm"
    else:
        flux_err_ujy = float(sky_std * np.sqrt(n_aper)) * cf
        err_model = "skyrms"

    return dict(
        instrument=product.instrument, band=product.band,
        wave_um=product.wave_um if np.isfinite(product.wave_um)
        else wave_um(f"{product.instrument}_{product.band}"),
        pixscale=pixscale, cf=cf,
        flux_ujy=flux_ujy, flux_err_ujy=flux_err_ujy, err_model=err_model,
        sky_level_ujy=sky_level * cf, sky_std_ujy=sky_std * cf,
        rgrid=rgrid, enclosed_ujy=enclosed,
        stamp=sub, rr=rr, mask=mask, mask_mode=mask_mode, nodata=nodata,
        outer_fill=outer_fill,
        annulus_srcmask=prep['annulus_srcmask'],
        cx=cx, cy=cy,
        aperture_arcsec=aperture_arcsec, sky_in=sky_in, sky_out=sky_out,
        n_masked_in_aperture=int((mask & in_aperture).sum()),
        aperture_coverage=coverage, masked_fraction=masked_fraction,
        cog_slope=slope, cog_conv_arcsec=conv_arcsec, cog_step=step,
        cog_end_slope=end_slope, cog_pedestal=pedestal, cog_fit_rms=fit_rms,
        stamp_raw=prep['stamp_raw'], n_deblended=prep['n_deblended'],
        target_ra=float(coord.ra.deg), target_dec=float(coord.dec.deg),
    )


def qa_flags(measurement: dict) -> str:
    """Machine-parsable QA tokens for the row's flags column.

    'key=value' joined by ';' -- always present for measured rows so
    downstream hazard filtering is uniform: cov (aperture coverage),
    maskfrac (masked fraction of the aperture), cogslope (relative outer
    curve-of-growth slope per arcsec; strongly negative = sky bias),
    cogconv (radius where the curve first converges, or 'none': not
    converged by the aperture edge), cogstep (fractional flux acquired
    between convergence and the aperture edge; large = contamination
    entered mid-aperture), cogend (only with cogconv=none: growth rate
    at the aperture edge, fraction of flux per arcsec -- ~+0.01-0.03 is
    ambient pickup, +0.05 and up is a blend), cogped (residual uniform
    background pedestal from the wide-range enclosed(r) = F + pi r^2 b
    fit, uJy/arcsec^2), cogrms (that fit's rms as a fraction of the
    flux -- the flatness proof), nbsub (interloper cores removed by the
    symmetric deblend).
    """
    tokens = []
    if 'aperture_coverage' in measurement:
        tokens.append(f"cov={measurement['aperture_coverage']:.3f}")
    if 'masked_fraction' in measurement:
        tokens.append(f"maskfrac={measurement['masked_fraction']:.3f}")
    slope = measurement.get('cog_slope')
    if slope is not None and np.isfinite(slope):
        tokens.append(f"cogslope={slope:+.4f}")
    conv = measurement.get('cog_conv_arcsec')
    if conv is not None:
        if np.isfinite(conv):
            tokens.append(f"cogconv={conv:g}")
            step = measurement.get('cog_step')
            if step is not None and np.isfinite(step):
                tokens.append(f"cogstep={step:+.3f}")
        else:
            tokens.append("cogconv=none")
            end = measurement.get('cog_end_slope')
            if end is not None and np.isfinite(end):
                tokens.append(f"cogend={end:+.3f}")
    ped = measurement.get('cog_pedestal')
    if ped is not None and np.isfinite(ped):
        tokens.append(f"cogped={ped:+.4f}")
    rms = measurement.get('cog_fit_rms')
    if rms is not None and np.isfinite(rms):
        tokens.append(f"cogrms={rms:.4f}")
    n_deb = measurement.get('n_deblended')
    if n_deb is not None:
        tokens.append(f"nbsub={int(n_deb)}")
    return ";".join(tokens)


def measurement_to_row(measurement: dict) -> dict:
    """Convert a measurement dict to a schema table row."""
    flux = measurement['flux_ujy']
    err = measurement['flux_err_ujy']
    mag = ujy_to_mag(flux)
    return make_row(
        band=f"{measurement['instrument']}_{measurement['band']}",
        flux_ujy=flux,
        flux_err_ujy=err,
        mag=mag,
        mag_err=flux_err_to_mag_err(flux, err),
        target_ra=measurement['target_ra'],
        target_dec=measurement['target_dec'],
        match_ra=measurement['target_ra'],     # forced at the target position
        match_dec=measurement['target_dec'],
        sep_arcsec=0.0,
        flags=qa_flags(measurement),
        source=(f"sedphot_aperture_r{measurement['aperture_arcsec']:g}as_"
                f"{measurement['mask_mode']}mask_{measurement['err_model']}"),
    )
