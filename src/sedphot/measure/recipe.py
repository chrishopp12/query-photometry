"""
recipe.py

Scene-Engine Recipe Constants
---------------------------------------------------------
Every science knob of the scene measurement engine in one place. The
measurement recipe is: build a scene from the survey catalog, subtract
measured stars, jointly solve component amplitudes (and shapes, where the
catalog declares misfit) against a bin-level-plane background, then mask,
fill, and integrate a curve of growth to the aperture flux. Stage-local
implementation constants (the PSF ring schedule, for example) stay in
their own module and are labeled there.

Requirements:
    numpy

Notes:
    Distances are arcsec, fluxes microjansky (uJy), surface brightness
    uJy/arcsec^2 unless a name says otherwise. Bound pairs are (low, high).
    The aperture radius and stamp size are runtime parameters (CLI), not
    constants; witness windows that depend on the aperture derive from it.
"""
from __future__ import annotations

import contextlib

import numpy as np


# ------------------------------------
# Curve of growth, witnesses, coverage
# ------------------------------------
# Curve-of-growth radii (arcsec). Witness radii are interpolated on the
# curve, so the grid does not need to contain them exactly. The grid runs
# to 40" -- well past where ordinary galaxies converge, but the most
# extended members (large cD/BCG halos) are still growing beyond 30", and
# storing the empirical curve that far lets a larger-aperture remeasure
# read a true enclosed value instead of capping at the grid end. 40" stays
# within the default 120" stamp (60" half-width) in every band, so the
# aperture never runs off the footprint.
DEFAULT_RGRID = np.arange(2.0, 41.0, 1.0)

# Excess-growth witness: measured growth minus the target model's own
# growth, from the aperture out to this radius. Growth the model cannot
# account for is contamination (or unmodeled envelope) by definition.
EXCESS_OUT_AS = 25.0

# Pedestal witness window: fit enclosed(r) = F + pi r^2 b over it. b is
# any residual uniform background (uJy/arcsec^2); 0 when the plane is right.
PED_WINDOW_AS = (6.0, 25.0)

# A curve-of-growth increment reads as converged below this fraction of
# the aperture flux per arcsec; a plateau is this many consecutive
# converged increments. Per-increment quietness alone cannot tell flat
# from a steady sub-threshold drift, so a plateau must also HOLD: total
# drift from the plateau to the grid end within HOLD_MAX x the flux.
PLATEAU_EPS = 0.01
PLATEAU_RUN = 3
HOLD_MAX = 0.02

# Aperture pixels with real data below this fraction demote the band to
# no_coverage: past it there is no honest fill, and a silently biased
# flux is worse than a refused one.
COVERAGE_MIN = 0.95
# The seeing-scale core is held tighter than the aperture: the twin/model
# fill can carry a modest core gap but not a large one. The peak itself is
# inviolable -- its twin reflection IS the peak, so no fill reconstructs a
# dead central pixel; an absolute (arcsec) radius protects only those few
# pixels regardless of the band's pixel scale.
CORE_MASKFRAC_MAX = 0.10   # of the seeing core may be masked-and-filled
PEAK_PROTECT_AS = 0.5      # a dead pixel within this radius refuses the band


# ------------------------------------
# Scene catalog
# ------------------------------------
# Scene-query cone (Tractor, Gaia). The cone must reach past the stamp's
# CORNERS or corner sources are simply absent from the scene, so the
# effective radius is
#     max(QUERY_RADIUS_AS, stamp half-diagonal + QUERY_PAD_AS)
# The floor alone covers the default 120-arcsec stamp; larger stamps grow
# the cone, and the pad keeps some just-off-stamp margin sources in reach.
QUERY_RADIUS_AS = 100.0
QUERY_PAD_AS = 15.0
# flux floor of the Tractor scene query, on the row's BRIGHTEST optical band
TRACTOR_MIN_NMGY = 0.5
# Identity radius: the closest catalog row within this of the request is
# the target. Also matches Gaia stars and registry/patch positions to
# components.
TARGET_MATCH_AS = 1.5

# The halo gate. A gated source receives a shape solve instead of a
# fixed catalog profile: a Sersic core, plus a Nuker halo only where the
# halo family is granted (see GATED_HALO). The second profile exists to
# fix MISFIT, and the catalog's own reduced chi-square is its misfit
# statement -- the necessary condition. Point sources, the target
# itself, and rows beyond the gate reach (the stamp half-width less
# GATE_EDGE_MARGIN_AS) never gate: a shape solve needs its source's
# pixels on the stamp, a distant halo seat with center freedom
# degenerates into a flat sheet across the field, and a RADIAL reach
# keeps the gate census identical on every instrument (a square-stamp
# test would admit corner sources on a rotated grid that an aligned grid
# excludes).
#
# The ceiling: far past the gate threshold, reduced chi-square flips
# meaning from "this galaxy needs an envelope" to "these pixels are not
# a galaxy model at all" (bleed trails and their shredded catalog
# echoes). Such a row neither gates NOR claims pixels in the artifact
# test -- a shape solve pointed at it can only rail its parameters
# trying to become the artifact.
GATE_FLUX_UJY = 100.0
GATE_RCHISQ = 6.0
GATE_RCHISQ_MAX = 1000.0

# Gate reach stops short of the stamp edge: a source with part of its
# flux off-frame cannot support an honest shape solve, and an edge
# vantage is never the one to define a shared source's decomposition.
# Edge sources keep their fixed catalog render (or a registry entry
# solved from a better vantage).
GATE_EDGE_MARGIN_AS = 15.0

# Ownership of blended catalog rows. A row inside the science aperture
# whose fracflux says the light at its position is dominated by OTHER
# sources is the catalog's rendering of the target's own substructure
# (knots and asymmetries a smooth profile cannot carry) -- it is target
# light, and subtracting it steals flux. Such rows leave the scene
# entirely. Outside the aperture the same signature usually means a
# real compact source embedded in a neighbor's envelope, which must
# stay modeled, so the rule is scoped to the aperture.
SHRED_FRACFLUX = 1.0

PATCH_FILENAME = 'patches.json'   # optional per-galaxy custom inputs

# A patch request (row replacement, free seat) must land on a real
# catalog row or component within this radius, or it is skipped loudly.
PATCH_MATCH_AS = 2.0


# ------------------------------------
# Components and margins
# ------------------------------------
# Extended sources this far off-stamp still enter the scene -- but only
# when their catalog-shape render lands MARGIN_MIN_UJY on the stamp.
# Design columns are normalized to unit in-stamp flux, so a near-empty
# render is a numerically explosive basis whose amplitude rails at any
# bound. An off-stamp giant whose light truly reaches the stamp is
# patches.json territory: components enter blind scenes on
# data-supported presence only.
MARGIN_AS = 25.0
MARGIN_MIN_UJY = 1.0

# Off-stamp point sources at least this bright keep analytic full-wing
# Moffat components: a rendered kernel stamp truncates, but a bright
# star's wings still reach across the edge.
BRIGHT_PSF_UJY = 100.0


# ------------------------------------
# Stars
# ------------------------------------
# A Gaia source is a confirmed star only with a 5-parameter astrometric
# solution at parallax or proper-motion significance above this. Gaia
# membership alone is not enough -- compact galaxy nuclei are in Gaia.
STAR_ASTROM_SIG = 3.0

STAR_MIN_UJY = 100.0      # fainter confirmed stars keep their catalog component
STAR_PROF_MAX_AS = 45.0   # measured stellar-profile terminus
STAR_RING_MIN_PX = 40     # a profile ring votes only with this many pixels

# A measured profile recovering less than MIN_FRAC of the (color-
# scaled) in-stamp render flux (flux0) is a FAILED measurement (rings
# starved by the target-region and bright-neighbor exclusions -- or the
# "star" is a galaxy with spurious Gaia astrometry). One recovering more
# than MAX_FRAC is a CONTAMINATED measurement (rings sitting in another
# source's light; subtracting it would excavate). Either way the
# source reverts: light must never leave the scene without something
# accounting for it. Thresholds are judged against the render's in-stamp
# flux scaled to the band through BAND_COLOR_COL, so a red star's honest
# faintness in g cannot read as failure.
STAR_PROFILE_MIN_FRAC = 0.8
STAR_PROFILE_MAX_FRAC = 1.3

# A reverted star inside the aperture zone gets NO design column: a
# free point-source column there can absorb target light wholesale
# (and its over-subtraction cannot be masked after the fact). Its
# predicted catalog-amplitude footprint is masked and twin-filled
# instead. Beyond the zone the component stays, with its amplitude
# leashed to the color-scaled catalog expectation.
#
# The floor is ZERO: the expectation is a catalog prediction, and data
# showing no light where the catalog predicts some must be able to say
# so. A positive floor forces predicted light into the scene that the
# stamp rejects, which is the catalog overriding the measurement. The
# ceiling is the real rail -- a stamp wanting MORE than the star can
# provide means the star model is too faint or the light is not the
# star's, and that is the miss worth catching and recording.
STAR_ZONE_BUFFER_AS = 3.0
STAR_REVERT_AMP_BAND = (0.0, 2.0)


# ------------------------------------
# Background: one owner, one estimator
# ------------------------------------
# The background is a plane through sigma-clipped bin MEANS
# (photometry sums pixels; a median mode-locks on the spiked
# histograms real survey frames carry),
# alternating with the amplitude solve until its constant converges. It
# never sits in a design matrix next to component amplitudes.
BIN_AS = 5.0            # bin-grid bin size
# BG_RMIN_AS is the one target/sky boundary; six stages read it (see
# sky_floor). Background bins inside it are excluded.
BG_RMIN_AS = 15.0
BIN_MIN_FRAC = 0.5      # a bin votes only if half its pixels are usable

# Bin-level MAD rejection: bins coherently elevated beyond this many
# sigma of the bin-to-bin scatter are source structure (halo skirts,
# tidal light) and lose their vote. The plane owns cutout-scale
# background only; ownership of light is positional, not statistical.
BG_REJ_SIGMA = 3.0

# Far-field witness: a robust level measured beyond this radius, where
# target and halo light are weakest. Recorded per band as an independent
# zero point -- never fed back into the fit.
FARFIELD_RMIN_AS = 50.0
FARFIELD_MIN_PX = 5000

ALT_MAX_ITER = 4        # background <-> amplitude alternation cap
ALT_TOL_SIGMA = 0.02    # converged: plane constant moves < this x sigma


# ------------------------------------
# Seats and the joint solve
# ------------------------------------
# A "seat" is a component whose shape parameters enter the nonlinear
# solve. Every seat carries SEAT_NPARAMS parameters:
# (size, profile, ellipticity, position angle, dx, dy).
SEAT_NPARAMS = 6

# Sersic index bounds for every seat. render.sersic_extent_px clamps to
# the same range when it sizes a render box.
SERSIC_N_RANGE = (0.4, 6.0)

# Sersic-seat ellipticity ceiling: a lower ceiling clips true edge-on
# disks. Nuker halo seats keep the stricter ceiling -- an envelope that
# flat is not an envelope.
SERSIC_E_MAX = 0.92
NUKER_E_MAX = 0.85

# The gated halo family: Nuker profile with frozen inner slope and break
# sharpness, Gaussian-truncated. Sersic-family outer profiles refuse cD
# envelopes; the data want shallow-with-an-edge.
NUKER_GAMMA = 0.5         # inner slope, frozen
NUKER_ALPHA = 2.0         # break sharpness, frozen
NUKER_TRUNC_AS = 120.0    # truncation scale; must stay above the break-
                          # radius ceiling or the two degenerate
NUKER_RB_AS = (2.0, 85.0)   # break-radius bounds
NUKER_RB0_AS = 15.0         # break-radius seed

# Nuker outer-slope bounds. The floor matters: a slope at the floor with
# the break radius at its ceiling is a flat sheet, not an envelope --
# measured cD envelopes fall like r^-1.6..-2.4, so the floor sits at the
# flat end of that range and a rail there is the witness that a profile
# wants flatter than an envelope.
NUKER_BETA = (1.8, 8.0)

# Center freedom. Halo centers move in a wide box under the ownership
# penalty (see solve): a halo displaced beyond its own break radius is
# not that galaxy's halo. Sersic seats get a small box: pinned centers
# on blends bake in mutual pulls (the observed peak is the true center
# plus the neighbor's slope), and a small box fixes the geometry without
# re-splitting the blend.
DXY_OUT_AS = 8.0
SEAT_DXY_AS = 1.0

# Role enforcement by bounds. A gated core seat is a nucleus, and nuclei
# have scales: without the cap the core can impersonate the envelope --
# globally cheaper in the pixel objective while digging a hole at the
# science aperture.
GATED_CORE_REFF_MAX_AS = 5.0
FREE_SEAT_REFF_MAX_AS = 6.0   # patch free seats (companion nuclei)

# Gated NEIGHBORS get a Sersic core only. The extended Nuker halo is a cD
# envelope model and belongs to the TARGET, granted through the patches'
# "target_halo". On an ordinary bright galaxy the neighbor halo fits to a
# few percent of the light with a railed beta -- a degenerate seat that
# wanders the solve and buys nothing. True gives every gated seat a
# core+halo pair.
GATED_HALO = False
REFIT_REFF_MAX_AS = 10.0      # the standard target-refit seat

PA_BOX_DEG = 95.0       # position-angle freedom about the catalog value
SNAP_BOX_AS = 2.0       # snap-to-peak search box (nearest local maximum)

SOLVE_NFEV = 450        # optimizer budget per COLD shape solve stage
SOLVE_FSCALE = 3.0      # soft-L1 transition, in units of the per-pixel
                        # loss scale below

# Warm re-solves (alternation iterates, transfer-band neighbor seats)
# start at a converged neighborhood and finish in tens of evaluations;
# a warm solve still burning hundreds is fighting a pathology the
# extra budget cannot fix. The reduced cap contains that cost.
SOLVE_NFEV_WARM = 150

# Fractional model-error floor in the shape solve's per-pixel scale:
# scale = sky rms + this x |source counts|. Against a bare scalar sky
# sigma, a bright core's model imperfection reads as thousands of
# sigma; the soft-L1 loss saturates there, gradients flatten, and the
# optimizer burns its whole budget on micro-steps (worst on deep
# stacks with no inverse-variance map). No pixel is trusted beyond a
# few percent of its own flux.
SOLVE_MODEL_ERR_FRAC = 0.02

# Amplitude ceiling, as a multiple of a seat's catalog flux or of a
# fixed component's in-stamp render flux. Pure safety: an unbounded
# degenerate column can solve to astronomically large amplitude on a
# near-zero render, harmless in-band but poisonous to every sibling
# band that leashes against it.
AMP_MAX_X_CAT = 100.0


# ------------------------------------
# Transfer bands
# ------------------------------------
# Bands that consume a reference band's seat shapes bound each seat's
# flux to this window around the color-scaled reference flux. Colors
# come from the owner's own catalog bands (nearest listed column);
# neutral 1.0 when the catalog cannot say.
TRANSFER_AMP_BAND = (0.1, 10.0)
BAND_COLOR_COL = {'u': 'flux_g', 'g': 'flux_g', 'r': 'flux_r',
                  'i': 'flux_z', 'z': 'flux_z', 'y': 'flux_z'}

# Per-instrument reference-band preference: the first available filter
# in this order solves seat shapes for its instrument's other bands.
REFERENCE_PREFERENCE = ('r', 'i', 'z', 'g', 'y', 'u')

# Where a GATING target's transfer bands get their neighbor shapes.
#
# Such a band needs two target shapes: free (harvested to the registry, so
# other fields consume this object's own centered view) and frozen at the
# reference shape (the science flux, so colors stay shape-consistent). Which
# of those two solves settles the NEIGHBOR seats is this knob.
#
# True    (default) the free solve runs first and settles every seat; the
#         science pass adopts its neighbor shapes and re-freezes the target,
#         leaving only amplitudes and background. One nonlinear solve, one
#         neighbor shape per band, stored and used. It is also the treatment a
#         neighbor gets when its shape comes from the registry, so a field's
#         science flux does not depend on whether that neighbor was solved by
#         an earlier field.
# False   each solve fits the neighbors itself. Two nonlinear solves, and the
#         neighbor shapes STORED (from the free pass) are not the ones USED
#         (from the frozen pass).
#
# Only gating targets have two solves to choose between. Elsewhere the
# science pass is the only place a gated neighbor's shape can come from, and
# it solves it either way.
#
# The science flux costs ~0.02 sigma (a gated target whose seated neighbor
# holds 13% of the aperture, across a real color gradient), no amplitude leash
# binds, and the pass is slightly cheaper. The gain is the HARVESTED shape
# other fields consume: on an injected neighbor of r_eff 4.0 px the free solve
# recovers 3.98, against 3.35 when the neighbor is fit around a frozen,
# over-extended target.
NEIGHBOR_SHAPE_FROM_FREE_SOLVE = True


# ------------------------------------
# Cross-field registry
# ------------------------------------
# A registry entry transports a solved source's full per-band
# decomposition (shapes, sky centers, per-band fluxes). Consumers leash
# frozen-component amplitudes tightly -- calibration headroom only. A
# wide leash lets amplitude refits re-park along the amplitude/
# background degeneracy the registry exists to pin; a tight leash around
# a wrong anchor is worse than none, so anchors are the solved per-band
# fluxes, never cross-band guesses.
REGISTRY_AMP_BAND = (0.8, 1.25)
REGISTRY_MATCH_AS = 2.0   # catalog rows within this of an entry are replaced
# A seat anchor drifts across bands (a coarse-pixel PS1 solve can land
# ~0.3" off the deep-band anchor), so keying each band on its own rounded
# position splits one source into near-duplicate entries. Coalesce a
# harvest onto any existing entry within this radius -- comfortably above
# the observed drift, well below the separation of distinct seated sources.
REGISTRY_KEY_COALESCE_AS = 1.0


# ------------------------------------
# Masks and fill
# ------------------------------------
K_ISO = 1.0             # isophote threshold (x sigma), all mask channels
GEO_REFF_FACTOR = 2.5   # intersection-mask geometric cap, x catalog reff
GEO_SEEING_FLOOR = 1.5  # ... with this x seeing floor
FLOOD_MAX_AS = 6.0      # maximum growth of the flood channel

# Radius inside which mask channels may not claim target pixels. Zero
# trusts the twin fill over a neighbor model's core subtraction:
# reconstructing the core from the mirror side of the galaxy stays
# data-true, while exposing the core keeps whatever over- or
# under-subtraction the neighbor model committed there. Set > 0 to keep
# every mask channel off the target inside that radius.
TARGET_MASK_FREE_AS = 0.0


# ------------------------------------
# PSF
# ------------------------------------
# Kernel width in FWHM units. Wide enough that under 0.1% of a
# beta=3 Moffat's flux falls outside the box; the edge taper rolls the
# remainder smoothly to zero so no square boundary survives into the
# rendered scene.
MOFFAT_KERNEL_FWHM = 16.0

# Empirical-PSF star window. The bright limit is the saturation guard;
# the faint limit is only a loose candidate cap -- the peak-S/N floor
# and the ring guards are the real filter, and they scale with each
# stamp's own depth where a fixed magnitude cannot: a deep stack's
# best kernel star is often fainter than any shallow-survey window.
PSF_STAR_GMAG = (15.8, 22.0)

# Profile rings below this S/N hand off to a Moffat wing graft: a faint
# star's measured outer rings are noise, and a monotone-floored noise
# wing is systematically zero.
PSF_WING_SNR = 5.0

# Kernel cleanliness. No PSF candidate within the target's structured
# zone: rings there measure the target's envelope, not the star (and
# the contamination is high-S/N, so the noise-triggered graft cannot
# catch it). A kernel carrying more than the wing-fraction ceiling
# beyond twice its own FWHM is contaminated regardless of source --
# clean PSFs sit near 4% -- and retries with the graft forced at
# PSF_GRAFT_FORCE_AS: measured core, analytic wings.
PSF_EXCLUDE_TARGET_AS = 25.0
PSF_WING_FRAC_MAX = 0.10
PSF_GRAFT_FORCE_AS = 2.5


# ------------------------------------
# Artifacts: mask, never fit
# ------------------------------------
# Catastrophic-pixel gate. A pixel is artifact when it clears the
# brightness floor (the LARGER of ARTIFACT_SIG x sigma above the outer
# level and the absolute ARTIFACT_SB_MIN below) AND exceeds
# ARTIFACT_RATIO x the catalog scene's own claim there AND the scene is
# QUIET there (claim below that same floor): an under-predicted real
# source fails the ratio test, a bright real core -- a cD cusp above its
# smooth model, a saturated star above its clipped catalog flux --
# fails the quiet test, and a bleed trail on quiet sky is orders of
# magnitude past all three. Damage beyond the ratio on quiet parts of
# the core is masked and the coverage gate demotes the band. Regions
# below ARTIFACT_AREA_MIN are left to the flood channel. A soft-edged
# artifact's sub-threshold skirt is taken by a seeded flood (K_ISO
# departure from the ambient surface, unclaimed pixels only,
# FLOOD_MAX_AS growth -- the neighbor-flood constants), so the boundary
# follows the artifact's own light and stops at real sources' claims.
# Masked artifact holes twin-fill exactly like neighbor masks, at every
# radius. Broad structure at the noise scale is background machinery,
# not artifact -- a per-pixel threshold cannot see it and must not try.
ARTIFACT_SIG = 20.0
ARTIFACT_RATIO = 5.0
ARTIFACT_AREA_MIN = 15.0     # arcsec^2

# Absolute brightness floor (uJy/arcsec^2). Condition 1 (20 sigma) is
# depth-RELATIVE: on a deep stack 20 sigma is mu ~ 23, inside ordinary
# astrophysics (galaxy outskirts, cD envelope skirts, star halos), so
# on deep frames the noise floor alone would mask real light. An
# instrument artifact of the bleed/streak class is brighter than the
# SKY itself, not merely brighter than the noise -- depth changes what
# you can see, not what a CCD bleed is. Candidacy takes the LARGER of
# the two floors: real trails run 300+ uJy/as^2 (10x over); a cD rim
# at a few uJy/as^2 never qualifies on any instrument.
ARTIFACT_SB_MIN = 30.0      # uJy/arcsec^2 (mu ~ 20.2)


# ------------------------------------
# Empty-aperture error
# ------------------------------------
# The reported flux error is the robust scatter of EMPAP_N source-free
# aperture sums of the final residual (scene, plane, and mesh all
# subtracted) -- correlation, background structure, and confusion
# included by measurement instead of excluded by assumption. A
# sigma x sqrt(N) propagation understates the truth by an order of
# magnitude on resampled stacks. Where an inverse-variance map exists
# the larger of the two is reported; when the stamp cannot host
# EMPAP_MIN_APS placements the white-noise path stands, labeled as
# such.
EMPAP_N = 200
EMPAP_MIN_APS = 30


# ------------------------------------
# Era stamping
# ------------------------------------
def snapshot() -> dict:
    """Every recipe constant as a JSON-safe dict, for provenance.

    A measured value is only reproducible against the exact recipe that
    produced it, so every measurement sidecar carries this snapshot.
    """
    out = {}
    for name, value in sorted(globals().items()):
        if not name.isupper():
            continue
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, tuple):
            value = list(value)
        out[name] = value
    return out


# ------------------------------------
# Runtime override
# ------------------------------------
@contextlib.contextmanager
def sky_floor(rmin_arcsec: float | None):
    """Scope BG_RMIN_AS -- the target/sky boundary -- to one run.

    BG_RMIN_AS is absolute arcsec, which suits the survey galaxies the
    defaults are tuned for and not a compact source on a 0.05"/pixel HST
    mosaic, where 15" of excluded core is most of the visit. Every
    consumer reads recipe.BG_RMIN_AS at call time -- background bins,
    star exclusion, stamp validity, artifact detection, and both
    empty-aperture placements -- so one override moves them together,
    which is the intent: one boundary, one owner. snapshot() reads
    globals() at call time too, so the sidecar records the value the run
    actually used, and a reconstruction can read it back.

    Parameters
    ----------
    rmin_arcsec : float or None
        Replacement boundary radius; None leaves the default standing.
    """
    global BG_RMIN_AS
    if rmin_arcsec is None:
        yield
        return
    previous = BG_RMIN_AS
    BG_RMIN_AS = float(rmin_arcsec)
    try:
        yield
    finally:
        BG_RMIN_AS = previous
