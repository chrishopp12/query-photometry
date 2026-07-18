"""
recipe.py

Scene-Engine Recipe Constants
---------------------------------------------------------
Every science knob of the scene measurement engine in one place. The
measurement recipe is: build a scene from the survey catalog, subtract
measured stars, jointly solve component amplitudes (and shapes, where the
catalog declares misfit) against a bin-median-plane background, then mask,
fill, and integrate a curve of growth to the aperture flux.

Notes:
    Distances are arcsec, fluxes microjansky (uJy), surface brightness
    uJy/arcsec^2 unless a name says otherwise. Bound pairs are (low, high).
    The aperture radius and stamp size are runtime parameters (CLI), not
    constants; witness windows that depend on the aperture derive from it.
"""
from __future__ import annotations

import numpy as np


# ------------------------------------
# Curve of growth, witnesses, coverage
# ------------------------------------
# Curve-of-growth radii (arcsec). Witness radii are interpolated on the
# curve, so the grid does not need to contain them exactly.
DEFAULT_RGRID = np.arange(2.0, 30.0, 1.0)

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
# flux is worse than a refused one. The seeing-scale core is gated
# absolutely -- no fill can reconstruct a clipped peak.
COVERAGE_MIN = 0.95


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
TRACTOR_MIN_NMGY = 0.5     # r-band flux floor of the Tractor scene query
TARGET_MATCH_AS = 1.5      # catalog row within this of the request = target

# The halo gate. A gated source receives a shape solve (Sersic core +
# Nuker halo) instead of a fixed catalog profile. The second profile
# exists to fix MISFIT, and the catalog's own reduced chi-square is its
# misfit statement -- the necessary condition. Point sources, the
# target itself, and rows beyond the stamp half-width never gate: a
# shape solve needs its source's pixels on the stamp, a distant halo
# seat with center freedom degenerates into a flat sheet across the
# field, and a RADIAL reach keeps the gate census identical on every
# instrument (a square-stamp test would admit corner sources on a
# rotated grid that an aligned grid excludes).
GATE_FLUX_UJY = 100.0
GATE_RCHISQ = 4.0

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


# ------------------------------------
# Background: one owner, one estimator
# ------------------------------------
# The background is a plane through sigma-clipped bin medians,
# alternating with the amplitude solve until its constant converges. It
# never sits in a design matrix next to component amplitudes.
BIN_AS = 5.0            # median-grid bin size
BG_RMIN_AS = 15.0       # bins inside this target radius are excluded
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
# measured cD envelopes fall like r^-1.6..-2.4, so a rail at the floor
# is the witness that a profile wants flatter than any physical halo.
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
REFIT_REFF_MAX_AS = 10.0      # the standard target-refit seat

PA_BOX_DEG = 95.0       # position-angle freedom about the catalog value
SNAP_BOX_AS = 2.0       # snap-to-peak search box (nearest local maximum)

SOLVE_NFEV = 450        # optimizer budget per shape solve
SOLVE_FSCALE = 3.0      # soft-L1 scale, in units of the pixel sigma

# Amplitude ceiling for fixed components and reference-band seats, as a
# multiple of the catalog expectation. Pure safety: an unbounded
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
                  'i': 'flux_z', 'z': 'flux_z'}

# Per-instrument reference-band preference: the first available filter
# in this order solves seat shapes for its instrument's other bands.
REFERENCE_PREFERENCE = ('r', 'i', 'z', 'g', 'y', 'u')


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
# under-subtraction the neighbor model committed there. Set > 0 to
# restore a mask-free core.
TARGET_MASK_FREE_AS = 0.0


# ------------------------------------
# PSF
# ------------------------------------
MOFFAT_KERNEL_FWHM = 8.0   # kernel half-extent in FWHM units

# Empirical-PSF star window: bright enough for measurable wings, faint
# enough to avoid saturated cores.
PSF_STAR_GMAG = (15.8, 19.5)

# Profile rings below this S/N hand off to a Moffat wing graft: a faint
# star's measured outer rings are noise, and a monotone-floored noise
# wing is systematically zero.
PSF_WING_SNR = 5.0


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
