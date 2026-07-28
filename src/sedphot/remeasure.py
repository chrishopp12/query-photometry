"""
remeasure.py

Re-report band fluxes from a stored fit
---------------------------------------------------------
Recompute a galaxy's band fluxes from the IMMUTABLE per-galaxy provenance
sidecar. The fit stored the target model's curve of growth (PSF-convolved,
circular apertures, arcsec radii) and the empirical neighbor-subtracted one,
so inside the stored radius grid a different aperture -- or the integrated
model total -- is an interpolation on values already on disk: no fetch, no
scene, no solve.

One request leaves that path. Aperture mode past the end of the stored grid
has no value left to interpolate, so `reconstruct` rebuilds the scene from
the pinned fit and integrates at R. That path re-reads the cached images
(fetching any that are missing), rebuilds the scene, and falls back to
SOLVING any band the sidecar cannot pin; such bands come back labeled
`solved_*` rather than `reconstruct_*`.

Re-reporting is stable against a registry that other galaxies keep
rewriting, because the sidecar carries its own registry_consumed snapshot
and that is read before the live registry. git_rev records which source tree
wrote the sidecar; it is a label, not a lock.

Both --mode values live here: 'sersic' reads the fitted model's COG (the
model IS the deblended target), 'aperture' the empirical neighbor-subtracted
one (already sky-subtracted and corrected -- it equals the science f_ap at
the measured aperture).
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


# Valid --mode values. _cog_source picks the exact field per band; this dict
# only whitelists the mode names.
_COG_FIELD = {'sersic': 'model_cog_uJy', 'aperture': 'enclosed_uJy'}


def _cog_source(mode: str, shape: str, fit_state: dict) -> tuple[str, str]:
    """(curve field, total field) for one band under a mode/shape request.

    'fitted' asks for the per-band FREE-target solve's own curve. Only a
    gating target's bands stored one -- the forced curve stands everywhere
    else, and the caller reports which bands fell back.

    'aperture' takes no shape here: the empirical curve is a MEASUREMENT of
    the pixels, not a rendering of a chosen shape, so there is no per-shape
    variant to pick. Shape reaches the empirical path only PAST the grid,
    where reconstruct rebuilds the scene and the target shape decides what
    gets subtracted from the aperture.
    """
    if mode == 'aperture':
        return 'enclosed_uJy', ''
    if shape == 'fitted' and fit_state.get('model_cog_free_uJy'):
        return 'model_cog_free_uJy', 'target_model_free_uJy'
    return 'model_cog_uJy', 'target_model_uJy'


def remeasure(provenance_path: str | Path,
              aperture_arcsec: float | None = None,
              mode: str = 'sersic',
              shape: str = 'forced',
              registry_path: str | None = None,
              write_qa: bool = False) -> pd.DataFrame:
    """Per-band fluxes at aperture_arcsec (None/<=0 = integrated), from a fit.

    Reads the stored curves of growth and reports one row per band. What you
    get depends on the mode and on whether the aperture is inside the stored
    radius grid:

      sersic,   inside grid   interpolate the fitted model's COG.
      sersic,   past grid     the model total -- the model has converged, so
                              every radius past the grid returns the same
                              number while aperture_as records what was asked.
      sersic,   integrated    the model total.
      aperture, inside grid   interpolate the empirical neighbor-subtracted COG.
      aperture, past grid     no stored value to read: `reconstruct` rebuilds
                              the scene from the pinned fit and integrates at
                              R. That re-reads the cached images (fetching any
                              that are missing) and SOLVES any band the sidecar
                              cannot pin.
      aperture, integrated    the outermost stored point, reported with
                              aperture_as = inf. There is no empirical total.

    'past grid' is judged against the SHORTEST grid in the sidecar, so it is a
    property of the galaxy, not of one band.

    shape selects which target shape the report is built on. 'forced' is the
    instrument's reference-band shape -- the one the science curve was built
    on. 'fitted' is each band's own free-target shape, which exists only for
    a GATING target (the engine solves such a target twice per transfer
    band: once frozen for the science flux, once free for the registry).
    Bands with no free-target record fall back to forced, name themselves in
    the log, and say so in their `source`.

    shape acts on sersic mode, and on aperture mode ONLY past the grid, where
    the target shape decides what gets subtracted from the aperture. Inside
    the grid under aperture mode it is accepted and ignored -- the empirical
    curve is a measurement of the pixels, not a rendering of a shape -- and
    anything but the default 'forced' says so on stdout, since a request
    that spans the grid boundary is meaningful on the far side.

    Returns
    -------
    report : pd.DataFrame
        (band, flux_uJy, mag_AB, aperture_as, mode, source). `source` records
        the shape actually used and distinguishes an interpolation
        (`*_remeasure`) from a rebuild (`reconstruct_*`) or a fresh fit
        (`solved_*`). Inside the grid a band whose provenance lacks the curve
        is skipped; past the grid it is solved and returned instead.
    """
    if mode not in _COG_FIELD:
        raise ValueError(f"mode must be one of {sorted(_COG_FIELD)}, got {mode!r}")
    prov = json.loads(Path(provenance_path).read_text(encoding='utf-8'))
    rev = prov.get('git_rev', '?')
    integrated = aperture_arcsec is None or aperture_arcsec <= 0
    # Past the stored grid the empirical curve holds no value to read;
    # reconstruct the scene from the pinned fit and integrate at R.
    per_band = prov.get('per_band') or {}
    # The SHORTEST stored grid decides for the whole galaxy: past its end a
    # band has no value left to read and would silently return its capped
    # outermost point. One band whose grid stopped early therefore sends
    # every band down the reconstruction path.
    ends = [(b.get('fit_state') or {}).get('rgrid')[-1]
            for b in per_band.values()
            if (b.get('fit_state') or {}).get('rgrid')]
    grid_max = min(ends) if ends else 0.0
    if mode == 'aperture' and not integrated and aperture_arcsec > grid_max:
        # The boundary is worth a line: on one side this is a millisecond
        # interpolation of values already on disk, on the other it re-reads
        # the images, rebuilds the whole scene, and may solve.
        print(f"  [remeasure] aperture {float(aperture_arcsec):g}\" is past "
              f"the stored curve of growth (ends at {float(grid_max):g}\"); "
              f"rebuilding the scene from the pinned fit instead of "
              f"interpolating")
        state: dict = {}
        recon = reconstruct(provenance_path, float(aperture_arcsec),
                            shape=shape, registry_path=registry_path,
                            write_qa=write_qa, status=state)
        solved = set(state.get('solved') or ())
        return pd.DataFrame(
            [dict(band=band, flux_uJy=round(flux, 4),
                  mag_AB=(round(UJY_AB_ZP - 2.5 * np.log10(flux), 4)
                          if flux > 0 else float('nan')),
                  aperture_as=float(aperture_arcsec), mode='aperture',
                  source=(f"solved_{shape}:{rev}" if band in solved
                          else f"reconstruct_{shape}:{rev}"))
             for band, flux in recon.items()],
            columns=['band', 'flux_uJy', 'mag_AB', 'aperture_as',
                     'mode', 'source'])
    # Everything below is the inside-the-grid path, where shape has nothing
    # to act on. The measure verb REFUSES the analogous combination; a
    # warning is right here because the same request past the grid IS
    # meaningful, so refusing would block one that spans both.
    if mode == 'aperture' and shape != 'forced':
        print(f"  [remeasure] WARNING: shape={shape!r} has no effect under "
              f"--mode aperture inside the stored grid (ends at "
              f"{float(grid_max):g}\"): the empirical curve is a measurement "
              f"of the pixels, not a rendering of a shape")
    rows = []
    demoted: list[str] = []
    for band, b in per_band.items():
        fs = b.get('fit_state') or {}
        field, total_key = _cog_source(mode, shape, fs)
        rgrid, cog = fs.get('rgrid'), fs.get(field)
        if not rgrid or not cog:
            continue
        used = shape
        if mode == 'sersic' and shape == 'fitted' \
                and field != 'model_cog_free_uJy':
            demoted.append(band)
            used = 'forced'
        total = b.get(total_key) if total_key else float(cog[-1])
        if total is None:
            continue
        flux = model_flux_within(aperture_arcsec, rgrid, cog, total)
        # The source records the shape actually used, not the one requested,
        # so a demoted band is visible in the table itself.
        tag = f"{mode}_{used}" if mode == 'sersic' else mode
        rows.append(dict(
            band=band,
            flux_uJy=round(flux, 4),
            mag_AB=(round(UJY_AB_ZP - 2.5 * np.log10(flux), 4)
                    if flux > 0 else float('nan')),
            aperture_as=(float('inf') if integrated else float(aperture_arcsec)),
            mode=mode,
            source=f"{tag}_remeasure:{rev}"))
    if demoted:
        print(f"  [remeasure] no per-band free-target model stored for "
              f"{sorted(demoted)}; those bands report the forced "
              f"(reference-band) shape")
    return pd.DataFrame(rows, columns=['band', 'flux_uJy', 'mag_AB',
                                       'aperture_as', 'mode', 'source'])


def _vector(record: dict | None) -> tuple:
    """(params, pix) out of a `shapes`, `solve`, or `solve_free` record.

    The grid is named `pix` on `shapes` and `pix_ref` on `solve`; either is
    accepted. Returns (None, None) when the record has no params.
    """
    params = (record or {}).get('params')
    if not params:
        return None, None
    return params, (record.get('pix') if record.get('pix') is not None
                    else record.get('pix_ref'))


def _build_pin_by_band(prov: dict, shape: str = 'forced') -> dict:
    """Per-band pin dict from the sidecar (owner->amp, shape, plane, mesh).

    'forced' rebuilds each band on the shapes it was MEASURED with -- the
    ones the science curve came from. 'fitted' substitutes each band's own
    free-target vector, which only a gating target has, falling back to
    forced (loudly) where a band stored none.

    Each pin carries seat_pix, the arcsec/px grid its radial parameters live
    in, so pinned_fit can rescale onto the band it re-renders on.

    `shapes` is the authoritative source: the whole seat list, on the band's
    own grid. Sidecars predating it fall back to the instrument's reference
    band `solve` record -- correct, because a transfer band's target was
    frozen at exactly that shape. What must never be used is a transfer
    band's OWN `solve`: it covers the free (neighbor) seats alone, so a
    length check refuses it.
    """
    per_band = prov.get('per_band') or {}
    # per_band preserves engine.order_bands order, so the FIRST band of each
    # instrument is its reference -- the one band that solved every seat
    # free. Position in the record is the rule; there is no band-name test.
    ref: dict = {}
    for band, b in per_band.items():
        params, pix = _vector(b.get('solve'))
        inst = band.split('_')[0]
        if params and inst not in ref:
            ref[inst] = (band, params, pix)

    pin: dict = {}
    demoted: list[str] = []
    for band, b in per_band.items():
        fs = b.get('fit_state') or {}
        if not fs.get('amps') or not fs.get('bg_coefs'):
            continue
        forced = ref.get(band.split('_')[0])
        # The shapes this band was measured with, else the reference band's.
        seat, seat_pix = _vector(b.get('shapes'))
        if seat is None and forced is not None:
            _, seat, seat_pix = forced
        if seat is None and b.get('seat_owners') == []:
            # A scene with NO seats needs no shape vector: pinned_fit renders
            # the fixed components at their stored amplitudes on the stored
            # plane, which is the whole fit. Pinnable, not a gap.
            seat, seat_pix = None, None
        elif seat is None:
            # Seats existed and no vector covers them, so this band cannot be
            # rebuilt. Leaving it out of the pin sends it down the SOLVING
            # path, which is a fresh measurement wearing a reconstruction
            # label -- the caller reports it instead.
            continue
        if shape == 'fitted' and seat is not None:
            own = _fitted_vector(band, b, forced, len(seat))
            if own is None:
                demoted.append(band)
            else:
                seat, seat_pix = own
        pin[band] = dict(seat_params=seat, seat_pix=seat_pix,
                         amps=fs['amps'], bg_coefs=fs['bg_coefs'],
                         mesh=fs.get('mesh'),
                         consumed=b.get('registry_consumed'))
    if demoted:
        print(f"  [remeasure] no per-band free-target shape stored for "
              f"{sorted(demoted)}; those bands use the shapes they were "
              f"measured with")
    return pin


def _fitted_vector(band: str, witness: dict, forced: tuple | None,
                   expect: int) -> tuple | None:
    """This band's own free-target shape vector, or None when it has none.

    The reference band solved every seat free, so the shapes it measured
    with ARE the free ones. On a transfer band the science pass had the
    target frozen, so only a gating target's separate free-target solve has
    one, recorded as `solve_free`. Either way the vector must cover the
    whole seat list.
    """
    if forced is not None and band == forced[0]:
        params, pix = _vector(witness.get('shapes'))
        if params is None:
            params, pix = _vector(witness.get('solve'))
    else:
        params, pix = _vector(witness.get('solve_free'))
    if not params or len(params) != expect:
        return None
    return params, pix


def reconstruct(provenance_path: str | Path, aperture_arcsec: float,
                shape: str = 'forced', registry_path: str | None = None,
                write_qa: bool = False,
                status: dict | None = None) -> dict:
    """Empirical flux at aperture_arcsec, past the stored grid, no solve.

    Rebuilds the galaxy's scene from the immutable sidecar (every shape,
    amplitude, and plane coefficient pinned; consumed neighbors from the
    sidecar's own snapshot, catalog neighbors from the cached catalog, the
    target/sky boundary from its recipe snapshot), re-renders, and
    integrates the aperture at R. Reuses the measurement
    pipeline in its pinned, no-write mode, so the science-aperture products
    are never touched. write_qa writes per-band scene figures to a scoped
    QA/remeasure_R<N>as/ subdir (never the science QA). Returns
    {band: flux_uJy}.

    status, when given, is filled with 'solved': the bands that had no
    pinnable fit and so were SOLVED rather than reconstructed. They are
    still returned -- best available answer -- but a caller that labels
    its output must not call them pinned.
    """
    import contextlib
    import io

    from astropy.coordinates import SkyCoord
    from .catalogs.legacy import LEGACY_DR_DEFAULT
    from .pipeline import run_measure

    prov = json.loads(Path(provenance_path).read_text(encoding='utf-8'))
    tgt = prov.get('target') or {}
    if tgt.get('ra_deg') is None or tgt.get('dec_deg') is None:
        raise ValueError(f"{provenance_path} records no target position, so "
                         f"there is nothing to rebuild the scene around")
    coord = SkyCoord(tgt['ra_deg'], tgt['dec_deg'], unit='deg')
    galaxy_dir = Path(provenance_path).parent.parent
    label = (tgt.get('label')
             or Path(provenance_path).name.split('_measured')[0])
    instruments = [i.lower() for i in (prov.get('instruments') or [])]
    cutout = float(prov.get('cutout_arcsec', 120.0))
    # A run that moved the target/sky boundary must be rebuilt on the same
    # one -- it decides which pixels the plane and mesh ever saw.
    sky_rmin = ((prov.get('scene') or {}).get('recipe') or {}).get('BG_RMIN_AS')
    # Replay the FETCH options too. They decide which pixels arrive, so a
    # default here silently rebuilds on different data: `bricks` swaps Legacy
    # brick coadds (with real inverse variance) for viewer cutouts, on a
    # different pixel grid. Reproducing a fit means reproducing its images.
    legacy = prov.get('legacy') or {}
    legacy_dr = legacy.get('dr') or LEGACY_DR_DEFAULT
    legacy_bricks = bool(legacy.get('bricks'))
    hst_proposal_id = prov.get('hst_proposal_id')
    # The SCENE belongs to the aperture the fit was built for: the
    # target-substructure rule and the star zone are both scoped to it,
    # so asking for a larger radius here would delete catalog rows as
    # 'target shreds' that the fit had modeled and subtracted.
    science_ap = float(prov.get('aperture_arcsec') or aperture_arcsec)
    if aperture_arcsec > cutout / 2.0:
        raise ValueError(
            f"aperture {aperture_arcsec:g}\" exceeds the stamp half-width "
            f"({cutout / 2.0:g}\"); re-fetch a larger cutout to reach it")
    pin = _build_pin_by_band(prov, shape)
    if not pin:
        raise ValueError("sidecar carries no pinnable fit for any band")
    if registry_path is None:
        cand = galaxy_dir.parent / 'registry.json'
        registry_path = str(cand) if cand.exists() else None
    grid = list(np.arange(2.0, float(np.ceil(aperture_arcsec)) + 2.0, 1.0))
    qa_dir = (galaxy_dir / 'Photometry' / 'QA'
              / f"remeasure_R{int(np.ceil(aperture_arcsec))}as"
              if write_qa else None)
    # The pinned pass reuses the measurement pipeline, whose progress log
    # is noise for a re-report; keep remeasure's output the table alone.
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        frame = run_measure(coord, label, str(galaxy_dir),
                            instruments=instruments,
                            aperture_arcsec=aperture_arcsec,
                            cutout_arcsec=cutout,
                            scene_aperture_arcsec=science_ap,
                            sky_rmin_arcsec=sky_rmin,
                            rgrid=grid, pin_by_band=pin,
                            legacy_dr=legacy_dr, legacy_bricks=legacy_bricks,
                            hst_proposal_id=hst_proposal_id,
                            registry_path=registry_path, write_outputs=False,
                            qa_dir=qa_dir)
    out = {row['band']: float(row['flux_uJy']) for _, row in frame.iterrows()}
    # A band with no pin took run_measure's SOLVING path, so it is a fresh
    # fit. Kept as the best available answer, but labeled separately.
    solved = sorted(band for band in out if band not in pin)
    if status is not None:
        status['solved'] = solved
    if solved:
        print(f"  [remeasure] {len(solved)} band(s) had no pinnable fit and "
              f"were SOLVED, not reconstructed: {solved}")
    # Compare the sidecar's band list against what came back: run_measure
    # reports a dead band by PRINTING and moving on, so under the redirect its
    # message is in the discarded buffer. The comparison catches every drop
    # mechanism, not only the ones the pipeline prints for; the replay below
    # recovers whatever reasons it did give.
    missing = [band for band in (prov.get('per_band') or {}) if band not in out]
    if missing:
        print(f"  [remeasure] {len(missing)} band(s) absent from the rebuild: "
              f"{sorted(missing)}")
        for line in log.getvalue().splitlines():
            if 'FAILED' in line or 'no_coverage' in line:
                print(f"    {line.strip()}")
    return out
