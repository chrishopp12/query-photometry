"""
remeasure.py

Re-report band fluxes from a stored fit
---------------------------------------------------------
Recompute a galaxy's band fluxes from the IMMUTABLE per-galaxy provenance
sidecar, with no re-fetch and no re-fit. The fit already stored the target
model's curve of growth (PSF-convolved, circular apertures, arcsec radii), so
a different aperture -- or the integrated model total -- is an interpolation on
values already on disk. The provenance is git_rev-pinned, so a re-report is
reproducible even after other galaxies rewrite the registry.

Both --remeasure modes live here and neither rebuilds the scene, because the fit
stored both curves of growth: 'sersic' reads the fitted model's COG (the model
IS the deblended target), 'aperture' the empirical neighbor-subtracted one
(already sky-subtracted and corrected -- it equals the science f_ap at the
measured aperture).
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


# Which stored curve of growth each mode reads (arcsec radii, uJy enclosed).
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

    mode 'sersic' reads the fitted model's curve of growth and extrapolates past
    the grid to the model total; mode 'aperture' reads the empirical, neighbor-
    subtracted curve of growth whose outermost measured value is the total.
    Returns a DataFrame (band, flux_uJy, mag_AB, aperture_as, mode, source);
    bands whose provenance lacks the curve (demoted/unmeasured) are skipped.

    shape selects which target shape the report is built on. 'forced' is the
    instrument's reference-band shape -- the one the science curve was built
    on. 'fitted' is each band's own free-target shape, which exists only for
    a GATING target (the engine solves such a target twice per transfer
    band: once frozen for the science flux, once free for the registry).
    Bands with no free-target record fall back to forced, name themselves in
    the log, and say so in their `source`.

    Where shape applies:
      sersic            reads the free-target model curve directly
                        (model_cog_free_uJy) -- pure interpolation.
      aperture          within the grid, shape is inert: the empirical curve
                        is a measurement, not a rendering of a shape. PAST
                        the grid, reconstruct rebuilds the scene and the
                        shape decides what is subtracted.
    """
    if mode not in _COG_FIELD:
        raise ValueError(f"mode must be one of {sorted(_COG_FIELD)}, got {mode!r}")
    prov = json.loads(Path(provenance_path).read_text())
    rev = prov.get('git_rev', '?')
    integrated = aperture_arcsec is None or aperture_arcsec <= 0
    # Past the stored grid the empirical curve holds no value to read;
    # reconstruct the scene from the pinned fit and integrate at R.
    per_band = prov.get('per_band') or {}
    grid_max = max((((b.get('fit_state') or {}).get('rgrid') or [0.0])[-1]
                    for b in per_band.values()), default=0.0)
    if mode == 'aperture' and not integrated and aperture_arcsec > grid_max:
        recon = reconstruct(provenance_path, float(aperture_arcsec),
                            shape=shape, registry_path=registry_path,
                            write_qa=write_qa)
        return pd.DataFrame(
            [dict(band=band, flux_uJy=round(flux, 4),
                  mag_AB=(round(UJY_AB_ZP - 2.5 * np.log10(flux), 4)
                          if flux > 0 else float('nan')),
                  aperture_as=float(aperture_arcsec), mode='aperture',
                  source=f"reconstruct_{shape}:{rev}")
             for band, flux in recon.items()],
            columns=['band', 'flux_uJy', 'mag_AB', 'aperture_as',
                     'mode', 'source'])
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
    """(params, pix) out of a shapes / solve / solve_free record.

    The three name their grid differently (`pix` vs `pix_ref`) because two
    of them are a solver's own output and one is the engine's statement of
    what it rendered. Read them uniformly.
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
    # free. Position in the record IS the rule: a band-name test would only
    # restate it, and only for instruments that happen to have an r band.
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
        if seat is None:
            continue
        if shape == 'fitted':
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
                write_qa: bool = False) -> dict:
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
    """
    import contextlib
    import io

    from astropy.coordinates import SkyCoord
    from .pipeline import run_measure

    prov = json.loads(Path(provenance_path).read_text())
    tgt = prov.get('target') or {}
    coord = SkyCoord(tgt['ra_deg'], tgt['dec_deg'], unit='deg')
    galaxy_dir = Path(provenance_path).parent.parent
    label = (tgt.get('label')
             or Path(provenance_path).name.split('_measured')[0])
    instruments = [i.lower() for i in (prov.get('instruments') or [])]
    cutout = float(prov.get('cutout_arcsec', 120.0))
    # A run that moved the target/sky boundary must be rebuilt on the same
    # one -- it decides which pixels the plane and mesh ever saw.
    sky_rmin = ((prov.get('scene') or {}).get('recipe') or {}).get('BG_RMIN_AS')
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
                            cutout_arcsec=cutout, sky_rmin_arcsec=sky_rmin,
                            rgrid=grid, pin_by_band=pin,
                            registry_path=registry_path, write_outputs=False,
                            qa_dir=qa_dir)
    out = {row['band']: float(row['flux_uJy']) for _, row in frame.iterrows()}
    # run_measure reports a dead band by PRINTING and moving on, so under the
    # redirect its only trace lands in the discarded buffer. Compare what was
    # asked for against what came back -- that catches every drop mechanism,
    # not just the two the pipeline has messages for -- and replay whatever
    # reasons it did give.
    missing = [band for band in pin if band not in out]
    if missing:
        print(f"  [remeasure] {len(missing)} band(s) dropped by the pinned "
              f"rebuild: {sorted(missing)}")
        for line in log.getvalue().splitlines():
            if 'FAILED' in line or 'no_coverage' in line:
                print(f"    {line.strip()}")
    return out
