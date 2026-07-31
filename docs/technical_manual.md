# sedphot Technical Manual

For a reader who will extend, debug, or audit the package. It assumes astronomy,
not this codebase. Paths are repository-relative.

**Contents**

1. [Overview](#1-overview)
2. [Glossary](#2-glossary)
3. [Architecture](#3-architecture)
4. [Module reference](#4-module-reference)
5. [The output contract](#5-the-output-contract)
6. [Extension points](#6-extension-points)
7. [Behavior worth knowing](#7-behavior-worth-knowing)
8. [Testing](#8-testing)

---

## 1. Overview

`sedphot` turns a galaxy name or position into SED-ready photometry tables. It has
two halves that share one table schema:

- **Retrieval** (`catalogs/`) — closest-source photometry from seven public
  archives, one row per band per provider.
- **Measurement** (`images/` + `measure/`) — survey cutouts fetched from five
  archives and measured through one identical scene engine, producing either a
  curve-of-growth aperture flux or a forced-model flux.

A third, independent path (`spherex.py`) submits an IRSA forced-photometry job and
stores the raw per-visit spectrophotometry table verbatim.

Everything is a subcommand of one CLI (`src/sedphot/cli.py`), and each verb maps to
one driver in `src/sedphot/pipeline.py`:

| Verb | Driver | Produces |
|---|---|---|
| `resolve` | `resolve.resolve_target` | prints position + output label |
| `catalogs` | `run_catalogs` | `<label>_catalog.csv` |
| `measure` | `run_measure` | `<label>_measured.csv` + QA figures |
| `spherex` | `run_spherex` | `SPHEREx/table_photometry.<tag>.csv` |
| `sed` | `run_sed` | `<label>_sed.png` |
| `overlay` | `run_overlay` | `<label>_overlay.png` |
| `remeasure` | `remeasure.remeasure` | re-reported fluxes from a stored fit |
| `run` | `run_all` | catalogs → measure → SPHEREx (opt-in) → SED |

Every product lands under `<out-dir>/Photometry/`, alongside a per-product JSON
provenance sidecar and a per-provider `coverage_*.json` status report. Fluxes are
microjansky, AB, statistical errors only, as-measured unless `--dered` is passed.

The measurement path is where the complexity is. One band's flux is produced by:
cutting a stamp, resolving a PSF, rendering every survey-catalog neighbor as a
model component, replacing confirmed stars with their own measured profiles,
solving all component amplitudes (and, where warranted, shapes) jointly against a
background plane, masking whatever residual neighbor light remains, reconstructing
the masked pixels from their mirror through the target, and integrating a curve of
growth. Every step writes a machine-readable witness.

---

## 2. Glossary

The package carries a large private vocabulary. Read this before the deep sections.

### The scene model

| Term | Owner | Meaning |
|---|---|---|
| **scene** | `measure/engine.py` | Three distinct senses. (1) The `scene` **dict** from `prepare_scene`: the per-galaxy inputs (survey catalog, confirmed stars, patches, registry). (2) `scene_img` inside `measure_band`: the **fitted model image** — every component at its solved amplitude, background excluded. (3) `measurement['scene']`: **model plus background**, which is what the QA figure draws. |
| **component** | `measure/components.py` | One rendered catalog row: `name`, `irow`, `cat` (catalog flux, µJy), `x`/`y`, `gate`, `base` (rendered image at catalog shape and amplitude), `flux0` (in-stamp flux of `base`), `shape`. The currency of the whole engine. |
| **seat** | `measure/seats.py` | A component whose **shape parameters enter the nonlinear solve**. Six parameters per seat (`recipe.SEAT_NPARAMS`): size, profile index, ellipticity, PA, dx, dy. Two kinds: `sersic` and `nuker`. A seat replaces its owner's fixed catalog column. |
| **gate / gated / gating** | `measure/components.py` `gated_row` | A catalog row **gates** when it is bright *and* the survey catalog's own reduced chi-square declares misfit (`recipe.GATE_FLUX_UJY`, `GATE_RCHISQ`), in any optical band, below the veto ceiling `GATE_RCHISQ_MAX`. A gated component earns a seat instead of a fixed profile. A **gating target** is a target whose own row would gate if it appeared as a neighbor elsewhere (`engine._target_gates`) — that triggers the per-band free solve whose shape the registry harvests. |
| **target system** | `measure/engine.py` `_system_names` | Components declared (via `patches.json` `target_system`) to be the target's *own* light — a dumbbell's second nucleus, a bound companion. A system member keeps its own component and seat, but its fitted model is attributed to the target: never subtracted, never masked, integrated into the aperture flux, and frozen with the target on transfer bands (`solve._target_side`). |
| **neighbor** | throughout | Any fitted component that is not in the target system. Neighbor light is subtracted before the curve of growth and masked where the model may be locally wrong. |
| **blind measurement** | `measure/engine.py` | A position with no survey-catalog rows. No components, no masks, no stars — background and curve of growth only. Flagged `scene=none`. |

### Bands and shape transfer

| Term | Owner | Meaning |
|---|---|---|
| **reference band** | `measure/engine.py` `order_bands` | The first filter of `recipe.REFERENCE_PREFERENCE` (`r, i, z, g, y, u`) an instrument has. It solves every seat shape free; its siblings inherit them. Per instrument — nothing crosses instruments but the catalogs and patches. |
| **transfer band** | `measure/solve.py` | Any non-reference band of an instrument. It reuses the reference band's seat list verbatim, freezes the target-system seats at the reference shape, and re-solves neighbor seats warm, with each seat's amplitude leashed to a color-scaled reference flux. |
| **frozen** | `measure/solve.py`, `measure/seats.py` | Held at a stored shape and not solved. Applies to a transfer band's target seats, and to a registry-consumed component (which is a fixed column with a tight amplitude leash). |
| **free** | `measure/solve.py` | Solved rather than held. `free_target=True` runs the pass in which *every* seat including the target is solved — the per-band shape a gating target harvests to the registry. `free_idx` is the index list of seats a transfer band actually solves. |
| **leash** | `measure/solve.py` | An amplitude bounded to a window around an *expected* flux instead of the generic sanity ceiling. Sources of an expectation: a star's color-scaled catalog flux (`recipe.STAR_REVERT_AMP_BAND`), a transfer band's color-scaled reference solution (`TRANSFER_AMP_BAND`), a registry entry's stored per-band flux (`REGISTRY_AMP_BAND`). Enforcement is a hard bound in `lsq_linear`, not a penalty: a forbidden amplitude parks exactly on the bound and the residual it would have absorbed is redistributed into the other columns and the background. An amplitude sitting on a bound is recorded as a witness (`leash_bound` names, `leash_detail` records, flag tokens `leash=N` / `leashhi=N`). A **zero** lower bound is the non-negativity floor every column carries, not a leash — an amplitude solving to zero against it is the data answering "no light", and it is not counted. |
| **`forced`** | two senses | (1) **Engine**: forced-photometry mode — a caller-supplied sky shape pins the target profile (`--sersic-params` / `--sersic-from` under `--mode sersic`, via `engine._pin_target`); the amplitude stays free. (2) **`remeasure --shape forced`**: report on the instrument's *reference-band* shape, the one the science curve was built on. These are unrelated mechanisms that share a word. |
| **`fitted`** | `remeasure.py` | `--shape fitted`: report on each band's own free-target shape. Only a gating target's transfer bands store one (`solve_free`); every other band falls back to `forced` and says so in the log and in its `source` string. |

### The cross-field registry

| Term | Owner | Meaning |
|---|---|---|
| **registry** | three senses | (1) The **provider dicts** `IMAGE_PROVIDERS` / `CATALOG_PROVIDERS` — module-level name→callable maps, also the `--all` run order. (2) The **CADC/IVOA service registry** that `astroquery.cadc` resolves its endpoints through (a 503 there blocks the CFHT provider entirely). (3) The **cross-field shape registry**: a JSON file of solved shared sources, in sky units, reusable across galaxies. Only sense (3) is meant when the docs say "the registry" unqualified. |
| **harvest** | `measure/seats.py` `harvest_seats` | Writing a solved seat's shape, sky center, and per-band flux **into** the registry. Only with `--registry-update`. |
| **consume** | `measure/seats.py` `apply_registry` | Reading a registry entry back **into** a later field: matched catalog rows are dropped, and the stored components are added as fixed, tightly-leashed columns with no gate. |
| **vantage** | `measure/seats.py` | The grade of a stored record. `target` — written by the source's own field, the one centered view; never overwritten by a neighbor solve. `neighbor` — written by a field that saw the source off-center; replaced when the home field runs. Vantage governs *write priority*, not consumption strength. |
| **tombstone** | `measure/seats.py` | A per-band marker meaning "the solve burned its evaluation cap here." Consumers kill the gate (the catalog render stands) so a doomed solve is not repeated field after field; only the home vantage retries. |
| **protect rule** | `measure/seats.py` `apply_registry` | An entry anchored on the consuming field's own target (or a declared system member) is **never** consumed. That field measures it fresh, and its harvest upgrades the entry. The one unfreeze. |

### Reconstruction

| Term | Owner | Meaning |
|---|---|---|
| **witness** | `measure/aperture.py` `witness_row` | A measured diagnostic recorded next to the flux. The witnesses are the reproducibility mechanism: nothing is eyeballed. The decisive few ride the table's `flags` column; the whole dict rides the provenance sidecar. |
| **pin / pinned** | `measure/solve.py` `pinned_fit` | Rebuilding a stored fit with **no solve at all**: seats render at the stored shape vector (rescaled onto the band's own pixel grid), amplitudes come from the sidecar in recorded order, the plane is re-evaluated from its stored coefficients, the mesh from its stored grid. `run_measure(pin_by_band=...)` is the hook. |
| **reconstruct** | `remeasure.py` `reconstruct` | The beyond-grid path: rebuild the whole scene from the immutable sidecar in pinned, no-write mode and integrate at a new radius. |

### Background and measurement

| Term | Owner | Meaning |
|---|---|---|
| **plane** | `measure/background.py` `bin_plane` | *The* background: a plane fit through sigma-clipped bin **means**, with bin-level MAD rejection. Owns level and tilt across the cutout, nothing sharper. Never sits in a design matrix beside component amplitudes — it alternates with the amplitude solve. |
| **mesh** | `measure/background.py` `residual_mesh` | The post-fit residual surface: bin levels of what no model claimed, one-bin Gaussian smoothed, DC-zeroed over the plane's own accepted-bin territory, subtracted only inside the curve of growth and never fed back into a fit. Structure sharper than about two bins is invisible to it by construction. |
| **ambient surface** | `measure/background.py` `ambient_surface` | A smoothed bin-mean surface on the same grid, with masked pixels also barred from voting and **no plane fit**. Consumers compare each pixel to the local ambient level: the flood mask channel and the twin fill's asymmetry correction. |
| **artifact** | `measure/artifacts.py` | A catastrophic-pixel region — bleed trails, satellite streaks. Masked, **never fit**. Candidacy needs three conditions at once: above the larger of a relative (`ARTIFACT_SIG`×σ) and an absolute (`ARTIFACT_SB_MIN` µJy/arcsec²) floor, above `ARTIFACT_RATIO`× the catalog scene's claim there, *and* the scene quiet there. |
| **flood** | `measure/aperture.py` `build_mask`, `measure/artifacts.py` | A seeded mask-growth channel: existing mask islands grow into contiguous pixels whose data departs from the ambient surface and that the scene does not claim. It cannot invent masks — only extend them, bounded by `FLOOD_MAX_AS`. Catches glow the catalog never admitted. |
| **twin fill** | `measure/aperture.py` `twin_fill` | Reconstruction of masked and blank aperture pixels from their **point reflection through the target center**, corrected by the odd part of the ambient surface, and clamped between the mirror value and the model fill (fitted target + background) so holes are impossible by construction. `twinfrac` is the fraction of masked aperture pixels that had a valid mirror. |
| **shred** | `measure/components.py` `drop_target_shreds` | A catalog row inside the science aperture whose `fracflux_r` exceeds `recipe.SHRED_FRACFLUX` — the catalog's rendering of the target's own substructure, not an independent source. Such rows **leave the scene entirely** and their light is measured as target flux. Scoped to the aperture; patch-named positions are exempt. |
| **plateau** | `measure/aperture.py` `plateau_hold` | The first curve-of-growth radius where the increments go quiet for `PLATEAU_RUN` consecutive steps **and hold** to the grid end. `-1` when the curve never certifies. Reported as `conv=`. |
| **star zone** | `measure/recipe.py` `STAR_ZONE_BUFFER_AS` | The disk of radius `aperture + 3″` around the target. A star whose profile measurement failed inside it gets **no design column** (a free point-source column there absorbs target light); its predicted footprint is masked and twin-filled instead. |
| **treated star** | `measure/stars.py` | A confirmed star routed by geometry alone. Inside the star zone it is *masked* and twin-filled with no design column (`mode: masked`); outside it keeps its catalog component with a leashed amplitude and `star_reverted=True` (`mode: leashed`), which makes `build_mask` use its full uncapped model isophote. A star is never subtracted from the data: both routes bound the damage a wrong star model can do. |
| **demote** | `measure/stamp.py` `check_coverage` | Refusing a band as `no_coverage` rather than reporting a biased flux. Three triggers: aperture coverage below `COVERAGE_MIN = 0.95`, any dead pixel inside `PEAK_PROTECT_AS = 0.5″`, or seeing-core mask fraction above `CORE_MASKFRAC_MAX = 0.10`. |

### Ambiguous words, resolved

| Word | Senses in this codebase |
|---|---|
| **reference** | (1) The instrument's **reference band**. (2) The catalog **`flux_r`** column, denominator of every color factor in `engine._band_colors` / `_seat_colors`. (3) A registry record's **`flux_ref`** — that seat's solved flux in that band, the anchor a consumer's amplitude leash is built around. Also **`pix_ref`**, the pixel grid a stored radial parameter vector lives in. |
| **registry** | Provider dict / CADC service registry / cross-field shape registry. |
| **scene** | Inputs dict / fitted-model image / model-plus-background. |
| **forced** | Caller-supplied target shape (engine) / reference-band shape (`remeasure --shape forced`). |
| **`free_seats`** vs **`free_target`** | `free_seats` is a **patches.json key** granting a named companion its own Sersic seat. `free_target` / `free_idx` are **solve arguments** naming which shape parameters are left free. Unrelated. |
| **fitted** | The generic English sense throughout, *and* `remeasure --shape fitted` specifically. |

### Other

| Term | Meaning |
|---|---|
| **patches** | An optional per-galaxy `patches.json` beside the galaxy directory. Custom knowledge the blind catalog rules cannot supply. No file means pure catalog behavior. See [§6](#6-extension-points). |
| **extraction / tag** | SPHEREx only. An **extraction** is one distinct configuration (source model + background region + MJD window). Its **tag** is `<model>-<hash6>`, a deterministic hash over the canonically-normalized configuration, used as the table filename infix and indexed in `extractions.json`. |
| **HAP** | Hubble Advanced Products — the MAST pipeline producing single-visit drizzled mosaics, per-filter point/segment catalogs, and per-field color composites. The `hst` image provider, the `hst` catalog provider, and `overlay` all consume HAP products. |

---

## 3. Architecture

### 3.1 A measurement, end to end

`pipeline.run_measure` has four phases. Everything after phase 1 is offline once
the image and scene caches exist.

```
CLI: sedphot measure --name X --instruments legacy cfht --aperture 12 --out-dir DIR
  |
  cli._cmd_measure -> _check_measure_args -> resolve.resolve_target
  |                                            (Sesame -> NED -> SIMBAD)
  v
pipeline.run_measure(coord, label, out_dir, instruments=[...], ...)
  |
  | validates: providers known; aperture inside the CoG grid;
  |            --registry-update implies --registry
  | enters recipe.sky_floor(sky_rmin_arcsec)   <- scopes BG_RMIN_AS to this call
  |
  +-- PHASE 1  fetch                images/<provider>.fetch(coord, ...) per instrument
  |            -> list[ImageProduct] cached under Photometry/<Inst>/
  |            -> or ProviderResult(no_coverage|error), recorded, run continues
  |            -> no products at all: report and STOP before any scene query
  |
  +-- PHASE 2  explicit shape       only when mode='sersic' and --sersic-params /
  |            (optional)           --sersic-from was given; pipeline._resolve_shape
  |
  +-- PHASE 3  scene                engine.prepare_scene(...) ONCE per galaxy
  |            - legacy.query_scene  -> Photometry/scene/tractor_scene_<dr>.csv
  |            - gaia.query_cone     -> Photometry/scene/gaia_scene.csv
  |            - stars.confirm_stars (astrometric significance)
  |            - patches.json read + validated
  |            - components.apply_patches -> drop_target_shreds
  |            - seats.load_registry
  |
  +-- PHASE 4  measure              for each instrument, engine.order_bands(products):
               for each band:  engine.measure_band(...) -> (measurement, new_ref)
                 aperture.measurement_to_row -> schema row
                 qa.qa_scene_figure           -> <Inst>/QA/<Inst>_<band>.png
               ApertureCoverageError -> band demoted, loop continues
               any other Exception   -> printed, band dropped, loop continues
  |
  +-- registry saved (if --registry-update), coverage report, growth curves,
      <label>_measured.csv + provenance sidecar
```

The band loop threads two pieces of state. `references[instrument]` carries the
reference band's seat list, solved parameter vector, pixel scale, and per-seat
fluxes forward to that instrument's siblings. `caches[instrument]` holds an
unconvolved-profile cache keyed by component name, valid only while the stamp
shape and center are unchanged (catalog shapes are band-independent; each band
convolves with its own PSF).

### 3.2 One band: `engine.measure_band`

```
load_stamp                cut, calibrate (counts -> µJy via stamp.cf), flag nodata,
                          measure global sigma and the far-field level
resolve_psf               brightest usable confirmed star -> ring-median profile
                          with grafted Moffat wings; Moffat fallback if none passes
check_coverage            first gate: aperture / peak / seeing core
build_components          every catalog row -> a rendered fixed profile
_system_names             which components are the target's own light
apply_registry            consume frozen shapes (sidecar snapshot first, else the
                          live registry); protect the target system
_pin_target               (forced mode only) replace the target profile
find_artifacts            catastrophic pixels -> out of `good`; a component whose
                          claim is majority-masked is voided; coverage re-judged
treat_stars               confirmed stars -> masked in-zone (no column) or
                          leashed out-of-zone; nothing is subtracted
build_seats               (reference band only) gated cores, patch free seats,
                          the standard target refit; transfer bands reuse ref['seats']
joint_fit                 {shapes + amplitudes} <-> background, block coordinate
   or pinned_fit          descent to a fixed point.  A gating target's transfer band
                          runs a second, free-target pass (see below).
build_mask                three channels: model-isophote ∩ geometry, star profiles,
                          ambient flood
residual_mesh             post-fit background structure, DC-zeroed on the plane's turf
twin_fill                 masked + blank aperture pixels from their mirror, clamped
curve                     enclosed(r) of (filled - plane - mesh) over rgrid
witness_row               every diagnostic, plus fit_state for reconstruction
flux_error / empap_error  white-noise floor, then the measured empty-aperture scatter
harvest_seats             (--registry-update only) write shapes back
```

**The two-solve.** On a transfer band the science flux always comes from the pass
with the target system *frozen* at the reference shape — that is what keeps colors
shape-consistent within an instrument. A **gating** target additionally needs its
own free per-band shape, because that is the data-driven envelope other fields will
consume. With `recipe.NEIGHBOR_SHAPE_FROM_FREE_SOLVE = True` (the default) the free
solve runs **first** and settles every seat; the science pass then adopts its
neighbor shapes and re-freezes the target, leaving only amplitudes and background
to solve. One nonlinear solve, and the neighbor shape stored is the one used. With
the flag off, each pass solves neighbors itself — two nonlinear solves, and the
stored shape is not the used one.

**Aperture attribution.** The flux is an integral of real photons minus stars,
neighbors, plane, and mesh. The target model reaches it only through two channels:
`build_mask` sees `scene_img` (so the flood channel's protection perimeter tracks
what the solve says the target is), and `twin_fill`'s `model_fill` supplies both
the fallback value and one end of the fill clamp. The identity
`f_ap − m_ap_fit = resid_unmasked_ap + fill_vs_model_ap` holds exactly and is
recorded per band.

### 3.3 A catalog retrieval

Much simpler. `run_catalogs` calls each provider's `query(coord, radius_arcsec)`,
which wraps its cone search in `retry.with_expanding_radius` (double the radius, up
to five attempts), picks the source closest to the target, and emits one schema row
per detected band. Providers never raise past their own boundary — they return a
`ProviderResult` with status `ok` / `no_coverage` / `no_match` / `error`. The driver
concatenates every provider's rows, optionally dereddens, and writes the table plus
a sidecar recording each provider's status and metadata.

### 3.4 Tracing a flux to a table row

For an aperture-mode measured row:

1. `enc = curve(...)` in `engine.measure_band` — the empirical curve of growth.
2. `flux_ap = aperture.enclosed_at(rgrid, enc, aperture_arcsec)` — linear
   interpolation of that curve at the science radius.
3. `err_ujy, err_model = aperture.flux_error(...)`, then raised to
   `aperture.empap_error(...)` if the measured empty-aperture scatter is larger and
   at least `EMPAP_MIN_APS` apertures were placed.
4. `aperture.measurement_to_row(measurement, mode='aperture')` →
   `schema.make_row(band=f"{instrument}_{band}", flux_ujy=flux_ap, ...,
   flags=qa_flags(...), source=f"sedphot_aperture_scene_{err_model}")`.
5. `schema.rows_to_frame` assembles and enforces uniqueness on `(band, source)`.

In `--mode sersic` step 4 reports `witness['target_model_uJy']` instead — the fitted
target system's total through the same scene fit and the same error model.

---

## 4. Module reference

### 4.1 Top level (`src/sedphot/`)

#### `cli.py`
The argparse tree. `build_parser()` assembles one subparser per verb; `main()` runs
`args.func(args)`.

Two shared argument groups are the load-bearing part. `_add_target_args` gives every
verb the same target spec (`--name` xor `--ra/--dec`) and, for verbs that write,
a required `--out-dir`. `_add_measure_args` defines the measurement options **once**
so `run` always accepts exactly what `measure` does.

`_check_measure_args` refuses contradictory flags rather than ignoring them: a shape
flag (`--sersic-from`, `--sersic-params`, `--sersic-seeing`) under `--mode aperture`
exits nonzero. The exception is `run` with `--spherex` on, where `--sersic-params` also
declares the SPHEREx extraction shape and so is meaningful alongside an
aperture-mode measurement.

**Invariant:** any new measurement option must go in `_add_measure_args`, not in the
`measure` subparser directly, or `run` will silently measure under different
settings.

#### `pipeline.py`
The drivers. All science lives elsewhere except `_resolve_shape`, which fits the
sky-frame Sersic shape an explicit `--sersic-from` request needs.

```python
run_catalogs(coord, label, out_dir, *, instruments, radius_arcsec=2.0,
             legacy_dr=LEGACY_DR_DEFAULT, dered=False,
             target_name=None) -> pd.DataFrame

run_measure(coord, label, out_dir, *, instruments, mode='aperture', bands=None,
            aperture_arcsec=12.0, cutout_arcsec=120.0,
            scene_aperture_arcsec=None, sky_rmin_arcsec=None, rgrid=None,
            sersic_from=None, sersic_params=None, sersic_seeing=None,
            registry_path=None, registry_update=False,
            pin_by_band=None, write_outputs=True, qa_dir=None,
            dump_arrays=False, legacy_dr=LEGACY_DR_DEFAULT,
            legacy_bricks=False, hst_proposal_id=None,
            target_name=None) -> pd.DataFrame

run_spherex(coord, label, out_dir, *, model='sersic', sersic_params=None,
            sersic_from=None, sersic_seeing=None, bkg_size=15.0,
            mjd_range=None, poll=5.0, timeout=3600.0, cutout_arcsec=120.0,
            legacy_dr=LEGACY_DR_DEFAULT, target_name=None) -> ProviderResult

run_sed(label, out_dir) -> Path | None
run_overlay(label, out_dir, *, zoom_arcsec=5.0, context_arcsec=15.0,
            wcs_from=None, dpi=200) -> Path | None
run_all(coord, label, out_dir, *, skip=None, ...) -> dict[str, str]
```

`run_measure`'s last four keyword groups are the **reconstruction hooks**, not CLI
surface: `pin_by_band` (rebuild each band from a stored fit instead of solving),
`write_outputs=False` (suppress every product so a re-report cannot touch the
science files — the frame is still returned), `qa_dir` (send figures to a scoped
directory), and `scene_aperture_arcsec` (build the scene at one radius while
integrating at another).

`run_all` returns a dict of **stage failures**; the `run` verb exits nonzero on a
non-empty return. Stages are isolated — a stage that raises is recorded and the
remaining stages still run. If `run_measure` dies before writing its own coverage
report, `run_all` writes an error entry in its place; a report the stage already
wrote is never overwritten (mtime comparison).

**Invariants.**
- `scene_aperture_arcsec` scopes the scene (shred rule, star zone);
  `aperture_arcsec` only integrates. Conflating them changes which catalog rows
  exist.
- The `recipe.sky_floor` context manager wraps the entire measurement, so the
  override cannot leak into another call.
- Phase 1 short-circuits before phase 3: a run with zero fetched images must not
  spend TAP calls building a scene nothing will consume.

#### `schema.py`
The one table contract. `BASE_COLS` (12 columns) is **frozen**: order and names do
not change, because `overlay.py` and downstream analysis select on them.
`EXTRA_COLS` is append-only.

```python
make_row(band, flux_ujy, flux_err_ujy, mag, mag_err, target_ra, target_dec,
         match_ra, match_dec, sep_arcsec, flags, source, *,
         retrieved=None, mw_transmission=nan, dered_applied=False) -> dict
rows_to_frame(rows: list[dict]) -> pd.DataFrame
```

`rows_to_frame` enforces the table invariant downstream selection depends on: **no
two rows may share `(band, source)`**. The same band from two different measurements
(unWISE vs AllWISE `WISE_W1`) is legitimate; the same band twice from one
measurement is a provider bug and raises.

`SOURCE_PREFIXES` is the machine half of the `source` contract — see [§5.3](#53-the-source-string-vocabulary).

#### `results.py`
`STATUS_OK` / `STATUS_NO_COVERAGE` / `STATUS_NO_MATCH` / `STATUS_ERROR`, plus two
dataclasses.

- `ProviderResult(provider, status, rows=[], message='', radius_used=None, meta={})`
- `ImageProduct(provider, instrument, band, path, calib, invvar_path=None, seeing_arcsec=1.0, wave_um=nan)`

`write_coverage_report(results, path) -> Path` and
`print_coverage_summary(results) -> None`.

**Invariant:** a provider never raises past its own boundary. Every catalog provider
returns a `ProviderResult`; every image provider returns `list[ImageProduct]` **or**
a `ProviderResult`. Image providers never report `no_match`.

#### `resolve.py`
`resolve_target(*, name=None, ra=None, dec=None, label=None, verbose=True) -> (SkyCoord, str)`
— exactly one of `name` or `(ra, dec)`. Name resolution tries Sesame
(`SkyCoord.from_name`), then NED, then SIMBAD. The label defaults to
`sanitize_label(name)` or `jname(coord)` (IAU `Jhhmmss.ss+ddmmss.s`).

**Invariant:** providers never take raw names. Resolution happens once, up front.

#### `units.py`
`NANOMAGGY_TO_UJY = 3.631`, `AB_ZP_UJY = 23.9`, and the conversions
`nanomaggy_to_ujy`, `mag_to_ujy`, `ujy_to_mag`, `mag_err_to_flux_err`,
`flux_err_to_mag_err`. Non-finite input returns NaN rather than raising.

#### `bands.py`
`WAVE_UM` (effective wavelength per band label) and `wave_um(band) -> float`. HST
filters are parsed from the name rather than tabulated: four digits are nm
(`F1042M` → 1.042 µm), three digits below 200 are 10 nm (`F160W` → 1.60 µm),
otherwise nm (`F475W` → 0.475 µm).

**Invariant:** these numbers are for figure coloring only and never enter a
measurement. SED fitters own bandpass physics.

#### `provenance.py`
```python
write_sidecar(product_path, meta: dict) -> Path      # <stem>.provenance.json
sha256_16(path) -> str
git_state() -> dict                                  # {git_rev, git_dirty}, fail-soft
```
Automatic fields (`product`, `written`, `sha256_16`, `package`, `package_version`,
`git_rev`, `git_dirty`) **win** a name collision with a caller key, and the dropped
key is named on stdout. A caller key silently replacing the hash or the revision
would leave a record that reads authoritative and is not.

#### `retry.py`
Four composable policies.

| Function | Handles |
|---|---|
| `with_expanding_radius(query_fn, coord, radius, label, *, max_retries=5, expand_factor=2.0)` | No-match: re-run the cone with a doubled radius. `query_fn` must return `[]` for empty, never raise. |
| `retry_transient(call, label, *, attempts=3, base_delay=2.0)` | One flaky endpoint: exponential backoff, re-raise the last exception. |
| `try_services(services, label, *, rounds=2, base_delay=2.0)` | Service rotation: skip an erroring endpoint immediately; back off only between whole rounds. A service answering cleanly with **no data** is authoritative — rotation stops and `None` is returned. |
| `query_vizier_mirrors(query_fn, label)` | VizieR: a mirror outage can present as empty results rather than errors, so the next mirror is asked before concluding no-match. |

**Invariant on VizieR:** re-pointing `astroquery.vizier.conf.server` at runtime does
not work — `VizierClass` captures the config value in a signature default at import.
`query_fn` receives the hostname and must construct `Vizier(vizier_server=server, ...)`.

#### `dered.py`
Opt-in Milky Way dereddening, three tiers per row: native `mw_transmission` from the
provider; else an `EXT_COEFF` coefficient times an IRSA SFD E(B−V) fetched once per
target; else left as-measured with a warning. `apply_dereddening(df, coord) ->
(dered_df, meta)`.

**Invariant:** only `run_catalogs` calls this. Every measured row keeps
`dered_applied = False`. See [§7](#7-behavior-worth-knowing).

#### `qa.py`
Headless matplotlib (`Agg`).

```python
qa_scene_figure(measurement, out_dir) -> Path   # <Inst>_<band>.png, 5 panels
plot_growth_curves(measurements, out_dir) -> Path
plot_sed(frames: dict[str, DataFrame], outpath, *, title='') -> Path
```

`qa_scene_figure` draws data | fitted scene + background | residual − mesh |
masked + filled − mesh | curve of growth. The panel directory belongs to the caller,
which is how `remeasure --write-qa` scopes its figures away from the science QA.

In `plot_sed` the frame key `'measured'` is load-bearing: that frame draws with open
faces, everything else filled. Marker = instrument, color = wavelength.

#### `overlay.py`
The positional counterpart to the SED plot: where each provider *matched*, drawn on
the HAP color composite. Reads only `match_ra`, `match_dec`, and `source` — which is
why the first twelve schema columns are frozen.

```python
build(tables, outpath, cache_dir, *, zoom_arcsec=5.0, context_arcsec=15.0,
      wcs_from=None, dpi=200) -> Path | None
style_for(source: str) -> dict
discover_hap_total(coord, radius_arcsec=60.0) -> dict | None
load_color_image_and_wcs(color_path, wcs_path) -> (rgb, WCS) | None
```

`PROVIDER_STYLES` is keyed by exact `source` string first (so HAP segment and point
centroids draw differently), then by the `SOURCE_PREFIXES` provider token — so a
provider that revs its data release keeps its marker.

**Invariants.**
- The composite carries no WCS of its own; the detection mosaic it was rendered from
  supplies one (~380 MB, cached). `wcs_from` substitutes a local file on the same
  drizzle grid. Pixel dimensions are checked either way and a mismatch **refuses**.
- A `wcs_from` path naming no file refuses before anything is queried — falling back
  to the download is the one outcome the caller asked by name to avoid.
- Overlapping HAP visits are ranked by `obs_id` and the lowest wins, so the same
  field yields the same composite every run.

#### `remeasure.py`
Re-report band fluxes from the immutable sidecar.

```python
remeasure(provenance_path, aperture_arcsec=None, mode='sersic',
          shape='forced', registry_path=None, write_qa=False) -> pd.DataFrame
reconstruct(provenance_path, aperture_arcsec, shape='forced',
            registry_path=None, write_qa=False, status=None) -> dict[str, float]
model_flux_within(aperture_arcsec, rgrid, cog, total) -> float
```

Behavior by mode and radius:

| mode | radius | what happens |
|---|---|---|
| `sersic` | inside grid | interpolate the fitted model's COG |
| `sersic` | past grid | the model total (it has converged) |
| `sersic` | `--integrated` | the model total |
| `aperture` | inside grid | interpolate the empirical neighbor-subtracted COG |
| `aperture` | past grid | `reconstruct`: rebuild the scene from the pinned fit and integrate at R |
| `aperture` | `--integrated` | the outermost stored point, `aperture_as = inf` |

"Past grid" is judged against the **shortest** stored grid across all bands, so it
is a property of the galaxy, not of one band: one band whose grid stopped early
sends every band down the reconstruction path.

`_build_pin_by_band` is where the shape selection lives. Its source of truth is
`witness['shapes']` — the whole seat vector on the band's own grid, whatever route
each shape took there. A sidecar without that record falls back to the
**instrument's reference band** `solve` record (correct: a transfer band's target was
frozen at exactly that shape). What must never be used is a transfer band's *own*
`solve` record, which covers the free neighbor seats alone; a length check refuses
it and the band is reported rather than silently rebuilt wrong.

`reconstruct` replays the run's cutout size, its `scene_aperture_arcsec` (the
sidecar's own science aperture), its `BG_RMIN_AS` from the recipe snapshot, and the
Legacy fetch options (`dr`, `bricks`) — different pixels would mean a different fit.
It refuses an aperture past the stamp half-width, and any band the sidecar cannot
pin is **solved** rather than reconstructed and labeled `solved_*` instead of
`reconstruct_*`.

#### `spherex.py`
Programmatic driver for the IRSA SPHEREx Spectrophotometry Tool — a GUI over an
IVOA UWS 1.1 async service. Direct UWS job creation is token-gated (403 for guests),
so submission goes through Firefly's command server as the public GUI does, then the
open UWS endpoint is polled by job id.

```python
@dataclass Sersic(n, axis_ratio, pa_deg, reff_arcsec)   # .reff_deg property
sersic_from_shape(shape_sky: dict) -> Sersic
fetch(coord, *, out_dir, model=None, bkg_region_size=15, mjd_range=None,
      poll=5, timeout=3600, shape_origin=None, label=None,
      target_name=None) -> ProviderResult
fetch_spectrophotometry(ra, dec, model=None, ...) -> pd.DataFrame
extraction_tag(model, bkg_region_size=15, mjd_range=None) -> str
config_payload(model, bkg_region_size=15, mjd_range=None) -> dict
```

**Unit trap:** Firefly's `ServerRequest` carries `effectiveRadius` in **degrees**;
the UWS `ELLIPTICAL` string and every CLI surface use **arcsec**. The `Sersic`
dataclass stores arcsec and converts.

**Invariants.**
- Nothing on disk is ever overwritten or renamed. A configuration already on disk is
  reused **only** when the table's own provenance sidecar records that exact
  configuration — a filename is a claim, a sidecar is the evidence.
- `config_payload` holds everything that changes what the tool computes (model
  parameters, background region, MJD window) and nothing that does not (shape
  origin, fetch date), so the same numeric shape from two sources is one extraction.
- The tool treats any source with `reff < 1″` as a point source; `fetch` warns that
  the Sersic parameters are then cosmetic.
- On failure, `fetch` prints the manual Data Explorer recipe and writes no partial
  file.

### 4.2 The scene engine (`src/sedphot/measure/`)

Stage-numbered modules form the chain; unnumbered modules are shared services.

#### `recipe.py` — every science constant
One place for every tunable. Distances arcsec, fluxes µJy, surface brightness
µJy/arcsec², bound pairs `(low, high)`. Stage-local implementation constants (the
PSF ring schedule, seat size floors) stay in their own module and are labeled there.

Selected constants, grouped as the file groups them:

| Constant | Value | Role |
|---|---|---|
| `DEFAULT_RGRID` | `arange(2, 41, 1)` | Curve-of-growth radii, 2–40″ |
| `EXCESS_OUT_AS` | 25.0 | Outer radius of the excess-growth witness |
| `PED_WINDOW_AS` | (6, 25) | Window of the pedestal fit `enc = F + b·πr²` |
| `PLATEAU_EPS` / `PLATEAU_RUN` / `HOLD_MAX` | 0.01 / 3 / 0.02 | Plateau certification |
| `COVERAGE_MIN` | 0.95 | Aperture coverage floor |
| `CORE_MASKFRAC_MAX` / `PEAK_PROTECT_AS` | 0.10 / 0.5 | Core and peak gates |
| `QUERY_RADIUS_AS` / `QUERY_PAD_AS` | 100 / 15 | Scene cone floor and corner pad |
| `TRACTOR_MIN_NMGY` | 0.5 | Scene-query flux floor, brightest optical band |
| `TARGET_MATCH_AS` | 1.5 | Identity radius (target, stars, patches, registry) |
| `GATE_FLUX_UJY` / `GATE_RCHISQ` / `GATE_RCHISQ_MAX` | 100 / 6 / 1000 | The gate and its veto ceiling |
| `GATE_EDGE_MARGIN_AS` | 15 | Radial gate reach, inset from the stamp half-width |
| `SHRED_FRACFLUX` | 1.0 | Target-substructure rule |
| `MARGIN_AS` / `MARGIN_MIN_UJY` | 25 / 1.0 | Off-stamp admission and its in-stamp flux floor |
| `BRIGHT_PSF_UJY` | 100 | Off-stamp point sources keeping analytic Moffat wings |
| `STAR_ASTROM_SIG` | 3.0 | Gaia parallax/PM significance for star confirmation |
| `STAR_ZONE_BUFFER_AS` | 3.0 | Star zone = aperture + this |
| `BIN_AS` / `BIN_MIN_FRAC` / `BG_REJ_SIGMA` | 5.0 / 0.5 / 3.0 | Background bins and rejection |
| **`BG_RMIN_AS`** | **15.0** | **The one target/sky boundary; six stages read it** |
| `FARFIELD_RMIN_AS` / `FARFIELD_MIN_PX` | 50 / 5000 | Independent far-field witness |
| `ALT_MAX_ITER` / `ALT_TOL_SIGMA` | 4 / 0.02 | Background↔amplitude alternation |
| `SEAT_NPARAMS` | 6 | Parameters per seat |
| `SERSIC_N_RANGE` / `SERSIC_E_MAX` / `NUKER_E_MAX` | (0.4, 6.0) / 0.92 / 0.85 | Profile bounds |
| `NUKER_GAMMA` / `NUKER_ALPHA` | 0.5 / 2.0 | Frozen inner slope and break sharpness |
| `NUKER_RB_AS` / `NUKER_RB0_AS` / `NUKER_BETA` / `NUKER_TRUNC_AS` | (2, 85) / 15 / (1.8, 8) / 120 | Halo family |
| `DXY_OUT_AS` / `SEAT_DXY_AS` | 8.0 / 1.0 | Halo and Sersic center boxes |
| `GATED_CORE_REFF_MAX_AS` / `FREE_SEAT_REFF_MAX_AS` / `REFIT_REFF_MAX_AS` | 5 / 6 / 10 | Role enforcement by size cap |
| `GATED_HALO` | `False` | Gated **neighbors** get a Sersic core only; the Nuker halo belongs to the target, granted via patches |
| `SOLVE_NFEV` / `SOLVE_NFEV_WARM` / `SOLVE_FSCALE` / `SOLVE_MODEL_ERR_FRAC` | 450 / 150 / 3.0 / 0.02 | Optimizer budget and loss scaling |
| `AMP_MAX_X_CAT` | 100 | Amplitude sanity ceiling |
| `TRANSFER_AMP_BAND` / `BAND_COLOR_COL` | (0.1, 10) / per-filter | Transfer leash and its color column |
| `REFERENCE_PREFERENCE` | `(r,i,z,g,y,u)` | Which band leads per instrument |
| **`NEIGHBOR_SHAPE_FROM_FREE_SOLVE`** | **`True`** | Free-solve-first ordering on a gating target's transfer bands |
| `REGISTRY_AMP_BAND` / `REGISTRY_MATCH_AS` / `REGISTRY_KEY_COALESCE_AS` | (0.8, 1.25) / 2.0 / 1.0 | Consumption leash, row match, key coalescing |
| `K_ISO` / `GEO_REFF_FACTOR` / `GEO_SEEING_FLOOR` / `FLOOD_MAX_AS` | 1.0 / 2.5 / 1.5 / 6.0 | Mask channels |
| `TARGET_MASK_FREE_AS` | 0.0 | Mask-free core radius (inert at 0) |
| `MOFFAT_KERNEL_FWHM` / `PSF_STAR_GMAG` / `PSF_WING_SNR` | 16 / (15.8, 22.0) / 5.0 | PSF kernel and star window |
| `PSF_EXCLUDE_TARGET_AS` / `PSF_WING_FRAC_MAX` / `PSF_GRAFT_FORCE_AS` | 25 / 0.10 / 2.5 | Kernel cleanliness |
| `ARTIFACT_SIG` / `ARTIFACT_RATIO` / `ARTIFACT_AREA_MIN` / `ARTIFACT_SB_MIN` | 20 / 5 / 15 arcsec² / 30 µJy/arcsec² | Artifact gate |
| `EMPAP_N` / `EMPAP_MIN_APS` | 200 / 30 | Empty-aperture error placements |

```python
snapshot() -> dict                 # every UPPERCASE global, JSON-safe
sky_floor(rmin_arcsec: float | None)   # context manager scoping BG_RMIN_AS
```

**Invariants.**
- Every consumer reads `recipe.BG_RMIN_AS` **at call time** — background bins, star
  exclusion, stamp validity, artifact detection, both empty-aperture placements — so
  one override moves them together. That is the intent: one boundary, one owner.
  Caching it in a module-level default would break `sky_floor`.
- `snapshot()` reads `globals()` at call time too, so the sidecar records the value
  the run actually used and `reconstruct` reads it back.
- `render.render_sersic_boxed` clamps `n` to a literal `(0.4, 6.0)` when sizing its
  render box; that literal is a deliberate copy of `SERSIC_N_RANGE` and must be kept
  in step with it.

#### `stamp.py` — Stage 1: stamp preparation and data-sufficiency gates

```python
@dataclass Stamp:
    data, wcs, header, cx, cy, pixscale, cf, rr, nodata, sigma,
    farfield_sb, invvar=None
    .good   -> ~nodata
    .shape  -> data.shape
    .sb     -> cf / pixscale**2        # counts -> µJy/arcsec²

load_stamp(path, calib, coord, *, cutout_half_arcsec, invvar_path=None) -> Stamp
check_coverage(stamp, *, aperture_arcsec, seeing_arcsec, nodata=None) -> float
radii_arcsec(shape, cx, cy, pixscale) -> np.ndarray
class ApertureCoverageError(RuntimeError)   # carries .coverage
```

`nodata` marks non-finite pixels, **exact archive zeros** (off-footprint fill), and
pixels more than 10σ below the outer clipped level (dead pixels, cosmic-ray holes).
`sigma` is a global clipped scatter of the outer stamp — an upper bound on the pixel
noise for thresholds and solver scales, **not a background estimate**. `farfield_sb`
is an independent robust zero measured beyond `FARFIELD_RMIN_AS`, recorded per band
and never fed back into the fit.

**Invariant:** no background is estimated at this stage. The background is owned
entirely by `background.bin_plane`.

`check_coverage` raises `ApertureCoverageError` on any of the three gates; the
pipeline catches it and demotes the band rather than reporting a biased flux. It is
called twice per band — once after the PSF, once again after artifact masking with
the augmented `nodata` map.

#### `psf.py` — Stage 2: empirical PSF, Moffat fallback

```python
resolve_psf(stamp, cat, stars, *, psfsize_col=None, fallback_arcsec=1.0,
            fallback_label='provider default') -> (kernel, fwhm_arcsec, provenance)
empirical_psf(stamp, stars) -> (kernel, fwhm, provenance) | None
resolve_seeing(cat, header, *, psfsize_col=None, ...) -> (fwhm, provenance)
moffat_kernel(seeing_arcsec, pixscale) -> np.ndarray
```

Empirical first: the brightest confirmed star inside `PSF_STAR_GMAG` that passes
every guard — a populated background annulus, at least `MIN_FINITE_RINGS` measured
rings, a monotone bright core, a plausible FWHM, and clean wings. The profile is a
circularized ring median; rings below `PSF_WING_SNR` hand off to a grafted Moffat
continuation whose **slope is fit on the star's own shoulder rings**. A kernel
carrying more than `PSF_WING_FRAC_MAX` beyond twice its own FWHM is contaminated and
retries with the graft forced at `PSF_GRAFT_FORCE_AS`; failing that, the next
candidate is tried.

**Invariants.**
- Kernels are unit-sum, square, odd-sized, edge-tapered to exactly zero (a square
  truncation renders every bright point source as a box in the fitted scene), and
  rendered at the stamp pixel scale.
- No candidate inside `PSF_EXCLUDE_TARGET_AS` of the target: those rings measure the
  envelope at high S/N, which the noise-triggered graft cannot catch.
- `moffat_kernel` **measures** its rendered FWHM and iterates the input width until
  they agree — pixel integration broadens a nominal-width Moffat by several percent
  at survey sampling, and a too-broad kernel is a floor no shape solve can get under.
- `r = 0` is anchored at the star's background-subtracted peak, not at the first ring
  median: the median of a peaked core is not its peak.
- The catalog per-source PSF-size column is read only by the instrument whose imaging
  the scene catalog describes (Legacy); everything else falls through to a header
  keyword or the provider-typical value.

#### `components.py` — Stage 3: catalog rows to scene components

```python
gated_row(row, dist_arcsec) -> bool
apply_patches(cat, patches) -> pd.DataFrame
drop_target_shreds(cat, coord, *, aperture_arcsec, patches=None) -> pd.DataFrame
build_components(cat, stamp, psf, seeing_arcsec, *, profile_cache=None,
                 gate_radius_arcsec=None) -> list[dict]
```

The component dict schema is in the module docstring and in [§2](#2-glossary).

**Invariants.**
- Components are named by **catalog row index** (`src<irow>`), never a running count.
  The component list differs between bands (margin cuts on different grids), and
  solved shapes transfer across bands by name.
- The target is the **closest** row within `TARGET_MATCH_AS`, one row, never every
  row within it — a tight pair straddling the requested position must keep separate
  identities.
- Design columns are normalized to unit in-stamp flux downstream, so a near-empty
  render is a numerically explosive basis. Hence the `MARGIN_MIN_UJY` admission test
  for off-stamp extended rows, with no exemptions; an off-stamp giant whose envelope
  truly reaches is patches territory.
- Render amplitude is the scene band floored at a tenth of the row's brightest band,
  so a very red source (r ≤ 0, z bright) does not render a dead zero column.
- On-stamp point sources are pasted with **bilinear sub-pixel placement** (the
  convolution of a delta is a translation) — dumping the flux into the nearest pixel
  leaves a dipole residual at every point source.
- `gated_row` is judged **per band, any optical band qualifying**; a catastrophic
  reduced chi-square in *any* band vetoes the row outright. Its `dist_arcsec`
  argument is what excludes the target, which is why `engine._target_gates` passes a
  deliberately non-self distance.

#### `stars.py` — Stage 4: the star stage

```python
confirm_stars(gaia) -> pd.DataFrame
treat_stars(stamp, comps, stars, *, colors=None, aperture_arcsec=None, tag='')
    -> (star_masks, comps, star_log)
```

Confirmation is **astrometric, not positional**: a Gaia row counts as a star only
with a five-parameter solution at parallax or proper-motion significance above
`STAR_ASTROM_SIG`. Gaia membership alone is not enough — compact galaxy nuclei are
in Gaia.

Stars are treated brightest-first and sequentially: each profile is measured on data
with brighter siblings already subtracted, with the treated stars' catalog bases
removed from the reference scene, and with already-measured star light excluded above
1σ. A component is treated once even when several Gaia rows land on it. Successfully
measured stars **leave the component list** and their light is pre-subtracted.

**Invariants.**
- The expectation a profile is judged against is the star's **in-stamp** flux
  (`flux0`), never its catalog total: amplitudes are in-stamp flux in this design,
  and for an off-stamp star a total-flux leash would force that much wing light onto
  the stamp.
- A star inside the star zone gets **no design column** — a free point-source
  column there can absorb target light wholesale, and its over-subtraction
  cannot be masked after the fact. Its predicted footprint is masked and the
  twin fill reconstructs beneath it.
- **No star is ever subtracted from the data.** The route is decided by
  geometry alone, so it cannot depend on how well any model happens to fit.
  Both routes bound the damage a wrong star model can do: the leash caps the
  amplitude, and the mask removes the region rather than trusting a model over
  it. Neither can excavate light that was never the star's.

#### `seats.py` — Stage 5: seats and the cross-field registry

```python
seat_slices(seats) -> list[slice]
snap_to_peak(image, x0, y0, pixscale) -> (x, y)
build_seats(comps, patches, stamp, image, *, tag='') -> (seats, drops)
apply_registry(comps, registry, stamp, psf, band_key, instrument, *,
               protect_px=None, snapshot=None, tag='') -> (comps, consumed)
harvest_seats(registry, seats, params, seat_amps, stamp, *, band_key,
              seat_col_flux=None, include_target=False, solve_health=None,
              tag='') -> list[str]
registry_name(ra_deg, dec_deg) -> str
resolve_registry_key(registry, ra_deg, dec_deg, tol_arcsec=None) -> str
load_registry(path) -> dict
save_registry(registry, path) -> None
```

`build_seats` produces, in order: a Sersic core for every gated component (plus a
Nuker halo only where `GATED_HALO` grants one); patch-granted `free_seats`; and the
standard **target refit** — the target's shape is always solved from the data, so the
catalog informs the photometry only through the neighbors. `patches.target_halo`
replaces the refit with a core + Nuker halo pair; `patches.target_refit: false`
disables it.

`apply_registry` runs in two passes so entries cannot eat each other's components:
first every catalog row matched by a live entry is dropped, then the frozen
components are added. Each consumed component carries `reg=True`, an `amp_lohi`
leash, and — for a Sersic record — its stored geometry, which `build_mask` needs to
size its mask on.

**Invariants.**
- Seat centers are stored as **sky coordinates**; every band renders the same scene
  on its own grid. Radial parameters are stored in *reference-band pixels* and
  rescaled arcsec-invariantly (`solve._scale_seat`, `_scale_params`).
- Registry entries transport **arcsec and ra/dec**, so a neighbor is portable across
  fields and instruments.
- A consumed component is rendered at its **fitted physical amplitude**
  (`flux_ref / (flux_home · this stamp's cf)`), never renormalized to this stamp.
  Renormalizing would cram a whole off-stamp source's flux into the few pixels that
  reach the edge. `flux_home` is stored cf-free, so per-stack zeropoints stay local.
- A record whose solved flux is below `MARGIN_MIN_UJY` is **never stored**. Zero
  amplitude is a two-part verdict — no such component here, and the attached shape
  was never constrained.
- A target-vantage record is never downgraded by a neighbor-vantage write.
- An entry is a **joint decomposition**: one tainted seat tombstones its entry's
  whole band.
- A Nuker halo record deliberately carries **no** mask geometry: masking is for
  compact neighbor light, and a halo sized on `rb` would swallow the aperture of
  anything embedded in an envelope.
- `save_registry` is atomic (write a sibling, then replace) — but atomicity is **not**
  serialization. Two concurrent updates finish last-writer-wins, so
  `--registry-update` sweeps must run one galaxy at a time.

#### `solve.py` — Stage 6: the joint fit

```python
joint_fit(image, good, stamp, psf, comps, seats, drops, *, ref=None,
          free_target=False, freeze_neighbors=None) -> dict
pinned_fit(image, good, stamp, psf, comps, seats, drops, *, pin) -> dict
solve_shapes(image, good, comps, bg_img, stamp, psf, seats, drops, *,
             p_seed=None, extra_fixed_cols=None, gram=None,
             stage_warm=False) -> dict
render_seats(seats, p, stamp, psf, s_px=1.0, anchors=None) -> (cols, owners)
seat_anchors(seats, stamp) -> list[(x, y, t0, slope)]
```

`joint_fit` returns `amps`, `mults`, `bg`, `track`, `solve_info`, `cols`, `owners`,
`fixed`, `col_flux`, `amp_bounds`, `seats_local`, `seat_params`, `seat_amps`.
`pinned_fit` returns the same minus `amp_bounds`, with `solve_info=None` and a
one-element `track`.

The structure is **block coordinate descent**: `{shapes + amplitudes} ↔ background`,
alternating up to `ALT_MAX_ITER` times until the plane constant moves less than
`ALT_TOL_SIGMA × σ`. Shapes re-solve *inside* the alternation, warm-started from the
previous iterate — on halo-dominated stamps the first background is contaminated, and
shapes solved once against it would inherit that bias frozen. The shape solve is
**variable projection**: every fixed amplitude is solved exactly at every trial of
the shape parameters, with the fixed columns' Gram block precomputed once and shared
across the alternation's warm re-solves.

Three transfer modes:

| Mode | Target seats | Neighbor seats | Shape solves |
|---|---|---|---|
| default | frozen at the reference shape | re-solved warm | 1 |
| `free_target=True` | free | free | 1 (full vector) |
| `freeze_neighbors=v` | frozen at the reference shape | frozen at `v`'s slices | 0 (`solve_info` is `None`) |

**Invariants.**
- **Amplitudes are microjanskys.** Every design column is divided by its own
  in-stamp flux, so a fitted amplitude *is* that component's in-stamp flux through
  the band — one unit system for catalog components and seat columns on every
  instrument.
- The background never sits in a design matrix beside component amplitudes.
- `_target_side(seat)` reads the `seat['system']` tag the engine sets, falling back
  to the literal name. A declared system member freezes exactly as the target does.
- `solve_shapes` uses a **two-stage start** — Sersic centers frozen, then released —
  on cold solves and on cross-band warm re-solves (`stage_warm`), because a seed
  scaled from another band lands off-center and can burn its budget crawling back.
  A same-band warm re-solve inside the alternation is already in its basin and skips
  the staging.
- The Jacobian is **per-seat finite differences**: perturbing one parameter re-renders
  one seat's column against the cached others, dropping the render count from
  `(6k+1)·k` to `7k`. Steps step *inward* from whichever bound the parameter sits on;
  both bounds need the check, because the staged center freeze is a box narrower than
  the step itself.
- `solve_info['seconds']` is the **last** warm iteration only. Never sum it as total
  solve time.
- `solve_info['pix_ref']` records the grid the radial parameters live in. Without it
  a pinned reconstruction cannot know whether a stored vector needs rescaling — a
  latent no-op for a single-pixel-scale instrument, wrong for HST.
- `pinned_fit` keys amplitudes by a **per-owner queue consumed in recorded order**,
  not by owner name: one owner can hold several columns (a `target_halo` target owns
  a core Sersic and a Nuker halo), and a name-keyed dict collapses them.
- An owner absent from `pin['amps']` falls back to amplitude one — its own catalog
  prediction — rather than vanishing from the scene.
- The nonlinear solve carries an **ownership penalty**: a halo displaced beyond its
  own break radius is not that galaxy's halo.
- The per-pixel loss scale is `σ + SOLVE_MODEL_ERR_FRAC·|counts|`. Against a bare
  scalar sky sigma a bright core's model imperfection reads as thousands of sigma,
  the soft-L1 loss saturates, and the optimizer burns its budget on micro-steps.

#### `aperture.py` — Stage 7: mask, fill, curve of growth, witnesses, output row

```python
build_mask(comps, fitted_by, star_masks, stamp, seeing_arcsec, scene, neighbors,
           image, good, *, tag='') -> (mask, flood_ujy)
twin_fill(image, neighbors, mask, good, stamp, model_fill, *,
          aperture_arcsec, tag='') -> dict
curve(img, rr, cf, rgrid) -> np.ndarray
enclosed_at(rgrid, enc, radius) -> float
ped_fit(enc, rgrid) -> (F, b, rms)
plateau_hold(enc, flux_ap, rgrid) -> float
witness_row(enc, model_cog, m_ap_cat, stamp, good, mask, twin_frac, neighbors,
            bg, track, flood_ujy, seeing_arcsec, seeing_src, *,
            rgrid, aperture_arcsec, solve_info=None, solve_free=None,
            shapes=None) -> dict
empap_error(resid, vote, stamp, *, aperture_arcsec) -> (err_ujy, n_placed)
flux_error(stamp, good, *, aperture_arcsec) -> (err_ujy, 'ivm' | 'skyrms')
qa_flags(witness, *, n_comps, consumed) -> str
measurement_to_row(measurement, *, mode='aperture') -> dict
```

`build_mask`'s three channels:
1. **Intersection** — a neighbor's own fitted model above `K_ISO·σ` *and* inside its
   own recorded geometry (catalog shape, or the registry's stored shape for a
   consumed component), capped at `GEO_REFF_FACTOR × reff` with a
   `GEO_SEEING_FLOOR × seeing` floor. A shapeless component falls back to a
   seeing-sized disc.
2. **Stars** — each measured profile above its own isophote, no geometric cap; a
   measured profile cannot claim light it does not see. A *reverted* star gets its
   full uncapped model isophote.
3. **Flood** — seeded growth into pixels departing from the ambient surface that the
   scene does not claim. Symmetric: escaped glow and over-subtraction holes both
   flood. Pixels the fitted target model claims are protected.

`witness_row`'s three optional records are distinct and must not be confused:
`solve` (what *this* band's solve varied — on a transfer band, the neighbor seats
alone), `solve_free` (the free-target pass's full seat vector — the only record of a
per-band free shape outside the mutable registry), and `shapes` (the whole seat
vector the band was **measured** with, on its own grid, however each shape got there).

**Invariants.**
- The reported error is **measured**: source-free apertures on the final residual
  (`empap_error`), floored by the white-noise value (`flux_error`). The larger wins.
  A stamp too small to place `EMPAP_MIN_APS` apertures keeps the white-noise value
  under its own label. Floors and inflation belong to the SED fitter, never to this
  table.
- `empap_error`'s RNG is seeded (`default_rng(0)`), so a re-measure reproduces its
  error bar exactly. Partly-masked placements are kept and rescaled — real target
  apertures have masked pixels too.
- `curve`'s per-radius masked sum is the **fast** version. The obvious sort+cumsum
  rewrite measures ~2× slower on a fine-pixel stamp (argsort over nine million
  elements costs more than 39 vectorized reductions). Measure before optimizing.
- In aperture mode the fitted target model appears only in witnesses and in the fill;
  it is never integrated into the reported flux.
- `plateau_hold` needs both quietness *and* a hold to the grid end — per-increment
  quietness alone cannot distinguish flat from a steady sub-threshold drift.

#### `background.py` — the one background estimator

```python
bin_grid(work, usable, pixscale) -> (row_starts, col_starts, bin_px, levels)
bin_plane(work, good, rr, pixscale) -> dict
ambient_surface(work, good, mask, rr, pixscale) -> np.ndarray | None
residual_mesh(resid, vote, pixscale, *, level_px=None, state=None) -> np.ndarray
eval_plane(coefs, shape) -> np.ndarray
eval_mesh(state, shape) -> np.ndarray
```

`bin_plane` returns `img`, `const` (the level at the stamp center), `coefs`
(`[level, x-tilt, y-tilt]` in a centered/normalized parametrization), `n_rej`,
`n_bins`, and `keep_px` (the accepted bins' pixel territory).

**Invariants.**
- **The bin statistic is a clipped MEAN, never a median.** Photometry sums pixels, so
  the background must estimate the mean sky under the aperture. Real survey frames
  carry spiked pixel histograms (heavily repeated values from upstream
  integerization) on which bin medians mode-lock: their bin-to-bin scatter collapses
  far below noise, the MAD rejection threshold collapses with it and thrashes, and
  the background inherits a `(mean − median) × area` flux systematic. The clip
  supplies the robustness the median was doing duty for.
- **Three estimators own three scales and none may take another's.** Plane: level and
  tilt across the cutout. `residual_mesh`: smooth structure below that, DC-zeroed
  over the plane's accepted-bin territory so the level keeps exactly one owner.
  The fitted source model: compact light.
- Ownership of light is **positional**, not statistical: a bin is background because
  of where it sits and what survives rejection there, never because a fit found it
  convenient.
- Pixels inside `BG_RMIN_AS` never vote. Target light is excluded by position, not
  left to rejection.
- The MAD-rejection keep decision is recomputed against **all** bins each pass, so a
  bin rejected by an early, still-biased fit can win its vote back.
- `bin_plane` builds its own plane through `eval_plane`, so the producer and the
  sidecar reconstruction cannot drift apart. Same for `residual_mesh` / `eval_mesh`
  through `_bin_surface`, whose queries are clamped to the bin-center hull so the
  surface never extrapolates outward at the stamp edge.
- `bin_plane`'s design uses bin-center coordinates offset by half a bin, which biases
  the level by `tilt × 0.5/n`. This is asserted in the reconstruction tests and held
  fixed deliberately: it is ~0.03% of a level whose tilt is itself small, and any
  change to it shifts the background of every measurement already on disk.

#### `artifacts.py`

```python
find_artifacts(raw, good, pred, rr, sigma, pixscale, sb=0.0)
    -> (mask, area_arcsec2, flood_area_arcsec2)
```

Only the catastrophic regime lives here. Broad low-surface-brightness structure is
invisible to any per-pixel threshold and belongs to the plane's bin rejection and the
far-field witness. Detection needs a rendered catalog scene to define "unclaimed", so
a blind scene skips it. Deeply negative counterparts are already `nodata` at stamp
preparation.

**Invariant:** the absolute floor `ARTIFACT_SB_MIN` exists because the 20σ condition
is depth-*relative*. On a deep stack 20σ is µ ≈ 23, well inside ordinary astrophysics.
An instrument artifact of the bleed/streak class is brighter than the **sky itself**,
not merely brighter than the noise.

#### `render.py`

```python
conv_same(img, psf) -> np.ndarray                    # cached kernel transform
sersic_extent_px(reff_px, n, frac=0.9999) -> float
sersic_profile(params, shape_2d) -> np.ndarray       # [ampl, reff, n, ellip, theta, x0, y0]
render_sersic(params, shape_2d, psf) -> np.ndarray
render_sersic_boxed(reff_px, n, ellip, theta, x0, y0, shape_2d, psf) -> np.ndarray
render_nuker(rb_px, beta, ellip, theta, x0, y0, shape_2d, psf, pixscale) -> np.ndarray
moffat_wings(counts, fwhm_px, x0, y0, shape_2d) -> np.ndarray
sersic_total(ampl, reff_px, n, ellip, cf) -> float
ampl_from_total(counts, reff_px, n, ellip) -> float
pa_map(wcs, x, y) -> (t0, slope)
```

`render_sersic_boxed` and `render_nuker` return **unit-amplitude shape columns**; the
amplitude solve scales them. `render_sersic_boxed` renders on an adaptive box
(the 99.9%-flux radius plus the kernel footprint, quantized up to a multiple of 32 px
so solver trials reuse a few FFT sizes), matching the full-frame render to ≤ 1e-4 of
total flux, and falls back to the full frame when a box would not help.

`conv_same` caches kernel transforms by **PSF object identity** and identity-checks
on every hit, so a stale transform can never serve a different kernel that inherited
the same `id`. Build one PSF array per band and reuse it.

`pa_map` gives a local affine sky-PA → pixel-theta map so a solver can vary a position
angle without a WCS round trip per trial; the profiles it feeds are 180°-symmetric, so
a branch-cut wrap between the two anchors is harmless.

`sersic_profile` floors `reff_px` at 0.3 px and floors the **minor axis** at the same
sampling scale — pixel-center evaluation of a sub-pixel sliver underflows to a
numerically empty image, which is an explosive normalized design column.

#### `sersic.py`

```python
moffat_psf(fwhm_arcsec, pixscale, *, beta=3.0, size=25) -> np.ndarray
sersic_basis(shape, fwhm_arcsec, pixscale, stamp_shape, *, oversample=3) -> np.ndarray
pa_east_of_north(stamp_wcs, cx, cy, theta_rad) -> float
theta_from_pa(stamp_wcs, cx, cy, pa_deg) -> float
fit_sersic_shape(stamp, sky_std, cx, cy, pixscale, seeing_arcsec, *,
                 mask=None, fit_radius_arcsec=12.0) -> dict
```

The standalone single-Sersic shape fit — the shape source for the SPHEREx forced
model and for pinning the engine's target profile under `--mode sersic`. It fits on a
background-subtracted sub-stamp with **no neighbor handling**; explicit parameters or
the scene fit's own target refit are the trusted paths.

**Invariant:** position angles cross module boundaries as **degrees east of north**
and convert to/from pixel-frame theta through each image's WCS, so a shape fit on one
instrument transfers to any other orientation. Fitted `n` and `r_eff` are
PSF-sensitive: an error in the assumed seeing maps directly into them.

#### `calibrate.py`

```python
load_image(path) -> (data, wcs, header)     # first HDU with a 2D image
pixel_scale_arcsec(wcs) -> float
hst_ab_zeropoint(photflam, photplam) -> float
calib_factor(calib, header) -> float
```

| `calib` key | Applies to | Factor |
|---|---|---|
| `nmgy` | Legacy cutouts/bricks, SDSS frames | `3.631` |
| `photzp` | CFHT MegaPipe (AB `PHOTZP`) | `10^(-(PHOTZP - 23.9)/2.5)` |
| `ps1` | PS1 stacks (`ZP = 25 + 2.5 log10 EXPTIME`) | `10^((23.9 - zp)/2.5)` |
| `hst` | drizzled e/s (`PHOTFLAM`/`PHOTPLAM`) | `10^((23.9 - zp_ab)/2.5)` |

#### `engine.py` — Stage 8: the per-galaxy driver

```python
prepare_scene(coord, *, phot_dir, out_dir, aperture_arcsec, cutout_half_arcsec,
              legacy_dr=LEGACY_DR_DEFAULT, registry_path=None) -> dict
order_bands(products) -> list
measure_band(product, coord, scene, ref, caches, *, aperture_arcsec,
             cutout_half_arcsec, rgrid, scene_aperture_arcsec=None,
             target_shape=None, registry_update=False, pin=None,
             dump_dir=None) -> (measurement, new_ref)
```

`prepare_scene` returns `{cat, stars, patches, registry, registry_path}`. Catalog
queries are cache-first under `<phot_dir>/scene/`. The cone radius is
`max(QUERY_RADIUS_AS, stamp half-diagonal + QUERY_PAD_AS)` — it must reach past the
stamp's **corners** or corner sources are simply absent. The cache filename carries
the radius only when the stamp pushed it past the recipe floor, so a larger-stamp run
cannot silently reuse a smaller cone.

`measure_band` returns `new_ref` non-`None` only on a band that solved seat shapes.
`measurement` carries the flux, error, witnesses, arrays for QA, and the geometry.

**Invariants.**
- A query that **finds nothing** (off the survey footprint) yields an empty catalog
  and the engine measures blind. A query that **fails** raises: a service outage must
  not silently downgrade the measurement.
- The unconvolved-profile cache is keyed by component name and invalidated whenever
  the stamp shape or center changes. The caller owns grid-identity verification.
- A catalog "wreck" (any `rchisq` column past `GATE_RCHISQ_MAX`) may not claim pixels
  in the artifact test: it is the catalog's echo of the artifact itself.
- A component is voided by artifact masking on **footprint ownership** (the mask
  covers the majority of its render above σ), never on center membership — a broad
  source nicked by a narrow trail keeps its column and solves from its clean pixels.
- A masked-mode (in-zone reverted) star's predicted footprint leaves the *solve's*
  pixel set; the measurement-side mask and fill cover it after the fit.
- A pinned band is treated as **self-contained**: `ref` must be `None`, and it neither
  solves nor harvests.
- Artifact holes ride the same fill channel as neighbor masks, at every radius. Left
  out, the curve of growth would count masked artifact pixels as zero flux.

### 4.3 Image providers (`src/sedphot/images/`)

Contract:

```python
fetch(coord, *, bands, size_arcsec, cache_dir, **options)
    -> list[ImageProduct] | ProviderResult
```

`IMAGE_PROVIDERS = {sdss, panstarrs, legacy, cfht, hst}` — dict order is the `--all`
run order. Downloads cache under `Photometry/<Instrument>/`.

| Module | Source | Notes |
|---|---|---|
| `legacy.py` | legacysurvey.org `fits-cutout`, NOIRLab Data Lab cutout, NERSC brick coadds | Two routes. **Cutouts** (default) are fast and sized to the request but serve no inverse variance; the two cutout hosts are rotated with `try_services`, so a dead primary costs one failed call rather than a backoff cycle. **Bricks** (`use_bricks`) give image + invvar per band at tens of MB. Layer and hemisphere follow the release and the Dec 32.375° boundary, with the other hemisphere tried on a miss. An all-zero cutout is treated as no coverage. Brick identity is read from cached filenames first, so a re-measure skips TAP entirely. |
| `cfht.py` | CADC SODA, collection `CFHTMEGAPIPE` | Candidate stacks are ordered by how well their footprint centers the target, then the first that actually **covers the science aperture** wins (same 95% rule as the measurement gate). When no single stack covers it, overlapping tile-edge stacks are combined by `common.mosaic_first_valid`. A stack with no `PHOTZP` is renamed `*.nophotzp.fits` and stays in the cache as the record that this band was already tried. A CADC outage falls back to whatever stacks are already cached. |
| `panstarrs.py` | `ps1filenames.py` + `fitscut.cgi` | Coverage is Dec > −30. Cache-complete short circuit skips the listing query. |
| `sdss.py` | `astroquery.sdss` | The frame is resolved **once** by `query_region` and each band fetched from it by `(run, rerun, camcol, field)`; `get_images(coordinates=...)` is unreliable. Full frames are saved; `size_arcsec` is unused. A band that fails after the frame resolves is dropped and the others kept. |
| `hst.py` | MAST HAP-SVM | Visit group ranked by closest pointing, then most filters, then deepest exposure. DRC and DRZ both accepted; SCI/WHT located by `EXTNAME`. Each mosaic is split into plain sci/wht files so the `ImageProduct` contract applies unchanged. Full mosaics are downloaded; `size_arcsec` is unused. |

`common.py` holds two shared guards:

```python
warn_undersized_cache(path, size_arcsec, label) -> bool
mosaic_first_valid(planes, coord, size_arcsec, pixscale) -> (data, wcs)
```

**Invariant:** image caches are keyed by band alone, never by size, so a re-run with a
larger `--cutout-size` silently reuses the smaller file. `warn_undersized_cache` makes
that loud; nothing is deleted or refetched automatically.

`mosaic_first_valid` reprojects every plane onto a target-centered TAN grid and takes
the **first** covering plane's value at each output pixel — tiles fill each other's
clipped side without averaging separately-calibrated pixels. Uncovered output is NaN,
never a zero fill.

### 4.4 Catalog providers (`src/sedphot/catalogs/`)

Contract:

```python
query(coord, radius_arcsec, **options) -> ProviderResult
```

`CATALOG_PROVIDERS = {galex, sdss, jplus, panstarrs, legacy, allwise, hst}` — dict
order is the `--all` run order (blue to red, HST last because its per-filter
downloads are the slow step).

| Module | Catalog / service | Photometry | Band labels |
|---|---|---|---|
| `galex.py` | GUVcat_AIS via VizieR (`II/335/galex_ais`) | native AB mags; MW transmission from the catalog's own E(B−V) | `GALEX_FUV`, `GALEX_NUV` |
| `sdss.py` | SDSS DR17 via `astroquery.sdss` | `cModelMag` (approximates a galaxy total); `mode == 1` primaries only; transmission from `extinction_*` | `SDSS_u..z` |
| `jplus.py` | J-PLUS DR3 `jplus.MagABDualObj` via the CEFCA TAP sync endpoint (POSTed directly) | `MAG_PSFCOR` | `JPLUS_<filter>`, 12 bands |
| `panstarrs.py` | PS1 DR1 via VizieR (`II/349/ps1`) | PSF magnitudes | `PS1_g..y` |
| `legacy.py` | Legacy Tractor via NOIRLab Datalab TAP | optical broadbands + unWISE forced photometry from the same table; per-band `mw_transmission` carried | `Legacy_<band>`, `WISE_Wn` |
| `allwise.py` | AllWISE `allwise_p3as_psd` via IRSA | `w*mpro` profile-fit, Vega→AB | `WISE_Wn` |
| `hst_hap.py` | HAP point/segment catalogs via MAST | prefers segment `MagSegment` (isophotal); falls back to point `MagAp2` with a warning | `HST_<FILTER>` |

`gaia.py` is **not** a photometry provider and is not in the registry. It is a
scene-input module: `query_cone(coord, radius_arcsec, *, cache_path=None) ->
DataFrame` over `gaia_dr3.gaia_source`, returning position, G magnitude, the
five-parameter astrometric solution with errors, and RUWE. Unlike the
`ProviderResult` providers, a TAP failure here **raises** after retries.

`legacy.py` also carries the scene-catalog surface:

```python
scene_cols(dr) -> tuple[str, ...]
query_scene(coord, radius_arcsec, *, dr=LEGACY_DR_DEFAULT, min_flux_nmgy=0.5,
            cache_path=None) -> pd.DataFrame
shape_from_tractor(type_, sersic_n, shape_r, e1, e2) -> dict | None
query_shape(coord, radius_arcsec=2.0, *, dr=LEGACY_DR_DEFAULT) -> (shape, origin) | None
```

**Invariants.**
- `LEGACY_DR_DEFAULT` is one value for every verb. Catalog rows, cutout pixels, and
  the scene catalog must come from the same reduction, so no stage may default
  differently.
- `scene_cols` is built from the release, because DR9 has no i-band columns and a
  `SELECT` naming a missing column fails outright.
- `query_scene`'s flux floor applies to the **brightest** optical band; an r-only
  floor silently drops very red sources that are bright in z and invisible in r —
  unmodeled, unmasked point sources in the red bands.
- `query_scene` and `gaia.query_cone` write their cache and then **read it back**, so
  the network path returns exactly what a later cache hit returns (the CSV round trip
  normalizes dtypes).
- `query_shape` distinguishes a **service failure** (raises `RuntimeError`) from **no
  usable extended shape** (returns `None`), so a caller can refuse to substitute a
  degenerate image-fit shape for a transient outage.
- Negative catalog fluxes are legitimate non-detections and are preserved. Sentinel
  magnitudes (SDSS −9999, J-PLUS 99, PS1 NaN, AllWISE null `sigmpro`) are skipped per
  band rather than propagated.

---

## 5. The output contract

### 5.1 The table schema

Fifteen columns, defined in `schema.py`.

| # | Column | Contract |
|---|---|---|
| 1 | `band` | `<Instrument>_<filter>` — `Legacy_g`, `PS1_g`, `GALEX_FUV`, `HST_F475W` |
| 2 | `flux_uJy` | Microjansky, AB. **Negative values are legitimate non-detections and are preserved.** |
| 3 | `flux_err_uJy` | **Statistical only.** Floors and inflation belong to the SED fitter. |
| 4 | `mag_AB` | AB view of the same measurement; NaN when flux ≤ 0 |
| 5 | `mag_err` | as above |
| 6–7 | `target_ra`, `target_dec` | the requested position (deg) |
| 8–9 | `match_ra`, `match_dec` | the matched source's position; for measured rows, equal to the target |
| 10 | `sep_arcsec` | target-to-match separation; 0 for measured rows |
| 11 | `flags` | provider-specific for catalog rows; **QA tokens** for measured rows (§5.2) |
| 12 | `source` | measurement provenance string (§5.3) |
| 13 | `retrieved` | ISO date the value was pulled |
| 14 | `mw_transmission` | per-band MW transmission where the provider supplies it; NaN otherwise |
| 15 | `dered_applied` | True only when flux/mag have been corrected |

**Columns 1–12 (`BASE_COLS`) are a frozen contract** — order and names do not change,
because `overlay.py` and downstream analysis select on them. `EXTRA_COLS` is
append-only: new columns go at the end.

**Table invariant:** no two rows may share `(band, source)`. Enforced by
`rows_to_frame`, which raises on violation.

### 5.2 The `flags` QA tokens

Measured rows only. `aperture.qa_flags` packs the decisive witnesses as
`key=value;key=value;…` so downstream selection filters without opening a figure.
The full witness dict rides the sidecar.

| Token | Witness key | Meaning |
|---|---|---|
| `cov=` | `cov` | fraction of aperture pixels with real data |
| `maskfrac=` | `maskfrac_ap` | masked fraction of the aperture |
| `twinfrac=` | `twinfrac` | fraction of masked aperture pixels with a valid mirror |
| `nbsub=` | `nbsub_ap_uJy` | neighbor flux subtracted inside the aperture |
| `excess=` | `excess_growth_uJy` | curve growth past the aperture the target model cannot account for |
| `pedb=` | `ped_b_sb` | residual uniform-background term of the curve (µJy/arcsec²; 0 when the plane is right) |
| `conv=` | `r_conv_as` | radius where the curve plateaus and holds; `-1` when it never does |
| `bg=` | `bg_sb` | fitted background level (µJy/arcsec²) |

Conditional tokens:

| Token | Emitted when |
|---|---|
| `far=` | the stamp had enough far field to measure an independent level. A sign or scale contradiction with `bg=` is the background-ownership warning — a flag, never a demotion. |
| `art=` | artifact area was masked (arcsec²) |
| `mesh=` | the residual mesh's flux inside the aperture (µJy) |
| `leash=N` | N amplitudes sit on a bound (the zero non-negativity floor excluded) |
| `leashhi=N` | N of those sit on a **ceiling**: the stamp wanted more light than the component can supply. Absent when none do |
| `refit=` | the target refit's flux over its catalog flux (native scene band only) |
| `atbound=N` | N shape parameters sit on a box bound |
| `reg=N` | N registry entries were consumed |
| `scene=none` | blind measurement — no catalog components at all |

### 5.3 The `source` string vocabulary

`schema.SOURCE_PREFIXES` maps a provider token to its `source` prefix.

| Token | Prefix | Emitted by |
|---|---|---|
| `legacy` | `Legacy_` | `Legacy_DR9` / `Legacy_DR10` |
| `unwise` | `unWISE_Legacy_` | `unWISE_Legacy_DR9` / `_DR10` |
| `allwise` | `AllWISE` | `AllWISE` |
| `galex` | `GALEX_` | `GALEX_GUVcat_AIS` |
| `sdss` | `SDSS_` | `SDSS_DR17_cModel` |
| `panstarrs` | `PanSTARRS_` | `PanSTARRS_DR1` |
| `jplus` | `JPLUS_` | `JPLUS_DR3_PSFCOR` |
| `hst` | `HST_HAP_` | `HST_HAP_segment` / `HST_HAP_point` |
| `aperture` | `sedphot_aperture_` | `sedphot_aperture_scene_<err_model>` |
| `sersic` | `sedphot_sersic_` | `sedphot_sersic_scene_<err_model>` |

`<err_model>` is `ivm`, `skyrms`, or `empap`.

**Frozen contract, two rules.** Append new kinds; never rename an existing one. The
set must stay **prefix-free** — no entry may be a prefix of another, so `startswith`
matching is unambiguous (`Legacy_` must never match `unWISE_Legacy_*`).

`remeasure` emits its own vocabulary in a **different** table shape
(`band, flux_uJy, mag_AB, aperture_as, mode, source`), not the schema table:

| `source` | Meaning |
|---|---|
| `aperture_remeasure:<rev>` | interpolation of the stored empirical curve |
| `sersic_forced_remeasure:<rev>` | interpolation of the stored forced-shape model curve |
| `sersic_fitted_remeasure:<rev>` | …of the per-band free-target model curve |
| `reconstruct_<shape>:<rev>` | beyond-grid rebuild from the pinned fit |
| `solved_<shape>:<rev>` | beyond-grid band the sidecar could not pin, so it was **solved** |

`<rev>` is the sidecar's `git_rev`.

### 5.4 The provenance sidecar

`provenance.write_sidecar` writes `<product-stem>.provenance.json` next to every
product. Automatic fields, always present and never overridable:

```
product, written, sha256_16, package, package_version, git_rev, git_dirty
```

The measurement sidecar (`<label>_measured.provenance.json`) additionally carries:

```
kind                 "aperture_photometry" | "sersic_photometry"
target               {name, label, ra_deg, dec_deg}
instruments          the provider list
mode, aperture_arcsec, cutout_arcsec
sersic_shape         the forced shape and its origin, or null
legacy               {dr, bricks}   (null when Legacy was not requested)
scene                {n_catalog_rows, n_confirmed_stars, patches,
                      registry_path, registry_updated,
                      recipe: recipe.snapshot()}
per_band             {band_key: witness}     -- in engine.order_bands order
```

Each `witness` (built by `aperture.witness_row`, extended by `engine.measure_band`):

| Group | Keys |
|---|---|
| flux and growth | `f_ap_uJy`, `aperture_as`, `excess_growth_uJy`, `model_own_growth_uJy`, `m_ap_fit_uJy`, `m_ap_cat_uJy` |
| coverage and masks | `cov`, `maskfrac_ap`, `twinfrac`, `nbsub_ap_uJy`, `flood_uJy` |
| background | `bg_sb`, `bg_tilt_sb`, `bg_rej_bins`, `farfield_sb`, `alt_track_sb`, `ped_b_sb`, `ped_rms_uJy`, `mesh_ap_uJy` |
| convergence | `r_conv_as`, `leash_bound`, `leash_detail` |
| PSF | `seeing_as`, `seeing_src` |
| scene | `n_comps`, `gated`, `seat_owners`, `stars`, `registry_consumed`, `artifact_as2`, `artifact_uJy`, `artifact_flood_as2` |
| attribution | `resid_unmasked_ap_uJy`, `fill_vs_model_ap_uJy`, `target_model_uJy`, `target_model_free_uJy`, `target_refit_x_cat` |
| shapes | `solve`, `solve_free`, `shapes` (see below) |
| reconstruction | `fit_state` |

`fit_state` is what a re-derivation needs without a solver:

```
amps             [[owner, amplitude_uJy], ...]   -- in the order the fit built them
bg_coefs         [level, x_tilt, y_tilt]
mesh             the residual mesh's re-evaluation record, or null
rgrid            the curve-of-growth radii
enclosed_uJy     the empirical (neighbor-subtracted) curve
model_cog_uJy    the forced-shape target model's curve
model_cog_free_uJy   the free-target model's curve (gating targets only)
```

The three shape records are distinct and must not be conflated:

| Record | Covers | Grid key |
|---|---|---|
| `solve` | only what **that** solve varied. On a transfer band: the neighbor seats alone, **not** a full seat vector. | `pix_ref` |
| `solve_free` | the free-target pass's **full** seat vector — the only record of a per-band free shape outside the mutable registry | `pix_ref` |
| `shapes` | the **whole** seat list the band was measured with, on its own grid, however each shape got there | `pix` |

`shapes` is what a reconstruction re-renders. `_build_pin_by_band` prefers it and
length-guards anything else.

**Invariant:** the sidecar is the source of truth for re-reporting. `remeasure` reads
it and never re-solves inside the grid, and `reconstruct` reads it — including the
`registry_consumed` snapshot, ahead of the live registry — so a fit reproduces even
after other galaxies have rewritten the shared registry.

### 5.5 Other products

```
<out-dir>/Photometry/
    <label>_catalog.csv        (+ .provenance.json)
    <label>_measured.csv       (+ .provenance.json)
    <label>_sed.png
    <label>_overlay.png
    coverage_catalogs.json     per-provider status, catalog run
    coverage_measure.json      per-provider status, measurement run
    QA/growth_curves.png       every measured band, one overlay
    QA/remeasure_R<N>as/       scoped figures from a beyond-grid rebuild
    Legacy/ PanSTARRS/ SDSS/ CFHT/ HST/   cached images + QA/ per-band figures
    scene/                     cached scene inputs (Tractor, Gaia)
    SPHEREx/table_photometry.<tag>.csv   raw per-visit × channel table, verbatim
    SPHEREx/extractions.json             the tag decoder ring
```

`coverage_*.json` maps provider → `{status, n_rows, message, radius_used_arcsec}`.

---

## 6. Extension points

### 6.1 `patches.json` — per-galaxy custom knowledge

An optional file in the galaxy directory (`recipe.PATCH_FILENAME`), read by
`prepare_scene`. **No file means pure catalog behavior.** Unknown keys are warned
about and ignored; required sub-keys are validated and a missing one raises.

| Key | Effect |
|---|---|
| `replace_rows` | `[{ra, dec, with: [{...}, ...]}]` — swap one catalog row for one or more replacements. Each replacement inherits every column of the replaced row and overrides the ones it names; `uJy` is rederived from `flux_r`, so replacements set `flux_r`. |
| `free_seats` | `[{ra, dec, snap?}]` — grant a named companion its own free Sersic seat (`seats.build_seats`). `snap: true` snaps to the nearest local maximum, unless that peak lands on the target. |
| `snap_gated` | `true` — snap every gated seat's center to its nearest local maximum. |
| `target_refit` | `false` — disable the standard target refit. |
| `target_halo` | `true` — replace the target refit with a gated-style Sersic core + Nuker halo pair. Needs an extended target shape; falls back to the standard refit otherwise. |
| `target_system` | `[{ra, dec}]` — declare components as the target's own light (§2). |
| `harvest_target` | `true` — force the target seat into the registry harvest even when it does not gate. |
| `comment` | free text; ignored. |

A patch request that lands on no component within `recipe.PATCH_MATCH_AS` (2″) is
skipped **loudly**, never silently. Patch-named positions are also exempt from the
target-shred rule — declared human knowledge wins over the blind rule.

### 6.2 The cross-field registry

`--registry FILE` consumes; `--registry-update` also writes back. Entry structure:

```json
{
  "J<position key>": {
    "ra": ..., "dec": ...,
    "components": {
      "Legacy_r": [ {"kind": "sersic", "ra": …, "dec": …, "ellip": …, "pa": …,
                     "reff_as": …, "n": …,
                     "flux_ref": …, "flux_home": …, "vantage": "target"} ],
      "…": []
    },
    "tombstones": { "Legacy_z": "solve hit its evaluation cap" }
  }
}
```

Consumption looks up `band_key` first (`Legacy_r`), then falls back to the
instrument-level key. Keys are position-derived (`registry_name`, IAU style) and
coalesced onto any existing entry within `REGISTRY_KEY_COALESCE_AS` (1″), because a
seat anchor drifts across bands and two keys for one source would subtract the same
light twice.

**Operational rules.** The registry is mutable state shared by every galaxy measured
against it. `harvest_seats` updates the loaded dict in memory as each band solves; the
file is rewritten once, at the end of a run, and only with `--registry-update`.
Updates are last-writer-wins with no locking, so `--registry-update` sweeps must run
**one galaxy at a time**.

### 6.3 `recipe.py`

Every science constant lives there and is snapshotted into every measurement sidecar,
so a measured value is only reproducible against the exact recipe that produced it.
Changing a constant changes future measurements; it does not change anything already
written.

`recipe.sky_floor(rmin)` is the one runtime override (`measure --sky-rmin`). It is
scoped to a call, recorded in the snapshot, and read back by `reconstruct`.

### 6.4 Adding a catalog provider

1. Create `src/sedphot/catalogs/<name>.py` exposing
   `query(coord, radius_arcsec, **options) -> ProviderResult`.
2. Write a private `_query_once(coord, radius_arcsec) -> list[dict]` that returns `[]`
   for both "no result" and "service failure" — **never raise** — and wrap it in
   `retry.with_expanding_radius`. Add `retry_transient` or `query_vizier_mirrors`
   around the transport if the service is known to flap.
3. Build rows with `schema.make_row`. Choose band labels `<Instrument>_<filter>` and a
   `source` string whose prefix you add to `schema.SOURCE_PREFIXES` — keeping the set
   prefix-free.
4. Skip sentinel/non-detection values per band rather than propagating them. Carry
   `mw_transmission` if the catalog supplies one.
5. Register in `catalogs/__init__.py`. Dict order is the `--all` run order.
6. Add a marker style in `overlay.PROVIDER_STYLES` keyed by the new token, and
   wavelengths in `bands.WAVE_UM` if the filters are new.
7. Add an `EXT_COEFF` entry in `dered.py` if the bands should be dereddenable
   (otherwise they fall to tier 3 and stay as-measured).

### 6.5 Adding an image provider

1. Create `src/sedphot/images/<name>.py` exposing
   `fetch(coord, *, bands, size_arcsec, cache_dir, **options)
   -> list[ImageProduct] | ProviderResult`.
2. Cache downloads under `cache_dir` and reuse them. Call
   `common.warn_undersized_cache` before reusing a cutout. Consider a cache-complete
   short circuit so a fully offline re-measure never touches the service.
3. Give every product a `calib` key. A new key needs a branch in
   `calibrate.calib_factor`; supply `invvar_path` when the archive serves an
   inverse-variance map.
4. Set `seeing_arcsec` (the fallback when no star qualifies for an empirical PSF) and
   `wave_um`.
5. Register in `images/__init__.py`, and add the instrument's cache directory name to
   `pipeline.INSTRUMENT_DIRS`.
6. If the provider needs a per-run option, thread it through `pipeline.run_measure`'s
   per-provider `options` dict and add the flag in `cli._add_measure_args` (not in the
   `measure` subparser) so `run` accepts it too.
7. Add the instrument to `qa.INSTRUMENT_STYLE` / `INSTRUMENT_MARKER` for the figures.

---

## 7. Behavior worth knowing

These are the non-obvious behaviors most likely to surprise a maintainer. Each is
verified against the code.

**A catalog archive outage usually looks like `no_match`, not `error`.** Every catalog
provider's `_query_once` swallows service failures and returns `[]`;
`with_expanding_radius` then exhausts its five doubling attempts and the provider
reports `no_match`. That is indistinguishable from a genuinely empty field. An
unexpected `no_match` on a target you know is covered is worth re-running later. The
exceptions: the Legacy *image* provider and `query_shape` do surface errors, and
`gaia.query_cone` / `query_scene` raise outright — a scene-catalog outage must not
silently downgrade a measurement to blind.

**`--dered` corrects only the catalog table.** `dered.apply_dereddening` is called
from `run_catalogs` alone; `run_measure` never calls it, so every measured row keeps
`dered_applied = False`. Drawing both tables on one axis therefore mixes corrected and
uncorrected fluxes. `run_sed` detects the disagreement across the tables'
`dered_applied` columns and prints
`WARNING … the plotted fluxes are NOT on a common extinction scale`. Nothing in the
figure itself shows it.

**The registry makes measurement order matter while it fills.** A source registered by
galaxy A is *consumed frozen* in galaxy B's scene rather than solved there, so B's
flux can depend on whether A ran first. Consumption also removes the seat, so once a
source is registered every later field freezes it — the order dependence is transient
by construction, and a re-run against a populated registry is uniform. To exercise the
free-solve path deliberately, run with no registry at all.

**`remeasure` behaves differently inside and past the stored grid.** Inside, it is a
millisecond interpolation of values already on disk — no fetch, no scene, no solve.
Past it, `reconstruct` re-reads the cached images (fetching any that are missing),
rebuilds the whole scene, and re-renders; a band the sidecar cannot pin is **solved**,
which is a fresh measurement wearing a reconstruction label, so it comes back tagged
`solved_*` and named on stdout. The boundary is announced on stdout when crossed.

**`--shape` is inert inside the grid under `--mode aperture`.** The empirical curve is
a measurement of the pixels, not a rendering of a chosen shape. `remeasure` accepts the
flag and warns rather than refusing, because the same request *past* the grid is
meaningful and refusing would block a request spanning both. The `measure` verb, by
contrast, refuses the analogous combination outright.

**A per-band measurement failure is caught and printed, not raised.** In
`run_measure`'s band loop, `ApertureCoverageError` demotes the band and any other
exception prints `<Inst> <band> FAILED: …` and continues. The provider's status
becomes `ok` if any band survived, `no_coverage` if only demotions occurred, and
`error` only if images were fetched and *every* measurement failed. The `measure` verb
does not inspect the returned frame, so **a run can exit 0 with fewer bands than
requested**. Check `coverage_measure.json` and the row count, not the exit status.
(`run` exits nonzero only when a whole *stage* raised.)

**Reconstruction runs under `contextlib.redirect_stdout`.** `reconstruct` redirects
the pipeline's progress log to a buffer so a re-report prints the table alone. Because
a dead band is reported by *printing*, its message lands in that buffer — so
`reconstruct` compares the sidecar's band list against what came back, reports the
difference, and replays the `FAILED` / `no_coverage` lines it finds. A new failure mode
that neither prints nor drops a band would be invisible.

**The scene is built at one aperture and integrated at another.**
`drop_target_shreds` and the star zone are both scoped to `scene_aperture_arcsec`, so
asking for a larger integration radius while rebuilding would reclassify catalog rows
as target substructure and delete sources the original fit had modeled and subtracted.
`reconstruct` pins the scene to the sidecar's own science aperture for exactly this
reason. Any new caller of `run_measure` that changes `aperture_arcsec` must decide
deliberately whether `scene_aperture_arcsec` moves with it.

**`solve.params` is not a seat vector.** On a transfer band it covers only the free
(neighbor) seats. `witness['shapes']` is the full vector. Any consumer that unpacks a
stored vector against a seat list must length-check.

**`solve.seconds` is the last warm iteration only**, not the total solve time. The
alternation runs `solve_shapes` repeatedly and only the final call's timing survives.

**Fetch options are part of the fit.** `reconstruct` replays `cutout_arcsec`,
`BG_RMIN_AS`, and the Legacy `dr` / `bricks` flags, because brick coadds and viewer
cutouts are different pixels on different grids. The sidecar does **not** record an
HST program restriction, so a reconstruction of an HST measurement fetched under
`--hst-proposal-id` re-selects its visit group by the provider's own ranking.

**A stack's coverage is judged at the science aperture, twice.** The CFHT provider
applies the same 95% rule at fetch time (driving the tile-edge mosaic) that
`check_coverage` applies at measure time, at the same aperture the run uses — so a
stack that would be demoted downstream is rejected upstream instead of accepted and
then failing.

**SPHEREx aborts rather than substitutes.** When the default Tractor shape lookup
fails — or the source has no extended shape — `run_spherex` returns an error result
instead of quietly falling back to an image fit. A written raw table is never
overwritten, so a wrong source model would be permanent; the shape must be one that
was asked for explicitly or looked up successfully.

**An untagged Legacy brick cache is assumed, loudly.** Brick names are shared across
data releases, so a file cached under the untagged name records no release. It is
reused under the requested one with an explicit `UNVERIFIED` warning and never
renamed — cached filenames are a downstream stability contract.

---

## 8. Testing

```bash
python -m pytest tests/
```

416 tests, **entirely offline**. No test performs network I/O: TAP clients, MAST
`Observations`, `requests`, and `astroquery` entry points are monkeypatched, and image
fixtures are synthesized in `tmp_path`. There is no `conftest.py` and no pytest
configuration — the suite runs from a bare invocation.

Coverage by area:

| Area | Files |
|---|---|
| Scene engine core — components, seats, the joint solve, the witnesses, one synthetic band end-to-end through the driver | `test_scene_engine.py` (the largest single file) |
| Reconstruction — plane/mesh round trips, `pinned_fit` grid rescaling, multi-seat owners, pin selection and its length guard | `test_reconstruct.py` |
| Re-reporting — interpolation and capping, mode/shape selection, the shortest-grid threshold, fetch-option replay, dropped-band reporting | `test_remeasure.py` |
| Background — bin statistics, rejection, mesh zeroing | `test_background.py`, `test_mesh_empap.py` |
| Stamp and gates | `test_stamp.py` |
| PSF resolution and kernel guards | `test_psf.py` |
| Stars — confirmation, profile measurement, revert paths | `test_stars.py` |
| Artifacts | `test_artifacts.py` |
| Rendering primitives | `test_render.py` |
| Scene catalogs — cache-first behavior, verified by a TAP stand-in that **raises on any contact** | `test_scene_catalogs.py` |
| Image providers — caching, CFHT stack selection and mosaicking, SDSS frame resolution | `test_image_cache.py`, `test_cfht_images.py`, `test_sdss_images.py` |
| CLI — flag refusal, `run`/`measure` option parity, exit codes | `test_cli.py` |
| Stage isolation and the SED dered warning | `test_run_stages.py` |
| Overlay — style keying, WCS checks, marker de-duplication | `test_overlay.py` |
| SPHEREx — configuration identity, tag stability, reuse rules, shape resolution | `test_spherex_config.py`, `test_spherex_shape.py` |
| Schema, units, bands, provenance, retry | `test_schema.py`, `test_units.py`, `test_bands.py`, `test_provenance.py`, `test_retry.py` |
| Tractor shape conversion | `test_legacy_shape.py` |
| AllWISE row construction | `test_allwise.py` |

The scene-engine and reconstruction tests are the ones that constrain behavior rather
than plumbing: they build synthetic scenes with known injected shapes and assert that
the solve recovers them, that a pinned rebuild reproduces the fit it was built from,
and that the shape records survive a full round trip through the sidecar.
