# sedphot

Galaxy in, SED photometry out. Give it a name or a position and it retrieves
catalog photometry from the common archives, fetches survey images and measures
uniform-aperture (or forced single-Sersic) fluxes, optionally pulls SPHEREx
spectrophotometry, and writes SED-ready tables with QA figures and provenance
sidecars.

```bash
pip install -e .

# resolve a name to a position
sedphot resolve --name "NGC 4889"

# catalog photometry from every archive, with per-provider fallback
sedphot catalogs --name M87 --all --out-dir Galaxies/M87

# fetch images and measure every band through one scene recipe
sedphot measure --name M87 --instruments cfht legacy \
    --aperture 12 --out-dir Galaxies/M87

# SPHEREx forced photometry; the Sersic shape comes from the Tractor catalog
sedphot spherex --name "NGC 4874" --out-dir Galaxies/NGC_4874

# everything for one galaxy, then a combined SED plot
sedphot run --name "NGC 4889" --out-dir Galaxies/NGC_4889
```

The per-galaxy verbs require `--out-dir`: products, image caches and the scene
cache all land under it. The multi-target verbs work from a target list or a
config instead and name their outputs individually (`--out`, `--report`,
`--registry-dir`, `--config`).

## Verbs

Per galaxy:

| Verb | What it does |
|---|---|
| `resolve`  | Name -> ICRS position (Sesame -> NED -> SIMBAD) + output label |
| `catalogs` | Closest-source photometry from the catalog archives -> `<label>_catalog.csv` |
| `measure`  | Fetch images, measure every band -> `<label>_measured.csv` + QA figures |
| `spherex`  | Raw SPHEREx spectrophotometry table (IRSA), forced-Sersic (default) or PSF model |
| `sed`      | Combined flux-vs-wavelength figure -> `<label>_sed.png`, or `<label>_spherex_sed.png` under `--spherex` |
| `overlay`  | Each provider's matched position drawn on the HAP color composite -> `<label>_overlay.png` |
| `remeasure`| Re-report band fluxes at a different aperture from a stored fit |
| `run`      | catalogs -> measure -> SPHEREx (opt-in) -> SED plot; stages are isolated (a dead stage is recorded and the rest still run) and the verb exits nonzero if any stage failed |

Many galaxies:

| Verb | What it does |
|---|---|
| `plan`          | Census a target list and decide the safe measurement order -> `<plan>.json` + `.csv` |
| `batch`         | Execute a plan: a grouped harvest pass, a registry merge, then an unbounded parallel pass |
| `spherex-plan`  | Group blended targets into joint SPHEREx jobs -> a reviewable config + `.csv` |
| `spherex-batch` | Run the joint SPHEREx jobs in a config -> one table per science member |

## Providers

Catalogs: `legacy` (Tractor via Datalab TAP; optical + unWISE-forced WISE,
MW transmission carried per band, dr9/dr10), `panstarrs` (VizieR),
`sdss` (DR17 cModel + native extinction), `galex` (GUVcat_AIS via VizieR),
`jplus` (DR3 PSFCOR via the CEFCA TAP), `allwise` (IRSA, Vega->AB),
`hst` (HAP point/segment catalogs via MAST).

Images (for `measure`): `legacy` (viewer cutouts, or NERSC bricks with
inverse variance via `--legacy-bricks`), `panstarrs` (fitscut stacks),
`sdss` (frames), `cfht` (MegaPipe stacks via CADC SODA), `hst` (HAP drizzled
mosaics, any instrument, DRC/DRZ).

Legacy cutouts come from either of two services, selected by
`--legacy-route {auto,viewer,noirlab}`. This is a reproducibility control,
not a convenience: the two routes do not frame identically, so under the
default `auto` a service outage silently splits a sample across two
griddings, and the resampling moves the PSF-star fit and hence the flux.
Pin the route for any sample meant to be internally comparable; a pinned
route reports an outage as an error rather than substituting the other
service. The sidecar records both the route asked for and, read back from
each image's own header, the route that actually answered.

Each provider reports its outcome into `coverage_*.json` and the run
continues, so one dead service never kills a fetch-all. The two interfaces
report different vocabularies: image providers return `ok`, `no_coverage` or
`error`; catalog providers return `ok` or `no_match`. A catalog service
outage is absorbed at the query boundary and surfaces as `no_match`, so an
unexpected `no_match` on a well-covered target is worth re-running later.

## Output conventions

```
<out-dir>/Photometry/
    <label>_catalog.csv  (+ .provenance.json)
    <label>_measured.csv (+ .provenance.json)
    <label>_sed.png
    <label>_spherex_sed.png                 broadbands + raw SPHEREx (sed --spherex)
    <label>_overlay.png
    coverage_catalogs.json / coverage_measure.json
    Legacy/ PanSTARRS/ SDSS/ CFHT/ HST/     cached images + QA/ figures
    QA/growth_curves.png                    all measured bands, one overlay
    scene/                                  cached scene inputs
    SPHEREx/table_photometry.<tag>.csv      raw per-visit x channel table,
                                            verbatim; one per extraction
                                            config (tag = <model>-<hash6>,
                                            or joint-<model>-<hash6>)
    SPHEREx/extractions.json                the tag decoder ring
```

The multi-target verbs write outside any one galaxy: `plan` and
`spherex-plan` write a plan/config JSON plus a flat `.csv` beside it, `batch`
writes one registry per harvest group and the merged frozen `registry.json`,
and both batch verbs write a per-target report and per-target logs. A joint
SPHEREx job also writes the raw multi-source table verbatim, plus any
ancillary member's spectrum, into its group directory.

`run` writes `<label>_sed.png` only. The SPHEREx figure is a deliberate
second pass (`sed --spherex`), so a missing `<label>_spherex_sed.png` means
"not generated yet", never "SPHEREx is absent".

Tables share one schema: `band, flux_uJy, flux_err_uJy, mag_AB, mag_err,
target_ra, target_dec, match_ra, match_dec, sep_arcsec, flags, source,
retrieved, mw_transmission, dered_applied`. The first twelve columns are a
frozen contract, because downstream consumers select on them.

Fluxes are microjansky, AB throughout. Errors are statistical only, so error
floors belong to the SED fitter. Negative catalog fluxes are legitimate
non-detections and are preserved. Fluxes are as-measured unless `--dered` is
passed, which applies to the catalog table; `sed` warns when it is asked to
plot tables on different extinction scales. Band labels are
`<Instrument>_<filter>`, and measurement provenance lives in `source`
(unWISE and AllWISE both label their bands `WISE_Wn` and differ in `source`).

## Measurement recipe

One recipe for every instrument, built around a fit to the whole scene:

1. **Scene** -- every Tractor catalog row near the target becomes a
   rendered component at its catalog shape. Design columns are normalized
   to unit in-stamp flux, so every fitted amplitude reads directly in
   microjanskys. A confirmed star is never subtracted from the data:
   inside the aperture plus a buffer it is masked and filled with no
   design column at all, and outside it the catalog component stays with
   a leashed amplitude. Both routes bound the damage a wrong star model
   can do; neither can excavate light that was never the star's.
2. **Joint fit** -- all component amplitudes solve together against a
   plane through sigma-clipped bin means (bin-level outlier rejection;
   the plane owns cutout-scale background), alternating until the
   background converges. A catalog row that declares its own misfit
   (bright and high reduced chi-square) additionally gets a shape solve,
   solved by variable projection with every amplitude re-fit at every
   trial. The target's own shape is always refit from the pixels -- the
   catalog informs the photometry through the neighbors.
3. **Measure** -- fitted neighbors and background are subtracted,
   residual neighbor light is masked (an intersection channel for the
   fitted models, a star channel, and an ambient flood channel that is
   symmetric on both escaped glow and over-subtraction holes), masked
   pixels are reconstructed from their point reflection through the
   target center (clamped by the model so holes are impossible), and the
   reported flux is the curve of growth at the aperture. The target
   model itself is never integrated into the measurement.

The background has one owner per spatial scale: the plane owns level and
tilt, a post-fit residual mesh owns smooth structure and is zeroed of any
uniform term on the plane's turf, and the Sersic components own compact
light. Sky is estimated only beyond `--sky-rmin`.

Bands are measured per instrument, reference band first: the reference
solves the shapes, and sibling bands re-solve neighbor shapes warm with
fluxes bounded to color-scaled reference values. The PSF is measured per
band from the field's own confirmed stars, with a Moffat fallback when no
star qualifies. A position with no Tractor coverage measures blind --
background and curve of growth only -- and says so in its flags.

The archive's inverse variance where it exists (Legacy bricks) and sky rms
otherwise are the *floor*. The reported error is the empirical
empty-aperture scatter, measured on the final residual, whenever enough
apertures can be placed and it exceeds the analytic value -- which is the
usual case, so most rows report the empirical number.

Off-footprint and blank pixels demote a band to `no_coverage` past 5% of the
aperture area. The seeing-scale core is held to a tighter threshold of its
own, and the peak itself is inviolable at any fraction, since no fill can
reconstruct it. Every measured row carries machine-parsable QA tokens in
`flags`: `cov` (aperture coverage), `maskfrac` (masked fraction of the
aperture), `twinfrac` (mirror-filled fraction of the masked area), `nbsub`
(neighbor flux subtracted inside the aperture), `mesh` (residual-mesh
contribution in the aperture), `excess` (curve growth past the aperture the
target model cannot account for), `pedb` (residual uniform-background term),
`conv` (radius where the curve plateaus and holds; -1 when it never does),
`bg` (fitted background level), `far` (an independent far-field level -- a
sign or scale contradiction with the fitted plane is the background-ownership
warning), `art` (artifact area), `leash` (amplitudes solved to a bound) and
`leashhi` (of those, how many at a ceiling -- the diagnostically significant
direction, a stamp wanting more light than the component can supply), plus
`refit`/`atbound`/`reg`/`scene=none` where they apply. The full per-band
witness set rides the `_measured.csv` provenance sidecar.

Two optional inputs extend the scene without touching the code:

- `<out-dir>/patches.json` -- per-galaxy custom knowledge: replace a
  blended catalog row with a known decomposition (`replace_rows`), grant a
  companion its own free shape (`free_seats`, optional `snap`), snap solved
  centers to image peaks (`snap_gated`), disable the standard target refit
  (`target_refit: false`), grant the target a Nuker halo (`target_halo`),
  write the target's own solved shape to the registry (`harvest_target`),
  or declare a companion to be part of the target rather than a contaminant
  (`target_system`, whose members freeze with the target on transfer bands).
  A `comment` key is accepted and ignored. No patch file means pure catalog
  behavior. A patches file is a science input and lives in the galaxy
  directory, so it travels only if copied deliberately.
- `--registry FILE` -- a cross-field registry of solved shared sources. A
  bright galaxy appearing in several targets' stamps is solved once and
  reused everywhere as a frozen, tightly-bounded component.
  `--registry-update` writes the current galaxy's solved shapes back.
  Updates replace the whole file with no locking, so a *bare* `measure`
  sweep with `--registry-update` must run one galaxy at a time. `batch`
  removes that constraint properly rather than by serializing -- see
  **Many galaxies** below.

`--mode sersic` reports the fitted target model's flux rather than the
aperture integral -- forced photometry through the same scene fit. The shape
is the standard reference-band refit, a fit on a chosen band
(`--sersic-from`), or explicit `--sersic-params`. Fitted `n` and `r_eff` are
PSF-sensitive, so explicit parameters from a trusted fit are the precision
path. The shape flags apply to `--mode sersic` only, and are refused rather
than ignored under `--mode aperture`.

`sedphot run` accepts every measurement option `measure` does, so the full
run never measures under different settings than the verb it drives. It
selects archives differently: `--catalogs` and `--images` each take provider
names, or `all` / `none` to run or skip that stage entirely, and `--skip`
drops a provider from both. Name the image providers explicitly for any
campaign -- `hst` is in the default set, so a bare run queries MAST for
every target, which is hundreds of megabytes per galaxy for typically no
measured band.

### Re-reporting from a stored fit

`sedphot remeasure <sidecar>` re-derives band fluxes from the immutable
provenance:

    sedphot remeasure Photometry/g1_measured.provenance.json \
        --mode aperture --aperture 10

The curve of growth is stored to 40", so changing the science aperture
inside that grid is an interpolation on values already on disk. Past the end
of the stored grid, aperture mode rebuilds the scene from the pinned fit and
integrates there, which re-reads the cached images and can solve any band
the sidecar cannot rebuild. The verb prints a line when a request crosses
that boundary. `--mode sersic` reads the fitted model's curve instead;
`--integrated` reports the total.

`--shape` picks which target shape the report is built on:

| | |
|---|---|
| `forced` (default) | the instrument's reference-band shape -- the one the science curve was built on |
| `fitted` | each band's own free-target shape, which the engine stores only where it solves the target twice |

Bands with no free-target record fall back to `forced`, name themselves in
the log, and say so in their `source` column. In `--mode aperture`, `--shape`
applies only past the stored grid, where the shape decides what gets
subtracted; inside the grid the empirical curve is a measurement of the
pixels, with no per-shape variant. `--write-qa` sends a beyond-grid rebuild's
figures to a scoped `QA/remeasure_R<N>as/` subdirectory, never the science QA.

## SPHEREx

`sedphot spherex` submits an IRSA forced-photometry job and writes the raw
per-visit x channel table verbatim, since quality cuts belong downstream. The
source model defaults to a forced Sersic -- a PSF model carries a chromatic
bias for extended sources -- with the shape resolved in order:

1. `--sersic-params N AXRATIO PA REFF` -- explicit, used as given
2. `--sersic-from <band>` -- fit on that band's image
3. default: the Legacy Tractor catalog shape (`type`, `sersic`, `shape_r`,
   `e1`/`e2` -> n, b/a, PA east of north; SER keeps its fitted index,
   DEV/EXP/REX fix n = 4/1/1). The TAP lookup is retried, and when it still
   fails -- or the source has no extended shape -- the verb aborts nonzero.
   A written table is never overwritten, so a wrong shape would be permanent.
   Proceed deliberately with `--sersic-params`, `--sersic-from`, or
   `--model psf`.

The shape's origin is recorded in the sidecar's model block. Every distinct
extraction configuration (model + shape + background region + MJD window)
owns its own `table_photometry.<model>-<hash6>.csv`, indexed in
`extractions.json`, so PSF and Sersic runs coexist without manual renames.
Re-requesting a configuration already on disk reuses it, provided the table's
own sidecar records that same configuration; a table no sidecar vouches for
is neither reused nor overwritten. Move a table aside deliberately to force a
re-fetch. `--mjd-range` restricts the job to a known-good visit window. The
verb exits nonzero when the fetch fails, so shell chains can trust `$?`.

### Blended neighbors: joint extraction

A SPHEREx pixel is about 6.2 arcseconds, so a companion within a few pixels
has its light in the target's aperture. The IRSA tool can fit several
positions simultaneously, which divides the light between them instead of
attributing all of it to one source, and `spherex-plan` / `spherex-batch`
drive that:

```bash
sedphot spherex-plan --targets targets.csv --out groups.json --blend-radius 45
sedphot spherex-batch --config groups.json --report groups.report.csv
```

A job's membership is a set of positions, each with its own frozen shape, so
it cannot be expressed on a command line -- hence a config file, and one
meant to be *read*: planning writes it (plus a flat `.csv` of every member),
a human reviews it, execution runs it. Planning works offline from each
field's cached Tractor scene catalog.

Each member is `science` (its spectrum is a product, and lands in its own
galaxy directory exactly where a single-position extraction would) or
`ancillary` (present only to absorb light; its spectrum is archived with the
group). So a joint job still leaves one table per galaxy, and nothing
downstream has to know a joint fit happened.

Three constraints are worth knowing before planning. A job takes at most 20
sources -- the tool keeps the first 20 rows of an uploaded list and drops the
rest *without an error*, so a larger group is refused here. A job is
point-source or elliptical as a whole, so a point-like member of a Sersic job
rides at a sub-threshold radius, which the tool reads as a point source. And
the tool does not validate an uploaded shape at all: it attempts the fit and
returns a table that looks like any other, so shapes are bounds-checked
before submission and the shape the tool echoes back is verified against the
one requested. On a mismatch the group writes no product and the raw table is
quarantined with no sidecar.

Membership is part of an extraction's identity: the same galaxy fit alone and
fit beside a companion are different measurements, so the joint tag hashes
the whole membership and leads with `joint-`, which keeps a solo glob from
ever matching a joint table. Both coexist in one directory.

The service runs two extraction jobs at once and queues the rest rather than
refusing them, so `--workers` above two is allowed and simply queues. Queued
time is budgeted separately from run time, so a deep queue cannot time out
its own tail.

## Many galaxies

Do not loop `sedphot run` over a target list. The registry is mutable state
shared across fields, and `--registry-update` rewrites the whole file with no
locking, so concurrent galaxies lose each other's entries -- while running
them serially forfeits hours for no reason.

```bash
sedphot plan  --targets targets.csv --out plan.json
sedphot batch --plan plan.json --registry-dir registries/ --report report.csv \
    --workers 3 --images legacy cfht sdss panstarrs --log-dir logs/
```

`plan` reads a target list (a name column, `ra_deg`/`dec_deg`, and optionally
`label`, `dir`, `priority` -- the same file `sed_fitting` reads, so one
catalog drives both tools), fetches each field's scene catalog, and splits
the sample in two:

- a **harvest pass** of the targets that both solve a shape worth
  propagating and have someone to propagate it to. These run in groups, the
  groups concurrently and each group's targets in sequence, and **every group
  writes its own registry file** -- so two writers to one registry never
  exist. The union is then merged and frozen, and a key claimed by two groups
  is a hard error, because the grouping guarantees it cannot happen.
- a **parallel pass** of everything else, run against the frozen registry
  with updates off. It writes no shared state, so its concurrency is bounded
  only by what the archives tolerate.

Two fields can share a registry entry only when they are closer than a
field's write reach plus its read reach, both derived from the stamp width,
plus a margin: a margin costs concurrency, a shortfall costs reproducibility.

Verify a sweep by counting bands in every table and every sidecar, not by
reading the report. A band that fails inside the measurement is logged and
skipped, but the run still exits zero with fewer bands than requested, and
the reference band -- measured first per instrument -- is the one most likely
to absorb an I/O stall.

## Position overlay

```bash
sedphot overlay --out-dir Galaxies/NGC_4889
```

The positional counterpart to `sed`: same tables, but it draws where each
provider *matched* rather than what it measured, on the Hubble Advanced
Products (HAP) color composite for the field. It catches the failure the flux
columns cannot show -- two providers confidently reporting photometry for two
different objects -- and reads only `match_ra`, `match_dec` and `source`,
which is why the first twelve table columns are a frozen contract. Marker
styles key on the `source` prefix vocabulary, so a provider that revs its
data release keeps its symbol.

The composite carries no WCS of its own, so the detection mosaic it was
rendered from supplies one: a ~380 MB download the first time a field is
used, cached under `Photometry/HST/` thereafter. `--wcs-from FITS` points at
a local file already on that drizzle grid; the pixel dimensions are checked
either way, and a mismatch refuses rather than misplacing every marker. The
verb needs HAP coverage, so it exits nonzero where there is none.

## Compact targets

The measurement defaults are galaxy-survey sized: a 12" aperture, a
curve-of-growth grid from 2" to 40", and sky estimated only beyond 15". A
compact source on an HST mosaic needs all three moved together --

```bash
sedphot measure --name "..." --instruments hst --aperture 1 \
    --cutout-size 20 --sky-rmin 3 --radii 0.1 0.25 0.5 1 1.5 2 2.5 \
    --out-dir Galaxies/compact_target
```

`--sky-rmin` is the target/sky boundary (`recipe.BG_RMIN_AS`): sky is
estimated only beyond it, and no pixel inside it votes on the background, on
star exclusion, or on artifact detection. It is scoped to the run and
recorded in the sidecar's recipe snapshot, so `remeasure` rebuilds the scene
on the same boundary the fit was solved on.

## Before you measure a sample

Three defaults are science commitments rather than settings, and each is
expensive to revisit once a sample is measured:

- **`--legacy-route`** -- leave it on `auto` and a service outage can split
  your sample across two griddings that do not frame identically. Pin it.
- **`--registry`** -- consuming a registry changes fluxes. `measure`
  deliberately does *not* auto-discover the sibling `registry.json` that
  `remeasure` does; that asymmetry is intentional, so that adding a registry
  to a sweep is always a stated decision.
- **the image provider list** -- name it explicitly. The default includes
  `hst`.

The other thing worth knowing early: the curve of growth is *stored*, out to
40 arcseconds. Changing the science aperture is therefore an interpolation on
data already on disk (`remeasure`), not a re-measurement -- which is the
difference between iterating in seconds and iterating in hours.

## Documentation

`docs/technical_manual.md` is the developer reference: module layout,
measurement flow, the project vocabulary (it leads with a glossary), the
output contract and the extension points.

`docs/sedphot_user_manual.pdf` is the operator's guide -- installing,
worked end-to-end examples, every flag explained, reading the QA figures, and
what to do when something goes wrong. It is built from
`docs/sedphot_user_manual.tex`; both are tracked, so a change to one needs a
rebuild of the other.

## Requirements

Python 3.11+; numpy, scipy, pandas, astropy, astroquery, matplotlib, pillow,
reproject, requests, defusedxml (see `pyproject.toml`).

Tests: `python -m pytest tests/` -- 511 tests, entirely offline.

## License

MIT
