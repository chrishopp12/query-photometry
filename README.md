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

Every verb that writes requires `--out-dir`: products, image caches and the
scene cache all land under it.

## Verbs

| Verb | What it does |
|---|---|
| `resolve`  | Name -> ICRS position (Sesame -> NED -> SIMBAD) + output label |
| `catalogs` | Closest-source photometry from the catalog archives -> `<label>_catalog.csv` |
| `measure`  | Fetch images, measure every band -> `<label>_measured.csv` + QA figures |
| `spherex`  | Raw SPHEREx spectrophotometry table (IRSA), forced-Sersic (default) or PSF model |
| `sed`      | Combined flux-vs-wavelength figure from the tables in `--out-dir` |
| `overlay`  | Each provider's matched position drawn on the HAP color composite -> `<label>_overlay.png` |
| `remeasure`| Re-report band fluxes at a different aperture from a stored fit |
| `run`      | catalogs -> measure -> SPHEREx (opt-in) -> SED plot; stages are isolated (a dead stage is recorded and the rest still run) and the verb exits nonzero if any stage failed |

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
    <label>_overlay.png
    coverage_catalogs.json / coverage_measure.json
    Legacy/ PanSTARRS/ SDSS/ CFHT/ HST/     cached images + QA/ figures
    QA/growth_curves.png                    all measured bands, one overlay
    scene/                                  cached scene inputs
    SPHEREx/table_photometry.<tag>.csv      raw per-visit x channel table,
                                            verbatim; one per extraction
                                            config (tag = <model>-<hash6>)
    SPHEREx/extractions.json                the tag decoder ring
```

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
   rendered component at its catalog shape (Gaia-confirmed stars are
   replaced by their own measured radial profiles and pre-subtracted).
   Design columns are normalized to unit in-stamp flux, so every fitted
   amplitude reads directly in microjanskys.
2. **Joint fit** -- all component amplitudes solve together against a
   plane through sigma-clipped bin means (bin-level outlier rejection;
   the plane owns cutout-scale background), alternating until the
   background converges. A catalog row that declares its own misfit
   (bright and high reduced chi-square) additionally gets a shape solve,
   solved by variable projection with every amplitude re-fit at every
   trial. The target's own shape is always refit from the pixels -- the
   catalog informs the photometry through the neighbors.
3. **Measure** -- fitted neighbors and background are subtracted,
   residual neighbor light is masked (model-isophote, star-profile and
   ambient-flood channels), masked pixels are reconstructed from their
   point reflection through the target center (clamped by the model so
   holes are impossible), and the reported flux is the curve of growth
   at the aperture. The target model itself is never integrated into the
   measurement.

Bands are measured per instrument, reference band first: the reference
solves the shapes, and sibling bands re-solve neighbor shapes warm with
fluxes bounded to color-scaled reference values. The PSF is measured per
band from the field's own confirmed stars, with a Moffat fallback when no
star qualifies. A position with no Tractor coverage measures blind --
background and curve of growth only -- and says so in its flags. Errors use
the archive's inverse variance where it exists (Legacy bricks) and sky rms
otherwise.

Off-footprint and blank pixels demote a band to `no_coverage` past 5% of the
aperture area, or at any fraction when they clip the seeing-scale core, where
no fill can reconstruct the peak. Every measured row carries machine-parsable
QA tokens in `flags`: `cov` (aperture coverage), `maskfrac` (masked fraction
of the aperture), `twinfrac` (mirror-filled fraction of the masked area),
`nbsub` (neighbor flux subtracted inside the aperture), `excess` (curve
growth past the aperture the target model cannot account for), `pedb`
(residual uniform-background term), `conv` (radius where the curve plateaus
and holds; -1 when it never does), `bg` (fitted background level), plus
`refit`/`atbound`/`reg`/`scene=none` where they apply. The full per-band
witness set rides the `_measured.csv` provenance sidecar.

Two optional inputs extend the scene without touching the code:

- `<out-dir>/patches.json` -- per-galaxy custom knowledge: replace a
  blended catalog row with a known decomposition (`replace_rows`), grant a
  companion its own free shape (`free_seats`, optional `snap`), snap solved
  centers to image peaks (`snap_gated`), or disable the standard target
  refit (`target_refit: false`). No patch file means pure catalog behavior.
- `--registry FILE` -- a cross-field registry of solved shared sources. A
  bright galaxy appearing in several targets' stamps is solved once and
  reused everywhere as a frozen, tightly-bounded component.
  `--registry-update` writes the current galaxy's solved shapes back.
  Updates replace the whole file (last writer wins), so run
  `--registry-update` sweeps one galaxy at a time.

`--mode sersic` reports the fitted target model's flux rather than the
aperture integral -- forced photometry through the same scene fit. The shape
is the standard reference-band refit, a fit on a chosen band
(`--sersic-from`), or explicit `--sersic-params`. Fitted `n` and `r_eff` are
PSF-sensitive, so explicit parameters from a trusted fit are the precision
path. The shape flags apply to `--mode sersic` only, and are refused rather
than ignored under `--mode aperture`.

`sedphot run` accepts every option `measure` does, so the full run never
measures under different settings than the verb it drives.

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

## Documentation

`docs/technical_manual.md` covers the module layout, the measurement flow,
the project vocabulary, the output contract and the extension points.

## Requirements

Python 3.11+; numpy, scipy, pandas, astropy, astroquery, matplotlib, pillow,
reproject, requests, defusedxml (see `pyproject.toml`).

Tests: `python -m pytest tests/` -- the suite is entirely offline.

## License

MIT
