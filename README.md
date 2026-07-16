# sedphot

Galaxy in, SED photometry out. Give it a name or a position and it retrieves
catalog photometry from the common archives, fetches images and measures
uniform aperture (or forced single-Sersic) fluxes, optionally pulls SPHEREx
spectrophotometry, and writes SED-ready tables with QA figures and provenance
sidecars.

```bash
pip install -e .

# resolve a name
sedphot resolve --name "NGC 4889"

# catalog photometry from every archive, with graceful per-provider fallback
sedphot catalogs --name M87 --all --out-dir Clusters/Virgo/Galaxies/M87

# fetch images and measure every band with one identical scene recipe
sedphot measure --ra 194.898792 --dec 27.959528 --instruments cfht legacy \
    --aperture 12 --legacy-dr dr9

# SPHEREx forced photometry; the Sersic shape comes from the Tractor catalog
sedphot spherex --name "NGC 4874" --out-dir Clusters/Coma/Galaxies/NGC_4874

# the flagship: everything, then a combined SED plot
sedphot run --name "NGC 4889" --out-dir Clusters/Coma/Galaxies/NGC_4889
```

## Verbs

| Verb | What it does |
|---|---|
| `resolve`  | Name -> ICRS position (Sesame -> NED -> SIMBAD) + output label |
| `catalogs` | Closest-source photometry from the catalog archives -> `<label>_catalog.csv` |
| `measure`  | Fetch images, measure every band -> `<label>_measured.csv` + QA figures |
| `spherex`  | Raw SPHEREx spectrophotometry table (IRSA), forced-Sersic (default) or PSF model |
| `sed`      | Combined flux-vs-wavelength figure from the tables in `out-dir` |
| `run`      | catalogs -> measure -> SPHEREx (opt-in) -> SED plot |

## Providers

Catalogs: `legacy` (Tractor via Datalab TAP; optical + unWISE-forced WISE,
MW transmission carried per band, dr9/dr10), `panstarrs` (VizieR),
`sdss` (DR17 cModel + native extinction), `galex` (GUVcat_AIS via VizieR),
`jplus` (DR3 PSFCOR via the CEFCA TAP), `allwise` (IRSA, Vega->AB),
`hst` (HAP point/segment catalogs via MAST).

Images (for `measure`): `legacy` (viewer cutouts, or NERSC bricks with real
inverse variance via `--legacy-bricks`), `panstarrs` (fitscut stacks),
`sdss` (frames), `cfht` (MegaPipe stacks via CADC SODA), `hst` (HAP drizzled
mosaics, any instrument, DRC/DRZ).

Every provider reports `ok / no_coverage / no_match / error` into
`coverage_*.json` and the run continues -- one dead service never kills a
fetch-all.

## Output conventions

```
<out-dir>/Photometry/
    <label>_catalog.csv  (+ .provenance.json)
    <label>_measured.csv (+ .provenance.json)
    <label>_sed.png
    coverage_catalogs.json / coverage_measure.json
    Legacy/ PanSTARRS/ SDSS/ CFHT/ HST/     cached images + QA/ figures
    SPHEREx/table_photometry.<tag>.csv      raw per-visit x channel table,
                                            verbatim; one per extraction
                                            config (tag = <model>-<hash6>)
    SPHEREx/extractions.json                the tag decoder ring
```

Tables share one schema (the v1 `OUT_COLS` plus `retrieved`,
`mw_transmission`, `dered_applied`): `band, flux_uJy, flux_err_uJy, mag_AB,
mag_err, target_ra, target_dec, match_ra, match_dec, sep_arcsec, flags,
source, ...`. Fluxes are microjansky, AB throughout; errors are statistical
only (error floors belong to the SED fitter); negative catalog fluxes are
legitimate non-detections and are preserved; fluxes are as-measured unless
`--dered` is passed (per-row corrections recorded). Band labels are
`<Instrument>_<filter>`; measurement provenance lives in `source` (unWISE
vs AllWISE both label their bands `WISE_Wn` and differ in `source`).

## Measurement recipe

One recipe for every instrument, built around a scene fit instead of a
sky annulus:

1. **Scene** -- every Tractor catalog row near the target becomes a
   rendered component at its catalog shape (Gaia-confirmed stars are
   replaced by their own measured radial profiles and pre-subtracted).
   Design columns are normalized to unit in-stamp flux, so every fitted
   amplitude reads directly in microjanskys.
2. **Joint fit** -- all component amplitudes solve together against a
   plane through sigma-clipped bin medians (bin-level outlier rejection;
   the plane owns cutout-scale background only), alternating until the
   background converges. A catalog row that declares its own misfit
   (bright and high reduced chi-square) additionally gets a shape solve:
   a Sersic core plus a truncated Nuker halo, solved by variable
   projection with every amplitude re-fit exactly at every trial. The
   target's own shape is always refit from the pixels -- the catalog
   informs the photometry only through the neighbors.
3. **Measure** -- fitted neighbors and background are subtracted,
   residual neighbor light is masked (model-isophote, star-profile, and
   ambient-flood channels), masked pixels are reconstructed from their
   point reflection through the target center (clamped by the model so
   holes are impossible), and the reported flux is the curve of growth
   at the aperture. The target model itself is never integrated into
   the measurement.

Bands are measured per instrument, reference band first: the reference
solves the shapes, sibling bands re-solve neighbor shapes warm with
fluxes leashed to color-scaled reference values. The PSF is measured per
band from the field's own confirmed stars (Moffat fallback when none
qualifies). A position with no Tractor coverage measures blind -- no
components, background and curve of growth only -- and says so in its
flags. Errors use the archive's inverse variance when it exists (Legacy
bricks), sky rms otherwise.

Off-footprint and blank pixels demote the band to `no_coverage` past 5%
of the aperture area -- or at any fraction when they clip the
seeing-scale core, where no fill can reconstruct the peak. Every
measured row carries machine-parsable QA tokens in `flags`: `cov`
(aperture coverage), `maskfrac` (masked fraction of the aperture),
`twinfrac` (mirror-filled fraction of the masked area), `nbsub`
(neighbor flux subtracted inside the aperture), `excess` (curve growth
past the aperture the target model cannot account for), `pedb` (residual
uniform-background term of the curve), `conv` (radius where the curve
plateaus and holds; -1 when it never does), `bg` (fitted background
level), plus `refit`/`atbound`/`reg`/`scene=none` where they apply. The
full per-band witness set (solve diagnostics, star log, background
track, the exact recipe constants) rides the `_measured.csv` provenance
sidecar.

Two optional inputs extend the scene without touching the code:

- `<out-dir>/patches.json` -- per-galaxy custom knowledge: replace a
  blended catalog row with a known decomposition (`replace_rows`), grant
  a companion a free shape seat (`free_seats`, optional `snap`), snap
  gated centers to image peaks (`snap_gated`), or disable the standard
  target refit (`target_refit: false`). No patch file means pure catalog
  behavior.
- `--registry FILE` -- a cross-field registry of solved shared sources
  (a bright galaxy appearing in several targets' stamps is solved once
  and consumed everywhere as frozen, tightly-leashed components).
  `--registry-update` writes the current galaxy's solved shapes back.

`--mode sersic` reports the fitted target model's flux instead of the
aperture integral -- forced photometry through the same scene fit. The
shape is the standard reference-band refit, a fit on a chosen band
(`--sersic-from`), or explicit `--sersic-params`; fitted n and r_eff are
PSF-sensitive, so explicit parameters from a trusted fit are the
precision path.

## SPHEREx

`sedphot spherex` submits an IRSA forced-photometry job and writes the raw
per-visit x channel table verbatim (quality cuts belong downstream). The
source model defaults to a forced Sersic -- a PSF model carries a chromatic
bias for extended sources -- with the shape resolved in order:

1. `--sersic-params N AXRATIO PA REFF` -- explicit, used as given
2. `--sersic-from <band>` -- fit on that band's image
3. default: the Legacy Tractor catalog shape (`type`, `sersic`, `shape_r`,
   `e1`/`e2` -> n, b/a, PA east of north; SER keeps its fitted index,
   DEV/EXP/REX fix n = 4/1/1). The TAP lookup is retried, and when it
   still fails -- or the source has no extended shape (PSF/DUP) -- the
   verb ABORTS nonzero instead of silently substituting an image fit: a
   wrong shape poisons an irreplaceable raw table. Proceed deliberately
   with `--sersic-params`, `--sersic-from`, or `--model psf`.

The shape's origin is recorded in the sidecar's model block. Every
distinct extraction configuration (model + shape + background region +
MJD window) owns its own `table_photometry.<model>-<hash6>.csv`, indexed
in `extractions.json`, so PSF and Sersic runs -- or different shapes --
coexist without manual renames. Re-requesting a configuration already on
disk reuses it (pre-tag bare tables are matched through their sidecars
and reused in place, never renamed); nothing is ever overwritten -- move
a table aside deliberately to force a re-fetch. `--mjd-range` restricts
the job to a known-good visit window (the IRSA workaround for
broken-metadata epochs). The verb exits nonzero when the fetch fails, so
shell chains can trust `$?`.

## Legacy scripts

`phot_coord_search.py`, `hst_aperture_photometry.py`, and
`plot_hst_image.py` remain standalone and untouched; `sedphot catalogs`
reproduces `phot_coord_search.py` row-for-row, and the HST curve-of-growth
workflow lives on in `hst_aperture_photometry.py` until its specialized
outputs are folded in.

## Requirements

Python 3.11+; numpy, scipy, pandas, astropy, astroquery, photutils,
matplotlib, requests, defusedxml (see `pyproject.toml`).

## License

MIT
