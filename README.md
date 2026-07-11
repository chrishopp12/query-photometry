# sedphot

Galaxy in, SED photometry out. Give it a name or a position and it retrieves
catalog photometry from the common archives, fetches images and measures
uniform aperture (or forced single-Sersic) fluxes, optionally pulls SPHEREx
spectrophotometry, and writes SED-ready tables with QA figures and provenance
sidecars.

```bash
pip install -e .

# resolve a name
sedphot resolve --name "SDSS J142800.81+570046.3"

# catalog photometry from every archive, with graceful per-provider fallback
sedphot catalogs --name M87 --all --out-dir Clusters/Virgo/Galaxies/M87

# fetch images and measure every band with one identical aperture recipe
sedphot measure --ra 194.898792 --dec 27.959528 --instruments cfht legacy \
    --aperture 25.5 --sky-in 30 --sky-out 43 --legacy-dr dr9

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

One recipe for every instrument: stamp at the target -> two-pass sky
(sigma-clipped annulus: pass one rejects bright peaks, then every detected
segment is masked and the annulus re-clipped, so the faint sources and
neighbor wings that bias a deep crowded annulus go too) -> two-channel
neighbor mask -> azimuthal-profile fill of masked and blank pixels ->
curve of growth -> aperture flux. Errors use the archive's inverse
variance when it exists (Legacy bricks, HST weights), sky rms otherwise.

Pixel ownership is structural, not radial: a detected segment that is not
the target's is masked wherever it sits -- inside the aperture included --
while the target's own segment is never masked, however asymmetric its
envelope. Sources whose isophotes merge with the target are caught by
subtracting the target's elliptical-median profile and detecting on the
residual above a local-brightness floor; `--protect-radius` guards only
that residual channel, and a merged companion inside it stays in the flux
with only the QA metrics to betray it. For pathological targets pass a
custom `--mask` (`.npz` staged masks pair with `--mask-ref` for their WCS).

Off-footprint and blank pixels are fill-corrected up to 5% of the aperture
area and demote the band to `no_coverage` beyond that -- or at any
fraction when they clip the seeing-scale core, where no fill can
reconstruct the peak. Every measured row carries machine-parsable QA
tokens in `flags`: `cov` (aperture coverage), `maskfrac` (masked fraction
of the aperture), `cogslope` (relative outer curve-of-growth slope per
arcsec; strongly negative means the sky was over-estimated). A warning
still fires when more than 20% of the aperture is masked.

`--mode sersic` forces one sky-frame Sersic shape (from `--sersic-params`,
or fit on a band with `--sersic-from`) across all bands and solves only the
amplitude -- profile-matched photometry, the convention behind the SPHEREx
work. Fitted n and r_eff are PSF-sensitive: supply `--sersic-seeing`, or
use `--sersic-params` from a trusted fit.

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
