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
sedphot measure --ra 216.988087 --dec 56.9878 --instruments cfht legacy \
    --aperture 25.5 --sky-in 30 --sky-out 43 --legacy-dr dr9

# the flagship: everything, then a combined SED plot
sedphot run --name "SDSS J142800.81+570046.3" --out-dir Galaxies/control_0
```

## Verbs

| Verb | What it does |
|---|---|
| `resolve`  | Name -> ICRS position (Sesame -> NED -> SIMBAD) + output label |
| `catalogs` | Closest-source photometry from the catalog archives -> `<label>_catalog.csv` |
| `measure`  | Fetch images, measure every band -> `<label>_measured.csv` + QA figures |
| `spherex`  | Raw SPHEREx spectrophotometry table (IRSA), PSF or forced-Sersic model |
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
    SPHEREx/table_photometry.csv            raw per-visit x channel table, verbatim
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

One recipe for every instrument: stamp at the target -> neighbor mask ->
sigma-clipped annulus sky (with matched-filter rejection of bright annulus
sources) -> azimuthal-profile fill of masked pixels -> curve of growth ->
aperture flux. Errors use the archive's inverse variance when it exists
(Legacy bricks, HST weights), sky rms otherwise.

The auto-mask subtracts the target's own elliptical-median profile and
detects neighbors on the residual, so it does not eat the galaxy's envelope;
for pathological targets (bright asymmetric cD envelopes) pass a custom
`--mask` (`.npz` staged masks pair with `--mask-ref` for their WCS). A
warning fires when more than 20% of the aperture is masked.

`--mode sersic` forces one sky-frame Sersic shape (from `--sersic-params`,
or fit on a band with `--sersic-from`) across all bands and solves only the
amplitude -- profile-matched photometry, the convention behind the SPHEREx
work. Fitted n and r_eff are PSF-sensitive: supply `--sersic-seeing`, or
use `--sersic-params` from a trusted fit.

## Legacy scripts

`phot_coord_search.py`, `hst_aperture_photometry.py`, and
`plot_hst_image.py` remain standalone and untouched; `sedphot catalogs`
reproduces `phot_coord_search.py` row-for-row (validated 2026-07-05), and
the HST curve-of-growth workflow lives on in `hst_aperture_photometry.py`
until its specialized outputs are folded in.

## Requirements

Python 3.11+; numpy, scipy, pandas, astropy, astroquery, photutils,
matplotlib, requests, defusedxml (see `pyproject.toml`).

## License

MIT
