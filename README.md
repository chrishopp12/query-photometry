# photometry

Source photometry tools for archival survey and HST data: multi-archive
retrieval for SED fitting, HST aperture curve-of-growth measurement, and
overlay verification plots. Each script is a standalone command-line tool.

## Scripts

### `phot_coord_search.py`

Retrieve archival photometry for a single sky position from Legacy Survey DR10
(grz + WISE W1/W2), Pan-STARRS DR1 (grizy), and HST HAP (per-band), converted
to a common flux unit (uJy) and AB magnitude. Writes one CSV row per band,
intended as input for SED fitting (e.g. Prospector).

```bash
python phot_coord_search.py --ra 150.0 --dec 2.2 --radius 2.0 --out target.csv
python phot_coord_search.py --ra 150.0 --dec 2.2 --no-panstarrs
```

### `hst_aperture_photometry.py`

Curve-of-growth aperture photometry on HST ACS/WFC drizzled (DRC) images, with
comparison to the HAP point and segment catalog values. Queries MAST, downloads
the DRC science and weight images, measures circular-aperture flux at a range of
radii with sigma-clipped annulus background subtraction, and writes a
curve-of-growth plot and a summary table.

```bash
python hst_aperture_photometry.py 150.0 2.2 --proposal-id 12345
```

### `plot_hst_image.py`

Overlay matched photometry positions (from `phot_coord_search.py` CSVs) on the
HAP color composite image, with a wide context panel and a zoomed detail panel
per target, for visual verification of the matches.

```bash
python plot_hst_image.py photometry.csv [more_photometry.csv ...]
```

## Requirements

Python 3.9+, with: numpy, scipy, pandas, matplotlib, astropy, astroquery,
photutils, pillow.

## Outputs

All data products (CSVs, FITS, catalogs, plots, cache directories) are written
to the working directory and are gitignored; only the code is tracked.

## License

MIT
