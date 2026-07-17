"""
pipeline.py

Retrieval and Measurement Drivers
---------------------------------------------------------

Orchestration only: resolve the target once, run the requested providers,
assemble the schema table, and write products + provenance. No science lives
here -- providers and the measurement engine own their own behavior.

Data products (under <out_dir>/Photometry/):
    <label>_catalog.csv               combined catalog photometry
    <label>_measured.csv              image measurements (scene-fit aperture
                                      or forced-model flux)
    <label>_sed.png                   combined SED figure
    <label>_*.provenance.json         provenance sidecars
    coverage_catalogs.json            per-provider status, catalog run
    coverage_measure.json             per-provider status, measurement run
    <Instrument>/                     cached images + QA/ per-band figures
    QA/growth_curves.png              all measured bands, one overlay
    scene/                            cached scene inputs (Tractor, Gaia)
    SPHEREx/table_photometry.<tag>.csv  raw spectrophotometry, one table
                                      per extraction config (run_spherex)

Requirements:
    numpy, pandas, astropy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .catalogs import CATALOG_PROVIDERS
from .catalogs.legacy import LEGACY_DR_DEFAULT
from .dered import apply_dereddening
from .images import IMAGE_PROVIDERS
from .measure import recipe
from .measure.aperture import measurement_to_row
from .measure.engine import measure_band, order_bands, prepare_scene
from .measure.seats import save_registry
from .measure.stamp import ApertureCoverageError
from .provenance import write_sidecar
from .qa import plot_growth_curves, qa_scene_figure
from .results import (
    STATUS_ERROR,
    STATUS_NO_COVERAGE,
    STATUS_OK,
    ImageProduct,
    ProviderResult,
    print_coverage_summary,
    write_coverage_report,
)
from .schema import rows_to_frame

# Cache directory per image provider, under <out_dir>/Photometry/.
INSTRUMENT_DIRS = {'legacy': 'Legacy', 'panstarrs': 'PanSTARRS',
                   'sdss': 'SDSS', 'cfht': 'CFHT', 'hst': 'HST'}


# ------------------------------------
# Catalog driver
# ------------------------------------
def run_catalogs(
        coord: SkyCoord,
        label: str,
        out_dir: str | Path,
        *,
        instruments: list[str],
        radius_arcsec: float = 2.0,
        legacy_dr: str = LEGACY_DR_DEFAULT,
        dered: bool = False,
        target_name: str | None = None,
) -> pd.DataFrame:
    """Query the requested catalog providers and write the combined table.

    Parameters
    ----------
    coord : SkyCoord
        Resolved target position.
    label : str
        Output stem (sanitized name or J-name).
    out_dir : str or Path
        Galaxy directory; products land in <out_dir>/Photometry/.
    instruments : list[str]
        Provider names from catalogs.CATALOG_PROVIDERS.
    radius_arcsec : float
        Starting search radius per provider. [default: 2.0]
    legacy_dr : str
        Legacy data release ('dr10' or 'dr9'). [default: LEGACY_DR_DEFAULT]
    dered : bool
        Apply MW dereddening (see dered.py tiers). [default: False]
    target_name : str, optional
        Original name string, recorded in the sidecar.

    Returns
    -------
    catalog_df : pd.DataFrame
        The combined photometry table (also written to CSV when non-empty).
    """
    unknown = [inst for inst in instruments if inst not in CATALOG_PROVIDERS]
    if unknown:
        raise ValueError(f"unknown catalog provider(s) {unknown}; "
                         f"known: {sorted(CATALOG_PROVIDERS)}")

    phot_dir = Path(out_dir) / "Photometry"
    phot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTarget: RA={coord.ra.deg:.6f}, Dec={coord.dec.deg:+.6f}  "
          f"(search radius={radius_arcsec:.1f}\")\n")

    results: list[ProviderResult] = []
    for name in instruments:
        print(f"=== {name} ===")
        provider = CATALOG_PROVIDERS[name]
        try:
            if name == 'legacy':
                result = provider(coord, radius_arcsec, dr=legacy_dr)
            else:
                result = provider(coord, radius_arcsec)
        except Exception as e:
            # Providers handle their own expected failures; this catches the
            # unexpected so one broken service never kills the run.
            result = ProviderResult(provider=name, status=STATUS_ERROR,
                                    message=f"{type(e).__name__}: {e}")
        results.append(result)
        print()

    catalog_df = rows_to_frame([row for r in results for row in r.rows])

    print("Provider summary:")
    print_coverage_summary(results)
    write_coverage_report(results, phot_dir / "coverage_catalogs.json")

    if catalog_df.empty:
        print("\nNo photometry retrieved from any catalog.")
        return catalog_df

    dered_meta = None
    if dered:
        print("\nApplying MW dereddening:")
        catalog_df, dered_meta = apply_dereddening(catalog_df, coord)

    out_csv = phot_dir / f"{label}_catalog.csv"
    catalog_df.to_csv(out_csv, index=False)
    write_sidecar(out_csv, {
        "kind": "catalog_photometry",
        "target": {"name": target_name, "label": label,
                   "ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg)},
        "radius_arcsec": radius_arcsec,
        "instruments": instruments,
        "legacy_dr": legacy_dr if 'legacy' in instruments else None,
        "dereddening": dered_meta,
        "providers": {r.provider: {"status": r.status, "message": r.message, **r.meta}
                      for r in results},
    })
    print(f"\nSaved {len(catalog_df)} photometric points to: {out_csv}")
    print(catalog_df.to_string(index=False))

    return catalog_df


# ------------------------------------
# Measurement driver
# ------------------------------------
def _resolve_shape(
        products,
        coord,
        *,
        sersic_from,
        sersic_params,
        cutout_half_arcsec,
        sersic_seeing=None,
):
    """Resolve an explicit sky-frame Sersic shape.

    Explicit --sersic-params wins; otherwise the shape is fit on the
    requested band ('z' or 'Legacy_z'), defaulting to the reddest
    available optical band. The fit runs on a background-subtracted
    stamp with no neighbor handling -- explicit parameters, or the scene
    fit's own target refit, are the trusted paths.
    """
    from .measure.background import bin_plane
    from .measure.sersic import SERSIC_N_MAX, fit_sersic_shape, pa_east_of_north
    from .measure.stamp import load_stamp

    if sersic_params is not None:
        n, axis_ratio, pa_deg, reff_arcsec = [float(v) for v in sersic_params]
        if axis_ratio < 1.0:
            raise ValueError("--sersic-params axis_ratio is a/b >= 1")
        shape_sky = dict(n=n, reff_arcsec=reff_arcsec,
                         ellip=1.0 - 1.0 / axis_ratio, pa_deg=pa_deg % 180.0)
        return shape_sky, {'source': 'explicit parameters'}

    if not products:
        raise ValueError("shape fit needs at least one fetched image "
                         "(or --sersic-params)")

    if sersic_from is not None:
        wanted = sersic_from.lower()
        matches = [p for p in products
                   if p.band.lower() == wanted
                   or f"{p.instrument}_{p.band}".lower() == wanted]
        if not matches:
            available = [f"{p.instrument}_{p.band}" for p in products]
            raise ValueError(f"--sersic-from {sersic_from!r} matches none of "
                             f"the fetched images: {available}")
        shape_product = matches[0]
    else:
        preference = ['z', 'y', 'i', 'r', 'g', 'u']
        ranked = sorted(products,
                        key=lambda p: preference.index(p.band.lower())
                        if p.band.lower() in preference else 99)
        shape_product = ranked[0]

    seeing = sersic_seeing if sersic_seeing is not None else shape_product.seeing_arcsec
    if sersic_seeing is None:
        print(f"  WARNING shape fit assumes PSF FWHM = {seeing:.2f}\" (a typical "
              f"value, not measured) -- n and r_eff are PSF-sensitive; supply "
              f"--sersic-seeing, or --sersic-params from a trusted fit")
    stamp = load_stamp(shape_product.path, shape_product.calib, coord,
                       cutout_half_arcsec=cutout_half_arcsec)
    raw = np.where(np.isfinite(stamp.data), stamp.data, 0.0)
    background = bin_plane(raw, stamp.good, stamp.rr, stamp.pixscale)
    sub = np.where(stamp.good, raw - background['img'], np.nan)
    fit = fit_sersic_shape(sub, stamp.sigma, stamp.cx, stamp.cy,
                           stamp.pixscale, seeing)
    if not fit['success']:
        print("  WARNING shape fit did not converge cleanly; inspect the QA")
    if fit['n'] >= SERSIC_N_MAX - 0.05:
        print(f"  WARNING fitted n={fit['n']:.2f} sits at the fit bound")
    pa_deg = pa_east_of_north(stamp.wcs, fit['xc'], fit['yc'], fit['theta'])
    shape_sky = dict(n=fit['n'], reff_arcsec=fit['reff_arcsec'],
                     ellip=fit['ellip'], pa_deg=pa_deg)
    origin = {'source': f"fit on {shape_product.instrument}_{shape_product.band} "
                        f"(redchi2 {fit['redchi2']:.2f})",
              'band': f"{shape_product.instrument}_{shape_product.band}",
              'assumed_seeing_arcsec': seeing,
              'redchi2': round(fit['redchi2'], 3)}
    return shape_sky, origin


def run_measure(
        coord: SkyCoord,
        label: str,
        out_dir: str | Path,
        *,
        instruments: list[str],
        mode: str = 'aperture',
        bands: list[str] | None = None,
        aperture_arcsec: float = 10.0,
        cutout_arcsec: float = 120.0,
        rgrid: list[float] | None = None,
        sersic_from: str | None = None,
        sersic_params: list[float] | None = None,
        sersic_seeing: float | None = None,
        registry_path: str | None = None,
        registry_update: bool = False,
        dump_arrays: bool = False,
        legacy_dr: str = LEGACY_DR_DEFAULT,
        legacy_bricks: bool = False,
        hst_proposal_id: str | None = None,
        target_name: str | None = None,
) -> pd.DataFrame:
    """Fetch images from the requested providers and measure every band.

    Every band runs through the scene engine: survey-catalog components,
    measured-star subtraction, a joint amplitude(+shape) solve against a
    bin-median-plane background, then mask, twin fill, and the curve of
    growth. Bands are measured per instrument, reference band first.

    Parameters
    ----------
    coord : SkyCoord
        Target position (the forced aperture center).
    label : str
        Output stem.
    out_dir : str or Path
        Galaxy directory; images cache under <out_dir>/Photometry/<Inst>/,
        and an optional patches file is read from <out_dir>/patches.json.
    instruments : list[str]
        Provider names from images.IMAGE_PROVIDERS.
    mode : str
        'aperture' (curve-of-growth aperture flux) or 'sersic' (the
        fitted target model's flux -- forced photometry through the same
        scene fit). [default: 'aperture']
    bands : list[str], optional
        Band subset applied to every provider. [default: provider defaults]
    aperture_arcsec : float
        Aperture radius. [default: 10.0]
    cutout_arcsec : float
        Stamp width. [default: 120]
    rgrid : list[float], optional
        Curve-of-growth radii override.
    sersic_from : str, optional
        Sersic mode: pin the target shape to a fit on this band
        ('z' or 'Legacy_z') instead of the standard per-instrument refit.
    sersic_params : list of float, optional
        Sersic mode: explicit shape [n, axis_ratio(a/b), pa_deg(E of N),
        reff_arcsec] -- pins the target profile in every band.
    sersic_seeing : float, optional
        PSF FWHM (arcsec) assumed by the --sersic-from shape fit.
    registry_path : str, optional
        Cross-field registry to consume (solved shared sources become
        frozen components).
    registry_update : bool
        Also write this galaxy's solved seats back to the registry.
    dump_arrays : bool
        Write per-band array bundles under <Inst>/QA/ (debug).
    legacy_dr : str
        Legacy release for the image provider and scene catalog.
        [default: LEGACY_DR_DEFAULT]
    legacy_bricks : bool
        Fetch NERSC bricks (image + invvar) instead of viewer cutouts.
    hst_proposal_id : str, optional
        Restrict the HST image provider to one program.
    target_name : str, optional
        Original name string for the sidecar.

    Returns
    -------
    measured_df : pd.DataFrame
        One row per measured band (also written to CSV when non-empty).
    """
    unknown = [inst for inst in instruments if inst not in IMAGE_PROVIDERS]
    if unknown:
        raise ValueError(f"unknown image provider(s) {unknown}; "
                         f"known: {sorted(IMAGE_PROVIDERS)}")
    if registry_update and registry_path is None:
        raise ValueError("--registry-update needs --registry PATH")

    phot_dir = Path(out_dir) / "Photometry"
    phot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTarget: RA={coord.ra.deg:.6f}, Dec={coord.dec.deg:+.6f}  "
          f"(aperture {aperture_arcsec:g}\", scene engine)\n")

    results: list[ProviderResult] = []
    measurements: list[dict] = []
    rows: list[dict] = []
    rgrid_arr = (np.unique(np.asarray(rgrid, dtype=float)) if rgrid
                 else recipe.DEFAULT_RGRID)
    if not rgrid_arr.min() <= aperture_arcsec <= rgrid_arr.max():
        raise ValueError(
            f"--aperture {aperture_arcsec:g}\" lies outside the "
            f"curve-of-growth grid [{rgrid_arr.min():g}, "
            f"{rgrid_arr.max():g}]\"; pass --radii covering it")

    # Phase 1 -- fetch every provider's images.
    fetched_products: list[tuple[str, list[ImageProduct]]] = []
    for name in instruments:
        print(f"=== {name} images ===")
        cache_dir = phot_dir / INSTRUMENT_DIRS.get(name, name)
        fetch = IMAGE_PROVIDERS[name]
        options: dict = {'bands': bands, 'size_arcsec': cutout_arcsec,
                         'cache_dir': cache_dir}
        if name == 'legacy':
            options.update(dr=legacy_dr, use_bricks=legacy_bricks)
        if name == 'hst' and hst_proposal_id:
            options.update(proposal_id=hst_proposal_id)
        try:
            fetched = fetch(coord, **options)
        except Exception as e:
            fetched = ProviderResult(provider=name, status=STATUS_ERROR,
                                     message=f"{type(e).__name__}: {e}")
        if isinstance(fetched, ProviderResult):
            results.append(fetched)
            print(f"  {fetched.status}: {fetched.message}")
        else:
            fetched_products.append((name, fetched))
            print(f"  {len(fetched)} band image(s) ready")
    print()

    # Phase 2 -- sersic mode with an explicit shape request: resolve the
    # one sky shape every band pins the target to. Without a request the
    # target shape is the standard per-instrument reference-band refit.
    shape_sky = None
    shape_origin = None
    if mode == 'sersic' and (sersic_params is not None
                             or sersic_from is not None):
        all_products = [p for _, products in fetched_products for p in products]
        shape_sky, shape_origin = _resolve_shape(
            all_products, coord, sersic_from=sersic_from,
            sersic_params=sersic_params,
            cutout_half_arcsec=cutout_arcsec / 2.0,
            sersic_seeing=sersic_seeing)
        print(f"Forced shape: n={shape_sky['n']:.2f}, "
              f"reff={shape_sky['reff_arcsec']:.2f}\", "
              f"ellip={shape_sky['ellip']:.2f}, PA={shape_sky['pa_deg']:.1f} deg "
              f"({shape_origin['source']})\n")
    elif mode == 'sersic':
        print("Forced shape: per-instrument reference-band refit\n")

    # Phase 3 -- scene inputs, once per galaxy: the survey catalog and
    # confirmed stars (cache-first under Photometry/scene/), the optional
    # patches file, and the cross-field registry.
    scene = prepare_scene(coord, phot_dir=phot_dir, out_dir=out_dir,
                          aperture_arcsec=aperture_arcsec,
                          legacy_dr=legacy_dr, registry_path=registry_path)
    print()

    # Phase 4 -- measure every band, reference band first per instrument.
    caches: dict = {}
    references: dict[str, dict] = {}
    for name, products in fetched_products:
        cache_dir = phot_dir / INSTRUMENT_DIRS.get(name, name)
        provider_rows: list[dict] = []
        measured_bands: list[str] = []
        demoted_bands: list[str] = []
        for product in order_bands(products):
            try:
                measurement, new_ref = measure_band(
                    product, coord, scene,
                    references.get(product.instrument), caches,
                    aperture_arcsec=aperture_arcsec,
                    cutout_half_arcsec=cutout_arcsec / 2.0,
                    rgrid=rgrid_arr,
                    target_shape=shape_sky if mode == 'sersic' else None,
                    registry_update=registry_update,
                    dump_dir=cache_dir / "QA" if dump_arrays else None)
                if new_ref is not None:
                    references[product.instrument] = new_ref
                row = measurement_to_row(measurement, mode=mode)
                figure = qa_scene_figure(measurement, cache_dir / "QA")
            except ApertureCoverageError as e:
                # The image exists but the aperture is off its footprint:
                # honest no_coverage, not a measurement of zero.
                print(f"  {product.instrument} {product.band}: "
                      f"no_coverage -- {e}")
                demoted_bands.append(f"{product.band} "
                                     f"(coverage {e.coverage:.2f})")
                continue
            except Exception as e:
                print(f"  {product.instrument} {product.band} FAILED: "
                      f"{type(e).__name__}: {e}")
                continue
            measurements.append(measurement)
            provider_rows.append(row)
            measured_bands.append(product.band)
            print(f"  {product.instrument} {product.band}: "
                  f"{row['flux_uJy']:.1f} +/- "
                  f"{measurement['flux_err_ujy']:.1f} uJy "
                  f"({measurement['err_model']}; QA {figure.name})")
        rows.extend(provider_rows)
        pieces = []
        if measured_bands:
            pieces.append(f"measured bands: {', '.join(measured_bands)}")
        if demoted_bands:
            pieces.append(f"aperture off footprint: {', '.join(demoted_bands)}")
        if measured_bands:
            status = STATUS_OK
        elif demoted_bands:
            status = STATUS_NO_COVERAGE
        else:
            status = STATUS_ERROR
            pieces.append("fetched images but every measurement failed")
        results.append(ProviderResult(
            provider=name, status=status, rows=provider_rows,
            message="; ".join(pieces)))
        print()

    if registry_update:
        save_registry(scene['registry'], registry_path)
        print(f"registry updated: {registry_path}\n")

    measured_df = rows_to_frame(rows)

    print("Provider summary:")
    print_coverage_summary(results)
    write_coverage_report(results, phot_dir / "coverage_measure.json")

    if measured_df.empty:
        print("\nNo bands measured.")
        return measured_df

    plot_growth_curves(measurements, phot_dir / "QA")
    out_csv = phot_dir / f"{label}_measured.csv"
    measured_df.to_csv(out_csv, index=False)
    write_sidecar(out_csv, {
        "kind": f"{mode}_photometry",
        "target": {"name": target_name, "label": label,
                   "ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg)},
        "instruments": instruments,
        "mode": mode,
        "aperture_arcsec": aperture_arcsec,
        "cutout_arcsec": cutout_arcsec,
        "sersic_shape": ({**shape_sky, **shape_origin} if shape_sky
                         else ({'source': 'reference-band refit'}
                               if mode == 'sersic' else None)),
        "scene": {
            "n_catalog_rows": int(len(scene['cat'])),
            "n_confirmed_stars": int(len(scene['stars'])),
            "patches": sorted(scene['patches'].keys()),
            "registry_path": str(registry_path) if registry_path else None,
            "registry_updated": bool(registry_update),
            "recipe": recipe.snapshot(),
        },
        "legacy": {"dr": legacy_dr, "bricks": legacy_bricks}
                  if 'legacy' in instruments else None,
        "per_band": {f"{m['instrument']}_{m['band']}": m['witness']
                     for m in measurements},
    })
    print(f"\nSaved {len(measured_df)} measured bands to: {out_csv}")
    print(measured_df.to_string(index=False))

    return measured_df


# ------------------------------------
# SPHEREx driver
# ------------------------------------
def run_spherex(
        coord: SkyCoord,
        label: str,
        out_dir: str | Path,
        *,
        model: str = 'sersic',
        sersic_params: list[float] | None = None,
        sersic_from: str | None = None,
        sersic_seeing: float | None = None,
        bkg_size: float = 15.0,
        mjd_range: list[float] | None = None,
        poll: float = 5.0,
        timeout: float = 3600.0,
        cutout_arcsec: float = 120.0,
        legacy_dr: str = LEGACY_DR_DEFAULT,
        target_name: str | None = None,
):
    """Fetch the raw SPHEREx spectrophotometry table for the target.

    Parameters
    ----------
    model : str
        'sersic' (elliptical forced model) or 'psf' (point source; carries
        a chromatic bias for extended sources).
    sersic_params : list of float, optional
        [n, axis_ratio(a/b), pa_deg, reff_arcsec] -- explicit shape.
    sersic_from : str, optional
        Fit the shape on this band's image instead of the default Tractor
        catalog lookup ('Legacy_z' fetches the Legacy z image; plain 'z'
        assumes Legacy). [default when model='sersic' and no params: the
        ls_dr9/dr10.tractor shape, falling back to a Legacy z image fit
        when the lookup yields nothing usable]
    sersic_seeing : float, optional
        PSF FWHM of the shape-fit band (see run_measure).
    bkg_size : float
        Tool BKG_REGION_SIZE in pixels. [default: 15]
    mjd_range : [float, float], optional
        Restrict to visits in this MJD window (the IRSA workaround for
        broken-metadata epochs).

    Returns
    -------
    result : ProviderResult
        ok with the table path, or error with the manual-GUI recipe.
    """
    from . import spherex as spherex_mod
    from .catalogs.legacy import query_shape

    print(f"\nTarget: RA={coord.ra.deg:.6f}, Dec={coord.dec.deg:+.6f}  "
          f"(SPHEREx {model} model)\n")

    tool_model = None
    shape_origin = None
    if model == 'sersic':
        shape_sky = origin = None
        if sersic_params is not None:
            shape_sky, origin = _resolve_shape(
                [], coord, sersic_from=None, sersic_params=sersic_params,
                cutout_half_arcsec=cutout_arcsec / 2.0)
        elif sersic_from is None:
            # Default: the Tractor catalog shape. A wrong shape poisons an
            # irreplaceable raw table, so a failed lookup ABORTS rather
            # than silently substituting the PSF-degenerate image fit --
            # a service outage must never change which shape convention
            # the extraction ran under.
            try:
                looked = query_shape(coord, dr=legacy_dr)
            except Exception as e:
                message = (f"Tractor shape lookup failed: {e} -- aborting; "
                           f"re-run later, or pass an explicit shape "
                           f"(--sersic-params), fit one deliberately "
                           f"(--sersic-from z), or use --model psf")
                print(f"  [spherex] {message}")
                return ProviderResult(provider='spherex', status=STATUS_ERROR,
                                      message=message)
            if looked is None:
                message = ("no usable extended Tractor shape at this "
                           "position -- pass --sersic-params, fit a band "
                           "with --sersic-from, or use --model psf")
                print(f"  [spherex] {message}")
                return ProviderResult(provider='spherex', status=STATUS_ERROR,
                                      message=message)
            shape_sky, origin = looked
        if shape_sky is None:
            spec = sersic_from
            instrument = spec.split('_')[0].lower() if '_' in spec else 'legacy'
            if instrument not in IMAGE_PROVIDERS:
                raise ValueError(f"--sersic-from {spec!r}: unknown instrument "
                                 f"{instrument!r}; known: {sorted(IMAGE_PROVIDERS)}")
            band = spec.split('_')[-1]
            cache_dir = (Path(out_dir) / "Photometry"
                         / INSTRUMENT_DIRS.get(instrument, instrument))
            options: dict = {'bands': (band,), 'size_arcsec': cutout_arcsec,
                             'cache_dir': cache_dir}
            if instrument == 'legacy':
                cache_dir = Path(out_dir) / "Photometry" / "Legacy"
                options.update(cache_dir=cache_dir, dr=legacy_dr)
            fetched = IMAGE_PROVIDERS[instrument](coord, **options)
            if isinstance(fetched, ProviderResult):
                raise RuntimeError(f"could not fetch the shape-fit image: "
                                   f"{fetched.message}")
            shape_sky, origin = _resolve_shape(
                fetched, coord, sersic_from=band, sersic_params=None,
                cutout_half_arcsec=cutout_arcsec / 2.0,
                sersic_seeing=sersic_seeing)
        print(f"Forced shape: n={shape_sky['n']:.2f}, "
              f"reff={shape_sky['reff_arcsec']:.2f}\", "
              f"ellip={shape_sky['ellip']:.2f}, PA={shape_sky['pa_deg']:.1f} deg "
              f"({origin['source']})\n")
        tool_model = spherex_mod.sersic_from_shape(shape_sky)
        shape_origin = origin['source']

    result = spherex_mod.fetch(coord, out_dir=out_dir, model=tool_model,
                               bkg_region_size=bkg_size,
                               mjd_range=tuple(mjd_range) if mjd_range else None,
                               poll=poll, timeout=timeout,
                               shape_origin=shape_origin)
    print(f"\n  spherex {result.status}: {result.message}")
    return result


# ------------------------------------
# Combined SED plot
# ------------------------------------
def run_sed(label: str | None, out_dir: str | Path) -> Path | None:
    """Combined SED figure from whatever schema tables exist in out_dir.

    Parameters
    ----------
    label : str, optional
        Output stem; inferred when exactly one <stem>_catalog.csv or
        <stem>_measured.csv family exists.
    out_dir : str or Path
        Galaxy directory.

    Returns
    -------
    figure_path : Path or None
        The written PNG, or None when no tables were found.
    """
    from .qa import plot_sed

    phot_dir = Path(out_dir) / "Photometry"
    if label is None:
        stems = {p.name.rsplit('_', 1)[0]
                 for p in list(phot_dir.glob("*_catalog.csv"))
                 + list(phot_dir.glob("*_measured.csv"))}
        if len(stems) != 1:
            raise ValueError(f"cannot infer --label in {phot_dir}: "
                             f"found stems {sorted(stems)}")
        label = stems.pop()

    frames = {}
    for kind in ("catalog", "measured"):
        path = phot_dir / f"{label}_{kind}.csv"
        if path.exists():
            frames[kind] = pd.read_csv(path)
    if not frames:
        print(f"  [sed] no tables for {label!r} in {phot_dir}")
        return None

    out = plot_sed(frames, phot_dir / f"{label}_sed.png", title=label)
    print(f"  [sed] wrote {out}")
    return out


# ------------------------------------
# The flagship: galaxy in, SED photometry out
# ------------------------------------
def run_all(
        coord: SkyCoord,
        label: str,
        out_dir: str | Path,
        *,
        skip: list[str] | None = None,
        radius_arcsec: float = 2.0,
        dered: bool = False,
        aperture_arcsec: float = 10.0,
        cutout_arcsec: float = 120.0,
        registry_path: str | None = None,
        registry_update: bool = False,
        spherex_model: str = 'off',
        sersic_params: list[float] | None = None,
        legacy_dr: str = LEGACY_DR_DEFAULT,
        legacy_bricks: bool = False,
        target_name: str | None = None,
) -> None:
    """Everything: catalogs -> images + aperture measurement -> SPHEREx
    (opt-in) -> combined SED plot, with per-provider graceful fallback.

    Parameters
    ----------
    skip : list[str], optional
        Provider names to leave out (catalog and image registries share
        names where they overlap).
    spherex_model : str
        'off' (default), 'psf', or 'sersic' (with sersic_params).
    Other parameters as in run_catalogs / run_measure.
    """
    skip = set(skip or [])
    catalog_set = [name for name in CATALOG_PROVIDERS if name not in skip]
    image_set = [name for name in IMAGE_PROVIDERS if name not in skip]

    print("\n===== catalogs =====")
    run_catalogs(coord, label, out_dir, instruments=catalog_set,
                 radius_arcsec=radius_arcsec, legacy_dr=legacy_dr,
                 dered=dered, target_name=target_name)

    print("\n===== images + measurement =====")
    run_measure(coord, label, out_dir, instruments=image_set,
                aperture_arcsec=aperture_arcsec, cutout_arcsec=cutout_arcsec,
                registry_path=registry_path, registry_update=registry_update,
                legacy_dr=legacy_dr, legacy_bricks=legacy_bricks,
                target_name=target_name)

    if spherex_model != 'off':
        print("\n===== SPHEREx =====")
        run_spherex(coord, label, out_dir, model=spherex_model,
                    sersic_params=sersic_params, legacy_dr=legacy_dr,
                    target_name=target_name)

    print("\n===== SED =====")
    run_sed(label, out_dir)
