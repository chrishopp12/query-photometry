"""
pipeline.py

Retrieval and Measurement Drivers
---------------------------------------------------------
Orchestration only: resolve the target once, run the requested providers,
assemble the schema table, and write products + provenance. No science lives
here -- providers and the measurement engine own their own behavior.

Data products (under <out_dir>/Photometry/):
    <label>_catalog.csv               combined catalog photometry
    <label>_measured.csv              image-based aperture measurements
    <label>_*.provenance.json         provenance sidecars
    coverage_catalogs.json            per-provider status, catalog run
    coverage_measure.json             per-provider status, measurement run
    <Instrument>/                     cached images + QA/ per-band figures
    QA/growth_curves.png              all measured bands, one overlay

Requirements:
    numpy, pandas, astropy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .catalogs import CATALOG_PROVIDERS
from .dered import apply_dereddening
from .images import IMAGE_PROVIDERS
from .measure.aperture import measure_aperture, measurement_to_row
from .measure.masks import load_user_mask
from .provenance import write_sidecar
from .qa import plot_growth_curves, qa_band_figure, qa_forced_figure
from .results import (
    STATUS_ERROR,
    STATUS_OK,
    ImageProduct,
    ProviderResult,
    print_coverage_summary,
    write_coverage_report,
)
from .schema import rows_to_frame


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
        legacy_dr: str = 'dr10',
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
        Legacy data release ('dr10' or 'dr9'). [default: 'dr10']
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
def _resolve_shape(products, coord, *, sersic_from, sersic_params,
                   sky_in, sky_out, cutout_half_arcsec, user_mask,
                   protect_radius, sersic_seeing=None):
    """Resolve the sky-frame Sersic shape for forced mode.

    Explicit --sersic-params wins; otherwise the shape is fit on the
    requested band ('z' or 'Legacy_z'), defaulting to the reddest available
    optical band (the A1925 convention was a frozen z-band fit).
    """
    from .measure.aperture import prepare_stamp
    from .measure.sersic import SERSIC_N_MAX, fit_sersic_shape, pa_east_of_north

    if sersic_params is not None:
        n, axis_ratio, pa_deg, reff_arcsec = [float(v) for v in sersic_params]
        if axis_ratio < 1.0:
            raise ValueError("--sersic-params axis_ratio is a/b >= 1")
        shape_sky = dict(n=n, reff_arcsec=reff_arcsec,
                         ellip=1.0 - 1.0 / axis_ratio, pa_deg=pa_deg % 180.0)
        return shape_sky, {'source': 'explicit parameters'}

    if not products:
        raise ValueError("sersic mode needs at least one fetched image "
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
    prep = prepare_stamp(shape_product, coord,
                         cutout_half_arcsec=cutout_half_arcsec,
                         sky_in=sky_in, sky_out=sky_out, user_mask=user_mask,
                         protect_radius=protect_radius)
    fit = fit_sersic_shape(prep['stamp'], prep['sky_std'], prep['cx'], prep['cy'],
                           prep['pixscale'], seeing,
                           mask=prep['mask'])
    if not fit['success']:
        print("  WARNING shape fit did not converge cleanly; inspect the QA")
    if fit['n'] >= SERSIC_N_MAX - 0.05:
        print(f"  WARNING fitted n={fit['n']:.2f} sits at the fit bound")
    pa_deg = pa_east_of_north(prep['stamp_wcs'], fit['xc'], fit['yc'], fit['theta'])
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
        sky_in: float = 30.0,
        sky_out: float = 45.0,
        cutout_arcsec: float = 120.0,
        rgrid: list[float] | None = None,
        mask_file: str | None = None,
        mask_ref: str | None = None,
        protect_radius: float = 4.0,
        sersic_from: str | None = None,
        sersic_params: list[float] | None = None,
        sersic_seeing: float | None = None,
        legacy_dr: str = 'dr9',
        legacy_bricks: bool = False,
        target_name: str | None = None,
) -> pd.DataFrame:
    """Fetch images from the requested providers and measure every band.

    Parameters
    ----------
    coord : SkyCoord
        Target position (the forced aperture center).
    label : str
        Output stem.
    out_dir : str or Path
        Galaxy directory; images cache under <out_dir>/Photometry/<Inst>/.
    instruments : list[str]
        Provider names from images.IMAGE_PROVIDERS.
    mode : str
        'aperture' (curve-of-growth aperture flux) or 'sersic' (forced
        single-Sersic amplitude with one shared sky shape). [default: 'aperture']
    bands : list[str], optional
        Band subset applied to every provider. [default: provider defaults]
    aperture_arcsec : float
        Aperture radius (aperture mode). [default: 10.0]
    sky_in, sky_out : float
        Background annulus (arcsec); must clear the source envelope.
        [default: 30-45]
    cutout_arcsec : float
        Stamp width; must contain the annulus. [default: 120]
    rgrid : list[float], optional
        Curve-of-growth radii override.
    mask_file : str, optional
        User mask (.npz neighbor_mask or FITS) instead of the auto-mask.
    mask_ref : str, optional
        Reference image whose WCS an .npz mask's grid is defined on.
    protect_radius : float
        Auto-mask protection radius around the target. [default: 4.0]
    sersic_from : str, optional
        Sersic mode: fit the shape on this band ('z' or 'Legacy_z').
        [default: reddest available optical band]
    sersic_params : list of float, optional
        Sersic mode: explicit shape [n, axis_ratio(a/b), pa_deg(E of N),
        reff_arcsec] -- skips the fit.
    legacy_dr : str
        Legacy release for the image provider. [default: 'dr9']
    legacy_bricks : bool
        Fetch NERSC bricks (image + invvar) instead of viewer cutouts.
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
    if sky_out > cutout_arcsec / 2.0 - 3.0:
        raise ValueError(f"sky_out={sky_out}\" does not fit in a "
                         f"{cutout_arcsec}\" cutout; raise --cutout-size")

    phot_dir = Path(out_dir) / "Photometry"
    phot_dir.mkdir(parents=True, exist_ok=True)
    user_mask = load_user_mask(mask_file, ref_image=mask_ref) if mask_file else None

    print(f"\nTarget: RA={coord.ra.deg:.6f}, Dec={coord.dec.deg:+.6f}  "
          f"(aperture {aperture_arcsec:g}\", sky {sky_in:g}-{sky_out:g}\")\n")

    instrument_dirs = {'legacy': 'Legacy', 'panstarrs': 'PanSTARRS',
                      'sdss': 'SDSS', 'cfht': 'CFHT'}
    results: list[ProviderResult] = []
    measurements: list[dict] = []
    rows: list[dict] = []

    # Phase 1 -- fetch every provider's images.
    fetched_products: list[tuple[str, list[ImageProduct]]] = []
    for name in instruments:
        print(f"=== {name} images ===")
        cache_dir = phot_dir / instrument_dirs.get(name, name)
        fetch = IMAGE_PROVIDERS[name]
        options: dict = {'bands': bands, 'size_arcsec': cutout_arcsec,
                         'cache_dir': cache_dir}
        if name == 'legacy':
            options.update(dr=legacy_dr, use_bricks=legacy_bricks)
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

    # Phase 2 -- sersic mode: resolve the one sky shape all bands share.
    shape_sky = None
    shape_origin = None
    if mode == 'sersic':
        all_products = [p for _, products in fetched_products for p in products]
        shape_sky, shape_origin = _resolve_shape(
            all_products, coord, sersic_from=sersic_from,
            sersic_params=sersic_params, sky_in=sky_in, sky_out=sky_out,
            cutout_half_arcsec=cutout_arcsec / 2.0, user_mask=user_mask,
            protect_radius=protect_radius, sersic_seeing=sersic_seeing)
        print(f"Forced shape: n={shape_sky['n']:.2f}, "
              f"reff={shape_sky['reff_arcsec']:.2f}\", "
              f"ellip={shape_sky['ellip']:.2f}, PA={shape_sky['pa_deg']:.1f} deg "
              f"({shape_origin['source']})\n")

    # Phase 3 -- measure every band.
    for name, products in fetched_products:
        cache_dir = phot_dir / instrument_dirs.get(name, name)
        provider_rows: list[dict] = []
        measured_bands: list[str] = []
        for product in products:
            try:
                if mode == 'sersic':
                    from .measure.sersic import forced_to_row, measure_forced
                    measurement = measure_forced(
                        product, coord, shape_sky,
                        sky_in=sky_in, sky_out=sky_out,
                        cutout_half_arcsec=cutout_arcsec / 2.0,
                        user_mask=user_mask, protect_radius=protect_radius,
                        rgrid=np.asarray(rgrid, dtype=float) if rgrid else None)
                    row = forced_to_row(measurement)
                    figure = qa_forced_figure(measurement, cache_dir / "QA")
                else:
                    measurement = measure_aperture(
                        product, coord,
                        aperture_arcsec=aperture_arcsec,
                        sky_in=sky_in, sky_out=sky_out,
                        cutout_half_arcsec=cutout_arcsec / 2.0,
                        rgrid=np.asarray(rgrid, dtype=float) if rgrid else None,
                        user_mask=user_mask,
                        protect_radius=protect_radius)
                    row = measurement_to_row(measurement)
                    figure = qa_band_figure(measurement, cache_dir / "QA")
            except Exception as e:
                print(f"  {product.instrument} {product.band} FAILED: "
                      f"{type(e).__name__}: {e}")
                continue
            measurements.append(measurement)
            provider_rows.append(row)
            measured_bands.append(product.band)
            print(f"  {product.instrument} {product.band}: "
                  f"{measurement['flux_ujy']:.1f} +/- "
                  f"{measurement['flux_err_ujy']:.1f} uJy "
                  f"({measurement['err_model']}; QA {figure.name})")
        rows.extend(provider_rows)
        results.append(ProviderResult(
            provider=name, status=STATUS_OK if measured_bands else STATUS_ERROR,
            rows=provider_rows,
            message=f"measured bands: {', '.join(measured_bands)}" if measured_bands
                    else "fetched images but every measurement failed"))
        print()

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
        "aperture_arcsec": aperture_arcsec if mode == 'aperture' else None,
        "sersic_shape": {**shape_sky, **shape_origin} if shape_sky else None,
        "sky_annulus_arcsec": [sky_in, sky_out],
        "cutout_arcsec": cutout_arcsec,
        "mask": {"mode": "user" if mask_file else "auto",
                 "file": mask_file, "ref_image": mask_ref,
                 "protect_radius_arcsec": protect_radius},
        "legacy": {"dr": legacy_dr, "bricks": legacy_bricks}
                  if 'legacy' in instruments else None,
        "per_band": {f"{m['instrument']}_{m['band']}":
                     {"err_model": m['err_model'],
                      "sky_level_ujy_per_px": round(m['sky_level_ujy'], 6),
                      "n_masked_in_aperture": m['n_masked_in_aperture']}
                     for m in measurements},
    })
    print(f"\nSaved {len(measured_df)} measured bands to: {out_csv}")
    print(measured_df.to_string(index=False))

    return measured_df
