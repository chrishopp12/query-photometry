#!/usr/bin/env python3
"""
cli.py

sedphot Command-Line Interface
---------------------------------------------------------
Galaxy in, SED photometry out. resolve and remeasure only print; every
other subcommand requires --out-dir, the galaxy directory products,
image caches and the scene cache all land under. resolve, catalogs,
measure, spherex and run take the same target spec: a resolvable name
(--name) or an explicit position (--ra --dec). sed and overlay work
from tables already on disk, and remeasure from a provenance sidecar
path.

Usage:
    sedphot resolve  (--name NAME | --ra DEG --dec DEG)
    sedphot catalogs (--name NAME | --ra DEG --dec DEG) --out-dir DIR
                     (--instruments legacy panstarrs hst ... | --all)
                     [--radius 2.0] [--legacy-dr {dr10,dr9}] [--dered]
    sedphot measure  (--name NAME | --ra DEG --dec DEG) --out-dir DIR
                     (--instruments legacy sdss cfht ... | --all)
                     [--mode {aperture,sersic}] [--aperture 12.0]
                     [--sky-rmin AS] [--registry FILE [--registry-update]]
                     [--legacy-route {auto,viewer,noirlab}]
    sedphot spherex  (--name NAME | --ra DEG --dec DEG) --out-dir DIR
                     [--model {psf,sersic}] [--sersic-params N AXR PA RE]
    sedphot plan     --targets CSV --out PLAN.json [--out-root DIR]
                     [--cutout-size 120.0] [--link-margin AS]
    sedphot batch    --plan PLAN.json --registry-dir DIR --report CSV
                     [--workers 4] [--pass {harvest,parallel,all}]
                     [--no-groups] [--no-resume] [--log-dir DIR]
                     [--catalogs ... | all | none] [--images ... | all | none]
                     [the measurement group, minus --registry*]
    sedphot sed      --out-dir DIR [--label STEM]
    sedphot overlay  --out-dir DIR [--label STEM] [--zoom-size 5.0]
                     [--context-size 15.0] [--wcs-from FITS]
    sedphot remeasure PROVENANCE.json [--mode {sersic,aperture}]
                     [--aperture 12.0 | --integrated]
                     [--shape {forced,fitted}] [--registry FILE]
                     [--write-qa] [--out CSV]
    sedphot run      (--name NAME | --ra DEG --dec DEG) --out-dir DIR
                     [--catalogs ... | all | none] [--images ... | all | none]
                     [--skip ...] [--spherex {off,psf,sersic}]
                     [every flag in the measurement group -- see
                      `sedphot measure --help`; --dump-arrays is
                      measure-only]

Examples:
    Resolve a name to coordinates and the default output label:
        sedphot resolve --name "NGC 4889"

    All catalog photometry for a position
    (add --legacy-dr dr10 for the i-band southern release):
        sedphot catalogs --ra 194.898792 --dec 27.959528 --all \\
            --out-dir Galaxies/J125935.7+275734

    Legacy + Pan-STARRS only:
        sedphot catalogs --name "M87" --instruments legacy panstarrs \\
            --out-dir Galaxies/M87

    Uniform aperture photometry on every available image:
        sedphot measure --name "M87" --all --aperture 12 \\
            --out-dir Galaxies/M87

    Pin the Legacy cutout service so a sample stays on one gridding.
    The routes do not frame identically, so under the default 'auto' a
    service outage silently splits a sample between them; pinning makes
    the outage an error instead. The route asked for is recorded in the
    sidecar's legacy.route, and the route that actually answered in its
    image_sources:
        sedphot measure --name "M87" --all --legacy-route noirlab \\
            --out-dir Galaxies/M87
"""
from __future__ import annotations

import argparse
import sys

from .batch import DEFAULT_STOP_AFTER_FAILURES
from .catalogs import CATALOG_PROVIDERS
from .catalogs.legacy import LEGACY_DR_DEFAULT
from .images import IMAGE_PROVIDERS
from .images.legacy import LEGACY_ROUTES, ROUTE_AUTO
from .pipeline import (run_all, run_catalogs, run_measure, run_overlay,
                       run_sed, run_spherex)
from .resolve import resolve_target
from .results import STATUS_OK
from .schedule import DEFAULT_LINK_MARGIN_AS


# ------------------------------------
# Shared argument groups
# ------------------------------------
def _add_target_args(parser: argparse.ArgumentParser, *,
                     needs_out_dir: bool = True) -> None:
    """The target spec + output location shared by every subcommand.

    A verb that writes requires --out-dir: products, image caches and the
    scene cache all land under it, so an unstated one silently builds a
    galaxy tree wherever the shell happens to be standing. resolve only
    prints, so it takes no output directory at all.
    """
    group = parser.add_argument_group("target")
    group.add_argument('--name', type=str, default=None,
                       help="Resolvable object name (Sesame -> NED -> SIMBAD)")
    group.add_argument('--ra', type=float, default=None,
                       help="RA in decimal degrees (with --dec, instead of --name)")
    group.add_argument('--dec', type=float, default=None,
                       help="Dec in decimal degrees")
    if needs_out_dir:
        group.add_argument('--out-dir', type=str, required=True,
                           help="Galaxy directory; products land in "
                                "<out-dir>/Photometry/ (required)")
    group.add_argument('--label', type=str, default=None,
                       help="Output filename stem [default: sanitized name or J-name]")


def _resolve_from_args(args: argparse.Namespace):
    """Resolve the target spec, exiting with a clean argparse-style error."""
    try:
        return resolve_target(name=args.name, ra=args.ra, dec=args.dec,
                              label=args.label)
    except ValueError as e:
        _usage_error(str(e))


def _add_measure_args(parser: argparse.ArgumentParser, *,
                      registry_args: bool = True) -> None:
    """Every measurement option, shared by the measure, run and batch verbs.

    Defined once so run always accepts the same set as measure; a flag run
    silently lacked would show up only as a wrong flux.

    Parameters
    ----------
    registry_args : bool
        Expose --registry/--registry-update. batch sets this False: the
        sweep owns one registry per group during the harvest pass and a
        frozen union afterward, so a caller-supplied pair would
        contradict the plan being executed. [default: True]
    """
    group = parser.add_argument_group("measurement")
    group.add_argument('--mode', type=str, default='aperture',
                       choices=('aperture', 'sersic'),
                       help="Measurement mode [default: aperture]")
    group.add_argument('--bands', nargs='+', default=None,
                       help="Band subset for every provider "
                            "[default: provider defaults]")
    group.add_argument('--aperture', type=float, default=12.0,
                       help="Aperture radius in arcsec [default: 12.0]")
    group.add_argument('--radii', nargs='+', type=float, default=None,
                       help="Curve-of-growth radii override, arcsec "
                            "[default: 2-40 in 1\" steps]")
    group.add_argument('--cutout-size', type=float, default=120.0,
                       help="Stamp width in arcsec [default: 120]")
    group.add_argument('--sky-rmin', type=float, default=None,
                       help="Target/sky boundary in arcsec: sky is "
                            "estimated only beyond it, and no pixel inside "
                            "it votes on the background. The default is "
                            "sized for survey galaxies; a compact HST target "
                            "wants a few arcsec "
                            "[default: recipe.BG_RMIN_AS = 15]")
    if not registry_args:
        parser.set_defaults(registry=None, registry_update=False)
        _add_shape_args(group)
        return
    group.add_argument('--registry', type=str, default=None,
                       help="Cross-field registry JSON to consume (solved "
                            "shared sources enter as frozen components). "
                            "Unlike remeasure, measure does NOT auto-discover "
                            "a sibling registry.json: consuming one changes "
                            "the fluxes, so a fresh measurement has to be "
                            "asked [default: none]")
    group.add_argument('--registry-update', action='store_true',
                       help="Also write this galaxy's solved seats back to "
                            "--registry (updates are last-writer-wins; run "
                            "sweeps serially)")
    _add_shape_args(group)


def _add_shape_args(group) -> None:
    """The shape and provider options, on both sides of the registry pair."""
    group.add_argument('--sersic-from', type=str, default=None,
                       help="Sersic mode: pin the target shape to a fit on "
                            "this band ('z' or 'Legacy_z') [default: "
                            "per-instrument reference-band refit]")
    group.add_argument('--sersic-params', nargs=4, type=float, default=None,
                       metavar=('N', 'AXRATIO', 'PA_DEG', 'REFF_AS'),
                       help="Sersic mode: explicit shape (0.4 <= n <= 6, "
                            "a/b >= 1, PA deg E of N, r_eff arcsec > 0) -- "
                            "pins the target profile in every band")
    group.add_argument('--sersic-seeing', type=float, default=None,
                       help="PSF FWHM (arcsec) assumed by the --sersic-from "
                            "shape fit; fitted n and r_eff are PSF-sensitive")
    group.add_argument('--legacy-dr', type=str, default=LEGACY_DR_DEFAULT,
                       choices=('dr10', 'dr9'),
                       help="Legacy Surveys data release for images and the "
                            f"scene catalog [default: {LEGACY_DR_DEFAULT}]")
    group.add_argument('--legacy-bricks', action='store_true',
                       help="Fetch NERSC brick coadds instead of viewer "
                            "cutouts: adds the inverse-variance map "
                            "(per-pixel errors), ~40 MB/file")
    group.add_argument('--legacy-route', type=str, default=ROUTE_AUTO,
                       choices=LEGACY_ROUTES,
                       help="Which Legacy cutout service to use. 'auto' "
                            "tries the NERSC viewer then falls back to "
                            "NOIRLab; 'viewer' or 'noirlab' pins one and "
                            "fails rather than substituting the other. The "
                            "two do not frame identically, so under 'auto' "
                            "an outage silently splits a sample across two "
                            f"griddings [default: {ROUTE_AUTO}]")
    group.add_argument('--hst-proposal-id', type=str, default=None,
                       help="Restrict the HST provider to one program")


def _usage_error(message: str) -> None:
    """Exit on a command line that cannot be honored.

    Exit 2 is argparse's own code for a usage error; 1 means the command
    ran and did not produce what was asked. Every refusal of the flags
    themselves belongs here so a driver can tell the two apart.
    """
    print(f"error: {message}", file=sys.stderr)
    sys.exit(2)


def _instruments_from_args(args: argparse.Namespace, registry: dict) -> list[str]:
    """Validate the --instruments/--all selection against a provider registry."""
    if args.all:
        return list(registry)
    if not args.instruments:
        _usage_error("give --instruments ... or --all")
    return args.instruments


# ------------------------------------
# Subcommands
# ------------------------------------
def _cmd_resolve(args: argparse.Namespace) -> None:
    coord, label = _resolve_from_args(args)
    print(f"RA  = {coord.ra.deg:.8f}")
    print(f"Dec = {coord.dec.deg:+.8f}")
    print(f"label = {label}")


def _cmd_catalogs(args: argparse.Namespace) -> None:
    coord, label = _resolve_from_args(args)
    instruments = _instruments_from_args(args, CATALOG_PROVIDERS)
    run_catalogs(
        coord, label, args.out_dir,
        instruments=instruments,
        radius_arcsec=args.radius,
        legacy_dr=args.legacy_dr,
        dered=args.dered,
        target_name=args.name,
    )


def _cmd_spherex(args: argparse.Namespace) -> None:
    coord, label = _resolve_from_args(args)
    result = run_spherex(
        coord, label, args.out_dir,
        model=args.model,
        sersic_params=args.sersic_params,
        sersic_from=args.sersic_from,
        sersic_seeing=args.sersic_seeing,
        bkg_size=args.bkg_size,
        mjd_range=args.mjd_range,
        poll=args.poll,
        timeout=args.timeout,
        legacy_dr=args.legacy_dr,
        target_name=args.name,
    )
    if result.status != STATUS_OK:
        sys.exit(1)


def _cmd_sed(args: argparse.Namespace) -> None:
    run_sed(args.label, args.out_dir)


def _cmd_plan(args: argparse.Namespace) -> None:
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from .measure import recipe
    from .schedule import (build_plan, census_from_catalog, fetch_catalog,
                           read_targets, write_plan)

    targets = read_targets(args.targets, out_root=args.out_root)
    print(f"{len(targets)} targets from {args.targets}")

    census = []
    for i, target in enumerate(targets, start=1):
        coord = SkyCoord(target['ra_deg'] * u.deg, target['dec_deg'] * u.deg)
        cat = fetch_catalog(coord, target['dir'],
                            cutout_arcsec=args.cutout_size,
                            legacy_dr=args.legacy_dr)
        entry = census_from_catalog(coord, cat,
                                    cutout_arcsec=args.cutout_size)
        census.append(entry)
        print(f"  [{i}/{len(targets)}] {target['name']}: "
              f"{len(cat)} scene rows, {entry['n_gated']} gated"
              + ("" if entry['matched'] else ", NO CATALOG MATCH")
              + (", gates" if entry['gates'] else ""))

    plan = build_plan(targets, census, cutout_arcsec=args.cutout_size,
                      margin_arcsec=args.link_margin)
    json_path, csv_path = write_plan(plan, args.out)

    print(f"\nscene cone {plan['scene_radius_arcsec']}\", "
          f"gate reach {plan['gate_reach_arcsec']}\", "
          f"link {plan['link_radius_arcsec']}\" "
          f"(margin {plan['link_margin_arcsec']}\")")
    print(f"harvest pass: {plan['n_harvest']} targets in "
          f"{plan['n_groups']} group(s), largest {plan['largest_group']}")
    print(f"parallel pass: {plan['n_parallel']} targets")
    if plan['n_unmatched']:
        print(f"WARNING: {plan['n_unmatched']} target(s) have no catalog "
              f"row within {recipe.TARGET_MATCH_AS}\"")
    print(f"{json_path}\n{csv_path}")

    if plan['n_violations']:
        for v in plan['violations'][:10]:
            print(f"  {v['ra_deg']:.6f} {v['dec_deg']:+.6f} written by "
                  f"{v['written_by']} (group {v['written_group']}), read by "
                  f"{v['read_by']} (group {v['read_group']})")
        sys.exit(f"sedphot plan: {plan['n_violations']} record(s) cross a "
                 f"group boundary, so running the groups concurrently would "
                 f"not equal running them in sequence; raise --link-margin "
                 f"above {plan['link_margin_arcsec']} and re-plan")


def _cmd_overlay(args: argparse.Namespace) -> None:
    if run_overlay(args.label, args.out_dir,
                   zoom_arcsec=args.zoom_size,
                   context_arcsec=args.context_size,
                   wcs_from=args.wcs_from, dpi=args.dpi) is None:
        sys.exit("sedphot overlay: no figure written "
                 "(no tables, or no HAP composite covers this position)")


def _cmd_remeasure(args: argparse.Namespace) -> None:
    from .remeasure import remeasure
    aperture = None if args.integrated else args.aperture
    try:
        table = remeasure(args.provenance, aperture, mode=args.mode,
                          shape=args.shape, registry_path=args.registry,
                          write_qa=args.write_qa)
    except ValueError as e:
        # Turn an actionable refusal (aperture past the stamp, a sidecar with
        # nothing pinnable) into a clean exit, as the other verbs do.
        sys.exit(f"sedphot remeasure: {e}")
    if table.empty:
        sys.exit(f"sedphot remeasure: no band has a stored model in "
                 f"{args.provenance}")
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"wrote {len(table)} bands -> {args.out}")
    else:
        print(table.to_string(index=False))


def _cmd_run(args: argparse.Namespace) -> None:
    _check_measure_args(args, 'run')
    coord, label = _resolve_from_args(args)
    failures = run_all(
        coord, label, args.out_dir,
        skip=args.skip,
        catalogs=args.catalogs,
        images=args.images,
        radius_arcsec=args.radius,
        dered=args.dered,
        mode=args.mode,
        bands=args.bands,
        aperture_arcsec=args.aperture,
        cutout_arcsec=args.cutout_size,
        sky_rmin_arcsec=args.sky_rmin,
        rgrid=args.radii,
        sersic_from=args.sersic_from,
        sersic_seeing=args.sersic_seeing,
        registry_path=args.registry,
        registry_update=args.registry_update,
        spherex_model=args.spherex,
        sersic_params=args.sersic_params,
        legacy_dr=args.legacy_dr,
        legacy_bricks=args.legacy_bricks,
        legacy_route=args.legacy_route,
        hst_proposal_id=args.hst_proposal_id,
        target_name=args.name,
    )
    if failures:
        sys.exit(1)


def _cmd_batch(args: argparse.Namespace) -> None:
    import json

    from .batch import run_sweep

    _check_measure_args(args, 'batch')
    with open(args.plan, encoding='utf-8') as handle:
        plan = json.load(handle)

    # registry_path and registry_update are the sweep's to set: the
    # harvest pass owns one registry per group and the parallel pass
    # runs against the frozen union, so a caller-supplied pair would
    # contradict the plan it is executing.
    run_options = dict(
        skip=args.skip,
        catalogs=args.catalogs,
        images=args.images,
        radius_arcsec=args.radius,
        dered=args.dered,
        mode=args.mode,
        bands=args.bands,
        aperture_arcsec=args.aperture,
        cutout_arcsec=args.cutout_size,
        sky_rmin_arcsec=args.sky_rmin,
        rgrid=args.radii,
        sersic_from=args.sersic_from,
        sersic_seeing=args.sersic_seeing,
        spherex_model=args.spherex,
        sersic_params=args.sersic_params,
        legacy_dr=args.legacy_dr,
        legacy_bricks=args.legacy_bricks,
        legacy_route=args.legacy_route,
        hst_proposal_id=args.hst_proposal_id,
    )
    try:
        summary = run_sweep(
            plan, registry_dir=args.registry_dir, run_options=run_options,
            which_pass=args.pass_, workers=args.workers,
            resume=not args.no_resume, groups=not args.no_groups,
            report_path=args.report, log_dir=args.log_dir,
            stop_after_failures=args.stop_after_failures)
    except ValueError as err:
        sys.exit(f"sedphot batch: {err}")

    for problem in summary['merge_problems'][:10]:
        print(f"  shared key: {problem}")
    for violation in summary['violations'][:10]:
        print(f"  crossing: {violation['ra_deg']:.6f} "
              f"{violation['dec_deg']:+.6f} written in group "
              f"{violation['written_group']}, read by {violation['read_by']} "
              f"in group {violation['read_group']}")
    if summary['aborted'] or summary['n_failed']:
        sys.exit(1)


def _check_measure_args(args: argparse.Namespace, verb: str) -> None:
    """Refuse a measurement request whose flags contradict each other.

    A shape flag under --mode aperture has nothing to act on: the run would
    report curve-of-growth fluxes while the caller expects a forced shape.
    Refuse rather than ignore -- a dropped flag is a wrong answer waiting to
    be trusted.
    """
    if args.registry_update and not args.registry:
        _usage_error(f"sedphot {verb}: --registry-update needs --registry PATH")
    shape_flags = [name for name, value in
                   (('--sersic-from', args.sersic_from),
                    ('--sersic-params', args.sersic_params),
                    ('--sersic-seeing', args.sersic_seeing))
                   if value is not None]
    # Under `run`, the shape flags also declare the SPHEREx extraction
    # shape, so they are meaningful with an aperture-mode measurement.
    # run_all forwards all three to run_spherex, so all three are exempt.
    if verb == 'run' and getattr(args, 'spherex', 'off') != 'off':
        shape_flags = []
    if args.mode != 'sersic' and shape_flags:
        _usage_error(f"sedphot {verb}: {', '.join(shape_flags)} only applies "
                     f"to --mode sersic (got --mode {args.mode}); drop the "
                     f"flag or pass --mode sersic")


def _cmd_measure(args: argparse.Namespace) -> None:
    _check_measure_args(args, 'measure')
    coord, label = _resolve_from_args(args)
    instruments = _instruments_from_args(args, IMAGE_PROVIDERS)
    try:
        measured = _run_measure_from_args(args, coord, label, instruments)
    except ValueError as e:
        # An actionable refusal -- an aperture outside the curve-of-growth
        # grid, an unknown provider -- reads as a crash when it arrives as a
        # traceback. Same treatment the other verbs give their own refusals.
        sys.exit(f"sedphot measure: {e}")
    # A band that raised leaves the table short while every provider still
    # reports ok. Exiting 0 there tells a driver the galaxy is done.
    failed = (measured.attrs.get('absent_bands') or {}).get('failed', [])
    if failed:
        sys.exit(f"sedphot measure: {len(failed)} band(s) failed and are "
                 f"absent from the table: {', '.join(failed)}")


def _run_measure_from_args(args: argparse.Namespace, coord, label,
                           instruments: list[str]):
    """Forward the parsed measurement options to the driver."""
    return run_measure(
        coord, label, args.out_dir,
        instruments=instruments,
        mode=args.mode,
        bands=args.bands,
        sersic_from=args.sersic_from,
        sersic_params=args.sersic_params,
        sersic_seeing=args.sersic_seeing,
        aperture_arcsec=args.aperture,
        cutout_arcsec=args.cutout_size,
        sky_rmin_arcsec=args.sky_rmin,
        rgrid=args.radii,
        registry_path=args.registry,
        registry_update=args.registry_update,
        dump_arrays=args.dump_arrays,
        legacy_dr=args.legacy_dr,
        legacy_bricks=args.legacy_bricks,
        legacy_route=args.legacy_route,
        hst_proposal_id=args.hst_proposal_id,
        target_name=args.name,
    )


# ------------------------------------
# Parser
# ------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Assemble the argparse tree: one subparser per verb."""
    parser = argparse.ArgumentParser(
        prog="sedphot",
        description="Galaxy in, SED photometry out: multi-archive retrieval and measurement.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_resolve = subparsers.add_parser(
        "resolve", help="Resolve a target name/position and print it")
    _add_target_args(p_resolve, needs_out_dir=False)
    p_resolve.set_defaults(func=_cmd_resolve)

    p_catalogs = subparsers.add_parser(
        "catalogs", help="Retrieve catalog photometry from the selected archives")
    _add_target_args(p_catalogs)
    p_catalogs.add_argument('--instruments', nargs='+', default=None,
                            choices=sorted(CATALOG_PROVIDERS),
                            help="Catalog providers to query")
    p_catalogs.add_argument('--all', action='store_true',
                            help="Query every registered catalog provider")
    p_catalogs.add_argument('--radius', type=float, default=2.0,
                            help="Starting search radius in arcsec [default: 2.0]")
    p_catalogs.add_argument('--legacy-dr', type=str, default=LEGACY_DR_DEFAULT,
                            choices=('dr10', 'dr9'),
                            help="Legacy Surveys data release "
                                 f"[default: {LEGACY_DR_DEFAULT}]")
    p_catalogs.add_argument('--dered', action='store_true',
                            help="Apply MW dereddening (default: as-measured; "
                                 "corrections recorded per row)")
    p_catalogs.set_defaults(func=_cmd_catalogs)

    p_measure = subparsers.add_parser(
        "measure", help="Fetch images and run uniform aperture photometry")
    _add_target_args(p_measure)
    p_measure.add_argument('--instruments', nargs='+', default=None,
                           choices=sorted(IMAGE_PROVIDERS),
                           help="Image providers to fetch and measure")
    p_measure.add_argument('--all', action='store_true',
                           help="Every registered image provider")
    _add_measure_args(p_measure)
    p_measure.add_argument('--dump-arrays', action='store_true',
                           help="Write per-band array bundles under <Inst>/QA/ "
                                "(debug)")
    p_measure.set_defaults(func=_cmd_measure)

    p_spherex = subparsers.add_parser(
        "spherex", help="Fetch the raw SPHEREx spectrophotometry table (IRSA)")
    _add_target_args(p_spherex)
    p_spherex.add_argument('--model', type=str, default='sersic',
                           choices=('psf', 'sersic'),
                           help="Forced-photometry source model; psf carries a "
                                "chromatic bias for extended sources "
                                "[default: sersic]")
    p_spherex.add_argument('--sersic-params', nargs=4, type=float, default=None,
                           metavar=('N', 'AXRATIO', 'PA_DEG', 'REFF_AS'),
                           help="Sersic mode: explicit shape (0.4 <= n <= 6, "
                                "a/b >= 1, PA deg E of N, r_eff arcsec > 0)")
    p_spherex.add_argument('--sersic-from', type=str, default=None,
                           help="Sersic mode: fit the shape on this band's "
                                "image ('Legacy_z' or 'z') instead of the "
                                "default Tractor catalog lookup")
    p_spherex.add_argument('--sersic-seeing', type=float, default=None,
                           help="PSF FWHM (arcsec) of the shape-fit band")
    p_spherex.add_argument('--bkg-size', type=float, default=15.0,
                           help="Background estimation region, pixels [default: 15]")
    p_spherex.add_argument('--mjd-range', nargs=2, type=float, default=None,
                           metavar=('MJD_START', 'MJD_END'),
                           help="Restrict to visits in this MJD window (the IRSA "
                                "workaround for broken-metadata epochs)")
    p_spherex.add_argument('--poll', type=float, default=5.0,
                           help="Job poll interval, seconds [default: 5]")
    p_spherex.add_argument('--timeout', type=float, default=3600.0,
                           help="Job timeout, seconds [default: 3600]")
    p_spherex.add_argument('--legacy-dr', type=str, default=LEGACY_DR_DEFAULT,
                           choices=('dr10', 'dr9'),
                           help="Legacy Surveys data release for a shape-fit "
                                f"image [default: {LEGACY_DR_DEFAULT}]")
    p_spherex.set_defaults(func=_cmd_spherex)

    p_sed = subparsers.add_parser(
        "sed", help="Combined SED plot from the tables already in out-dir")
    p_sed.add_argument('--out-dir', type=str, required=True,
                       help="Galaxy directory holding the tables (required)")
    p_sed.add_argument('--label', type=str, default=None,
                       help="Output stem [default: inferred when unambiguous]")
    p_sed.set_defaults(func=_cmd_sed)

    p_plan = subparsers.add_parser(
        "plan", help="Plan a multi-target sweep: gate census, groups, order")
    p_plan.add_argument('--targets', type=str, required=True,
                        help="Target CSV: a name column, ra_deg/dec_deg, "
                             "an optional dir and priority")
    p_plan.add_argument('--out', type=str, required=True,
                        help="Plan JSON to write; the flat CSV goes beside it")
    p_plan.add_argument('--out-root', type=str, default=None,
                        help="Parent directory for targets whose CSV row "
                             "names no dir")
    p_plan.add_argument('--cutout-size', type=float, default=120.0,
                        help="Stamp width the sweep will use; both the scene "
                             "cone and the gate reach derive from it "
                             "[default: 120.0]")
    p_plan.add_argument('--link-margin', type=float,
                        default=DEFAULT_LINK_MARGIN_AS,
                        help="Padding past the modeled write-plus-read span "
                             "before two fields are called independent "
                             f"[default: {DEFAULT_LINK_MARGIN_AS}]")
    p_plan.add_argument('--legacy-dr', type=str, default=LEGACY_DR_DEFAULT,
                        choices=['dr10', 'dr9'],
                        help=f"Scene catalog release [default: {LEGACY_DR_DEFAULT}]")
    p_plan.set_defaults(func=_cmd_plan)

    p_overlay = subparsers.add_parser(
        "overlay",
        help="Overlay each catalog's matched position on the HAP color image")
    p_overlay.add_argument('--out-dir', type=str, required=True,
                           help="Galaxy directory holding the tables (required)")
    p_overlay.add_argument('--label', type=str, default=None,
                           help="Output stem [default: inferred when "
                                "unambiguous]")
    p_overlay.add_argument('--zoom-size', type=float, default=5.0,
                           help="Zoom panel half-width in arcsec [default: 5]")
    p_overlay.add_argument('--context-size', type=float, default=15.0,
                           help="Context panel half-width in arcsec "
                                "[default: 15]")
    p_overlay.add_argument('--wcs-from', type=str, default=None,
                           help="Local FITS already on the composite's "
                                "drizzle grid, instead of fetching the "
                                "~380 MB detection mosaic for its WCS "
                                "(dimension-checked; a mismatch, or a path "
                                "naming no file, refuses)")
    p_overlay.add_argument('--dpi', type=int, default=200,
                           help="Figure resolution [default: 200]")
    p_overlay.set_defaults(func=_cmd_overlay)

    p_remeasure = subparsers.add_parser(
        "remeasure",
        help="Re-report band fluxes from a stored fit (no re-fetch, no refit)")
    p_remeasure.add_argument('provenance', type=str,
                             help="Path to a <label>_measured.provenance.json")
    p_remeasure.add_argument('--mode', choices=['aperture', 'sersic'],
                             default='aperture',
                             help="aperture: empirical neighbor-subtracted "
                                  "flux; sersic: fitted-model flux. Matches "
                                  "the measure default, like --aperture does, "
                                  "so a bare remeasure re-reports what the "
                                  "measurement measured [default: aperture]")
    p_remeasure.add_argument('--aperture', type=float, default=12.0,
                             help="Circular aperture radius, arcsec; matches "
                                  "the measure default, so a bare remeasure "
                                  "re-reports at the radius the measurement "
                                  "used [default: 12]")
    p_remeasure.add_argument('--shape', choices=['forced', 'fitted'],
                             default='forced',
                             help="Target shape the report is built on. "
                                  "forced: the instrument's reference-band "
                                  "shape. fitted: each band's own free-target "
                                  "shape, stored only for a gating target; a "
                                  "band without one falls back to forced and "
                                  "says so in `source`. Applies to sersic "
                                  "mode always, to aperture mode only past "
                                  "the stored grid [default: forced]")
    p_remeasure.add_argument('--registry', type=str, default=None,
                             help="Cross-field registry fallback for beyond-"
                                  "grid reconstruction [default: sibling of "
                                  "the galaxy directory; the sidecar's own "
                                  "consumed-shape snapshot is used first]")
    p_remeasure.add_argument('--write-qa', action='store_true',
                             help="Write per-band scene figures for a beyond-"
                                  "grid reconstruction to a scoped QA subdir")
    p_remeasure.add_argument('--integrated', action='store_true',
                             help="Report the total instead of an aperture "
                                  "(ignores --aperture): the integrated model "
                                  "in sersic mode, the outermost stored "
                                  "curve-of-growth value in aperture mode")
    p_remeasure.add_argument('--out', type=str, default=None,
                             help="Output CSV path [default: print to stdout]")
    p_remeasure.set_defaults(func=_cmd_remeasure)

    all_providers = sorted(set(CATALOG_PROVIDERS) | set(IMAGE_PROVIDERS))
    p_run = subparsers.add_parser(
        "run", help="Galaxy in, SED photometry out: catalogs + measurement "
                    "+ optional SPHEREx + SED plot")
    _add_target_args(p_run)
    p_run.add_argument('--catalogs', nargs='+', default=None,
                        metavar='PROVIDER',
                        help="Catalog providers to query: names, or 'all' / "
                             "'none'. 'none' skips the catalog stage "
                             "entirely [default: all]")
    p_run.add_argument('--images', nargs='+', default=None,
                        metavar='PROVIDER',
                        help="Image providers to measure: names, or 'all' / "
                             "'none'. 'none' skips the images + measurement "
                             "stage entirely [default: all]")
    p_run.add_argument('--skip', nargs='+', default=None,
                        choices=all_providers,
                        help="Providers to remove from both selections")
    p_run.add_argument('--radius', type=float, default=2.0,
                       help="Catalog search radius, arcsec [default: 2.0]")
    p_run.add_argument('--dered', action='store_true',
                       help="Apply MW dereddening to catalog fluxes")
    _add_measure_args(p_run)
    p_run.add_argument('--spherex', type=str, default='off',
                       choices=('off', 'psf', 'sersic'),
                       help="Also fetch SPHEREx spectrophotometry [default: off]")
    p_run.set_defaults(func=_cmd_run)

    p_batch = subparsers.add_parser(
        "batch", help="Execute a plan: the harvest pass, then the parallel one")
    p_batch.add_argument('--plan', type=str, required=True,
                         help="Plan JSON from `sedphot plan`")
    p_batch.add_argument('--registry-dir', type=str, required=True,
                         help="Per-group registries and the merged, frozen "
                              "one land here")
    p_batch.add_argument('--report', type=str, required=True,
                         help="Per-target report CSV")
    p_batch.add_argument('--log-dir', type=str, default=None,
                         help="Per-target stdout logs [default: interleaved]")
    p_batch.add_argument('--workers', type=int, default=4,
                         help="Concurrent groups, then concurrent targets. "
                              "The parallel pass is bounded by what the "
                              "archives tolerate, not by cores [default: 4]")
    p_batch.add_argument('--pass', dest='pass_', type=str, default='all',
                         choices=('harvest', 'parallel', 'all'),
                         help="Which pass to run [default: all]")
    p_batch.add_argument('--no-groups', action='store_true',
                         help="Run the harvest pass as one sequence against "
                              "one registry: the reference the grouped path "
                              "is judged against")
    p_batch.add_argument('--no-resume', action='store_true',
                         help="Re-measure targets that already have a "
                              "measured table and a readable sidecar")
    p_batch.add_argument('--stop-after-failures', type=int,
                         default=DEFAULT_STOP_AFTER_FAILURES,
                         help="Abandon the sweep after this many failures; "
                              f"0 disables [default: {DEFAULT_STOP_AFTER_FAILURES}]")
    p_batch.add_argument('--catalogs', nargs='+', default=None,
                        metavar='PROVIDER',
                        help="Catalog providers to query: names, or 'all' / "
                             "'none'. 'none' skips the catalog stage "
                             "entirely [default: all]")
    p_batch.add_argument('--images', nargs='+', default=None,
                        metavar='PROVIDER',
                        help="Image providers to measure: names, or 'all' / "
                             "'none'. 'none' skips the images + measurement "
                             "stage entirely [default: all]")
    p_batch.add_argument('--skip', nargs='+', default=None,
                        choices=all_providers,
                        help="Providers to remove from both selections")
    p_batch.add_argument('--radius', type=float, default=2.0,
                         help="Catalog match radius in arcsec [default: 2.0]")
    p_batch.add_argument('--dered', action='store_true',
                         help="Deredden the catalog table")
    p_batch.add_argument('--spherex', type=str, default='off',
                         choices=('off', 'psf', 'sersic'),
                         help="SPHEREx extraction [default: off]")
    _add_measure_args(p_batch, registry_args=False)
    p_batch.set_defaults(func=_cmd_batch)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
