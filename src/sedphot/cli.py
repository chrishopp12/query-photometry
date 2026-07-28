#!/usr/bin/env python3
"""
cli.py

sedphot Command-Line Interface
---------------------------------------------------------
Galaxy in, SED photometry out. resolve and remeasure only print; the
other subcommands write into the galaxy directory given by --out-dir
(default '.', so products land in ./Photometry/ of the current
directory). resolve, catalogs, measure, spherex and run take the same
target spec: a resolvable name (--name) or an explicit position
(--ra --dec). sed and overlay work from tables already on disk, and
remeasure from a provenance sidecar path.

Usage:
    sedphot resolve  (--name NAME | --ra DEG --dec DEG)
    sedphot catalogs (--name NAME | --ra DEG --dec DEG)
                     (--instruments legacy panstarrs hst ... | --all)
                     [--radius 2.0] [--legacy-dr {dr10,dr9}] [--dered]
    sedphot measure  (--name NAME | --ra DEG --dec DEG)
                     (--instruments legacy sdss cfht ... | --all)
                     [--mode {aperture,sersic}] [--aperture 10.0]
                     [--sky-rmin AS] [--registry FILE [--registry-update]]
    sedphot spherex  (--name NAME | --ra DEG --dec DEG)
                     [--model {psf,sersic}] [--sersic-params N AXR PA RE]
    sedphot sed      [--out-dir DIR] [--label STEM]
    sedphot overlay  [--out-dir DIR] [--label STEM] [--zoom-size 5.0]
                     [--context-size 15.0] [--wcs-from FITS]
    sedphot remeasure PROVENANCE.json [--mode {sersic,aperture}]
                     [--aperture 12.0 | --integrated]
                     [--shape {forced,fitted}] [--registry FILE]
                     [--write-qa] [--out CSV]
    sedphot run      (--name NAME | --ra DEG --dec DEG) [--skip ...]
                     [--spherex {off,psf,sersic}]
                     [every flag in the measurement group -- see
                      `sedphot measure --help`; --dump-arrays is
                      measure-only]

Examples:
    Resolve a name to coordinates and the default output label:
        sedphot resolve --name "NGC 4889"

    All catalog photometry for a position, into the current directory
    (add --legacy-dr dr10 for the i-band southern release):
        sedphot catalogs --ra 194.898792 --dec 27.959528 --all

    Legacy + Pan-STARRS only, into a galaxy directory:
        sedphot catalogs --name "M87" --instruments legacy panstarrs \\
            --out-dir Clusters/Virgo/Galaxies/M87

    Uniform aperture photometry on every available image:
        sedphot measure --name "M87" --all --aperture 10 --out-dir M87
"""
from __future__ import annotations

import argparse
import sys

from .catalogs import CATALOG_PROVIDERS
from .catalogs.legacy import LEGACY_DR_DEFAULT
from .images import IMAGE_PROVIDERS
from .pipeline import (run_all, run_catalogs, run_measure, run_overlay,
                       run_sed, run_spherex)
from .resolve import resolve_target
from .results import STATUS_OK


# ------------------------------------
# Shared argument groups
# ------------------------------------
def _add_target_args(parser: argparse.ArgumentParser) -> None:
    """The target spec + output location shared by every subcommand."""
    group = parser.add_argument_group("target")
    group.add_argument('--name', type=str, default=None,
                       help="Resolvable object name (Sesame -> NED -> SIMBAD)")
    group.add_argument('--ra', type=float, default=None,
                       help="RA in decimal degrees (with --dec, instead of --name)")
    group.add_argument('--dec', type=float, default=None,
                       help="Dec in decimal degrees")
    group.add_argument('--out-dir', type=str, default=".",
                       help="Galaxy directory; products land in <out-dir>/Photometry/ "
                            "[default: .]")
    group.add_argument('--label', type=str, default=None,
                       help="Output filename stem [default: sanitized name or J-name]")


def _resolve_from_args(args: argparse.Namespace):
    """Resolve the target spec, exiting with a clean argparse-style error."""
    try:
        return resolve_target(name=args.name, ra=args.ra, dec=args.dec,
                              label=args.label)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)


def _add_measure_args(parser: argparse.ArgumentParser) -> None:
    """Every measurement option, shared by the measure and run verbs.

    Defined once so run always accepts the same set as measure; a flag run
    silently lacked would show up only as a wrong flux.
    """
    group = parser.add_argument_group("measurement")
    group.add_argument('--mode', type=str, default='aperture',
                       choices=('aperture', 'sersic'),
                       help="Measurement mode [default: aperture]")
    group.add_argument('--bands', nargs='+', default=None,
                       help="Band subset for every provider "
                            "[default: provider defaults]")
    group.add_argument('--aperture', type=float, default=10.0,
                       help="Aperture radius in arcsec [default: 10.0]")
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
    group.add_argument('--registry', type=str, default=None,
                       help="Cross-field registry JSON to consume (solved "
                            "shared sources enter as frozen components)")
    group.add_argument('--registry-update', action='store_true',
                       help="Also write this galaxy's solved seats back to "
                            "--registry (updates are last-writer-wins; run "
                            "sweeps serially)")
    group.add_argument('--sersic-from', type=str, default=None,
                       help="Sersic mode: pin the target shape to a fit on "
                            "this band ('z' or 'Legacy_z') [default: "
                            "per-instrument reference-band refit]")
    group.add_argument('--sersic-params', nargs=4, type=float, default=None,
                       metavar=('N', 'AXRATIO', 'PA_DEG', 'REFF_AS'),
                       help="Sersic mode: explicit shape (n, a/b >= 1, PA "
                            "deg E of N, r_eff arcsec) -- pins the target "
                            "profile in every band")
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
    group.add_argument('--hst-proposal-id', type=str, default=None,
                       help="Restrict the HST provider to one program")


def _instruments_from_args(args: argparse.Namespace, registry: dict) -> list[str]:
    """Validate the --instruments/--all selection against a provider registry."""
    if args.all:
        return list(registry)
    if not args.instruments:
        print("error: give --instruments ... or --all", file=sys.stderr)
        sys.exit(2)
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
        hst_proposal_id=args.hst_proposal_id,
        target_name=args.name,
    )
    if failures:
        sys.exit(1)


def _check_measure_args(args: argparse.Namespace, verb: str) -> None:
    """Refuse a measurement request whose flags contradict each other.

    A shape flag under --mode aperture has nothing to act on: the run would
    report curve-of-growth fluxes while the caller expects a forced shape.
    Refuse rather than ignore -- a dropped flag is a wrong answer waiting to
    be trusted.
    """
    if args.registry_update and not args.registry:
        sys.exit(f"sedphot {verb}: --registry-update needs --registry PATH")
    shape_flags = [name for name, value in
                   (('--sersic-from', args.sersic_from),
                    ('--sersic-params', args.sersic_params),
                    ('--sersic-seeing', args.sersic_seeing))
                   if value is not None]
    # Under `run`, --sersic-params also declares the SPHEREx extraction
    # shape, so it is meaningful with an aperture-mode measurement.
    if verb == 'run' and getattr(args, 'spherex', 'off') != 'off':
        shape_flags = [f for f in shape_flags if f != '--sersic-params']
    if args.mode != 'sersic' and shape_flags:
        sys.exit(f"sedphot {verb}: {', '.join(shape_flags)} only applies to "
                 f"--mode sersic (got --mode {args.mode}); drop the flag or "
                 f"pass --mode sersic")


def _cmd_measure(args: argparse.Namespace) -> None:
    _check_measure_args(args, 'measure')
    coord, label = _resolve_from_args(args)
    instruments = _instruments_from_args(args, IMAGE_PROVIDERS)
    run_measure(
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
    _add_target_args(p_resolve)
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
                           help="Sersic mode: explicit shape (n<=6, a/b >= 1, "
                                "PA deg E of N, r_eff arcsec)")
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
    p_sed.add_argument('--out-dir', type=str, default=".",
                       help="Galaxy directory [default: .]")
    p_sed.add_argument('--label', type=str, default=None,
                       help="Output stem [default: inferred when unambiguous]")
    p_sed.set_defaults(func=_cmd_sed)

    p_overlay = subparsers.add_parser(
        "overlay",
        help="Overlay each catalog's matched position on the HAP color image")
    p_overlay.add_argument('--out-dir', type=str, default=".",
                           help="Galaxy directory [default: .]")
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
    p_remeasure.add_argument('--mode', choices=['sersic', 'aperture'],
                             default='sersic',
                             help="sersic: fitted-model flux; aperture: "
                                  "empirical neighbor-subtracted flux "
                                  "[default: sersic]")
    p_remeasure.add_argument('--aperture', type=float, default=12.0,
                             help="Circular aperture radius, arcsec. Note "
                                  "this differs from `measure --aperture` "
                                  "(10); pass the aperture the sidecar "
                                  "records to reproduce the original "
                                  "measurement [default: 12]")
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
    p_run.add_argument('--skip', nargs='+', default=None, choices=all_providers,
                       help="Providers to leave out")
    p_run.add_argument('--radius', type=float, default=2.0,
                       help="Catalog search radius, arcsec [default: 2.0]")
    p_run.add_argument('--dered', action='store_true',
                       help="Apply MW dereddening to catalog fluxes")
    _add_measure_args(p_run)
    p_run.add_argument('--spherex', type=str, default='off',
                       choices=('off', 'psf', 'sersic'),
                       help="Also fetch SPHEREx spectrophotometry [default: off]")
    p_run.set_defaults(func=_cmd_run)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
