#!/usr/bin/env python3
"""
cli.py

sedphot Command-Line Interface
---------------------------------------------------------
Galaxy in, SED photometry out. Every subcommand takes the same target spec:
a resolvable name (--name) or an explicit position (--ra --dec), plus the
galaxy directory products land in (--out-dir, default '.').

Usage:
    sedphot resolve  (--name NAME | --ra DEG --dec DEG)
    sedphot catalogs (--name NAME | --ra DEG --dec DEG)
                     (--instruments legacy panstarrs hst | --all)
                     [--radius 2.0] [--legacy-dr {dr10,dr9}]
                     [--out-dir DIR] [--label STEM]

Examples:
    Resolve a name to coordinates and the default output label:
        sedphot resolve --name "SDSS J142800.81+570046.3"

    All catalog photometry for a position, into the current directory:
        sedphot catalogs --ra 216.988087 --dec 56.987800 --all --legacy-dr dr9

    Legacy + Pan-STARRS only, into a galaxy directory:
        sedphot catalogs --name "M87" --instruments legacy panstarrs \\
            --out-dir Clusters/Virgo/Galaxies/M87
"""
from __future__ import annotations

import argparse
import sys

from .catalogs import CATALOG_PROVIDERS
from .images import IMAGE_PROVIDERS
from .pipeline import run_catalogs, run_measure
from .resolve import resolve_target


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


def _cmd_measure(args: argparse.Namespace) -> None:
    coord, label = _resolve_from_args(args)
    instruments = _instruments_from_args(args, IMAGE_PROVIDERS)
    if args.mode != 'aperture':
        print("error: --mode sersic lands with the forced-Sersic milestone; "
              "only 'aperture' is available", file=sys.stderr)
        sys.exit(2)
    run_measure(
        coord, label, args.out_dir,
        instruments=instruments,
        bands=args.bands,
        aperture_arcsec=args.aperture,
        sky_in=args.sky_in,
        sky_out=args.sky_out,
        cutout_arcsec=args.cutout_size,
        rgrid=args.radii,
        mask_file=args.mask,
        mask_ref=args.mask_ref,
        protect_radius=args.protect_radius,
        legacy_dr=args.legacy_dr,
        legacy_bricks=args.legacy_bricks,
        target_name=args.name,
    )


# ------------------------------------
# Parser
# ------------------------------------
def build_parser() -> argparse.ArgumentParser:
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
    p_catalogs.add_argument('--legacy-dr', type=str, default='dr10',
                            choices=('dr10', 'dr9'),
                            help="Legacy Surveys data release [default: dr10]")
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
    p_measure.add_argument('--mode', type=str, default='aperture',
                           choices=('aperture', 'sersic'),
                           help="Measurement mode [default: aperture]")
    p_measure.add_argument('--bands', nargs='+', default=None,
                           help="Band subset for every provider "
                                "[default: provider defaults]")
    p_measure.add_argument('--aperture', type=float, default=10.0,
                           help="Aperture radius in arcsec [default: 10.0]")
    p_measure.add_argument('--radii', nargs='+', type=float, default=None,
                           help="Curve-of-growth radii override (arcsec)")
    p_measure.add_argument('--sky-in', type=float, default=30.0,
                           help="Sky annulus inner radius, arcsec [default: 30]")
    p_measure.add_argument('--sky-out', type=float, default=45.0,
                           help="Sky annulus outer radius, arcsec [default: 45]")
    p_measure.add_argument('--cutout-size', type=float, default=120.0,
                           help="Stamp width in arcsec; must contain the sky "
                                "annulus [default: 120]")
    p_measure.add_argument('--mask', type=str, default=None,
                           help="User mask file (.npz neighbor_mask or FITS) "
                                "instead of the auto-mask")
    p_measure.add_argument('--mask-ref', type=str, default=None,
                           help="Reference image whose WCS an .npz mask's grid "
                                "is defined on (A1925 staged-mask pairing)")
    p_measure.add_argument('--protect-radius', type=float, default=4.0,
                           help="Auto-mask: radius never masked around the "
                                "target, arcsec [default: 4.0]")
    p_measure.add_argument('--legacy-dr', type=str, default='dr9',
                           choices=('dr10', 'dr9'),
                           help="Legacy release for images [default: dr9]")
    p_measure.add_argument('--legacy-bricks', action='store_true',
                           help="Fetch NERSC brick coadds (image + invvar; "
                                "real per-pixel errors, ~40 MB/file)")
    p_measure.set_defaults(func=_cmd_measure)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
