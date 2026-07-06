"""
resolve.py

Target Name and Position Resolution
---------------------------------------------------------
Turn a galaxy name or an explicit (ra, dec) into the SkyCoord + output label
the rest of the package runs on. Name resolution tries, in order:

    1. CDS Sesame (astropy SkyCoord.from_name -- covers most catalogs,
       including SDSS Jhhmmss designations)
    2. NED  (astroquery.ipac.ned)
    3. SIMBAD (astroquery.simbad)

Requirements:
    astropy, astroquery

Notes:
    resolve_target is the single entry point; providers never take raw names.
    Names go to the services as-is -- no cluster-alias rewriting
    (ACO/redMaPPer); this package resolves galaxies.
"""
from __future__ import annotations

import re

import astropy.units as u
from astropy.coordinates import SkyCoord


# ------------------------------------
# Name resolution chain
# ------------------------------------
def _sesame(name: str) -> SkyCoord:
    return SkyCoord.from_name(name)


def _ned(name: str) -> SkyCoord:
    from astroquery.ipac.ned import Ned
    result = Ned.query_object(name)
    if result is None or len(result) == 0:
        raise ValueError(f"NED returned no results for {name!r}")
    return SkyCoord(float(result['RA'][0]), float(result['DEC'][0]), unit=u.deg)


def _simbad(name: str) -> SkyCoord:
    from astroquery.simbad import Simbad
    simbad = Simbad()
    result = simbad.query_object(name)
    if result is None or len(result) == 0:
        raise ValueError(f"SIMBAD returned no results for {name!r}")
    cols = {c.lower(): c for c in result.colnames}
    ra_col, dec_col = cols.get('ra'), cols.get('dec')
    if ra_col is None or dec_col is None:
        raise ValueError(f"No usable RA/Dec columns in SIMBAD result for {name!r}")
    return SkyCoord(result[ra_col][0], result[dec_col][0], unit=u.deg)


def resolve_name(name: str, *, verbose: bool = True) -> SkyCoord:
    """Resolve a target name to ICRS coordinates via Sesame -> NED -> SIMBAD.

    Parameters
    ----------
    name : str
        Any resolvable object name.
    verbose : bool
        Print which service answered. [default: True]

    Returns
    -------
    coord : SkyCoord
        ICRS position of the target.

    Raises
    ------
    ValueError
        If every service fails.
    """
    for label, resolver in (("Sesame", _sesame), ("NED", _ned), ("SIMBAD", _simbad)):
        try:
            coord = resolver(name)
            if verbose:
                print(f"  [resolve] {name!r} -> RA={coord.ra.deg:.6f}, "
                      f"Dec={coord.dec.deg:+.6f}  ({label})")
            return coord
        except Exception as e:
            if verbose:
                print(f"  [resolve] {label} failed for {name!r}: {e}")
    raise ValueError(f"Could not resolve {name!r} with Sesame, NED, or SIMBAD.")


# ------------------------------------
# Labels
# ------------------------------------
def sanitize_label(name: str) -> str:
    """Reduce a target name to a filesystem-safe output stem."""
    label = re.sub(r"[^\w+-]+", "_", name.strip()).strip("_")
    return label or "target"


def jname(coord: SkyCoord) -> str:
    """IAU-style Jhhmmss.ss+ddmmss.s label for an anonymous position."""
    ra = coord.ra.to_string(unit=u.hourangle, sep="", precision=2, pad=True)
    dec = coord.dec.to_string(unit=u.deg, sep="", precision=1,
                              alwayssign=True, pad=True)
    return f"J{ra}{dec}"


# ------------------------------------
# Entry point
# ------------------------------------
def resolve_target(
        *,
        name: str | None = None,
        ra: float | None = None,
        dec: float | None = None,
        label: str | None = None,
        verbose: bool = True,
) -> tuple[SkyCoord, str]:
    """Resolve the CLI target spec to (coord, output label).

    Exactly one of name or (ra, dec) must be given.

    Parameters
    ----------
    name : str, optional
        Resolvable object name.
    ra, dec : float, optional
        Explicit ICRS position in decimal degrees.
    label : str, optional
        Output stem override; defaults to the sanitized name or a J-name.
    verbose : bool
        Print resolution diagnostics. [default: True]

    Returns
    -------
    coord : SkyCoord
    out_label : str
    """
    has_name = name is not None
    has_pos = ra is not None and dec is not None
    if has_name == has_pos:
        raise ValueError("Give exactly one of --name or --ra/--dec.")

    if has_name:
        coord = resolve_name(name, verbose=verbose)
        out_label = label or sanitize_label(name)
    else:
        coord = SkyCoord(float(ra), float(dec), unit=u.deg)
        out_label = label or jname(coord)
    return coord, out_label
