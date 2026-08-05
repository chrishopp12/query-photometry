"""
groups.py

Joint SPHEREx Extraction: Planning and Execution
---------------------------------------------------------

A SPHEREx pixel is ~6.2 arcsec, so a source with a companion inside a
few pixels has that companion's light in its aperture. The IRSA
Spectrophotometry Tool fits several positions simultaneously, which
divides the light between them instead of attributing it all to one.
This module decides which sources belong in a job and runs the jobs.

A job's membership cannot be expressed on a command line: it is a set
of positions, each with its own frozen shape, and the same galaxy fit
alone and fit beside a companion are different measurements. So a
config file is the interface, and it is a REVIEWABLE record -- planning
writes one, a human reads it, execution runs it.

Roles. A science member's spectrum is a product in its own right and
lands in its own galaxy directory, exactly where a single-position
extraction would. An ancillary member is present only to absorb light;
its spectrum is archived with the group. That split is what keeps a
joint extraction consumable by anything that reads one table per galaxy.

Data products:
  <config>              the group plan, JSON
  <config>.csv          the same members, flat, for inspection
  <report>              one row per group
  <log-dir>/<id>.log    per-group stdout

Requirements:
  - numpy, pandas, astropy; requests via spherex

Notes:
  - The tool takes the first 20 rows of an uploaded list and drops the
    rest without an error, so a group over the cap is refused here.
  - The service admits two concurrent jobs; the runner caps at that
    rather than discovering the throttle mid-sweep.
  - A job is point-source or elliptical as a whole. Under the sersic
    model a point-like member is carried at a sub-threshold radius,
    which the tool reads as a point source.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from .catalogs.legacy import LEGACY_DR_DEFAULT, shape_from_tractor
from .measure import recipe
from .results import STATUS_OK
from .spherex import (GROUP_SOURCES_MAX, GROUP_ROLES, ROLE_ANCILLARY,
                      ROLE_SCIENCE, SPHEREX_POINT_SOURCE_REFF, GroupSource,
                      Sersic, check_shape, fetch_group, group_config_payload,
                      group_extraction_tag, group_is_complete,
                      sersic_from_shape)

# ------------------------------------
# Constants
# ------------------------------------

SCHEMA_VERSIONS = (1,)

# Members closer than this share a job. Roughly seven SPHEREx pixels:
# far enough out that the PSF wings have fallen away, close enough that
# grouping stays local.
DEFAULT_BLEND_RADIUS_AS = 45.0

# A neighbor is worth a seat when it is at least this fraction of the
# science member's own flux. Below it the neighbor cannot move the
# science flux by more than the noise, and it costs one of twenty seats.
DEFAULT_COMPANION_FLUX_RATIO = 0.1

# Radius given to a point-like member of an elliptical job. The tool
# reads any radius below SPHEREX_POINT_SOURCE_REFF as a point source, so
# this is how a PSF neighbor rides in a sersic group -- one job is one
# model, and the tool has no per-row switch.
POINT_LIKE_REFF_AS = 0.5

MODELS = ("psf", "sersic")

DEFAULT_KEYS = ("model", "bkg_size", "mjd_range", "poll", "timeout",
                "tol_arcsec")
CONFIG_KEYS = ("schema_version", "description", "data_root", "group_dir",
               "defaults", "groups", "planning_notes", "path")
CONFIG_REQUIRED = ("schema_version", "groups")
GROUP_KEYS = ("id", "members", "note")
MEMBER_KEYS = ("label", "name", "role", "dir", "ra_deg", "dec_deg", "shape",
               "shape_origin", "note")
MEMBER_REQUIRED = ("label", "ra_deg", "dec_deg")
SHAPE_KEYS = ("n", "axis_ratio", "pa_deg", "reff_arcsec")

MEMBER_COLUMNS = ("group", "label", "role", "ra_deg", "dec_deg", "n",
                  "axis_ratio", "pa_deg", "reff_arcsec", "shape_origin",
                  "dir")
REPORT_COLUMNS = ("group", "status", "tag", "n_members", "n_science",
                  "seconds", "job_url", "members", "message")

STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

# The service admits two concurrent spectrophotometry jobs.
MAX_WORKERS = 2


# ------------------------------------
# Config: validation
# ------------------------------------

def _require_keys(where: str, mapping: dict, *, allowed, required=()) -> None:
    """Refuse an unknown key and a missing required one, by name.

    A typo'd optional key silently takes a default, which for an
    extraction is a wrong measurement rather than a missing one.
    """
    if not isinstance(mapping, dict):
        raise ValueError(f"{where}: expected a mapping, got "
                         f"{type(mapping).__name__}")
    unknown = sorted(set(mapping) - set(allowed))
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed "
                         f"{sorted(allowed)}")
    missing = [k for k in required if mapping.get(k) is None]
    if missing:
        raise ValueError(f"{where}: missing required key(s) {missing}")


def _shape_from_config(where: str, raw: dict) -> Sersic:
    _require_keys(where, raw, allowed=SHAPE_KEYS, required=SHAPE_KEYS)
    try:
        model = Sersic(n=float(raw["n"]), axis_ratio=float(raw["axis_ratio"]),
                       pa_deg=float(raw["pa_deg"]),
                       reff_arcsec=float(raw["reff_arcsec"]))
    except (TypeError, ValueError) as e:
        raise ValueError(f"{where}: shape values must be numbers ({e})") from e
    complaints = check_shape(model, where=where)
    if complaints:
        raise ValueError("; ".join(complaints))
    return model


def load_config(path: str | Path) -> dict:
    """Read and validate a group config.

    Every check that can be made without the network is made here: a
    job costs tens of minutes and writes a product that is never
    overwritten, so a config error must surface before submission.

    Parameters
    ----------
    path : str or Path
        Config JSON.

    Returns
    -------
    config : dict
        The config with defaults filled in and 'path' / 'data_root'
        resolved to absolute directories.
    """
    path = Path(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    _require_keys(str(path), config, allowed=CONFIG_KEYS,
                  required=CONFIG_REQUIRED)
    if config["schema_version"] not in SCHEMA_VERSIONS:
        raise ValueError(f"{path}: schema_version "
                         f"{config['schema_version']!r} is not one of "
                         f"{list(SCHEMA_VERSIONS)}")

    defaults = dict(config.get("defaults") or {})
    _require_keys(f"{path}.defaults", defaults, allowed=DEFAULT_KEYS)
    defaults.setdefault("model", "sersic")
    defaults.setdefault("bkg_size", 15.0)
    defaults.setdefault("mjd_range", None)
    defaults.setdefault("poll", 5.0)
    defaults.setdefault("timeout", 10800.0)
    defaults.setdefault("tol_arcsec", 1.0)
    if defaults["model"] not in MODELS:
        raise ValueError(f"{path}.defaults.model: {defaults['model']!r} is "
                         f"not one of {list(MODELS)}")
    config["defaults"] = defaults

    groups = config["groups"]
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"{path}: groups must be a non-empty list")
    ids = [g.get("id") for g in groups]
    duplicated = sorted({i for i in ids if ids.count(i) > 1})
    if duplicated:
        raise ValueError(f"{path}: duplicate group id(s) {duplicated}")

    data_root = (path.parent / (config.get("data_root") or ".")).resolve()
    config["data_root"] = str(data_root)
    config["path"] = str(path)
    if config.get("group_dir"):
        config["group_dir"] = str(
            (data_root / config["group_dir"]).resolve())

    for group in groups:
        _require_keys(f"{path}:{group.get('id')}", group, allowed=GROUP_KEYS,
                      required=("id", "members"))
        # Builds every GroupSource, so a config that cannot be executed
        # fails here rather than after the first job has run.
        sources_for(group, config)
    return config


def sources_for(group: dict, config: dict) -> list[GroupSource]:
    """Build one group's GroupSource list from the config.

    Raises
    ------
    ValueError
        On a malformed member, an unknown role, a science member with no
        directory, a group over the seat cap, or a shape outside the
        tool's stated bounds.
    """
    model_name = config["defaults"]["model"]
    data_root = Path(config["data_root"])
    members = group.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError(f"group {group.get('id')!r}: members must be a "
                         f"non-empty list")
    if len(members) > GROUP_SOURCES_MAX:
        raise ValueError(
            f"group {group.get('id')!r}: {len(members)} members exceeds the "
            f"tool's {GROUP_SOURCES_MAX} per job; the extra rows would be "
            f"dropped without an error")

    sources = []
    for raw in members:
        where = f"group {group.get('id')!r} member {raw.get('label')!r}"
        _require_keys(where, raw, allowed=MEMBER_KEYS,
                      required=MEMBER_REQUIRED)
        role = raw.get("role") or ROLE_SCIENCE
        if role not in GROUP_ROLES:
            raise ValueError(f"{where}: role {role!r} is not one of "
                             f"{list(GROUP_ROLES)}")
        out_dir = raw.get("dir")
        if role == ROLE_SCIENCE and not out_dir:
            raise ValueError(f"{where}: a science member needs a dir -- its "
                             f"table has nowhere to land")
        if out_dir and not Path(out_dir).is_absolute():
            out_dir = str(data_root / out_dir)

        shape = raw.get("shape")
        if model_name == "psf":
            if shape:
                raise ValueError(f"{where}: the job's model is psf but this "
                                 f"member carries a shape; it would be "
                                 f"silently ignored")
            model = None
        else:
            if not shape:
                raise ValueError(f"{where}: the job's model is sersic, so "
                                 f"every member needs a shape")
            model = _shape_from_config(f"{where}.shape", shape)
        sources.append(GroupSource(
            label=str(raw["label"]), name=raw.get("name"), role=role,
            ra_deg=float(raw["ra_deg"]), dec_deg=float(raw["dec_deg"]),
            model=model, out_dir=out_dir,
            shape_origin=raw.get("shape_origin")))

    labels = [s.label for s in sources]
    duplicated = sorted({x for x in labels if labels.count(x) > 1})
    if duplicated:
        raise ValueError(f"group {group.get('id')!r}: duplicate member "
                         f"label(s) {duplicated}")
    if not any(s.is_science for s in sources):
        raise ValueError(f"group {group.get('id')!r}: no science member; "
                         f"the job would write nothing")
    return sources


# ------------------------------------
# Planning
# ------------------------------------

def link_groups(coords: SkyCoord, radius_arcsec: float) -> list[list[int]]:
    """Single-linkage clusters of positions within a radius.

    Single linkage, not a fixed radius about each target: blending is
    transitive at the PSF scale, so A near B near C belong in one job
    even when A and C are apart. Returns index lists in first-member
    order.
    """
    n = len(coords)
    if n == 0:
        return []
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    left, right, _, _ = coords.search_around_sky(coords,
                                                 radius_arcsec * u.arcsec)
    for a, b in zip(np.asarray(left), np.asarray(right)):
        ra_, rb = find(int(a)), find(int(b))
        if ra_ != rb:
            parent[rb] = ra_

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return [members for _, members in
            sorted(clusters.items(), key=lambda kv: min(kv[1]))]


def _shape_from_row(row) -> tuple[Sersic, str] | None:
    """A Sersic and its origin from one Tractor catalog row, or None."""
    shape_sky = shape_from_tractor(row["type"], row["sersic"], row["shape_r"],
                                   row["shape_e1"], row["shape_e2"])
    if shape_sky is None:
        return None
    origin = (f"ls tractor {str(row['type']).strip().upper()} "
              f"(scene catalog)")
    return sersic_from_shape(shape_sky), origin


def _point_like(row) -> tuple[Sersic, str]:
    """A point-like neighbor as an elliptical job can carry it."""
    return (Sersic(n=1.0, axis_ratio=1.0, pa_deg=0.0,
                   reff_arcsec=POINT_LIKE_REFF_AS),
            f"ls tractor {str(row['type']).strip().upper()}; carried at "
            f"reff {POINT_LIKE_REFF_AS:g}\" (below the tool's "
            f"{SPHEREX_POINT_SOURCE_REFF:g}\" point-source threshold)")


def plan_groups(
        targets: list[dict],
        *,
        blend_radius_as: float = DEFAULT_BLEND_RADIUS_AS,
        model: str = "sersic",
        companions: bool = True,
        companion_flux_ratio: float = DEFAULT_COMPANION_FLUX_RATIO,
        mjd_range: list[float] | None = None,
        bkg_size: float = 15.0,
        legacy_dr: str = LEGACY_DR_DEFAULT,
        cutout_arcsec: float = 120.0,
        description: str | None = None,
        progress: Callable[[str], None] = print,
) -> dict:
    """Group targets into joint jobs and resolve every member's shape.

    Targets within blend_radius_as of one another share a job, since
    each is in the others' PSF wings. Optionally, catalog neighbors that
    are not themselves targets ride along as ancillary members, so their
    light is fit rather than left in a science member's aperture.

    Shapes come from each field's cached Tractor scene catalog -- the
    same cache the measurement engine reads -- so planning a measured
    tree needs no network.

    Parameters
    ----------
    targets : list of dict
        From schedule.read_targets: name, label, ra_deg, dec_deg, dir.
    blend_radius_as : float
        Link distance. [default: 45]
    model : str
        'sersic' or 'psf', for the whole plan. [default: 'sersic']
    companions : bool
        Seat non-target catalog neighbors as ancillary members.
        [default: True]
    companion_flux_ratio : float
        A neighbor rides along when its r flux is at least this fraction
        of the science member's. [default: 0.1]
    mjd_range : list of float, optional
        Visit window recorded in the config's defaults.
    bkg_size : float
        Background region, pixels, recorded in the defaults. [default: 15]
    legacy_dr : str
        Data release for the scene catalogs. [default: LEGACY_DR_DEFAULT]
    cutout_arcsec : float
        Stamp width the scene cache was built at. [default: 120]
    description : str, optional
        Free text recorded in the config.
    progress : callable
        Progress sink. [default: print]

    Returns
    -------
    config : dict
        A group config, ready to write.
    """
    from .schedule import fetch_catalog

    if model not in MODELS:
        raise ValueError(f"model {model!r} is not one of {list(MODELS)}")
    coords = SkyCoord([t["ra_deg"] for t in targets] * u.deg,
                      [t["dec_deg"] for t in targets] * u.deg)
    clusters = link_groups(coords, blend_radius_as)
    progress(f"{len(targets)} target(s) -> {len(clusters)} group(s) at "
             f"{blend_radius_as:g}\" linking; largest "
             f"{max((len(c) for c in clusters), default=0)}")

    groups, notes = [], []
    for order, indices in enumerate(clusters, start=1):
        group_id = f"g{order:03d}"
        members, seated = [], []
        for i in indices:
            target = targets[i]
            member = {"label": target["label"], "name": target.get("name"),
                      "role": ROLE_SCIENCE, "dir": target.get("dir"),
                      "ra_deg": float(target["ra_deg"]),
                      "dec_deg": float(target["dec_deg"])}
            if model == "sersic":
                catalog = fetch_catalog(coords[i], target["dir"],
                                        cutout_arcsec=cutout_arcsec,
                                        legacy_dr=legacy_dr)
                resolved = _resolve_member_shape(coords[i], catalog)
                if resolved is None:
                    notes.append(f"{group_id}/{target['label']}: no usable "
                                 f"Tractor shape; supply one by hand or "
                                 f"plan with --model psf")
                    progress(f"  WARNING {target['label']}: no usable shape")
                else:
                    member["shape"], member["shape_origin"] = resolved
                seated.append((coords[i], catalog, target["label"]))
            members.append(member)

        if companions and model == "sersic" and seated:
            extra, dropped = _companions_for(
                seated, members, blend_radius_as=blend_radius_as,
                flux_ratio=companion_flux_ratio)
            members.extend(extra)
            if dropped:
                notes.append(f"{group_id}: {dropped} companion(s) above the "
                             f"flux floor did not fit in the "
                             f"{GROUP_SOURCES_MAX}-seat job and were dropped")
                progress(f"  NOTE {group_id}: dropped {dropped} companion(s) "
                         f"at the seat cap")
        groups.append({"id": group_id, "members": members})
        progress(f"  {group_id}: {len(indices)} science + "
                 f"{len(members) - len(indices)} ancillary")

    config = {
        "schema_version": 1,
        "description": description or
                       f"Joint SPHEREx extractions, {blend_radius_as:g}\" "
                       f"linking, {model} model",
        "data_root": ".",
        "group_dir": "spherex_groups",
        "defaults": {"model": model, "bkg_size": bkg_size,
                     "mjd_range": list(mjd_range) if mjd_range else None},
        "groups": groups,
    }
    if notes:
        config["planning_notes"] = notes
    return config


def _resolve_member_shape(coord, catalog):
    """A science member's own shape from its field's scene catalog."""
    if catalog is None or not len(catalog):
        return None
    rows = SkyCoord(catalog["ra"].to_numpy() * u.deg,
                    catalog["dec"].to_numpy() * u.deg)
    separations = coord.separation(rows).arcsec
    nearest = int(np.argmin(separations))
    if separations[nearest] > recipe.TARGET_MATCH_AS:
        return None
    resolved = _shape_from_row(catalog.iloc[nearest])
    if resolved is None:
        return None
    model, origin = resolved
    if check_shape(model):
        return None
    return ({"n": model.n, "axis_ratio": model.axis_ratio,
             "pa_deg": model.pa_deg, "reff_arcsec": model.reff_arcsec},
            f"{origin}, sep {separations[nearest]:.2f}\"")


def _companions_for(seated, members, *, blend_radius_as, flux_ratio):
    """Catalog neighbors worth a seat, brightest first.

    A neighbor already seated as a science member is skipped: the point
    is to seat light that nothing else accounts for.
    """
    claimed = SkyCoord([m["ra_deg"] for m in members] * u.deg,
                       [m["dec_deg"] for m in members] * u.deg)
    candidates: dict[tuple, dict] = {}
    for coord, catalog, _label in seated:
        if catalog is None or not len(catalog):
            continue
        rows = SkyCoord(catalog["ra"].to_numpy() * u.deg,
                        catalog["dec"].to_numpy() * u.deg)
        separations = coord.separation(rows).arcsec
        own = float(catalog.iloc[int(np.argmin(separations))].get("flux_r")
                    or 0.0)
        floor = abs(own) * flux_ratio
        for i in range(len(catalog)):
            if separations[i] > blend_radius_as:
                continue
            row = catalog.iloc[i]
            flux = float(row.get("flux_r") or 0.0)
            if flux < floor:
                continue
            here = SkyCoord(float(row["ra"]) * u.deg, float(row["dec"]) * u.deg)
            if here.separation(claimed).arcsec.min() <= recipe.TARGET_MATCH_AS:
                continue
            key = (round(float(row["ra"]), 6), round(float(row["dec"]), 6))
            if key in candidates and candidates[key]["_flux"] >= flux:
                continue
            resolved = _shape_from_row(row)
            model, origin = resolved if resolved else _point_like(row)
            if check_shape(model):
                continue
            candidates[key] = {
                "label": f"J{key[0]:.5f}{key[1]:+.5f}",
                "role": ROLE_ANCILLARY,
                "ra_deg": key[0], "dec_deg": key[1],
                "shape": {"n": model.n, "axis_ratio": model.axis_ratio,
                          "pa_deg": model.pa_deg,
                          "reff_arcsec": model.reff_arcsec},
                "shape_origin": origin, "_flux": flux,
            }

    ranked = sorted(candidates.values(), key=lambda c: -c["_flux"])
    seats = GROUP_SOURCES_MAX - len(members)
    taken = ranked[:max(seats, 0)]
    for candidate in taken:
        candidate.pop("_flux")
    return taken, len(ranked) - len(taken)


def write_config(config: dict, path: str | Path) -> tuple[Path, Path]:
    """Write the config and its flat member listing.

    The CSV is the artifact to read: a config is reviewed member by
    member, and JSON nesting hides exactly the thing under review.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MEMBER_COLUMNS))
        writer.writeheader()
        for group in config["groups"]:
            for member in group["members"]:
                shape = member.get("shape") or {}
                writer.writerow({
                    "group": group["id"], "label": member.get("label"),
                    "role": member.get("role") or ROLE_SCIENCE,
                    "ra_deg": member.get("ra_deg"),
                    "dec_deg": member.get("dec_deg"),
                    "n": shape.get("n"),
                    "axis_ratio": shape.get("axis_ratio"),
                    "pa_deg": shape.get("pa_deg"),
                    "reff_arcsec": shape.get("reff_arcsec"),
                    "shape_origin": member.get("shape_origin"),
                    "dir": member.get("dir"),
                })
    return path, csv_path


# ------------------------------------
# Execution
# ------------------------------------

def _run_group(group: dict, config: dict, *, log_dir=None) -> dict:
    """Run one group; never raises."""
    import time

    group_id = group["id"]
    row = {column: None for column in REPORT_COLUMNS}
    row["group"] = group_id
    job_url: list[str] = []

    def body():
        defaults = config["defaults"]
        sources = sources_for(group, config)
        row["n_members"] = len(sources)
        row["n_science"] = sum(1 for s in sources if s.is_science)
        row["members"] = " ".join(
            f"{s.label}:{'S' if s.is_science else 'A'}" for s in sources)
        mjd = defaults["mjd_range"]
        return fetch_group(
            sources, group_id=group_id,
            group_dir=config.get("group_dir"),
            bkg_region_size=defaults["bkg_size"],
            mjd_range=tuple(mjd) if mjd else None,
            poll=defaults["poll"], timeout=defaults["timeout"],
            tol_arcsec=defaults["tol_arcsec"],
            on_job_url=job_url.append)

    started = time.monotonic()
    try:
        if log_dir is not None:
            path = Path(log_dir) / f"{group_id}.log"
            # Line-buffered: a block-buffered log is empty until the job
            # ends, which for a job measured in tens of minutes reads as
            # a hang to whoever is watching.
            with path.open("w", encoding="utf-8", buffering=1) as handle:
                with redirect_stdout(handle):
                    result = body()
        else:
            result = body()
    except Exception as err:
        row.update(status=STATUS_FAILED, seconds=round(
            time.monotonic() - started, 1),
            job_url=job_url[0] if job_url else None,
            message=f"{type(err).__name__}: {err}")
        return row

    row.update(status=result.status, seconds=round(
        time.monotonic() - started, 1),
        tag=result.meta.get("tag"), job_url=job_url[0] if job_url else None,
        message=result.message)
    return row


def _write_report(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["group"]))


def run_config(
        config: dict,
        *,
        report_path: str | Path,
        workers: int = 1,
        resume: bool = True,
        only: list[str] | None = None,
        log_dir: str | Path | None = None,
        progress: Callable[[str], None] = print,
) -> dict:
    """Run every group in a config.

    Parameters
    ----------
    config : dict
        From load_config.
    report_path : str or Path
        One row per group lands here, rewritten as each finishes.
    workers : int
        Concurrent jobs, capped at MAX_WORKERS. [default: 1]
    resume : bool
        Skip a group whose science members already carry this exact
        extraction. [default: True]
    only : list of str, optional
        Run just these group ids.
    log_dir : str or Path, optional
        Per-group logs land here.
    progress : callable
        Progress sink. [default: print]

    Returns
    -------
    summary : dict
        Per-status counts and the report path.
    """
    if workers > MAX_WORKERS:
        raise ValueError(
            f"--workers {workers} would submit {workers} concurrent IRSA "
            f"extractions; the service admits {MAX_WORKERS}. Each job runs "
            f"for tens of minutes, so discovering the throttle mid-sweep "
            f"is expensive.")
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    if config.get("group_dir"):
        Path(config["group_dir"]).mkdir(parents=True, exist_ok=True)

    groups = [g for g in config["groups"]
              if only is None or g["id"] in set(only)]
    if only:
        unknown = sorted(set(only) - {g["id"] for g in config["groups"]})
        if unknown:
            raise ValueError(f"no such group(s) in this config: {unknown}")

    rows: list[dict] = []
    pending = []
    for group in groups:
        sources = sources_for(group, config)
        defaults = config["defaults"]
        mjd = tuple(defaults["mjd_range"]) if defaults["mjd_range"] else None
        payload = group_config_payload(sources, defaults["bkg_size"], mjd)
        tag = group_extraction_tag(sources, defaults["bkg_size"], mjd)
        complete, missing = group_is_complete(sources, payload, tag)
        if resume and complete:
            rows.append({**{c: None for c in REPORT_COLUMNS},
                         "group": group["id"], "status": STATUS_SKIPPED,
                         "tag": tag, "n_members": len(sources),
                         "n_science": sum(1 for s in sources if s.is_science),
                         "message": "already on disk"})
            continue
        pending.append(group)

    progress(f"{len(groups)} group(s): {len(pending)} to run, "
             f"{len(groups) - len(pending)} already on disk; "
             f"{min(workers, max(len(pending), 1))} worker(s)")
    _write_report(report_path, rows)

    def record(row):
        rows.append(row)
        progress(f"  {row['status']:7s} {row['group']} "
                 f"{row.get('message') or ''}")
        _write_report(report_path, rows)

    if pending:
        # Threads, not processes: a job is a long poll, so the work is
        # waiting on IRSA rather than on this machine.
        with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
            futures = [pool.submit(_run_group, group, config,
                                   log_dir=log_dir) for group in pending]
            for future in as_completed(futures):
                record(future.result())

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    progress(f"{counts.get(STATUS_OK, 0)} ok, "
             f"{counts.get(STATUS_FAILED, 0)} failed, "
             f"{counts.get(STATUS_SKIPPED, 0)} skipped -> {report_path}")
    return {"counts": counts, "report": str(report_path),
            "n_groups": len(groups)}
