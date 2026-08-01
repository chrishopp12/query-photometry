"""
schedule.py

Campaign Scheduling
---------------------------------------------------------

The planning pass of a multi-target sweep. Reads a target list, fetches
each field's scene catalog (cache-first, the same cache the measurement
reads), and derives the order the measuring passes follow: which targets
harvest a shape worth propagating, which fields can share a registry
entry, and the sequence within each set of fields that can.

Grouping exists because the registry is cross-field mutable state. A
field WRITES within its gate reach and READS within its scene cone, so
two fields can share an entry only when their separation is under the
sum of the two. Fields further apart than that touch disjoint entries
and may be measured concurrently against separate registry files, merged
afterward. Both radii derive from the stamp size rather than being
assumed, and the link distance carries a margin on top: a margin costs
concurrency, a shortfall costs reproducibility.

The order within a set follows the vantage grade. A target whose own
catalog row qualifies for a shape gate harvests its shape at target
vantage from its own centered view, and that record outranks any
neighbor-vantage copy; measuring such a target before its consumers
means they freeze the better record. A target that does not gate
harvests nothing about itself and its position in the order is free.

Data products:
  <plan>.json   census, groups, and pass assignment for every target
  <plan>.csv    the same rows, flat, for inspection

Requirements:
  - numpy, pandas, astropy

Notes:
  - Gate membership is purely radial. The gate reach is the stamp half
    width less an edge margin, always inside the stamp's inscribed
    circle, so no stamp geometry enters the census.
  - A target with no catalog row within the identity radius cannot
    gate; it is counted and reported rather than silently passed over.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord

from .catalogs.legacy import LEGACY_DR_DEFAULT, query_scene
from .measure import recipe
from .measure.components import gated_row
from .resolve import sanitize_label

# ------------------------------------
# Constants
# ------------------------------------

# Added to the write-plus-read span before two fields are called
# independent. The engine models a cone; light does not stop at one.
DEFAULT_LINK_MARGIN_AS = 120.0

TARGET_COLUMN_ALIASES = ("name", "target")
POSITION_COLUMNS = (("ra_deg", "dec_deg"), ("ra", "dec"))
# Every column this reader consumes. A target list may serve other
# tools too, so anything else is reported rather than refused -- but a
# typo in an optional column silently takes a default, which is what
# naming the ignored set makes visible.
TARGET_COLUMNS = ("name", "target", "ra_deg", "dec_deg", "ra", "dec",
                  "dir", "label", "priority")

PASS_HARVEST = "harvest"
PASS_PARALLEL = "parallel"

PLAN_COLUMNS = ("name", "label", "ra_deg", "dec_deg", "dir", "pass",
                "group", "order", "gates", "matched", "n_consumers",
                "n_gated", "priority")


# ------------------------------------
# Geometry
# ------------------------------------

def scene_radius_arcsec(cutout_arcsec: float) -> float:
    """The scene cone: what a field renders, and so what it can read."""
    return max(recipe.QUERY_RADIUS_AS,
               cutout_arcsec / 2.0 * np.sqrt(2.0) + recipe.QUERY_PAD_AS)


def gate_reach_arcsec(cutout_arcsec: float) -> float:
    """The gate reach: what a field solves, and so what it can write."""
    return cutout_arcsec / 2.0 - recipe.GATE_EDGE_MARGIN_AS


def link_radius_arcsec(cutout_arcsec: float,
                       margin_arcsec: float = DEFAULT_LINK_MARGIN_AS) -> float:
    """Separation under which two fields can share a registry entry.

    The floor is the write reach plus the read reach: under it a record
    can be harvested by one field and consumed by another the grouping
    calls independent, so a negative margin is refused rather than
    left for the audit to discover.

    Parameters
    ----------
    cutout_arcsec : float
        Stamp width; both component radii derive from it.
    margin_arcsec : float
        Padding past the modeled span; must not be negative.
        [default: 120.0]
    """
    if margin_arcsec < 0:
        raise ValueError(f"link margin {margin_arcsec} is negative; the link "
                         f"distance may not fall below the write-plus-read "
                         f"span")
    return (gate_reach_arcsec(cutout_arcsec)
            + scene_radius_arcsec(cutout_arcsec) + margin_arcsec)


def _pairs_within(coords: SkyCoord, radius_arcsec: float) -> list[tuple]:
    """Index pairs (i < j) separated by less than a radius."""
    if len(coords) < 2:
        return []
    left, right, _, _ = coords.search_around_sky(coords,
                                                 radius_arcsec * u.arcsec)
    return sorted({(int(a), int(b)) for a, b in zip(left, right) if a < b})


def connected_groups(coords: SkyCoord, radius_arcsec: float) -> list[int]:
    """Group id per position; positions closer than the radius share one.

    Returns
    -------
    labels : list of int
        Group index per input, numbered by first appearance so the
        labeling does not depend on iteration order.
    """
    n = len(coords)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, j in _pairs_within(coords, radius_arcsec):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[max(root_i, root_j)] = min(root_i, root_j)

    labels, seen = [], {}
    for i in range(n):
        root = find(i)
        if root not in seen:
            seen[root] = len(seen)
        labels.append(seen[root])
    return labels


def consumer_counts(coords: SkyCoord, scene_arcsec: float) -> list[int]:
    """How many OTHER fields would read an entry harvested at each position."""
    counts = [0] * len(coords)
    for i, j in _pairs_within(coords, scene_arcsec):
        counts[i] += 1
        counts[j] += 1
    return counts


# ------------------------------------
# The gate census
# ------------------------------------

def census_from_catalog(coord: SkyCoord, cat: pd.DataFrame, *,
                        cutout_arcsec: float) -> dict:
    """Gate facts for one field, from its scene catalog alone.

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    cat : pd.DataFrame
        The field's Tractor scene catalog.
    cutout_arcsec : float
        Stamp width; sets the gate reach.

    Returns
    -------
    census : dict
        matched (a catalog row claims the target), gates (that row
        qualifies for a shape gate), n_gated (gated neighbors in reach),
        and entries -- the positions this field would harvest, which is
        every gated neighbor plus the target itself when it gates.
    """
    if not len(cat):
        return {"matched": False, "gates": False, "n_gated": 0,
                "entries": []}

    rows = SkyCoord(cat["ra"].to_numpy() * u.deg,
                    cat["dec"].to_numpy() * u.deg)
    separations = coord.separation(rows).arcsec

    nearest = int(np.argmin(separations))
    matched = bool(separations[nearest] < recipe.TARGET_MATCH_AS)
    # The target's own row is judged at a non-self distance: the question
    # is whether another field would seat it, not whether this one does.
    gates = bool(matched and gated_row(cat.iloc[nearest],
                                       recipe.TARGET_MATCH_AS + 1.0))

    reach = gate_reach_arcsec(cutout_arcsec)
    entries = [[float(cat.iloc[i]["ra"]), float(cat.iloc[i]["dec"])]
               for i in range(len(cat))
               if separations[i] < reach
               and gated_row(cat.iloc[i], float(separations[i]))]
    n_gated = len(entries)
    if gates:
        entries.append([float(coord.ra.deg), float(coord.dec.deg)])
    return {"matched": matched, "gates": gates, "n_gated": n_gated,
            "entries": entries}


def fetch_catalog(coord: SkyCoord, target_dir: str | Path, *,
                  cutout_arcsec: float,
                  legacy_dr: str = LEGACY_DR_DEFAULT) -> pd.DataFrame:
    """One field's scene catalog, into the cache the measurement reads."""
    radius = scene_radius_arcsec(cutout_arcsec)
    suffix = ("" if radius == recipe.QUERY_RADIUS_AS
              else f"_r{int(round(radius))}")
    cache_dir = Path(target_dir) / "Photometry" / "scene"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return query_scene(coord, radius, dr=legacy_dr,
                       min_flux_nmgy=recipe.TRACTOR_MIN_NMGY,
                       cache_path=cache_dir
                       / f"tractor_scene_{legacy_dr}{suffix}.csv")


# ------------------------------------
# The target list
# ------------------------------------

def read_targets(path: str | Path,
                 out_root: str | Path | None = None) -> list[dict]:
    """Targets from a CSV, in file order.

    Parameters
    ----------
    path : str or Path
        CSV with a name column and a position pair; a ``dir`` column
        names each target's directory, and ``priority`` overrides the
        derived order (higher runs earlier).
    out_root : str or Path, optional
        Root that ``dir`` resolves against, matching how sed_fitting
        resolves it against a campaign's data_root -- so one shared
        catalog drives both tools. An absolute ``dir`` wins regardless.
        [default: None]

    Returns
    -------
    targets : list of dict
        name, label, ra_deg, dec_deg, dir, priority. A ``label`` column
        overrides the sanitized name; it is the product filename stem,
        so an existing tree's stems must be named rather than guessed.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) if rows else []

    name_col = next((c for c in TARGET_COLUMN_ALIASES if c in fields), None)
    if name_col is None:
        raise ValueError(f"{path}: no {' or '.join(TARGET_COLUMN_ALIASES)} "
                         f"column (found: {fields})")
    position = next((pair for pair in POSITION_COLUMNS
                     if pair[0] in fields and pair[1] in fields), None)
    if position is None:
        raise ValueError(f"{path}: no ra_deg/dec_deg column pair "
                         f"(found: {fields})")
    used = [c for c in fields if c in TARGET_COLUMNS]
    ignored = [c for c in fields if c not in TARGET_COLUMNS]
    print(f"{path.name}: {len(rows)} rows; using {', '.join(used)}"
          + (f"; ignoring {', '.join(ignored)}" if ignored else ""))

    targets, seen = [], set()
    for row in rows:
        name = (row.get(name_col) or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        # dir is RELATIVE to out_root, the same way sed_fitting resolves it
        # against a campaign's data_root -- so one shared catalog serves
        # both tools. An absolute dir still wins (pathlib absorbs it), and
        # without an out_root the value is used as given.
        relative = (row.get("dir") or "").strip() or name
        directory = (str(Path(out_root) / relative) if out_root
                     else relative)
        priority = (row.get("priority") or "").strip()
        targets.append({"name": name,
                        "label": (row.get("label") or "").strip()
                                 or sanitize_label(name),
                        "ra_deg": float(row[position[0]]),
                        "dec_deg": float(row[position[1]]),
                        "dir": directory,
                        "priority": float(priority) if priority else None})
    if not targets:
        raise ValueError(f"{path}: no target rows")
    return targets


# ------------------------------------
# The plan
# ------------------------------------

def audit_groups(rows: list[dict], *, scene_arcsec: float) -> list[dict]:
    """Records one group would write that another group's field would read.

    The grouping claims that running groups concurrently equals running
    every field in one sequence. It does not claim the groups are
    physically separate -- light from off the stamp reaches the stamp at
    any separation, and no radius changes that. What it does claim is
    that no record crosses a group boundary, and that is checkable: a
    harvested position inside another group's scene cone is a record
    that field WOULD have consumed had the passes been sequential, and
    did not. Every hit is a reason to raise the margin and re-plan.

    Only harvest-pass fields are consumers here. The parallel pass reads
    the merged registry after the freeze, so it sees every record
    regardless of which group wrote it.

    Parameters
    ----------
    rows : list of dict
        Plan rows carrying pass, group, name, and entries.
    scene_arcsec : float
        The consuming reach.

    Returns
    -------
    violations : list of dict
        entry position, the group that writes it, and the field in
        another group that would read it.
    """
    harvest = [r for r in rows if r["pass"] == PASS_HARVEST]
    entries = [{"ra_deg": position[0], "dec_deg": position[1],
                "group": r["group"], "owner": r["name"]}
               for r in harvest for position in r.get("entries") or []]
    fields = [{"ra_deg": r["ra_deg"], "dec_deg": r["dec_deg"],
               "group": r["group"], "name": r["name"]} for r in harvest]
    return audit_entries(entries, fields, scene_arcsec=scene_arcsec)


def audit_entries(entries: list[dict], fields: list[dict], *,
                  scene_arcsec: float) -> list[dict]:
    """Records inside the consuming reach of a field in another group.

    The same check on either side of the harvest pass: predicted records
    before it runs, actual registry entries after. Positions carry the
    group that wrote them; a hit is a record one group would have handed
    another had the passes been sequential.

    Parameters
    ----------
    entries : list of dict
        ra_deg, dec_deg, group, owner.
    fields : list of dict
        Consuming positions: ra_deg, dec_deg, group, name.
    scene_arcsec : float
        The consuming reach.

    Returns
    -------
    violations : list of dict
        One per (record, out-of-group reader) pair.
    """
    if not entries or len(fields) < 2:
        return []

    entry_coords = SkyCoord([e["ra_deg"] for e in entries] * u.deg,
                            [e["dec_deg"] for e in entries] * u.deg)
    field_coords = SkyCoord([f["ra_deg"] for f in fields] * u.deg,
                            [f["dec_deg"] for f in fields] * u.deg)

    left, right, _, _ = field_coords.search_around_sky(entry_coords,
                                                       scene_arcsec * u.arcsec)
    violations = []
    for entry_index, field_index in zip(left, right):
        entry = entries[int(entry_index)]
        reader = fields[int(field_index)]
        if entry["group"] == reader["group"]:
            continue
        violations.append({"ra_deg": entry["ra_deg"],
                           "dec_deg": entry["dec_deg"],
                           "written_by": entry["owner"],
                           "written_group": entry["group"],
                           "read_by": reader["name"],
                           "read_group": reader["group"]})
    return sorted(violations, key=lambda v: (str(v["written_by"]),
                                             str(v["read_by"])))


def build_plan(targets: list[dict], census: list[dict], *,
               cutout_arcsec: float,
               margin_arcsec: float = DEFAULT_LINK_MARGIN_AS) -> dict:
    """Assign every target a pass, a group, and an order within it.

    The harvest pass holds the targets that both gate and have a
    consumer: only they write a record another field will read, so only
    they need a sequence. Everything else goes to the parallel pass,
    which reads a frozen registry and writes nothing.

    Parameters
    ----------
    targets : list of dict
        From read_targets.
    census : list of dict
        From census_from_catalog, aligned with targets.
    cutout_arcsec : float
        Stamp width; sets both radii.
    margin_arcsec : float
        Link-distance padding. [default: 120.0]

    Returns
    -------
    plan : dict
        radii, counts, and one row per target.
    """
    if len(targets) != len(census):
        raise ValueError(f"{len(targets)} targets but {len(census)} census "
                         f"entries")

    coords = SkyCoord([t["ra_deg"] for t in targets] * u.deg,
                      [t["dec_deg"] for t in targets] * u.deg)
    scene = scene_radius_arcsec(cutout_arcsec)
    link = link_radius_arcsec(cutout_arcsec, margin_arcsec)
    consumers = consumer_counts(coords, scene)

    harvest = [i for i in range(len(targets))
               if census[i]["gates"] and consumers[i] > 0]
    groups = dict(zip(harvest, connected_groups(coords[harvest], link))) \
        if harvest else {}

    rows = []
    for i, target in enumerate(targets):
        in_harvest = i in groups
        rows.append({
            "name": target["name"], "label": target["label"],
            "ra_deg": target["ra_deg"], "dec_deg": target["dec_deg"],
            "dir": target["dir"],
            "pass": PASS_HARVEST if in_harvest else PASS_PARALLEL,
            "group": groups.get(i),
            "order": None,
            "gates": census[i]["gates"], "matched": census[i]["matched"],
            "n_consumers": consumers[i], "n_gated": census[i]["n_gated"],
            "priority": target["priority"],
            "entries": census[i].get("entries") or [],
        })

    # Within a group: the most-read record first, an explicit priority
    # ahead of the derived one, and the name as the deterministic tie.
    for group in sorted({rows[i]["group"] for i in groups}):
        members = [i for i in groups if rows[i]["group"] == group]
        members.sort(key=lambda i: (-(rows[i]["priority"] or 0.0),
                                    -rows[i]["n_consumers"],
                                    -rows[i]["n_gated"],
                                    rows[i]["name"]))
        for position, i in enumerate(members):
            rows[i]["order"] = position

    group_sizes = [sum(1 for r in rows if r["group"] == g)
                   for g in sorted({r["group"] for r in rows
                                    if r["group"] is not None})]
    violations = audit_groups(rows, scene_arcsec=scene)
    return {
        "cutout_arcsec": cutout_arcsec,
        "scene_radius_arcsec": round(scene, 2),
        "gate_reach_arcsec": round(gate_reach_arcsec(cutout_arcsec), 2),
        "link_radius_arcsec": round(link, 2),
        "link_margin_arcsec": margin_arcsec,
        "n_targets": len(rows),
        "n_harvest": len(groups),
        "n_parallel": len(rows) - len(groups),
        "n_groups": len(group_sizes),
        "largest_group": max(group_sizes) if group_sizes else 0,
        "n_unmatched": sum(1 for r in rows if not r["matched"]),
        "n_violations": len(violations),
        "violations": violations,
        "targets": rows,
    }


def write_plan(plan: dict, path: str | Path) -> tuple[Path, Path]:
    """Write the plan as JSON, and its target rows as a flat CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n",
                    encoding="utf-8")

    csv_path = path.with_suffix(".csv")
    ordered = sorted(plan["targets"],
                     key=lambda r: (r["pass"] != PASS_HARVEST,
                                    r["group"] if r["group"] is not None else -1,
                                    r["order"] if r["order"] is not None else -1,
                                    r["name"]))
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PLAN_COLUMNS))
        writer.writeheader()
        for row in ordered:
            writer.writerow({k: row.get(k) for k in PLAN_COLUMNS})
    return path, csv_path
