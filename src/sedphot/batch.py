"""
batch.py

Sweep Execution
---------------------------------------------------------

Runs a plan (schedule.py) over many targets in two passes.

The harvest pass runs its groups concurrently and each group's targets
in sequence, every group writing its OWN registry file. Two writers to
one registry therefore never exist, and the whole-file rewrite that
loses entries under concurrency cannot happen. The merge then unions
those files -- disjoint by construction, so a key in two of them is a
contradiction, not a conflict to resolve -- audits the union against the
group boundary, and freezes it.

The parallel pass runs every remaining target concurrently against the
frozen registry with updates off. It writes no shared state, so its
concurrency is bounded by what the archives will tolerate rather than by
correctness.

Data products:
  <registry-dir>/group_<n>.json   one registry per harvest group
  <registry-dir>/registry.json    the merged, frozen registry
  <report>                        one row per target
  <log-dir>/<label>.log           per-target stdout

Requirements:
  - astropy (via schedule); the measurement stack inside each worker

Notes:
  - Workers are spawned, never forked, so a worker's state is only what
    its arguments carry.
  - The plan's stamp width is asserted against the sweep's: every radius
    in the grouping derives from it, so running a different one would
    execute a grouping that describes a different geometry.
  - A group is one unit of work. Its targets run in the plan's order,
    which is what makes a shared record deterministic rather than a
    race between whichever worker reached it first.
"""

from __future__ import annotations

import csv
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable

from .measure.seats import load_registry, save_registry
from .schedule import PASS_HARVEST, PASS_PARALLEL, audit_entries

# ------------------------------------
# Constants
# ------------------------------------

DEFAULT_STOP_AFTER_FAILURES = 20
MERGED_REGISTRY_NAME = "registry.json"

REPORT_COLUMNS = ("name", "label", "pass", "group", "order", "status",
                  "stage", "seconds", "failures", "error")

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


# ------------------------------------
# Report rows
# ------------------------------------

def _row(target: dict, **fields) -> dict:
    """One report row with every column present."""
    row = {column: None for column in REPORT_COLUMNS}
    row.update({"name": target["name"], "label": target["label"],
                "pass": target["pass"], "group": target["group"],
                "order": target["order"]})
    row.update({k: v for k, v in fields.items() if k in row})
    return row


# ------------------------------------
# One target
# ------------------------------------

def measured_products(target: dict) -> tuple[Path, Path]:
    """The measured table and its sidecar for a target."""
    phot_dir = Path(target["dir"]) / "Photometry"
    return (phot_dir / f"{target['label']}_measured.csv",
            phot_dir / f"{target['label']}_measured.provenance.json")


def is_complete(target: dict) -> bool:
    """Whether a target already has a measured table with a readable sidecar."""
    table, sidecar = measured_products(target)
    if not (table.is_file() and sidecar.is_file()):
        return False
    try:
        json.loads(sidecar.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return False
    return True


def _run_target(target: dict, *, registry_path: str | None,
                registry_update: bool, options: dict) -> dict:
    """Run every stage for one target; never raises."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    from .pipeline import run_all

    if options["resume"] and is_complete(target):
        return _row(target, status=STATUS_SKIPPED, stage="resume")

    started = time.monotonic()
    coord = SkyCoord(target["ra_deg"] * u.deg, target["dec_deg"] * u.deg)
    try:
        failures = run_all(coord, target["label"], target["dir"],
                           registry_path=registry_path,
                           registry_update=registry_update,
                           target_name=target["name"],
                           **options["run"])
    except Exception as err:
        return _row(target, status=STATUS_FAILED, stage="run",
                    seconds=round(time.monotonic() - started, 1),
                    error=f"{type(err).__name__}: {err}")

    elapsed = round(time.monotonic() - started, 1)
    if failures:
        return _row(target, status=STATUS_FAILED, stage="stages",
                    seconds=elapsed,
                    failures="; ".join(f"{k}: {v}"
                                       for k, v in sorted(failures.items())))
    return _row(target, status=STATUS_OK, seconds=elapsed)


def _run_logged(target: dict, *, registry_path: str | None,
                registry_update: bool, options: dict) -> dict:
    """_run_target with the target's stdout captured to its own log."""
    log_dir = options.get("log_dir")
    if log_dir is None:
        return _run_target(target, registry_path=registry_path,
                           registry_update=registry_update, options=options)
    path = Path(log_dir) / f"{target['label']}.log"
    with path.open("w", encoding="utf-8") as handle:
        with redirect_stdout(handle):
            return _run_target(target, registry_path=registry_path,
                               registry_update=registry_update,
                               options=options)


def _run_group(targets: list[dict], registry_path: str,
               options: dict) -> list[dict]:
    """One harvest group: its targets in plan order, against one registry."""
    return [_run_logged(target, registry_path=registry_path,
                        registry_update=True, options=options)
            for target in sorted(targets, key=lambda t: t["order"])]


# ------------------------------------
# The registry merge
# ------------------------------------

def merge_registries(paths: list[Path]) -> tuple[dict, list[str]]:
    """Union the per-group registries; a shared key is a contradiction.

    Groups are built so no source lies in reach of two of them, so the
    same key appearing twice means the grouping was wrong -- reported
    rather than silently resolved by a precedence rule that would be
    inventing an answer.

    Returns
    -------
    merged : dict
        The union.
    problems : list of str
        One per key claimed by more than one group.
    """
    merged: dict = {}
    owner: dict = {}
    problems = []
    for path in sorted(paths, key=lambda p: p.name):
        for key, entry in load_registry(path).items():
            if key in merged:
                problems.append(f"{key}: written by both {owner[key]} and "
                                f"{path.name}")
                continue
            merged[key] = entry
            owner[key] = path.name
    return merged, problems


def audit_merged(registries: dict[int, dict], rows: list[dict], *,
                 scene_arcsec: float) -> list[dict]:
    """The plan-time boundary audit, re-run on the entries actually written.

    The predicted audit runs on gated catalog rows. What a solve really
    harvests can differ -- entries coalesce, patches seat sources the
    catalog does not, and a sub-margin solve is dropped -- so the same
    question is asked again of the records on disk.
    """
    entries = [{"ra_deg": entry.get("ra"), "dec_deg": entry.get("dec"),
                "group": group, "owner": key}
               for group, registry in registries.items()
               for key, entry in registry.items()
               if entry.get("ra") is not None and entry.get("dec") is not None]
    fields = [{"ra_deg": r["ra_deg"], "dec_deg": r["dec_deg"],
               "group": r["group"], "name": r["name"]}
              for r in rows if r["pass"] == PASS_HARVEST]
    return audit_entries(entries, fields, scene_arcsec=scene_arcsec)


# ------------------------------------
# The sweep
# ------------------------------------

def _write_report(path: Path, rows: list[dict]) -> None:
    """Write the report in pass, group, then plan order."""
    ordered = sorted(rows, key=lambda r: (r["pass"] != PASS_HARVEST,
                                          r["group"] if r["group"] is not None
                                          else -1,
                                          r["order"] if r["order"] is not None
                                          else -1,
                                          r["name"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REPORT_COLUMNS))
        writer.writeheader()
        writer.writerows(ordered)


def _execute(workers: int, calls: list[tuple],
             record: Callable[[list], None],
             over_budget: Callable[[], bool]) -> bool:
    """Run (function, args, kwargs) calls; True if the budget abandoned them.

    A single worker runs in THIS process rather than through a pool of
    one: the traceback survives, and a caller can substitute the stage
    it wants to exercise. Workers are otherwise spawned, so a worker
    sees only what its arguments carry.
    """
    if workers == 1:
        for function, args, kwargs in calls:
            record(function(*args, **kwargs))
            if over_budget():
                return True
        return False

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(workers, len(calls)),
                             mp_context=context) as pool:
        futures = [pool.submit(function, *args, **kwargs)
                   for function, args, kwargs in calls]
        for future in as_completed(futures):
            record(future.result())
            if over_budget():
                for pending in futures:
                    pending.cancel()
                return True
    return False


def run_sweep(
        plan: dict,
        *,
        registry_dir: str | Path,
        run_options: dict,
        which_pass: str = "all",
        workers: int = 4,
        resume: bool = True,
        groups: bool = True,
        report_path: str | Path,
        log_dir: str | Path | None = None,
        stop_after_failures: int = DEFAULT_STOP_AFTER_FAILURES,
        progress: Callable[[str], None] = print,
    ) -> dict:
    """Execute a plan: the harvest pass, the merge, then the parallel pass.

    Parameters
    ----------
    plan : dict
        From schedule.build_plan.
    registry_dir : str or Path
        Per-group registries and the merged one land here.
    run_options : dict
        Forwarded verbatim to pipeline.run_all; ``cutout_arcsec`` must
        match the plan's.
    which_pass : str
        'harvest', 'parallel', or 'all'. [default: 'all']
    groups : bool
        Run the harvest groups concurrently. False runs every harvest
        target in one sequence against one registry -- slower, and the
        reference the grouped path is judged against. [default: True]
    resume : bool
        Skip targets that already have a measured table and a readable
        sidecar. [default: True]
    stop_after_failures : int
        Abandon the sweep once this many targets have failed; 0 disables.
        [default: 20]

    Returns
    -------
    summary : dict
        Per-pass counts, merge problems, audit violations, report path.
    """
    if run_options.get("cutout_arcsec") != plan["cutout_arcsec"]:
        raise ValueError(
            f"the sweep's stamp width {run_options.get('cutout_arcsec')} "
            f"differs from the plan's {plan['cutout_arcsec']}; every radius "
            f"in the grouping derives from it, so re-plan at this width")
    if plan.get("n_violations"):
        raise ValueError(
            f"the plan records {plan['n_violations']} boundary crossing(s); "
            f"raise its link margin and re-plan before sweeping")

    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)

    options = {"resume": resume, "run": run_options,
               "log_dir": None if log_dir is None else str(log_dir)}
    merged_path = registry_dir / MERGED_REGISTRY_NAME
    rows: list[dict] = []
    counts = {STATUS_OK: 0, STATUS_FAILED: 0, STATUS_SKIPPED: 0}
    summary = {"merge_problems": [], "violations": [], "aborted": False}

    def record(new: list[dict] | dict) -> None:
        for row in [new] if isinstance(new, dict) else new:
            rows.append(row)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            detail = row["error"] or row["failures"] or ""
            progress(f"  {row['status']:7s} {row['name']} {detail}")
        _write_report(report_path, rows)

    def over_budget() -> bool:
        return bool(stop_after_failures
                    and counts[STATUS_FAILED] >= stop_after_failures)

    # ---- harvest pass -------------------------------------------------
    harvest = [r for r in plan["targets"] if r["pass"] == PASS_HARVEST]
    if which_pass in ("harvest", "all") and harvest:
        by_group: dict[int, list[dict]] = {}
        for row in harvest:
            by_group.setdefault(row["group"], []).append(row)

        if not groups:
            progress(f"harvest pass: {len(harvest)} targets, one sequence "
                     f"(grouping off)")
            ordered = sorted(harvest, key=lambda r: (r["group"], r["order"]))
            calls = [(_run_group, (ordered, str(merged_path), options), {})]
            summary["aborted"] = _execute(1, calls, record, over_budget)
        else:
            progress(f"harvest pass: {len(harvest)} targets in "
                     f"{len(by_group)} group(s), "
                     f"{min(workers, len(by_group))} worker(s)")
            calls = [(_run_group,
                      (members, str(registry_dir / f"group_{group}.json"),
                       options), {})
                     for group, members in sorted(by_group.items())]
            summary["aborted"] = _execute(workers, calls, record, over_budget)

    # ---- merge, audit, freeze -----------------------------------------
    if which_pass in ("harvest", "all") and harvest and not summary["aborted"]:
        if groups:
            paths = sorted(registry_dir.glob("group_*.json"))
            if not paths:
                # Every harvest target was resumed, so no group wrote
                # anything this time. Merging the empty set would replace
                # a registry an earlier sweep built with nothing.
                progress(f"no group registries written; keeping "
                         f"{merged_path} as it stands")
                paths = None
            merged, problems = merge_registries(paths or [])
            summary["merge_problems"] = problems
            registries = {int(p.stem.split("_")[1]): load_registry(p)
                          for p in paths or []}
            summary["violations"] = audit_merged(
                registries, plan["targets"],
                scene_arcsec=plan["scene_radius_arcsec"])
            if problems or summary["violations"]:
                progress(f"MERGE REFUSED: {len(problems)} shared key(s), "
                         f"{len(summary['violations'])} boundary crossing(s)")
                summary["aborted"] = True
            elif paths:
                save_registry(merged, merged_path)
                progress(f"merged {len(merged)} entries from {len(paths)} "
                         f"group registr(ies) -> {merged_path}")
        else:
            progress(f"registry: {len(load_registry(merged_path))} entries "
                     f"-> {merged_path}")

    # ---- parallel pass ------------------------------------------------
    parallel = [r for r in plan["targets"] if r["pass"] == PASS_PARALLEL]
    if which_pass in ("parallel", "all") and parallel \
            and not summary["aborted"]:
        registry_arg = str(merged_path) if merged_path.exists() else None
        progress(f"parallel pass: {len(parallel)} targets, {workers} "
                 f"worker(s), registry frozen")
        calls = [(_run_logged, (target,),
                  {"registry_path": registry_arg, "registry_update": False,
                   "options": options})
                 for target in parallel]
        summary["aborted"] = _execute(workers, calls, record, over_budget)

    summary.update({"n_ok": counts[STATUS_OK],
                    "n_failed": counts[STATUS_FAILED],
                    "n_skipped": counts[STATUS_SKIPPED],
                    "n_harvest": len(harvest), "n_parallel": len(parallel),
                    "registry": str(merged_path),
                    "report": str(report_path)})
    progress(f"{summary['n_ok']} ok, {summary['n_failed']} failed, "
             f"{summary['n_skipped']} skipped -> {report_path}")
    return summary
