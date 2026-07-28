"""
provenance.py

Product Provenance Sidecars
---------------------------------------------------------
Every table or figure sedphot writes gets a JSON sidecar recording how it was
made: the query/measurement parameters (caller-supplied), plus automatic
fields -- package version, git revision of the code that ran, timestamp, and
a content hash of the product itself.

Data products:
    <product-stem>.provenance.json    next to every written product (the
                                      product's extension is replaced)

Requirements:
    (stdlib only)

Notes:
    Git fields are fail-soft: a pip-installed copy outside the repo records
    null rather than raising.
    The automatic field names are reserved: a caller key of the same name is
    dropped with a printed warning rather than replacing the field the
    sidecar exists to guarantee.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path

from . import __version__


# ------------------------------------
# Helpers
# ------------------------------------
def sha256_16(path: str | Path) -> str:
    """First 16 hex chars of the sha256 of a file's bytes."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return digest[:16]


def git_state() -> dict:
    """Git revision + dirty flag of the repository containing the installed
    sedphot source; fail-soft (null outside a repo)."""
    repo_dir = Path(__file__).resolve().parent
    try:
        rev = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty_probe = subprocess.run(
            ["git", "-C", str(repo_dir), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        return {"git_rev": rev, "git_dirty": bool(dirty_probe)}
    except Exception:
        return {"git_rev": None, "git_dirty": None}


# ------------------------------------
# Sidecar writer
# ------------------------------------
def write_sidecar(product_path: str | Path, meta: dict) -> Path:
    """Write the sidecar next to a written product, replacing the product's
    extension with .provenance.json.

    Parameters
    ----------
    product_path : str or Path
        The product file (must already exist -- its hash goes in the sidecar).
    meta : dict
        Caller-supplied provenance: query parameters, service endpoints,
        match separations, measurement settings. Keys naming an automatic
        field are dropped (see Notes).

    Returns
    -------
    sidecar_path : Path
        The written sidecar.
    """
    product_path = Path(product_path)
    automatic = {
        "product": product_path.name,
        "written": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "sha256_16": sha256_16(product_path),
        "package": "sedphot",
        "package_version": __version__,
        **git_state(),
    }
    # The automatic fields are the whole point of the sidecar, so they win a
    # name collision -- and the loser is named: a caller key silently
    # replacing the hash or the git revision would leave a record that reads
    # authoritative and is not.
    clash = sorted(set(meta) & set(automatic))
    if clash:
        print(f"  [provenance] {product_path.name}: caller key(s) "
              f"{', '.join(clash)} name automatic fields and are dropped")
    record = {**automatic,
              **{key: value for key, value in meta.items() if key not in automatic}}
    sidecar_path = product_path.with_suffix(".provenance.json")
    sidecar_path.write_text(json.dumps(record, indent=2, default=str) + "\n",
                            encoding='utf-8')
    return sidecar_path
