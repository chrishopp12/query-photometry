"""
spherex.py

SPHEREx Spectrophotometry Retrieval (IRSA)
---------------------------------------------------------
Programmatic driver for the IRSA SPHEREx Spectrophotometry Tool. The output
is the raw per-exposure table (one row per visit x LVF channel: lambda,
lambda_width, flux, flux_err, flags, ...) written VERBATIM -- downstream
SED machinery owns binning and quality cuts; this module never coerces it
to the broadband schema.

The tool is a GUI over an IVOA UWS 1.1 async service. Direct UWS job
creation is token-gated (403 for guests), so submission goes the way the
public GUI does: through Firefly's command server (guest session), then the
open UWS endpoint is polled by job id. The direct-UWS path is kept for the
day IRSA documents the credential.

Source model: POINT, or ELLIPTICAL with a frozen Sersic shape. UNIT TRAP:
Firefly's ServerRequest carries effectiveRadius in DEGREES; the UWS
ELLIPTICAL string and every CLI surface use ARCSEC. The Sersic dataclass
stores arcsec and converts.

Data products (under <out_dir>/Photometry/SPHEREx/):
    table_photometry.<tag>.csv (+ .provenance.json)   one raw per-exposure
        table per extraction configuration, tag = '<model>-<hash6>' over
        the config (model + params + bkg region + MJD window)
    extractions.json                                  manifest: the tag
        decoder ring, regenerable from the sidecars

Requirements:
    numpy, pandas, requests, astropy (defusedxml used for XML when installed)

Notes:
    Nothing on disk is overwritten or renamed: hand-downloaded raw tables
    are irreplaceable ground truth, results are not byte-reproducible
    across fetch dates (server-side calibration evolves), and existing
    filenames are baked into roster/run provenance. Re-requesting a
    configuration that is already on disk -- under its tagged name, or as
    a pre-tag bare table_photometry.csv whose sidecar records the same
    config -- reuses it; move a table aside deliberately to force a
    re-fetch. When the service fails, fetch() prints the manual Data
    Explorer recipe -- "save as CSV" -- instead of writing a partial file.
    Epochs with broken file metadata kill jobs server-side; the
    IRSA-documented workaround is restricting to a known-good MJD window
    (mjd_range).
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import secrets
import time
import warnings
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path

try:
    # Hardened parser (XXE/entity-expansion safe) when available.
    from defusedxml import ElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import requests

from .provenance import write_sidecar
from .results import STATUS_ERROR, STATUS_OK, ProviderResult

# ------------------------------------
# Endpoints
# ------------------------------------
APP_BASE = "https://irsa.ipac.caltech.edu/applications/spherex"
CMDSRV_ASYNC = APP_BASE + "/CmdSrv/async"                  # Firefly job servlet
UWS_BASE = "https://irsa.ipac.caltech.edu/api/spherex/spectrophotometry"
UWS_ASYNC = UWS_BASE + "/async"                            # IVOA UWS 1.1 service

# ------------------------------------
# Constants
# ------------------------------------
UWS_NS = {"uws": "http://www.ivoa.net/xml/UWS/v1.0",
          "xlink": "http://www.w3.org/1999/xlink"}

# UWS / Firefly phases that mean "still working".
_PENDING = {"PENDING", "QUEUED", "EXECUTING", "RUN", "UNKNOWN", "HELD", "SUSPENDED"}

SPHEREX_N_MAX = 6.0    # the tool rejects Sersic indices above 6

MANUAL_RECIPE = (
    "Manual fallback: IRSA Data Explorer -> SPHEREx Spectrophotometry Tool, "
    "enter the position (and Sersic shape for elliptical mode), run, then "
    "'save as CSV' the per-exposure table to Photometry/SPHEREx/. Raw tables "
    "are irreplaceable -- archive before overwriting."
)


# ------------------------------------
# Source model
# ------------------------------------
@dataclass
class Sersic:
    """Frozen Sersic profile for forced photometry (matches the GUI fields).

    Parameters
    ----------
    n : float
        Sersic index (the tool caps this at 6).
    axis_ratio : float
        "Major/Minor Axis Ratio" = a/b >= 1.
    pa_deg : float
        Position angle, degrees E of N.
    reff_arcsec : float
        Effective radius, arcsec.
    """
    n: float
    axis_ratio: float
    pa_deg: float
    reff_arcsec: float

    @property
    def reff_deg(self) -> float:
        # Firefly's ServerRequest carries effectiveRadius in DEGREES; the
        # server converts it to ARCSEC for the UWS ELLIPTICAL string.
        return self.reff_arcsec / 3600.0


def sersic_from_shape(shape_sky: dict) -> Sersic:
    """Bridge a measure-module sky shape into the tool's Sersic convention.

    Converts ellip (1 - b/a) to axis_ratio (a/b) and clips the index to the
    tool's cap of 6, printing a note when the clip fires.

    Parameters
    ----------
    shape_sky : dict
        Sky-frame shape with keys 'n', 'ellip', 'pa_deg', 'reff_arcsec'.

    Returns
    -------
    model : Sersic
        Tool-convention shape, index clipped to SPHEREX_N_MAX.
    """
    n = float(shape_sky['n'])
    if n > SPHEREX_N_MAX:
        print(f"  [spherex] Sersic n={n:.2f} clipped to the tool cap "
              f"{SPHEREX_N_MAX:g}")
        n = SPHEREX_N_MAX
    ellip = float(shape_sky['ellip'])
    return Sersic(n=n,
                  axis_ratio=1.0 / max(1.0 - ellip, 1e-3),
                  pa_deg=float(shape_sky['pa_deg']),
                  reff_arcsec=float(shape_sky['reff_arcsec']))


# ------------------------------------
# Firefly ServerRequest construction
# ------------------------------------
def _table_id() -> str:
    return "Spec-photo-tbl-" + secrets.token_hex(4)


def build_server_request(ra, dec, model=None, bkg_region_size=15,
                         mjd_range=None, tbl_id=None, job_id=None,
                         ff_session_id=None):
    """Build the Firefly ServerRequest dict for SpectrophotometryProcessor.

    Parameters
    ----------
    ra, dec : float
        Target position, ICRS degrees.
    model : Sersic, optional
        None requests point-source forced photometry (shapeFit=false); a
        Sersic requests the elliptical model with the shape fields the GUI
        sends. [default: None]
    bkg_region_size : float
        Background estimation region, pixels; the backend validates it as
        an integer. [default: 15]
    mjd_range : (float, float), optional
        Limit the visits used -- the GUI's "Time Range / MJD values" fields
        (startTime/endTime), which map to the UWS TIME parameter. Epochs
        with broken file metadata kill jobs server-side; the IRSA-documented
        workaround is a cut to a known-good MJD window.
    tbl_id, job_id, ff_session_id : str, optional
        Firefly bookkeeping; tbl_id is generated when not given.

    Returns
    -------
    req : dict
        ServerRequest payload for the CmdSrv tableSearch command.
    """
    tbl_id = tbl_id or _table_id()
    req = {
        "id": "SpectrophotometryProcessor",
        "UserTargetWorldPt": f"{ra};{dec};EQ_J2000",
        # The backend validates this as an INTEGER in pixels ('15.0' is
        # rejected with "Command line validation error").
        "bgEstimationRegion": str(int(round(float(bkg_region_size)))),
        "exposureTimeMode": "mjd",
        "CONE_AREA_KEY_RESERVED": "CONE",
        "startIdx": 0,
        "pageSize": 2147483647,
        "tbl_id": tbl_id,
        "META_INFO": {"title": "Spectrophotometry Targets", "tbl_id": tbl_id},
    }
    if mjd_range is not None:
        start, end = sorted(float(v) for v in mjd_range)
        req["startTime"] = f"{start:.7f}".rstrip('0').rstrip('.')
        req["endTime"] = f"{end:.7f}".rstrip('0').rstrip('.')
    if model is not None:
        req.update({
            "shapeFit": "true",
            "sersicIdx": str(model.n),
            "axisRatio": str(model.axis_ratio),
            "positionAngle": str(model.pa_deg),
            "effectiveRadius": f"{model.reff_deg:.10f}",       # DEGREES
        })
    else:
        req["shapeFit"] = "false"
    if job_id:
        req["META_INFO"]["jobId"] = job_id
    if ff_session_id:
        req["ffSessionId"] = ff_session_id
    return req


# ------------------------------------
# Session + submission (Firefly guest path)
# ------------------------------------
def make_session(bootstrap=True):
    """Return a requests.Session prepared to talk to Firefly.

    The guest CmdSrv submit path needs no ffSessionId -- the server mints
    its own on submit -- so priming cookies with a single GET of the app is
    all that's required; job reads and polls hit the open UWS endpoint.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "sedphot-spherex/0.2 (research script)"})
    if bootstrap:
        try:
            session.get(APP_BASE + "/", timeout=30)
        except requests.RequestException:
            pass
    return session


def submit_firefly(session, ra, dec, model=None, bkg_region_size=15,
                   mjd_range=None, ff_session_id=None):
    """Submit a spectrophotometry job via Firefly's command server.

    Returns
    -------
    firefly_job_id : str
        Server-assigned Firefly job id.
    uws_url : str or None
        UWS job URL when the submit response already carries it; otherwise
        the caller polls CmdSrv until it appears.
    """
    req = build_server_request(ra, dec, model=model,
                               bkg_region_size=bkg_region_size,
                               mjd_range=mjd_range,
                               ff_session_id=ff_session_id)
    resp = session.post(
        CMDSRV_ASYNC,
        data={"cmd": "tableSearch", "request": json.dumps(req)},
        timeout=60,
    )
    resp.raise_for_status()
    info = resp.json()
    job_id = info.get("jobId")
    if not job_id:
        raise RuntimeError(f"Firefly submit returned no jobId: {info!r}")
    uws_url = (info.get("jobInfo") or {}).get("jobUrl")
    return job_id, uws_url


def firefly_status(session, firefly_job_id):
    """Poll Firefly's view of the job.

    A freshly submitted job can briefly read back as JSON null before the
    server registers it; that case is treated as still-pending rather than
    an error.

    Returns
    -------
    phase, uws_url, payload : tuple
        Firefly phase (or None), UWS job URL (or None), and the raw JSON.
    """
    response = session.get(f"{CMDSRV_ASYNC}/{firefly_job_id}", timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not payload:
        return None, None, payload
    return (payload.get("phase"),
            (payload.get("jobInfo") or {}).get("jobUrl"),
            payload)


# ------------------------------------
# UWS retrieval (open by id)
# ------------------------------------
def uws_status(session, uws_url):
    """Return (phase, [ {id, href}, ... ]) for a UWS job document."""
    response = session.get(uws_url, timeout=30)
    response.raise_for_status()
    root = ET.fromstring(response.text)
    phase = root.findtext("uws:phase", namespaces=UWS_NS)
    results = [
        {"id": res.get("id"),
         "href": res.get("{http://www.w3.org/1999/xlink}href")}
        for res in root.findall("uws:results/uws:result", UWS_NS)
    ]
    return phase, results


def _votable_to_flat(content):
    """Parse an IRSA spectrophotometry VOTable into a flat per-exposure table.

    The tool returns ONE row per source with per-visit measurements packed
    as array-valued columns; this explodes them to one row per visit (what
    the GUI's "save as CSV" produces). Two IRSA quirks are handled: the
    non-spec arraysize="54x*" on obs_publisher_did (captured, rewritten,
    re-split fixed-width) and array-valued columns generally.
    """
    from astropy.io.votable import parse_single_table

    text = content.decode("utf-8", "replace")
    packed = {}  # field name -> char width, for non-spec "Nx*" char columns

    def _fix(match):
        tag = match.group(0)
        name = re.search(r'name="([^"]*)"', tag)
        width = re.search(r'arraysize="(\d+)x\*?"', tag)
        if name and width:
            packed[name.group(1)] = int(width.group(1))
        return re.sub(r'arraysize="\d+x\*?"', 'arraysize="*"', tag)

    text = re.sub(r"<FIELD\b[^>]*>", _fix, text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        table = parse_single_table(
            BytesIO(text.encode("utf-8")), verify="warn").to_table()

    cols, n = {}, 1
    for c in table.colnames:
        cell = table[c][0]
        if c in packed:                       # packed char: split fixed-width
            s = cell.decode() if isinstance(cell, bytes) else str(cell)
            w = packed[c]
            cols[c] = [s[i:i + w].strip() for i in range(0, len(s), w)]
            n = max(n, len(cols[c]))
        elif np.ndim(cell) > 0:               # per-visit array -> explode
            cols[c] = [x.decode() if isinstance(x, bytes) else x
                       for x in np.asarray(cell)]
            n = max(n, len(cols[c]))
        else:                                  # source-level scalar -> broadcast
            cols[c] = cell.decode() if isinstance(cell, bytes) else cell
    return pd.DataFrame(
        {c: (v if isinstance(v, list) else [v] * n) for c, v in cols.items()})


def read_result_table(session, href):
    """Download a UWS result href and return the flat per-exposure DataFrame."""
    response = session.get(href, timeout=120)
    response.raise_for_status()
    text = response.text
    head = text.lstrip()[:256]
    upper = head.upper()
    try:
        if upper.startswith("<?XML") or "<VOTABLE" in upper:
            return _votable_to_flat(response.content)
        if head[:1] in ("\\", "|"):                 # IPAC table markers
            from astropy.table import Table
            return Table.read(StringIO(text), format="ipac").to_pandas()
        return pd.read_csv(StringIO(text))          # CSV / TSV
    except Exception as exc:
        raise RuntimeError(
            f"Could not parse result table from {href}; inspect the payload "
            f"and add a parser. First bytes:\n{text[:300]}"
        ) from exc


# ------------------------------------
# Orchestration
# ------------------------------------
def _wait(poll_fn, interval=5, timeout=3600, on_update=None, done=None):
    """Poll poll_fn -> (phase, payload) until done, or raise on timeout."""
    t0 = time.time()
    while True:
        phase, payload = poll_fn()
        if on_update:
            on_update(phase, payload)
        if done is not None:
            finished = done(phase, payload)
        else:
            finished = bool(phase) and phase.upper() not in _PENDING
        if finished:
            return phase, payload
        if time.time() - t0 > timeout:
            raise TimeoutError(f"job still {phase!r} after {timeout}s")
        time.sleep(interval)


def fetch_spectrophotometry(ra, dec, model=None, bkg_region_size=15,
                            mjd_range=None, out_csv=None, session=None,
                            ff_session_id=None, poll=5, timeout=3600,
                            verbose=True):
    """End-to-end: submit -> wait -> download the per-exposure table.

    Parameters
    ----------
    ra, dec : float
        Target position, ICRS degrees.
    model : Sersic, optional
        Elliptical source model; None for point-source forced photometry.
        [default: None]
    bkg_region_size : float
        Background estimation region, pixels (validated server-side as an
        integer). [default: 15]
    mjd_range : (float, float), optional
        Restrict to visits in this MJD window.
    out_csv : str or Path, optional
        Also write the table here when given.
    session : requests.Session, optional
        Session to reuse. [default: a fresh make_session()]
    ff_session_id : str, optional
        Explicit Firefly session id.
    poll : float
        Poll interval, seconds. [default: 5]
    timeout : float
        Give up after this many seconds. [default: 3600]
    verbose : bool
        Print progress. [default: True]

    Returns
    -------
    df : pd.DataFrame
        Flat per-exposure table, one row per visit x channel.
    """
    session = session or make_session()

    def log(*args):
        if verbose:
            print("[spherex]", *args)

    fjob, uws_url = submit_firefly(session, ra, dec, model=model,
                                   bkg_region_size=bkg_region_size,
                                   mjd_range=mjd_range,
                                   ff_session_id=ff_session_id)
    log("submitted; firefly job", fjob)

    # The UWS jobUrl appears while Firefly is still EXECUTING -- key the
    # wait off the URL, not the phase.
    if not uws_url:
        phase, uws_url = _wait(
            lambda: firefly_status(session, fjob)[:2],
            interval=poll, timeout=timeout,
            on_update=lambda p, _u: log("firefly", p),
            done=lambda _p, url: bool(url))
    if not uws_url:
        raise RuntimeError("Firefly never exposed a UWS jobUrl")
    log("UWS job", uws_url)

    phase, results = _wait(
        lambda: uws_status(session, uws_url),
        interval=poll, timeout=timeout,
        on_update=lambda p, _r: log("uws", p))
    if phase.upper() != "COMPLETED":
        raise RuntimeError(f"UWS job ended in phase {phase} (not COMPLETED)")
    if not results:
        raise RuntimeError("UWS job COMPLETED but exposed no results")

    df = read_result_table(session, results[0]["href"])
    log(f"got table: {len(df)} rows, {len(df.columns)} cols")
    if out_csv:
        df.to_csv(out_csv, index=False)
        log("wrote", out_csv)
    return df


# ------------------------------------
# Direct-UWS path (needs an IRSA API token)
# ------------------------------------
def elliptical_param(ra, dec, model):
    """UWS ELLIPTICAL string: ra,dec,n,axis_ratio,PA,reff_arcsec (ARCSEC)."""
    return (f"{ra},{dec},{model.n},{model.axis_ratio},"
            f"{model.pa_deg},{model.reff_arcsec}")


def submit_uws_direct(session, ra, dec, model=None, bkg_region_size=15,
                      mjd_range=None, token=None):
    """Create a job straight on the UWS service. Returns the job URL.

    Gated: a direct POST returns 403 for guests; supply token once IRSA
    documents the credential and this becomes the clean path (no Firefly).
    """
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = {"BKG_REGION_SIZE": str(int(round(float(bkg_region_size))))}
    if model is not None:
        data["ELLIPTICAL"] = elliptical_param(ra, dec, model)
    else:
        data["POINT"] = f"{ra},{dec}"
    if mjd_range is not None:
        start, end = sorted(float(v) for v in mjd_range)
        data["TIME"] = f"{start},{end}"
    response = session.post(UWS_ASYNC, data=data, headers=headers,
                            allow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.url


# ------------------------------------
# Extraction configurations
# ------------------------------------
MANIFEST_NAME = "extractions.json"
PRETAG_TABLE_NAME = "table_photometry.csv"   # fetched before tags existed


def _canon(value: float) -> float:
    """Canonical float for configuration identity: 6 significant digits."""
    return float(f"{float(value):.6g}")


def config_payload(model: Sersic | None, bkg_region_size: float = 15,
                   mjd_range=None) -> dict:
    """The extraction-defining parameters, canonically normalized.

    Everything that changes what the tool computes belongs here (source
    model, background region, visit window); provenance-only detail (shape
    origin, fetch date) does not -- the same numeric shape from Tractor or
    from a hand-typed --sersic-params is the same extraction.
    """
    if model is None:
        model_payload: str | dict = "psf"
    else:
        model_payload = {
            "type": "sersic",
            "n": _canon(model.n),
            "axis_ratio": _canon(model.axis_ratio),
            "pa_deg": _canon(model.pa_deg),
            "reff_arcsec": _canon(model.reff_arcsec),
        }
    return {
        "model": model_payload,
        "bkg_region_size_px": int(round(float(bkg_region_size))),
        "mjd_range": ([_canon(v) for v in sorted(float(x) for x in mjd_range)]
                      if mjd_range else None),
    }


def extraction_tag(model: Sersic | None, bkg_region_size: float = 15,
                   mjd_range=None) -> str:
    """Deterministic '<model>-<hash6>' tag naming one configuration."""
    payload = config_payload(model, bkg_region_size, mjd_range)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:6]
    kind = "psf" if model is None else "sersic"
    return f"{kind}-{digest}"


def _sidecar_payload(sidecar: dict) -> dict | None:
    """Rebuild a config payload from a table's provenance sidecar."""
    model = sidecar.get("model")
    if model in ("point", "psf"):
        model_payload: str | dict = "psf"
    elif isinstance(model, dict):
        try:
            model_payload = {
                "type": "sersic",
                "n": _canon(model["n"]),
                "axis_ratio": _canon(model["axis_ratio"]),
                "pa_deg": _canon(model["pa_deg"]),
                "reff_arcsec": _canon(model["reff_arcsec"]),
            }
        except (KeyError, TypeError, ValueError):
            return None
    else:
        return None
    try:
        mjd = sidecar.get("mjd_range")
        return {
            "model": model_payload,
            "bkg_region_size_px": int(round(float(sidecar["bkg_region_size_px"]))),
            "mjd_range": ([_canon(v) for v in sorted(float(x) for x in mjd)]
                          if mjd else None),
        }
    except (KeyError, TypeError, ValueError):
        return None


def _matching_existing_table(spherex_dir: Path, payload: dict,
                             tag: str) -> Path | None:
    """An already-fetched table for this configuration, if any.

    The tagged filename is checked first, then the pre-tagging bare name
    whose sidecar records the same configuration. Pre-tag tables are
    recognized in place and never renamed: their names are baked into
    roster entries and run provenance.
    """
    tagged = spherex_dir / f"table_photometry.{tag}.csv"
    if tagged.exists():
        return tagged
    pretag = spherex_dir / PRETAG_TABLE_NAME
    sidecar_path = pretag.with_suffix(".provenance.json")
    if pretag.exists() and sidecar_path.exists():
        try:
            recorded = json.loads(sidecar_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if _sidecar_payload(recorded) == payload:
            return pretag
    return None


def _index_extraction(spherex_dir: Path, tag: str, filename: str,
                      payload: dict, *, shape_origin=None, n_rows=None) -> None:
    """Record one extraction in the manifest (the tag decoder ring).

    The manifest is a regenerable convenience index; the per-table
    provenance sidecars remain authoritative.
    """
    manifest_path = spherex_dir / MANIFEST_NAME
    manifest = {"kind": "spherex_extractions", "entries": {}}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            print(f"  [spherex] unreadable {MANIFEST_NAME}; rebuilding it")
        manifest.setdefault("entries", {})
    manifest["entries"][tag] = {
        "file": filename,
        **payload,
        "shape_origin": shape_origin,
        "n_rows": n_rows,
        "indexed": datetime.datetime.now().astimezone().isoformat(
            timespec="seconds"),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


# ------------------------------------
# Provider-style wrapper
# ------------------------------------
def fetch(coord, *, out_dir, model: Sersic | None = None,
          bkg_region_size: float = 15, mjd_range: tuple | None = None,
          poll: float = 5, timeout: float = 3600,
          shape_origin: str | None = None) -> ProviderResult:
    """Fetch the raw SPHEREx table for ONE extraction configuration.

    Each distinct configuration (source model + background region + MJD
    window) owns a tagged table under Photometry/SPHEREx/, so PSF and
    Sersic extractions -- or different Sersic shapes -- coexist without
    manual renames. Re-requesting an existing configuration reuses its
    table (status ok, nothing fetched); move a table aside deliberately to
    force a re-fetch. Nothing on disk is ever overwritten or renamed (raw
    tables can be irreplaceable manual downloads, and existing names are
    baked into run provenance).

    Parameters
    ----------
    coord : SkyCoord
        Target position.
    out_dir : str or Path
        Target directory; tables land in Photometry/SPHEREx/ under it.
    model : Sersic, optional
        Elliptical source model; None for point-source forced photometry.
        [default: None]
    bkg_region_size : float
        Background estimation region, pixels. [default: 15]
    mjd_range : (float, float), optional
        Restrict to visits in this MJD window (the GUI's Time Range). The
        IRSA-prescribed workaround for pipeline failures on files with
        broken metadata is a cut to the known-good window.
    poll : float
        Poll interval, seconds. [default: 5]
    timeout : float
        Give up after this many seconds. [default: 3600]
    shape_origin : str, optional
        Provenance of the Sersic shape (e.g. Tractor table + type, or the
        image fit); recorded in the sidecar's model block and the manifest.

    Returns
    -------
    result : ProviderResult
        status ok with meta.path and meta.tag on success (meta.reused when
        the configuration was already on disk); error with the manual
        recipe (and no partial file) otherwise.
    """
    spherex_dir = Path(out_dir) / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True, exist_ok=True)
    payload = config_payload(model, bkg_region_size, mjd_range)
    tag = extraction_tag(model, bkg_region_size, mjd_range)

    existing = _matching_existing_table(spherex_dir, payload, tag)
    if existing is not None:
        # Index it if the manifest does not know it yet (pre-tag tables,
        # rebuilt manifests); origin and row count come from its sidecar.
        recorded = {}
        sidecar_path = existing.with_suffix(".provenance.json")
        if sidecar_path.exists():
            try:
                recorded = json.loads(sidecar_path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        manifest_path = spherex_dir / MANIFEST_NAME
        known = {}
        if manifest_path.exists():
            try:
                known = json.loads(manifest_path.read_text()).get("entries", {})
            except (OSError, json.JSONDecodeError):
                pass
        if tag not in known:
            recorded_model = recorded.get("model")
            origin = (recorded_model.get("shape_origin")
                      if isinstance(recorded_model, dict) else None)
            _index_extraction(spherex_dir, tag, existing.name, payload,
                              shape_origin=origin or shape_origin,
                              n_rows=recorded.get("n_rows"))
        print(f"  [spherex] extraction {tag} already on disk: {existing.name}")
        return ProviderResult(provider='spherex', status=STATUS_OK,
                              message=f"reusing extraction {tag} "
                                      f"({existing.name})",
                              meta={'path': str(existing), 'tag': tag,
                                    'reused': True})

    out_csv = spherex_dir / f"table_photometry.{tag}.csv"
    try:
        df = fetch_spectrophotometry(
            float(coord.ra.deg), float(coord.dec.deg), model=model,
            bkg_region_size=bkg_region_size, mjd_range=mjd_range,
            out_csv=str(out_csv), poll=poll, timeout=timeout)
    except Exception as e:
        print(f"  [spherex] fetch failed: {type(e).__name__}: {e}")
        print(f"  [spherex] {MANUAL_RECIPE}")
        return ProviderResult(provider='spherex', status=STATUS_ERROR,
                              message=f"{type(e).__name__}: {e}; {MANUAL_RECIPE}")

    write_sidecar(out_csv, {
        "kind": "spherex_spectrophotometry_raw",
        "extraction_tag": tag,
        "target": {"ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg)},
        "model": ("point" if model is None else
                  {"type": "sersic", "n": model.n, "axis_ratio": model.axis_ratio,
                   "pa_deg": model.pa_deg, "reff_arcsec": model.reff_arcsec,
                   "shape_origin": shape_origin}),
        "bkg_region_size_px": int(round(float(bkg_region_size))),
        "mjd_range": list(mjd_range) if mjd_range else None,
        "service": UWS_BASE,
        "n_rows": len(df),
        "columns": list(df.columns),
    })
    _index_extraction(spherex_dir, tag, out_csv.name, payload,
                      shape_origin=shape_origin, n_rows=len(df))
    return ProviderResult(provider='spherex', status=STATUS_OK,
                          message=f"{len(df)} visit-channel rows ({tag})",
                          meta={'path': str(out_csv), 'tag': tag})
