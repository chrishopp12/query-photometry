"""
spherex.py

SPHEREx Spectrophotometry Retrieval (IRSA)
---------------------------------------------------------
Programmatic driver for the IRSA SPHEREx Spectrophotometry Tool, ported
whole from a1925_nbcg/sed_photoz/spherex_fetch.py. The output is the raw
per-exposure table (one row per visit x LVF channel: lambda, lambda_width,
flux, flux_err, flags, ...) written VERBATIM -- downstream SED machinery
owns binning and quality cuts; this module never coerces it to the
broadband schema.

The tool is a GUI over an IVOA UWS 1.1 async service. Direct UWS job
creation is token-gated (403 for guests), so submission goes the way the
public GUI does: through Firefly's command server (guest session), then the
open UWS endpoint is polled by job id. The direct-UWS path is kept for the
day IRSA documents the credential.

Source model: POINT, or ELLIPTICAL with a frozen Sersic shape. UNIT TRAP,
preserved from the original: Firefly's ServerRequest carries
effectiveRadius in DEGREES; the UWS ELLIPTICAL string and every CLI surface
use ARCSEC. The Sersic dataclass stores arcsec and converts.

Data products (under <out_dir>/Photometry/SPHEREx/):
    table_photometry.csv (+ .provenance.json)   the raw per-exposure table

Requirements:
    numpy, pandas, requests, astropy

Notes:
    The programmatic path failed upstream for a period in 2026-06 (IRSA
    pipeline errors); the manual fallback is the Data Explorer GUI's
    "save as CSV", saved to the same path -- fetch() prints the recipe when
    the service fails. Hand-downloaded raw tables are irreplaceable
    ground truth: never overwrite one without archiving it.
"""
from __future__ import annotations

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

    n           : Sersic index (the tool caps this at 6).
    axis_ratio  : "Major/Minor Axis Ratio" = a/b >= 1.
    pa_deg      : position angle, degrees E of N.
    reff_arcsec : effective radius, arcsec.
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
    tool's cap of 6 (with a printed note -- the A1925 control fit n=6.13 was
    submitted as exactly this clip).
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
                         tbl_id=None, job_id=None, ff_session_id=None):
    """Build the Firefly ServerRequest dict for SpectrophotometryProcessor.

    model=None -> point-source forced photometry (shapeFit=false); a Sersic
    -> elliptical model with the shape fields the GUI sends.
    """
    tbl_id = tbl_id or _table_id()
    req = {
        "id": "SpectrophotometryProcessor",
        "UserTargetWorldPt": f"{ra};{dec};EQ_J2000",
        # The backend validates this as an INTEGER ('15.0' is rejected with
        # "Command line validation error" -- verified on the live service
        # 2026-07-05).
        "bgEstimationRegion": str(int(round(float(bkg_region_size)))),
        "exposureTimeMode": "mjd",
        "CONE_AREA_KEY_RESERVED": "CONE",
        "startIdx": 0,
        "pageSize": 2147483647,
        "tbl_id": tbl_id,
        "META_INFO": {"title": "Spectrophotometry Targets", "tbl_id": tbl_id},
    }
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
                   ff_session_id=None):
    """Submit a spectrophotometry job via Firefly.

    Returns (firefly_job_id, uws_url_or_None): the submit response either
    already carries the UWS jobUrl or the caller polls CmdSrv until it
    appears. The id is server-assigned.
    """
    req = build_server_request(ra, dec, model=model,
                               bkg_region_size=bkg_region_size,
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
    """Poll Firefly's view of the job: (phase, uws_url_or_None, raw).

    A freshly-submitted job can briefly read back as JSON null before the
    server registers it; treat that as still-pending rather than crashing.
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
                            out_csv=None, session=None, ff_session_id=None,
                            poll=5, timeout=3600, verbose=True):
    """End-to-end: submit -> wait -> download the per-exposure table.

    Returns a DataFrame (also written to out_csv if given).
    """
    session = session or make_session()

    def log(*args):
        if verbose:
            print("[spherex]", *args)

    fjob, uws_url = submit_firefly(session, ra, dec, model=model,
                                   bkg_region_size=bkg_region_size,
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
                      token=None):
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
    response = session.post(UWS_ASYNC, data=data, headers=headers,
                            allow_redirects=True, timeout=60)
    response.raise_for_status()
    return response.url


# ------------------------------------
# Provider-style wrapper
# ------------------------------------
def fetch(coord, *, out_dir, model: Sersic | None = None,
          bkg_region_size: float = 15, poll: float = 5,
          timeout: float = 3600) -> ProviderResult:
    """Fetch the raw SPHEREx table into <out_dir>/Photometry/SPHEREx/.

    Refuses to overwrite an existing table (raw tables may be irreplaceable
    hand downloads); prints the manual-GUI recipe on service failure.

    Returns
    -------
    result : ProviderResult
        status ok with meta.path on success; error with the recipe (and no
        partial file) otherwise.
    """
    spherex_dir = Path(out_dir) / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True, exist_ok=True)
    out_csv = spherex_dir / "table_photometry.csv"
    if out_csv.exists():
        return ProviderResult(
            provider='spherex', status=STATUS_ERROR,
            message=f"{out_csv} already exists -- raw SPHEREx tables can be "
                    f"irreplaceable manual downloads; move it aside deliberately "
                    f"before re-fetching")

    try:
        df = fetch_spectrophotometry(
            float(coord.ra.deg), float(coord.dec.deg), model=model,
            bkg_region_size=bkg_region_size, out_csv=str(out_csv),
            poll=poll, timeout=timeout)
    except Exception as e:
        print(f"  [spherex] fetch failed: {type(e).__name__}: {e}")
        print(f"  [spherex] {MANUAL_RECIPE}")
        return ProviderResult(provider='spherex', status=STATUS_ERROR,
                              message=f"{type(e).__name__}: {e}; {MANUAL_RECIPE}")

    write_sidecar(out_csv, {
        "kind": "spherex_spectrophotometry_raw",
        "target": {"ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg)},
        "model": ("point" if model is None else
                  {"type": "sersic", "n": model.n, "axis_ratio": model.axis_ratio,
                   "pa_deg": model.pa_deg, "reff_arcsec": model.reff_arcsec}),
        "bkg_region_size_arcsec": bkg_region_size,
        "service": UWS_BASE,
        "n_rows": len(df),
        "columns": list(df.columns),
    })
    return ProviderResult(provider='spherex', status=STATUS_OK,
                          message=f"{len(df)} visit-channel rows",
                          meta={'path': str(out_csv)})
