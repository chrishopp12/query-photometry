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
open UWS endpoint is polled by job id. The direct-UWS path is unusable
without a credential IRSA does not currently issue to guests;
submit_uws_direct takes a token for when one is available.

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
    a pre-tag bare table_photometry.csv -- reuses it, provided the table's
    own sidecar records that same config; a table no sidecar vouches for is
    neither reused nor overwritten. Move a table aside deliberately to
    force a re-fetch. When the service fails, fetch() prints the manual Data
    Explorer recipe -- "save as CSV" -- instead of writing a partial file.
    Epochs with broken file metadata kill jobs server-side; the
    IRSA-documented workaround is restricting to a known-good MJD window
    (mjd_range).

    Because a written table is never overwritten, a wrong source model is
    permanent: an extraction runs only on a shape that was asked for
    explicitly or looked up successfully, never on a substituted one.
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
CMDSRV_SYNC = APP_BASE + "/CmdSrv/sync"                    # uploads land here
UWS_BASE = "https://irsa.ipac.caltech.edu/api/spherex/spectrophotometry"
UWS_ASYNC = UWS_BASE + "/async"                            # IVOA UWS 1.1 service

# ------------------------------------
# Constants
# ------------------------------------
UWS_NS = {"uws": "http://www.ivoa.net/xml/UWS/v1.0",
          "xlink": "http://www.w3.org/1999/xlink"}

# UWS / Firefly phases that mean "still working".
_PENDING = {"PENDING", "QUEUED", "EXECUTING", "RUN", "UNKNOWN", "HELD", "SUSPENDED"}

# Of those, the ones that mean "has not started yet". A job waiting its
# turn has consumed no extraction time, so it is timed against its own
# budget: the service is allowed to be busy, and killing a job for the
# depth of someone else's queue abandons work that would have run.
_QUEUED = {"PENDING", "QUEUED", "HELD", "SUSPENDED"}

# Default ceiling on time spent waiting to start. Generous, because the
# service is documented to run two jobs at a time and queue the rest --
# so a deep queue is normal operation, not a fault.
QUEUE_TIMEOUT = 21600.0    # seconds

SPHEREX_N_MAX = 6.0    # the tool rejects Sersic indices above 6
SPHEREX_N_MIN = 0.5    # below this the tool is outside its tested range

# The tool treats any source with reff below this as a POINT SOURCE: a
# sub-threshold Sersic request returns photometry bit-identical to a psf
# request of the same window, so the Sersic parameters are cosmetic.
SPHEREX_POINT_SOURCE_REFF = 1.0    # arcsec
SPHEREX_REFF_MAX = 20.0            # arcsec; the tool's stated upper bound

# Sources per joint job. The tool takes the first this many rows of an
# uploaded list and DISCARDS the rest without an error, so a longer list
# is a silent truncation rather than a refusal.
GROUP_SOURCES_MAX = 20

# Firefly ServerRequest keys. A single-position request carries the
# position itself; an upload request names a server-side table and the
# columns to read each parameter from.
FF_MODE_KEY = "CONE_AREA_KEY_RESERVED"
FF_MODE_SINGLE = "CONE"
FF_MODE_UPLOAD = "UPLOAD"

# Column names sedphot writes into an uploaded target table. They are
# ours to choose -- the request names each one explicitly rather than
# relying on the server to guess.
UPLOAD_COL_LABEL = "label"
UPLOAD_COL_RA = "ra"
UPLOAD_COL_DEC = "dec"
UPLOAD_COL_N = "sersic_n"
UPLOAD_COL_AXIS_RATIO = "axis_ratio"
UPLOAD_COL_PA = "pa_deg"
UPLOAD_COL_REFF = "reff_arcsec"

# UNIT TRAP, and it is not the same trap as the single-position path.
# A single-position request carries effectiveRadius in DEGREES; an
# uploaded table is read by column and the tool's own echo of the shape
# it used (shape_r) is in ARCSEC. The upload is therefore written in
# arcsec and verify_group_shapes asserts the echo, so a wrong assumption
# fails loudly on the first job instead of quietly scaling every radius
# by 3600.
UPLOAD_REFF_UNIT = "arcsec"

# Columns the result table carries per source rather than per visit.
# They identify which requested source a block of rows belongs to, and
# echo back the shape the tool actually used.
RESULT_COL_SOURCE_ID = "source_id"
RESULT_COL_RA = "ra"
RESULT_COL_DEC = "dec"
RESULT_SHAPE_COLS = {"n": "sersic", "axis_ratio": "ab_ratio",
                     "pa_deg": "phi", "reff_arcsec": "shape_r"}

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


ROLE_SCIENCE = "science"
ROLE_ANCILLARY = "ancillary"
GROUP_ROLES = (ROLE_SCIENCE, ROLE_ANCILLARY)


@dataclass
class GroupSource:
    """One member of a joint extraction.

    A joint job fits every member simultaneously, so a member's flux
    depends on which other members were present. The role records what
    the member is FOR: a science member's spectrum is a product in its
    own right, an ancillary member is present only to absorb light that
    would otherwise land on a science member.

    Parameters
    ----------
    label : str
        Output stem, and the identity carried through the upload.
    ra_deg, dec_deg : float
        Position, ICRS degrees.
    model : Sersic, optional
        Per-source shape; None requests point-source photometry for this
        member. A job is point-source or elliptical as a whole, so a
        group mixing the two is refused rather than silently coerced.
    role : str
        'science' or 'ancillary'.
    out_dir : str, optional
        Galaxy directory for a science member's products. Required for
        science members, meaningless for ancillary ones.
    name : str, optional
        Original name string, recorded in provenance.
    shape_origin : str, optional
        Provenance of the shape (catalog + type, or the image fit).
    """
    label: str
    ra_deg: float
    dec_deg: float
    model: Sersic | None = None
    role: str = "science"
    out_dir: str | None = None
    name: str | None = None
    shape_origin: str | None = None

    @property
    def is_science(self) -> bool:
        return self.role == ROLE_SCIENCE


def check_shape(model: Sersic, *, where: str = "shape") -> list[str]:
    """Complaints about a Sersic shape the tool would accept regardless.

    The tool validates a typed single-position shape but NOT an uploaded
    one -- an unphysical row is attempted rather than refused, and the
    resulting table looks like every other table. These are the stated
    bounds, checked here so a group config fails before it reaches an
    irreplaceable raw product.

    Parameters
    ----------
    model : Sersic
        Shape to check.
    where : str
        Label for the messages.

    Returns
    -------
    complaints : list of str
        Empty when the shape is inside every stated bound.
    """
    complaints = []
    if not SPHEREX_N_MIN <= model.n <= SPHEREX_N_MAX:
        complaints.append(f"{where}: Sersic n={model.n:g} outside the tool's "
                          f"tested range [{SPHEREX_N_MIN:g}, {SPHEREX_N_MAX:g}]")
    if model.axis_ratio < 1.0:
        complaints.append(f"{where}: axis_ratio={model.axis_ratio:g} is a/b "
                          f"and must be >= 1")
    if not -360.0 <= model.pa_deg <= 360.0:
        complaints.append(f"{where}: pa_deg={model.pa_deg:g} outside "
                          f"[-360, 360]")
    if model.reff_arcsec >= SPHEREX_REFF_MAX:
        complaints.append(f"{where}: reff {model.reff_arcsec:g}\" is at or "
                          f"above the tool's {SPHEREX_REFF_MAX:g}\" ceiling")
    return complaints


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


def _base_request(bkg_region_size, mjd_range, tbl_id, job_id, ff_session_id):
    """The half of the ServerRequest that a single and a joint job share.

    Everything here applies to the whole job: the background region and
    the visit window are one setting per job, not per source.
    """
    tbl_id = tbl_id or _table_id()
    req = {
        "id": "SpectrophotometryProcessor",
        # The backend validates this as an INTEGER in pixels ('15.0' is
        # rejected with "Command line validation error").
        "bgEstimationRegion": str(int(round(float(bkg_region_size)))),
        "exposureTimeMode": "mjd",
        "startIdx": 0,
        "pageSize": 2147483647,
        "tbl_id": tbl_id,
        "META_INFO": {"title": "Spectrophotometry Targets", "tbl_id": tbl_id},
    }
    if mjd_range is not None:
        start, end = sorted(float(v) for v in mjd_range)
        req["startTime"] = f"{start:.7f}".rstrip('0').rstrip('.')
        req["endTime"] = f"{end:.7f}".rstrip('0').rstrip('.')
    if job_id:
        req["META_INFO"]["jobId"] = job_id
    if ff_session_id:
        req["ffSessionId"] = ff_session_id
    return req


def build_server_request(ra, dec, model=None, bkg_region_size=15,
                         mjd_range=None, tbl_id=None, job_id=None,
                         ff_session_id=None):
    """Build the Firefly ServerRequest dict for one position.

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
    req = _base_request(bkg_region_size, mjd_range, tbl_id, job_id,
                        ff_session_id)
    req[FF_MODE_KEY] = FF_MODE_SINGLE
    req["UserTargetWorldPt"] = f"{ra};{dec};EQ_J2000"
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
    return req


def build_group_request(server_file, *, sersic, bkg_region_size=15,
                        mjd_range=None, tbl_id=None, job_id=None,
                        ff_session_id=None):
    """Build the ServerRequest for a joint extraction over an uploaded list.

    The position and the shape are named by COLUMN rather than carried as
    values, so the uploaded table is the whole source list and this
    request only says how to read it.

    Parameters
    ----------
    server_file : str
        Server-side token for the uploaded table (see upload_table).
    sersic : bool
        True runs the elliptical model, reading each source's shape from
        its row; False runs point-source photometry for every member.
        One job is one mode -- the tool has no per-row model switch.
    bkg_region_size : float
        Background estimation region, pixels. [default: 15]
    mjd_range : (float, float), optional
        Restrict to visits in this MJD window.
    tbl_id, job_id, ff_session_id : str, optional
        Firefly bookkeeping; tbl_id is generated when not given.

    Returns
    -------
    req : dict
        ServerRequest payload for the CmdSrv tableSearch command.
    """
    req = _base_request(bkg_region_size, mjd_range, tbl_id, job_id,
                        ff_session_id)
    req.update({
        FF_MODE_KEY: FF_MODE_UPLOAD,
        "uploadFile": server_file,
        # The tool takes this many rows and drops the rest silently; the
        # request states the cap the caller was held to.
        "uploadRowsMax": GROUP_SOURCES_MAX,
        "uploadCenterLonColumns": UPLOAD_COL_RA,
        "uploadCenterLatColumns": UPLOAD_COL_DEC,
        "nameColumn": UPLOAD_COL_LABEL,
        "shapeFit": "true" if sersic else "false",
    })
    if sersic:
        req.update({
            "sersicIdxColumn": UPLOAD_COL_N,
            "axisRatioColumn": UPLOAD_COL_AXIS_RATIO,
            "positionAngleColumn": UPLOAD_COL_PA,
            "effectiveRadiusColumn": UPLOAD_COL_REFF,
        })
    return req


def build_upload_frame(sources: list[GroupSource]) -> pd.DataFrame:
    """The target table a joint job uploads.

    One row per source, in the order given -- the tool keeps the first
    GROUP_SOURCES_MAX rows, so order is what survives truncation.

    Parameters
    ----------
    sources : list of GroupSource
        Members of the group. Either all carry a model or none does.

    Returns
    -------
    frame : pd.DataFrame
        Columns the request names; radii in UPLOAD_REFF_UNIT.

    Raises
    ------
    ValueError
        On an empty group, a group over the cap, duplicate labels, or a
        group mixing point-source and Sersic members.
    """
    if not sources:
        raise ValueError("a group needs at least one source")
    if len(sources) > GROUP_SOURCES_MAX:
        raise ValueError(
            f"{len(sources)} sources exceeds the tool's {GROUP_SOURCES_MAX} "
            f"per job; the extra rows would be dropped without an error")
    labels = [s.label for s in sources]
    duplicated = sorted({x for x in labels if labels.count(x) > 1})
    if duplicated:
        raise ValueError(f"duplicate labels in one group: {duplicated}")
    modelled = [s.model is not None for s in sources]
    if any(modelled) and not all(modelled):
        bare = [s.label for s in sources if s.model is None]
        raise ValueError(
            f"a job is point-source or elliptical as a whole; these members "
            f"carry no shape while others do: {bare}")

    columns = {UPLOAD_COL_LABEL: labels,
               UPLOAD_COL_RA: [float(s.ra_deg) for s in sources],
               UPLOAD_COL_DEC: [float(s.dec_deg) for s in sources]}
    if all(modelled):
        columns.update({
            UPLOAD_COL_N: [float(s.model.n) for s in sources],
            UPLOAD_COL_AXIS_RATIO: [float(s.model.axis_ratio)
                                    for s in sources],
            UPLOAD_COL_PA: [float(s.model.pa_deg) for s in sources],
            UPLOAD_COL_REFF: [float(s.model.reff_arcsec) for s in sources],
        })
    return pd.DataFrame(columns)


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


def _find_server_file(payload):
    """The upload token anywhere in a JSON upload response, or None.

    The documented answer is the delimited text form; this covers a JSON
    one by locating the token by key rather than by position.
    """
    if isinstance(payload, dict):
        for key in ("cacheKey", "serverFile", "file", "path"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                return value
        for value in payload.values():
            found = _find_server_file(value)
            if found:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_server_file(item)
            if found:
                return found
    return None


# Statuses the upload servlet reports as success in its delimited reply.
_UPLOAD_OK = {"200", "ok", "success", "0", "true"}


def _parse_upload_response(text: str) -> str | None:
    """The uploaded file's server-side token, or None.

    Firefly answers an upload as '<status>::<message>::<cacheKey>[::...]'
    -- positionally, so a success with no message reads as '200::::<key>'.
    The key is what a later request names as uploadFile, and it can carry
    an unresolved '${upload-dir}' placeholder that the server expands;
    it is passed through verbatim. A JSON body is accepted as a fallback.
    """
    text = (text or "").strip()
    if not text:
        return None
    if "::" in text and text[:1] not in ("{", "["):
        parts = text.split("::")
        if len(parts) >= 3 and parts[0].strip().lower() in _UPLOAD_OK:
            return parts[2].strip() or None
        return None
    try:
        return _find_server_file(json.loads(text))
    except (ValueError, TypeError):
        return None


def upload_table(session, frame, *, filename="sedphot_targets.csv",
                 timeout=60):
    """Upload a target table to Firefly; return its server-side token.

    A joint job names an uploaded table rather than carrying its rows,
    so this is the step that has no counterpart on the single-position
    path.

    Parameters
    ----------
    session : requests.Session
        Session with the app's cookies primed (see make_session).
    frame : pd.DataFrame
        Target table, from build_upload_frame.
    filename : str
        Name the upload is announced under. [default: sedphot_targets.csv]
    timeout : float
        Transport timeout, seconds. [default: 60]

    Returns
    -------
    server_file : str
        Token to put in the request's uploadFile.

    Raises
    ------
    RuntimeError
        When the response carries no recognizable token; the message
        quotes the payload, since a changed response shape is the one
        failure this cannot work around on its own.
    """
    payload = frame.to_csv(index=False).encode("utf-8")
    # The command rides in the QUERY STRING; the multipart body carries
    # only the file. Sending cmd in the body is answered with a 500.
    response = session.post(
        CMDSRV_SYNC + "?cmd=upload",
        files={"file": (filename, payload, "text/csv")},
        timeout=timeout,
    )
    response.raise_for_status()
    text = (response.text or "").strip()
    server_file = _parse_upload_response(text)
    if not server_file:
        raise RuntimeError(
            f"Firefly upload returned no server file token; the response "
            f"shape changed and the uploadFile key cannot be filled. "
            f"First bytes:\n{text[:300]}")
    return server_file


def submit_request(session, req):
    """POST a built ServerRequest to Firefly's command server.

    Returns
    -------
    firefly_job_id : str
        Server-assigned Firefly job id.
    uws_url : str or None
        UWS job URL when the submit response already carries it; otherwise
        the caller polls CmdSrv until it appears.
    """
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


def submit_firefly(session, ra, dec, model=None, bkg_region_size=15,
                   mjd_range=None, ff_session_id=None):
    """Submit a single-position spectrophotometry job.

    Returns
    -------
    firefly_job_id, uws_url : see submit_request.
    """
    return submit_request(session, build_server_request(
        ra, dec, model=model, bkg_region_size=bkg_region_size,
        mjd_range=mjd_range, ff_session_id=ff_session_id))


def submit_firefly_group(session, sources, bkg_region_size=15,
                         mjd_range=None, ff_session_id=None):
    """Upload a group's target table and submit the joint job.

    Parameters
    ----------
    session : requests.Session
        Session with the app's cookies primed.
    sources : list of GroupSource
        Members to fit simultaneously.
    bkg_region_size : float
        Background estimation region, pixels. [default: 15]
    mjd_range : (float, float), optional
        Restrict to visits in this MJD window.
    ff_session_id : str, optional
        Explicit Firefly session id.

    Returns
    -------
    firefly_job_id, uws_url : see submit_request.
    """
    frame = build_upload_frame(sources)
    server_file = upload_table(session, frame)
    return submit_request(session, build_group_request(
        server_file, sersic=sources[0].model is not None,
        bkg_region_size=bkg_region_size, mjd_range=mjd_range,
        ff_session_id=ff_session_id))


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


def uws_error_detail(session, uws_url):
    """The service's own explanation for a failed job, or None.

    IRSA carries it in the job document's uws:errorSummary and answers
    {job}/error with "This endpoint is not used"; the UWS spec allows
    either, so the job document is read first and /error only as a
    fallback for services that populate it. Without this, a failed
    extraction reports only its phase -- which cannot distinguish a
    rejected request from a service-side fault, and so cannot say
    whether retrying is worth anything.
    """
    def _clean(text):
        return " ".join((text or "").split())[:1000] or None

    try:
        response = session.get(uws_url, timeout=30)
        if response.status_code == 200:
            root = ET.fromstring(response.text)
            summary = root.find("uws:errorSummary", UWS_NS)
            if summary is not None:
                message = _clean(summary.findtext("uws:message",
                                                  namespaces=UWS_NS))
                kind = summary.get("type")
                if message:
                    return f"{message} [{kind}]" if kind else message
    except (requests.RequestException, ET.ParseError):
        pass

    try:
        response = session.get(uws_url.rstrip("/") + "/error", timeout=30)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    text = (response.text or "").strip()
    if not text:
        return None
    try:
        return _clean(" ".join(ET.fromstring(text).itertext()))
    except ET.ParseError:
        pass
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return _clean(text)
    if isinstance(payload, dict):
        for key in ("detail", "message", "error", "errorSummary"):
            if payload.get(key):
                return _clean(str(payload[key]))
    return _clean(text)


def _explode_source_row(table, index, packed):
    """One source's packed VOTable row as a flat per-visit frame.

    Array-valued columns become one row per visit; source-level scalars
    broadcast across them.
    """
    cols, n = {}, 1
    for c in table.colnames:
        cell = table[c][index]
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


def _votable_to_flat(content):
    """Parse an IRSA spectrophotometry VOTable into a flat per-exposure table.

    The tool returns ONE ROW PER SOURCE with per-visit measurements packed
    as array-valued columns; every row is exploded to one row per visit
    (what the GUI's "save as CSV" produces) and the sources concatenated.
    Reading only the first row would return a joint job's first source
    and silently discard the rest, which no downstream check would catch:
    the result is a well-formed table of the wrong length.

    Two IRSA quirks are handled: a non-spec fixed-width char column
    (arraysize="Nx*", rewritten to "*" and re-split at width N) and
    array-valued columns generally.
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

    if len(table) == 0:
        return pd.DataFrame(columns=list(table.colnames))
    return pd.concat(
        [_explode_source_row(table, i, packed) for i in range(len(table))],
        ignore_index=True)


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
# Joint result -> one table per source
# ------------------------------------
def _separations_arcsec(ra, dec, ra0, dec0):
    """Angular separation in arcsec between arrays and one position."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord

    here = SkyCoord(np.asarray(ra, dtype=float) * u.deg,
                    np.asarray(dec, dtype=float) * u.deg)
    there = SkyCoord(float(ra0) * u.deg, float(dec0) * u.deg)
    return here.separation(there).arcsec


def verify_group_shapes(block, source, *, rtol=1e-3) -> list[str]:
    """Complaints where the tool's echoed shape differs from the request.

    The result table repeats the parameters the fit actually used, in
    ARCSEC for the radius. That echo is the only direct evidence of how
    an uploaded column was interpreted -- the single-position request
    carries the radius in degrees, and nothing documents the column's
    unit -- so it is checked rather than assumed. A silent factor of
    3600 would produce a plausible table of the wrong source model.

    Parameters
    ----------
    block : pd.DataFrame
        The rows belonging to one source.
    source : GroupSource
        What was asked for.
    rtol : float
        Relative tolerance. [default: 1e-3]

    Returns
    -------
    complaints : list of str
        Empty when every echoed parameter matches, or when the table
        carries no echo columns to check.
    """
    if source.model is None:
        return []
    complaints = []
    for attr, column in RESULT_SHAPE_COLS.items():
        if column not in block.columns:
            continue
        echoed = pd.to_numeric(block[column], errors='coerce').dropna()
        if echoed.empty:
            continue
        got = float(echoed.iloc[0])
        want = float(getattr(source.model, attr))
        if attr == 'pa_deg':
            got, want = got % 360.0, want % 360.0
        if not np.isclose(got, want, rtol=rtol, atol=1e-6):
            complaints.append(
                f"{source.label}: the tool used {column}={got:g} for a "
                f"requested {attr}={want:g}"
                + (f" (ratio {got / want:.4g})" if want else ""))
    return complaints


def split_group_table(frame, sources, *, tol_arcsec=1.0):
    """Split a joint result into one frame per requested source.

    Sources are matched by POSITION, not by row order or by source_id:
    the tool assigns its own ids and sedphot's guarantee is about the
    position it asked for. A requested source that matches nothing, or
    an output source that matches no request, is an error -- an
    unclaimed spectrum means the job did not fit what the caller thinks
    it fitted.

    Parameters
    ----------
    frame : pd.DataFrame
        Flat per-visit table covering every source in the job.
    sources : list of GroupSource
        The members requested, in upload order.
    tol_arcsec : float
        Match radius. [default: 1.0]

    Returns
    -------
    blocks : dict
        label -> that source's rows, index reset.
    complaints : list of str
        Shape-echo disagreements; empty when every echo matched.

    Raises
    ------
    RuntimeError
        On a missing, ambiguous or unclaimed source.
    """
    for column in (RESULT_COL_RA, RESULT_COL_DEC):
        if column not in frame.columns:
            raise RuntimeError(
                f"the result table has no {column!r} column, so its rows "
                f"cannot be attributed to the requested sources; columns "
                f"are {list(frame.columns)}")

    key = (RESULT_COL_SOURCE_ID if RESULT_COL_SOURCE_ID in frame.columns
           else None)
    groups = (list(frame.groupby(key, sort=True)) if key
              else list(frame.groupby([RESULT_COL_RA, RESULT_COL_DEC],
                                      sort=True)))
    positions = [(float(pd.to_numeric(block[RESULT_COL_RA]).iloc[0]),
                  float(pd.to_numeric(block[RESULT_COL_DEC]).iloc[0]))
                 for _, block in groups]

    blocks, complaints, claimed = {}, [], set()
    for source in sources:
        seps = _separations_arcsec([p[0] for p in positions],
                                   [p[1] for p in positions],
                                   source.ra_deg, source.dec_deg)
        near = [i for i, s in enumerate(seps) if s <= tol_arcsec]
        if not near:
            closest = int(np.argmin(seps)) if len(seps) else None
            detail = (f"closest output source is {seps[closest]:.2f}\" away"
                      if closest is not None else "the job returned none")
            raise RuntimeError(
                f"{source.label}: no source within {tol_arcsec:g}\" of the "
                f"requested position in the joint result; {detail}")
        if len(near) > 1:
            raise RuntimeError(
                f"{source.label}: {len(near)} output sources within "
                f"{tol_arcsec:g}\" of the requested position; the members "
                f"cannot be told apart at this tolerance")
        index = near[0]
        if index in claimed:
            raise RuntimeError(
                f"{source.label}: matches an output source already claimed "
                f"by another member; two members are the same position")
        claimed.add(index)
        block = groups[index][1].reset_index(drop=True)
        blocks[source.label] = block
        complaints.extend(verify_group_shapes(block, source))

    unclaimed = [positions[i] for i in range(len(groups))
                 if i not in claimed]
    if unclaimed:
        raise RuntimeError(
            f"the joint result carries {len(unclaimed)} source(s) no member "
            f"claims, at {unclaimed}; the job did not fit the requested set")
    return blocks, complaints


# ------------------------------------
# Orchestration
# ------------------------------------
def _wait(poll_fn, interval=5, timeout=10800, on_update=None, done=None,
          max_poll_failures=5, queue_timeout=QUEUE_TIMEOUT):
    """Poll poll_fn -> (phase, payload) until done, or raise on timeout.

    Two budgets, because queueing and running fail differently. The
    timeout bounds how long a job may RUN; queue_timeout bounds how long
    it may wait to start. A single wall-clock budget makes the client's
    patience depend on how many jobs the caller submitted -- the tail of
    a deep queue would be killed for the service being busy, discarding
    an extraction that had not begun and would have completed.

    Transient poll failures are tolerated up to max_poll_failures in a
    row: a long job is polled hundreds of times, and one dropped HTTP
    read must not abandon an extraction that is still running
    server-side.
    """
    t0 = time.time()
    queued_since = None
    failures = 0
    while True:
        try:
            phase, payload = poll_fn()
            failures = 0
        except requests.RequestException as e:
            failures += 1
            if failures >= max_poll_failures:
                raise
            print(f"  [spherex] poll failed ({type(e).__name__}); "
                  f"retrying ({failures}/{max_poll_failures})")
            if time.time() - t0 > timeout:
                raise TimeoutError(f"job unreachable after {timeout}s") from e
            time.sleep(interval)
            continue
        if bool(phase) and phase.upper() in _QUEUED:
            if queued_since is None:
                queued_since = time.time()
            waited = time.time() - queued_since
            if queue_timeout and waited > queue_timeout:
                raise TimeoutError(
                    f"job still {phase!r} after {waited:.0f}s of queueing "
                    f"(queue_timeout {queue_timeout:g}s); it never started")
            # The run budget starts when the job does.
            t0 = time.time()
        else:
            queued_since = None
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
                            ff_session_id=None, poll=5, timeout=10800,
                            verbose=True, on_job_url=None,
                            queue_timeout=QUEUE_TIMEOUT):
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
    on_job_url : callable, optional
        Called with the UWS job URL as soon as it exists, before the
        wait, so a crashed run can be resumed against a job that is
        still executing server-side.

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
    df = _await_result(session, fjob, uws_url, poll=poll, timeout=timeout,
                       log=log, on_job_url=on_job_url,
                       queue_timeout=queue_timeout)
    if out_csv:
        df.to_csv(out_csv, index=False)
        log("wrote", out_csv)
    return df


def _await_result(session, fjob, uws_url, *, poll, timeout, log,
                  on_job_url=None, queue_timeout=QUEUE_TIMEOUT):
    """Wait for a submitted job and return its result table.

    Shared by the single-position and joint paths: the difference
    between them is entirely in what was submitted, so the wait, the
    failure reporting and the download are one implementation.
    """
    # The UWS jobUrl appears while Firefly is still EXECUTING -- key the
    # wait off the URL, not the phase.
    if not uws_url:
        _phase, uws_url = _wait(
            lambda: firefly_status(session, fjob)[:2],
            interval=poll, timeout=timeout, queue_timeout=queue_timeout,
            on_update=lambda p, _u: log("firefly", p),
            done=lambda _p, url: bool(url))
    if not uws_url:
        raise RuntimeError("Firefly never exposed a UWS jobUrl")
    log("UWS job", uws_url)
    if on_job_url is not None:
        # Handed out before the wait, so a caller can record it and poll
        # a job that outlives this process.
        on_job_url(uws_url)

    phase, results = _wait(
        lambda: uws_status(session, uws_url),
        interval=poll, timeout=timeout, queue_timeout=queue_timeout,
        on_update=lambda p, _r: log("uws", p))
    if phase.upper() != "COMPLETED":
        # The service's own explanation, when it offers one: a phase alone
        # cannot say whether the request was rejected or the service was
        # overloaded, and those call for opposite responses.
        detail = uws_error_detail(session, uws_url)
        raise RuntimeError(
            f"UWS job ended in phase {phase} (not COMPLETED)"
            + (f" -- service says: {detail}" if detail else "")
            + f" [job {uws_url}]")
    if not results:
        raise RuntimeError("UWS job COMPLETED but exposed no results")
    if len(results) > 1:
        # Only the first is read. Saying so keeps a quietly-dropped
        # result visible; split_group_table refuses on the sources it
        # would have carried.
        log(f"WARNING job exposed {len(results)} results; reading the first "
            f"({[r.get('id') for r in results]})")

    df = read_result_table(session, results[0]["href"])
    log(f"got table: {len(df)} rows, {len(df.columns)} cols")
    return df


def fetch_group_spectrophotometry(sources, bkg_region_size=15,
                                  mjd_range=None, session=None,
                                  ff_session_id=None, poll=5, timeout=10800,
                                  verbose=True, on_job_url=None,
                                  tol_arcsec=1.0,
                                  queue_timeout=QUEUE_TIMEOUT):
    """End-to-end joint extraction: upload -> submit -> wait -> split.

    Every member is fit simultaneously, so each member's flux depends on
    which other members were present. That is the point -- a companion
    inside the PSF absorbs light that would otherwise be attributed to
    its neighbor -- and it is also why the group belongs in the
    extraction's identity rather than being run-time detail.

    Parameters
    ----------
    sources : list of GroupSource
        Members to fit together, at most GROUP_SOURCES_MAX.
    bkg_region_size : float
        Background estimation region, pixels; one value for the job.
        [default: 15]
    mjd_range : (float, float), optional
        Restrict to visits in this MJD window; one window for the job.
    session : requests.Session, optional
        Session to reuse. [default: a fresh make_session()]
    ff_session_id : str, optional
        Explicit Firefly session id.
    poll : float
        Poll interval, seconds. [default: 5]
    timeout : float
        Give up after this many seconds. [default: 10800]
    verbose : bool
        Print progress. [default: True]
    on_job_url : callable, optional
        Called with the UWS job URL as soon as it exists, before the
        wait, so a crashed run can be resumed against a job that is
        still executing server-side.
    tol_arcsec : float
        Position match radius for attributing rows to members.
        [default: 1.0]

    Returns
    -------
    blocks : dict
        label -> that member's per-visit table.
    frame : pd.DataFrame
        The joint table verbatim, every member included.
    complaints : list of str
        Shape-echo disagreements; empty when every echo matched.
    """
    session = session or make_session()

    def log(*args):
        if verbose:
            print("[spherex]", *args)

    fjob, uws_url = submit_firefly_group(
        session, sources, bkg_region_size=bkg_region_size,
        mjd_range=mjd_range, ff_session_id=ff_session_id)
    log(f"submitted {len(sources)} source(s); firefly job {fjob}")
    frame = _await_result(session, fjob, uws_url, poll=poll, timeout=timeout,
                          log=log, on_job_url=on_job_url,
                          queue_timeout=queue_timeout)
    blocks, complaints = split_group_table(frame, sources,
                                           tol_arcsec=tol_arcsec)
    log(f"split into {len(blocks)} source table(s): "
        + ", ".join(f"{k} ({len(v)} rows)" for k, v in blocks.items()))
    return blocks, frame, complaints


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

    Gated: a direct POST returns 403 for guests. With a token this is the
    one-hop path, skipping Firefly entirely.
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
PRETAG_TABLE_NAME = "table_photometry.csv"   # untagged; never renamed

# Leads a joint extraction's tag. A solo glob cannot match it, which is
# what lets both live in one directory without a consumer picking the
# wrong one by accident.
GROUP_TAG_PREFIX = "joint"


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


def _member_payload(source: GroupSource) -> dict:
    """One member's contribution to a joint extraction's identity.

    Position and shape only. Role, output directory and name are
    bookkeeping -- they change nothing the tool computes, and two runs
    that differ only in which member is called science must reuse one
    table rather than fetch it twice.
    """
    if source.model is None:
        model_payload: str | dict = "psf"
    else:
        model_payload = {
            "type": "sersic",
            "n": _canon(source.model.n),
            "axis_ratio": _canon(source.model.axis_ratio),
            "pa_deg": _canon(source.model.pa_deg),
            "reff_arcsec": _canon(source.model.reff_arcsec),
        }
    return {"ra_deg": _canon(source.ra_deg), "dec_deg": _canon(source.dec_deg),
            "model": model_payload}


def group_config_payload(sources: list[GroupSource],
                         bkg_region_size: float = 15,
                         mjd_range=None) -> dict:
    """The parameters defining a JOINT extraction, canonically normalized.

    The whole membership is in here, because a joint fit divides the
    light between its members: the same galaxy fit alone and fit beside
    a companion are two different measurements, and must not share a
    tag or be reused for one another. Members are sorted by position so
    the identity does not depend on upload order.
    """
    return {
        "joint": True,
        "members": sorted((_member_payload(s) for s in sources),
                          key=lambda m: (m["ra_deg"], m["dec_deg"])),
        "bkg_region_size_px": int(round(float(bkg_region_size))),
        "mjd_range": ([_canon(v) for v in sorted(float(x) for x in mjd_range)]
                      if mjd_range else None),
    }


def _tag_for(payload: dict, kind: str) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:6]
    return f"{kind}-{digest}"


def extraction_tag(model: Sersic | None, bkg_region_size: float = 15,
                   mjd_range=None) -> str:
    """Deterministic '<model>-<hash6>' tag naming one configuration."""
    return _tag_for(config_payload(model, bkg_region_size, mjd_range),
                    "psf" if model is None else "sersic")


def group_extraction_tag(sources: list[GroupSource],
                         bkg_region_size: float = 15,
                         mjd_range=None) -> str:
    """Deterministic 'joint-<model>-<hash6>' tag naming one joint job.

    The 'joint' token leads so a glob written for solo tables
    ('table_photometry.sersic-*.csv') cannot match a joint one. The two
    coexist in a directory and a consumer selects between them
    deliberately rather than by whichever the filesystem offers first.
    """
    kind = "psf" if sources[0].model is None else "sersic"
    return _tag_for(group_config_payload(sources, bkg_region_size, mjd_range),
                    f"{GROUP_TAG_PREFIX}-{kind}")


def _model_payload_from_record(model) -> dict | str | None:
    """Rebuild one source model from its recorded form."""
    if model in ("point", "psf"):
        return "psf"
    if isinstance(model, dict):
        try:
            return {"type": "sersic",
                    "n": _canon(model["n"]),
                    "axis_ratio": _canon(model["axis_ratio"]),
                    "pa_deg": _canon(model["pa_deg"]),
                    "reff_arcsec": _canon(model["reff_arcsec"])}
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _sidecar_payload(sidecar: dict) -> dict | None:
    """Rebuild a config payload from a table's provenance sidecar.

    A joint table's sidecar rebuilds the GROUP payload, so the reuse
    check compares like with like: a solo table can never satisfy a
    joint request, nor a joint table a request for a different
    membership.
    """
    joint = sidecar.get("joint")
    if isinstance(joint, dict):
        members = joint.get("members")
        if not isinstance(members, list) or not members:
            return None
        rebuilt = []
        for member in members:
            model_payload = _model_payload_from_record(
                (member or {}).get("model"))
            if model_payload is None:
                return None
            try:
                rebuilt.append({"ra_deg": _canon(member["ra_deg"]),
                                "dec_deg": _canon(member["dec_deg"]),
                                "model": model_payload})
            except (KeyError, TypeError, ValueError):
                return None
        try:
            mjd = sidecar.get("mjd_range")
            return {
                "joint": True,
                "members": sorted(rebuilt,
                                  key=lambda m: (m["ra_deg"], m["dec_deg"])),
                "bkg_region_size_px": int(round(float(
                    sidecar["bkg_region_size_px"]))),
                "mjd_range": ([_canon(v) for v in
                               sorted(float(x) for x in mjd)] if mjd else None),
            }
        except (KeyError, TypeError, ValueError):
            return None

    model_payload = _model_payload_from_record(sidecar.get("model"))
    if model_payload is None:
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


def _records_config(table_path: Path, payload: dict) -> bool:
    """True when the table's own sidecar records exactly this configuration.

    The sidecar is written AFTER the CSV, so a table without a readable one
    was never verified and never reached the manifest -- the filename alone,
    tag included, is not evidence of what was extracted.
    """
    sidecar_path = table_path.with_suffix(".provenance.json")
    if not sidecar_path.exists():
        return False
    try:
        recorded = json.loads(sidecar_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return False
    return _sidecar_payload(recorded) == payload


def _matching_existing_table(spherex_dir: Path, payload: dict,
                             tag: str) -> Path | None:
    """An already-fetched table for this configuration, if any.

    The tagged filename is checked first, then the untagged name. Either
    way the table's own provenance sidecar must exist and record this exact
    configuration -- a name is a claim, a sidecar is the evidence. Untagged
    tables are recognized in place and never renamed: their names are baked
    into roster entries and run provenance.
    """
    for candidate in (spherex_dir / f"table_photometry.{tag}.csv",
                      spherex_dir / PRETAG_TABLE_NAME):
        if candidate.exists() and _records_config(candidate, payload):
            return candidate
    return None


def find_table(spherex_dir: str | Path) -> Path | None:
    """The one raw table a SPHEREx directory vouches for, config unknown.

    _matching_existing_table answers "is THIS configuration already on
    disk"; this answers "which table is the canonical one here", for a
    caller that has no configuration to match against -- a figure, a
    report. The evidence is the same and so is the standard: a directory
    can also hold hand-named files predating this package, so only a
    table with a provenance sidecar beside it is a candidate. A name is
    a claim, a sidecar is the evidence.

    Tagged names rank ahead of the pre-tag bare one, and ties break on
    the sorted filename, so the choice never depends on directory order.

    Parameters
    ----------
    spherex_dir : str or Path
        <out-dir>/Photometry/SPHEREx.

    Returns
    -------
    table : Path or None
        The selected CSV, or None when the directory is absent or holds
        no sidecar-backed table.
    """
    spherex_dir = Path(spherex_dir)
    if not spherex_dir.is_dir():
        return None
    vouched = [path
               for path in sorted(spherex_dir.glob("table_photometry*.csv"))
               if path.with_suffix(".provenance.json").exists()]
    tagged = [path for path in vouched if path.name != PRETAG_TABLE_NAME]
    for candidates in (tagged, vouched):
        if candidates:
            return candidates[0]
    return None


def _index_extraction(spherex_dir: Path, tag: str, filename: str,
                      payload: dict, *, shape_origin=None, n_rows=None) -> None:
    """Record one extraction in the manifest (the tag decoder ring).

    The manifest is a regenerable convenience index; the per-table
    provenance sidecars remain authoritative. Written atomically
    (sibling, then replace) so an interrupted run cannot leave a torn
    index behind.
    """
    manifest_path = spherex_dir / MANIFEST_NAME
    manifest = {"kind": "spherex_extractions", "entries": {}}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
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
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding='utf-8')
    tmp.replace(manifest_path)


# ------------------------------------
# Provider-style wrapper
# ------------------------------------
def fetch(coord, *, out_dir, model: Sersic | None = None,
          bkg_region_size: float = 15, mjd_range: tuple | None = None,
          poll: float = 5, timeout: float = 10800,
          shape_origin: str | None = None,
          label: str | None = None,
          target_name: str | None = None) -> ProviderResult:
    """Fetch the raw SPHEREx table for ONE extraction configuration.

    Each distinct configuration (source model + background region + MJD
    window) owns a tagged table under Photometry/SPHEREx/, so PSF and
    Sersic extractions -- or different Sersic shapes -- coexist without
    manual renames. Re-requesting an existing configuration reuses its
    table (status ok, nothing fetched) when the table's own sidecar records
    that configuration; a table no sidecar vouches for is an error, not a
    reuse and not a fetch target. Move a table aside deliberately to force
    a re-fetch. Nothing on disk is ever overwritten or renamed (raw tables
    can be irreplaceable manual downloads, and existing names are baked
    into run provenance).

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
    label, target_name : str, optional
        Output stem and original name string, recorded in the sidecar's
        target block so it matches the catalog and measurement sidecars.
        The IRSA request itself takes only a position -- its name box is a
        resolver input, and sedphot resolves upstream -- so neither reaches
        the query.

    Returns
    -------
    result : ProviderResult
        status ok with meta.path and meta.tag on success (meta.reused when
        the configuration was already on disk); error with the manual
        recipe (and no partial file) otherwise.
    """
    spherex_dir = Path(out_dir) / "Photometry" / "SPHEREx"
    spherex_dir.mkdir(parents=True, exist_ok=True)
    if model is not None and model.reff_arcsec < SPHEREX_POINT_SOURCE_REFF:
        print(f"  [spherex] WARNING: reff {model.reff_arcsec:.2f}\" is below "
              f"the tool's {SPHEREX_POINT_SOURCE_REFF:g}\" point-source "
              f"threshold -- this job returns PSF photometry regardless of "
              f"the Sersic parameters (a psf-tag table of the same window "
              f"would be identical)")
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
                recorded = json.loads(sidecar_path.read_text(encoding='utf-8'))
            except (OSError, json.JSONDecodeError):
                pass
        manifest_path = spherex_dir / MANIFEST_NAME
        known = {}
        if manifest_path.exists():
            try:
                known = json.loads(
                    manifest_path.read_text(encoding='utf-8')).get("entries", {})
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
    if out_csv.exists():
        # The reuse check above already refused it, so its sidecar is
        # missing or records something else. Nothing on disk is ever
        # overwritten, and an unverified raw table is not a fetch target.
        message = (f"{out_csv.name} exists but no sidecar vouches for this "
                   f"configuration; move it aside deliberately to re-fetch")
        print(f"  [spherex] {message}")
        print(f"  [spherex] {MANUAL_RECIPE}")
        return ProviderResult(provider='spherex', status=STATUS_ERROR,
                              message=f"{message}; {MANUAL_RECIPE}")
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
        "target": {"name": target_name, "label": label,
                   "ra_deg": float(coord.ra.deg),
                   "dec_deg": float(coord.dec.deg)},
        "model": ("point" if model is None else
                  {"type": "sersic", "n": model.n, "axis_ratio": model.axis_ratio,
                   "pa_deg": model.pa_deg, "reff_arcsec": model.reff_arcsec,
                   "shape_origin": shape_origin,
                   **({"effectively_point_source": True}
                      if model.reff_arcsec < SPHEREX_POINT_SOURCE_REFF
                      else {})}),
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


# ------------------------------------
# Joint extraction, provider-style
# ------------------------------------
def _member_record(source: GroupSource) -> dict:
    """One member as the sidecar records it.

    Carries what the extraction depended on (position, shape) plus the
    bookkeeping that identifies it. _sidecar_payload reads the first
    half back to answer "is this configuration already on disk".
    """
    return {
        "label": source.label,
        "name": source.name,
        "role": source.role,
        "ra_deg": float(source.ra_deg),
        "dec_deg": float(source.dec_deg),
        "model": ("point" if source.model is None else
                  {"type": "sersic", "n": source.model.n,
                   "axis_ratio": source.model.axis_ratio,
                   "pa_deg": source.model.pa_deg,
                   "reff_arcsec": source.model.reff_arcsec,
                   "shape_origin": source.shape_origin}),
    }


def group_table_path(out_dir, tag: str) -> Path:
    """Where a science member's joint table lives."""
    return (Path(out_dir) / "Photometry" / "SPHEREx"
            / f"table_photometry.{tag}.csv")


def group_is_complete(sources: list[GroupSource], payload: dict,
                      tag: str) -> tuple[bool, list[str]]:
    """Whether every science member already carries this exact extraction.

    Returns
    -------
    complete : bool
        True when nothing needs fetching.
    missing : list of str
        Labels whose table is absent or vouched for by no sidecar
        recording this configuration.
    """
    missing = []
    for source in sources:
        if not source.is_science:
            continue
        path = group_table_path(source.out_dir, tag)
        if not (path.exists() and _records_config(path, payload)):
            missing.append(source.label)
    return (not missing), missing


def fetch_group(sources: list[GroupSource], *, group_id: str,
                group_dir: str | Path | None = None,
                bkg_region_size: float = 15,
                mjd_range: tuple | None = None,
                poll: float = 5, timeout: float = 10800,
                tol_arcsec: float = 1.0,
                queue_timeout: float = QUEUE_TIMEOUT,
                on_job_url=None) -> ProviderResult:
    """Run ONE joint extraction and write each science member its own table.

    The tool returns every member in a single table; sedphot splits it
    so each science member's directory holds exactly one table for this
    extraction, which is the shape every downstream consumer already
    reads. The membership is in the tag, so a galaxy measured alone and
    the same galaxy measured beside a companion are two extractions that
    coexist rather than one overwriting the other.

    Nothing on disk is overwritten or renamed. A member whose table
    already exists and is vouched for keeps it; a member whose table
    exists with no matching sidecar is reported and left alone.

    Parameters
    ----------
    sources : list of GroupSource
        Members to fit together. Science members need out_dir.
    group_id : str
        Group identity, recorded in provenance.
    group_dir : str or Path, optional
        Directory for the joint table verbatim and any ancillary
        member's spectrum. Without it an ancillary member's spectrum is
        computed and discarded, which is reported.
    bkg_region_size : float
        Background estimation region, pixels. [default: 15]
    mjd_range : (float, float), optional
        Restrict to visits in this MJD window.
    poll : float
        Poll interval, seconds. [default: 5]
    timeout : float
        Give up after this many seconds. [default: 10800]
    tol_arcsec : float
        Position match radius when attributing rows to members.
        [default: 1.0]
    queue_timeout : float
        Give up on a job that never starts. Separate from timeout so a
        deep queue -- normal operation when many jobs are in flight --
        cannot kill an extraction that has not begun.
        [default: QUEUE_TIMEOUT]
    on_job_url : callable, optional
        Called with the UWS job URL as soon as it exists.

    Returns
    -------
    result : ProviderResult
        ok with meta.tag, meta.paths (label -> table) and meta.reused;
        error with the manual recipe and no partial product otherwise.
    """
    science = [s for s in sources if s.is_science]
    if not science:
        return ProviderResult(provider='spherex', status=STATUS_ERROR,
                              message=f"group {group_id} has no science "
                                      f"member; nothing would be written")
    without_dir = [s.label for s in science if not s.out_dir]
    if without_dir:
        return ProviderResult(provider='spherex', status=STATUS_ERROR,
                              message=f"group {group_id}: science members "
                                      f"with no out_dir: {without_dir}")
    complaints = [c for s in sources if s.model is not None
                  for c in check_shape(s.model, where=s.label)]
    if complaints:
        # The tool does not validate an uploaded shape -- it attempts the
        # fit and returns a table that looks like any other -- so this is
        # the only place an unphysical member is refused.
        message = f"group {group_id}: " + "; ".join(complaints)
        print(f"  [spherex] {message}")
        return ProviderResult(provider='spherex', status=STATUS_ERROR,
                              message=message)

    payload = group_config_payload(sources, bkg_region_size, mjd_range)
    tag = group_extraction_tag(sources, bkg_region_size, mjd_range)
    complete, missing = group_is_complete(sources, payload, tag)
    if complete:
        print(f"  [spherex] group {group_id} extraction {tag} already on disk")
        return ProviderResult(
            provider='spherex', status=STATUS_OK,
            message=f"reusing joint extraction {tag} "
                    f"({len(science)} science member(s))",
            meta={'tag': tag, 'group_id': group_id, 'reused': True,
                  'paths': {s.label: str(group_table_path(s.out_dir, tag))
                            for s in science}})

    if group_dir is not None:
        Path(group_dir).mkdir(parents=True, exist_ok=True)
    ancillary = [s.label for s in sources if not s.is_science]
    if ancillary and group_dir is None:
        print(f"  [spherex] NOTE group {group_id} has ancillary member(s) "
              f"{ancillary} and no group_dir; their spectra are fit and "
              f"then discarded")

    try:
        blocks, joint, echo = fetch_group_spectrophotometry(
            sources, bkg_region_size=bkg_region_size, mjd_range=mjd_range,
            poll=poll, timeout=timeout, tol_arcsec=tol_arcsec,
            queue_timeout=queue_timeout, on_job_url=on_job_url)
    except Exception as e:
        print(f"  [spherex] group {group_id} fetch failed: "
              f"{type(e).__name__}: {e}")
        print(f"  [spherex] {MANUAL_RECIPE}")
        return ProviderResult(provider='spherex', status=STATUS_ERROR,
                              message=f"{type(e).__name__}: {e}; "
                                      f"{MANUAL_RECIPE}")

    if echo:
        # The tool used a shape other than the one requested, so every
        # member's flux is against an unknown model. The raw table is
        # kept unvouched-for -- a rerun costs the whole job -- but no
        # product is written under a name a consumer would trust.
        message = (f"group {group_id}: the tool's echoed shape disagrees "
                   f"with the request -- " + "; ".join(echo))
        print(f"  [spherex] {message}")
        if group_dir is not None:
            quarantine = (Path(group_dir)
                          / f"joint_photometry.{tag}.unverified.csv")
            joint.to_csv(quarantine, index=False)
            print(f"  [spherex] raw table kept for diagnosis (no sidecar, "
                  f"not a product): {quarantine}")
        return ProviderResult(provider='spherex', status=STATUS_ERROR,
                              message=message)

    written, reused, refused = {}, [], []
    for source in science:
        out_csv = group_table_path(source.out_dir, tag)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        if out_csv.exists():
            if _records_config(out_csv, payload):
                reused.append(source.label)
                written[source.label] = str(out_csv)
            else:
                refused.append(source.label)
            continue
        block = blocks[source.label]
        block.to_csv(out_csv, index=False)
        write_sidecar(out_csv, {
            "kind": "spherex_spectrophotometry_joint",
            "extraction_tag": tag,
            "target": {"name": source.name, "label": source.label,
                       "ra_deg": float(source.ra_deg),
                       "dec_deg": float(source.dec_deg)},
            "model": _member_record(source)["model"],
            "joint": {"group_id": group_id,
                      "role": source.role,
                      "n_members": len(sources),
                      "members": [_member_record(s) for s in sources]},
            "bkg_region_size_px": int(round(float(bkg_region_size))),
            "mjd_range": list(mjd_range) if mjd_range else None,
            "service": UWS_BASE,
            "n_rows": len(block),
            "columns": list(block.columns),
        })
        _index_extraction(out_csv.parent, tag, out_csv.name, payload,
                          shape_origin=source.shape_origin, n_rows=len(block))
        written[source.label] = str(out_csv)

    if group_dir is not None:
        joint_csv = Path(group_dir) / f"joint_photometry.{tag}.csv"
        if joint_csv.exists():
            print(f"  [spherex] {joint_csv.name} exists; leaving it in place")
        else:
            joint.to_csv(joint_csv, index=False)
            write_sidecar(joint_csv, {
                "kind": "spherex_spectrophotometry_joint_raw",
                "extraction_tag": tag,
                "joint": {"group_id": group_id, "n_members": len(sources),
                          "members": [_member_record(s) for s in sources]},
                "bkg_region_size_px": int(round(float(bkg_region_size))),
                "mjd_range": list(mjd_range) if mjd_range else None,
                "service": UWS_BASE,
                "n_rows": len(joint),
                "columns": list(joint.columns),
            })

    if refused:
        message = (f"group {group_id}: {refused} already carry "
                   f"table_photometry.{tag}.csv that no sidecar vouches for; "
                   f"move those aside deliberately. The rest were written.")
        print(f"  [spherex] {message}")
        return ProviderResult(provider='spherex', status=STATUS_ERROR,
                              message=message,
                              meta={'tag': tag, 'group_id': group_id,
                                    'paths': written, 'refused': refused})
    return ProviderResult(
        provider='spherex', status=STATUS_OK,
        message=f"{len(joint)} visit-channel rows across {len(sources)} "
                f"member(s) -> {len(written)} science table(s) ({tag})"
                + (f"; {len(reused)} already on disk" if reused else ""),
        meta={'tag': tag, 'group_id': group_id, 'paths': written,
              'reused': reused, 'n_members': len(sources)})
