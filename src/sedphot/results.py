"""
results.py

Provider Result Types and Coverage Report
---------------------------------------------------------
The two small containers every provider returns, and the coverage report the
drivers assemble from them. A provider never raises past its own boundary:
it reports ok / no_coverage / no_match / error and the run continues, so a
fetch-all over many archives degrades gracefully instead of dying on the
first service outage.

Data products:
    coverage_report.json    per-provider status written by the drivers

Requirements:
    (stdlib only)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


# Provider status vocabulary.
STATUS_OK = "ok"                    # rows returned
STATUS_NO_COVERAGE = "no_coverage"  # footprint does not include the target
STATUS_NO_MATCH = "no_match"        # covered, but nothing within the search radius
STATUS_ERROR = "error"              # service or parse failure (message has details)


# ------------------------------------
# Result containers
# ------------------------------------
@dataclass
class ProviderResult:
    """Outcome of one catalog provider query.

    Attributes
    ----------
    provider : str
        Registry name ('legacy', 'galex', ...).
    status : str
        One of ok | no_coverage | no_match | error.
    rows : list[dict]
        schema.make_row dicts; empty unless status == 'ok'.
    message : str
        Human-readable detail ("no GUVcat source within 32 arcsec").
    radius_used : float or None
        Final search radius in arcsec, if radius expansion was involved.
    meta : dict
        Provider-specific extras for the sidecar (Legacy: brickname, release).
    """
    provider: str
    status: str
    rows: list[dict] = field(default_factory=list)
    message: str = ""
    radius_used: float | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class ImageProduct:
    """One downloaded science image ready for measurement.

    Attributes
    ----------
    provider : str
        Registry name of the image provider.
    instrument : str
        Instrument label used in band names and directory layout ('Legacy').
    band : str
        Filter label within the instrument ('g').
    path : str
        Local science FITS.
    calib : str
        Calibration key for measure.calibrate ('nmgy' | 'photzp' | 'ps1' | 'hst').
    invvar_path : str or None
        Inverse-variance map when the archive serves one (Legacy bricks, HST wht).
    seeing_arcsec : float
        Approximate PSF FWHM; sizes detection kernels and PSF convolution.
    wave_um : float
        Effective wavelength for QA coloring and the SED plot.
    """
    provider: str
    instrument: str
    band: str
    path: str
    calib: str
    invvar_path: str | None = None
    seeing_arcsec: float = 1.0
    wave_um: float = float("nan")


# ------------------------------------
# Coverage report
# ------------------------------------
def write_coverage_report(results: list[ProviderResult], path: str | Path) -> Path:
    """Write the per-provider status summary consumed by humans and reruns."""
    report = {
        r.provider: {
            "status": r.status,
            "n_rows": len(r.rows),
            "message": r.message,
            "radius_used_arcsec": r.radius_used,
        }
        for r in results
    }
    path = Path(path)
    path.write_text(json.dumps(report, indent=2) + "\n")
    return path


def print_coverage_summary(results: list[ProviderResult]) -> None:
    """One line per provider, for the end-of-run console summary."""
    for r in results:
        detail = f" -- {r.message}" if r.message else ""
        print(f"  {r.provider:12s} {r.status:12s} {len(r.rows):3d} rows{detail}")
