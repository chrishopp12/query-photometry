"""
pipeline.py

Retrieval and Measurement Drivers
---------------------------------------------------------
Orchestration only: resolve the target once, run the requested providers,
assemble the schema table, and write products + provenance. No science lives
here -- providers and the measurement engine own their own behavior.

Data products (under <out_dir>/Photometry/):
    <label>_catalog.csv               combined catalog photometry
    <label>_catalog.provenance.json   query provenance sidecar
    coverage_report.json              per-provider status for the run

Requirements:
    pandas, astropy
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from astropy.coordinates import SkyCoord

from .catalogs import CATALOG_PROVIDERS
from .provenance import write_sidecar
from .results import (
    STATUS_ERROR,
    ProviderResult,
    print_coverage_summary,
    write_coverage_report,
)
from .schema import rows_to_frame


# ------------------------------------
# Catalog driver
# ------------------------------------
def run_catalogs(
        coord: SkyCoord,
        label: str,
        out_dir: str | Path,
        *,
        instruments: list[str],
        radius_arcsec: float = 2.0,
        legacy_dr: str = 'dr10',
        target_name: str | None = None,
) -> pd.DataFrame:
    """Query the requested catalog providers and write the combined table.

    Parameters
    ----------
    coord : SkyCoord
        Resolved target position.
    label : str
        Output stem (sanitized name or J-name).
    out_dir : str or Path
        Galaxy directory; products land in <out_dir>/Photometry/.
    instruments : list[str]
        Provider names from catalogs.CATALOG_PROVIDERS.
    radius_arcsec : float
        Starting search radius per provider. [default: 2.0]
    legacy_dr : str
        Legacy data release ('dr10' or 'dr9'). [default: 'dr10']
    target_name : str, optional
        Original name string, recorded in the sidecar.

    Returns
    -------
    catalog_df : pd.DataFrame
        The combined photometry table (also written to CSV when non-empty).
    """
    unknown = [inst for inst in instruments if inst not in CATALOG_PROVIDERS]
    if unknown:
        raise ValueError(f"unknown catalog provider(s) {unknown}; "
                         f"known: {sorted(CATALOG_PROVIDERS)}")

    phot_dir = Path(out_dir) / "Photometry"
    phot_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nTarget: RA={coord.ra.deg:.6f}, Dec={coord.dec.deg:+.6f}  "
          f"(search radius={radius_arcsec:.1f}\")\n")

    results: list[ProviderResult] = []
    for name in instruments:
        print(f"=== {name} ===")
        provider = CATALOG_PROVIDERS[name]
        try:
            if name == 'legacy':
                result = provider(coord, radius_arcsec, dr=legacy_dr)
            else:
                result = provider(coord, radius_arcsec)
        except Exception as e:
            # Providers handle their own expected failures; this catches the
            # unexpected so one broken service never kills the run.
            result = ProviderResult(provider=name, status=STATUS_ERROR,
                                    message=f"{type(e).__name__}: {e}")
        results.append(result)
        print()

    catalog_df = rows_to_frame([row for r in results for row in r.rows])

    print("Provider summary:")
    print_coverage_summary(results)
    write_coverage_report(results, phot_dir / "coverage_report.json")

    if catalog_df.empty:
        print("\nNo photometry retrieved from any catalog.")
        return catalog_df

    out_csv = phot_dir / f"{label}_catalog.csv"
    catalog_df.to_csv(out_csv, index=False)
    write_sidecar(out_csv, {
        "kind": "catalog_photometry",
        "target": {"name": target_name, "label": label,
                   "ra_deg": float(coord.ra.deg), "dec_deg": float(coord.dec.deg)},
        "radius_arcsec": radius_arcsec,
        "instruments": instruments,
        "legacy_dr": legacy_dr if 'legacy' in instruments else None,
        "providers": {r.provider: {"status": r.status, "message": r.message, **r.meta}
                      for r in results},
    })
    print(f"\nSaved {len(catalog_df)} photometric points to: {out_csv}")
    print(catalog_df.to_string(index=False))

    return catalog_df
