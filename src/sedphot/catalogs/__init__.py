"""
catalogs

Catalog Provider Registry
---------------------------------------------------------
One flat module per archive, each exposing

    query(coord, radius_arcsec, **options) -> ProviderResult

registered here by its CLI instrument token. Providers own their retry
behavior (radius expansion, transport backoff) and never raise past their
boundary -- see results.ProviderResult for the status vocabulary.
"""
from __future__ import annotations

from . import hst_hap, legacy, panstarrs

CATALOG_PROVIDERS = {
    'legacy': legacy.query,
    'panstarrs': panstarrs.query,
    'hst': hst_hap.query,
}
