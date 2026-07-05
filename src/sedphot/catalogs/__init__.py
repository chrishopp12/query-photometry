"""
catalogs

Catalog Provider Registry
---------------------------------------------------------
One flat module per archive, each exposing

    query(coord, radius_arcsec, **options) -> ProviderResult

registered here by its CLI instrument token. Providers own their retry
behavior (radius expansion, transport backoff, VizieR mirrors) and never
raise past their boundary -- see results.ProviderResult for the status
vocabulary. Registry order is the --all run order (blue to red, HST last
because its per-filter downloads are the slow step).
"""
from __future__ import annotations

from . import allwise, galex, hst_hap, jplus, legacy, panstarrs, sdss

CATALOG_PROVIDERS = {
    'galex': galex.query,
    'sdss': sdss.query,
    'jplus': jplus.query,
    'panstarrs': panstarrs.query,
    'legacy': legacy.query,
    'allwise': allwise.query,
    'hst': hst_hap.query,
}
