"""
images

Image Provider Registry
---------------------------------------------------------
One flat module per archive, each exposing

    fetch(coord, *, bands, size_arcsec, cache_dir, **options)
        -> list[ImageProduct] | ProviderResult

Downloads are cached under the target's Photometry/<Instrument>/ directory
and reused on re-runs. A provider returns a ProviderResult (no_coverage /
error) instead of raising when the archive has nothing here.
"""
from __future__ import annotations

from . import cfht, legacy, panstarrs, sdss

IMAGE_PROVIDERS = {
    'sdss': sdss.fetch,
    'panstarrs': panstarrs.fetch,
    'legacy': legacy.fetch,
    'cfht': cfht.fetch,
}
