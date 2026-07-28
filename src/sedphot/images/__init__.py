"""
images

Image Provider Registry
---------------------------------------------------------
One flat module per archive, each exposing

    fetch(coord, *, bands, size_arcsec, cache_dir, **options)
        -> list[ImageProduct] | ProviderResult

Downloads are cached under the target's Photometry/<Instrument>/ directory
and reused on re-runs. Registry order is the --all run order.

A provider never raises past its own boundary: it returns a ProviderResult
reporting no_coverage (the footprint excludes the target) or error (service
or parse failure) -- see results.ProviderResult for the status vocabulary.
Image providers never report no_match.
"""
from __future__ import annotations

from . import cfht, hst, legacy, panstarrs, sdss

IMAGE_PROVIDERS = {
    'sdss': sdss.fetch,
    'panstarrs': panstarrs.fetch,
    'legacy': legacy.fetch,
    'cfht': cfht.fetch,
    'hst': hst.fetch,
}
