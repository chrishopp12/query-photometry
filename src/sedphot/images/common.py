"""
common.py

Shared Image-Provider Helpers
---------------------------------------------------------
Guards shared by the cutout-fetching providers. Image caches are keyed
by band alone (never by size), so a changed request can quietly reuse a
file fetched under different settings; the checks here make that loud.

Requirements:
    numpy, astropy
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

# Tolerance for the size comparison: a cached cutout within this factor
# of the request counts as full size (requests round to whole pixels,
# and a stack near its footprint edge can legitimately trim a little).
SIZE_TOLERANCE = 0.98


def warn_undersized_cache(path: str | Path, size_arcsec: float,
                          label: str) -> bool:
    """Warn when a cached image spans less sky than the requested cutout.

    A re-run with a larger --cutout-size reuses the smaller cached file:
    the measurement engine then pads the missing area with no-data
    pixels, quietly shrinking the background reach and the curve-of-
    growth tail. Nothing is deleted or refetched here -- remove the file
    by hand to refetch at the larger size.

    Parameters
    ----------
    path : str or Path
        The cached FITS file about to be reused.
    size_arcsec : float
        The cutout width the current run asked for.
    label : str
        Provider tag for the message.

    Returns
    -------
    warned : bool
        True when the warning printed. Any read or parse failure
        returns False silently -- a diagnostic must not kill a fetch.
    """
    try:
        with fits.open(path) as hdul:
            hdu = next(h for h in hdul
                       if h.data is not None and h.data.ndim == 2)
            shape = hdu.data.shape
            scale = float(np.mean(proj_plane_pixel_scales(
                WCS(hdu.header).celestial))) * 3600.0
    except Exception:
        return False
    extent = min(shape) * scale
    if extent >= size_arcsec * SIZE_TOLERANCE:
        return False
    print(f"  [{label}] WARNING cached {Path(path).name} spans "
          f"{extent:.0f}\" but {size_arcsec:.0f}\" was requested -- "
          f"likely cached by an earlier smaller --cutout-size run; "
          f"delete the file to refetch at full size")
    return True
