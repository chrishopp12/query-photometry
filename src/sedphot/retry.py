"""
retry.py

Query Retry Helpers
---------------------------------------------------------
Three retry policies used by the providers:

    with_expanding_radius   no-match handling -- re-run a cone search with a
                            doubled radius
    retry_transient         transport handling -- exponential backoff around a
                            flaky HTTP/TAP call (CADC, IRSA)
    query_vizier_mirrors    VizieR-specific: fall back across mirror servers,
                            because a VizieR outage can present as empty
                            results rather than errors

Providers wrap their public query in with_expanding_radius and, where a
service is known to flap, wrap the transport call itself in retry_transient.
The policies compose without knowing about each other.

Requirements:
    astropy; astroquery for query_vizier_mirrors
"""
from __future__ import annotations

import time
from typing import Callable

from astropy.coordinates import SkyCoord


# ------------------------------------
# Constants
# ------------------------------------
EXPAND_FACTOR = 2.0    # multiply the search radius by this on each retry
MAX_RETRIES = 5        # max radius expansions before giving up on a catalog

TRANSIENT_ATTEMPTS = 3     # transport retries
TRANSIENT_BASE_DELAY = 2.0  # seconds; doubles each attempt

# Tried in order by query_vizier_mirrors; the primary CDS server comes first
# so the mirror only answers when the primary is empty or erroring.
VIZIER_MIRRORS = (
    "vizier.cds.unistra.fr",
    "vizier.cfa.harvard.edu",
)


# ------------------------------------
# Retry wrappers
# ------------------------------------
def with_expanding_radius(
        query_fn: Callable[[SkyCoord, float], list[dict]],
        coord: SkyCoord,
        radius_arcsec: float,
        label: str,
        *,
        max_retries: int = MAX_RETRIES,
        expand_factor: float = EXPAND_FACTOR,
) -> list[dict]:
    """Call query_fn(coord, radius) up to max_retries times, expanding the radius.

    Parameters
    ----------
    query_fn : callable
        Signature (coord, radius_arcsec) -> list[dict]; returns [] on no
        results (it must not raise for an empty match).
    coord : SkyCoord
        Target position.
    radius_arcsec : float
        Starting search radius.
    label : str
        Catalog name for logging.
    max_retries : int
        Attempts before giving up. [default: 5]
    expand_factor : float
        Radius multiplier per attempt. [default: 2.0]

    Returns
    -------
    rows : list[dict]
        Rows from the first successful attempt, or [] if all fail.
    """
    r = radius_arcsec
    for attempt in range(1, max_retries + 1):
        print(f"  [{label}] Attempt {attempt}/{max_retries}, radius={r:.1f}\"")
        rows = query_fn(coord, r)
        if rows:
            print(f"  [{label}] Found {len(rows)} match(es) at radius={r:.1f}\"")
            return rows
        print(f"  [{label}] No results. Expanding radius.")
        r *= expand_factor
    print(f"  [{label}] No results after {max_retries} attempts.")
    return []


def query_vizier_mirrors(query_fn: Callable[[str], object], label: str):
    """Run a VizieR query against each mirror until one returns rows.

    query_fn receives the mirror hostname and must construct its Vizier
    instance with Vizier(vizier_server=server, ...). Re-pointing
    astroquery.vizier.conf.server at runtime does NOT work: VizierClass
    captures the config value in a signature default when astroquery is
    imported, so even instances constructed after the re-point keep the
    original hostname (astroquery 0.4.x behavior) and a "mirror" attempt
    silently queries the dead host.

    An empty result from one mirror may be a genuine no-match, so the next
    mirror is asked before concluding; the cost is one redundant query in the
    true-no-match case, and the benefit is surviving a mirror outage that
    presents as empty results rather than errors.

    Parameters
    ----------
    query_fn : callable
        One-argument callable (mirror hostname) performing the VizieR
        query; its result is returned as-is when truthy (astroquery
        TableList).
    label : str
        Provider name for logging.

    Returns
    -------
    The first truthy query result, or None when every mirror returns nothing.
    """
    for server in VIZIER_MIRRORS:
        try:
            result = query_fn(server)
        except Exception as e:
            print(f"  [{label}] VizieR {server} error: {type(e).__name__}: {e}")
            continue
        if result:
            return result
        if server != VIZIER_MIRRORS[-1]:
            print(f"  [{label}] VizieR {server} returned nothing; trying a mirror")
    return None


def retry_transient(
        call: Callable[[], object],
        label: str,
        *,
        attempts: int = TRANSIENT_ATTEMPTS,
        base_delay: float = TRANSIENT_BASE_DELAY,
):
    """Run call() with exponential backoff on any exception.

    Parameters
    ----------
    call : callable
        Zero-argument callable performing the transport operation.
    label : str
        Service name for logging.
    attempts : int
        Total attempts before re-raising. [default: 3]
    base_delay : float
        First retry delay in seconds; doubles each attempt. [default: 2.0]

    Returns
    -------
    The return value of call().

    Raises
    ------
    The last exception if every attempt fails, or ValueError when
    attempts is not at least 1.
    """
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    delay = base_delay
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as e:
            last_error = e
            if attempt == attempts:
                break
            print(f"  [{label}] Transient failure ({type(e).__name__}: {e}); "
                  f"retrying in {delay:.0f}s ({attempt}/{attempts})")
            time.sleep(delay)
            delay *= 2.0
    raise last_error
