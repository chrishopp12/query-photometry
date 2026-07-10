"""query_vizier_mirrors: the fallback must actually re-point the server."""
import pytest

from sedphot.retry import VIZIER_MIRRORS, query_vizier_mirrors


def test_error_falls_through_to_mirror():
    seen = []

    def query(server):
        seen.append(server)
        if len(seen) == 1:
            raise ConnectionError("primary is down")
        return ["rows"]

    assert query_vizier_mirrors(query, "test") == ["rows"]
    assert seen == list(VIZIER_MIRRORS)


def test_empty_result_tries_next_mirror():
    seen = []

    def query(server):
        seen.append(server)
        return None if len(seen) == 1 else ["rows"]

    assert query_vizier_mirrors(query, "test") == ["rows"]
    assert seen == list(VIZIER_MIRRORS)


def test_primary_success_asks_no_mirror():
    seen = []

    def query(server):
        seen.append(server)
        return ["rows"]

    assert query_vizier_mirrors(query, "test") == ["rows"]
    assert seen == [VIZIER_MIRRORS[0]]


def test_all_empty_returns_none():
    assert query_vizier_mirrors(lambda server: None, "test") is None


def test_all_errors_return_none():
    def query(server):
        raise ConnectionError("everything is down")

    assert query_vizier_mirrors(query, "test") is None


def test_astroquery_vizier_server_kwarg_takes_effect():
    # The canary this design rests on: constructing with vizier_server=
    # must bind the instance to that host. (Re-pointing conf.server does
    # not -- the signature default is captured at import.)
    pytest.importorskip("astroquery")
    from astroquery.vizier import Vizier

    mirror = VIZIER_MIRRORS[-1]
    assert Vizier(vizier_server=mirror).VIZIER_SERVER == mirror
