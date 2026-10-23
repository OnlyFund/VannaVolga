"""The daily Bloomberg fetch's command line: the market, then today's pillar dates."""
import types

import pytest


@pytest.fixture
def fetch_module():
    pytest.importorskip("xbbg")
    import market_data.bloomberg_fetch as module
    return module


def _run(module, argv):
    calls = []
    module.main(argv, fetch=lambda day, vols: calls.append(("market", day, vols)),
                store_dates=lambda: calls.append(("dates",)), today="2026-10-23")
    return calls


def test_today_stores_the_market_then_the_dates(fetch_module):
    assert _run(fetch_module, []) == [("market", None, True), ("dates",)]
    assert _run(fetch_module, ["2026-10-23"])[-1] == ("dates",)


def test_a_past_date_or_no_dates_skips_them(fetch_module):
    assert _run(fetch_module, ["2026-10-21"]) == [("market", "2026-10-21", True)]
    assert _run(fetch_module, ["--no-dates"]) == [("market", None, True)]


def test_store_live_dates_runs_its_own_process_and_never_raises(fetch_module):
    seen = []

    def run(command, cwd):
        seen.append(command[1:])
        return types.SimpleNamespace(returncode=1)
    assert fetch_module.store_live_dates(run) == 1
    assert seen == [["-m", "market_data.live_dates", "--store"]]

    def broken(command, cwd):
        raise OSError("no python")
    assert fetch_module.store_live_dates(broken) == 1
