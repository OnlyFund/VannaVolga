"""Golden tests: the Pricer's end-to-end numbers on frozen markets and trades.

Each case prices a reference portfolio through the app's own pricing path
(see ``golden/app_harness.py``) and compares every output against
``golden/expected/<case>.json``: prices, vols, all fixed and smile Greeks,
theta currency, per-line errors, delta-hedge forwards, the by-pair
aggregation and a spot ladder.

These tests pin current behaviour, not correctness. When a change is meant
to move numbers, regenerate and review the diff of the expected files::

    UPDATE_GOLDEN=1 python -m pytest tests/test_golden.py      (bash)
    $env:UPDATE_GOLDEN=1; python -m pytest tests/test_golden.py (PowerShell)
"""
import json
import math
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from golden.app_harness import AppHarness, load_markets
import app.ui.app as ui

pytestmark = [pytest.mark.filterwarnings("ignore"), pytest.mark.golden]

EXPECTED = Path(__file__).resolve().parent / "golden" / "expected"
UPDATE = os.environ.get("UPDATE_GOLDEN") == "1"
REL_TOL, ABS_TOL = 1e-7, 1e-6

BUNDLE = "MARKETS_20261015.json"
EVENTS = "EURUSD_20261015_events.json"


def reference_portfolio(pair, spot):
    """Trades covering every input path, plus lines the app must reject."""
    base, term = pair[:3], pair[3:]

    def k(moneyness):
        return ui.format_strike_for_pair(pair, spot * moneyness)

    def line(strike, expiry, call_put, buy_sell, notional, premium=0.0, premium_ccy=term,
             currency_pair=pair):
        return dict(currency_pair=currency_pair, strike=strike, expiry=expiry,
                    call_put=call_put, buy_sell=buy_sell, notional=notional,
                    premium=premium, premium_ccy=premium_ccy)

    return [
        # Priced lines: aliases, pillar tenors, broken dates in every input
        # format, both premium currencies, buys and sells.
        line("atm", "1M", "CALL", "BUY", "10m"),
        line("25d", "3M", "PUT", "BUY", "5m"),
        line("10d", "6M", "CALL", "SELL", "2.5m"),
        line(k(1.00), "2026-12-01", "CALL", "BUY", "7m", 70_000.0, base),
        line(k(0.98), "1W", "PUT", "SELL", "3m", round(3e6 * spot * 0.002, 2)),
        line("atmf", "2W", "PUT", "BUY", "500k"),
        line("25d", "1Y", "CALL", "BUY", "1m", 15_000.0, base),
        line(k(1.03), "21jan27", "CALL", "SELL", "4m"),
        line(k(0.95), "10/05/2027", "PUT", "BUY", "1,000,000"),
        # Before the first (1W) pillar: priced from the O/N-to-1W short end.
        line(k(1.00), "2026-10-19", "CALL", "BUY", "1m"),
        # Rejected: expiring today, beyond the last pillar, and a pair with
        # no loaded market.
        line(k(1.00), "2026-10-15", "CALL", "BUY", "1m"),
        line(k(1.00), "2027-12-15", "CALL", "BUY", "1m"),
        line("1.10", "1M", "CALL", "BUY", "1m", currency_pair="GBPUSD"),
    ]


#: Leading lines used for delta hedges and spot ladders (the short-end line
#: is priced but left out, so those outputs stay comparable).
PRICED_LINES = 9
#: Lines of the reference portfolio that price and that must be rejected.
EXPECTED_PRICED, EXPECTED_REJECTED = 10, 3

#: name -> (market file, pair, model, run spot ladders). Ladders are the slow
#: part (~10 s each under vv, ~6x that under vv-sabr), so they run on one
#: pip-quoted and one JPY-quoted pair only.
CASES = {
    "vv_EURUSD": (BUNDLE, "EURUSD", "vv", True),
    "vv_USDJPY": (BUNDLE, "USDJPY", "vv", True),
    "vv_USDMXN": (BUNDLE, "USDMXN", "vv", False),
    "vv-sabr_EURUSD": (BUNDLE, "EURUSD", "vv-sabr", False),
    "vv_EURUSD_events": (EVENTS, "EURUSD", "vv", False),
}
PORTFOLIO_CASES = ("vv_EURUSD", "vv_USDJPY", "vv_USDMXN")


def _plain(value):
    """Make app outputs JSON-comparable."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return pd.Timestamp(value).date().isoformat()
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


_RUNS = {}


def run_case(name):
    """Price one case once per session; the portfolio test reuses the lines."""
    if name not in _RUNS:
        market_file, pair, model, ladders = CASES[name]
        harness = AppHarness(load_markets(market_file)[pair], pricing_model=model)
        rows = reference_portfolio(pair, harness.spot)
        results, errors = harness.price_rows(rows)
        hedges, hedge_errors = harness.build_delta_hedges(
            pd.DataFrame(rows[:PRICED_LINES]), harness.mods
        )
        output = {
            "inputs": {"market_file": market_file, "pair": pair, "model": model,
                       "valuation_date": harness.valuation_date, "spot": harness.spot},
            "lines": results,
            "errors": errors,
            "aggregation": ui.aggregate_pricer_results_by_pair(results).to_dict("records"),
            "delta_hedges": hedges,
            "delta_hedge_errors": hedge_errors,
        }
        if ladders:
            output["spot_ladder"], output["spot_ladder_errors"] = harness.spot_ladder(
                rows[:PRICED_LINES], steps=2
            )
            (output["hedged_spot_ladder"],
             output["hedged_spot_ladder_errors"]) = harness.spot_ladder(
                rows[:PRICED_LINES], steps=1, forward_positions=hedges
            )
        _RUNS[name] = (results, _plain(output))
    return _RUNS[name]


def run_portfolio():
    """The three pairs' lines aggregated together, as one multi-pair book."""
    results = [line for name in PORTFOLIO_CASES for line in run_case(name)[0]]
    return _plain({
        "cases": list(PORTFOLIO_CASES),
        "aggregation": ui.aggregate_pricer_results_by_pair(results).to_dict("records"),
    })


def _differences(expected, actual, path="$"):
    if isinstance(expected, dict) and isinstance(actual, dict):
        out = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                out.append(f"{path}.{key}: missing")
            elif key not in expected:
                out.append(f"{path}.{key}: unexpected {actual[key]!r}")
            else:
                out += _differences(expected[key], actual[key], f"{path}.{key}")
        return out
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{path}: length {len(actual)} != expected {len(expected)}"]
        return [d for i, (e, a) in enumerate(zip(expected, actual))
                for d in _differences(e, a, f"{path}[{i}]")]
    if (isinstance(expected, (int, float)) and isinstance(actual, (int, float))
            and not isinstance(expected, bool) and not isinstance(actual, bool)):
        if math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL):
            return []
        return [f"{path}: {actual!r} != expected {expected!r}"]
    return [] if expected == actual else [f"{path}: {actual!r} != expected {expected!r}"]


def _check(name, actual):
    path = EXPECTED / f"{name}.json"
    if UPDATE:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(actual, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"golden file {path.name} regenerated")
    if not path.exists():
        pytest.fail(f"Missing {path.name}; run with UPDATE_GOLDEN=1 to create it")
    expected = json.loads(path.read_text(encoding="utf-8"))
    differences = _differences(expected, actual)
    assert not differences, (
        f"{len(differences)} golden differences in {path.name}:\n" + "\n".join(differences[:40])
    )


@pytest.mark.parametrize("name", list(CASES))
def test_pricer_matches_golden(name):
    _check(name, run_case(name)[1])


def test_multi_pair_aggregation_matches_golden():
    _check("portfolio_all_pairs", run_portfolio())


def test_market_set_prices_the_mixed_book_exactly_as_each_pair_alone():
    """Routing through a MarketSet must not move a single golden number."""
    from app import pricer
    from app.market_set import MarketSet

    payloads = load_markets(BUNDLE)
    harnesses = {CASES[name][1]: AppHarness(payloads[CASES[name][1]]) for name in PORTFOLIO_CASES}
    first = next(iter(harnesses.values()))
    markets = MarketSet.from_payloads(payloads, first.valuation_date)
    book = [line for pair, harness in harnesses.items()
            for line in reference_portfolio(pair, harness.spot)]
    results, errors = pricer.price_lines(markets, pd.DataFrame(book), first.mods)

    alone = [dict(line, row=None) for name in PORTFOLIO_CASES for line in run_case(name)[0]]
    assert _differences(_plain(alone), _plain([dict(line, row=None) for line in results])) == []
    assert len(errors) == EXPECTED_REJECTED * len(PORTFOLIO_CASES)
    assert sum("No market loaded for GBPUSD" in error for error in errors) == len(PORTFOLIO_CASES)
    _check("portfolio_all_pairs", _plain({
        "cases": list(PORTFOLIO_CASES),
        "aggregation": ui.aggregate_pricer_results_by_pair(results).to_dict("records"),
    }))


@pytest.mark.parametrize("name", PORTFOLIO_CASES)
def test_booked_trades_price_exactly_as_the_pricer(name, tmp_path):
    """The Portfolio tab's path (book, then price the booked trades through
    ``MarketSet.with_spots``) must match the Pricer to the last digit.

    Booking stores a resolved alias strike (atm, 25d ...) at dealing
    precision, so those lines are compared with the Pricer at the booked
    strike; lines typed with a numeric strike must match the golden file.
    """
    from app import pricer
    from app.market_set import MarketSet
    from app.trades import service
    from app.trades.store import SqliteTradeStore

    market_file, pair, model, _ = CASES[name]
    golden = json.loads((EXPECTED / f"{name}.json").read_text())["lines"]
    payloads = load_markets(market_file)
    harness = AppHarness(payloads[pair], pricing_model=model)
    rows = reference_portfolio(pair, harness.spot)
    priced_rows = [rows[line["row"] - 1] for line in golden]
    markets = MarketSet.from_payloads(payloads, harness.valuation_date)
    store = SqliteTradeStore(tmp_path / "trades.sqlite")
    booked = service.book_lines(store, markets, priced_rows, harness.mods,
                                trade_date=harness.valuation_date)
    assert booked.ok, booked.errors
    results, _, errors = service.price_trades(markets.with_spots({}), store.trades(), harness.mods)
    assert errors == []
    by_id = {line["trade_id"]: line for line in _plain(results)}
    as_booked = [by_id[booked.lines[number]] for number in range(1, len(priced_rows) + 1)]

    at_booked_strikes = [dict(row, strike=str(line["strike"]))
                         for row, line in zip(priced_rows, as_booked)]
    repriced, _ = pricer.price_lines(markets, pd.DataFrame(at_booked_strikes), harness.mods)
    strip = lambda line: {k: v for k, v in line.items() if k not in ("row", "trade_id", "book",
                                                                       "counterparty")}  # noqa: E731
    assert _differences([strip(line) for line in _plain(repriced)],
                        [strip(line) for line in as_booked]) == []

    def typed_as_booked(row, line):
        try:
            return float(row["strike"]) == line["strike"]
        except ValueError:                      # an alias: atm, 25d ...
            return False
    typed = [i for i, (row, line) in enumerate(zip(priced_rows, as_booked)) if typed_as_booked(row, line)]
    assert len(typed) >= 4
    assert _differences([strip(golden[i]) for i in typed], [strip(as_booked[i]) for i in typed]) == []


def test_reference_portfolio_exercises_expected_paths():
    """Guard the fixture itself: priced lines price and rejected lines fail."""
    for name in PORTFOLIO_CASES:
        output = run_case(name)[1]
        assert len(output["lines"]) == EXPECTED_PRICED, name
        assert len(output["errors"]) == EXPECTED_REJECTED, name
