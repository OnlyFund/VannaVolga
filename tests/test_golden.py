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
        # Rejected today: before the first pillar, expiring today, beyond the
        # last pillar, and a pair with no loaded market.
        line(k(1.00), "2026-10-19", "CALL", "BUY", "1m"),
        line(k(1.00), "2026-10-15", "CALL", "BUY", "1m"),
        line(k(1.00), "2027-12-15", "CALL", "BUY", "1m"),
        line("1.10", "1M", "CALL", "BUY", "1m", currency_pair="GBPUSD"),
    ]


PRICED_LINES = 9

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


def test_reference_portfolio_exercises_expected_paths():
    """Guard the fixture itself: priced lines price and rejected lines fail."""
    for name in PORTFOLIO_CASES:
        output = run_case(name)[1]
        assert len(output["lines"]) == PRICED_LINES, name
        assert len(output["errors"]) == 4, name
