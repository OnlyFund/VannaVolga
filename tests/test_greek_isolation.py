"""A vol Greek that cannot be computed is unavailable on its line (with the
reason), never the cause of a line not pricing: the price and the other
Greeks are exactly as before. Delta still has to compute (hedges need it)."""
from datetime import date

import numpy as np
import pytest

from app import pricer
from app.market_set import MarketSet
from Greeks.smile_greeks import SmileGreeks
from golden.app_harness import load_markets
from market_data import Modifications

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
ROW = dict(currency_pair="EURUSD", expiry="2027-01-15", call_put="CALL", strike="1.15",
           buy_sell="BUY", notional="10m", premium=0.0, premium_ccy="TERM")


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def boom(*args, **kwargs):
    raise ValueError("VV price is outside the implied-volatility bracket")


def test_a_failing_sega_leaves_the_line_priced_and_says_why(markets, monkeypatch):
    (normal,), _ = pricer.price_lines(markets, [ROW], MODS)
    assert "greek_problems" not in normal                 # unchanged when nothing fails
    monkeypatch.setattr(SmileGreeks, "sega", boom)
    pricer._GREEK_MEMO.clear()     # the engine changed: nothing priced before applies
    (line,), errors = pricer.price_lines(markets, [ROW], MODS)
    assert errors == []
    assert line["sega"] is None and line["fixed_sega"] is None      # both families blank
    assert "smile sega failed" in line["greek_problems"]["sega"]
    for key in ("mtm_term", "vol_pct", "delta", "gamma", "vega", "vanna", "volga", "rega",
                "theta", "fixed_delta", "fixed_vega"):
        assert line[key] == normal[key], key


def test_totals_skip_the_unavailable_greek(markets, monkeypatch):
    monkeypatch.setattr(SmileGreeks, "sega", boom)
    rows = [ROW, dict(ROW, strike="1.10", call_put="PUT")]
    results, _ = pricer.price_lines(markets, rows, MODS)
    totals = pricer.aggregate_by_pair(results).set_index("currency_pair")
    assert np.isnan(totals.at["EURUSD", "sega"])                     # no partial sum shown
    assert totals.at["EURUSD", "vega"] == pytest.approx(sum(r["vega"] for r in results))


def test_a_digital_carries_its_legs_problem(markets, monkeypatch):
    monkeypatch.setattr(SmileGreeks, "sega", boom)
    (line,), errors = pricer.price_lines(markets, [dict(ROW, option_type="digital")], MODS)
    assert errors == [] and line["sega"] is None and "sega" in line["greek_problems"]


def test_delta_is_still_required(markets, monkeypatch):
    monkeypatch.setattr(SmileGreeks, "delta", boom)
    results, errors = pricer.price_lines(markets, [ROW], MODS)
    assert results == [] and "implied-volatility bracket" in errors[0]


def test_the_ladder_shows_an_unavailable_greek_blank(markets, monkeypatch):
    from app.reports.spot_ladder import calculate_spot_ladder
    monkeypatch.setattr(SmileGreeks, "sega", boom)
    context = markets.market("EURUSD")
    s = context.settings
    from app.trades.model import FX_OPTION, Trade
    trade = Trade(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=VALUATION,
                  expiry=date(2027, 1, 15), buy_sell="BUY", notional=1e6, strike=1.15,
                  call_put="CALL", premium=0.0, premium_ccy="USD").normalized()
    ladder, errors = calculate_spot_ladder(
        [(0, trade.to_pricer_row())], [context.spot], VALUATION, context.market_for_expiry,
        context.resolve_market_expiry, s.pricing_model, s.atm_type, s.delta_base, MODS,
        mode="reference", scenario_workers=1)
    assert errors == [] and np.isnan(ladder[0]["sega"]) and np.isfinite(ladder[0]["vega"])
