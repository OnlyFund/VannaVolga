"""Exotics in the Risk tab: left out unless "Include exotics" is ticked; then
digitals join the spot ladder (as their replicating vanilla legs), the strike
grid and the Greek buckets, and types not priced yet stay out."""
from datetime import date

import pandas as pd
import pytest

from app.market_set import MarketSet
from app.reports import risk_grid
from app.reports.spot_ladder import calculate_spot_ladder
from app.trades import service
from app.trades.model import FX_FORWARD, FX_OPTION, Trade
from app.ui.risk_block import ladder_rows, risk_options
from golden.app_harness import load_markets
from market_data import Modifications
from pricing.digital import leg_rows

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def option(trade_id, option_type=None, payout=None, call_put="CALL", buy_sell="BUY",
           pair="EURUSD", strike=1.15, premium=12_345.0):
    return Trade(product_type=FX_OPTION, currency_pair=pair, trade_date=VALUATION,
                 expiry=date(2027, 1, 15), buy_sell=buy_sell, notional=1e6, strike=strike,
                 call_put=call_put, premium=premium, premium_ccy=pair[3:],
                 option_type=option_type, payout_ccy=payout, trade_id=trade_id).normalized()


TRADES = [option("V1"), option("D1", "digital"), option("K1", "knock_out"),
          option("J1", pair="USDJPY", strike=150.0),
          Trade(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=VALUATION,
                expiry=date(2027, 1, 15), buy_sell="SELL", notional=5e5, contract_rate=1.16,
                trade_id="F1").normalized()]


def ids(trades):
    return [t.trade_id for t in trades]


def test_exotics_are_left_out_unless_included():
    options, left_out = risk_options(TRADES, "EURUSD")
    assert ids(options) == ["V1"] and ids(left_out) == ["D1", "K1"]
    options, left_out = risk_options(TRADES, "EURUSD", include_exotics=True)
    # Digitals are priced; a knock-out is not yet, so it stays out.
    assert ids(options) == ["V1", "D1"] and ids(left_out) == ["K1"]
    assert ids(risk_options(TRADES, "USDJPY", True)[0]) == ["J1"]


@pytest.mark.parametrize("call_put, buy_sell, payout, legs", [
    ("CALL", "BUY", "EUR", 3), ("PUT", "SELL", "USD", 2), ("PUT", "BUY", "EUR", 3)])
def test_a_digital_is_its_legs_in_the_ladder_with_its_premium_once(call_put, buy_sell, payout,
                                                                   legs):
    digital = option("D1", "digital", payout, call_put, buy_sell)
    rows = ladder_rows([option("V1"), digital])
    assert rows[0] == option("V1").to_pricer_row()
    legs_ = rows[1:]
    assert len(legs_) == legs and all(r["option_type"] == "vanilla" for r in legs_)
    priced = [r for r in legs_ if r["premium"]]
    # The premium rides once, on a leg on the trade's own side.
    assert len(priced) == 1 and priced[0]["premium"] == 12_345.0
    assert priced[0]["buy_sell"] == buy_sell
    assert legs_ == leg_rows(digital.to_pricer_row(), payout == "EUR")


def ladder(markets, rows, mode):
    context = markets.market("EURUSD")
    s = context.settings
    spots = [context.spot * (1 + step / 100) for step in (-1, 0, 1)]
    out, errors = calculate_spot_ladder(
        list(pd.DataFrame(rows).iterrows()), spots, VALUATION, context.market_for_expiry,
        context.resolve_market_expiry, s.pricing_model, s.atm_type, s.delta_base, MODS,
        mode=mode, scenario_workers=1)
    assert errors == []
    return out


@pytest.mark.parametrize("mode", ["reference", "optimized"])
@pytest.mark.parametrize("call_put, buy_sell, payout", [("CALL", "BUY", "EUR"),
                                                        ("PUT", "SELL", "USD")])
def test_the_ladder_at_spot_matches_the_pricer(markets, mode, call_put, buy_sell, payout):
    digital = option("D1", "digital", payout, call_put, buy_sell)
    (priced,), _, errors = service.price_trades(markets, [digital], MODS)
    assert errors == []
    centre = ladder(markets, ladder_rows([digital]), mode)[1]
    assert centre["pnl_base"] == pytest.approx(priced["pnl_base"], rel=1e-10)
    for key in ("delta", "fixed_delta", "gamma", "fixed_gamma", "vega", "fixed_vega", "theta",
                "fixed_theta", "vanna", "volga"):
        assert centre[key] == pytest.approx(priced[key], rel=1e-6, abs=1e-6), key


def test_digitals_join_the_grid_and_buckets_without_a_notional(markets):
    options, _ = risk_options(TRADES, "EURUSD", include_exotics=True)
    results, forwards, errors = service.price_trades(markets, options, MODS)
    assert errors == []
    lines, line_errors = risk_grid.risk_lines(markets, "EURUSD", options, results, forwards)
    assert line_errors == []
    by_id = {line.trade_id: line for line in lines}
    digital = next(r for r in results if r["trade_id"] == "D1")
    assert by_id["D1"].values["notional"] == 0.0 and by_id["V1"].values["notional"] == 1e6
    assert by_id["D1"].values["delta"] == digital["delta"]
    assert by_id["D1"].values["mtm_base"] == digital["mtm_base"]
