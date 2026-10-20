"""Standalone spot/forward deals valued at their own delivery date: the
outright is the forward curve at the value date (linear between quoted
deliveries), MTM is discounted on the term curve to that date, and a deal
delivering on an option's delivery date values exactly as the hedge forward on
that option expiry. Settlement P&L, term-entered amounts, deals inside spot or
settling tomorrow, deals beyond the vol range and the Risk ladder."""
from datetime import date

import pandas as pd
import pytest

from app import pricer
from app.market_set import MarketSet
from app.reports.spot_ladder import calculate_spot_ladder
from app.trades import service
from app.trades.model import FX_FORWARD, Trade
from golden.app_harness import load_markets
from market_data import Modifications

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
SPOT, DIVISOR = 1.1213, 10000.0          # EURUSD in MARKETS_20261015


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def deal(value_date, buy_sell="BUY", notional=10e6, rate=1.13, amount_ccy=None, trade_id="TR-1"):
    return Trade(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=VALUATION,
                 expiry=None, value_date=value_date, buy_sell=buy_sell, notional=notional,
                 contract_rate=rate, amount_ccy=amount_ccy, trade_id=trade_id).normalized()


def priced(markets, trades, extras=()):
    results, forwards, errors = service.price_trades(markets, trades, MODS, extras=extras)
    assert errors == []
    return {row["trade_id"]: row for row in forwards}


def test_on_an_options_delivery_date_it_values_as_that_hedge(markets):
    hedge = Trade(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=VALUATION,
                  expiry=date(2027, 1, 15), buy_sell="SELL", notional=10e6,
                  contract_rate=1.13, trade_id="H").normalized()
    same = deal(date(2027, 1, 20), buy_sell="SELL", trade_id="D")   # delivery of 15-Jan
    rows = priced(markets, [hedge, same], extras=("rho", "phi"))
    for key in ("mtm_term", "mtm_base", "delta_offset", "spot_delta", "fwd_delta", "theta",
                "rho", "phi"):
        assert rows["D"][key] == pytest.approx(rows["H"][key], rel=1e-12, abs=1e-9), key


def test_a_broken_date_takes_the_linear_outright_and_simple_discounting(markets):
    # 2027-03-05 lies between 3M (20-Jan, 44.810, 4.3144%) and 6M (19-Apr, 88.805, 4.5242%).
    t = (date(2027, 3, 5) - date(2027, 1, 20)).days / (date(2027, 4, 19) - date(2027, 1, 20)).days
    points = 44.810 + t * (88.805 - 44.810)
    rate = 0.043144 + t * (0.045242 - 0.043144)
    outright = SPOT + points / DIVISOR
    discount = 1.0 / (1.0 + rate * (date(2027, 3, 5) - VALUATION).days / 360.0)   # USD ACT/360
    row = priced(markets, [deal(date(2027, 3, 5))])["TR-1"]
    assert row["outright"] == pytest.approx(outright, abs=2e-7)
    assert row["settlement_term"] == pytest.approx(10e6 * (outright - 1.13), rel=1e-5)
    assert row["mtm_term"] == pytest.approx(10e6 * (outright - 1.13) * discount, rel=1e-5)
    assert row["mtm_base"] == pytest.approx(row["mtm_term"] / SPOT, rel=1e-12)
    assert row["settlement_base"] == pytest.approx(row["settlement_term"] / outright, rel=1e-12)
    # Spot delta: the amount at the term-discounted forward over spot; forward delta: the amount.
    assert row["fwd_delta"] == 10e6
    assert row["spot_delta"] == pytest.approx(10e6 * discount * outright / SPOT, rel=1e-5)
    assert row["forward_note"] is None and row["value_date"] == date(2027, 3, 5)


def test_a_term_entered_deal_values_as_its_base_equivalent(markets):
    base = deal(date(2027, 3, 5), buy_sell="SELL", notional=10e6, trade_id="B")
    term = deal(date(2027, 3, 5), buy_sell="BUY", notional=11.3e6, amount_ccy="USD", trade_id="T")
    rows = priced(markets, [base, term])
    for key in ("mtm_term", "mtm_base", "delta_offset", "theta", "settlement_term"):
        assert rows["T"][key] == pytest.approx(rows["B"][key], rel=1e-12), key
    assert rows["T"]["base_amount"] == pytest.approx(-10e6)


def test_beyond_the_vol_range_a_deal_still_values(markets):
    last_vol = max(pd.Timestamp(e) for e in markets.market("EURUSD").vol_rows["expiry"])
    last_fwd = max(pd.to_datetime(markets.market("EURUSD").forward_rows["delivery_date"]))
    if last_fwd <= last_vol + pd.Timedelta(days=3):
        pytest.skip("this market's forward curve ends with its vols")
    row = priced(markets, [deal(last_fwd.date())])["TR-1"]
    assert row["mtm_term"] != 0.0


def test_inside_spot_it_values_at_spot_and_says_so(markets):
    # Booked two days ago at spot: its value date (16-Oct) is before today's spot date.
    row = priced(markets, [deal(date(2026, 10, 16))])["TR-1"]
    assert row["outright"] == SPOT
    assert "inside spot" in row["forward_note"] and "no TN/SN points" in row["forward_note"]


def test_a_deal_settling_before_the_next_business_day_loses_its_mtm_as_theta(markets):
    row = priced(markets, [deal(date(2026, 10, 16))])["TR-1"]     # tomorrow
    assert row["theta"] == pytest.approx(-row["mtm_term"])        # USD term: theta in USD


def test_theta_is_the_next_days_curve_at_the_fixed_delivery(markets):
    from app.curve_roll import next_business_day
    from pricing.fx_forward import FxForwardPosition
    context = markets.market("EURUSD")
    tomorrow = next_business_day("EURUSD", VALUATION)
    position = FxForwardPosition("EURUSD", date(2027, 3, 5), 10e6, 1.13, date(2027, 3, 5))
    today = position.value(context.market_for_delivery(date(2027, 3, 5)), VALUATION)
    later = position.value(context.market_for_delivery(date(2027, 3, 5), as_of=tomorrow),
                           tomorrow)
    row = priced(markets, [deal(date(2027, 3, 5))])["TR-1"]
    assert row["theta"] == pytest.approx(later["mtm_term"] - today["mtm_term"], rel=1e-12)
    assert row["theta"] != 0.0


def test_the_risk_ladder_values_deals_at_their_delivery(markets):
    trade = deal(date(2027, 3, 5))
    context = markets.market("EURUSD")
    s = context.settings
    spots = [SPOT * 0.99, SPOT, SPOT * 1.01]
    positions = [{"currency_pair": "EURUSD", "expiry": trade.expiry,
                  "base_amount": trade.signed_base_amount(), "contract_rate": 1.13,
                  "value_date": trade.value_date}]
    ladder, errors = calculate_spot_ladder(
        [], spots, VALUATION, context.market_for_expiry, context.resolve_market_expiry,
        s.pricing_model, s.atm_type, s.delta_base, MODS, mode="optimized",
        forward_positions=positions, scenario_workers=1,
        market_for_delivery=context.market_for_delivery)
    assert errors == []
    row = priced(markets, [trade])["TR-1"]
    middle = ladder[1]
    assert middle["pnl_base"] == pytest.approx(row["mtm_base"], rel=1e-12)
    assert middle["delta"] == pytest.approx(row["delta_offset"], rel=1e-12)
    assert middle["theta"] == pytest.approx(row["theta"], rel=1e-12)
    for level, spot in zip(ladder, spots):
        moved = priced(markets.with_spots({"EURUSD": spot}), [trade])["TR-1"]
        assert level["pnl_base"] == pytest.approx(moved["mtm_base"], rel=1e-12)
    # Without the delivery-date market the ladder refuses rather than misdating it.
    _, refused = calculate_spot_ladder(
        [], spots[:1], VALUATION, context.market_for_expiry, context.resolve_market_expiry,
        s.pricing_model, s.atm_type, s.delta_base, MODS, mode="optimized",
        forward_positions=positions, scenario_workers=1)
    assert refused and "needs the market at its delivery date" in refused[0]


def test_before_the_valuation_date_is_refused(markets):
    with pytest.raises(ValueError, match="before the valuation date"):
        markets.market("EURUSD").market_for_delivery(date(2026, 10, 14))
