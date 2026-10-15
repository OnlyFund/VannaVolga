"""Strike grid and Greek buckets: 5-delta buckets, proportional allocation."""
from datetime import date

import pytest

from app.reports.risk_grid import (
    ATM, TOTAL, RiskLine, delta_buckets, greek_buckets, linear_weights, strike_grid,
)

TENORS = ["1M", "3M"]
DATES = [date(2026, 11, 17), date(2027, 1, 15)]


def line(call_delta, expiry=DATES[0], **values):
    return RiskLine("T", expiry, call_delta, values or {"vega": 100.0})


def test_five_delta_buckets_run_from_low_to_high_strike():
    buckets = delta_buckets()
    labels = [label for label, _ in buckets]
    assert labels[:2] == ["5P", "10P"] and labels[9] == ATM and labels[-2:] == ["10C", "5C"]
    assert len(labels) == 19
    assert dict(buckets)["25P"] == pytest.approx(0.75) and dict(buckets)["25C"] == pytest.approx(0.25)
    assert [label for label, _ in delta_buckets(25)] == ["25P", ATM, "25C"]
    with pytest.raises(ValueError):
        delta_buckets(7)


def test_linear_weights_split_between_neighbours_and_clamp():
    nodes = [0.95, 0.5, 0.05]                        # descending, like call deltas
    assert linear_weights(0.5, nodes) == [(1, 1.0)]
    assert sorted(linear_weights(0.275, nodes)) == [(1, pytest.approx(0.5)), (2, pytest.approx(0.5))]
    assert linear_weights(0.99, nodes) == [(0, 1.0)]
    assert linear_weights(0.01, nodes) == [(2, 1.0)]


def test_option_on_a_bucket_lands_in_one_cell():
    grid = strike_grid([line(0.25)], TENORS, DATES, "vega")
    assert grid.at["1M", "25C"] == pytest.approx(100.0)
    assert grid.drop(index=TOTAL).drop(columns=TOTAL).values.sum() == pytest.approx(100.0)


def test_intermediate_delta_and_expiry_are_split_proportionally():
    # 22.5 call delta: half 20C, half 25C. Expiry 1/4 of the way from 1M to 3M.
    quarter = date.fromordinal(DATES[0].toordinal() + (DATES[1].toordinal() - DATES[0].toordinal()) // 4)
    grid = strike_grid([line(0.225, quarter)], TENORS, DATES, "vega")
    share = (quarter.toordinal() - DATES[0].toordinal()) / (DATES[1].toordinal() - DATES[0].toordinal())
    assert grid.at["1M", "20C"] == pytest.approx(50.0 * (1 - share))
    assert grid.at["3M", "25C"] == pytest.approx(50.0 * share)
    assert grid.at[TOTAL, TOTAL] == pytest.approx(100.0)


def test_totals_equal_the_portfolio_and_hedges_go_to_the_last_call_bucket():
    lines = [line(0.62, vega=40.0, delta=1e6), line(0.08, DATES[1], vega=-15.0, delta=-2e5),
             RiskLine("H", DATES[0], None, {"delta": -8e5})]
    vega = strike_grid(lines, TENORS, DATES, "vega")
    assert vega.at[TOTAL, TOTAL] == pytest.approx(25.0)
    delta = strike_grid(lines, TENORS, DATES, "delta")
    assert delta.at["1M", "5C"] == pytest.approx(-8e5)
    assert delta.at[TOTAL, TOTAL] == pytest.approx(0.0)
    buckets = greek_buckets(lines, TENORS, DATES)
    assert buckets.at["vega", TOTAL] == pytest.approx(25.0)
    assert buckets.at["delta", "1M"] == pytest.approx(2e5)


def test_risk_lines_from_a_booked_book(tmp_path):
    from datetime import datetime, timedelta, timezone
    from app.market_set import MarketSet
    from app.reports.risk_grid import risk_lines, tenor_axis
    from app.trades import service
    from app.trades.store import SqliteTradeStore
    from golden.app_harness import load_markets
    from market_data import Modifications

    class Clock:
        now = datetime(2026, 10, 15, tzinfo=timezone.utc)

        def __call__(self):
            Clock.now += timedelta(seconds=1)
            return Clock.now

    mods = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
    markets = MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), date(2026, 10, 15))
    store = SqliteTradeStore(tmp_path / "t.sqlite", clock=Clock())
    rows = [dict(currency_pair="EURUSD", strike=s, expiry="1M", call_put=cp, buy_sell="BUY",
                 notional="1m", premium=0.0, premium_ccy="USD")
            for s, cp in (("25d", "CALL"), ("25d", "PUT"), ("atm", "CALL"))]
    service.book_lines(store, markets, rows, mods, with_hedges=True)
    trades = store.trades()
    results, forwards, _ = service.price_trades(markets, trades, mods)
    lines, errors = risk_lines(markets, "EURUSD", trades, results, forwards, include_hedges=True)
    assert errors == [] and len(lines) == 4
    deltas = sorted(l.call_delta for l in lines if l.call_delta is not None)
    assert deltas[0] == pytest.approx(0.25, abs=0.03) and deltas[-1] == pytest.approx(0.75, abs=0.03)
    labels, dates, tenor_errors = tenor_axis(markets.market("EURUSD"))
    grid = strike_grid(lines, labels, dates, "delta")
    assert grid.at[TOTAL, TOTAL] == pytest.approx(0.0, abs=1e-6)     # hedged book
    assert grid[TOTAL].drop(TOTAL).sum() == pytest.approx(0.0, abs=1e-6)
    store.close()


def test_default_tenor_buckets_are_the_market_pillars():
    from app.market_context import MarketContext
    from app.reports.risk_grid import market_tenors, tenor_axis
    from golden.app_harness import load_markets
    context = MarketContext.from_payload(load_markets("MARKETS_20261015.json")["EURUSD"])
    pillars = market_tenors(context)
    assert pillars == ["1W", "2W", "1M", "2M", "3M", "6M", "9M", "1Y"]
    labels, dates, errors = tenor_axis(context, pillars)
    assert labels == pillars and errors == [] and dates == sorted(dates)
    _, _, errors = tenor_axis(context, ["1M", "2Y"])
    assert errors and "2Y" in errors[0]
