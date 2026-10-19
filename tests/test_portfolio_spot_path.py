"""The Portfolio's spot edit reprices along the smile dynamics: at a moved spot
a book's P&L and smile delta match the Risk ladder there; at the market spot
nothing changes; "quotes held" is the old repricing; a failure falls back to
the quotes held and says so."""
from datetime import date

import pandas as pd
import pytest

from app import pricer
from app.market_set import MarketSet
from app.reports.spot_ladder import calculate_spot_ladder
from app.trades import service
from app.trades.model import FX_OPTION, Trade
from app.ui.risk_block import ladder_rows
from golden.app_harness import load_markets
from market_data import Modifications

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def option(trade_id, strike, call_put, buy_sell, expiry, option_type=None):
    return Trade(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=VALUATION,
                 expiry=expiry, buy_sell=buy_sell, notional=10e6, strike=strike,
                 call_put=call_put, premium=0.0, premium_ccy="USD", option_type=option_type,
                 trade_id=trade_id).normalized()


BOOK = [option("A", 1.10, "PUT", "SELL", date(2027, 1, 15)),
        option("B", 1.13, "PUT", "BUY", date(2027, 4, 15)),
        option("C", 1.17, "CALL", "SELL", date(2026, 12, 1)),
        option("D", 1.15, "CALL", "BUY", date(2027, 1, 15), option_type="digital")]


def totals(results):
    return sum(r["pnl_base"] for r in results), sum(r["delta"] for r in results)


def ladder(markets, spot):
    context = markets.market("EURUSD")
    s = context.settings
    out, errors = calculate_spot_ladder(
        list(pd.DataFrame(ladder_rows(BOOK)).iterrows()), [context.spot, spot], VALUATION,
        context.market_for_expiry, context.resolve_market_expiry, s.pricing_model,
        s.atm_type, s.delta_base, MODS, mode="optimized", scenario_workers=1)
    assert errors == []
    return out[1]["pnl_base"] - out[0]["pnl_base"], out[1]["delta"]


@pytest.mark.parametrize("move", [0.01, -0.005])
def test_a_moved_spot_reprices_as_the_risk_ladder(markets, move):
    spot = markets.market("EURUSD").spot * (1 + move)
    base, _, errors = service.price_trades(markets, BOOK, MODS)
    moved, _, moved_errors = service.price_trades(markets.with_spots({"EURUSD": spot}), BOOK, MODS)
    assert errors == moved_errors == []
    assert all(r["spot_path"] == pricer.SMILE_PATH for r in moved)
    pnl_change = totals(moved)[0] - totals(base)[0]
    ladder_pnl, ladder_delta = ladder(markets, spot)
    assert pnl_change == pytest.approx(ladder_pnl, rel=1e-5)
    assert totals(moved)[1] == pytest.approx(ladder_delta, rel=1e-5)


def test_at_the_market_spot_both_paths_price_the_same(markets):
    smile, _, _ = service.price_trades(markets, BOOK, MODS)
    held, _, _ = service.price_trades(markets, BOOK, MODS, spot_path=pricer.QUOTES_HELD)
    assert smile == held
    assert not any("spot_path" in r for r in smile)


def test_quotes_held_reprices_on_the_unmoved_quotes(markets):
    spot = markets.market("EURUSD").spot * 1.01
    moved = markets.with_spots({"EURUSD": spot})
    held, _, _ = service.price_trades(moved, BOOK, MODS, spot_path=pricer.QUOTES_HELD)
    smile, _, _ = service.price_trades(moved, BOOK, MODS)
    assert totals(held) != pytest.approx(totals(smile))
    context = moved.market("EURUSD")
    for r, trade in zip(held, BOOK):
        quotes = context.market_for_expiry(pd.Timestamp(trade.expiry))
        assert quotes.s == spot and "spot_path" not in r


def test_the_anchor_is_the_market_data_spot_however_often_moved(markets):
    context = markets.market("EURUSD")
    twice = context.with_spot(context.spot * 1.01).with_spot(context.spot * 0.99)
    assert twice.spot_anchor == context.spot


def test_failing_dynamics_fall_back_to_the_quotes_held_and_say_so(markets, monkeypatch):
    import app.reports.spot_ladder as ladder_module

    def boom(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(ladder_module, "move_quotes_with_spot", boom)
    moved = markets.with_spots({"EURUSD": markets.market("EURUSD").spot * 1.01})
    fallback, _, errors = service.price_trades(moved, BOOK, MODS)
    held, _, _ = service.price_trades(moved, BOOK, MODS, spot_path=pricer.QUOTES_HELD)
    assert errors == []
    assert all(r["spot_path"] == pricer.QUOTES_HELD and "boom" in r["spot_path_note"]
               for r in fallback)
    assert [r["pnl_base"] for r in fallback] == [r["pnl_base"] for r in held]
