"""Progress callbacks: pricing and the spot ladder report as they go, on the
calling thread, and a failing callback never changes or stops a result."""
import threading
from datetime import date

import pytest

from app.market_set import MarketSet
from app.progress import current, reporting, safe
from app.reports.spot_ladder import calculate_spot_ladder
from app.trades import service
from app.trades.model import FX_FORWARD, FX_OPTION, Trade
from golden.app_harness import load_markets
from market_data import Modifications

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def trades():
    options = [Trade(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=VALUATION,
                     expiry=date(2027, 1, 15), buy_sell="BUY", notional=1e6, strike=strike,
                     call_put="CALL", premium=0.0, premium_ccy="USD",
                     trade_id=f"TR-00000{n}").normalized()
               for n, strike in enumerate((1.10, 1.12, 1.14), start=1)]
    hedge = Trade(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=VALUATION,
                  expiry=date(2027, 1, 15), buy_sell="SELL", notional=1e6, contract_rate=1.13,
                  trade_id="TR-000004").normalized()
    return [*options, hedge]


def test_safe_and_the_current_callback():
    def boom(*args):
        raise RuntimeError("display gone")
    safe(boom)(1, 2, "x")                          # swallowed
    safe(None)(1, 2)                               # no-op
    seen = []
    with reporting(lambda *a: seen.append(a)):
        current()(1, 3, "a")
        with reporting(None):
            current()(2, 3, "b")                   # nested: none
        current()(3, 3, "c")
    current()(9, 9)                                # outside: none
    assert seen == [(1, 3, "a"), (3, 3, "c")]


def test_pricing_reports_every_trade_and_a_failing_callback_changes_nothing(markets):
    seen = []
    with_bar = service.price_trades(markets, trades(), MODS,
                                    progress=lambda *a: seen.append(a))
    assert seen == [(1, 4, "TR-000001"), (2, 4, "TR-000002"), (3, 4, "TR-000003"),
                    (4, 4, "TR-000004")]

    def boom(*args):
        raise RuntimeError("display gone")
    assert service.price_trades(markets, trades(), MODS, progress=boom) == with_bar
    assert service.price_trades(markets, trades(), MODS) == with_bar


@pytest.mark.parametrize("workers", [1, 3])
def test_the_ladder_reports_on_the_calling_thread_and_keeps_its_results(markets, workers):
    context = markets.market("EURUSD")
    s = context.settings
    rows = [(i, t.to_pricer_row()) for i, t in enumerate(trades()[:3])]
    spots = [context.spot * f for f in (0.99, 1.0, 1.01)]
    caller, seen = threading.get_ident(), []

    def record(done, total, stage):
        seen.append((threading.get_ident(), done, total, stage))

    def ladder(progress=None):
        return calculate_spot_ladder(
            rows, spots, VALUATION, context.market_for_expiry, context.resolve_market_expiry,
            s.pricing_model, s.atm_type, s.delta_base, MODS, mode="optimized",
            scenario_workers=workers, progress=progress)
    assert ladder(record) == ladder()
    assert seen and all(thread == caller for thread, *_ in seen)
    for stage in {stage for *_, stage in seen}:
        counts = [(done, total) for _, done, total, s_ in seen if s_ == stage]
        assert [d for d, _ in counts] == list(range(1, counts[0][1] + 1)), stage
