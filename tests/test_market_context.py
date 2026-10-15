"""Backend market and pricer: validation, fallbacks and per-line isolation."""
import ast
from pathlib import Path

import pandas as pd
import pytest

from app import pricer
from app.market_context import (
    MarketContext, MarketContextError, PricingSettings, UnavailableMarket, build_context,
)
from app.market_store import validate_market
from golden.app_harness import load_markets
from market_data import Modifications

pytestmark = pytest.mark.filterwarnings("ignore")

MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def payload():
    return load_markets("MARKETS_20261015.json")["EURUSD"]


@pytest.fixture(scope="module")
def context(payload):
    return MarketContext.from_payload(payload)


def trade(**changes):
    line = dict(currency_pair="EURUSD", strike="atm", expiry="1M", call_put="CALL",
                buy_sell="BUY", notional="1m", premium=0.0, premium_ccy="USD")
    line.update(changes)
    return line


def loaded(payload):
    data = validate_market(payload)
    return dict(currency_pair=data["currency_pair"], valuation_date=data["valuation_date"],
                spot=data["spot"], forward_divisor=data["forward_divisor"],
                broker_strangle=data["broker_strangle"], vol_rows=data["vol_rows"],
                forward_rows=data["forward_rows"], event_rows=data["event_rows"])


def test_invalid_inputs_name_every_problem_at_once(payload):
    args = loaded(payload)
    args.update(currency_pair="EUR", spot=0.0, forward_divisor=float("nan"),
                vol_rows=args["vol_rows"].drop(columns=["bf_10"]))
    with pytest.raises(MarketContextError) as error:
        MarketContext(**args, settings=PricingSettings(pricing_model="black"))
    message = str(error.value)
    for expected in ("currency pair", "spot", "forward divisor", "bf_10", "pricing model"):
        assert expected in message


def test_invalid_payload_is_a_market_error_not_a_crash():
    with pytest.raises(MarketContextError, match="Invalid market"):
        MarketContext.from_payload({"currency_pair": "EURUSD"})


def test_unavailable_market_reports_the_reason_on_every_line(payload):
    context = build_context(**dict(loaded(payload), spot=-1.0))
    assert isinstance(context, UnavailableMarket)
    results, errors = pricer.price_lines(context, [trade(), trade(expiry="3M")], MODS)
    assert results == []
    assert len(errors) == 2
    assert all("spot must be a finite positive number" in error for error in errors)
    positions, hedge_errors = pricer.build_delta_hedges(context, pd.DataFrame([trade()]), MODS)
    assert positions == [] and len(hedge_errors) == 1


def test_context_prices_from_a_snapshot_of_the_tables(payload):
    args = loaded(payload)
    vol_rows = args["vol_rows"]
    context = MarketContext(**args)
    before = context.market_for_expiry("2026-11-16").v_atm
    vol_rows["atm"] = vol_rows["atm"] + 5.0       # caller keeps editing its table
    assert context.market_for_expiry("2026-11-16").v_atm == before


def test_one_bad_line_never_stops_the_book(context):
    frame = pd.DataFrame(
        [trade(), trade(premium_ccy="JPY"), trade(expiry="2030-01-01"), trade(strike="25d")],
        index=[7, 3, 11, 0],                       # editor rows after deletions
    )
    results, errors = pricer.price_lines(context, frame, MODS)
    assert [result["row"] for result in results] == [1, 4]
    assert errors[0].startswith("Line 2: Premium currency must be EUR or USD")
    assert errors[1].startswith("Line 3: Expiry is outside the quoted volatility range")
    assert all(result["smile_status"] == "resolved" for result in results)


def test_failed_hedge_group_is_reported_and_others_still_hedge(context, monkeypatch):
    results, _ = pricer.price_lines(context, [trade(), trade(expiry="3M")], MODS)
    original = context.market_for_expiry
    failing = results[1]["expiry"]

    def flaky(expiry, *args, **kwargs):
        if pd.Timestamp(expiry) == pd.Timestamp(failing):
            raise ValueError("forward curve unavailable")
        return original(expiry, *args, **kwargs)
    monkeypatch.setattr(context, "market_for_expiry", flaky)
    trades, errors = pricer.hedge_trades_for(context, results)
    assert [t["expiry"] for t in trades] == [results[0]["expiry"]]
    assert len(errors) == 1 and "forward curve unavailable" in errors[0]


def test_non_finite_delta_is_a_hedge_error_not_an_exception(context):
    results, _ = pricer.price_lines(context, [trade()], MODS)
    broken = [dict(results[0], delta=float("nan"))]
    trades, errors = pricer.hedge_trades_for(context, broken)
    assert trades == [] and "Smile-adjusted delta is invalid" in errors[0]


def test_render_app_defines_no_pricing_logic_of_its_own():
    """The app prices through the backend; nested copies would drift from it."""
    import app.ui.app as ui
    source = ast.parse(Path(ui.__file__).read_text(encoding="utf-8-sig"))
    render = next(node for node in source.body
                  if isinstance(node, ast.FunctionDef) and node.name == "render_app")
    nested = {node.name for node in ast.walk(render) if isinstance(node, ast.FunctionDef)}
    moved = {"market_for_expiry", "resolve_market_expiry", "variance_days_between",
             "forward_curve", "calculate_row", "build_delta_hedges",
             "auto_resolve_portfolio_strikes"}
    assert not nested & moved
