from dataclasses import replace

import numpy as np
import pytest
import QuantLib as ql

from market_data import MarketData
from resolved_smile_inputs import (
    MarketSnapshot, PreparationError, ResolvedQuotes, build_pillars,
    resolve_conventions,
)
from helper_functions.fx_conventions import joint_calendar, spot_date


def market(pair="EURUSD"):
    return MarketData(s=150 if pair == "USDJPY" else 1.1, fwd_pts=30,
                      v_atm=10, rr_25=-.5, rr_10=-1, bf_25=.2, bf_10=.6,
                      currency_pair=pair)


@pytest.mark.parametrize("pair,expiry", [
    ("EURUSD", "2026-12-02"), ("EURUSD", "2028-10-02"),
    ("USDJPY", "2026-12-02"), ("USDJPY", "2028-10-02"),
    ("USDMXN", "2026-10-03"), ("USDMXN", "2026-12-02")])
def test_pillars_independent_quantlib_delta(pair, expiry):
    raw = MarketSnapshot.capture(market(pair))
    ctx = resolve_conventions(raw, expiry, "2026-10-02")
    pillars = build_pillars(ctx, ResolvedQuotes(10, -.5, -1, .2, .6), raw.s)
    kind = {"spot": (ql.DeltaVolQuote.PaSpot if ctx.premium_adjusted else
                     ql.DeltaVolQuote.Spot),
            "forward": (ql.DeltaVolQuote.PaFwd if ctx.premium_adjusted else
                        ql.DeltaVolQuote.Fwd)}[ctx.delta_convention]
    for i, delta in ((0, -.10), (1, -.25), (3, .25), (4, .10)):
        calc = ql.BlackDeltaCalculator(
            ql.Option.Put if delta < 0 else ql.Option.Call, kind, raw.s,
            ctx.domestic_discount, ctx.foreign_discount,
            pillars.vols[i] / 100 * ctx.time ** .5)
        assert calc.deltaFromStrike(pillars.strikes[i]) == pytest.approx(delta, abs=2e-11)
    calc = ql.BlackDeltaCalculator(ql.Option.Call, kind, raw.s,
                                  ctx.domestic_discount, ctx.foreign_discount,
                                  .1 * ctx.time ** .5)
    assert pillars.strikes[2] == pytest.approx(calc.atmStrike(ql.DeltaVolQuote.AtmDeltaNeutral))
    if pair == "USDMXN":
        assert ctx.delta_convention == "spot"     # EM too: spot delta up to 1Y
    if pair == "USDJPY":
        assert ctx.forward_divisor == 100


def test_cutoff_and_independent_adjustment():
    trade = ql.Date(2, 10, 2026)
    cutoff = joint_calendar("EURUSD").advance(spot_date("EURUSD", trade), ql.Period("1Y"))
    raw = MarketSnapshot.capture(market())
    ctx = resolve_conventions(raw, cutoff, trade)
    assert ctx.delta_convention == "spot"
    assert resolve_conventions(raw, cutoff + 1, trade).delta_convention == "forward"
    quotes = ResolvedQuotes(10, -.5, -1, .2, .6)
    for delta in ("spot", "forward"):
        for pa in (False, True):
            build_pillars(replace(ctx, delta_convention=delta, premium_adjusted=pa), quotes, raw.s)


@pytest.mark.parametrize("pair", ["USDEUR", "ABCDEF", None])
def test_unsupported_conventions(pair):
    with pytest.raises(PreparationError):
        resolve_conventions(MarketSnapshot.capture(market(pair)), "2026-12-02", "2026-10-02")


def test_snapshot_detached():
    m = market()
    raw = MarketSnapshot.capture(m)
    m.bf_25 = 9
    assert raw.bf_25 == .2
    raw.to_market().bf_25 = 8
    assert raw.bf_25 == .2


def test_snapshot_detaches_and_freezes_arrays_in_nested_options():
    m = market()
    values = np.array([.12, -.3, .85])
    m.frozen_sabr_params = {"params": values}
    raw = MarketSnapshot.capture(m)
    values[0] = 9
    assert raw.frozen_sabr_params["params"][0] == pytest.approx(.12)
    with pytest.raises(ValueError):
        raw.frozen_sabr_params["params"][0] = 1
    thawed = raw.to_market()
    thawed.frozen_sabr_params["params"][0] = .5
    assert raw.frozen_sabr_params["params"][0] == pytest.approx(.12)
