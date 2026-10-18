"""European digitals: replicated with vanilla legs on the line's smile
(pricing/digital.py), priced, Greeked, booked and revalued like vanillas."""
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from app import pricer
from app.hedge_delta import HEDGE_DELTA, HEDGE_MODE
from app.market_set import MarketSet
from app.trades import service
from app.trades.store import SqliteTradeStore
from app.vol_override import VOL_INPUT, VOL_MODE
from golden.app_harness import load_markets
from market_data import Modifications
from pricing import Pricable
from pricing.digital import RELATIVE_WIDTH, DigitalPricable, legs
from pricing_utils.BlackScholes import BlackScholes

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
PAYOUT = 1e6


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def line(pair="EURUSD", call_put="CALL", buy_sell="BUY", strike="1.13", payout="BASE",
         **changes):
    row = dict(currency_pair=pair, expiry="3M", option_type="Digital", call_put=call_put,
               strike=strike, buy_sell=buy_sell, notional="1m", payout_ccy=payout,
               premium=np.nan, premium_mode="auto", premium_ccy="TERM")
    row.update(changes)
    return row


def priced(markets, lines, extras=()):
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame(lines))
    results, errors = pricer.price_lines(markets, rows, MODS, extras=extras)
    assert errors == []
    return rows, results


@pytest.mark.parametrize("call_put, payout_base, expected", [
    ("CALL", False, [(0.995, 100.0), (1.005, -100.0)]),
    ("PUT", False, [(1.005, 100.0), (0.995, -100.0)]),
    # Base payout: the vanilla plus the strike times the term digital.
    ("CALL", True, [(1.0, 1.0), (0.995, 100.0), (1.005, -100.0)]),
    ("PUT", True, [(1.0, -1.0), (1.005, 100.0), (0.995, -100.0)]),
])
def test_replicating_legs(call_put, payout_base, expected):
    got = legs(1.0, call_put, payout_base, 0.01)
    assert [(round(k, 12), round(w, 12)) for k, w in got] == expected


def test_auto_premium_greeks_and_vol_of_a_digital(markets):
    _, (digital,) = priced(markets, [line()])
    assert digital["premium_auto"] and digital["pnl"] == 0.0
    assert digital["auto_premium"] == abs(digital["mtm"]) > 0
    assert digital["payout_ccy"] == "EUR" and digital["notional"] == "1,000,000"
    # The vol shown is the smile's at the strike, as for a vanilla there.
    _, (vanilla,) = priced(markets, [line(option_type="Vanilla")])
    assert digital["vol_pct"] == vanilla["vol_pct"] and digital["strike"] == vanilla["strike"]
    for key in ("delta", "fixed_delta", "gamma", "vega", "theta", "fixed_theta", "spot_delta",
                "fwd_delta", "pair_delta"):
        assert np.isfinite(digital[key]), key
    assert digital["delta_convention"] == vanilla["delta_convention"] == "S"


def test_every_greek_is_the_sum_of_the_legs(markets):
    context = pricer.market_for_pair(markets, "EURUSD")
    expiry = context.resolve_market_expiry("3M")
    market = context.market_for_expiry(expiry)
    from app.curve_roll import next_business_day
    next_day = next_business_day("EURUSD", VALUATION)
    next_market = context.market_for_expiry(expiry, as_of=next_day, allow_short_end=True)
    s = context.settings
    args = (market, s.pricing_model, s.atm_type, s.delta_base, MODS)
    kwargs = dict(next_market=next_market, next_valuation=next_day, extras=("delta_decay", "vega_decay"))
    row = pd.Series(dict(currency_pair="EURUSD", expiry=expiry, call_put="PUT", strike=1.11,
                         buy_sell="SELL", notional=PAYOUT, valuation=VALUATION))
    digital = DigitalPricable(row=row, premium=0.0, premium_ccy="USD",
                              payout_ccy="EUR").price(*args, **kwargs)
    total = {}
    for strike, weight in legs(1.11, "PUT", True, RELATIVE_WIDTH * 1.11):
        leg = row.copy()
        leg["strike"], leg["notional"] = strike, abs(weight) * PAYOUT
        leg["buy_sell"] = "SELL" if weight > 0 else "BUY"
        result = Pricable(row=leg, premium=0.0, premium_ccy="USD").price(*args, **kwargs)
        for key in ("mtm", "delta", "fixed_delta", "gamma", "fixed_gamma", "vega", "vanna",
                    "volga", "theta", "fixed_theta", "rega", "sega", "delta_decay",
                    "fixed_delta_decay", "vega_decay"):
            total[key] = total.get(key, 0.0) + result[key]
    for key, value in total.items():
        assert digital[key] == pytest.approx(value, rel=1e-12, abs=1e-9), key


def test_flat_smile_matches_the_black_digital(markets):
    context = pricer.market_for_pair(markets, "EURUSD")
    expiry = context.resolve_market_expiry("6M")
    market = context.market_for_expiry(expiry).copy_with(rr_25=0.0, rr_10=0.0, bf_25=0.0,
                                                         bf_10=0.0)
    s = context.settings
    for call_put, payout_ccy in (("CALL", "USD"), ("PUT", "USD"), ("CALL", "EUR")):
        row = pd.Series(dict(currency_pair="EURUSD", expiry=expiry, call_put=call_put,
                             strike=1.14, buy_sell="BUY", notional=PAYOUT, valuation=VALUATION))
        d = DigitalPricable(row=row, premium=0.0, premium_ccy="USD", payout_ccy=payout_ccy).price(
            market, s.pricing_model, s.atm_type, s.delta_base, MODS)
        b = BlackScholes(S=market.s, K=1.14, fwd_pts=market.fwd_pts, v=d["vol_pct"],
                         expiry_date=str(pd.Timestamp(expiry).date()), valuation_date=str(VALUATION),
                         CallPut=call_put, BuySell="BUY", Notional=1.0,
                         fwd_divisor=market.forward_divisor, term_discount=market.term_discount,
                         currency_pair="EURUSD")
        F, T, DF = float(b.f), float(b.t), float(b.term_discount_factor)
        sd = d["vol_pct"] / 100 * math.sqrt(T)
        d1 = (math.log(F / 1.14) + 0.5 * sd * sd) / sd
        phi = 1 if call_put == "CALL" else -1
        unit = F * norm.cdf(phi * d1) if payout_ccy == "EUR" else norm.cdf(phi * (d1 - sd))
        assert d["mtm_term"] == pytest.approx(PAYOUT * DF * unit, rel=1e-6)


def test_smile_digital_is_minus_the_strike_slope_of_the_vanilla(markets):
    rows, (digital,) = priced(markets, [line(pair="USDMXN", strike="18.5", payout="TERM")])
    strike = digital["strike"]

    def vanilla(k):
        _, (v,) = priced(markets, [line(pair="USDMXN", strike=str(k), option_type="Vanilla",
                                        notional="1m")])
        return v["mtm_term"]
    def slope(h):
        return -(vanilla(strike + h) - vanilla(strike - h)) / (2 * h)
    h = 1e-3 * strike
    richardson = (4 * slope(h / 2) - slope(h)) / 3
    assert digital["mtm_term"] == pytest.approx(richardson * PAYOUT / 1e6, rel=5e-6)


@pytest.mark.parametrize("payout", ["TERM", "BASE"])
def test_call_plus_put_is_the_discounted_payout(markets, payout):
    _, (call, put) = priced(markets, [line(payout=payout), line(call_put="PUT", payout=payout)])
    context = pricer.market_for_pair(markets, "EURUSD")
    market = context.market_for_expiry(call["expiry"])
    b = BlackScholes(S=market.s, K=1.13, fwd_pts=market.fwd_pts, v=10.0,
                     expiry_date=str(pd.Timestamp(call["expiry"]).date()),
                     valuation_date=str(VALUATION), CallPut="CALL", BuySell="BUY", Notional=1.0,
                     fwd_divisor=market.forward_divisor, term_discount=market.term_discount,
                     currency_pair="EURUSD")
    expected = PAYOUT * float(b.term_discount_factor) * (float(b.f) if payout == "BASE" else 1.0)
    assert call["mtm_term"] + put["mtm_term"] == pytest.approx(expected, rel=1e-10)
    # Zero to rounding on a 1m payout.
    assert call["vega"] + put["vega"] == pytest.approx(0.0, abs=1e-3)


def test_payout_defaults_to_base(markets):
    _, (blank, base) = priced(markets, [line(payout=None), line(payout="BASE")])
    assert blank["payout_ccy"] == base["payout_ccy"] == "EUR"
    assert blank["mtm"] == base["mtm"]


def test_typed_vol_on_a_digital(markets):
    _, (plain,) = priced(markets, [line()])
    _, (typed,) = priced(markets, [line(**{VOL_INPUT: plain["vol_pct"] + 1.0,
                                           VOL_MODE: "typed"})])
    assert typed["vol_pct"] == pytest.approx(plain["vol_pct"] + 1.0, abs=1e-6)
    # MTM stays at market; the auto premium is the price at the typed vol.
    assert typed["mtm"] == pytest.approx(plain["mtm"], rel=1e-12)
    assert typed["auto_premium"] == typed["prem_at_vol"] != plain["auto_premium"]
    assert typed["pnl"] == pytest.approx(typed["mtm"] + typed["premium"])


def test_digital_hedge_delta_is_not_capped_at_150(markets):
    _, (digital,) = priced(markets, [line(**{HEDGE_DELTA: 400.0, HEDGE_MODE: "typed"})])
    assert digital["hedge_delta_pct"] == 400.0
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame([line(
        option_type="Vanilla", **{HEDGE_DELTA: 400.0, HEDGE_MODE: "typed"})]))
    _, errors = pricer.price_lines(markets, rows, MODS)
    assert "Hedge delta must be a percent of notional from 0 to 150" in errors[0]


def test_a_digital_books_and_revalues_as_priced(markets, tmp_path):
    store = SqliteTradeStore(tmp_path / "trades.sqlite")
    rows, (digital,) = priced(markets, [line(payout="TERM")])
    result = service.book_lines(store, markets, rows, MODS, with_hedges=True)
    assert result.ok, result.errors
    option = next(t for t in store.trades() if t.is_option)
    assert option.option_type == "digital" and option.payout_ccy == "USD"
    assert option.notional == PAYOUT and option.premium == pytest.approx(digital["auto_premium"])
    assert any(t.is_forward for t in store.trades())
    revalued, _, errors = service.price_trades(markets, store.trades(), MODS)
    assert errors == []
    assert revalued[0]["mtm"] == pytest.approx(digital["mtm"], rel=1e-12)
    assert revalued[0]["delta"] == pytest.approx(digital["delta"], rel=1e-12)


def test_a_zero_rounding_residue_shows_as_zero():
    from app.ui.portfolio_block import _with_line_results
    shown = _with_line_results(pd.DataFrame([{"currency_pair": "EURUSD"}]),
                               [{"row": 1, "pnl_base": -1e-9, "mtm_base": -0.6}])
    assert shown.at[0, "pnl_base"] == "0" and shown.at[0, "mtm_base"] == "-1"
