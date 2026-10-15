"""Sticky-tenor curve roll used by the rolled smile theta."""
import ast
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from app.curve_roll import (
    anchor_forward_rows, next_business_day, roll_tenor_dates, roll_forward_rows,
    roll_vol_rows, spot_date,
)
from app.market_service import MarketService, pair_delivery, pair_divisor
from app.smile_service import DiagnosticsCollector
from pricing.greeks import calculate_greeks

pytestmark = pytest.mark.filterwarnings("ignore")

PAIR = "EURUSD"
T0, T1 = date(2026, 10, 2), date(2026, 10, 5)          # Friday -> Monday
TENORS = ["1W", "1M", "2M"]
EXPIRIES = pd.to_datetime(["2026-10-08", "2026-11-04", "2026-12-03"])  # tenor rules at T0


def app_curve(events=None):
    """Run the real nested UI market builder on a three-pillar curve."""
    import app.ui.app as ui
    vol_rows = pd.DataFrame(dict(
        tenor=TENORS, expiry=EXPIRIES, atm=[8., 9., 11.], rr_25=[-.5] * 3,
        rr_10=[-1.] * 3, bf_25=[.2] * 3, bf_10=[.6] * 3))
    forward_rows = pd.DataFrame(dict(
        tenor=TENORS, delivery_date=[pair_delivery(PAIR, x) for x in EXPIRIES],
        fwd_pts=[3., 14., 29.], term_discount=[.04, .045, .05]))
    event_rows = pd.DataFrame(events or [], columns=["event_date", "variance_days"])
    collector = DiagnosticsCollector()
    namespace = dict(vars(ui), st=SimpleNamespace(session_state=SimpleNamespace(
        vol_rows=vol_rows, forward_rows=forward_rows, event_rows=event_rows,
        currency_pair=PAIR)),
        valuation_date=T0, spot=1.1, currency_pair=PAIR,
        forward_divisor=pair_divisor(PAIR), pricing_model="vv", broker_strangle=False,
        atm_type="dn", delta_base=True, smooth_pillar_interpolation=True,
        diagnostics=collector, market_service=MarketService(collector))
    source = ast.parse(Path(ui.__file__).read_text(encoding="utf-8-sig"))
    render = next(n for n in source.body
                  if isinstance(n, ast.FunctionDef) and n.name == "render_app")
    names = {"naive_date", "variance_days_between", "forward_curve",
             "forward_points_for_expiry", "term_discount_for_expiry", "market_for_expiry",
             "resolve_market_expiry"}
    module = ast.Module(body=[n for n in render.body
                             if isinstance(n, ast.FunctionDef) and n.name in names])
    exec(compile(module, ui.__file__, "exec"), namespace)
    return namespace


def rolled_expiry(tenor):
    index = TENORS.index(tenor)
    return roll_tenor_dates(PAIR, [tenor], [EXPIRIES[index]], T0, T1)[0]


def test_theta_horizon_is_the_next_fx_trading_day():
    assert next_business_day(PAIR, T0) == T1
    # Columbus Day is a USD settlement holiday but an FX trading day.
    assert next_business_day(PAIR, date(2026, 10, 9)) == date(2026, 10, 12)
    assert next_business_day(PAIR, date(2026, 12, 24)) == date(2026, 12, 28)
    assert next_business_day(PAIR, date(2026, 12, 31)) == date(2027, 1, 4)


def test_zero_day_roll_leaves_the_market_unchanged():
    build = app_curve()["market_for_expiry"]
    today, same_day = build("2026-11-17"), build("2026-11-17", as_of=T0)
    assert same_day.v_atm == pytest.approx(today.v_atm, abs=1e-12)
    assert same_day.fwd_pts == pytest.approx(today.fwd_pts, abs=1e-12)


def test_variance_clock_counts_full_calendar_days():
    clock = app_curve()["variance_days_between"]
    assert clock(T0, T1) == 3.0
    assert clock(T1, date(2026, 10, 6)) == 1.0


def test_usdzar_friday_rolls_curves_even_when_spot_settlement_is_unchanged():
    from pricing.fx_forward import FxForwardPosition
    from pricing.greeks import calculate_forward_theta
    from pricing.pricing import Pricable, price_option

    namespace = app_curve()
    t0, t1 = date(2026, 10, 9), date(2026, 10, 12)
    pair = "USDZAR"
    assert spot_date(pair, t0) == spot_date(pair, t1)
    vol_rows = pd.DataFrame(dict(
        tenor=["9M", "1Y"], expiry=pd.to_datetime(["2027-07-01", "2027-09-28"]),
        atm=[11.78, 12.07], rr_25=[2.03, 2.01], rr_10=[4.115, 4.07],
        bf_25=[.318, .323], bf_10=[1.095, 1.148],
    ))
    forward_rows = pd.DataFrame(dict(
        tenor=["9M", "1Y"],
        delivery_date=pd.to_datetime(["2027-07-06", "2027-09-30"]),
        fwd_pts=[3145., 4215.], term_discount=[.05, .05],
    ))
    namespace.update(
        valuation_date=t0, spot=16.59, currency_pair=pair, broker_strangle=True,
    )
    namespace["st"].session_state.vol_rows = vol_rows
    namespace["st"].session_state.forward_rows = forward_rows
    namespace["st"].session_state.currency_pair = pair
    build = namespace["market_for_expiry"]
    expiry = date(2027, 9, 28)
    today = build(expiry)
    tomorrow = build(expiry, as_of=t1, allow_short_end=True)
    assert today.fwd_pts == 4215.
    assert tomorrow.fwd_pts == pytest.approx(4215. - (4215. - 3145.) * 3. / 86.)
    assert tomorrow.v_atm < today.v_atm
    rolled_vol = roll_vol_rows(vol_rows, pair, t0, t1)
    rolled_forward = roll_forward_rows(
        forward_rows, rolled_vol, pair, t0, t1, lambda e: pair_delivery(pair, e)
    )
    assert rolled_vol.expiry.iloc[-1] == pd.Timestamp("2027-10-01")
    # Sunday is a valid synthetic risk coordinate, not a settlement date.
    assert rolled_forward.delivery_date.iloc[-1] == pd.Timestamp("2027-10-03")
    row = pd.Series(dict(
        currency_pair=pair, strike=17., expiry=expiry, valuation=t0,
        call_put="CALL", buy_sell="BUY", notional=700_000_000., premium_ccy="USD",
    ))
    result = Pricable(row, 0., "USD").price(
        today, "vv", "dn", True, {}, next_market=tomorrow, next_valuation=t1
    )
    assert abs(result["theta"] - result["fixed_theta"]) > 10_000.
    hedge = FxForwardPosition(
        pair, expiry, -result["delta"], today.s + today.fwd_pts / today.forward_divisor
    )
    hedge_theta = calculate_forward_theta(hedge, today, tomorrow, t0, t1)
    assert hedge_theta > 0.
    rolled_row = row.copy()
    rolled_row["valuation"] = t1
    _, rolled_volatility, _ = price_option(tomorrow, rolled_row, "vv", "dn", True)
    _, frozen_forward_volatility, _ = price_option(
        tomorrow.copy_with(fwd_pts=today.fwd_pts), rolled_row, "vv", "dn", True
    )
    assert rolled_volatility > frozen_forward_volatility


def test_flat_tenor_vols_stay_flat_across_weekend_and_between_pillars():
    namespace = app_curve()
    rows = namespace["st"].session_state.vol_rows
    rows["atm"] = 10.0
    build = namespace["market_for_expiry"]
    for expiry in ("2026-10-08", "2026-10-23", "2026-11-17"):
        today = build(expiry)
        tomorrow = build(expiry, as_of=T1, allow_short_end=True)
        for field in ("v_atm", "rr_25", "rr_10", "bf_25", "bf_10"):
            assert getattr(today, field) == pytest.approx(
                getattr(tomorrow, field), abs=1e-10
            )
        assert tomorrow.v_atm == pytest.approx(10.0, abs=1e-10)


def test_pillar_dates_roll_with_their_tenors():
    for tenor, stored in zip(TENORS, EXPIRIES):
        assert rolled_expiry(tenor) == stored + pd.Timedelta(days=3)


def test_rolled_pillar_date_keeps_its_tenor_quote():
    build = app_curve()["market_for_expiry"]
    on_pillar = build(rolled_expiry("2M"), as_of=T1, allow_short_end=True)
    assert on_pillar.v_atm == pytest.approx(11.0, abs=1e-9)


def test_fixed_expiry_rolls_down_the_vol_and_forward_curves():
    build = app_curve()["market_for_expiry"]
    expiry = EXPIRIES[1]                       # today's 1M pillar date
    today = build(expiry)
    tomorrow = build(expiry, as_of=T1, allow_short_end=True)
    assert today.v_atm == pytest.approx(9.0, abs=1e-9)
    assert 8.0 < tomorrow.v_atm < 9.0          # now short of the rolled 1M
    assert today.fwd_pts == pytest.approx(14.0)
    assert 3.0 < tomorrow.fwd_pts < 14.0       # forward points roll down


def test_short_end_takes_first_pillar_variance_by_event_weights():
    namespace = app_curve(events=[(pd.Timestamp("2026-10-07"), 2.0)])
    build, vdays = namespace["market_for_expiry"], namespace["variance_days_between"]
    expiry = pd.Timestamp("2026-10-08")        # before the rolled 1W pillar
    first = rolled_expiry("1W")
    assert expiry < first
    short = build(expiry, as_of=T1, allow_short_end=True)
    curve = pd.Timestamp(T1)
    first_variance = 8.0 ** 2 * (first - curve).days
    expected = np.sqrt(first_variance * vdays(curve, expiry) / vdays(curve, first)
                       / (expiry - curve).days)
    assert short.v_atm == pytest.approx(expected, rel=1e-9)
    # The event inside the short window raises its share of the variance.
    plain = app_curve()["market_for_expiry"](expiry, as_of=T1, allow_short_end=True)
    assert short.v_atm > plain.v_atm


def test_expiry_before_the_first_pillar_prices_today():
    build = app_curve()["market_for_expiry"]
    short = build("2026-10-06")                 # before the 1W pillar, at T0
    assert 0.0 < short.v_atm < 8.0 * 1.5
    with pytest.raises(ValueError, match="after the valuation date"):
        build(T0)


def test_overnight_input_resolves_to_the_next_good_business_day():
    from app.curve_roll import overnight_expiry
    resolve = app_curve()["resolve_market_expiry"]
    assert overnight_expiry(PAIR, T0) == date(2026, 10, 5)      # Friday -> Monday
    for token in ("ON", "o/n", "1D"):
        assert resolve(token) == date(2026, 10, 5)


def test_short_end_equals_an_overnight_node_interpolated_to_1w():
    """O/N vol from 1W and the event weights, then interpolated O/N -> 1W."""
    from app.curve_roll import overnight_expiry
    namespace = app_curve(events=[(pd.Timestamp("2026-10-07"), 2.0)])
    build, vdays = namespace["market_for_expiry"], namespace["variance_days_between"]
    curve = pd.Timestamp(T0)
    on, week = pd.Timestamp(overnight_expiry(PAIR, T0)), pd.Timestamp(EXPIRIES[0])
    days = lambda d: (d - curve).days

    # Explicit O/N node: 1W total variance scaled by its share of variance days.
    week_variance = 8.0 ** 2 * days(week)
    on_variance = week_variance * vdays(curve, on) / vdays(curve, week)
    assert build(on).v_atm == pytest.approx(np.sqrt(on_variance / days(on)), rel=1e-9)

    # Between the node and 1W: linear in variance days between the two nodes.
    for expiry in (pd.Timestamp("2026-10-06"), pd.Timestamp("2026-10-07")):
        weight = (vdays(curve, expiry) - vdays(curve, on)) / (vdays(curve, week) - vdays(curve, on))
        variance = on_variance + weight * (week_variance - on_variance)
        assert build(expiry).v_atm == pytest.approx(np.sqrt(variance / days(expiry)), rel=1e-9)
    # The event on 10-07 carries extra variance into that expiry.
    assert build("2026-10-07").v_atm > build("2026-10-06").v_atm


def test_forward_curve_is_anchored_at_zero_points_on_spot_date():
    rows = pd.DataFrame(dict(tenor=["1W"], delivery_date=[pd.Timestamp("2026-10-14")],
                             fwd_pts=[3.0], term_discount=[0.04]))
    anchored = anchor_forward_rows(rows, PAIR, T1)
    assert anchored["delivery_date"].iloc[0] == pd.Timestamp(spot_date(PAIR, T1))
    assert anchored["fwd_pts"].iloc[0] == 0.0
    assert anchored["term_discount"].iloc[0] == 0.04


def test_rolled_theta_is_one_day_price_change_in_base_currency():
    build = app_curve()["market_for_expiry"]
    market = build("2026-11-17")
    row = pd.Series(dict(currency_pair=PAIR, strike=1.1, expiry=date(2026, 11, 17),
                         valuation=T0, call_put="CALL", buy_sell="BUY",
                         notional="1000000", premium=0., premium_ccy="USD"))
    greeks = calculate_greeks(
        market, row, None, "vv", "dn", True, {},
        next_market=build(row.expiry, as_of=T1, allow_short_end=True),
        next_valuation=T1,
    )
    # A long ATM option loses value over a weekend; not divided by 252.
    assert -5_000 < greeks["theta"] < -100


def test_friday_theta_reprices_date_and_forward_on_the_rolled_skew():
    from pricing.greeks import _rolled_smile_theta
    from pricing.pricing import price_option, price_option_with_vol

    namespace = app_curve()
    build = namespace["market_for_expiry"]
    # Keep all tenor smile quotes flat to isolate time and forward/skew roll.
    namespace["st"].session_state.vol_rows["atm"] = 10.0
    expiry = date(2026, 11, 4)
    today = build(expiry)
    tomorrow = build(expiry, as_of=T1, allow_short_end=True)
    row = pd.Series(dict(
        strike=1.1, expiry=expiry, valuation=T0, call_put="CALL",
        buy_sell="BUY", notional=1_000_000.,
    ))
    _, today_vol, today_price = price_option(today, row, "vv", "dn", True)
    rolled_row = row.copy()
    rolled_row["valuation"] = T1
    _, rolled_vol, rolled_price = price_option(tomorrow, rolled_row, "vv", "dn", True)
    common = dict(
        k=1.1, expiry_date=expiry.isoformat(), valuation_date=T0.isoformat(),
        CallPut="CALL", BuySell="BUY", Notional=1_000_000.,
        atm_type="dn", deltaBase=True, pricing_model="vv",
    )
    theta = _rolled_smile_theta(today, tomorrow, common, {}, T1)
    assert (T1 - T0).days == 3
    assert tomorrow.fwd_pts < today.fwd_pts
    assert 1.1 / (tomorrow.s + tomorrow.fwd_pts / tomorrow.forward_divisor) > (
        1.1 / (today.s + today.fwd_pts / today.forward_divisor)
    )
    assert rolled_vol != pytest.approx(today_vol, abs=1e-7)
    assert theta == pytest.approx((rolled_price - today_price) / today.s)
    _, _, frozen_vol_price = price_option_with_vol(
        tomorrow, rolled_row, today_vol, "dn", True
    )
    assert theta != pytest.approx((frozen_vol_price - today_price) / today.s)
    _, _, frozen_forward_price = price_option(
        tomorrow.copy_with(fwd_pts=today.fwd_pts), rolled_row, "vv", "dn", True
    )
    assert theta != pytest.approx((frozen_forward_price - today_price) / today.s)


def test_lower_forward_moves_fixed_call_strike_up_the_positive_skew():
    from market_data import MarketData
    from pricing.greeks import _rolled_smile_theta
    from pricing.pricing import price_option, price_option_with_vol

    today = MarketData(
        s=16.6, fwd_pts=4215., v_atm=12.07, rr_25=2.01, rr_10=4.07,
        bf_25=.35, bf_10=1.15, currency_pair="USDZAR",
        wrapped_smile=True, broker_strangle=False,
    )
    tomorrow = today.copy_with(fwd_pts=4202.)
    row = pd.Series(dict(
        strike=17.2, expiry="2027-09-28", valuation="2026-10-09",
        call_put="CALL", buy_sell="BUY", notional=700_000_000.,
    ))
    rolled_row = row.copy()
    rolled_row["valuation"] = "2026-10-12"
    _, _, today_price = price_option(today, row, "vv", "dn", True)
    _, rolled_vol, rolled_price = price_option(tomorrow, rolled_row, "vv", "dn", True)
    # Compare on the same rolled date to isolate the forward/skew effect.
    _, unchanged_forward_vol, _ = price_option(today, rolled_row, "vv", "dn", True)
    assert rolled_vol > unchanged_forward_vol
    common = dict(
        k=row.strike, expiry_date=row.expiry, valuation_date=row.valuation,
        CallPut="CALL", BuySell="BUY", Notional=row.notional,
        atm_type="dn", deltaBase=True, pricing_model="vv",
    )
    theta = _rolled_smile_theta(today, tomorrow, common, {}, "2026-10-12")
    assert theta == pytest.approx((rolled_price - today_price) / today.s)
    _, _, frozen_skew_price = price_option_with_vol(
        tomorrow, rolled_row, unchanged_forward_vol, "dn", True
    )
    assert rolled_price > frozen_skew_price


@pytest.mark.parametrize("pair,expected", [
    ("EURUSD", "USD"), ("GBPUSD", "USD"), ("USDJPY", "USD"),
    ("EURGBP", "EUR"), ("AUDJPY", "AUD"), ("EUR/USD", "USD"), ("", None), ("EURUS", None),
])
def test_conventional_premium_currency(pair, expected):
    from helper_functions.premium_quotes import conventional_premium_ccy
    assert conventional_premium_ccy(pair) == expected


def test_theta_is_reported_in_the_premium_currency(monkeypatch):
    import pricing.greeks as greeks_module
    build = app_curve()["market_for_expiry"]
    market = build("2026-11-17")
    row = pd.Series(dict(currency_pair=PAIR, strike=1.1, expiry=date(2026, 11, 17),
                         valuation=T0, call_put="CALL", buy_sell="BUY",
                         notional="1000000", premium=0., premium_ccy="EUR"))
    kwargs = dict(next_market=build(row.expiry, as_of=T1, allow_short_end=True),
                  next_valuation=T1)
    in_usd = calculate_greeks(market, row, None, "vv", "dn", True, {}, **kwargs)
    # Without a premium-currency rule theta stays in the base currency.
    monkeypatch.setattr(greeks_module, "theta_currency", lambda row, market: None)
    in_eur = calculate_greeks(market, row, None, "vv", "dn", True, {}, **kwargs)
    # EURUSD premium is USD whatever the line's entered premium currency.
    for key in ("theta", "fixed_theta"):
        assert in_usd[key] == pytest.approx(in_eur[key] * market.s, rel=1e-12)
    assert in_usd["vega"] == pytest.approx(in_eur["vega"], rel=1e-12)
