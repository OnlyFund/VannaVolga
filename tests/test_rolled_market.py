"""The rolled market: a day's quotes rolled to a later date (ROADMAP item 6).

Each property of the roll is checked against the market it was rolled from:
pillar dates are the new date's tenor dates, vols keep forward variance,
forward points stay with their tenor, passed events drop out; and every
fail-safe falls back with a reason.
"""
import math
from copy import deepcopy

import pandas as pd
import pytest

from apeiron.gaia.market_dates import pillars
from app.market_context import MarketContext, naive_date, wing_vols
from app.rolled_market import RollError, roll_context, rolling_enabled
from golden.app_harness import load_markets
from helper_functions.fx_conventions import settlement

pytestmark = pytest.mark.filterwarnings("ignore")

T0 = "2026-10-15"                  # Thursday, the quotes' date


@pytest.fixture(scope="module")
def payloads():
    return load_markets("MARKETS_20261015.json")


def _context(payloads, pair="EURUSD", **changes):
    payload = deepcopy(payloads[pair])
    payload.update(changes)
    return MarketContext.from_payload(payload)


def _var(market, start, expiry, wing="ATM"):
    """Total variance (vol squared times calendar years) of one smile vol."""
    vol = wing_vols(market)[wing] / 100.0
    return vol * vol * (naive_date(expiry) - naive_date(start)).days / 365.0


def test_a_roll_to_the_same_day_is_no_roll(payloads):
    context = _context(payloads)
    assert roll_context(context, T0) == (context, [])


@pytest.mark.parametrize("to_date", ["2026-10-16", "2026-10-19", "2026-10-22"])
def test_pillars_are_the_new_dates_tenor_dates(payloads, to_date):
    rolled, notes = roll_context(_context(payloads), to_date)
    rules = settlement("EURUSD")
    for tenor, expiry in zip(rolled.vol_rows["tenor"], rolled.vol_rows["expiry"]):
        assert naive_date(expiry) == pd.Timestamp(pillars.tenor_dates(rules, to_date, tenor).expiry)
    for tenor, delivery in zip(rolled.forward_rows["tenor"], rolled.forward_rows["delivery_date"]):
        assert naive_date(delivery) == pd.Timestamp(pillars.forward_delivery(rules, to_date, tenor))
    assert naive_date(rolled.valuation_date) == pd.Timestamp(to_date)
    assert rolled.rolled_from == pd.Timestamp(T0) and notes[0].startswith("EURUSD quotes of 2026-10-15")


@pytest.mark.parametrize("to_date", ["2026-10-16", "2026-10-19"])
@pytest.mark.parametrize("wing", ["PUT_10", "PUT_25", "ATM", "CALL_25", "CALL_10"])
def test_each_smile_vol_keeps_todays_forward_variance(payloads, to_date, wing):
    """var_rolled(T1 -> E) == var_today(T0 -> E) - var_today(T0 -> T1), at every rolled pillar."""
    context = _context(payloads)
    rolled, _ = roll_context(context, to_date)
    passed = _var(context._market_for_expiry(to_date, allow_short_end=True), T0, to_date, wing)
    checked = 0
    for tenor, expiry in zip(rolled.vol_rows["tenor"], rolled.vol_rows["expiry"]):
        if tenor == "1Y":                    # beyond the last quote: its own test
            continue
        today = _var(context._market_for_expiry(expiry), T0, expiry, wing)
        after = _var(rolled.market_for_expiry(expiry), to_date, expiry, wing)
        assert after == pytest.approx(today - passed, rel=1e-9, abs=1e-12), tenor
        checked += 1
    assert checked == 7


def test_a_fixed_expiry_between_pillars_is_close_to_its_forward_variance(payloads):
    """Off the pillars the rolled smile interpolates its own pillars, so it is close, not exact."""
    context = _context(payloads)
    rolled, _ = roll_context(context, "2026-10-16")
    expiry = "2026-12-01"
    today = _var(context._market_for_expiry(expiry), T0, expiry)
    passed = _var(context._market_for_expiry("2026-10-16", allow_short_end=True), T0, "2026-10-16")
    after = _var(rolled.market_for_expiry(expiry), "2026-10-16", expiry)
    assert after == pytest.approx(today - passed, rel=2e-3)


def test_forward_points_and_rates_stay_with_their_tenor(payloads):
    context = _context(payloads)
    rolled, _ = roll_context(context, "2026-10-19")
    before = context.forward_rows.set_index("tenor")
    after = rolled.forward_rows.set_index("tenor")
    for column in ("fwd_pts", "term_discount"):
        assert list(after[column]) == list(before.loc[after.index, column])
    assert rolled.short_points == context.short_points


def test_the_last_pillar_beyond_todays_quotes_holds_the_last_segments_forward_vol(payloads):
    """1Y rolled 4 days lands past the last quote: total variance runs on at the 9M -> 1Y
    rate per variance day (the surface's interpolation clock), less the days that passed."""
    context = _context(payloads)
    rolled, notes = roll_context(context, "2026-10-19")
    old = context.vol_rows.set_index("tenor")["expiry"].map(naive_date)
    new = naive_date(rolled.vol_rows.set_index("tenor").loc["1Y", "expiry"])
    assert new > old["1Y"]
    for wing in ("PUT_10", "ATM", "CALL_10"):
        last = _var(context._market_for_expiry(old["1Y"]), T0, old["1Y"], wing)
        previous = _var(context._market_for_expiry(old["9M"]), T0, old["9M"], wing)
        rate = (last - previous) / context.variance_days_between(old["9M"], old["1Y"])
        passed = _var(context._market_for_expiry("2026-10-19", allow_short_end=True), T0, "2026-10-19", wing)
        expected = last + rate * context.variance_days_between(old["1Y"], new) - passed
        assert _var(rolled.market_for_expiry(new), "2026-10-19", new, wing) == pytest.approx(expected, rel=1e-9)
    assert any("1Y beyond the last quote" in note for note in notes)
    assert not any("kept for" in note for note in notes)


def test_a_pillar_whose_forward_variance_cannot_be_formed_keeps_its_tenor_smile(payloads, monkeypatch):
    import app.rolled_market as module

    def no_extension(*args):
        raise ValueError("no forward variance to extend in the ATM vol")
    monkeypatch.setattr(module, "_extended_quotes", no_extension)
    context = _context(payloads)
    rolled, notes = roll_context(context, "2026-10-19")
    quoted = wing_vols(context._market_for_expiry(context.vol_rows.set_index("tenor").loc["1Y", "expiry"]))
    year = rolled.vol_rows.set_index("tenor").loc["1Y", "expiry"]
    assert wing_vols(rolled.market_for_expiry(year))["ATM"] == pytest.approx(quoted["ATM"], abs=1e-9)
    assert any("tenor smile kept for 1Y (no forward variance to extend" in note for note in notes)


def test_quotes_become_smile_butterflies(payloads):
    context = _context(payloads)
    assert context.broker_strangle
    rolled, _ = roll_context(context, "2026-10-16")
    assert not rolled.broker_strangle


def test_passed_events_drop_out(payloads):
    events = [{"event_date": "2026-10-16", "variance_days": 1.0, "event": "passed"},
              {"event_date": "2026-11-05", "variance_days": 0.5, "event": "ahead"}]
    rolled, _ = roll_context(_context(payloads, event_calendar=events), "2026-10-19")
    assert [str(day)[:10] for day in rolled.event_rows["event_date"]] == ["2026-11-05"]


def test_a_pillar_without_a_tenor_cannot_roll(payloads):
    payload = deepcopy(payloads["EURUSD"])
    payload["vol_data"][0]["tenor"] = ""
    with pytest.raises(RollError, match="has no tenor"):
        roll_context(MarketContext.from_payload(payload), "2026-10-16")


def test_forward_points_without_tenors_move_by_the_days_rolled(payloads):
    """Saved markets always have tenors; a market typed into the Markets tab may not."""
    context = _context(payloads)
    context.forward_rows = context.forward_rows.drop(columns=["tenor"])
    rolled, notes = roll_context(context, "2026-10-19")
    before = pd.to_datetime(context.forward_rows["delivery_date"]).dt.tz_localize(None)
    after = pd.to_datetime(rolled.forward_rows["delivery_date"])
    assert list(after - before) == [pd.Timedelta(days=4)] * len(before)
    assert list(rolled.forward_rows["fwd_pts"]) == list(context.forward_rows["fwd_pts"])
    assert "forward points moved by the days rolled (no tenors)" in notes[0]


def test_rolling_can_be_turned_off():
    assert rolling_enabled({}) and rolling_enabled({"VV_ROLL_MARKET": "1"})
    assert not rolling_enabled({"VV_ROLL_MARKET": "0"})


@pytest.mark.parametrize("pair", ["EURUSD", "USDJPY", "USDMXN"])
def test_every_pair_rolls_a_week_and_prices_at_every_pillar(payloads, pair):
    rolled, notes = roll_context(_context(payloads, pair), "2026-10-22")
    for expiry in rolled.vol_rows["expiry"]:
        market = rolled.market_for_expiry(expiry)
        assert all(math.isfinite(v) and v > 0 for v in wing_vols(market).values())
