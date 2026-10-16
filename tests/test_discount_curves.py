"""Discount curves: SOFR / forward-implied, then quoted, stale and unavailable."""
import pytest

from helper_functions.fx_conventions import CCY, spot_date
from market_data.discount_curves import (
    IMPLIED, QUOTED, SOFR, STALE, TENORS, UNAVAILABLE, DiscountCurves, fill_tenors,
)
from market_data.market_tables import _ql_date, market, market_bundle, tenor_dates
from market_data.store import VOL_FIELDS, MarketDataStore

TODAY = "2026-10-16"
USD = {tenor: 0.04 for tenor in TENORS}
SURFACE = {tenor: dict(zip(VOL_FIELDS, (8.0, -0.5, 0.2, -0.9, 0.6))) for tenor in TENORS}


def points(pair, valuation, spot, base_rate, term_rate, divisor):
    """Forward points consistent with flat simple rates on each leg's day count."""
    start = spot_date(pair, _ql_date(valuation))
    out = {}
    for tenor in TENORS:
        end = _ql_date(tenor_dates(pair, valuation, tenor)[1])
        tau = {c: CCY[c].day_count.yearFraction(start, end) for c in (pair[:3], pair[3:])}
        forward = spot * (1 + term_rate * tau[pair[3:]]) / (1 + base_rate * tau[pair[:3]])
        out[tenor] = (forward - spot) * divisor
    return out


@pytest.fixture
def store(tmp_path):
    return MarketDataStore(tmp_path / "market_data.sqlite")


def save(store, valuation, as_of, **data):
    snapshot_id = store.save(valuation, as_of, **data)
    return store.load(snapshot_id=snapshot_id)


def test_fill_tenors_interpolates_in_time_and_holds_the_ends_flat():
    rates, filled = fill_tenors({"1M": 0.03, "3M": 0.05})
    assert rates["1W"] == rates["2W"] == 0.03 and rates["1Y"] == 0.05
    assert rates["2M"] == pytest.approx(0.03 + 0.02 * 31 / 61)
    assert filled == ["1W", "2W", "2M", "6M", "9M", "1Y"]


def test_usd_is_sofr_and_others_are_implied_from_forward_points(store):
    jpy = points("USDJPY", TODAY, 150.0, 0.04, 0.01, 100)
    del jpy["2M"]
    snapshot = save(store, TODAY, f"{TODAY}T12:00:00", spots={"USDJPY": 150.0},
                    depo_curves={"USD": {k: v for k, v in USD.items() if k != "9M"},
                                 "JPY": {tenor: 0.02 for tenor in TENORS}},
                    forward_points={"USDJPY": jpy})
    curves = DiscountCurves(snapshot, store)
    usd = curves.curve("USD")
    assert usd.source == SOFR and usd.rates["9M"] == pytest.approx(0.04)
    assert usd.notes == ["USD: SOFR 9M interpolated"]
    implied = curves.curve("JPY")
    assert implied.source == IMPLIED           # not the quoted 2%: the forwards say 1%
    assert implied.rates == pytest.approx({tenor: 0.01 for tenor in TENORS}, abs=1e-9)
    assert implied.notes == ["JPY: 2M interpolated (no USDJPY forward points)"]


def test_quoted_depo_when_the_pair_has_no_forward_points(store):
    snapshot = save(store, TODAY, f"{TODAY}T12:00:00", spots={"USDJPY": 150.0},
                    depo_curves={"USD": USD, "JPY": {"1M": 0.012, "1Y": 0.016}})
    curve = DiscountCurves(snapshot, store).curve("JPY")
    assert curve.source == QUOTED and curve.rates["1W"] == 0.012 and curve.rates["1Y"] == 0.016
    assert "basis not included" in curve.notes[0]


def test_stale_curve_within_five_business_days_then_unavailable(store):
    czk = points("USDCZK", "2026-10-12", 21.0, 0.04, 0.035, 1000)
    first = save(store, "2026-10-12", "2026-10-12T17:00:00", spots={"USDCZK": 21.0},
                 depo_curves={"USD": USD}, forward_points={"USDCZK": czk})
    today = save(store, TODAY, f"{TODAY}T12:00:00", depo_curves={"USD": USD})
    curve = DiscountCurves(today, store).curve("CZK")
    assert curve.source == STALE and curve.as_of == "2026-10-12"
    assert curve.rates["3M"] == pytest.approx(0.035)
    assert f"snapshot {first.id} (2026-10-12)" in curve.notes[-1]
    # A week later the 12 October curve is too old: CZK is unavailable.
    later = save(store, "2026-10-23", "2026-10-23T12:00:00", depo_curves={"USD": USD})
    assert DiscountCurves(later, store).curve("CZK").source == UNAVAILABLE
    # Without a store there is nothing to fall back to.
    assert DiscountCurves(today).curve("CZK").source == UNAVAILABLE


def test_a_pair_without_a_discount_rate_is_unavailable_not_priced_at_a_default(store):
    czk = points("USDCZK", TODAY, 21.0, 0.04, 0.035, 1000)
    eur = points("EURUSD", TODAY, 1.12, 0.02, 0.04, 10000)
    snapshot = save(store, TODAY, f"{TODAY}T12:00:00",
                    spots={"USDCZK": 21.0, "EURUSD": 1.12},
                    vols={"USDCZK": SURFACE, "EURUSD": SURFACE},
                    forward_points={"USDCZK": czk, "EURUSD": eur})
    payload, notes = market(snapshot, "USDCZK")             # no SOFR: nothing to imply from
    assert payload is None and notes[-1] == "USDCZK: unavailable, no CZK discount rate"
    bundle, _ = market_bundle(snapshot, store=store)
    assert bundle["markets"] == {}                           # EURUSD needs SOFR too


def test_basis_alerts_flag_jumps_beyond_the_limit(store):
    def snap(as_of, jpy_quote, try_quote):
        return save(store, TODAY, as_of, spots={"USDJPY": 150.0, "USDTRY": 41.0},
                    depo_curves={"USD": USD, "JPY": {"3M": jpy_quote}, "TRY": {"3M": try_quote}},
                    forward_points={"USDJPY": points("USDJPY", TODAY, 150.0, 0.04, 0.01, 100),
                                    "USDTRY": points("USDTRY", TODAY, 41.0, 0.04, 0.35, 10000)})
    snap(f"{TODAY}T09:00:00", 0.012, 0.36)
    now = snap(f"{TODAY}T12:00:00", 0.019, 0.375)   # JPY basis -70bp; TRY -150bp (EM limit 200)
    alerts = DiscountCurves(now, store).basis_alerts()
    assert len(alerts) == 1 and alerts[0].startswith("JPY 3M: basis moved -70bp")
