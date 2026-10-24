"""The per-context market caches (performance step 1) and their fail-safes:
``market_for_expiry`` built once per request, forward curves read once per
curve date, the converted pillar curve kept per curve date and spot."""
import gc

import pandas as pd
import pytest

from app import market_context as mc
from app import pricer
from app.market_context import DeliveryCurve, MarketContext, interpolate_delivery_curve
from app.smile_service import DiagnosticsCollector, _caches
from golden.app_harness import load_markets
from market_data import Modifications

pytestmark = pytest.mark.filterwarnings("ignore")

MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def payload():
    return load_markets("MARKETS_20261015.json")["EURUSD"]


def fresh(payload):
    return MarketContext.from_payload(payload)


def expiry(context, months=2):
    return pd.Timestamp(context.valuation_date) + pd.DateOffset(months=months)


def builds(context, monkeypatch):
    """Uncached builds on ``context`` and its ``with_spot`` copies (which share
    its cache), not on the rolled markets theta builds."""
    calls = []
    original = MarketContext._build_market_for_expiry

    def counted(self, *args, **kwargs):
        if self.__dict__.get("_market_cache_lock") is context.__dict__.get("_market_cache_lock"):
            calls.append(args)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(MarketContext, "_build_market_for_expiry", counted)
    return calls


def fields(market):
    return {name: value for name, value in vars(market).items()
            if name not in ("diagnostics_token", "_prepared_reuse")}


def test_a_request_is_built_once_and_each_caller_gets_its_own_copy(payload, monkeypatch):
    context = fresh(payload)
    context.market_for_expiry(expiry(context, 7))       # the cache exists
    calls = builds(context, monkeypatch)
    first = context.market_for_expiry(expiry(context))
    second = context.market_for_expiry(expiry(context))
    assert len(calls) == 1
    assert first is not second and fields(first) == fields(second)
    first.s = 99.0                                   # a caller changing its copy ...
    assert context.market_for_expiry(expiry(context)).s != 99.0    # ... changes nobody else's


def test_the_key_carries_the_spot_short_end_and_as_of(payload, monkeypatch):
    context = fresh(payload)
    context.market_for_expiry(expiry(context, 7))       # the cache exists
    calls = builds(context, monkeypatch)
    day = expiry(context)
    context.market_for_expiry(day)
    moved = context.with_spot(context.spot * 1.01)     # shares the cache
    assert moved.market_for_expiry(day).s == pytest.approx(context.spot * 1.01)
    context.market_for_expiry(day, spot_value=context.spot * 0.99)
    next_day = pd.Timestamp(context.valuation_date) + pd.Timedelta(days=1)
    context.market_for_expiry(day, as_of=next_day)
    assert len(calls) == 4


def test_cached_markets_price_exactly_as_uncached(payload, monkeypatch):
    lines = [dict(currency_pair="EURUSD", strike=strike, expiry=tenor, call_put="CALL",
                  buy_sell="BUY", notional="1m", premium=0.0, premium_ccy="USD")
             for strike in ("atm", "25d") for tenor in ("1M", "3M", "1M")]
    cached, _ = pricer.price_lines(fresh(payload), pd.DataFrame(lines), MODS)
    monkeypatch.setenv(mc.MARKET_CACHE_ENV, "0")
    plain, _ = pricer.price_lines(fresh(payload), pd.DataFrame(lines), MODS)
    assert cached == plain


def test_the_off_switch_builds_every_time(payload, monkeypatch):
    monkeypatch.setenv(mc.MARKET_CACHE_ENV, "0")
    context = fresh(payload)
    calls = builds(context, monkeypatch)              # no lock on either side: all counted
    context.market_for_expiry(expiry(context))
    context.market_for_expiry(expiry(context))
    assert len(calls) == 2 and "_market_cache" not in context.__dict__


def test_the_check_mode_catches_a_wrong_cached_market_and_rebuilds(payload, monkeypatch):
    context = fresh(payload)
    good = context.market_for_expiry(expiry(context))
    (key, (market, records)), = context._market_cache.items()
    market.v_atm += 1.0                              # the cache gone wrong
    monkeypatch.setenv(mc.MARKET_CACHE_CHECK_ENV, "1")
    again = context.market_for_expiry(expiry(context))
    assert again.v_atm == good.v_atm
    assert any(record[1] == "market cache" and "v_atm" in record[2]
               for record in context.diagnostics.records())
    assert key not in context._market_cache          # the wrong entry is dropped


def test_a_reuse_records_the_builds_diagnostics_again(payload):
    context = fresh(payload)
    context._market_cache = mc.OrderedDict()
    day = expiry(context)
    context.market_for_expiry(day)
    key = next(iter(context._market_cache))
    market, _ = context._market_cache[key]
    context._market_cache[key] = (market, (("vv", "curve", "a note from the build"),))
    context.diagnostics = DiagnosticsCollector()     # a new run's collector
    context.market_for_expiry(day)
    assert ("vv", "curve", "a note from the build") in context.diagnostics.records()


def test_a_failed_build_is_not_kept(payload, monkeypatch):
    context = fresh(payload)
    original = MarketContext._build_market_for_expiry
    failing = {"on": True}

    def flaky(self, *args, **kwargs):
        if failing["on"]:
            raise ValueError("not this time")
        return original(self, *args, **kwargs)
    monkeypatch.setattr(MarketContext, "_build_market_for_expiry", flaky)
    with pytest.raises(ValueError):
        context.market_for_expiry(expiry(context))
    failing["on"] = False
    assert context.market_for_expiry(expiry(context)).v_atm > 0


def test_the_cache_is_bounded(payload, monkeypatch):
    monkeypatch.setattr(mc, "MARKET_CACHE_SIZE", 3)
    context = fresh(payload)
    for step in range(6):
        context.market_for_expiry(expiry(context), spot_value=context.spot * (1 + step / 1000))
    assert len(context._market_cache) == 3


def test_the_converted_curve_is_kept_per_curve_date_and_spot(payload, monkeypatch):
    context = fresh(payload)
    conversions = []
    original = context.market_service.convert_curve

    def counted(*args, **kwargs):
        conversions.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(context.market_service, "convert_curve", counted)
    for months in (1, 2, 4, 5):
        context.market_for_expiry(expiry(context, months))
    assert len(conversions) == 1                     # one curve date, one spot
    context.with_spot(context.spot * 1.01).market_for_expiry(expiry(context, 3))
    assert len(conversions) == 2


def test_a_delivery_curve_reads_once_and_fails_as_before():
    rows = pd.DataFrame({"delivery_date": ["2026-11-20", "2026-12-21", "2027-01-20"],
                         "fwd_pts": [10.0, 20.0, 30.0]})
    curve = DeliveryCurve(rows, "delivery_date", "fwd_pts", "Forward points")
    assert curve("2026-12-06") == interpolate_delivery_curve(
        rows, "delivery_date", "fwd_pts", "2026-12-06", "Forward points")
    with pytest.raises(ValueError, match="outside the quoted forward range"):
        curve("2027-02-01")
    unused = DeliveryCurve(rows.assign(fwd_pts=float("nan")), "delivery_date", "fwd_pts", "Forward points")
    with pytest.raises(ValueError, match="valid delivery dates"):
        unused("2026-12-01")                         # raises when used, not when made
    with pytest.raises(ValueError, match="valid delivery dates"):
        unused("2026-12-01")                         # and again on every use


def test_a_context_keeps_its_own_surface_cache_alive(payload):
    context = fresh(payload)
    gc.collect()
    assert _caches.get(context.diagnostics.token) is context._surface_cache
