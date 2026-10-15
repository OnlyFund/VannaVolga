import numpy as np
import pytest

from resolved_smile_cores import make_core
from resolved_smile_inputs import MarketSnapshot, build_pillars, ResolvedQuotes, resolve_conventions
from test_resolved_smile_inputs import market
from wrapped_vol_resolver import WrappedVolResolver


@pytest.mark.parametrize("pair,expiry", [
    ("EURUSD", "2028-10-02"),
    ("USDMXN", "2026-12-02"),
])
@pytest.mark.parametrize("model", ["vv", "vv-sabr"])
def test_no_broker_core_matches_legacy_at_pillars_intervals_and_wings(
        pair, expiry, model):
    raw = MarketSnapshot.capture(market(pair))
    context = resolve_conventions(raw, expiry, "2026-10-02")
    quotes = ResolvedQuotes(raw.v_atm, raw.rr_25, raw.rr_10,
                            raw.bf_25, raw.bf_10)
    pillars = build_pillars(context, quotes, raw.s)
    core = make_core(model, context, pillars)
    resolver = WrappedVolResolver(
        raw.to_market(), expiry, "2026-10-02", model=model,
        broker_strangle=False,
        benchmark_kwargs={
            "deltaBase": context.premium_adjusted,
            "atm_conv": context.atm_convention,
        },
        fallback=False)
    # Wings beyond the 10-delta pillars intentionally differ from legacy.
    strikes = np.linspace(pillars.strikes[0], pillars.strikes[-1], 19)
    for strike in strikes:
        actual = core.get_vol_for_strike(strike)
        expected = resolver.benchmark(strike)
        assert actual == pytest.approx(expected, rel=3e-5, abs=5e-4), (
            "{} {} mismatch at {}: core={}, legacy={}".format(
                model, pair, strike, actual, expected))


@pytest.mark.parametrize("model", ["vv", "vv-sabr"])
def test_core_uses_explicit_spot_delta_context(model):
    raw = MarketSnapshot.capture(market())
    context = resolve_conventions(raw, "2026-12-02", "2026-10-02")
    assert context.delta_convention == "spot"
    pillars = build_pillars(
        context,
        ResolvedQuotes(raw.v_atm, raw.rr_25, raw.rr_10,
                       raw.bf_25, raw.bf_10),
        raw.s)
    core = make_core(model, context, pillars)
    assert np.all(np.isfinite(core.get_vol_for_strikes(pillars.strikes)))


def _skewed_core(expiry, model="vv"):
    from market_data import MarketData
    raw = MarketSnapshot.capture(MarketData(
        s=1.1, fwd_pts=30, v_atm=8, rr_25=-1.0, rr_10=-1.9, bf_25=.2, bf_10=.65,
        currency_pair="EURUSD"))
    context = resolve_conventions(raw, expiry, "2026-10-02")
    pillars = build_pillars(context, ResolvedQuotes(
        raw.v_atm, raw.rr_25, raw.rr_10, raw.bf_25, raw.bf_10), raw.s)
    return make_core(model, context, pillars), context, pillars


def _delta_vol(core, context, delta, phi):
    from resolved_smile_cores import _strike_from_delta
    vol = core.get_vol_for_strike(context.forward)
    for _ in range(100):
        strike = _strike_from_delta(context.forward, vol / 100, context.time,
                                    context.delta_factor, phi * delta, phi,
                                    context.premium_adjusted)
        vol = core.get_vol_for_strike(strike)
    return vol


@pytest.mark.parametrize("expiry", ["2026-11-02", "2027-04-02", "2027-10-04"])
def test_wing_is_continuous_in_vol_and_slope_at_10_delta(expiry):
    core, _, pillars = _skewed_core(expiry)
    h = 1e-5
    for edge, outward in ((0, -1.0), (4, 1.0)):
        k = pillars.strikes[edge]
        inside, at, outside = (core.get_vol_for_strike(k * np.exp(outward * s * h))
                               for s in (-1, 0, 1))
        assert at == pytest.approx(pillars.vols[edge], abs=1e-12)
        assert outside - at == pytest.approx(at - inside, rel=2e-2, abs=1e-6)


@pytest.mark.parametrize("expiry", ["2026-11-02", "2027-04-02", "2027-10-04"])
def test_wing_carries_the_25_10_curvature_and_widens_skew(expiry):
    core, context, pillars = _skewed_core(expiry)
    h = 1e-4
    for edge, inner, outward in ((0, 1, -1.0), (4, 3, 1.0)):
        wing = core._wing(edge)
        k_edge, k_inner = pillars.strikes[edge], pillars.strikes[inner]
        slope = lambda k, s: s * (
            core.get_vol_for_strike(k * np.exp(outward * s * h)) - core.get_vol_for_strike(k)) / h
        expected = (slope(k_edge, -1) - slope(k_inner, 1)) / abs(np.log(k_edge / k_inner))
        assert wing["slope"] == pytest.approx(slope(k_edge, -1), rel=1e-2, abs=1e-3)
        assert wing["curvature"] == pytest.approx(max(expected, 0.0), rel=2e-2, abs=0.5)
        assert np.isfinite(wing["cap"]) and wing["cap"] > 0
    rr = [_delta_vol(core, context, d, 1) - _delta_vol(core, context, d, -1)
          for d in (.10, .05, .02, .01)]
    assert rr[0] == pytest.approx(-1.9, abs=1e-6)
    assert all(b < a for a, b in zip(rr, rr[1:])), rr
    calls = [_delta_vol(core, context, d, 1) for d in (.10, .05, .02, .01)]
    assert all(b >= a - 1e-9 for a, b in zip(calls, calls[1:])), calls


def test_wing_is_linear_beyond_min_delta_and_lee_bounded():
    core, context, pillars = _skewed_core("2027-10-04")
    for edge, outward in ((0, -1.0), (4, 1.0)):
        wing = core._wing(edge)
        x = [wing["cap"] + s for s in (.1, .2, .3)]
        vols = [core.get_vol_for_strike(np.exp(wing["log_edge"] + outward * d)) for d in x]
        assert vols[2] - vols[1] == pytest.approx(vols[1] - vols[0], rel=1e-9, abs=1e-9)
    for strike in (context.forward * 1e-3, context.forward * 1e3):
        vol = core.get_vol_for_strike(strike)
        assert 0 < vol <= 100 * np.sqrt(2 * abs(np.log(strike / context.forward)) / context.time) + 1e-9


def test_wing_parameters_are_built_once_per_side():
    core, _, pillars = _skewed_core("2027-04-02")
    core.get_vol_for_strike(pillars.strikes[0] * .9)
    wing = core._wings[0]
    core.get_vol_for_strike(pillars.strikes[0] * .8)
    assert core._wings[0] is wing and 4 not in core._wings
