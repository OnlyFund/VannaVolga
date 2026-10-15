"""Signed, reference-based smile dynamics and visible numerical fallbacks."""
import numpy as np
import pytest

from app.smile_service import DiagnosticsCollector, market_diagnostics
from market_data import MarketData
from smile_perturbed_market import PerturbedMarket
from surface_models.smile_extension import SmileExtension
from wrapped_vol_resolver import DegradedSmileWarning


def market(**changes):
    values = dict(
        s=16.6, fwd_pts=3150., v_atm=11.78, rr_25=2.03, rr_10=4.115,
        bf_25=.4, bf_10=1.3, currency_pair="USDZAR",
        wrapped_smile=True, broker_strangle=False,
    )
    values.update(changes)
    return MarketData(**values)


def extension(m=None, **changes):
    values = dict(
        market=market() if m is None else m, K=16.95,
        expiry_date="2027-07-09", valuation_date="2026-10-09",
        mods={"beta_atm":.8, "beta_rr":.5, "beta_bf":.2},
    )
    values.update(changes)
    return SmileExtension(**values)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("model", ["vv", "vv-sabr"])
def test_skews_have_consistent_log_strike_units(wrapped, model):
    sk = extension(m=market(wrapped_smile=wrapped), pricing_model=model).get_local_skews()
    p, a, c = sk["local_skew"]
    assert sk["rr_skew"] == pytest.approx(c-p)
    assert sk["bf_skew"] == pytest.approx((c+p)/2-a)


def test_wing_shocks_are_signed_and_zero_at_reference():
    sk = dict(local_skew=(2., 3., 8.), rr_skew=6., bf_skew=2.,
              main_vols=np.array([10., 11., 12.]), rr_ratio=2., bf_ratio=3.)
    pv = dict(v_atm=11., v_25p=10., v_25c=12.)
    for move in [-.001, 0., .001]:
        se = extension(mods={"shifted_spot":16.6*np.exp(move),
                             "beta_rr":.5, "beta_bf":.2})
        w = se.extend_wings_convexity(pv, sk)
        assert w["rr_25"] == pytest.approx(2.+.5*move*6.)
        assert w["bf_25"] == pytest.approx(.2*move*2., abs=1e-14)


@pytest.mark.parametrize("model", ["vv", "vv-sabr"])
def test_spot_paths_agree_and_zero_bump_preserves_surface(model):
    m = market()
    pm = PerturbedMarket(m, mods={"beta_atm":.8,"beta_rr":.5,"beta_bf":.2},
                         K=16.95, expiry_date="2027-07-09",
                         valuation_date="2026-10-09", pricing_model=model)
    original = extension(m=m, pricing_model=model)._compute_vol(16.95)
    assert pm.shift_spot(delta_spot=0.) == original
    for bump in [-.0001, .0001]:
        shifted = m.s*(1+bump)
        assert pm.shift_spot(new_spot=shifted) == pytest.approx(
            pm.vol_for_strike(new_spot=shifted), rel=1e-12)
    assert m.s == 16.6


@pytest.mark.parametrize("mods", [
    {"beta_atm":-1.}, {"beta_rr":float("nan")}, {"beta_bf":float("inf")},
    {"shifted_spot":0.}, {"shifted_spot":float("nan")},
])
def test_invalid_dynamics_are_rejected_not_fallbacks(mods):
    with pytest.raises(ValueError):
        extension(mods=mods)


def test_spot_bump_probes_use_unbumped_reference(monkeypatch):
    seen = []
    original = SmileExtension.get_local_skews
    def capture(self):
        seen.append(self.market.s)
        return original(self)
    monkeypatch.setattr(SmileExtension, "get_local_skews", capture)
    pm = PerturbedMarket(market(), mods={"beta_atm":.8}, K=16.95,
                         expiry_date="2027-07-09", valuation_date="2026-10-09")
    pm.shift_spot(delta_spot=.0001)
    pm.shift_spot(delta_spot=-.0001)
    assert seen == [16.6, 16.6]


@pytest.mark.parametrize("wrapped", [False, True])
def test_numerical_failure_falls_back_at_shifted_spot_with_warning(monkeypatch, wrapped):
    collector = DiagnosticsCollector()
    se = extension(m=market(wrapped_smile=wrapped, diagnostics_token=collector.token),
                   mods={"shifted_spot":16.7, "beta_atm":.8})
    def fail():
        raise RuntimeError("synthetic skew failure")
    monkeypatch.setattr(se, "get_local_skews", fail)
    expected = extension(m=se.market.copy_with(s=16.7))._compute_vol(16.95)
    with pytest.warns(DegradedSmileWarning, match="unchanged quotes at shifted spot"):
        assert se.get_vol_for_strike() == expected
    if wrapped:
        assert market_diagnostics(se.market)[0] == "degraded"


def test_programming_errors_are_not_swallowed(monkeypatch):
    se = extension(mods={"shifted_spot":16.7})
    def fail():
        raise AttributeError("synthetic programming error")
    monkeypatch.setattr(se, "get_local_skews", fail)
    with pytest.raises(AttributeError):
        se.get_vol_for_strike()


def test_failed_fallback_propagates(monkeypatch):
    se = extension(mods={"shifted_spot":16.7})
    def fail():
        raise RuntimeError("synthetic skew failure")
    def broken_model(**kwargs):
        raise RuntimeError("fallback also failed")
    monkeypatch.setattr(se, "get_local_skews", fail)
    se.model_cls = broken_model
    with pytest.warns(DegradedSmileWarning):
        with pytest.raises(RuntimeError, match="fallback also failed"):
            se.get_vol_for_strike()


@pytest.mark.parametrize("bad_vol", [float("nan"), 0., -1.])
def test_invalid_repriced_vol_triggers_visible_fallback(bad_vol):
    from types import SimpleNamespace
    se = extension(mods={"shifted_spot":16.7, "beta_atm":.8})
    model = se.model_cls
    def numerical_failure(**kwargs):
        surface = kwargs["market"]
        if surface.s == 16.7 and surface.v_atm != se.market.v_atm:
            return SimpleNamespace(get_vol_for_strike=lambda: bad_vol)
        return model(**kwargs)
    se.model_cls = numerical_failure
    expected = extension(m=se.market.copy_with(s=16.7))._compute_vol(16.95)
    with pytest.warns(DegradedSmileWarning):
        assert se.get_vol_for_strike() == expected
