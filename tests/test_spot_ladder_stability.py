import os
import sys
from datetime import date
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.reports.spot_ladder import _compare_ladder_results, calculate_spot_ladder
from market_data import MarketData, Modifications
from helper_functions.vanna_volga_helpers import GetMainPillars
from Greeks.smile_greeks import SmileGreeks
from surface_models.smile_extension import SmileExtension


def test_spot_ladder_smile_is_smooth_across_spots():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
        smooth_pillar_interpolation=True,
    )

    ladder = []
    for spot in np.linspace(0.95, 1.45, 25):
        pillars = GetMainPillars(
            k=1.18,
            expiry_date="2026-02-16",
            valuation_date="2025-08-14",
            atm_conv="dn",
            deltaBase=False,
            s=spot,
            fwd_pts=market.fwd_pts,
            v_atm=market.v_atm,
            rr_25=market.rr_25,
            rr_10=market.rr_10,
            bf_25=market.bf_25,
            bf_10=market.bf_10,
            forward_divisor=market.forward_divisor,
            smooth_pillar_interpolation=True,
        )
        ordered = sorted(pillars.get_main_strikes_vols().items())
        ladder.append([vol for _, vol in ordered])

    ladder = np.asarray(ladder, dtype=float)
    assert np.all(np.isfinite(ladder))
    assert np.max(np.abs(np.diff(ladder, axis=0))) < 2.0
    assert np.max(np.abs(np.diff(ladder, axis=1))) < 3.5


def test_smile_price_delta_and_gamma_match_finite_difference():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
        smooth_pillar_interpolation=True,
    )
    mods = Modifications(beta_atm=0.0, beta_rr=0.0, beta_bf=0.0)
    smile = SmileGreeks(
        market=market,
        mods=mods,
        k=1.18,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        CallPut="CALL",
        pricing_model="vv",
        deltaBase=False,
        atm_type="dn",
    )

    eps = 1e-4
    S = market.s
    price_up = smile._price_given_market(market.copy_with(s=S * (1 + eps)))
    price_dn = smile._price_given_market(market.copy_with(s=S * (1 - eps)))
    price_mid = smile._price_given_market(market.copy_with(s=S))
    delta_fd = (price_up - price_dn) / (2.0 * S * eps)
    gamma_fd = ((price_up - 2.0 * price_mid + price_dn) / ((S * eps) ** 2)) * 0.01

    assert smile.price_delta(eps=eps) == pytest.approx(delta_fd, rel=1e-6, abs=1e-8)
    assert smile.price_gamma(eps_small=eps) == pytest.approx(gamma_fd, rel=1e-6, abs=1e-8)


def test_spot_ladder_pnl_follows_smile_delta():
    valuation = date(2025, 8, 14)
    expiry = date(2026, 2, 16)
    spot = 1.17
    eps = 1e-4

    def market_for_expiry(
        _expiry, as_of=None, allow_short_end=False, spot_value=None,
    ):
        return MarketData(
            s=spot if spot_value is None else float(spot_value),
            fwd_pts=10.0,
            v_atm=7.0,
            rr_25=0.5,
            rr_10=1.0,
            bf_25=0.2,
            bf_10=0.7,
            smooth_pillar_interpolation=True,
        )

    ladder_args = dict(
        rows=[(
            0,
            {
                "currency_pair": "EURUSD",
                "strike": 1.18,
                "expiry": expiry,
                "call_put": "CALL",
                "buy_sell": "BUY",
                "notional": "1m",
                "premium": 0.0,
                "premium_ccy": "USD",
            },
        )],
        ladder_spots=[spot * (1.0 - eps), spot, spot * (1.0 + eps)],
        valuation_date=valuation,
        market_for_expiry=market_for_expiry,
        resolve_expiry=lambda value: value,
        pricing_model="vv",
        atm_type="dn",
        delta_base=False,
        mods=Modifications(beta_atm=0.8, beta_rr=0.3, beta_bf=0.2),
        mode="reference",
    )
    serial_levels, serial_errors = calculate_spot_ladder(
        **ladder_args, scenario_workers=1
    )
    levels, errors = calculate_spot_ladder(
        **ladder_args, scenario_workers=2
    )

    assert errors == serial_errors == []
    assert _compare_ladder_results(serial_levels, levels)
    levels = sorted(levels, key=lambda level: level["spot"])
    lower, middle, upper = levels
    lower_term_pnl = lower["pnl_base"] * lower["spot"]
    upper_term_pnl = upper["pnl_base"] * upper["spot"]
    ladder_delta = (upper_term_pnl - lower_term_pnl) / (upper["spot"] - lower["spot"])

    assert middle["delta"] == pytest.approx(ladder_delta, rel=2e-3)
    assert abs(middle["delta"] - middle["fixed_delta"]) > 1.0


def test_smile_extension_reuses_fresh_local_skews_only_within_one_evaluation():
    market = MarketData(
        s=1.17,
        fwd_pts=10.0,
        v_atm=7.0,
        rr_25=0.5,
        rr_10=1.0,
        bf_25=0.2,
        bf_10=0.7,
        smooth_pillar_interpolation=True,
    )
    smile = SmileExtension(
        market=market,
        K=1.18,
        expiry_date="2026-02-16",
        valuation_date="2025-08-14",
        mods=Modifications(
            beta_atm=0.8, beta_rr=0.3, beta_bf=0.2,
            shifted_spot=market.s * 1.0001,
        ),
        deltaBase=False,
    )

    with patch.object(
        smile, "get_local_skews", wraps=smile.get_local_skews
    ) as get_local_skews:
        vol = smile.get_vol_for_strike()

    assert np.isfinite(vol)
    assert get_local_skews.call_count == 1
