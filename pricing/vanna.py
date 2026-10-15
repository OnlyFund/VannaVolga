"""Reload-safe fixed and smile-adjusted point-vega sensitivities."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from helper_functions.vanna_volga_helpers import get_model_map
from pricing_utils.BlackScholes import BlackScholes
from smile_perturbed_market import PerturbedMarket
from helper_functions.fx_discounting import discount_kwargs


def fixed_vanna(market, strike, expiry, valuation, call_put, buy_sell,
                notional, atm_type, pricing_model, delta_base, spot_eps=1e-4):
    """Return point-vega change for a 1% relative spot move.

    This helper intentionally computes point vega directly at each bumped
    spot. It does not call quote-bump ``FixedVega`` and therefore cannot mix
    market-vega and point-vega definitions.
    """
    model_cls = get_model_map(market)[pricing_model]

    def point_vega(spot):
        bumped_market = market.copy_with(s=spot)
        vol = model_cls(
            market=bumped_market,
            K=strike,
            expiry_date=expiry,
            valuation_date=valuation,
            CallPut=call_put,
            atm_conv=atm_type,
            deltaBase=delta_base,
        ).get_vol_for_strike()
        vega = BlackScholes(
            S=spot,
            K=strike,
            fwd_pts=market.fwd_pts,
            v=vol,
            expiry_date=expiry,
            valuation_date=valuation,
            CallPut=call_put,
            BuySell=buy_sell,
            Notional=notional,
            deltaBase=delta_base,
            fwd_divisor=market.forward_divisor,
            term_discount=market.term_discount,
            **(discount_kwargs(bumped_market, expiry, valuation) if market.wrapped_smile else {}),
        ).vega()
        # ``vega()`` converts to base currency at its own spot. Re-express both
        # bumped vegas at the valuation spot so the difference is pure cross
        # risk and not the derivative of the reporting-currency conversion.
        return vega * spot / market.s

    up = point_vega(market.s * (1.0 + spot_eps))
    down = point_vega(market.s * (1.0 - spot_eps))
    return (up - down) / (2.0 * spot_eps) * 0.01


def smile_vanna(market, strike, expiry, valuation, call_put, buy_sell,
                notional, atm_type, pricing_model, delta_base, mods,
                spot_eps=1e-4):
    """Return smile-adjusted point-vega change for a 1% relative spot move."""
    perturbation = PerturbedMarket(
        base_market=market,
        mods=mods,
        K=strike,
        expiry_date=expiry,
        valuation_date=valuation,
        CallPut=call_put,
        deltaBase=delta_base,
        atm_type=atm_type,
        pricing_model=pricing_model,
    )

    def point_vega(spot, relative_bump):
        vol = perturbation.shift_spot(
            delta_spot=relative_bump,
            new_spot=spot,
        )
        vega = BlackScholes(
            S=spot,
            K=strike,
            fwd_pts=market.fwd_pts,
            v=vol,
            expiry_date=expiry,
            valuation_date=valuation,
            CallPut=call_put,
            BuySell=buy_sell,
            Notional=notional,
            deltaBase=delta_base,
            fwd_divisor=market.forward_divisor,
            term_discount=market.term_discount,
            **(discount_kwargs(market.copy_with(s=spot), expiry, valuation) if market.wrapped_smile else {}),
        ).vega()
        # Hold the base-currency conversion at the valuation spot; see
        # ``fixed_vanna``.
        return vega * spot / market.s

    up_spot = market.s * (1.0 + spot_eps)
    down_spot = market.s * (1.0 - spot_eps)
    up = point_vega(up_spot, spot_eps)
    down = point_vega(down_spot, -spot_eps)
    return (up - down) / (2.0 * spot_eps) * 0.01
