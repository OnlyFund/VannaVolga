"""Model-dependent joint broker-strangle conversion, outside strike cores."""
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from helper_functions.fx_strangle_converter import (
    _price, _strike_from_delta, broker_to_twovol_bf,
)
from resolved_smile_inputs import PreparationError, ResolvedQuotes, build_pillars
from resolved_smile_cores import make_core


@dataclass(frozen=True)
class BrokerTarget:
    delta: float
    common_vol: float
    put_strike: float
    call_strike: float
    premium: float


@dataclass(frozen=True)
class PremiumMatch:
    target: BrokerTarget
    model_premium: float
    residual: float
    tolerance: float


def broker_targets(raw, context):
    result = []
    for delta, bf in ((25, raw.bf_25), (10, raw.bf_10)):
        vol = (raw.v_atm + bf) / 100
        if vol <= 0:
            raise ValueError("broker common volatilities must be positive")
        kp = _strike_from_delta(context.forward, vol, context.time, context.delta_factor,
                                -delta / 100, -1, context.premium_adjusted)
        kc = _strike_from_delta(context.forward, vol, context.time, context.delta_factor,
                                delta / 100, 1, context.premium_adjusted)
        premium = (_price(context.forward, kp, vol, context.time, context.domestic_discount, -1)
                   + _price(context.forward, kc, vol, context.time, context.domestic_discount, 1))
        result.append(BrokerTarget(delta, vol * 100, kp, kc, premium))
    return tuple(result)


def evaluate_matches(core, targets):
    ctx = core.context
    result = []
    for target in targets:
        premium = sum(_price(ctx.forward, k, core.get_vol_for_strike(k) / 100,
                             ctx.time, ctx.domestic_discount, phi)
                      for k, phi in ((target.put_strike, -1), (target.call_strike, 1)))
        tolerance = max(1e-13 * ctx.domestic_discount * ctx.forward,
                        1e-8 * target.premium)
        result.append(PremiumMatch(target, premium, premium - target.premium, tolerance))
    return tuple(result)


def prepare_core(raw, context, model, broker_strangle, beta=.5):
    quotes = ResolvedQuotes(raw.v_atm, raw.rr_25, raw.rr_10, raw.bf_25, raw.bf_10)
    if not broker_strangle:
        pillars = build_pillars(context, quotes, raw.s)
        return quotes, pillars, make_core(model, context, pillars, beta), ()
    targets = broker_targets(raw, context)
    seeds = []
    for delta, rr, bf in ((25, raw.rr_25, raw.bf_25), (10, raw.rr_10, raw.bf_10)):
        try:
            seed = broker_to_twovol_bf(
                raw.s, raw.fwd_pts, context.forward_divisor, raw.v_atm, rr, bf,
                delta, context.valuation_date, context.expiry_date,
                context.delivery_date, context.deposit_basis, raw.term_discount * 100,
                is_pa=context.premium_adjusted, atm_conv=context.atm_convention,
                delta_convention=context.delta_convention).bf_2v
        except (ValueError, RuntimeError):
            # Scalar three-pivot conversion is merely a seed, not a fallback
            # result. Only the final five-pillar model premium can be accepted.
            seed = bf
        seeds.append(seed)
    lower = np.array([abs(raw.rr_25) / 2, abs(raw.rr_10) / 2]) - raw.v_atm + 1e-5
    upper = np.array([raw.v_atm * 4 + abs(raw.rr_25), raw.v_atm * 4 + abs(raw.rr_10)])

    def candidate(bfs):
        trial = replace(quotes, bf_25=float(bfs[0]), bf_10=float(bfs[1]))
        pillars = build_pillars(context, trial, raw.s)
        return trial, pillars, make_core(model, context, pillars, beta)

    def residual(bfs):
        _, _, core = candidate(bfs)
        return np.array([match.residual / match.target.premium
                         for match in evaluate_matches(core, targets)])

    try:
        solved = least_squares(
            residual, np.clip(seeds, lower + 1e-6, upper - 1e-6),
            bounds=(lower, upper), xtol=1e-11, ftol=1e-11, gtol=1e-11,
            max_nfev=100)
        quotes, pillars, core = candidate(solved.x)
        matches = evaluate_matches(core, targets)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        raise PreparationError("joint {} conversion failed: {}".format(model, exc)) from exc
    if not solved.success or any(abs(match.residual) > match.tolerance for match in matches):
        errors = tuple(match.residual / match.target.premium for match in matches)
        raise PreparationError(
            "joint {} conversion premium residual exceeds tolerance "
            "(success={}, bfs={}, relative residuals={})"
            .format(model, solved.success, tuple(solved.x), errors))
    return quotes, pillars, core, matches
