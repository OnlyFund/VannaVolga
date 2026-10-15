"""Model-dependent joint broker-strangle conversion, outside strike cores.

A broker strangle quotes one common volatility for the put and call at the
quoted delta. The smile needs two-vol butterflies instead: the ones whose
five-pillar smile reprices both broker strangles (25D and 10D) at their fixed
common-vol strikes. ``prepare_core`` finds them with a fallback chain; every
step after the first is reported in its ``note`` rather than hidden:

1. Exact joint solve with the requested model, from the scalar three-pivot
   seeds. Accepted only within ``evaluate_matches`` tolerance (1e-8 relative).
2. Models other than VV (SABR has three parameters and cannot generally
   reprice both strangles): the exact VV two-vol butterflies, with the
   requested model built on them. Its own premium error is reported.
3. A SABR smile that cannot be calibrated on the pillars falls back to the VV
   smile on the same pillars.

Every accepted result must pass a failsafe: positive finite wing vols,
ordered pillar strikes, and pillar call prices that are decreasing and convex
in strike (no butterfly arbitrage). If nothing passes, ``PreparationError``
is raised and the caller degrades visibly; the curve layer then fills that
pillar from its neighbours (``app.market_service.convert_pillars``).

When VV itself fails, it is the quotes, not the solver: for very steep skews
with small butterflies no butterfly pair reprices both strangles (the 25D
and 10D zero-residual curves do not cross), so no retry strategy can help.
"""
import math
from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import least_squares

from helper_functions.fx_strangle_converter import (
    _N, _price, _strike_from_delta, broker_to_twovol_bf,
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


def _relative(matches):
    return tuple(match.residual / match.target.premium for match in matches)


def _matched(matches):
    return all(abs(match.residual) <= match.tolerance for match in matches)


def check_pillars_arbitrage_free(context, pillars):
    """Raise unless pillar call prices are decreasing and convex in strike.

    Uses undiscounted Black prices at the five pillar strikes and vols, the
    discrete form of the Breeden-Litzenberger condition.
    """
    strikes, vols = pillars.strikes, pillars.vols
    sqrt_t = math.sqrt(context.time)
    prices = []
    for strike, vol in zip(strikes, vols):
        s = vol / 100.0 * sqrt_t
        d1 = (math.log(context.forward / strike) + 0.5 * s * s) / s
        prices.append(context.forward * _N(d1) - strike * _N(d1 - s))
    slopes = [(b - a) / (kb - ka) for a, b, ka, kb
              in zip(prices, prices[1:], strikes, strikes[1:])]
    if not all(math.isfinite(s) and s < 0.0 for s in slopes):
        raise PreparationError("pillar call prices are not decreasing in strike")
    if not all(b > a for a, b in zip(slopes, slopes[1:])):
        raise PreparationError("pillar call prices are not convex in strike (butterfly arbitrage)")


def _core_with_fallback(model, context, pillars, beta):
    """Build ``model`` on ``pillars``; a SABR smile that fails falls back to VV."""
    try:
        return make_core(model, context, pillars, beta), None
    except PreparationError as exc:
        if model == "vv":
            raise
        return (make_core("vv", context, pillars, beta),
                "{} smile unavailable ({}); using the VV smile on the same pillars"
                .format(model, exc))


def _seeds(raw, context):
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
    return np.array(seeds, dtype=float)


def _solve_joint(raw, context, targets, model, beta):
    """Solve the two-vol butterflies that make ``model`` reprice the strangles.

    Returns ``(quotes, pillars, core, matches)`` or raises ``PreparationError``
    naming the closest miss.
    """
    base = ResolvedQuotes(raw.v_atm, raw.rr_25, raw.rr_10, raw.bf_25, raw.bf_10)
    lower = np.array([abs(raw.rr_25) / 2, abs(raw.rr_10) / 2]) - raw.v_atm + 1e-5
    upper = np.array([raw.v_atm * 4 + abs(raw.rr_25), raw.v_atm * 4 + abs(raw.rr_10)])

    def candidate(bfs):
        trial = replace(base, bf_25=float(bfs[0]), bf_10=float(bfs[1]))
        pillars = build_pillars(context, trial, raw.s)
        return trial, pillars, make_core(model, context, pillars, beta)

    def residual(bfs):
        _, _, core = candidate(bfs)
        return np.array(_relative(evaluate_matches(core, targets)))

    try:
        solved = least_squares(
            residual, np.clip(_seeds(raw, context), lower + 1e-6, upper - 1e-6),
            bounds=(lower, upper), xtol=1e-11, ftol=1e-11, gtol=1e-11, max_nfev=100)
        quotes, pillars, core = candidate(solved.x)
        matches = evaluate_matches(core, targets)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        raise PreparationError("joint {} conversion failed: {}".format(model, exc)) from exc
    if not solved.success or not _matched(matches):
        raise PreparationError(
            "joint {} conversion premium residual exceeds tolerance "
            "(success={}, bfs={}, relative residuals={})"
            .format(model, solved.success, tuple(solved.x), _relative(matches)))
    return quotes, pillars, core, matches


def _accept(context, quotes, pillars):
    """Failsafe applied to any converted result before it is used."""
    values = (quotes.atm, quotes.rr_25, quotes.rr_10, quotes.bf_25, quotes.bf_10)
    if not all(math.isfinite(v) for v in values):
        raise PreparationError("converted quotes are not finite")
    check_pillars_arbitrage_free(context, pillars)


def prepare_core(raw, context, model, broker_strangle, beta=.5):
    """Return ``(quotes, pillars, core, matches, note)`` for one expiry.

    ``note`` is ``None`` when the requested model reprices the broker
    strangles exactly (or quotes are already two-vol and the model built);
    otherwise it describes the fallback used. See the module docstring.
    """
    quotes = ResolvedQuotes(raw.v_atm, raw.rr_25, raw.rr_10, raw.bf_25, raw.bf_10)
    if not broker_strangle:
        pillars = build_pillars(context, quotes, raw.s)
        core, note = _core_with_fallback(model, context, pillars, beta)
        return quotes, pillars, core, (), note
    try:
        targets = broker_targets(raw, context)
    except RuntimeError as exc:
        # e.g. a premium-adjusted call delta with no strike at the common vol.
        raise PreparationError("broker strangle strikes: {}".format(exc)) from exc
    notes = []
    try:
        quotes, pillars, core, matches = _solve_joint(raw, context, targets, model, beta)
    except PreparationError as exc:
        if model == "vv":
            raise
        # The requested model cannot reprice both strangles: take the exact
        # VV conversion and build the requested model on its pillars.
        quotes, pillars, _, _ = _solve_joint(raw, context, targets, "vv", beta)
        core, core_note = _core_with_fallback(model, context, pillars, beta)
        matches = evaluate_matches(core, targets)
        notes.append(
            "{} cannot reprice both broker strangles ({}); using the VV two-vol "
            "butterflies, {} strangle premium error 25D {:+.4%}, 10D {:+.4%}".format(
                model, _short(exc), model, *_relative(matches)))
        if core_note:
            notes.append(core_note)
    _accept(context, quotes, pillars)
    return quotes, pillars, core, matches, "; ".join(notes) or None


def _short(exc):
    text = str(exc)
    if "residual exceeds tolerance" in text:
        return "premium residual over tolerance"
    return text.split(":", 1)[-1].strip()[:120]
