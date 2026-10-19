"""Reusable fixed and smile Greek calculations for option rows."""

import sys
from math import exp

import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Greeks.fixed_greeks import (
    FixedDelta, FixedGamma, FixedVega, FixedVanna, FixedVolga,
    FixedTheta, FixedRega, FixedSega,
)
from Greeks.smile_greeks import SmileGreeks
from pricing_utils.BlackScholes import BlackScholes
from smile_perturbed_market import PerturbedMarket
from helper_functions.input_resolvers import resolve_strike
from helper_functions.helper_functions import year_fraction
from pricing.vanna import fixed_vanna, smile_vanna
from helper_functions.fx_discounting import discount_kwargs
from helper_functions.premium_quotes import convert_premium_quote, conventional_premium_ccy


def _date_text(value):
    """Convert Streamlit date values to the core library's ISO date format."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _pair_code(row, market):
    pair = row.get("currency_pair") or getattr(market, "currency_pair", None) or ""
    return str(pair).replace("/", "").replace(" ", "").strip().upper()


def theta_currency(row, market):
    """Return the currency theta is reported in for this option line.

    The pair's conventional premium currency: USD when USD is the term
    currency, otherwise the base currency. It follows the pair rather than the
    line's entered premium currency, so theta sums consistently within a pair.
    Without a valid pair theta stays in the base currency, like the other
    Greeks, and ``None`` is returned.
    """
    return conventional_premium_ccy(_pair_code(row, market))


def calculate_forward_theta(position, market, next_market, valuation, next_valuation):
    """Return the existing forward's rolled P&L in the pair's theta currency."""
    today = position.value(market, valuation)
    tomorrow = position.value(next_market, next_valuation)
    term_theta = tomorrow["mtm_term"] - today["mtm_term"]
    row = {"currency_pair": position.currency_pair}
    if theta_currency(row, market) == _pair_code(row, market)[3:]:
        return term_theta
    return term_theta / market.s


def forward_delta_factor(market, expiry, valuation):
    """Delta reporting units per unit forward hedge notional."""
    if market.delta_reporting != "spot":
        return 1.0
    forward = market.s + market.fwd_pts / market.forward_divisor
    overrides = discount_kwargs(market, expiry, valuation)
    term_factor = overrides.get(
        "term_discount_factor",
        exp(-market.term_discount * year_fraction(_date_text(expiry), _date_text(valuation))),
    )
    return term_factor * forward / market.s


#: Delta convention when the pair's own convention cannot be read (the app's
#: reporting before deltas followed the pair).
DEFAULT_DELTA_CONVENTION = "forward"


def delta_convention(pair, expiry, valuation):
    """``"spot"`` or ``"forward"``: the delta ``pair`` quotes at ``expiry``.

    The rule the smile reads its delta pillars with
    (``resolved_smile_inputs.resolve_conventions``): G10 spot delta up to the
    pair's cutoff (1Y), forward beyond; EM forward at every tenor. Falls back
    to ``DEFAULT_DELTA_CONVENTION`` when the convention cannot be read.
    """
    try:
        import QuantLib as ql
        from helper_functions import fx_conventions as conventions
        from helper_functions.fx_discounting import _to_date, _to_ql
        convention = conventions.get_pair(str(pair).replace("/", "").strip().upper())
        kind = conventions.delta_type(convention.name, _to_ql(_to_date(expiry)),
                                      _to_ql(_to_date(valuation)))
        forward = (convention.fwd_delta_beyond == "0D"
                   or kind in (ql.DeltaVolQuote.Fwd, ql.DeltaVolQuote.PaFwd))
        return "forward" if forward else "spot"
    except Exception:
        return DEFAULT_DELTA_CONVENTION


def convention_deltas(market, row, expiry, valuation, fixed_delta, delta):
    """A line's fixed and smile deltas in spot, forward and the pair's
    convention, from the deltas as reported on ``market``.

    Spot delta is the spot-equivalent amount every total adds up; forward
    delta is the forward amount (at the option's delivery) that hedges it;
    the pair delta is the one the pair quotes at this expiry
    (:func:`delta_convention`), marked ``"S"`` or ``"F"``. Returns ``{}``
    when the deltas cannot be converted (legacy reporting, a bad factor), so
    callers keep the reported deltas.
    """
    try:
        mode = market.delta_reporting
        if mode not in ("spot", "forward"):
            return {}
        factor = forward_delta_factor(market.copy_with(delta_reporting="spot"), expiry, valuation)
        if not (factor > 0 and np.isfinite(factor)):
            return {}
        values = {}
        for prefix, reported in (("fixed_", fixed_delta), ("", delta)):
            reported = float(reported)
            if not np.isfinite(reported):
                return {}
            spot = reported if mode == "spot" else reported * factor
            values[f"{prefix}spot_delta"] = spot
            values[f"{prefix}fwd_delta"] = spot / factor
        convention = delta_convention(_pair_code(row, market), expiry, valuation)
        key = "spot_delta" if convention == "spot" else "fwd_delta"
        values["pair_delta"] = values[key]
        values["fixed_pair_delta"] = values[f"fixed_{key}"]
        values["delta_convention"] = "S" if convention == "spot" else "F"
        return values
    except Exception:
        return {}


def _report_delta(raw_delta, value, market, expiry, valuation, premium_adjusted):
    """Convert the fixed-points price derivative to an FX quote delta."""
    forward = market.s + market.fwd_pts / market.forward_divisor
    # dF/dS is one in the raw bump. Spot quote delta holds FX discount curves
    # fixed instead, giving dF/dS = F/S.
    spot_delta = raw_delta * forward / market.s
    if premium_adjusted:
        spot_delta -= value / market.s
    if market.delta_reporting == "forward":
        return spot_delta / forward_delta_factor(
            market.copy_with(delta_reporting="spot"), expiry, valuation
        )
    return spot_delta


#: Relative spot step of the gamma price grid S * (1 + k * GAMMA_BUMP), k = -2..2.
GAMMA_BUMP = 1e-4


def _base_premium(row):
    """True when the line's premium is paid in the base currency (legacy
    delta reporting nets it out); validates the pair and premium currency."""
    premium_ccy = row.get("premium_ccy")
    if premium_ccy is None:
        return False
    pair = str(row.get("currency_pair", "")).strip().upper()
    premium_ccy = str(premium_ccy).strip().upper()
    if len(pair) != 6 or not pair.isalpha():
        raise ValueError("Currency pair must contain two 3-letter ISO codes")
    if premium_ccy not in (pair[:3], pair[3:]):
        raise ValueError("Premium currency must match the pair's base or term currency")
    return premium_ccy == pair[:3]


def _display_delta(raw_delta, value, market, row, expiry, valuation, base_premium):
    """The delta the app reports, from a raw price derivative ``dV/dS``
    (forward points fixed) and the option value ``value`` (term currency)."""
    if market.delta_reporting is not None:
        from helper_functions.fx_conventions import get_pair
        premium_adjusted = get_pair(_pair_code(row, market)).premium_adjusted
        return _report_delta(raw_delta, value, market, expiry, valuation, premium_adjusted)
    if base_premium:
        return raw_delta - convert_premium_quote(value, "base_cash", market.s)
    return raw_delta


def _path_delta_gamma(price_at, market, row, expiry, valuation, base_premium,
                      eps=GAMMA_BUMP):
    """``(delta, gamma)`` at ``market.s`` along one spot path, as reported.

    ``price_at(spot)`` prices the option along the path: the unchanged smile
    quotes (fixed) or the smile dynamics (smile). Gamma is the change in the
    reported delta for a 1% relative spot move: the derivative of the very
    delta reported on that path, in its spot/forward and premium conventions,
    so a spot ladder's P&L curvature matches it. (Differentiating the
    analytic Black-Scholes delta at the bumped vol, as before, missed the
    vol's own move with spot.) ``market`` gives the spot and the reporting
    conventions.
    """
    spot = market.s
    prices = {k: price_at(spot * (1 + k * eps)) for k in (-2, -1, 0, 1, 2)}

    def reported(center, step):
        raw = (prices[center + step] - prices[center - step]) / (2 * spot * eps * step)
        return _display_delta(raw, prices[center],
                              market.copy_with(s=spot * (1 + center * eps)),
                              row, expiry, valuation, base_premium)
    delta = reported(0, 1)
    gamma = (reported(1, 1) - reported(-1, 1)) / (2 * eps) * 0.01
    return delta, gamma


def _path_gamma(price_at, market, row, expiry, valuation, base_premium, eps=GAMMA_BUMP):
    """Gamma at ``market.s`` along one spot path (see ``_path_delta_gamma``)."""
    return _path_delta_gamma(price_at, market, row, expiry, valuation, base_premium, eps)[1]


def _smile_path_pricer(anchor, common, mods):
    """``price(spot)`` on the smile dynamics anchored at ``anchor``'s spot
    and quotes: the path the smile delta and a spot ladder's P&L follow."""
    smile = SmileGreeks(mods=mods, **dict(common, market=anchor))
    path = smile._get_pm()

    def price(spot):
        return BlackScholes(
            S=spot, K=smile.k, fwd_pts=anchor.fwd_pts, v=path.shift_spot(new_spot=spot),
            expiry_date=smile.expiry_date, valuation_date=smile.valuation_date,
            CallPut=smile.CallPut, BuySell=smile.BuySell, Notional=smile.notional,
            deltaBase=smile.deltaBase, fwd_divisor=anchor.forward_divisor,
            term_discount=anchor.term_discount, **smile._discount_kwargs(),
        ).price()
    return price


def _consistent_gammas(market, row, common, mods, base_premium):
    """``(fixed gamma, smile gamma)`` on the fixed-quote and smile-dynamics paths."""
    expiry, valuation = common["expiry_date"], common["valuation_date"]
    fixed_pricer = FixedDelta(**common)

    def fixed_price(spot):
        return fixed_pricer._price_given_market(market.copy_with(s=spot))

    return (_path_gamma(fixed_price, market, row, expiry, valuation, base_premium),
            _path_gamma(_smile_path_pricer(market, common, mods), market, row, expiry,
                        valuation, base_premium))


def smile_path_delta_gamma(anchor, market, row, mods, pricing_model, atm_type, delta_base):
    """Smile ``(delta, gamma)`` at ``market.s`` along the smile dynamics
    anchored at ``anchor`` (today's market): the path a spot ladder prices
    its P&L on, so the deltas shown at each ladder spot add up to its P&L.
    ``market`` (the scenario market at the ladder spot) sets the reporting.
    """
    common = _greek_inputs(anchor, row, pricing_model, atm_type, delta_base)
    return _path_delta_gamma(_smile_path_pricer(anchor, common, mods), market, row,
                             common["expiry_date"], common["valuation_date"],
                             _base_premium(row))


def _rolled_smile_theta(market, next_market, common, mods, next_valuation):
    """Reprice the fixed strike on the rolled date, forward and smile.

    The horizon is measured in calendar days to the next trading day, including
    the full weekend on Friday. Re-solving strike volatility on ``next_market``
    captures movement along the skew as the forward rolls down.
    """
    next_valuation = _date_text(next_valuation)

    def smile_price(surface, valuation):
        pm = PerturbedMarket(
            base_market=surface,
            mods=mods,
            K=common["k"],
            expiry_date=common["expiry_date"],
            valuation_date=valuation,
            CallPut=common["CallPut"],
            deltaBase=common["deltaBase"],
            atm_type=common["atm_type"],
            pricing_model=common["pricing_model"],
        )
        vol = pm.shift_spot(delta_spot=0.0, new_spot=surface.s)
        return BlackScholes(
            S=surface.s,
            K=common["k"],
            fwd_pts=surface.fwd_pts,
            v=vol,
            expiry_date=common["expiry_date"],
            valuation_date=valuation,
            CallPut=common["CallPut"],
            BuySell=common["BuySell"],
            Notional=common["Notional"],
            deltaBase=common["deltaBase"],
            fwd_divisor=surface.forward_divisor,
            term_discount=surface.term_discount,
            **(discount_kwargs(surface, common["expiry_date"], valuation)
               if surface.wrapped_smile else {}),
        ).price()

    # Prices are term currency; Greeks are reported in base currency to match
    # delta, so convert the rolled price change with the current spot.
    term_theta = smile_price(next_market, next_valuation) - smile_price(
        market, common["valuation_date"]
    )
    return term_theta / market.s


def _greek_inputs(market, row, pricing_model, atm_type, delta_base):
    """The Greek calculators' common keyword arguments for one line.

    A wrapped smile uses its own ATM and delta conventions.
    """
    if getattr(market, "wrapped_smile", False) and not market.force_legacy_smile:
        atm_type, delta_base = market.atm_conv, market.deltaBase
    return dict(
        market=market,
        k=resolve_strike(market, row, atm_type, delta_base),
        expiry_date=_date_text(row["expiry"]),
        valuation_date=_date_text(row["valuation"]),
        CallPut=row["call_put"],
        BuySell=row["buy_sell"],
        Notional=float(row["notional"]),
        atm_type=atm_type,
        pricing_model=pricing_model,
        deltaBase=delta_base,
    )


def calculate_greeks(
    market, row, mode, pricing_model, atm_type, delta_base, mods,
    next_market=None, next_valuation=None, option_value_term=None,
):
    """Calculate Greeks with optional spot/forward quote-delta reporting.

    Explicit ``market.delta_reporting`` uses pair-conventional premium
    adjustment, independent of the entered trade premium currency. Without
    it, legacy base-premium deltas subtract the signed
    option value divided by spot; term-premium deltas remain price derivatives.
    ``option_value_term`` can reuse an already priced value at row notional.
    Rows without premium currency retain the raw finite-difference deltas.
    Theta is reported in the pair's conventional premium currency
    (:func:`theta_currency`); every other Greek is in the base currency.
    Smile pillar conventions are independent of this reporting adjustment.
    """
    expiry = _date_text(row["expiry"])
    valuation = _date_text(row["valuation"])
    base_premium = _base_premium(row)
    if year_fraction(expiry, valuation) == 0.0:
        return {
            key: 0.0
            for name in ("delta", "gamma", "vega", "vanna", "volga", "theta", "rega", "sega")
            for key in (f"fixed_{name}", name)
        }
    common = _greek_inputs(market, row, pricing_model, atm_type, delta_base)
    atm_type, delta_base = common["atm_type"], common["deltaBase"]
    calculators = {
        "delta": FixedDelta,
        "gamma": FixedGamma,
        "vega": FixedVega,
        "vanna": FixedVanna,
        "volga": FixedVolga,
        "theta": FixedTheta,
        "rega": FixedRega,
        "sega": FixedSega,
    }
    fixed = {}
    for name, calculator_cls in calculators.items():
        if name == "vanna":
            fixed[name] = fixed_vanna(
                market=market,
                strike=common["k"],
                expiry=expiry,
                valuation=valuation,
                call_put=common["CallPut"],
                buy_sell=common["BuySell"],
                notional=common["Notional"],
                atm_type=atm_type,
                pricing_model=pricing_model,
                delta_base=delta_base,
            )
        else:
            fixed[name] = calculator_cls(**common).calculate()

    smile_calculator = SmileGreeks(mods=mods, **common)
    smile = {
        "delta": smile_calculator.delta(),
        "gamma": smile_calculator.gamma(),
        "vega": smile_calculator.vega(),
        "vanna": smile_vanna(
            market=market,
            strike=common["k"],
            expiry=expiry,
            valuation=valuation,
            call_put=common["CallPut"],
            buy_sell=common["BuySell"],
            notional=common["Notional"],
            atm_type=atm_type,
            pricing_model=pricing_model,
            delta_base=delta_base,
            mods=mods,
        ),
        "volga": smile_calculator.volga(),
        "theta": _rolled_smile_theta(
            market, next_market, common, mods, next_valuation
        ) if next_market is not None and next_valuation is not None else None,
        "rega": smile_calculator.rega(),
        "sega": smile_calculator.sega(),
    }
    if market.delta_reporting is not None or base_premium:
        if option_value_term is None:
            option_value_term = smile_calculator._price_given_market(market)
        for values in (fixed, smile):
            values["delta"] = _display_delta(values["delta"], option_value_term, market,
                                             row, expiry, valuation, base_premium)
    # Gamma is the derivative of the reported delta on each path. Should that
    # fail, the calculators' gammas (analytic delta at the bumped vol) stay.
    try:
        fixed["gamma"], smile["gamma"] = _consistent_gammas(
            market, row, common, mods, base_premium)
    except Exception:
        pass
    # Theta is a P&L amount, so it is reported in the pair's conventional
    # premium currency (see ``theta_currency``); the calculators return base.
    if theta_currency(row, market) == _pair_code(row, market)[3:]:
        for values in (fixed, smile):
            if values["theta"] is not None:
                values["theta"] *= market.s
    result = {}
    for name in calculators:
        result[f"fixed_{name}"] = fixed[name]
        # The unprefixed column is the smile-adjusted value; this keeps the
        # fixed/smile pair adjacent in the portfolio output.
        result[name] = smile[name]
    return result


def _theta_parts_term(market, next_market, common, mods, next_valuation):
    """``(static, vol roll)`` of a line's smile theta, in the term currency.

    Static: tomorrow's price with today's strike vol, forward points and
    rates (the calendar-day decay). Vol roll: tomorrow's price at the strike
    vol on the next-day surface instead. The rest of the smile theta is carry.
    """
    def strike_vol(surface, valuation):
        pm = PerturbedMarket(
            base_market=surface, mods=mods, K=common["k"], expiry_date=common["expiry_date"],
            valuation_date=valuation, CallPut=common["CallPut"], deltaBase=common["deltaBase"],
            atm_type=common["atm_type"], pricing_model=common["pricing_model"])
        return pm.shift_spot(delta_spot=0.0, new_spot=surface.s)

    def price(vol, valuation):
        return BlackScholes(
            S=market.s, K=common["k"], fwd_pts=market.fwd_pts, v=vol,
            expiry_date=common["expiry_date"], valuation_date=valuation,
            CallPut=common["CallPut"], BuySell=common["BuySell"], Notional=common["Notional"],
            deltaBase=common["deltaBase"], fwd_divisor=market.forward_divisor,
            term_discount=market.term_discount,
            **(discount_kwargs(market, common["expiry_date"], valuation)
               if market.wrapped_smile else {}),
        ).price()

    tomorrow = _date_text(next_valuation)
    vol_today = strike_vol(market, common["valuation_date"])
    vol_tomorrow = strike_vol(next_market, tomorrow)
    held = price(vol_today, tomorrow)
    return held - price(vol_today, common["valuation_date"]), price(vol_tomorrow, tomorrow) - held


def calculate_extra_greeks(
    market, row, names, base_greeks, pricing_model, atm_type, delta_base, mods,
    next_market=None, next_valuation=None,
):
    """``(values, problems)`` for the extra Greeks in ``names``
    (``pricing.greek_catalog.EXTRA_GREEKS``), at the row's notional.

    ``base_greeks`` is the row's ``calculate_greeks`` result (reused, never
    recomputed). ``values`` has ``fixed_<name>`` and ``<name>`` for each
    Greek; one that cannot be computed is ``None`` in both, with the reason in
    ``problems[name]``, and never affects the others or the base Greeks.
    """
    from Greeks.cross_greeks import CrossGreeks
    from Greeks.weighted_greeks import scale_t
    from pricing.greek_catalog import DECAY_GREEKS, THETA_PARTS, WEIGHTED_GREEKS, extras

    wanted = extras(names)
    values, problems = {}, {}

    def failed(name, reason):
        values[f"fixed_{name}"] = values[name] = None
        problems[name] = reason

    expiry, valuation = _date_text(row["expiry"]), _date_text(row["valuation"])
    expired = year_fraction(expiry, valuation) == 0.0

    decays = [name for name in wanted if name in DECAY_GREEKS]
    if decays:
        tomorrow, reason = None, None
        if next_market is None or next_valuation is None:
            reason = "no next-business-day market"
        elif expired or _date_text(next_valuation) >= expiry:
            tomorrow = {key: 0.0 for key in base_greeks}     # expires before tomorrow
        else:
            try:
                rolled = row.copy()
                rolled["valuation"] = next_valuation
                tomorrow = calculate_greeks(next_market, rolled, None, pricing_model,
                                            atm_type, delta_base, mods)
            except Exception as exc:
                reason = f"next-day Greeks failed: {exc}"
        for name in decays:
            greek = name[:-len("_decay")]
            if tomorrow is None:
                failed(name, reason)
                continue
            for key, out in ((f"fixed_{greek}", f"fixed_{name}"), (greek, name)):
                today, later = base_greeks.get(key), tomorrow.get(key)
                values[out] = (None if today is None or later is None
                               else float(later) - float(today))

    if "gamma_vol" in wanted:
        if expired:
            values["fixed_gamma_vol"] = values["gamma_vol"] = 0.0
        else:
            try:
                common = _greek_inputs(market, row, pricing_model, atm_type, delta_base)
                cross = CrossGreeks(mods=mods, **common)
                eps = 0.01
                base_premium = _base_premium(row)
                # Same gamma definition as calculate_greeks, on scaled vols.
                up, dn = (_consistent_gammas(bumped, row, dict(common, market=bumped), mods,
                                             base_premium)
                          for bumped in (cross._scaled_market(eps), cross._scaled_market(-eps)))
                values["fixed_gamma_vol"] = (up[0] - dn[0]) / (2 * eps)
                values["gamma_vol"] = (up[1] - dn[1]) / (2 * eps)
            except Exception as exc:
                failed("gamma_vol", f"gamma per vol failed: {exc}")

    parts = [name for name in wanted if name in THETA_PARTS]
    if parts:
        reason = None
        if next_market is None or next_valuation is None:
            reason = "no next-business-day market"
        elif expired or _date_text(next_valuation) >= expiry:
            for name in parts:
                values[f"fixed_{name}"] = values[name] = 0.0
        else:
            try:
                common = _greek_inputs(market, row, pricing_model, atm_type, delta_base)
                static, roll = _theta_parts_term(market, next_market, common, mods, next_valuation)
                # The premium currency theta is in (calculate_greeks): term
                # amounts as they are, base ones at today's spot.
                if theta_currency(row, market) != _pair_code(row, market)[3:]:
                    static, roll = static / market.s, roll / market.s
                smile_theta, fixed_theta = base_greeks.get("theta"), base_greeks.get("fixed_theta")
                if smile_theta is None or fixed_theta is None:
                    raise ValueError("no theta to split")
                split = {
                    "theta_static": (static, static),
                    "theta_vol_roll": (float(fixed_theta) - static, roll),
                    "theta_carry": (0.0, float(smile_theta) - static - roll),
                }
                for name in parts:
                    values[f"fixed_{name}"], values[name] = split[name]
            except Exception as exc:
                reason = f"theta split failed: {exc}"
        if reason:
            for name in parts:
                failed(name, reason)

    weighted = [name for name in wanted if name in WEIGHTED_GREEKS]
    if weighted:
        try:
            weight = 0.0 if expired else scale_t(expiry, valuation)
        except Exception as exc:
            weight, reason = None, f"expiry weight failed: {exc}"
        for name in weighted:
            greek = name[len("weighted_"):]
            if weight is None:
                failed(name, reason)
                continue
            for key, out in ((f"fixed_{greek}", f"fixed_{name}"), (greek, name)):
                base = base_greeks.get(key)
                values[out] = None if base is None else float(base) * weight
    return values, problems
