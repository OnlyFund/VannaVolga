"""Reusable fixed and smile Greek calculations for option rows."""

import sys
from math import exp
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
    premium_ccy = row.get("premium_ccy")
    base_premium = False
    if premium_ccy is not None:
        pair = str(row.get("currency_pair", "")).strip().upper()
        premium_ccy = str(premium_ccy).strip().upper()
        if len(pair) != 6 or not pair.isalpha():
            raise ValueError("Currency pair must contain two 3-letter ISO codes")
        if premium_ccy not in (pair[:3], pair[3:]):
            raise ValueError("Premium currency must match the pair's base or term currency")
        base_premium = premium_ccy == pair[:3]
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
    if market.delta_reporting is not None:
        from helper_functions.fx_conventions import get_pair
        premium_adjusted = get_pair(_pair_code(row, market)).premium_adjusted
        if option_value_term is None:
            option_value_term = smile_calculator._price_given_market(market)
        for values in (fixed, smile):
            values["delta"] = _report_delta(
                values["delta"], option_value_term, market,
                expiry, valuation, premium_adjusted,
            )
    elif base_premium:
        if option_value_term is None:
            option_value_term = smile_calculator._price_given_market(market)
        premium_exposure = convert_premium_quote(
            option_value_term, "base_cash", market.s
        )
        fixed["delta"] -= premium_exposure
        smile["delta"] -= premium_exposure
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
    from pricing.greek_catalog import DECAY_GREEKS, WEIGHTED_GREEKS, extras

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

                def fixed_gamma(bumped):
                    return FixedGamma(**dict(common, market=bumped)).calculate()
                values["fixed_gamma_vol"] = (fixed_gamma(cross._scaled_market(eps))
                                             - fixed_gamma(cross._scaled_market(-eps))) / (2 * eps)
                values["gamma_vol"] = cross.dGammaDvol(eps)
            except Exception as exc:
                failed("gamma_vol", f"gamma per vol failed: {exc}")

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
