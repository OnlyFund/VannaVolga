"""Reusable option pricing functions and the portfolio-line ``Pricable`` API."""

from dataclasses import dataclass
from typing import Optional
from math import exp, isfinite, sqrt

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pricing_utils.BlackScholes import BlackScholes
from pricing_utils.black_scholes_numba import black_scholes_values
from helper_functions.input_resolvers import resolve_strike
from helper_functions.helper_functions import year_fraction
from helper_functions.cashflows import signed_premium_cashflow
from helper_functions.fx_discounting import discount_kwargs
from helper_functions.vanna_volga_helpers import get_model_map
from helper_functions.premium_quotes import (
    PREMIUM_QUOTES,
    safe_convert_premium_quote,
)

try:
    import QuantLib as ql
except ImportError:  # QuantLib is an optional pricing accelerator.
    ql = None


def _date_text(value):
    """Convert Streamlit date values to the core library's ISO date format."""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


def _price_with_fallback(
    market, strike, vol, expiry, valuation, call_put, buy_sell,
    notional, delta_base,
):
    """Use QuantLib for scalar MTM, falling back to Numba then library BS."""
    time = float(year_fraction(_date_text(expiry), _date_text(valuation)))
    spot = float(market.s)
    forward_divisor = float(market.forward_divisor)
    forward = spot + float(market.fwd_pts) / forward_divisor
    strike = float(strike)
    vol = float(vol)
    notional = float(notional)
    call_put = str(call_put).upper()
    buy_sell = str(buy_sell).upper()
    numeric_inputs = (spot, forward_divisor, forward, strike, vol, notional)
    # Empty unless settlement discounting is switched on for a known pair, in
    # which case every engine below uses the same convention-based factor.
    discount_overrides = discount_kwargs(market, expiry, valuation)
    if not all(isfinite(value) for value in numeric_inputs):
        raise ValueError("Black inputs must be finite")
    if spot <= 0.0 or forward_divisor <= 0.0 or forward <= 0.0:
        raise ValueError("Spot, forward divisor, and derived forward must be positive")
    if strike <= 0.0 or vol <= 0.0 or notional < 0.0:
        raise ValueError("Strike and volatility must be positive and notional non-negative")
    if call_put not in {"CALL", "PUT"}:
        raise ValueError("Call/Put must be CALL or PUT")
    if buy_sell not in {"BUY", "SELL"}:
        raise ValueError("Buy/Sell must be BUY or SELL")

    if ql is not None:
        try:
            term_discount = float(market.term_discount)
            if time <= 0.0:
                raise ValueError("Invalid Black inputs")
            option_type = ql.Option.Call if call_put == "CALL" else ql.Option.Put
            unsigned_price = ql.blackFormula(
                option_type,
                float(strike),
                forward,
                float(vol) / 100.0 * sqrt(time),
                discount_overrides.get(
                    "term_discount_factor", exp(-term_discount * time)
                ),
            )
            side = 1.0 if buy_sell == "BUY" else -1.0
            price = side * notional * float(unsigned_price)
            if not isfinite(price):
                raise ValueError("QuantLib Black formula returned a non-finite price")
            return price
        except Exception:
            # Preserve the existing calculation as a dependency/runtime fallback.
            pass
    try:
        return float(black_scholes_values(
            spot=spot,
            strike=strike,
            forward_points=market.fwd_pts,
            volatility_pct=float(vol),
            expiry_date=expiry,
            valuation_date=valuation,
            call_put=call_put,
            buy_sell=buy_sell,
            notional=notional,
            delta_base=delta_base,
            forward_divisor=market.forward_divisor,
            term_discount=market.term_discount,
            **discount_overrides,
        )[0])
    except Exception:
        try:
            price = float(BlackScholes(
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
                fwd_divisor=forward_divisor,
                term_discount=market.term_discount,
                **discount_overrides,
            ).price())
        except Exception as exc:
            raise ValueError("All Black pricing engines failed") from exc
        if not isfinite(price):
            raise ValueError("Black-Scholes fallback returned a non-finite price")
        return price


def price_option(market, row, pricing_model, atm_type, delta_base):
    """Return model volatility and MTM for one option row."""
    if getattr(market, "wrapped_smile", False) and not market.force_legacy_smile:
        atm_type, delta_base = market.atm_conv, market.deltaBase
    model = get_model_map(market)[pricing_model]
    expiry = _date_text(row["expiry"])
    valuation = _date_text(row["valuation"])
    strike = resolve_strike(market, row, atm_type, delta_base)
    if year_fraction(expiry, valuation) == 0.0:
        vol = float(market.v_atm)
    else:
        vol = model(
            market=market,
            K=strike,
            expiry_date=expiry,
            valuation_date=valuation,
            CallPut=row["call_put"],
            BuySell=row["buy_sell"],
            Notional=float(row["notional"]),
            atm_conv=atm_type,
            deltaBase=delta_base,
        ).get_vol_for_strike()
    price = _price_with_fallback(
        market, strike, vol, expiry, valuation, row["call_put"],
        row["buy_sell"], float(row["notional"]), delta_base,
    )
    return float(strike), float(vol), float(price)


def price_option_with_vol(market, row, vol, atm_type, delta_base):
    """Price a prepared option using a previously solved volatility."""
    expiry = _date_text(row["expiry"])
    valuation = _date_text(row["valuation"])
    strike = float(row["strike"])
    price = _price_with_fallback(
        market, strike, vol, expiry, valuation, row["call_put"],
        row["buy_sell"], float(row["notional"]), delta_base,
    )
    return strike, float(vol), float(price)


@dataclass
class Pricable:
    """A single FX option line valued as one unit and scaled by its notional.

    ``premium`` is the absolute cash amount entered by the user. Its sign is
    derived from BUY/SELL, and it is never scaled by the option notional.
    ``None`` asks for the auto premium: the option's value in the premium
    currency, so P&L is zero (the result then carries ``premium_auto`` and
    ``auto_premium``). MTM and Greeks are calculated for one unit and then
    multiplied by notional; the premium changes neither.
    """

    row: object
    premium: Optional[float]
    premium_ccy: str

    def price(
        self, market, pricing_model, atm_type, delta_base, mods=None,
        next_market=None, next_valuation=None, extras=(),
    ):
        """Return the line's MTM, P&L, implied vol, and complete Greek vector.

        ``extras`` names on-demand Greeks (``pricing.greek_catalog``) to add;
        any that fail are ``None`` with the reason in ``greek_problems``.
        """
        from pricing.greeks import calculate_extra_greeks, calculate_greeks, theta_currency

        row = self.row.copy()
        notional = float(row["notional"])
        if notional < 0:
            raise ValueError("Notional must be non-negative")
        pair = str(row.get("currency_pair", "")).strip().upper()
        if len(pair) != 6:
            raise ValueError("Currency pair must contain two 3-letter ISO codes")
        base_ccy, term_ccy = pair[:3], pair[3:]
        premium_ccy = str(self.premium_ccy).strip().upper()
        if premium_ccy not in (base_ccy, term_ccy):
            raise ValueError("Premium currency must match the pair's base or term currency")
        buy_sell = str(row["buy_sell"]).strip().upper()
        auto_premium = self.premium is None
        # Validates the side (and an entered premium) before any pricing.
        signed_premium = signed_premium_cashflow(0.0 if auto_premium else self.premium, buy_sell)

        # Existing pricing and Greek calculators apply notional internally.
        # Request unit values here so this API owns the one scaling step.
        row["notional"] = 1.0
        row["premium_ccy"] = premium_ccy
        strike, vol, unit_mtm = price_option(
            market, row, pricing_model, atm_type, delta_base
        )
        unit_greeks = calculate_greeks(
            market, row, None, pricing_model, atm_type, delta_base, mods,
            next_market=next_market, next_valuation=next_valuation,
            option_value_term=unit_mtm,
        )
        greek_problems = {}
        if extras:
            unit_extras, greek_problems = calculate_extra_greeks(
                market, row, extras, unit_greeks, pricing_model, atm_type, delta_base, mods,
                next_market=next_market, next_valuation=next_valuation,
            )
            unit_greeks = {**unit_greeks, **unit_extras}
        conversion = 1.0 / float(market.s) if premium_ccy == base_ccy else 1.0
        mtm = float(unit_mtm) * notional * conversion
        if auto_premium:
            # The option's value in the premium currency: paid on a buy,
            # received on a sale, so premium + MTM is zero.
            signed_premium = signed_premium_cashflow(abs(mtm), buy_sell)
        mtm_base = float(unit_mtm) * notional / float(market.s)
        mtm_term = float(unit_mtm) * notional
        premium_base = signed_premium if premium_ccy == base_ccy else signed_premium / float(market.s)
        premium_term = signed_premium * float(market.s) if premium_ccy == base_ccy else signed_premium
        scaled_greeks = {
            name: (None if value is None else float(value) * notional)
            for name, value in unit_greeks.items()
        }
        # Quoted premium/MTM are presentation-only.  They must never be able to
        # break a valuation, so every conversion falls back to None.
        strike_value = float(strike)
        quotes = {}
        for quote in PREMIUM_QUOTES:
            quotes["mtm_" + quote] = safe_convert_premium_quote(
                mtm_term, quote, market.s, strike=strike_value, notional=notional,
            )
            quotes["premium_" + quote] = safe_convert_premium_quote(
                premium_term, quote, market.s, strike=strike_value, notional=notional,
            )
            pnl_term_quote = quotes["mtm_" + quote]
            premium_quote = quotes["premium_" + quote]
            quotes["pnl_" + quote] = (
                None if pnl_term_quote is None or premium_quote is None
                else pnl_term_quote + premium_quote
            )
        # Only with extras requested: results without them stay exactly as before.
        problems = {"greek_problems": greek_problems} if extras else {}
        auto = {"premium_auto": True, "auto_premium": abs(signed_premium)} if auto_premium else {}
        return {
            **problems,
            **auto,
            "currency_pair": pair,
            "strike": float(strike),
            "expiry": row["expiry"],
            "notional": notional,
            "vol_pct": float(vol),
            "premium": signed_premium,
            "premium_ccy": premium_ccy,
            "mtm": mtm,
            "mtm_ccy": premium_ccy,
            "pnl": signed_premium + mtm,
            "pnl_ccy": premium_ccy,
            "mtm_base": mtm_base,
            "mtm_base_ccy": base_ccy,
            "mtm_term": mtm_term,
            "mtm_term_ccy": term_ccy,
            "pnl_base": premium_base + mtm_base,
            "pnl_base_ccy": base_ccy,
            "pnl_term": premium_term + mtm_term,
            "pnl_term_ccy": term_ccy,
            "theta_ccy": theta_currency(row, market) or base_ccy,
            **quotes,
            **scaled_greeks,
        }
