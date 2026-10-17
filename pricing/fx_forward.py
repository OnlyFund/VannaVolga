"""Valuation for a linear FX forward position."""

from dataclasses import dataclass
from math import exp, isfinite


def term_discount_factor(market, expiry, valuation_date):
    """Term-currency discount factor to the forward's delivery date.

    The same factor the option pricers discount premiums with (settlement
    conventions via ``discount_kwargs``, else ``exp(-r_term * T)``), so a
    hedge and the options it hedges are valued on one basis. Falls back to
    1.0 (undiscounted) when the market or dates cannot give one.
    """
    try:
        from helper_functions.fx_discounting import discount_kwargs
        from helper_functions.helper_functions import year_fraction
        factor = discount_kwargs(market, expiry, valuation_date).get("term_discount_factor")
        if factor is None:
            text = lambda value: value.isoformat() if hasattr(value, "isoformat") else str(value)
            years = max(year_fraction(text(expiry), text(valuation_date)), 0.0)
            factor = exp(-float(getattr(market, "term_discount", 0.0) or 0.0) * years)
        factor = float(factor)
        return factor if isfinite(factor) and 0.0 < factor < 10.0 else 1.0
    except Exception:
        return 1.0


@dataclass(frozen=True)
class FxForwardPosition:
    """A signed base-currency amount exchanged at a fixed FX forward rate.

    Positive ``base_amount`` receives base currency and pays term currency.
    ``expiry`` follows the pricer's option-expiry convention; the market
    builder maps it to the corresponding delivery-date forward quote.
    """

    currency_pair: str
    expiry: object
    base_amount: float
    contract_rate: float

    def value(self, market, valuation_date):
        """Return MTM (discounted to delivery at the term rate, like option
        premiums) and delta in term and base currencies."""
        pair = str(self.currency_pair).strip().upper()
        amount = float(self.base_amount)
        strike = float(self.contract_rate)
        spot = float(market.s)
        divisor = float(market.forward_divisor)
        forward = spot + float(market.fwd_pts) / divisor
        values = (amount, strike, spot, divisor, forward)
        if len(pair) != 6 or not all(isfinite(value) for value in values):
            raise ValueError("Forward position and market inputs must be finite and valid")
        if amount == 0.0 or strike <= 0.0 or spot <= 0.0 or divisor <= 0.0:
            raise ValueError("Forward amount, rate, spot, and divisor must be valid")
        if forward <= 0.0:
            raise ValueError("Forward market rate must be positive")

        discount = term_discount_factor(market, self.expiry, valuation_date)
        mtm_term = amount * (forward - strike) * discount
        return {
            "mtm_term": mtm_term,
            "discount_factor": discount,
            "mtm_base": mtm_term / spot,
            # With forward points held fixed across spot shocks, dF/dS = 1.
            "delta_term": amount,
        }
