"""Valuation for a linear FX forward position."""

from dataclasses import dataclass
from math import isfinite


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
        """Return undiscounted MTM and delta in term and base currencies."""
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

        mtm_term = amount * (forward - strike)
        return {
            "mtm_term": mtm_term,
            "mtm_base": mtm_term / spot,
            # With forward points held fixed across spot shocks, dF/dS = 1.
            "delta_term": amount,
        }
