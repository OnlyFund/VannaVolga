from math import exp, isfinite
from numbers import Real

from market_data import MarketData
from apeiron.atlas.resolved import _day as _to_date, simple_discount_factor
from helper_functions.helper_functions import year_fraction


class PerturbedMarket:
    # Suggested defaults for 10Δ / 25Δ ratios
    RR_D10_RATIO = 1.85
    BF_D10_RATIO = 3.8

    def __init__(self, base_market: MarketData):
        self.base = base_market

    def _bump(self, attr: str, delta: float) -> MarketData:
        """Generic bump method for any single market attribute."""
        return self.base.copy_with(**{attr: getattr(self.base, attr) + delta})

    # --- RR / BF bumping ---
    def bump_rr(self, delta: float, which="25") -> MarketData:
        return self._bump(f"rr_{which}", delta)

    def bump_bf(self, delta: float, which="25") -> MarketData:
        return self._bump(f"bf_{which}", delta)

    def bump_rr_with_ratio(self, eps=0.01, d10_ratio=None) -> MarketData:
        if d10_ratio is None:
            d10_ratio = self.RR_D10_RATIO
        return self.base.copy_with(
            rr_25=self.base.rr_25 + eps,
            rr_10=self.base.rr_10 + eps * d10_ratio
        )

    def bump_bf_with_ratio(self, eps=0.01, d10_ratio=None) -> MarketData:
        if d10_ratio is None:
            d10_ratio = self.BF_D10_RATIO
        return self.base.copy_with(
            bf_25=self.base.bf_25 + eps,
            bf_10=self.base.bf_10 + eps * d10_ratio
        )

    # Vol scaler - bumps ATM by ``eps`` vol points and scales the whole smile
    # by the same factor, so RR and BF keep their sign and their ATM ratios.
    def scale_all_vols(self, eps: float) -> MarketData:
        atm = self.base.v_atm
        if atm <= 0:
            raise ValueError("ATM volatility must be positive for proportional scaling")
        scale = (atm + eps) / atm
        return self.base.copy_with(
            v_atm=atm * scale,
            rr_25=self.base.rr_25 * scale,
            rr_10=self.base.rr_10 * scale,
            bf_25=self.base.bf_25 * scale,
            bf_10=self.base.bf_10 * scale,
        )

    # Spot & ATM vol bump
    def shift_spot(self, delta_spot: float) -> MarketData:
        return self.base.copy_with(s=self.base.s * (1 + delta_spot))

    def bump_vol(self, delta_vol: float) -> MarketData:
        return self.base.copy_with(v_atm=self.base.v_atm + delta_vol)

    def bump_rate(self, currency, delta_rate, expiry_date, valuation_date,
                  delivery_date=None) -> MarketData:
        """Bump a domestic (term) or foreign (base) rate, keeping spot fixed.

        Rates and bumps are decimal (10bp = 0.001). Infer the base discount
        factor from F/S = D_base/D_term; a bump preserves the other currency's
        factor. Use simple deposit rates to delivery on each currency's basis
        when settlement discounting is enabled and the market carries its
        resolved dates (``market.resolved``), else continuous rates to expiry.
        ``delivery_date`` is a value-dated forward's own delivery date (else
        the delivery follows from the expiry).
        """
        if currency not in ("domestic", "foreign"):
            raise ValueError("Rate currency must be 'domestic' or 'foreign'")
        if (isinstance(delta_rate, bool) or not isinstance(delta_rate, Real)
                or not isfinite(delta_rate)):
            raise ValueError("Rate bump must be finite")
        expiry, valuation = _to_date(expiry_date), _to_date(valuation_date)
        if delivery_date is not None:
            expiry = _to_date(delivery_date)
        if expiry <= valuation:
            raise ValueError("Rate risk requires an expiry after valuation")
        market = self.base
        forward = market.s + market.fwd_pts / market.forward_divisor
        rate = market.term_discount
        resolved = getattr(market, "resolved", None)
        if (market.settlement_discounting and market.currency_pair and resolved is not None
                and resolved.matches(expiry, valuation, delivery_date)
                and resolved.base_year_fraction is not None):
            term_fraction = resolved.year_fraction_from(valuation)
            discount_factor = simple_discount_factor(rate, term_fraction)
            if currency == "domestic":
                rate += delta_rate
                new_discount = simple_discount_factor(rate, term_fraction)
                forward *= discount_factor / new_discount
            else:
                tau = resolved.base_year_fraction
                base_discount = discount_factor * forward / market.s
                # D_base' / D_base = 1 / (1 + dr * tau_base * D_base).
                forward *= simple_discount_factor(delta_rate * base_discount, tau)
        else:
            tau = year_fraction(expiry.isoformat(), valuation.isoformat())
            if currency == "domestic":
                rate += delta_rate
            forward *= exp((1 if currency == "domestic" else -1) * delta_rate * tau)
        return market.copy_with(
            term_discount=rate,
            fwd_pts=(forward - market.s) * market.forward_divisor,
        )

    def scale_vol(self, method='atm', eps=0.01):
        if method == 'atm':
            return self.bump_vol(delta_vol=eps)
        elif method == 'smile':
            return self.scale_all_vols(eps=eps)
        else:
            raise ValueError(f"Unknown method: {method}")

