from abc import ABC, abstractmethod
import pandas as pd
from helper_functions.vanna_volga_helpers import get_model_map
from helper_functions.fx_discounting import discount_kwargs
from pricing_utils.BlackScholes import BlackScholes
from market_data import MarketData
from perturbed_market import PerturbedMarket


class FixedGreekBase(ABC):
    """Shared pricing and perturbation infrastructure for fixed Greeks.

    Fixed means the market ATM/RR/BF quotes do not change during a spot bump.
    The model is re-evaluated at each bumped spot, so the fixed-strike option
    moves across the unchanged delta-based smile.
    """
    def __init__(self, market: MarketData, k, expiry_date, valuation_date,
                 CallPut='CALL', BuySell='BUY', Notional=1, atm_type='dn',
                 pricing_model='vv', deltaBase=True, **kwargs):
        if pricing_model not in get_model_map():
            raise ValueError(
                f"Invalid pricing_model '{pricing_model}'. Must be one of {list(get_model_map().keys())}"
            )

        self.market = market
        self.k = k
        self.expiry_date = expiry_date
        self.valuation_date = valuation_date
        self.CallPut = CallPut
        self.BuySell = BuySell
        self.notional = Notional
        self.atm_type = atm_type
        self.pricing_model = pricing_model
        self.model_cls = get_model_map(market)[pricing_model]
        self.deltaBase = market.deltaBase if market.wrapped_smile and not market.force_legacy_smile else deltaBase
        self.model_kwargs = kwargs

    def _discount_kwargs(self, market=None, valuation_date=None):
        """Return convention-discounting overrides, or ``{}`` for legacy behaviour.

        Pass the bumped market for rate risk so the discount factor changes
        with the rate. A valuation-date bump (theta) must
        pass its own ``valuation_date`` so the discount factor rolls with it.
        """
        return discount_kwargs(
            self.market if market is None else market,
            self.expiry_date,
            self.valuation_date if valuation_date is None else valuation_date,
        )

    def _to_base_ccy(self, term_amount: float) -> float:
        """Convert a term-currency risk amount into base currency.

        Prices are term currency, so every Greek built from a price difference
        inherits that currency. Market Greeks are quoted in base currency, the
        same currency as delta, so divide by spot.
        """
        return term_amount / self.market.s

    def _price_given_market(self, market: MarketData) -> float:
        model_kwargs = dict(self.model_kwargs)
        model_kwargs["atm_conv"] = self.atm_type
        model_kwargs["deltaBase"] = self.deltaBase
        vol = self.model_cls(
            market=market,
            K=self.k,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            **model_kwargs
        ).get_vol_for_strike()

        return BlackScholes(
            S=market.s,
            K=self.k,
            fwd_pts=market.fwd_pts,
            v=vol,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=market.forward_divisor, term_discount=market.term_discount, **self._discount_kwargs(market)
        ).price()

    def _delta_given_market(self, market: MarketData) -> float:
        """Return the option's quote delta on the unchanged sticky-delta smile."""
        model_kwargs = dict(self.model_kwargs)
        model_kwargs["atm_conv"] = self.atm_type
        model_kwargs["deltaBase"] = self.deltaBase
        vol = self.model_cls(
            market=market,
            K=self.k,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            **model_kwargs
        ).get_vol_for_strike()

        return BlackScholes(
            S=market.s,
            K=self.k,
            fwd_pts=market.fwd_pts,
            v=vol,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=market.forward_divisor, term_discount=market.term_discount, **self._discount_kwargs(market)
        ).delta()

    def _vega_given_market(self, market: MarketData) -> float:
        """Return point vega at the model-implied fixed-strike volatility."""
        model_kwargs = dict(self.model_kwargs)
        model_kwargs["atm_conv"] = self.atm_type
        model_kwargs["deltaBase"] = self.deltaBase
        vol = self.model_cls(
            market=market,
            K=self.k,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            **model_kwargs
        ).get_vol_for_strike()
        return BlackScholes(
            S=market.s,
            K=self.k,
            fwd_pts=market.fwd_pts,
            v=vol,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=market.forward_divisor, term_discount=market.term_discount, **self._discount_kwargs(market)
        ).vega()

    # -------------------- Perturbation helpers --------------------
    def _perturb_spot(self, eps=0.001):
        return {
            "base": self.market,
            "up": PerturbedMarket(self.market).shift_spot(eps),
            "dn": PerturbedMarket(self.market).shift_spot(-eps),
            "eps": eps
        }

    def _perturb_vol(self, eps=0.01):
        return {
            "base": self.market,
            "up": PerturbedMarket(self.market).bump_vol(delta_vol=eps),
            "dn": PerturbedMarket(self.market).bump_vol(delta_vol=-eps),
            "eps": eps
        }

    def _perturb_smile(self, method='atm', eps=0.01):
        return {
            "base": self.market,
            "up": PerturbedMarket(self.market).scale_vol(method=method, eps=eps),
            "dn": PerturbedMarket(self.market).scale_vol(method=method, eps=-eps),
            "eps": eps
        }

    def _perturb_rr(self, eps=0.1, ratio=1.85):
        return {
            "base": self.market,
            "up": PerturbedMarket(self.market).bump_rr_with_ratio(eps=eps, d10_ratio=ratio),
            "dn": PerturbedMarket(self.market).bump_rr_with_ratio(eps=-eps, d10_ratio=ratio),
            "eps": eps
        }

    def _perturb_bf(self, eps=0.1, ratio=None):
        if ratio is None:
            ratio = PerturbedMarket.BF_D10_RATIO
        return {
            "base": self.market,
            "up": PerturbedMarket(self.market).bump_bf_with_ratio(eps=eps, d10_ratio=ratio),
            "dn": PerturbedMarket(self.market).bump_bf_with_ratio(eps=-eps, d10_ratio=ratio),
            "eps": eps
        }

    @abstractmethod
    def calculate(self):
        pass


# -------------------- Standard Greeks --------------------
class FixedRho(FixedGreekBase):
    """Sticky-delta domestic/term rate derivative, scaled to a +10bp move."""

    rate_currency = "domestic"

    def calculate(self, eps=0.001):
        if eps <= 0:
            raise ValueError("Rate finite-difference step must be positive")
        pm = PerturbedMarket(self.market)
        up, dn = (
            pm.bump_rate(self.rate_currency, step, self.expiry_date, self.valuation_date)
            for step in (eps, -eps)
        )
        return self._to_base_ccy(
            (self._price_given_market(up) - self._price_given_market(dn)) / (2 * eps) * 0.001
        )


class FixedPhi(FixedRho):
    """Sticky-delta foreign/base rate derivative, scaled to a +10bp move."""

    rate_currency = "foreign"


class FixedDelta(FixedGreekBase):
    """Sticky-delta spot Greek with unchanged market smile quotes."""

    def calculate(self):
        bumps = self._perturb_spot()
        up_price = self._price_given_market(bumps["up"])
        dn_price = self._price_given_market(bumps["dn"])
        return (up_price - dn_price) / (2 * self.market.s * bumps["eps"])


class FixedGamma(FixedGreekBase):
    """Quote-delta sensitivity to spot on the unchanged sticky-delta smile.

    The result is the base-currency delta change for a 1% relative spot move,
    matching the quote-delta gamma convention used by :class:`BlackScholes`.
    """

    def calculate(self):
        bumps = self._perturb_spot()
        up_delta = self._delta_given_market(bumps["up"])
        dn_delta = self._delta_given_market(bumps["dn"])
        d_delta_dS = (up_delta - dn_delta) / (2 * self.market.s * bumps["eps"])
        return d_delta_dS * self.market.s * 0.01


class FixedVega(FixedGreekBase):
    """Price sensitivity to an ATM or proportional smile-volatility bump."""
    def __init__(self, market, k, expiry_date, valuation_date,
                 CallPut='CALL', BuySell='BUY', Notional=1,
                 atm_type='dn', deltaBase=True,
                 pricing_model='vv', method='atm', **kwargs):
        self.method = method
        super().__init__(market, k, expiry_date, valuation_date,
                         CallPut, BuySell, Notional, atm_type,
                         pricing_model=pricing_model, deltaBase=deltaBase, **kwargs)

    def calculate(self):
        bumps = self._perturb_smile(method=self.method)
        up_price = self._price_given_market(bumps["up"])
        dn_price = self._price_given_market(bumps["dn"])
        return self._to_base_ccy((up_price - dn_price) / (2 * bumps["eps"]))


class FixedTheta(FixedGreekBase):
    """One-business-day price decay with the market smile held unchanged."""
    def _shift_date(self):
        date_obj = pd.Timestamp(self.valuation_date)
        shifted_date = (date_obj + pd.offsets.BDay(1)).strftime('%Y-%m-%d')
        return {"base": self.valuation_date, "shift": shifted_date}

    def calculate(self):
        dates = self._shift_date()
        def price_at(valuation_date):
            model_kwargs = dict(self.model_kwargs)
            model_kwargs["atm_conv"] = self.atm_type
            model_kwargs["deltaBase"] = self.deltaBase
            vol = self.model_cls(
                market=self.market,
                K=self.k,
                expiry_date=self.expiry_date,
                valuation_date=valuation_date,
                CallPut=self.CallPut,
                **model_kwargs
            ).get_vol_for_strike()
            return BlackScholes(
                S=self.market.s,
                K=self.k,
                fwd_pts=self.market.fwd_pts,
                v=vol,
                expiry_date=self.expiry_date,
                valuation_date=valuation_date,
                CallPut=self.CallPut,
                BuySell=self.BuySell,
                Notional=self.notional,
                deltaBase=self.deltaBase,
                fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount,
                **self._discount_kwargs(valuation_date=valuation_date)
            ).price()

        return self._to_base_ccy(price_at(dates["shift"]) - price_at(dates["base"]))


# -------------------- Cross Greeks --------------------
class FixedVanna(FixedDelta):
    """Fixed-surface point-vega change for a 1% relative spot move.

    Point vega is recalculated from the model-implied volatility at the
    fixed strike after each spot bump. This deliberately differs from
    ``FixedVega``, which is a market-quote bump risk.
    """
    def calculate(self, spot_eps=0.0001):
        bumps = self._perturb_spot(eps=spot_eps)
        # ``BlackScholes.vega`` converts to base currency at its own spot, so
        # re-express both bumped vegas at the valuation spot. Otherwise the
        # difference also picks up the derivative of that conversion, which is
        # a reporting-unit artifact rather than cross risk.
        vega_up = self._vega_given_market(bumps["up"]) * bumps["up"].s / self.market.s
        vega_dn = self._vega_given_market(bumps["dn"]) * bumps["dn"].s / self.market.s

        d_vega_d_spot = (vega_up - vega_dn) / (2 * self.market.s * spot_eps)
        return d_vega_d_spot * self.market.s * 0.01


class FixedVolga(FixedVega):
    """Second price difference with respect to volatility points."""
    def calculate(self):
        bumps = self._perturb_vol()
        base_px = self._price_given_market(bumps['base'])
        px_up = self._price_given_market(bumps['up'])
        px_dn = self._price_given_market(bumps['dn'])
        return self._to_base_ccy(
            (px_up - 2 * base_px + px_dn) / (bumps['eps'] ** 2)
        )


class FixedRega(FixedGreekBase):
    """Price sensitivity to a risk-reversal quote bump."""
    def __init__(self, market, k, expiry_date, valuation_date,
                 CallPut='CALL', BuySell='BUY', Notional=1,
                 atm_type='dn', deltaBase=True,
                 pricing_model='vv', method='atm', **kwargs):
        self.method = method
        super().__init__(market, k, expiry_date, valuation_date,
                         CallPut, BuySell, Notional, atm_type,
                         pricing_model=pricing_model, deltaBase=deltaBase, **kwargs)

    def calculate(self, eps=0.01):
        bumps = self._perturb_rr(eps=eps)
        px_up = self._price_given_market(bumps['up'])
        px_dn = self._price_given_market(bumps['dn'])
        return self._to_base_ccy((px_up - px_dn) / (2 * bumps['eps']) / 10)


class FixedSega(FixedGreekBase):
    """Price sensitivity to a butterfly quote bump."""
    def __init__(self, market, k, expiry_date, valuation_date,
                 CallPut='CALL', BuySell='BUY', Notional=1,
                 atm_type='dn', deltaBase=True,
                 pricing_model='vv', method='atm', **kwargs):
        self.method = method
        super().__init__(market, k, expiry_date, valuation_date,
                         CallPut, BuySell, Notional, atm_type,
                         pricing_model=pricing_model, deltaBase=deltaBase, **kwargs)

    def calculate(self, eps=0.01):
        bumps = self._perturb_bf(eps=eps)
        px_up = self._price_given_market(bumps['up'])
        px_dn = self._price_given_market(bumps['dn'])
        return self._to_base_ccy((px_up - px_dn) / (2 * bumps['eps']) / 10)

if __name__=='__main__':

    # Usage example:
    market = MarketData(s=1.1641, fwd_pts=129.5, v_atm=7.21, rr_25=0.69, rr_10=1.26, bf_25=0.21, bf_10=0.77)

    print(FixedDelta(
        market=market,
        k=1.20,
        expiry_date='2026-02-13',
        valuation_date='2025-08-13',
        CallPut='CALL',
        Notional=1000000,
        deltaBase=False,
        pricing_model='vv-sabr'
    ).calculate())

    print(FixedVega(
        market=market,
        k=1.242,
        expiry_date='2026-02-16',
        valuation_date='2025-08-12',
        CallPut='CALL',
        Notional=1000000,
        deltaBase=False,
        pricing_model='vv-sabr',method='smile'
    ).calculate())

    print(FixedRega(
        market=market,
        k=1.2420,
        expiry_date='2026-02-16',
        valuation_date='2025-08-14',
        CallPut='CALL',
        Notional=1000000,
        deltaBase=False,
        pricing_model='vv-sabr',method='smile'
    ).calculate())

    print(FixedSega(
        market=market,
        k=1.242,
        expiry_date='2026-02-16',
        valuation_date='2025-08-14',
        CallPut='CALL',
        Notional=1000000,
        deltaBase=False,
        pricing_model='vv-sabr',method='smile'
    ).calculate(eps=0.01))
"""Finite-difference Greeks on an unchanged delta-quoted FX smile."""
