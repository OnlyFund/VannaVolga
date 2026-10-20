import pandas as pd
from pricing_utils.BlackScholes import BlackScholes
from market_data import MarketData, Modifications
from smile_perturbed_market import PerturbedMarket
from helper_functions.vanna_volga_helpers import get_model_map
from helper_functions.fx_discounting import discount_kwargs

class SmileGreeks:
    """Greeks for a spot-dependent, recalibrated FX volatility smile.

    Gamma is the base-currency delta change for a 1% relative spot move.
    Vanna is the finite-difference change in base-currency point vega for a
    1% relative spot move. Volga is the change in point-vega for a 10bp
    proportional smile-vol bump. Rega and sega are price changes for 10bp RR
    and BF quote bumps. All cash Greeks are reported in base currency, the
    same currency as delta.
    """
    def __init__(self, market: MarketData, mods: Modifications, k, expiry_date, valuation_date,
                 CallPut='CALL', BuySell='BUY', Notional=1, atm_type='dn',
                 pricing_model='vv', deltaBase=True, **kwargs):
        if pricing_model not in get_model_map():
            raise ValueError(f"Invalid pricing_model '{pricing_model}'. Must be one of {list(get_model_map().keys())}")

        self.market = market
        self.mods = mods
        self.k = k
        self.expiry_date = expiry_date
        self.valuation_date = valuation_date
        self.CallPut = CallPut
        self.BuySell = BuySell
        self.notional = Notional
        self.atm_type = atm_type
        self.pricing_model = pricing_model
        self.deltaBase = market.deltaBase if market.wrapped_smile and not market.force_legacy_smile else deltaBase
        self.model_cls = get_model_map(market)[pricing_model]
        self.model_kwargs = kwargs

    # --- Currency convention helper ---
    def _discount_kwargs(self, market=None):
        """Return convention-discounting overrides, or ``{}`` for legacy behaviour.

        Rate risk must pass the bumped market so its changed term rate also
        changes the settlement discount factor.
        """
        return discount_kwargs(
            self.market if market is None else market,
            self.expiry_date,
            self.valuation_date,
        )

    def _to_base_ccy(self, term_amount: float) -> float:
        """Convert a term-currency risk amount into base currency.

        Prices are term currency, so every Greek built from a price difference
        inherits that currency. Market Greeks are quoted in base currency, the
        same currency as delta, so divide by spot.
        """
        return term_amount / self.market.s

    # --- Core pricing helper ---
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

    # --- Perturbed Market helper ---
    def _get_pm(self) -> PerturbedMarket:
        return PerturbedMarket(
            base_market=self.market,
            mods=self.mods,
            K=self.k,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            debug=False,
            deltaBase=self.deltaBase,
            atm_type=self.atm_type,
            pricing_model=self.pricing_model
        )

    # --- First-order Greeks ---
    def _rate_risk(self, currency, eps):
        if eps <= 0:
            raise ValueError("Rate finite-difference step must be positive")
        pm = self._get_pm()
        up, dn = (pm.rate_market(currency, step) for step in (eps, -eps))
        return self._to_base_ccy(
            (self._price_given_market(up) - self._price_given_market(dn)) / (2 * eps) * 0.001
        )

    def rho(self, eps=0.001):
        """Domestic/term rate derivative per +10bp, including forward betas."""
        return self._rate_risk("domestic", eps)

    def phi(self, eps=0.001):
        """Foreign/base rate derivative per +10bp, including forward betas."""
        return self._rate_risk("foreign", eps)

    def price_delta(self, eps=1e-4):
        """Return a finite-difference price delta for the current smile surface."""
        S = self.market.s
        price_up = self._price_given_market(self.market.copy_with(s=S * (1 + eps)))
        price_dn = self._price_given_market(self.market.copy_with(s=S * (1 - eps)))
        return (price_up - price_dn) / (2 * S * eps)

    def delta(self, eps=1e-4):
        """Return smile-adjusted ``dPrice/dSpot`` using a central bump."""
        pm = self._get_pm()

        vol_up = pm.shift_spot(delta_spot=eps)
        vol_dn = pm.shift_spot(delta_spot=-eps)

        price_up = BlackScholes(
            S=self.market.s * (1 + eps),
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_up,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).price()

        price_dn = BlackScholes(
            S=self.market.s * (1 - eps),
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_dn,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).price()

        # ``eps`` is a relative spot bump; normalize to an absolute spot
        # derivative so this is comparable with FixedDelta and BlackScholes.
        return (price_up - price_dn) / (2 * self.market.s * eps)

    # --- Second-order Greeks ---
    def price_gamma(self, eps_small=1e-4):
        """Return a finite-difference price gamma per 1% absolute spot move."""
        S = self.market.s
        price_up = self._price_given_market(self.market.copy_with(s=S * (1 + eps_small)))
        price_mid = self._price_given_market(self.market.copy_with(s=S))
        price_dn = self._price_given_market(self.market.copy_with(s=S * (1 - eps_small)))
        return ((price_up - 2.0 * price_mid + price_dn) / ((S * eps_small) ** 2)) * 0.01

    def gamma(self, eps_small=1e-4):
        """Return the base-currency delta change for a 1% relative spot move."""
        S = self.market.s
        pm = self._get_pm()

        # Spot up
        S_up = S * (1 + eps_small)
        vol_up = pm.shift_spot(delta_spot=eps_small, new_spot=S_up)
        delta_up = BlackScholes(
            S=S_up,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_up,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).delta()

        # Spot down
        S_dn = S * (1 - eps_small)
        vol_dn = pm.shift_spot(delta_spot=eps_small, new_spot=S_dn)
        delta_dn = BlackScholes(
            S=S_dn,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_dn,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).delta()

        # Gamma per market convention: delta is already a base-currency
        # amount, so scale the spot derivative by ``S * 0.01`` for a 1%
        # relative spot move.
        d_delta_dS = (delta_up - delta_dn) / (2 * self.market.s * eps_small)
        return d_delta_dS * self.market.s * 0.01

    def vega(self):
        """Return price sensitivity to a proportional five-point smile bump."""
        pm = self._get_pm()
        px_vol_up = pm.scale_vol(method='smile', eps=0.01)
        px_vol_dn = pm.scale_vol(method='smile', eps=-0.01)

        px_vol_up = BlackScholes(
            S=self.market.s,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=px_vol_up,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).price()

        px_vol_dn = BlackScholes(
            S=self.market.s,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=px_vol_dn,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).price()

        return self._to_base_ccy((px_vol_up - px_vol_dn) / (2 * 0.01))

    def theta(self, next_market=None, next_valuation_date=None):
        """Return rolled one-business-day smile theta.

        ``next_market`` must contain the quotes for the option's remaining
        tenor on ``next_valuation_date``.  This makes theta include both the
        passage of calendar time to the next trading day and re-interpolation
        of the ATM/RR/BF and forward curves. Spot stays unchanged; the rolled
        forward changes the fixed strike's position on the skew.
        """
        if next_market is None or next_valuation_date is None:
            raise ValueError(
                "next_market and next_valuation_date are required for rolled theta"
            )
        next_date = pd.Timestamp(next_valuation_date)
        expiry = pd.Timestamp(self.expiry_date)
        if next_date >= expiry:
            raise ValueError("The next business day must be before expiry")

        next_greeks = SmileGreeks(
            market=next_market,
            mods=self.mods,
            k=self.k,
            expiry_date=self.expiry_date,
            valuation_date=next_valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            atm_type=self.atm_type,
            pricing_model=self.pricing_model,
            deltaBase=self.deltaBase,
            **self.model_kwargs
        )
        return self._to_base_ccy(
            next_greeks._price_given_market(next_market)
            - self._price_given_market(self.market)
        )

    def vanna(self, eps_spot=1e-4):
        """Return point-vega change for a 1% relative spot move.

        The base-currency conversion inside ``BlackScholes.vega`` is held at
        the valuation spot so the difference is pure cross risk.
        """
        pm = self._get_pm()
        s_up, s_dn = self.market.s * (1 + eps_spot), self.market.s * (1 - eps_spot)
        vol_up, vol_dn = pm.shift_spot(delta_spot=eps_spot), pm.shift_spot(delta_spot=-eps_spot)

        vega_up = BlackScholes(
            S=s_up,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_up,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).vega()

        vega_dn = BlackScholes(
            S=s_dn,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_dn,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).vega()

        d_vega_d_spot = (
            vega_up * s_up / self.market.s - vega_dn * s_dn / self.market.s
        ) / (2 * self.market.s * eps_spot)
        return d_vega_d_spot * self.market.s * 0.01

    def volga(self, eps_vol=0.10, method='smile'):
        """Return the point-vega change for a 10bp smile-vol bump."""
        pm = self._get_pm()
        vol_up, vol_dn = pm.scale_vol(method=method, eps=eps_vol), pm.scale_vol(method=method, eps=-eps_vol)

        vega_up = BlackScholes(
            S=self.market.s, K=self.k, fwd_pts=self.market.fwd_pts, v=vol_up,
            expiry_date=self.expiry_date, valuation_date=self.valuation_date,
            CallPut=self.CallPut, BuySell=self.BuySell, Notional=self.notional,
            deltaBase=self.deltaBase, fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).vega()

        vega_dn = BlackScholes(
            S=self.market.s, K=self.k, fwd_pts=self.market.fwd_pts, v=vol_dn,
            expiry_date=self.expiry_date, valuation_date=self.valuation_date,
            CallPut=self.CallPut, BuySell=self.BuySell, Notional=self.notional,
            deltaBase=self.deltaBase, fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).vega()

        return (vega_up - vega_dn) / 2.0

    def rega(self, eps=0.10):
        """Return the price change for a 10bp RR quote move."""
        pm = self._get_pm()
        vol_up = pm.bump_rr_with_ratio(eps)
        vol_dn = pm.bump_rr_with_ratio(-eps)

        px_up = BlackScholes(
            S=self.market.s,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_up,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).price()

        px_dn = BlackScholes(
            S=self.market.s,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_dn,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).price()

        return self._to_base_ccy((px_up - px_dn) / 2.0)

    def sega(self, eps=0.10):
        """Return the price change for a 10bp BF quote move."""
        pm = self._get_pm()
        vol_up = pm.bump_bf_with_ratio(eps)
        vol_dn = pm.bump_bf_with_ratio(-eps)

        px_up = BlackScholes(
            S=self.market.s,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_up,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).price()

        px_dn = BlackScholes(
            S=self.market.s,
            K=self.k,
            fwd_pts=self.market.fwd_pts,
            v=vol_dn,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.market.forward_divisor, term_discount=self.market.term_discount, **self._discount_kwargs()
        ).price()

        return self._to_base_ccy((px_up - px_dn) / 2.0)


if __name__ == '__main__':
    market = MarketData(
        s=16.59,
        fwd_pts=4215,
        v_atm=12.07,
        rr_25=2.01,
        rr_10=4.07,
        bf_25=0.343,
        bf_10=1.148
    )

    mods = Modifications(beta_atm=0, beta_rr=0.5, beta_bf=0)

    greeks = SmileGreeks(
        market=market,
        mods=mods,
        k=17.00,
        expiry_date="2027-09-28",
        valuation_date="2026-10-07",
        CallPut="CALL",
        pricing_model="vv",Notional=700000000
    )

    print("Delta:", greeks.delta())
    print("Gamma:", greeks.gamma())
    print("Vega:", greeks.vega())
    print('Vanna:',greeks.vanna())
    print('Volga:',greeks.volga(method='smile'))
    print('rega:', greeks.rega())
    print('sega:',greeks.sega())
    print('theta:',greeks.theta(next_market=market,next_valuation_date="2026-10-08"))
"""Finite-difference Greeks for spot-dependent FX volatility smiles."""
