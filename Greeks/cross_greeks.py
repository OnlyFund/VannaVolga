from market_data import MarketData, Modifications
from helper_functions.vanna_volga_helpers import get_model_map
try:
    from .smile_greeks import SmileGreeks
except ImportError:
    from smile_greeks import SmileGreeks


class CrossGreeks:
    """Calculate smile gamma sensitivity to a proportional vol bump.

    ``dGammaDvol`` returns the first derivative of the smile-adjusted
    quote-delta gamma with respect to volatility points. The gamma convention
    remains quote-delta sensitivity to spot, scaled per 1% spot move.
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
        self.deltaBase = deltaBase
        self.model_cls = get_model_map()[pricing_model]
        self.model_kwargs = kwargs

        self.base_market = SmileGreeks(
            market=market,
            mods=mods,
            k=k,
            expiry_date=expiry_date,
            valuation_date=valuation_date,
            CallPut=CallPut,
            BuySell=BuySell,
            Notional=Notional,
            atm_type=atm_type,
            pricing_model=pricing_model,
            deltaBase=deltaBase,
            **kwargs
        )

    def _scaled_market(self, eps: float) -> MarketData:
        """Return a market copy with all five quoted vols scaled together."""
        atm = self.market.v_atm
        if atm <= 0:
            raise ValueError("ATM volatility must be positive for proportional scaling")

        scale = (atm + eps) / atm
        v10p = (atm + self.market.bf_10 - self.market.rr_10 / 2.0) * scale
        v25p = (atm + self.market.bf_25 - self.market.rr_25 / 2.0) * scale
        v_atm = atm * scale
        v25c = (atm + self.market.bf_25 + self.market.rr_25 / 2.0) * scale
        v10c = (atm + self.market.bf_10 + self.market.rr_10 / 2.0) * scale
        return self.market.copy_with(
            v_atm=v_atm,
            rr_25=v25c - v25p,
            rr_10=v10c - v10p,
            bf_25=(v25c + v25p) / 2.0 - v_atm,
            bf_10=(v10c + v10p) / 2.0 - v_atm,
        )

    def _smile_gamma(self, market: MarketData) -> float:
        """Calculate smile gamma using the same conventions as SmileGreeks."""
        return SmileGreeks(
            market=market,
            mods=self.mods,
            k=self.k,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            atm_type=self.atm_type,
            pricing_model=self.pricing_model,
            deltaBase=self.deltaBase,
            **self.model_kwargs
        ).gamma()

    def dGammaDvol(self, eps=0.01):
        """Return ``d(smile gamma)/d(vol point)`` using a central bump."""
        if eps <= 0:
            raise ValueError("eps must be positive")
        gamma_up = self._smile_gamma(self._scaled_market(eps))
        gamma_dn = self._smile_gamma(self._scaled_market(-eps))
        return (gamma_up - gamma_dn) / (2.0 * eps)





if __name__ == '__main__':
    market = MarketData(
        s=1.1740,
        fwd_pts=128,
        v_atm=6.65,
        rr_25=0.48,
        rr_10=0.87,
        bf_25=0.21,
        bf_10=0.80
    )

    mods = Modifications(beta_atm=1, beta_rr=0, beta_bf=0)

    greeks = CrossGreeks(
        market=market,
        mods=mods,
        k=1.25,
        expiry_date="2026-04-06",
        valuation_date="2025-10-06",
        CallPut="CALL",
        pricing_model="vv",Notional=1000000
    )

    print(greeks.dGammaDvol())
"""Cross-sensitivities between smile volatility and quote-delta gamma."""
