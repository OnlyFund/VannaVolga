from helper_functions.helper_functions import year_fraction
from helper_functions.vanna_volga_helpers import get_model_map
from Greeks.fixed_greeks import FixedGamma, FixedVega
from Greeks.smile_greeks import SmileGreeks
from market_data import MarketData, Modifications

def scale_t(expiry_date, valuation_date, alpha=0.4):
    """Return the repository's expiry-dependent risk scaling factor."""
    time_to_expiry = year_fraction(expiry_date, valuation_date)
    if time_to_expiry <= 0:
        raise ValueError("Expiry must be after valuation date")
    return 0.25 / (time_to_expiry ** alpha)


class WeightedGreeks(FixedGamma):
    """Apply expiry scaling to fixed or smile gamma and fixed vega."""

    def __init__(self, market: MarketData, k, expiry_date, valuation_date,
                 CallPut='CALL', BuySell='BUY', Notional=1, atm_type='dn',
                 pricing_model: str = 'vv', greek_type='smile', deltaBase=True, mods=None, **kwargs) -> None:

        if pricing_model not in get_model_map():
            raise ValueError(f"Invalid pricing_model '{pricing_model}'. Must be one of {list(get_model_map().keys())}")

        self.market = market
        self.k = k
        self.expiry_date = expiry_date
        self.valuation_date = valuation_date
        self.CallPut = CallPut
        self.BuySell = BuySell
        self.notional = Notional
        self.atm_type = atm_type
        self.model_cls = get_model_map()[pricing_model]
        self.deltaBase = deltaBase
        self.model_kwargs = kwargs
        self.pricing_model = pricing_model
        self.greek_type = greek_type
        # default mods if None
        self.mods = mods if mods is not None else Modifications(shifted_spot=self.market.s,beta_atm=0.7, beta_rr=0.5, beta_bf=0.5)

    def weighted_gamma(self):
        """Return selected gamma multiplied by :func:`scale_t`."""
        fixed_gamma = FixedGamma(
            market=self.market,
            k=self.k,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            atm_type=self.atm_type,
            pricing_model=self.pricing_model,
            deltaBase=self.deltaBase
        ).calculate()

        smile_gamma = SmileGreeks(
            market=self.market,
            mods=self.mods,
            k=self.k,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            atm_type=self.atm_type,
            pricing_model=self.pricing_model,
            deltaBase=self.deltaBase
        ).gamma()

        if self.greek_type == 'smile':
            return smile_gamma * scale_t(self.expiry_date, self.valuation_date)
        else:
            return fixed_gamma * scale_t(self.expiry_date, self.valuation_date)

    def weighted_vega(self):
        """Return fixed vega multiplied by :func:`scale_t`."""
        fixed_vega = FixedVega(
            market=self.market,
            k=self.k,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            atm_type=self.atm_type,
            pricing_model=self.pricing_model,
            deltaBase=self.deltaBase
        ).calculate()

        return fixed_vega * scale_t(self.expiry_date, self.valuation_date)


if __name__ == "__main__":
    market = MarketData(
        s=1.1641, fwd_pts=129.5, v_atm=7.21,
        rr_25=0.69, rr_10=1.26, bf_25=0.21, bf_10=0.77
    )
    print(WeightedGreeks(
        market=market, k=1.20,
        expiry_date='2026-02-13', valuation_date='2025-08-13',
        CallPut='CALL', Notional=1000000, deltaBase=False,
        pricing_model='vv', greek_type='fixed'
    ).weighted_gamma())
"""Time-weighted variants of fixed and smile Greeks."""
