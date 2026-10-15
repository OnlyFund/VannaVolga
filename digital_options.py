from vanna_volga_sabr import VannaVolga_SABR, VannaVolga
from market_data import MarketData
from helper_functions.vol_data import VolData
from pricing_utils.BlackScholes import BlackScholes
from helper_functions.VannaVolga_function import normalize_atm_type
from itertools import islice
from Greeks.smile_greeks import SmileGreeks
import numpy as np


class DigitalOption:

    def __init__(self, market: MarketData, k: float,
                 expiry_date: str, valuation_date: str,
                 CallPut: str, BuySell: str='BUY',
                 notional: float=1.0,
                 isPayoutBase=True,
                 vanilla_width=0.0001,
                 deltaBase=True,
                 atm_conv='dn',isSettle_base=False):

        self.market = market
        self.k = float(k)
        self.expiry_date = expiry_date
        self.valuation_date = valuation_date
        self.CallPut = CallPut.upper()
        self.BuySell = BuySell.upper()
        self.notional = notional
        self.PayoutBase = isPayoutBase
        self.vanilla_width = vanilla_width
        self.deltaBase = deltaBase
        self.atm_conv = normalize_atm_type(atm_conv)
        # --- original asymmetric strike logic preserved ---

        if self.CallPut == 'CALL':
            self.k1 = self.k
            self.k2 = self.k +self.vanilla_width/2
        else:
            self.k1 = self.k
            self.k2 = self.k -self.vanilla_width/2
        self.isSettle_base = isSettle_base
        # Cache vols
        self._vol1 = None
        self._vol2 = None

    # ===============================================================
    # INTERNAL HELPERS
    # ===============================================================

    def _get_vol(self, strike):

        return VannaVolga_SABR(
            market=self.market,
            K=strike,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut='CALL',
            deltaBase=self.deltaBase,
            atm_conv=self.atm_conv
        ).get_vol_for_strike()

    def _load_vols(self):
        if self._vol1 is None:
            self._vol1 = self._get_vol(self.k1)
        if self._vol2 is None:
            self._vol2 = self._get_vol(self.k2)
        return self._vol1, self._vol2

    def _get_notional_per_leg(self):
        """
        Preserve original notional logic:
        - Asymmetric scaling
        - PayoutBase adjustments
        - Multiplied by 2
        """
        k1, k2 = self.k1, self.k2

        sign = 1 if self.BuySell == 'BUY' else -1

        n1 = (self.notional / self.vanilla_width) * sign
        n2 = (self.notional / self.vanilla_width) * -sign

        if self.isSettle_base is True:
            if self.PayoutBase:
                n1 *= k2
                n2 *= k1

            return n1 * 2, n2 * 2
        else:

            return n1 , n2

    def _price_leg(self, K, vol, Notional, BuySellOverride):
        """Call the BlackScholes pricer exactly like original code."""
        return BlackScholes(
            S=self.market.s,
            K=K,
            fwd_pts=self.market.fwd_pts,
            v=vol,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut=self.CallPut,
            BuySell=BuySellOverride,
            Notional=Notional,
            deltaBase=self.deltaBase,
            term_discount=getattr(self.market, "term_discount", 0.05),
        )

    # ===============================================================
    # PRICE
    # ===============================================================

    def get_vanilla_px(self):
        vol1, vol2 = self._load_vols()
        n1, n2 = self._get_notional_per_leg()

        # original BuySell logic preserved
        opt1 = self._price_leg(
            self.k1, vol1, n1,
            'BUY' if self.BuySell == 'BUY' else 'SELL'
        ).price()

        opt2 = self._price_leg(
            self.k2, vol2, n2,
            'SELL' if self.BuySell == 'BUY' else 'BUY'
        ).price()



        return (opt1 - opt2) / self.notional

    # ===============================================================
    # GREEKS
    # ===============================================================

    def _greek(self, greek_method_name):
        vol1, vol2 = self._load_vols()
        n1, n2 = self._get_notional_per_leg()

        bs1 = self._price_leg(
            self.k1, vol1, n1,
            'BUY' if self.BuySell == 'BUY' else 'SELL'
        )
        bs2 = self._price_leg(
            self.k2, vol2, n2,
            'SELL' if self.BuySell == 'BUY' else 'BUY'
        )

        g1 = getattr(bs1, greek_method_name)()
        g2 = getattr(bs2, greek_method_name)()

        return round(g1 - g2, 0)  # preserve original rounding

    def delta(self):
        return self._greek('delta')

    def gamma(self):
        return self._greek('gamma')

    def vega(self):
        return self._greek('vega')

if __name__ == "__main__":
    vd = VolData('vol_data.csv', valuation_date='2025-08-29')
    market = MarketData(s=1.1598,
                        fwd_pts=53.99,
                        v_atm=5.7,
                        rr_25=0.205,
                        rr_10=0.185,
                        bf_25=0.185, bf_10=0.38)

    example_price = DigitalOption(
        market=market,
        k=1.165,
        expiry_date='2025-11-28',
        valuation_date='2025-08-31',
        CallPut='PUT',
        BuySell='BUY',
        isPayoutBase=True,
        notional=1,
        vanilla_width=0.001,
        atm_conv='dn',
        deltaBase=True,
        isSettle_base=True,
    ).get_vanilla_px()
    print(f"Digital option example price: {example_price}")
