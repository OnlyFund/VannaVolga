from pandas.tseries.offsets import BDay
from market_data import MarketData, Modifications
import pandas as pd
try:
    from .smile_greeks import SmileGreeks
except ImportError:
    from smile_greeks import SmileGreeks


def get_next_bday_date(start_date):
    """Return the next business day date as YYYY-MM-DD string."""
    return (pd.to_datetime(start_date) + BDay(1)).strftime('%Y-%m-%d')


class DecayGreeks(SmileGreeks):
    """
    Compute one-business-day changes in smile Greeks.

    Each result is ``Greek(next business day) - Greek(valuation date)``.
    The market quotes, spot, and smile-dynamics parameters are held fixed;
    only the valuation date and time to expiry advance.
    """

    def __init__(self, market, mods, k, expiry_date, valuation_date,
                 CallPut='CALL', BuySell='BUY', Notional=1, atm_type='dn',
                 pricing_model='vv', deltaBase=True, **kwargs):
        super().__init__(
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

        # Create base SmileGreeks object (valuation_date)
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

        # Precompute next business day for convenience
        self.next_valuation_date = get_next_bday_date(valuation_date)
        if pd.Timestamp(self.next_valuation_date) >= pd.Timestamp(expiry_date):
            raise ValueError("The next business day must be before expiry")

    # -------------------------------------------------------------------------
    # --- Internal Helper
    # -------------------------------------------------------------------------
    def _get_tomorrow_greeks(self):
        """Return a SmileGreeks object with valuation_date = next business day."""
        return SmileGreeks(
            market=self.market,
            mods=self.mods,
            k=self.k,
            expiry_date=self.expiry_date,
            valuation_date=self.next_valuation_date,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.notional,
            atm_type=self.atm_type,
            pricing_model=self.pricing_model,
            deltaBase=self.deltaBase,
            **self.model_kwargs
        )

    # -------------------------------------------------------------------------
    # --- Core Decay Methods
    # -------------------------------------------------------------------------
    def delta_decay(self):
        """Return next-business-day delta minus current delta."""
        greeks_tom = self._get_tomorrow_greeks()
        return greeks_tom.delta() - self.base_market.delta()

    def gamma_decay(self):
        """Return next-business-day gamma minus current gamma."""
        greeks_tom = self._get_tomorrow_greeks()
        return greeks_tom.gamma() - self.base_market.gamma()

    def dVegaDtime(self):
        """Change in Vega per business day."""
        greeks_tom = self._get_tomorrow_greeks()
        return greeks_tom.vega() - self.base_market.vega()

    def dVannaDtime(self):
        """Change in Vanna per business day."""
        greeks_tom = self._get_tomorrow_greeks()
        return greeks_tom.vanna() - self.base_market.vanna()

    def dVolgaDtime(self):
        """Change in Volga per business day."""
        greeks_tom = self._get_tomorrow_greeks()
        return greeks_tom.volga() - self.base_market.volga()

    def dRegaDtime(self):
        """Change in 10bp RR P&L risk over one business day."""
        greeks_tom = self._get_tomorrow_greeks()
        return greeks_tom.rega() - self.base_market.rega()

    def dSegaDtime(self):
        """Change in 10bp BF P&L risk over one business day."""
        greeks_tom = self._get_tomorrow_greeks()
        return greeks_tom.sega() - self.base_market.sega()


if __name__ == "__main__":
    market = MarketData(
        s=1.1740,
        fwd_pts=128,
        v_atm=6.69,
        rr_25=0.51,
        rr_10=0.91,
        bf_25=0.24,
        bf_10=0.84
    )

    mods = Modifications(beta_atm=0.5, beta_rr=0, beta_bf=0)

    greeks = DecayGreeks(
        market=market,
        mods=mods,
        k=1.25,
        expiry_date="2026-04-06",
        valuation_date="2025-10-06",
        CallPut="CALL",
        pricing_model="vv-sabr",
        Notional=1_000_000
    )

    print("Δ-decay:", greeks.delta_decay())
    print("Γ-decay:", greeks.gamma_decay())
    print("dVega/dt:", greeks.dVegaDtime())
    print("dVanna/dt:", greeks.dVannaDtime())
    print("dVolga/dt:", greeks.dVolgaDtime())
    print("dRega/dt:", greeks.dRegaDtime())
    print("dSega/dt:", greeks.dSegaDtime())
"""One-business-day changes in spot-dependent smile Greeks."""
