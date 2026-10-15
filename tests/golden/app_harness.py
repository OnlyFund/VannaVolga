"""Run the Pricer's pricing path for golden tests, outside Streamlit.

The app prices through ``app.market_context.MarketContext`` and the
functions in ``app.pricer``; this harness drives the same code with one
market and the sidebar defaults, so golden tests record what the running app
computes rather than a re-implementation of it.
"""
from pathlib import Path

import pandas as pd

from app import pricer
from app.market_context import MarketContext, PricingSettings
from app.market_store import read_market_file
from app.reports.spot_ladder import calculate_spot_ladder
from market_data import Modifications

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def load_markets(name):
    """Return ``{pair: market dict}`` from a fixture file, failing on any bad pair."""
    markets, errors = read_market_file((FIXTURES / "markets" / name).read_bytes())
    if errors:
        raise ValueError(f"Fixture {name} has invalid markets: {errors}")
    return markets


class AppHarness:
    """One pair's market as the app would hold it after loading the file."""

    def __init__(self, market_payload, pricing_model="vv", premium_quote="term_cash",
                 beta_atm=0.7, beta_rr=1.2, beta_bf=0.0, delta_reporting="forward"):
        settings = PricingSettings(pricing_model=pricing_model, delta_reporting=delta_reporting)
        self.context = MarketContext.from_payload(market_payload, settings)
        self.pair = self.context.currency_pair
        self.valuation_date = self.context.valuation_date
        self.spot = self.context.spot
        self.mods = Modifications(beta_atm=beta_atm, beta_rr=beta_rr, beta_bf=beta_bf)
        self.pricing_model = pricing_model
        self.premium_quote = premium_quote

    def price_rows(self, rows):
        """Price ``rows`` exactly as the Pricer's "Price portfolio" button does."""
        return pricer.price_lines(self.context, pd.DataFrame(rows), self.mods,
                                  self.premium_quote)

    def build_delta_hedges(self, rows, mods):
        return pricer.build_delta_hedges(self.context, rows, mods)

    def spot_ladder(self, rows, steps=1, increment_pct=1.0, mode="optimized",
                    forward_positions=()):
        frame = pd.DataFrame(rows)
        spots = [self.spot * (1.0 + step * increment_pct / 100.0)
                 for step in range(-steps, steps + 1)]
        settings = self.context.settings
        return calculate_spot_ladder(
            list(frame.iterrows()), spots, self.valuation_date,
            self.context.market_for_expiry, self.context.resolve_market_expiry,
            self.pricing_model, settings.atm_type, settings.delta_base, self.mods,
            mode=mode, forward_positions=forward_positions, scenario_workers=1,
        )
