"""Apeiron's levels and the rules between them, as data.

``MODULE_LEVELS`` places every module of the library (today's locations) in
its level; the import-boundary test (``tests/test_architecture.py``) reads it
and checks every internal import against ``ALLOWED``. A new module must be
placed here, so the map can never silently fall behind the code.

Levels:

* ``atlas`` - core numerical engine; imports nothing else in the suite;
* ``gaia.market_dates``, ``gaia.market``, ``gaia.book`` - the data foundation; each
  imports only itself;
* ``hyperion`` - analytics, risk, services and cache; combines the above;
* ``athena`` (strategy, optimizer, backtest), ``delphi`` (research) and
  ``olympus`` (UI) - consumers; none imports another. Whether they go only
  through Hyperion's services is decided when they are built, so for now
  they may import any lower level.

Modules move to ``apeiron/<level>/`` one level at a time; their entry here
changes path, not level.
"""

BASE = ("atlas", "gaia.market_dates", "gaia.market", "gaia.book")
CONSUMERS = ("athena", "delphi", "olympus")

#: For each level, the levels its modules may import (besides their own).
ALLOWED = {
    "atlas": set(),
    "gaia.market_dates": set(),
    "gaia.market": set(),
    "gaia.book": set(),
    "hyperion": set(BASE),
    "athena": set(BASE) | {"hyperion"},
    "delphi": set(BASE) | {"hyperion"},
    "olympus": set(BASE) | {"hyperion"},
}

_ATLAS = [
    # smile and pricing numerics at the repository root
    "VannaVolga_ImpliedVol", "perturbed_market", "resolved_smile_cores",
    "resolved_smile_inputs", "smile_perturbed_market", "smile_preparation",
    "vanna_volga_sabr", "wrapped_vol_resolver",
    "pricing", "pricing.pricing", "pricing.greeks", "pricing.vanna", "pricing.digital",
    "pricing.fx_forward", "pricing.greek_catalog",
    "pricing_utils", "pricing_utils.BlackScholes", "pricing_utils.black_scholes_numba",
    "pricing_utils.breeden_litzenberger", "pricing_utils.sabr_numba",
    "pricing_utils.sabr_price",
    "Greeks.cross_greeks", "Greeks.decay_greeks", "Greeks.fixed_greeks",
    "Greeks.smile_greeks", "Greeks.weighted_greeks",
    "surface_models.smile_extension",
    "helper_functions", "helper_functions.VannaVolga_function", "helper_functions.cashflows",
    "helper_functions.delta_convention", "helper_functions.fx_strangle_converter",
    "helper_functions.helper_functions", "helper_functions.input_resolvers",
    "helper_functions.math_utils", "helper_functions.model_constants",
    "helper_functions.premium_quotes", "helper_functions.strike_solvers",
    "helper_functions.vanna_volga_helpers",
    # the market objects the numerics consume
    "market_data.model",
]

_GAIA_MARKET_DATES = [
    "app.curve_roll", "market_data.variance_calendar", "market_data.event_calendar",
    # pair conventions + calendars + option dates: to be split (conventions to
    # gaia.market, calendar logic stays here)
    "helper_functions.fx_conventions",
    "helper_functions.fx_discounting",
    # Bloomberg checks of the date rules (edge cases, historical expiries)
    "market_data.bloomberg_edge_cases", "market_data.bloomberg_historical_expiry_dates",
]

_GAIA_MARKET = [
    "market_data", "market_data.bloomberg_fetch", "market_data.depo_tickers",
    "market_data.discount_curves", "market_data.events", "market_data.fwd_tickers",
    "market_data.implied_rates", "market_data.market_tables", "market_data.sample_market",
    "market_data.settlement_holidays",
    "market_data.snapshot_filter", "market_data.store", "market_data.universe",
    "market_data.vol_data", "market_data.vol_tickers",
    "app.market_store", "app.usd_spots",
    # Bloomberg probes some checkouts keep (untracked)
    "market_data.swap_dates", "market_data.bloomberg_holiday_calendar",
]

_GAIA_BOOK = ["app.trades", "app.trades.model", "app.trades.store", "app.option_types"]

_HYPERION = [
    "app", "app.calibration_cache", "app.calibration_plan", "app.event_calendar",
    "app.hedge_delta", "app.market_context", "app.market_layers", "app.market_service",
    "app.market_set", "app.pair_input", "app.premium_side", "app.pricer", "app.progress",
    "app.smile_service", "app.spot_forwards", "app.store_markets", "app.vol_override",
    "app.trades.service", "app.reports", "app.reports.risk_grid", "app.reports.spot_ladder",
]

_OLYMPUS = [
    "app.streamlit_app", "app.ui", "app.ui.app", "app.ui.event_calendar_block",
    "app.ui.greek_picker", "app.ui.market_block", "app.ui.market_picker",
    "app.ui.model_block", "app.ui.negatives", "app.ui.portfolio_block",
    "app.ui.progress_bar", "app.ui.risk_block", "app.ui.spot_forward_block",
    "app.ui.store_sync", "app.ui.theme", "app.ui.trades_block",
]

_DELPHI = [
    "research.Vanna_Volga_Implement", "research.digital_options",
]

MODULE_LEVELS = {
    **{m: "atlas" for m in _ATLAS},
    **{m: "gaia.market_dates" for m in _GAIA_MARKET_DATES},
    **{m: "gaia.market" for m in _GAIA_MARKET},
    **{m: "gaia.book" for m in _GAIA_BOOK},
    **{m: "hyperion" for m in _HYPERION},
    **{m: "olympus" for m in _OLYMPUS},
    **{m: "delphi" for m in _DELPHI},
}

#: Packages whose ``__init__`` only re-exports a module: importing names from
#: the package is importing that module (``from market_data import MarketData``).
RE_EXPORTS = {"market_data": "market_data.model"}

#: Whole folders placed by prefix (scripts and studies, not imported by the library).
PREFIX_LEVELS = {"research.": "delphi", "apeiron.atlas": "atlas",
                 "apeiron.gaia.market_dates": "gaia.market_dates", "apeiron.gaia.market": "gaia.market",
                 "apeiron.gaia.book": "gaia.book", "apeiron.hyperion": "hyperion",
                 "apeiron.athena": "athena", "apeiron.delphi": "delphi",
                 "apeiron.olympus": "olympus"}


def level_of(module):
    """The level of ``module`` (dotted name), or ``None`` when it is unplaced."""
    if module in MODULE_LEVELS:
        return MODULE_LEVELS[module]
    for prefix, level in PREFIX_LEVELS.items():
        if module == prefix.rstrip(".") or module.startswith(prefix):
            return level
    return None


def may_import(importer_level, imported_level):
    """True when a module of ``importer_level`` may import ``imported_level``."""
    return importer_level == imported_level or imported_level in ALLOWED.get(importer_level, ())
