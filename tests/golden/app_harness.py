"""Run the Pricer's own nested functions from ``render_app`` outside Streamlit.

The market builder and the per-line pricing loop are nested inside
``app.ui.app.render_app`` and read ``st.session_state``. This harness compiles
those exact function definitions from the app source into a namespace that
supplies one market's tables and the sidebar settings, so golden tests record
what the running app computes rather than a re-implementation of it.

When that logic moves into a backend module, only this file should need to
change; the golden expectations must not.
"""
import ast
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

import app.ui.app as ui
from app.market_service import MarketService
from app.market_store import read_market_file, validate_market
from app.smile_service import DiagnosticsCollector, SurfaceCache, bind_cache
from market_data import Modifications

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"

#: Nested ``render_app`` functions the pricer path needs, by name.
_NESTED = (
    "tenor_days", "naive_date", "variance_days_between", "forward_curve",
    "forward_points_for_expiry", "term_discount_for_expiry", "market_for_expiry",
    "resolve_market_expiry", "format_notional", "build_delta_hedges",
)


def load_markets(name):
    """Return ``{pair: market dict}`` from a fixture file, failing on any bad pair."""
    markets, errors = read_market_file((FIXTURES / "markets" / name).read_bytes())
    if errors:
        raise ValueError(f"Fixture {name} has invalid markets: {errors}")
    return markets


def _render_body():
    source = ast.parse(Path(ui.__file__).read_text(encoding="utf-8-sig"))
    return next(node for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name == "render_app").body


def _nested_definitions():
    body = _render_body()
    found = {node.name: node for node in body
             if isinstance(node, ast.FunctionDef) and node.name in _NESTED}
    # ``calculate_row`` lives inside the ``if calculate:`` block.
    for node in body:
        if isinstance(node, ast.If) and getattr(node.test, "id", None) == "calculate":
            for inner in node.body:
                if isinstance(inner, ast.FunctionDef) and inner.name == "calculate_row":
                    found["calculate_row"] = inner
    missing = set(_NESTED + ("calculate_row",)).difference(found)
    if missing:
        raise RuntimeError("render_app no longer defines: " + ", ".join(sorted(missing)))
    return list(found.values())


class AppHarness:
    """One pair's market as the app would hold it after loading the file."""

    def __init__(self, market_payload, pricing_model="vv", premium_quote="term_cash",
                 beta_atm=0.7, beta_rr=1.2, beta_bf=0.0, delta_reporting="forward"):
        loaded = validate_market(market_payload)
        self.pair = loaded["currency_pair"]
        self.valuation_date = loaded["valuation_date"]
        self.spot = loaded["spot"]
        self.mods = Modifications(beta_atm=beta_atm, beta_rr=beta_rr, beta_bf=beta_bf)
        self.pricing_model = pricing_model
        self.diagnostics = DiagnosticsCollector()
        bind_cache(self.diagnostics.token, SurfaceCache())
        session = SimpleNamespace(
            vol_rows=loaded["vol_rows"], forward_rows=loaded["forward_rows"],
            event_rows=loaded["event_rows"], currency_pair=self.pair,
            delta_reporting=delta_reporting,
        )
        self.ns = dict(
            vars(ui),
            st=SimpleNamespace(session_state=session),
            valuation_date=self.valuation_date, spot=self.spot,
            currency_pair=self.pair, forward_divisor=loaded["forward_divisor"],
            pricing_model=pricing_model, broker_strangle=loaded["broker_strangle"],
            atm_type="dn", delta_base=True, smooth_pillar_interpolation=True,
            diagnostics=self.diagnostics, market_service=MarketService(self.diagnostics),
            premium_quote=premium_quote, mods=self.mods,
        )
        module = ast.Module(body=_nested_definitions(), type_ignores=[])
        exec(compile(module, ui.__file__, "exec"), self.ns)

    def __getattr__(self, name):
        try:
            return self.ns[name]
        except KeyError:
            raise AttributeError(name) from None

    def price_rows(self, rows):
        """Price ``rows`` exactly as the Pricer's "Price portfolio" button does."""
        frame = pd.DataFrame(rows)
        calculated = [self.ns["calculate_row"]((line, row)) for line, (_, row)
                      in enumerate(frame.iterrows(), start=1)]
        results = [item[1] for item in calculated if item[1] is not None]
        errors = [item[2] for item in calculated if item[2] is not None]
        return results, errors

    def spot_ladder(self, rows, steps=1, increment_pct=1.0, mode="optimized",
                    forward_positions=()):
        frame = pd.DataFrame(rows)
        spots = [self.spot * (1.0 + step * increment_pct / 100.0)
                 for step in range(-steps, steps + 1)]
        return ui.calculate_spot_ladder(
            list(frame.iterrows()), spots, self.valuation_date,
            self.ns["market_for_expiry"], self.ns["resolve_market_expiry"],
            self.pricing_model, "dn", True, self.mods, mode=mode,
            forward_positions=forward_positions, scenario_workers=1,
        )
