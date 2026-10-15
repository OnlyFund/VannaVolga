from datetime import date

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import importlib

from app.reports.spot_ladder import (
    _add_option_pnl, _empty_level, calculate_spot_ladder,
)
import app.reports.spot_ladder as spot_ladder
from app.ui.portfolio_block import (
    GREEK_NAMES, LINE_RESULT_COLUMNS, REMOVE_COLUMN, _apply_editor_state,
    _marked_row_indices, _with_line_results,
)
from app.ui.app import (
    aggregate_pricer_results_by_pair, format_pricer_aggregation,
    format_strike_for_pair,
)
from app.ui.app import interpolate_delivery_curve
from helper_functions.cashflows import signed_premium_cashflow
from market_data import MarketData
from pricing.fx_forward import FxForwardPosition


def _market_for_expiry(expiry, as_of=None, allow_short_end=False, spot_value=None):
    return MarketData(
        s=1.20 if spot_value is None else float(spot_value),
        fwd_pts=1_500.0,
        v_atm=10.0,
        rr_25=0.0,
        rr_10=0.0,
        bf_25=0.0,
        bf_10=0.0,
    )


def test_forward_position_uses_undiscounted_term_and_base_mtm():
    position = FxForwardPosition(
        currency_pair="EURUSD",
        expiry=date(2025, 9, 12),
        base_amount=-2_000_000.0,
        contract_rate=1.30,
    )

    value = position.value(_market_for_expiry(None), date(2025, 8, 29))

    assert value["mtm_term"] == pytest.approx(-100_000.0)
    assert value["mtm_base"] == pytest.approx(-100_000.0 / 1.20)
    assert value["delta_term"] == pytest.approx(-2_000_000.0)


def test_delta_hedge_notional_offsets_smile_delta():
    fixed_delta = 600_000.0
    smile_delta = 750_000.0

    hedge_amount = -smile_delta

    assert hedge_amount == pytest.approx(-750_000.0)
    assert smile_delta + hedge_amount == pytest.approx(0.0)
    assert fixed_delta + hedge_amount == pytest.approx(-150_000.0)


@pytest.mark.parametrize("pair", ["EURUSD", "USDZAR"])
def test_forward_theta_keeps_contract_rate_and_uses_pair_theta_currency(pair):
    from pricing.greeks import calculate_forward_theta

    today = _market_for_expiry(None)
    tomorrow = today.copy_with(fwd_pts=1_490.0)
    position = FxForwardPosition(pair, date(2027, 1, 15), -2_000_000., 1.30)
    theta = calculate_forward_theta(
        position, today, tomorrow, date(2026, 10, 9), date(2026, 10, 12)
    )
    term_theta = -2_000_000. * ((1_490. - 1_500.) / 10_000.)
    expected = term_theta if pair == "EURUSD" else term_theta / today.s
    assert theta == pytest.approx(expected)
    assert theta > 0.0


def test_editor_state_persists_all_added_portfolio_rows():
    rows = pd.DataFrame([
        {"strike": "1.10", "notional": "1m"},
        {"strike": "1.20", "notional": "2m"},
    ])
    updated = _apply_editor_state(
        rows,
        {
            "edited_rows": {1: {"notional": "3m"}},
            "added_rows": [
                {"strike": "1.30", "notional": "4m"},
                {"strike": "1.40", "notional": "5m"},
            ],
            "deleted_rows": [],
        },
    )

    assert list(updated["strike"]) == ["1.10", "1.20", "1.30", "1.40"]
    assert list(updated["notional"]) == ["1m", "3m", "4m", "5m"]


def test_editor_state_deletes_original_and_dynamic_portfolio_rows():
    rows = pd.DataFrame([
        {"strike": "1.10", "notional": "1m"},
        {"strike": "1.20", "notional": "2m"},
    ])

    updated = _apply_editor_state(
        rows,
        {
            "edited_rows": {2: {"notional": "4.5m"}},
            "added_rows": [
                {"strike": "1.30", "notional": "4m"},
                {"strike": "1.40", "notional": "5m"},
            ],
            "deleted_rows": [0, 2],
        },
    )

    assert list(updated["strike"]) == ["1.20", "1.40"]
    assert list(updated["notional"]) == ["2m", "5m"]


def test_editor_state_explicit_remove_marks_delete_rows_without_persisting_column():
    rows = pd.DataFrame([
        {"strike": "1.10", "notional": "1m"},
        {"strike": "1.20", "notional": "2m"},
    ])
    editor_state = {
        "edited_rows": {1: {REMOVE_COLUMN: True}},
        "added_rows": [{REMOVE_COLUMN: False, "strike": "1.30", "notional": "3m"}],
        "deleted_rows": [],
    }
    editor_state["deleted_rows"] = _marked_row_indices(editor_state)

    updated = _apply_editor_state(rows, editor_state)

    assert list(updated["strike"]) == ["1.10", "1.30"]
    assert REMOVE_COLUMN not in updated.columns


def test_editor_state_normalizes_currency_pairs_to_uppercase():
    rows = pd.DataFrame([
        {"currency_pair": "eurusd", "strike": "1.10"},
    ])

    updated = _apply_editor_state(
        rows,
        {
            "edited_rows": {0: {"currency_pair": "gbpusd"}},
            "added_rows": [{"currency_pair": "usdjpy", "strike": "155.00"}],
            "deleted_rows": [],
        },
    )

    assert list(updated["currency_pair"]) == ["GBPUSD", "USDJPY"]


def test_line_results_attach_to_matching_pricer_rows():
    rows = pd.DataFrame([
        {"currency_pair": "EURUSD", "strike": "1.10"},
        {"currency_pair": "USDJPY", "strike": "155.00"},
    ])
    first_result = {
        "row": 1,
        "vol_pct": 8.25,
        "mtm_base": 125.0,
        "pnl_base": 25.0,
        **{
            column: 10.0
            for greek in GREEK_NAMES
            for column in (f"fixed_{greek}", greek)
        },
    }

    display = _with_line_results(rows, [first_result])

    assert set(LINE_RESULT_COLUMNS).issubset(display.columns)
    assert display.loc[0, "vol_pct"] == "8.2500"
    assert display.loc[0, "mtm_base"] == "125"
    assert pd.isna(display.loc[1, "vol_pct"])
    assert list(display["currency_pair"]) == ["EURUSD", "USDJPY"]


def test_pricer_results_aggregate_by_pair_including_hedges():
    def result(pair, row, amount):
        return {
            "row": row,
            "currency_pair": pair,
            "mtm_base": amount,
            "pnl_base": amount / 2.0,
            **{
                column: amount
                for greek in GREEK_NAMES
                for column in (f"fixed_{greek}", greek)
            },
        }

    aggregation = aggregate_pricer_results_by_pair(
        [result("EURUSD", 1, 10.0), result("EURUSD", 2, 20.0), result("USDJPY", 3, 5.0)],
        [{
            "currency_pair": "EURUSD",
            "mtm_base": 3.0,
            "pnl_base": 4.0,
            "delta_offset": -7.0,
            "theta": 2.0,
        }],
    ).set_index("currency_pair")

    assert aggregation.loc["EURUSD", "mtm_base"] == pytest.approx(33.0)
    assert aggregation.loc["EURUSD", "pnl_base"] == pytest.approx(19.0)
    assert aggregation.loc["EURUSD", "fixed_delta"] == pytest.approx(23.0)
    assert aggregation.loc["EURUSD", "delta"] == pytest.approx(23.0)
    assert aggregation.loc["EURUSD", "theta"] == pytest.approx(32.0)
    assert aggregation.loc["EURUSD", "fixed_theta"] == pytest.approx(30.0)
    assert aggregation.loc["USDJPY", "mtm_base"] == pytest.approx(5.0)


def test_pricer_display_formats_monetary_and_greek_values_with_commas():
    aggregation = pd.DataFrame([
        {"currency_pair": "EURUSD", "mtm_base": 1234567.89, "pnl_base": -12.5,
         "fixed_delta": 2500000.0, "delta": -1500.0},
    ])

    display = format_pricer_aggregation(aggregation)

    assert display.loc[0, "mtm_base"] == "1,234,568"
    assert display.loc[0, "pnl_base"] == "-12"
    assert display.loc[0, "fixed_delta"] == "2,500,000"
    assert display.loc[0, "delta"] == "-1,500"


def test_strike_display_uses_pair_specific_precision():
    assert format_strike_for_pair("EURUSD", 1.123456) == "1.1235"
    assert format_strike_for_pair("USDJPY", 156.789) == "156.79"
    assert format_strike_for_pair("JPYEUR", 0.006789) == "0.01"


def test_forward_curve_interpolation_rejects_out_of_range_delivery():
    curve = pd.DataFrame(
        {
            "delivery_date": ["2025-01-10", "2025-02-10"],
            "fwd_pts": [10.0, 20.0],
        }
    )
    assert interpolate_delivery_curve(
        curve, "delivery_date", "fwd_pts", "2025-01-20", "Forward points"
    ) == pytest.approx(10.0 + 10.0 * 10.0 / 31.0)
    with pytest.raises(ValueError, match="outside the quoted forward range"):
        interpolate_delivery_curve(
            curve, "delivery_date", "fwd_pts", "2025-03-10", "Forward points"
        )


def test_weighted_greeks_imports_as_a_package_module():
    assert importlib.import_module("Greeks.weighted_greeks").WeightedGreeks


def test_spot_ladder_includes_forward_pnl_and_both_delta_totals():
    position = {
        "currency_pair": "EURUSD",
        "expiry": date(2025, 9, 12),
        "base_amount": -2_000_000.0,
        "contract_rate": 1.30,
    }
    levels, errors = calculate_spot_ladder(
        rows=[],
        ladder_spots=[1.20],
        valuation_date=date(2025, 8, 29),
        market_for_expiry=_market_for_expiry,
        resolve_expiry=lambda value: value,
        pricing_model="vv",
        atm_type="dn",
        delta_base=False,
        mods=None,
        mode="reference",
        forward_positions=[position],
    )

    assert errors == []
    level = levels[0]
    assert level["pnl_base"] == pytest.approx(-100_000.0 / 1.20)
    assert "mtm" not in level
    assert "option_pnl_base" not in level
    assert "hedge_pnl_base" not in level
    assert level["fixed_delta"] == pytest.approx(-2_000_000.0)
    assert level["delta"] == pytest.approx(-2_000_000.0)


@pytest.mark.parametrize("mode", ["reference", "optimized"])
def test_spot_ladder_includes_weekend_forward_roll_theta(mode):
    calls = []

    def market(expiry, as_of=None, allow_short_end=False, spot_value=None):
        calls.append(as_of)
        result = _market_for_expiry(expiry, spot_value=spot_value)
        return result if as_of is None else result.copy_with(fwd_pts=1_490.)

    levels, errors = calculate_spot_ladder(
        rows=[], ladder_spots=[1.20], valuation_date=date(2026, 10, 9),
        market_for_expiry=market, resolve_expiry=lambda value: value,
        pricing_model="vv", atm_type="dn", delta_base=False, mods=None,
        mode=mode, scenario_workers=1,
        forward_positions=[dict(
            currency_pair="EURUSD", expiry=date(2027, 1, 15),
            base_amount=-2_000_000., contract_rate=1.35,
        )],
    )
    assert errors == []
    assert date(2026, 10, 12) in calls
    assert levels[0]["theta"] == pytest.approx(2_000.)
    assert levels[0]["fixed_theta"] == 0.0


def test_spot_ladder_uses_forward_base_amount_not_discounted_delta(monkeypatch):
    class DiscountedDeltaForward:
        def __init__(self, **kwargs):
            self.base_amount = kwargs["base_amount"]
            self.expiry = kwargs["expiry"]

        def value(self, market, valuation_date):
            return {
                "mtm_term": 0.0,
                "mtm_base": 0.0,
                "delta_term": self.base_amount * 0.98,
            }

    monkeypatch.setattr(spot_ladder, "FxForwardPosition", DiscountedDeltaForward)
    level = _empty_level(1.20)
    spot_ladder._add_forward_pnl(
        level,
        {
            "currency_pair": "EURUSD",
            "expiry": date(2025, 9, 12),
            "base_amount": -2_000_000.0,
            "contract_rate": 1.30,
        },
        _market_for_expiry(None),
        date(2025, 8, 29),
    )

    assert level["fixed_delta"] == pytest.approx(-2_000_000.0)
    assert level["delta"] == pytest.approx(-2_000_000.0)


@pytest.mark.parametrize(
    ("buy_sell", "expected_cashflow"),
    [("BUY", -100.0), ("SELL", 100.0)],
)
def test_option_premium_cashflow_has_the_correct_trade_sign(
    buy_sell, expected_cashflow,
):
    assert signed_premium_cashflow(100.0, buy_sell) == expected_cashflow

    level = _empty_level(1.25)
    _add_option_pnl(
        level,
        {
            "currency_pair": "EURUSD",
            "premium": 100.0,
            "premium_ccy": "USD",
            "buy_sell": buy_sell,
        },
        mtm_term=25.0,
        spot=1.25,
    )

    assert level["pnl_base"] == pytest.approx(
        (expected_cashflow + 25.0) / 1.25
    )
