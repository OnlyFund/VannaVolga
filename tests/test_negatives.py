import pandas as pd
import pytest
from pandas.io.formats.style import Styler

from app.reports.spot_ladder import format_move, spot_moves
from app.ui.negatives import NEGATIVE_STYLE, is_negative, red_negatives


def test_spot_ladder_shows_each_spots_move_from_spot():
    moves = spot_moves([0.98, 0.99, 1.0, 1.015, 1.02], 1.0)
    assert moves == pytest.approx([-2.0, -1.0, 0.0, 1.5, 2.0])
    assert [format_move(m) for m in moves] == ["-2.00%", "-1.00%", "0.00%", "+1.50%", "+2.00%"]
    assert format_move(spot_moves([1.00005], 1.0)[0]) == "+0.0050%"


@pytest.mark.parametrize("value", [-1, -0.5, -1e-9, "-1,234", "-0.5", " -12", "−3",
                                   "-1,000", "-0.01"])
def test_negative_values(value):
    assert is_negative(value)


@pytest.mark.parametrize("value", [0, 1.5, float("nan"), None, True, "", "1,234", "-0",
                                   "-0.00", "-", "2026-10-17", "CALL", "nan"])
def test_values_that_are_not_negative(value):
    assert not is_negative(value)


def _styles(styler):
    """{(row, column): css} for the styled cells."""
    styler._compute()
    return {(row, styler.data.columns[col]): "; ".join(f"{k}: {v}" for k, v in css)
            for (row, col), css in styler.ctx.items() if css}


def test_red_negatives_colours_negative_cells_only():
    frame = pd.DataFrame({"pair": ["EURUSD", "USDJPY"], "vega": ["-1,200", "300"],
                          "delta": [5.0, -2.0]})
    styled = red_negatives(frame)
    assert isinstance(styled, Styler)
    assert _styles(styled) == {(0, "vega"): NEGATIVE_STYLE, (1, "delta"): NEGATIVE_STYLE}
    assert styled.data.equals(frame)          # values untouched


def test_red_negatives_limits_to_columns():
    frame = pd.DataFrame({"strike": ["-25d"], "pnl_base": ["-10"]})
    assert _styles(red_negatives(frame, columns=["pnl_base", "absent"])) == {
        (0, "pnl_base"): NEGATIVE_STYLE}


def test_red_negatives_keeps_existing_styles():
    frame = pd.DataFrame({"spot": [1.1, 1.2], "pnl": [-1.0, 2.0]})
    styled = frame.style.apply(lambda row: ["font-weight: bold"] * len(row), axis=1)
    styles = _styles(red_negatives(styled))
    assert styles[(0, "pnl")] == "font-weight: bold; " + NEGATIVE_STYLE
    assert styles[(1, "pnl")] == "font-weight: bold"


def test_red_negatives_falls_back_to_the_plain_table():
    frame = pd.DataFrame({"pnl": [-1.0]})
    assert red_negatives(frame, columns=[]) is frame
    assert red_negatives(frame.iloc[:0]) is not None
    assert red_negatives("not a table") == "not a table"
