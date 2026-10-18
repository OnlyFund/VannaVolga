"""Negative values in red, for the report tables.

``red_negatives`` wraps a table in a pandas Styler that colours negative
cells red; the minus sign stays. Cells may be numbers or text already
formatted for display ("-1,234"). A "-0" left by rounding is not negative.

Colour is cosmetic: if styling fails the table is returned unstyled, so a
report never breaks over it. In ``st.data_editor`` Streamlit applies the
styles to read-only columns only, which is where results sit.

A new Styler gets a random id, which Streamlit sends with the table: an
editor whose input changes every run is rebuilt every run, dropping a line
being added. The Styler made here takes an id fixed by the table's columns.
"""
import hashlib
import re

import pandas as pd
from pandas.io.formats.style import Styler

NEGATIVE_STYLE = "color: #d32f2f"
# A minus (ASCII or Unicode), then a number with at least one non-zero digit.
_NEGATIVE_TEXT = re.compile(r"^\s*[-−]\s*[\d,]*\.?\d*[1-9]")


def is_negative(value):
    """True for a negative number, or text showing one."""
    if isinstance(value, str):
        return bool(_NEGATIVE_TEXT.match(value))
    if value is None or isinstance(value, bool):
        return False
    try:
        return float(value) < 0     # NaN compares False
    except (TypeError, ValueError):
        return False


def negative_style(value):
    return NEGATIVE_STYLE if is_negative(value) else ""


def stable_uuid(columns):
    """A Styler id that stays the same from run to run for these columns."""
    text = "".join(str(column) for column in columns)
    return "neg" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def red_negatives(data, columns=None):
    """``data`` (a DataFrame or a Styler) with negative cells in red.

    ``columns`` limits the colouring to those columns (missing ones are
    ignored); by default every column is coloured.
    """
    try:
        if isinstance(data, Styler):
            styler = data
        else:
            styler = data.style
            styler.set_uuid(stable_uuid(data.columns))
        frame = styler.data
        subset = list(frame.columns) if columns is None else [c for c in columns if c in frame]
        if not subset or frame.empty:
            return data
        # Styler.map replaced applymap in pandas 2.1.
        apply_cells = getattr(styler, "map", None) or styler.applymap
        return apply_cells(negative_style, subset=pd.IndexSlice[:, subset])
    except Exception:              # cosmetic: never break a table over its colour
        return data
