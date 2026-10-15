import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from event_calendar import EventCalendar
from helper_functions.vol_data import VolData


ROOT = os.path.dirname(os.path.dirname(__file__))


def test_event_calendar_normalizes_timezone_and_includes_events():
    calendar = EventCalendar(os.path.join(ROOT, "event_calendar.csv"))
    assert calendar.get_event_weights(["2025-09-05"]).iloc[0] == pytest.approx(1.5)
    assert calendar.get_var_days("2025-09-04", "2025-09-05") == pytest.approx(2.5)
    assert calendar.get_var_days("2025-09-04", "2025-09-06") == pytest.approx(3.5)
    assert calendar.get_var_days("2025-09-05", "2025-09-06") == pytest.approx(1.0)


def _expected_interpolated_vol(valuation_date, target_date, pillars, column="ATM"):
    """Independent implementation of the documented interpolation convention.

    Total variance is accumulated in calendar-annualized units (quote**2 *
    calendar days), interpolated linearly against the event-weighted
    variance-day clock, then annualized over the target's calendar days.
    """
    calendar = EventCalendar(os.path.join(ROOT, "event_calendar.csv"))
    valuation_date = pd.Timestamp(valuation_date)
    target_date = pd.Timestamp(target_date)
    x = np.concatenate(([0.0], pillars["vdays"].to_numpy(dtype=float)))
    total_variance = (
        pillars[column].to_numpy(dtype=float) ** 2
        * pillars["days"].to_numpy(dtype=float)
    )
    y = np.concatenate(([0.0], total_variance))
    target_vdays = calendar.get_var_days(valuation_date, target_date)
    calendar_days = (target_date - valuation_date).days
    return float(np.sqrt(np.interp(target_vdays, x, y) / calendar_days))


def test_pre_event_maturity_is_discounted_relative_to_event_pillar():
    data = VolData(
        os.path.join(ROOT, "vol_data.csv"),
        valuation_date="2025-08-29",
        calendar_csv=os.path.join(ROOT, "event_calendar.csv"),
    )
    pillars = data.vols_5pt_df.dropna(subset=["expiry"])
    pre_event = data.interpolate_vols("2025-09-04")
    event_pillar = data.interpolate_vols("2025-09-05")
    assert pre_event["ATM"] < event_pillar["ATM"]
    assert pre_event["ATM"] == pytest.approx(
        _expected_interpolated_vol("2025-08-29", "2025-09-04", pillars), rel=1e-12
    )


def test_interpolated_vols_are_not_rescaled_by_the_variance_day_clock():
    """The interpolation convention must reproduce quotes at every pillar.

    Accumulating total variance on the variance-day clock while annualizing it
    over calendar days rescaled every interpolated quote by
    sqrt(vdays / calendar days), which pushed a 5.92 ATM quote down to ~5.09.
    The pillar dates below exercise the interpolation formula itself, without
    relying on the exact-match shortcut inside ``interpolate_vols``.
    """
    data = VolData(
        os.path.join(ROOT, "vol_data.csv"),
        valuation_date="2025-08-29",
        calendar_csv=os.path.join(ROOT, "event_calendar.csv"),
    )
    pillars = data.vols_5pt_df.dropna(subset=["expiry"])
    for column in ("PUT_10", "PUT_25", "ATM", "CALL_25", "CALL_10"):
        for _, pillar in pillars.iterrows():
            assert _expected_interpolated_vol(
                "2025-08-29", pillar["expiry"], pillars, column
            ) == pytest.approx(pillar[column], rel=1e-12)


def test_rolled_valuation_excludes_already_observed_event():
    data = VolData(
        os.path.join(ROOT, "vol_data.csv"),
        valuation_date="2025-09-05",
        calendar_csv=os.path.join(ROOT, "event_calendar.csv"),
    )
    pillars = data.vols_5pt_df.dropna(subset=["expiry"])
    overnight = data.interpolate_vols("2025-09-06")
    # The 2025-09-05 event is observed on the valuation date, so it no longer
    # loads the overnight bucket, which is left with a single business day.
    assert overnight["vday_to_expiry"] == pytest.approx(1.0)
    assert overnight["ATM"] == pytest.approx(
        _expected_interpolated_vol("2025-09-05", "2025-09-06", pillars), rel=1e-12
    )
    # One full business day is denser than the average day of the next pillar,
    # whose window is diluted by a weekend, so overnight vol prints above it.
    assert overnight["ATM"] > pillars.iloc[1]["ATM"]


def test_forward_curve_is_valid_and_first_bucket_is_defined():
    data = VolData(
        os.path.join(ROOT, "vol_data.csv"),
        valuation_date="2025-08-29",
        calendar_csv=os.path.join(ROOT, "event_calendar.csv"),
    )
    curve = data.vols_5pt_df.dropna(subset=["expiry"])
    forward_cols = [c for c in curve.columns if c.startswith("FWDVOL_")]
    assert curve["vdays"].is_monotonic_increasing
    assert (curve[forward_cols] >= 0).all().all()
    assert (curve[forward_cols].iloc[0] > 0).all()


def test_interpolation_reproduces_quotes_and_rejects_out_of_range_dates():
    data = VolData(
        os.path.join(ROOT, "vol_data.csv"),
        valuation_date="2025-08-29",
        calendar_csv=os.path.join(ROOT, "event_calendar.csv"),
    )
    valid = data.vols_5pt_df.dropna(subset=["expiry"])
    for _, row in valid.iterrows():
        interpolated = data.interpolate_vols(row["expiry"])
        assert interpolated["ATM"] == pytest.approx(row["ATM"])

    with pytest.raises(ValueError, match="after valuation"):
        data.interpolate_vols("2025-08-29")
    with pytest.raises(ValueError, match="outside"):
        data.interpolate_vols("2026-09-01")
