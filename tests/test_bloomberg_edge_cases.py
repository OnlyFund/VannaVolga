"""Judging the tenor edge cases on Bloomberg's answers, without Bloomberg."""
from datetime import date

from market_data.bloomberg_edge_cases import PAGE_SOURCE, SURFACE_SOURCE, validate


def _row(case_id, expiry="2024-01-18", delivery="2024-01-22", market_expiry="", market_delivery="",
         status="to_confirm", source=""):
    return dict(case_id=case_id, pair="EURUSD", trade_date="2024-01-11", tenor="1W", expiry=expiry,
                delivery=delivery, market_expiry=market_expiry, market_delivery=market_delivery,
                status=status, source=source, checked_on="", note="")


def _run(rows, expiries=None, spots=None):
    asked = {}

    def fetch_expiries(wanted):
        asked["expiries"] = wanted
        return expiries or {}

    def fetch_spots(wanted):
        asked["spots"] = wanted
        return spots or {}

    return validate(rows, fetch_expiries, fetch_spots, today=date(2026, 10, 22)), asked


def test_fetched_answers_confirm_or_dispute():
    rows = [_row("O1"), dict(_row("O2"), pair="GBPUSD")]
    report, asked = _run(
        rows,
        expiries={("EURUSD", "2024-01-11", "1W"): "2024-01-18", ("GBPUSD", "2024-01-11", "1W"): "2024-01-17"},
        spots={("EURUSD", "2024-01-18"): "2024-01-22", ("GBPUSD", "2024-01-17"): "2024-01-19"})
    assert asked["spots"] == {("EURUSD", "2024-01-18"), ("GBPUSD", "2024-01-17")}
    assert report["confirmed"] == ["O1"] and report["disputed"] == ["O2"]
    assert rows[0]["status"] == "confirmed" and rows[0]["source"] == SURFACE_SOURCE
    assert rows[0]["checked_on"] == "2026-10-22"
    assert rows[1]["status"] == "disputed" and "pricing page" in rows[1]["note"]


def test_hand_answers_are_judged_not_fetched():
    rows = [_row("O1", market_expiry="2024-01-18", market_delivery="2024-01-22")]
    report, asked = _run(rows)
    assert "expiries" not in asked
    assert report["confirmed"] == ["O1"] and rows[0]["source"] == PAGE_SOURCE


def test_settled_rows_are_left_alone_and_missing_answers_stay_open():
    rows = [_row("O1", status="confirmed", source="x"), _row("O2"),
            _row("O3", expiry="2024-02-01", status="disputed")]
    report, _ = _run(rows, expiries={("EURUSD", "2024-01-11", "1W"): "2024-01-18"})
    assert report["unanswered"] == ["O2"] and rows[1]["status"] == "to_confirm"
    assert rows[0]["status"] == "confirmed" and rows[2]["status"] == "disputed"
