"""The date edge cases (tests/edge_cases/) against the market and the solver.

Confirmed rows hold the solver to the market's answer; every row holds it to
the answer stored when the case was proposed, so a rule change that moves a
case fails until the case is re-checked. See tests/edge_cases/README.md.
"""
import pytest

from apeiron.gaia.market_dates import edge_cases, pillars
from helper_functions.fx_conventions import settlement

OPTIONS = edge_cases.read(edge_cases.OPTION_FILE)
TYPED = edge_cases.read(edge_cases.TYPED_FILE)
ROLLS = edge_cases.read(edge_cases.ROLLOVER_FILE)
STATUSES = {edge_cases.CONFIRMED, edge_cases.TO_CONFIRM, edge_cases.DISPUTED}


def _ids(rows):
    return [row["case_id"] for row in rows]


def test_the_case_files_are_well_formed():
    assert OPTIONS and TYPED and ROLLS
    for rows in (OPTIONS, TYPED, ROLLS):
        assert len(set(_ids(rows))) == len(rows)
        for row in rows:
            assert row["status"] in STATUSES, row["case_id"]
            if row["status"] == edge_cases.CONFIRMED:
                answer = row.get("market_delivery") or row.get("market_fx_date")
                assert answer and row["source"] and row["checked_on"], row["case_id"]


@pytest.mark.parametrize("row", OPTIONS, ids=_ids(OPTIONS))
def test_tenor_case(row):
    dates = pillars.tenor_dates(settlement(row["pair"]), row["trade_date"], row["tenor"])
    got = (str(dates.expiry), str(dates.delivery))
    if row["status"] == edge_cases.DISPUTED:
        pytest.skip(f"disputed: {row['note']}")
    if row["status"] == edge_cases.CONFIRMED:
        if row["market_expiry"]:
            assert got[0] == row["market_expiry"], "the market's expiry"
        assert got[1] == row["market_delivery"], "the market's delivery"
    assert got == (row["expiry"], row["delivery"]), "moved since proposed: re-check it"


@pytest.mark.parametrize("row", TYPED, ids=_ids(TYPED))
def test_typed_expiry(row):
    rules = settlement(row["pair"])
    got = (str(pillars.typed_expiry(row["typed_expiry"])),
           str(pillars.delivery_for_expiry(rules, row["typed_expiry"])))
    if row["status"] == edge_cases.DISPUTED:
        pytest.skip(f"disputed: {row['note']}")
    if row["status"] == edge_cases.CONFIRMED:
        assert got[1] == row["market_delivery"], "the market's delivery"
    assert got == (row["expiry"], row["delivery"]), "moved since proposed: re-check it"


def test_merge_keeps_market_answers_and_numbers_new_cases():
    existing = [{"case_id": "O0007", "pair": "EURUSD", "trade_date": "2025-01-10", "tenor": "1W",
                 "market_delivery": "2025-01-21", "status": "confirmed"}]
    proposed = [{"pair": "EURUSD", "trade_date": "2025-01-10", "tenor": "1W", "status": "to_confirm"},
                {"pair": "USDJPY", "trade_date": "2025-01-10", "tenor": "1W", "status": "to_confirm"}]
    merged = edge_cases.merge(existing, proposed, ("pair", "trade_date", "tenor"), "O")
    assert merged[0] is existing[0] and merged[0]["status"] == "confirmed"
    assert [row["case_id"] for row in merged] == ["O0007", "O0008"]


def test_categories_name_why_a_case_is_an_edge_case():
    _, found = edge_cases.categories("EURUSD", "2025-12-18", "1W")      # 25 Dec -> 26 Dec
    assert found[0] == "closed_day_roll"
    _, found = edge_cases.categories("EURUSD", "2025-11-20", "1W")      # Thanksgiving expiry
    assert "expiry_on_holiday" in found
    _, found = edge_cases.categories("EURUSD", "2025-01-10", "1W")      # MLK between
    assert "usd_intermediate" in found


@pytest.mark.parametrize("row", ROLLS, ids=_ids(ROLLS))
def test_rollover_case(row):
    from apeiron.gaia.market_dates.rollover import fx_date
    got = str(fx_date(row["moment"], row["timezone"]))
    if row["status"] == edge_cases.DISPUTED:
        pytest.skip(f"disputed: {row['note']}")
    if row["status"] == edge_cases.CONFIRMED:
        assert got == row["market_fx_date"], "the market's FX date"
    assert got == row["fx_date"], "moved since proposed: re-check it"
