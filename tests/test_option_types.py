"""Option types: vanilla is the default; vanillas and digitals are priced,
other types are stored and shown but report that they are not priced yet,
never priced as vanillas. Trades booked before types existed read as
vanillas, and the store migrates to the type columns without touching them."""
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pytest

from app import option_types, pricer
from app.market_set import MarketSet
from app.trades import service
from app.trades.model import FX_FORWARD, FX_OPTION, Booking, Trade, TradeValidationError
from app.trades.store import _SCHEMA, SCHEMA_VERSION, SqliteTradeStore
from app.ui.portfolio_block import _apply_editor_state, with_premium_modes
from golden.app_harness import load_markets
from market_data import Modifications

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def option(**changes):
    terms = dict(product_type=FX_OPTION, currency_pair="EURUSD", trade_date=VALUATION,
                 expiry=date(2027, 1, 15), buy_sell="BUY", notional=1e6, strike=1.15,
                 call_put="CALL", premium=1000.0, premium_ccy="USD")
    terms.update(changes)
    return Trade(**terms)


@pytest.mark.parametrize("value, expected", [
    (None, "vanilla"), (np.nan, "vanilla"), ("", "vanilla"), ("  ", "vanilla"),
    ("Vanilla", "vanilla"), ("digital", "digital"), ("DIGITAL", "digital"),
    ("Knock-out", "knock_out"), ("knock_out", "knock_out"), ("Double no-touch", "double_no_touch"),
    ("one touch", "one_touch"),
])
def test_option_type_names(value, expected):
    assert option_types.option_type_of(value) == expected


def test_an_unknown_type_is_rejected():
    with pytest.raises(ValueError, match="Option type must be one of"):
        option_types.option_type_of("asian")


def test_only_vanillas_and_digitals_are_priced_for_now():
    assert option_types.check_priced(None) == "vanilla"
    assert option_types.check_priced("Digital") == "digital"
    for key in option_types.OPTION_TYPES[2:]:
        with pytest.raises(ValueError, match="not priced yet"):
            option_types.check_priced(key)


def test_trades_default_to_vanilla_and_digitals_pay_in_base():
    assert option().normalized().option_type == "vanilla"
    assert option().is_vanilla and option(option_type="vanilla").is_vanilla
    digital = option(option_type="Digital").normalized()
    assert digital.option_type == "digital" and digital.payout_ccy == "EUR"
    assert not digital.is_vanilla
    assert option(option_type="digital", payout_ccy="usd").normalized().payout_ccy == "USD"
    with pytest.raises(TradeValidationError, match="payout currency must be EUR or USD"):
        option(option_type="digital", payout_ccy="JPY").normalized()
    with pytest.raises(TradeValidationError, match="only a binary-payout option"):
        option(payout_ccy="EUR").normalized()
    with pytest.raises(TradeValidationError, match="option type must be one of"):
        option(option_type="asian").normalized()


def test_forwards_have_no_option_type():
    forward = dict(product_type=FX_FORWARD, currency_pair="EURUSD", trade_date=VALUATION,
                   expiry=date(2027, 1, 15), buy_sell="BUY", notional=1e6, contract_rate=1.16)
    normalized = Trade(**forward).normalized()
    assert normalized.option_type is None and not normalized.is_vanilla
    with pytest.raises(TradeValidationError, match="no option type"):
        Trade(**forward, option_type="digital").normalized()


def test_pricer_rows_carry_the_type():
    assert option().to_pricer_row()["option_type"] == "vanilla"
    row = option(option_type="digital").normalized().to_pricer_row()
    assert row["option_type"] == "digital" and row["payout_ccy"] == "EUR"


def line(**changes):
    row = dict(currency_pair="EURUSD", expiry="3M", call_put="CALL", strike="25d",
               buy_sell="BUY", notional="10m", premium=np.nan, premium_mode="auto",
               premium_ccy="TERM")
    row.update(changes)
    return row


def test_lines_of_types_not_priced_yet_report_it_and_vanillas_are_unchanged(markets):
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame([
        line(), line(option_type="Vanilla"), line(option_type="One-touch"),
        line(option_type="Knock-out"), line(option_type="asian")]))
    results, errors = pricer.price_lines(markets, rows, MODS)
    assert [r["row"] for r in results] == [1, 2]
    strip = [{k: v for k, v in r.items() if k != "row"} for r in results]
    assert strip[0] == strip[1]
    assert errors == ["Line 3: One-touch options are not priced yet",
                      "Line 4: Knock-out options are not priced yet",
                      "Line 5: Option type must be one of Vanilla, Digital, Knock-out, Knock-in, "
                      "One-touch, No-touch, Double no-touch, not 'asian'"]


def test_booking_a_type_not_priced_yet_is_refused_and_vanillas_record_their_type(
        markets, tmp_path):
    store = SqliteTradeStore(tmp_path / "trades.sqlite")
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame(
        [line(), line(option_type="Knock-in")]))
    refused = service.book_lines(store, markets, rows, MODS)
    assert not refused.ok and store.trades() == []
    booked = service.book_lines(store, markets, rows.iloc[[0]], MODS)
    assert booked.ok and [t.option_type for t in store.trades()] == ["vanilla"]


def test_stored_trades_of_types_not_priced_yet_are_reported(markets, tmp_path):
    store = SqliteTradeStore(tmp_path / "trades.sqlite")
    store.book(Booking(trades=(option(), option(option_type="no_touch"))))
    trades = store.trades()
    assert [t.option_type for t in trades] == ["vanilla", "no_touch"]
    assert trades[1].payout_ccy == "EUR"
    results, _, errors = service.price_trades(markets, trades, MODS)
    assert [r["trade_id"] for r in results] == [trades[0].trade_id]
    assert errors == [f"{trades[1].trade_id}: No-touch options are not priced yet"]


def _version_1_store(path):
    """A store as schema 1 wrote it, holding one option."""
    schema = (_SCHEMA.replace("    option_type TEXT,\n", "").replace("    payout_ccy TEXT,\n", ""))
    assert "option_type" not in schema
    db = sqlite3.connect(str(path))
    db.executescript(schema + "INSERT INTO schema_info (version) VALUES (1);")
    db.execute(
        "INSERT INTO trade_versions (trade_id, version, action, recorded_at, status, "
        "product_type, currency_pair, trade_date, expiry, buy_sell, notional, strike, "
        "call_put, premium, premium_ccy) VALUES ('TR-000001', 1, 'new', "
        "'2026-10-15T09:00:00Z', 'live', 'fx_option', 'EURUSD', '2026-10-15', "
        "'2027-01-15', 'BUY', 1000000, 1.15, 'CALL', 1000, 'USD')")
    db.execute("INSERT INTO sequences (name, next) VALUES ('trade', 2)")
    db.commit()
    db.close()


def test_a_version_1_store_migrates_and_its_trades_are_vanillas(tmp_path):
    path = tmp_path / "trades.sqlite"
    _version_1_store(path)
    store = SqliteTradeStore(path, backup_dir=tmp_path / "backups")
    (trade,) = store.trades()
    assert trade.option_type is None and trade.is_vanilla
    assert trade.strike == 1.15 and trade.premium == 1000.0
    assert store._schema_version() == SCHEMA_VERSION == 2
    assert list((tmp_path / "backups").glob("*before_schema_2*"))
    # New types book alongside it.
    store.book(Booking(trades=(option(option_type="digital", payout_ccy="USD"),)))
    assert [t.option_type for t in store.trades()] == [None, "digital"]


def test_a_version_1_store_opened_read_only_still_reads(tmp_path):
    path = tmp_path / "trades.sqlite"
    _version_1_store(path)
    store = SqliteTradeStore(path, read_only=True)
    (trade,) = store.trades()
    assert trade.is_vanilla and trade.option_type is None


@pytest.mark.parametrize("column, value", [("option_type", "Digital"), ("payout_ccy", "TERM")])
def test_changing_the_type_or_payout_is_a_change_of_terms(column, value):
    rows = with_premium_modes(pd.DataFrame([line(premium=5.0, premium_mode="typed",
                                                 option_type="Vanilla", payout_ccy="BASE",
                                                 trade_id="TR-000009")]))
    changed = _apply_editor_state(rows, {"edited_rows": {0: {column: value}}})
    assert changed.at[0, "premium_mode"] == "auto" and pd.isna(changed.at[0, "premium"])
    assert changed.at[0, "trade_id"] == ""
