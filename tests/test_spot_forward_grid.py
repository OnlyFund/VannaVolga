"""The Pricer's Spot / forward grid: value-date tokens and tenors, the market
rate, amounts in either currency, refusals, and the failsafe that every line
prices exactly as the trade it books, once read back from the store."""
import copy
from datetime import date

import pytest

from app import spot_forwards as sf
from app.market_set import MarketSet
from app.trades import service
from app.trades.store import SqliteTradeStore
from golden.app_harness import load_markets

VALUATION = date(2026, 10, 15)       # Thursday: TOM Fri 16, spot Mon 19, SN Tue 20


def market_set(short_points=None):
    payloads = copy.deepcopy(load_markets("MARKETS_20261015.json"))
    if short_points:
        payloads["EURUSD"]["short_points"] = short_points
    return MarketSet.from_payloads(payloads, VALUATION)


@pytest.fixture(scope="module")
def markets():
    return market_set()


def line(**changes):
    row = dict(currency_pair="EURUSD", buy_sell="BUY", amount="10m", amount_ccy="BASE",
               value_date="3M", rate="", book="", counterparty="", trade_id="")
    row.update(changes)
    return row


@pytest.mark.parametrize("pair, token, expected", [
    ("EURUSD", "SPOT", date(2026, 10, 19)), ("EURUSD", "", date(2026, 10, 19)),
    ("EURUSD", "TOM", date(2026, 10, 16)), ("EURUSD", "SN", date(2026, 10, 20)),
    ("EURUSD", "1W", date(2026, 10, 26)),
    ("EURUSD", "3M", date(2027, 1, 19)),          # spot plus 3M; 18-Jan is a USD holiday
    ("EURUSD", "1Y", date(2027, 10, 19)), ("EURUSD", "18M", date(2028, 4, 19)),
    ("EURUSD", "2027-03-05", date(2027, 3, 5)), ("EURUSD", date(2027, 3, 5), date(2027, 3, 5)),
    ("USDCAD", "TOM", date(2026, 10, 16)),         # T+1: TOM is spot
    ("USDCAD", "SPOT", date(2026, 10, 16)),
])
def test_value_dates(pair, token, expected):
    assert sf.value_date_for(pair, token, VALUATION) == expected


def test_tenors_from_a_month_end_spot_stay_at_month_end():
    # Wed 28-Oct trades for spot Fri 30-Oct, October's last business day.
    assert sf.value_date_for("EURUSD", "1M", date(2026, 10, 28)) == date(2026, 11, 30)


@pytest.mark.parametrize("token, message", [
    ("TOD", "settles today"), ("0M", "must be positive"),
    ("2027-03-06", "not a business day for EURUSD \\(next: 2027-03-08\\)"),
])
def test_value_dates_refused(token, message):
    with pytest.raises(ValueError, match=message):
        sf.value_date_for("EURUSD", token, VALUATION)


def test_a_blank_rate_is_the_rounded_market_outright(markets):
    (result,), errors = sf.price_lines(markets, [line()])
    assert errors == [] and result["rate_auto"]
    assert result["contract_rate"] == round(result["outright"], 6)
    assert abs(result["mtm_term"]) <= 10e6 * 0.5e-6
    assert sf.rate_decimals("USDJPY") == 4


def test_a_typed_rate_and_the_amounts(markets):
    (result,), _ = sf.price_lines(markets, [line(buy_sell="SELL", rate="1.13")])
    assert result["contract_rate"] == 1.13 and not result["rate_auto"]
    assert result["base_amount"] == -10e6 and result["term_amount"] == pytest.approx(11.3e6)
    assert result["settlement_term"] == pytest.approx(-10e6 * (result["outright"] - 1.13))
    assert result["mtm_usd"] == pytest.approx(result["mtm_term"])          # USD term


def test_a_term_amount_is_the_opposite_base_amount(markets):
    results, _ = sf.price_lines(markets, [
        line(buy_sell="BUY", amount="11.3m", amount_ccy="TERM", rate="1.13"),
        line(buy_sell="SELL", amount="10m", rate="1.13")])
    term, base = results
    assert term["base_amount"] == pytest.approx(-10e6)
    for key in ("mtm_term", "mtm_base", "theta", "spot_delta", "settlement_term"):
        assert term[key] == pytest.approx(base[key], rel=1e-12), key
    assert term["amount_ccy"] == "USD"


def test_usd_mtm_when_usd_is_the_base(markets):
    (result,), _ = sf.price_lines(markets, [line(currency_pair="USDMXN", value_date="6M",
                                                 buy_sell="SELL", amount="5m", rate="19")])
    assert result["mtm_usd"] == pytest.approx(result["mtm_base"])


@pytest.mark.parametrize("changes, message", [
    (dict(currency_pair="EUR"), "is not a pair"), (dict(buy_sell="HOLD"), "BUY or SELL"),
    (dict(amount=""), "enter an amount"), (dict(amount="0"), "must be positive"),
    (dict(amount_ccy="JPY"), "amount currency must be EUR \\(base\\) or USD \\(term\\)"),
    (dict(rate="abc"), "rate must be a number"), (dict(rate="-1"), "positive"),
    (dict(value_date="18M"), "outside the quoted forward range"),       # curve ends at 1Y
    (dict(value_date="TOM"), "ON/TN points are not quoted.*not booked off-market"),
])
def test_bad_lines_are_reported(markets, changes, message):
    results, errors = sf.price_lines(markets, [line(**changes)])
    assert results == [] and len(errors) == 1
    assert errors[0].startswith("Line 1: ")
    assert __import__("re").search(message, errors[0]), errors[0]


def test_blank_lines_are_skipped_and_numbering_follows_the_grid(markets):
    results, errors = sf.price_lines(markets, [dict(), line(), line(amount="x")])
    assert [r["row"] for r in results] == [2]
    assert errors[0].startswith("Line 3: ")


def test_with_tn_points_tom_books(tmp_path):
    quoted = market_set({"ON": 0.5, "TN": 0.4, "SN": 0.45})
    (result,), errors = sf.price_lines(quoted, [line(value_date="TOM")])
    assert errors == [] and result["value_date"] == date(2026, 10, 16)
    assert result["theta"] == pytest.approx(-result["mtm_term"])


GRID = [
    line(),                                                          # market rate, 3M
    line(buy_sell="SELL", rate="1.13", value_date="2027-03-05"),     # broken date
    line(amount="11.3m", amount_ccy="TERM", rate="1.13", value_date="1Y"),
    line(value_date="SPOT", amount="2.5m", rate="1.12"),
    line(currency_pair="USDJPY", buy_sell="SELL", amount="500m", amount_ccy="TERM",
         value_date="6M", rate="150"),
    line(currency_pair="USDMXN", amount="3m", value_date="9M", rate="20.1"),
    line(value_date="SN", rate="1.121", amount="1m"),
]


def test_every_line_prices_exactly_as_the_trade_it_books(markets, tmp_path):
    store = SqliteTradeStore(tmp_path / "trades.sqlite")
    grid, errors = sf.price_lines(markets, GRID, extras=("rho", "phi"))
    assert errors == [] and len(grid) == len(GRID)
    booked = sf.book_lines(store, markets, GRID, book="B1", counterparty="C1", user="t")
    assert booked.ok and list(booked.lines) == list(range(1, len(GRID) + 1))
    trades = store.trades()
    assert {t.book for t in trades} == {"B1"} and len({t.package_id for t in trades}) == 1
    _, book, errors = service.price_trades(markets, trades, None, extras=("rho", "phi"))
    assert errors == []
    by_id = {row["trade_id"]: row for row in book}
    for number, result in enumerate(grid, start=1):
        row = by_id[booked.lines[number]]
        for key in ("value_date", "base_amount", "forward_rate", "outright", "mtm_term",
                    "mtm_base", "settlement_term", "spot_delta", "fwd_delta", "theta",
                    "rho", "phi"):
            assert row[key] == result[key], (number, key)


def test_booking_is_all_or_nothing_and_never_twice(markets, tmp_path):
    store = SqliteTradeStore(tmp_path / "trades.sqlite")
    refused = sf.book_lines(store, markets, [line(), line(value_date="TOD")])
    assert not refused.ok and refused.errors[0].startswith("Line 2: ") and store.trades() == []
    again = sf.book_lines(store, markets, [line(trade_id="TR-000001")])
    assert not again.ok and "already booked as TR-000001" in again.errors[0]
    assert not sf.book_lines(store, markets, []).ok


def test_booked_deals_export_and_import(markets, tmp_path):
    # The Blotter's export: a value date must serialise and come back.
    import json
    store = SqliteTradeStore(tmp_path / "trades.sqlite")
    booked = sf.book_lines(store, markets, [line(amount="11.3m", amount_ccy="TERM",
                                                 value_date="2027-03-05", rate="1.13")])
    assert booked.ok
    text = json.dumps(service.export_book(store))
    assert '"value_date": "2027-03-05"' in text and "2027-03-05" in service.export_csv(store)
    target = SqliteTradeStore(tmp_path / "copy.sqlite")
    service.import_book(target, text)
    (original,), (copied,) = store.trades(), target.trades()
    for name in ("value_date", "expiry", "amount_ccy", "notional", "contract_rate", "buy_sell"):
        assert getattr(copied, name) == getattr(original, name), name


def test_remove_drops_the_ticked_lines_and_keeps_one_line():
    import pandas as pd
    from app.ui.spot_forward_block import REMOVE, TICK, remove_ticked
    rows = pd.DataFrame([dict(line(trade_id="TR-1"), **{TICK: False, REMOVE: False}),
                         dict(line(amount="5m"), **{TICK: True, REMOVE: True}),
                         dict(line(amount="7m"), **{TICK: False, REMOVE: False})])
    assert remove_ticked(rows.assign(**{REMOVE: False})) is None          # nothing ticked
    kept = remove_ticked(rows)
    assert list(kept["amount"]) == ["10m", "7m"] and not kept[REMOVE].any()
    everything = remove_ticked(rows.assign(**{REMOVE: True}))
    assert len(everything) == 1 and (everything["amount"] == "").all()     # a fresh line stays
    assert everything.iloc[0][["buy_sell", "amount_ccy", "value_date"]].tolist() == [
        "BUY", "BASE", "SPOT"]
