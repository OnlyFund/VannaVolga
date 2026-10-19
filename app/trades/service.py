"""Booking and portfolio services: between Pricer lines, the store and pricing.

* ``book_lines`` turns Pricer lines into booked option trades, optionally
  with delta-hedge forwards linked to the options they hedge;
* ``select_trades`` reads a filtered book; ``price_trades`` prices it
  through a ``MarketSet``, options as Pricer lines and forwards as hedge
  positions, each result carrying its ``trade_id``;
* ``export_book`` / ``import_book`` move a book between files and stores.

Failsafes: a booking is all or nothing. Every selected line must resolve
to booked values (pair, dated expiry, numeric strike) and price before
anything is written; one bad line books nothing and is reported, so a
package is never half booked. Hedges are computed exactly as the Pricer
computes its hedge forwards and must sum to them.
"""
import csv
import io
import json
from dataclasses import dataclass, field
from datetime import date
from typing import List, Optional

import pandas as pd

from app import pricer
from app.option_types import PAYOUT_TYPES, option_type_of
from app.premium_side import premium_currency
from app.trades.model import (
    FX_FORWARD, FX_OPTION, LIVE, Booking, HedgeAllocation, Trade, TradeValidationError,
    as_date,
)

EXPORT_FORMAT, EXPORT_VERSION = "trade_book", 1
#: Optional per-line booking columns in the Pricer grid.
BOOKING_COLUMNS = ("book", "counterparty")


@dataclass
class BookingResult:
    """What a booking call did; ``booked`` is empty when ``errors`` is not."""

    booked: List[Trade] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    lines: dict = field(default_factory=dict)      # line number -> trade id

    @property
    def ok(self):
        return bool(self.booked) and not self.errors


def _text(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _amount(value):
    """A premium cell as a number; an empty cell is no premium."""
    if value is None or (not isinstance(value, str) and pd.isna(value)) or not str(value).strip():
        return 0.0
    return value


def _type_terms(pair, line):
    """``option_type`` and ``payout_ccy`` for a Pricer line's trade."""
    option_type = option_type_of(line.get("option_type"))
    payout = (premium_currency(pair, line.get("payout_ccy")) or pair[:3]
              if option_type in PAYOUT_TYPES else None)
    return {"option_type": option_type, "payout_ccy": payout}


def booking_from_lines(markets, rows, mods, trade_date=None, book="", counterparty="",
                       with_hedges=False, default_pair=None):
    """Build a ``Booking`` from Pricer lines; returns ``(booking, line_numbers, errors)``.

    Lines are resolved and priced first: the booked strike and expiry are
    the ones the Pricer shows, and hedges come from the priced deltas.
    """
    trade_date = as_date(trade_date or markets.valuation_date)
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    frame = frame.reset_index(drop=True)
    resolved, _, _ = pricer.resolve_line_inputs(markets, frame, default_pair)
    results, errors = pricer.price_lines(markets, resolved, mods)
    if errors:
        return None, [], errors
    trades, numbers = [], []
    for result in results:
        line = resolved.iloc[result["row"] - 1]
        try:
            trade = Trade(
                product_type=FX_OPTION, currency_pair=result["currency_pair"],
                trade_date=trade_date, expiry=result["expiry"], buy_sell=_text(line["buy_sell"]),
                notional=pricer.parse_notional(line["notional"]), strike=result["strike"],
                call_put=_text(line["call_put"]),
                premium=(result["auto_premium"] if result.get("premium_auto")
                         else _amount(line.get("premium"))),
                premium_ccy=premium_currency(result["currency_pair"],
                                             line.get("premium_ccy")) or None,
                book=_text(line.get("book")) or book,
                counterparty=_text(line.get("counterparty")) or counterparty,
                # The line's type (pricing has already refused any it cannot
                # price) and, for a binary payout, its currency.
                **_type_terms(result["currency_pair"], line),
            ).normalized()
        except (TradeValidationError, ValueError) as exc:
            errors.append(f"Line {result['row']}: {exc}")
            continue
        trades.append(trade)
        numbers.append(result["row"])
    if errors:
        return None, [], errors
    hedges = {}
    if with_hedges:
        groups, hedge_errors = pricer.hedge_amounts(markets, results)
        if hedge_errors:
            return None, [], hedge_errors
        index_of = {result["row"]: index for index, result in enumerate(results)}
        for (pair, expiry, rate), members in groups.items():
            total = sum(amount for _, amount in members)
            if abs(total) < 1e-6:
                continue      # delta-flat group: nothing to hedge
            first = trades[index_of[members[0][0]["row"]]]
            hedge = Trade(
                product_type=FX_FORWARD, currency_pair=pair, trade_date=trade_date,
                expiry=expiry, buy_sell="BUY" if total > 0 else "SELL", notional=abs(total),
                contract_rate=rate, book=first.book, counterparty=first.counterparty,
            ).normalized()
            hedges[len(trades)] = [HedgeAllocation(index_of[result["row"]], amount)
                                   for result, amount in members]
            trades.append(hedge)
            numbers.append(None)
    return Booking(tuple(trades), hedges), numbers, []


def book_lines(store, markets, rows, mods, trade_date=None, book="", counterparty="",
               with_hedges=False, user="", reason="", default_pair=None):
    """Book Pricer lines (and their hedges); all or nothing."""
    booking, numbers, errors = booking_from_lines(
        markets, rows, mods, trade_date, book, counterparty, with_hedges, default_pair)
    if errors:
        return BookingResult(errors=errors)
    try:
        booked = store.book(booking, user=user, reason=reason)
    except (TradeValidationError, ValueError) as exc:
        return BookingResult(errors=[str(exc)])
    lines = {number: trade.trade_id for number, trade in zip(numbers, booked) if number}
    return BookingResult(booked=booked, lines=lines)


def select_trades(store, status=LIVE, currency_pair=None, book=None, counterparty=None,
                  product_type=None, live_on=None, as_of=None):
    """The latest versions matching every filter given (see the store)."""
    return store.trades(status=status, currency_pair=currency_pair, book=book,
                        counterparty=counterparty, product_type=product_type,
                        as_of=as_of, live_on=live_on)


def price_trades(markets, trades, mods, extras=(), spot_path=pricer.SMILE_PATH):
    """Price booked trades; returns ``(option results, forward rows, errors)``.

    Each result carries its ``trade_id``. Expired or cancelled trades are
    skipped with a note, never priced. ``extras`` adds on-demand Greeks;
    ``spot_path`` is how a moved spot reprices (``pricer.price_line``).
    """
    valuation = as_date(markets.valuation_date)
    options, forwards, errors = [], [], []
    for trade in trades:
        if trade.status != LIVE:
            continue
        if trade.expiry <= valuation:
            errors.append(f"{trade.trade_id}: expired on {trade.expiry}")
        elif trade.is_option:
            options.append(trade.to_pricer_row())
        else:
            forwards.append(trade)
    results, line_errors = pricer.price_lines(markets, options, mods, extras=extras,
                                              spot_path=spot_path)
    by_row = {number: row["trade_id"] for number, row in enumerate(options, start=1)}
    errors += [_with_trade_id(message, by_row) for message in line_errors]
    forward_rows = []
    for trade in forwards:
        try:
            row = pricer.value_forward(markets, trade.currency_pair, trade.expiry,
                                       trade.signed_base_amount(), trade.contract_rate)
        except Exception as exc:
            errors.append(f"{trade.trade_id}: {exc}")
            continue
        forward_rows.append(dict(row, trade_id=trade.trade_id, book=trade.book,
                                 counterparty=trade.counterparty))
    return results, forward_rows, errors


def _with_trade_id(message, by_row):
    """``Line 3: ...`` -> ``TR-000012: ...`` for booked trades."""
    head, _, rest = message.partition(": ")
    if head.startswith("Line "):
        try:
            return f"{by_row[int(head[5:])]}: {rest}"
        except (KeyError, ValueError):
            pass
    return message


# -- export / import --------------------------------------------------------------

def _plain(trade):
    terms = trade.terms()
    terms["trade_date"] = trade.trade_date.isoformat()
    terms["expiry"] = trade.expiry.isoformat()
    return terms


def export_book(store, as_of=None, include_cancelled=True):
    """The current book as a JSON-ready dict, with hedge allocations."""
    trades = store.trades(status=None if include_cancelled else LIVE, as_of=as_of)
    return {
        "format": EXPORT_FORMAT, "version": EXPORT_VERSION,
        "exported_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "trades": [_plain(trade) for trade in trades],
        "hedges": {trade.trade_id: [
            {"option": allocation.option, "base_amount": allocation.base_amount}
            for allocation in store.allocations(trade.trade_id, as_of)]
            for trade in trades if trade.is_forward},
    }


def export_csv(store, as_of=None):
    """The current book as CSV text (one row per trade, no allocations)."""
    trades = store.trades(status=None, as_of=as_of)
    buffer = io.StringIO()
    columns = list(Trade.__dataclass_fields__)
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for trade in trades:
        writer.writerow(_plain(trade))
    return buffer.getvalue()


def import_book(store, payload, user="", reason="import"):
    """Book an exported book's live trades into ``store`` as new trades.

    Returns ``(mapping, notes)``: trades get new ids in the target store and
    ``mapping`` relates them to the source ids. Hedge links are carried
    across; a hedge whose options were not all imported (cancelled in the
    source) is imported unlinked and named in ``notes``. All or nothing.
    """
    data = json.loads(payload) if isinstance(payload, (str, bytes)) else payload
    if not isinstance(data, dict) or data.get("format") != EXPORT_FORMAT:
        raise TradeValidationError("not a trade book export")
    if int(data.get("version", 0)) > EXPORT_VERSION:
        raise TradeValidationError(f"export version {data.get('version')} is newer than supported")
    source = [Trade(**{key: value for key, value in terms.items()
                       if key in Trade.__dataclass_fields__})
              for terms in data.get("trades", []) if terms.get("status", LIVE) == LIVE]
    index = {trade.trade_id: position for position, trade in enumerate(source)}
    hedges, notes = {}, []
    for hedge_id, allocations in (data.get("hedges") or {}).items():
        if hedge_id not in index or not allocations:
            continue
        missing = [item["option"] for item in allocations if item["option"] not in index]
        if missing:
            notes.append(f"{hedge_id} imported without its links: {', '.join(missing)} "
                         "not imported")
            continue
        hedges[index[hedge_id]] = [HedgeAllocation(index[item["option"]], item["base_amount"])
                                   for item in allocations]
    fresh = [Trade(**{**trade.terms(), "trade_id": None, "version": 0, "package_id": None})
             for trade in source]
    if not fresh:
        raise TradeValidationError("the export has no live trades")
    booked = store.book(Booking(tuple(fresh), hedges), user=user, reason=reason)
    return {old.trade_id: new.trade_id for old, new in zip(source, booked)}, notes
