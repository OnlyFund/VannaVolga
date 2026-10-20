"""Spot and forward deals: the Pricer's Spot / forward grid, priced and booked.

A line is ``currency_pair``, ``buy_sell``, ``amount`` (``10m``, ``250k``)
in ``amount_ccy`` (``BASE``, the default, or ``TERM``, or either currency's
code: buying term currency is selling base), ``value_date``, ``rate`` (blank
takes the market outright at the value date), ``book`` and ``counterparty``.

Value dates, on the pair's joint calendar:

* ``SPOT`` (or blank): the spot date;
* ``TOM``: the business day after today, when it is before spot;
* ``SN``: the business day after spot;
* a tenor (``1W``, ``3M``, ``18M``, ``1Y``): spot plus the tenor, modified
  following, end of month (the forward and option-delivery convention);
* a date: it must be a good business day.

A deal settling today (``TOD``) or earlier is refused. Before spot it needs
the market's short swaps (``TN``, and ``ON`` for today): without them the
deal would value at spot, so it is refused rather than booked off-market.

Each line prices exactly as the booked trade will (``pricer.value_forward``
on the trade's own terms), and booking is all or nothing.
"""
import math
import re
from datetime import date

import pandas as pd
import QuantLib as ql

from app import pricer
from app.premium_side import BASE, TERM
from app.trades.model import FX_FORWARD, Booking, Trade, TradeValidationError, as_date
from helper_functions.fx_conventions import get_pair, joint_calendar, spot_date
from helper_functions.input_resolvers import parse_expiry, parse_notional

COLUMNS = ("currency_pair", "buy_sell", "amount", "amount_ccy", "value_date", "rate",
           "book", "counterparty", "trade_id")
SPOT_TOKENS = ("", "SPOT", "SP")
_TENOR = re.compile(r"(\d+)([DWMY])")


def _text(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    return str(value).strip()


def _ql(day):
    day = as_date(day)
    return ql.Date(day.day, day.month, day.year)


def _py(day):
    return date(day.year(), day.month(), day.dayOfMonth())


def value_date_for(pair, value, valuation_date):
    """The value date ``value`` (a token, a tenor or a date) names for ``pair``."""
    calendar = joint_calendar(pair)
    today = _ql(valuation_date)
    spot = spot_date(pair, today)
    if isinstance(value, (date, pd.Timestamp)):
        token = ""
        typed = as_date(value)
    else:
        token = _text(value).upper().replace("/", "")
        typed = None
    if typed is None:
        if token in SPOT_TOKENS:
            return _py(spot)
        if token in ("TOD", "ON"):
            raise ValueError("a deal for value today settles today: book it for TOM or later")
        if token in ("TOM", "TN"):
            tomorrow = calendar.advance(today, ql.Period(1, ql.Days), ql.Following)
            if tomorrow >= spot:
                return _py(spot)          # a T+1 pair's TOM is its spot
            return _py(tomorrow)
        if token == "SN":
            return _py(calendar.advance(spot, ql.Period(1, ql.Days), ql.Following))
        match = _TENOR.fullmatch(token)
        if match:
            count, unit = int(match.group(1)), match.group(2)
            if count <= 0:
                raise ValueError(f"tenor {token} must be positive")
            period = ql.Period(count, {"D": ql.Days, "W": ql.Weeks, "M": ql.Months,
                                       "Y": ql.Years}[unit])
            return _py(calendar.advance(spot, period, ql.ModifiedFollowing, True))
        typed = parse_expiry(value, valuation_date)
    if not calendar.isBusinessDay(_ql(typed)):
        good = _py(calendar.adjust(_ql(typed), ql.Following))
        raise ValueError(f"value date {typed} is not a business day for {pair} "
                         f"(next: {good})")
    return typed


def _amount_ccy(pair, value):
    token = _text(value).upper()
    if token in ("", BASE, pair[:3]):
        return pair[:3]
    if token in (TERM, pair[3:]):
        return pair[3:]
    raise ValueError(f"amount currency must be {pair[:3]} (base) or {pair[3:]} (term), "
                     f"not {token}")


def _rate(value):
    """The typed rate, or ``None`` for the market outright."""
    text = _text(value)
    if not text:
        return None
    try:
        rate = float(text.replace(",", ""))
    except ValueError:
        raise ValueError(f"rate must be a number, not {text!r}") from None
    if not rate > 0 or rate != rate or rate == float("inf"):
        raise ValueError("rate must be a positive number")
    return rate


def rate_decimals(pair):
    """Decimals a booked rate carries: two past the pip (EURUSD 6, USDJPY 4)."""
    return max(0, round(-math.log10(float(get_pair(pair).pip)))) + 2


def market_outright(markets, pair, value_date):
    """``(outright, note)`` at ``value_date``: spot plus the curve's points."""
    market = pricer.market_for_pair(markets, pair).market_for_delivery(value_date)
    return market.s + market.fwd_pts / market.forward_divisor, getattr(
        market, "forward_note", None)


def line_trade(markets, row, default_pair=None, trade_date=None, book="", counterparty=""):
    """The normalized ``Trade`` a grid line books; raises ``ValueError`` on bad input.

    A blank rate takes the market outright, rounded to ``rate_decimals``
    (a booked rate is a quotable one). A value date before spot the
    market cannot date (no short swaps) is refused.
    """
    pair = _text(row.get("currency_pair")).upper().replace("/", "") or _text(
        default_pair).upper()
    if not re.fullmatch(r"[A-Z]{6}", pair):
        raise ValueError(f"currency pair {pair or '(blank)'} is not a pair such as EURUSD")
    valuation = as_date(markets.valuation_date)
    side = _text(row.get("buy_sell")).upper()
    if side not in ("BUY", "SELL"):
        raise ValueError("buy/sell must be BUY or SELL")
    amount_text = _text(row.get("amount"))
    if not amount_text:
        raise ValueError("enter an amount")
    amount = parse_notional(amount_text)
    if not amount > 0:
        raise ValueError("amount must be positive (use SELL to sell)")
    amount_ccy = _amount_ccy(pair, row.get("amount_ccy"))
    value_date = value_date_for(pair, row.get("value_date"), valuation)
    if value_date <= valuation:
        raise ValueError(f"value date {value_date} must be after today ({valuation})")
    outright, note = market_outright(markets, pair, value_date)
    if note:
        raise ValueError(f"{note}; not booked off-market, choose spot or later")
    rate = _rate(row.get("rate"))
    return Trade(
        product_type=FX_FORWARD, currency_pair=pair,
        trade_date=as_date(trade_date or valuation), expiry=None, value_date=value_date,
        buy_sell=side, notional=amount,
        contract_rate=round(outright, rate_decimals(pair)) if rate is None else rate,
        amount_ccy=amount_ccy,
        book=_text(row.get("book")) or book,
        counterparty=_text(row.get("counterparty")) or counterparty,
    ).normalized()


def price_trade(markets, trade, extras=()):
    """The valuation row of one deal, as the book prices it."""
    return pricer.value_forward(markets, trade.currency_pair, trade.expiry,
                                trade.signed_base_amount(), trade.contract_rate,
                                extras=extras, value_date=trade.value_date)


def usd_per(markets, ccy):
    """USD per one ``ccy`` from the loaded markets' spots, or ``nan``."""
    ccy = str(ccy).upper()
    if ccy == "USD":
        return 1.0
    for pair, invert in ((f"{ccy}USD", False), (f"USD{ccy}", True)):
        if pair in markets.pairs:
            try:
                spot = float(markets.market(pair).spot)
            except Exception:
                continue
            return 1.0 / spot if invert else spot
    return float("nan")


def price_lines(markets, rows, extras=(), default_pair=None):
    """Price grid lines; returns ``(results, errors)``.

    Blank lines are skipped. Each result has the line number (``row``), the
    resolved terms (value date, rate, whether it is the market's, amounts in
    both currencies) and the valuation, with MTM in USD.
    """
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    results, errors = [], []
    for number, (_, row) in enumerate(frame.reset_index(drop=True).iterrows(), start=1):
        row = row.to_dict()
        if not any(_text(row.get(c)) for c in ("currency_pair", "amount", "value_date",
                                                "rate")):
            continue
        try:
            trade = line_trade(markets, row, default_pair)
            value = price_trade(markets, trade, extras)
        except (TradeValidationError, ValueError) as exc:
            errors.append(f"Line {number}: {exc}")
            continue
        usd = usd_per(markets, trade.base)
        results.append(dict(
            value, row=number, rate_auto=_rate(row.get("rate")) is None,
            contract_rate=trade.contract_rate, amount_ccy=trade.amount_ccy,
            term_amount=trade.signed_term_amount(),
            mtm_usd=value["mtm_base"] * usd, settlement_usd=value["settlement_base"] * usd,
        ))
    return results, errors


def book_lines(store, markets, rows, book="", counterparty="", user="", default_pair=None):
    """Book grid lines as one package; all or nothing.

    Returns a ``service.BookingResult``; its ``lines`` map line numbers to
    the booked trade ids. A line already booked (it has a trade id) is
    refused: add a new line to book the same terms again.
    """
    from app.trades.service import BookingResult
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
    trades, numbers, errors = [], [], []
    for number, (_, row) in enumerate(frame.reset_index(drop=True).iterrows(), start=1):
        row = row.to_dict()
        if _text(row.get("trade_id")):
            errors.append(f"Line {number} is already booked as {_text(row['trade_id'])}")
            continue
        try:
            trade = line_trade(markets, row, default_pair, book=book,
                               counterparty=counterparty)
            price_trade(markets, trade)          # books only what prices
        except (TradeValidationError, ValueError) as exc:
            errors.append(f"Line {number}: {exc}")
            continue
        trades.append(trade)
        numbers.append(number)
    if errors:
        return BookingResult(errors=errors)
    if not trades:
        return BookingResult(errors=["nothing to book"])
    try:
        booked = store.book(Booking(tuple(trades)), user=user, reason="spot/forward")
    except (TradeValidationError, ValueError) as exc:
        return BookingResult(errors=[str(exc)])
    return BookingResult(booked=booked,
                         lines={n: t.trade_id for n, t in zip(numbers, booked)})
