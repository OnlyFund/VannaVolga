"""Cashflow conventions shared by portfolio valuation reports."""

from math import isfinite


def signed_premium_cashflow(premium, buy_sell):
    """Return premium cashflow: negative for a buy and positive for a sale."""
    amount = float(premium)
    if not isfinite(amount) or amount < 0.0:
        raise ValueError("Premium must be a finite non-negative total amount")
    side = str(buy_sell).strip().upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("Buy/Sell must be BUY or SELL")
    return -amount if side == "BUY" else amount
