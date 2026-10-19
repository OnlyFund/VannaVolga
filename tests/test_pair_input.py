"""The Markets tab's typed currency pair: only a supported six-letter pair is
used; anything else falls back to the last valid pair, EURUSD, or the market
set's first pair, and says what was rejected."""
import pytest

from app.pair_input import DEFAULT_PAIR, fallback_pair, pair_problem, resolve_pair_input


@pytest.mark.parametrize("typed, pair", [
    ("EURUSD", "EURUSD"), ("eur/usd", "EURUSD"), (" usd jpy ", "USDJPY"), ("EURJPY", "EURJPY")])
def test_a_supported_pair_is_accepted_normalized(typed, pair):
    assert resolve_pair_input(typed, "GBPUSD", ["GBPUSD"]) == (pair, None)


@pytest.mark.parametrize("typed, reason", [
    ("EURUS", "six letters"), ("EURUSDX", "six letters"), ("", "six letters"),
    ("EUR1SD", "six letters"), ("ABCDEF", "unknown pair ABCDEF"),
    ("USDEUR", "quoted as EURUSD")])
def test_anything_else_keeps_the_last_valid_pair_and_says_why(typed, reason):
    pair, message = resolve_pair_input(typed, "GBPUSD", ["GBPUSD", "USDJPY"])
    assert pair == "GBPUSD"
    assert reason in message and f"'{typed}'" in message and "kept GBPUSD" in message
    assert "invert" not in message


def test_fallback_order():
    # 1. the last valid active pair
    assert fallback_pair("USDJPY", ["EURUSD", "USDJPY"]) == "USDJPY"
    # 2. EURUSD when the market set has it (no active pair, or an invalid one)
    assert fallback_pair(None, ["USDJPY", "EURUSD"]) == "EURUSD"
    assert fallback_pair("ABCDEF", ["USDJPY", "EURUSD"]) == "EURUSD"
    # 3. the market set's first pair
    assert fallback_pair(None, ["USDJPY", "GBPUSD"]) == "USDJPY"
    # 4. EURUSD
    assert fallback_pair(None, []) == DEFAULT_PAIR == "EURUSD"
    assert fallback_pair("bad", ["also bad"]) == "EURUSD"


def test_pair_problem():
    assert pair_problem("EURUSD") is None
    assert pair_problem(None) is not None
