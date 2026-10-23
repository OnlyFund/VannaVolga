"""Several pairs priced together: routing, isolation and stale-market guards."""
import re
from copy import deepcopy
from datetime import date

import pandas as pd
import pytest

from app import pricer
from app.market_context import MarketContext, MarketContextError, UnavailableMarket
from app.market_set import MarketSet
from golden.app_harness import load_markets
from market_data import Modifications

pytestmark = pytest.mark.filterwarnings("ignore")

MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)
VALUATION = date(2026, 10, 15)


@pytest.fixture(scope="module")
def payloads():
    return load_markets("MARKETS_20261015.json")


@pytest.fixture(scope="module")
def markets(payloads):
    return MarketSet.from_payloads(payloads, VALUATION)


def trade(pair, **changes):
    line = dict(currency_pair=pair, strike="atm", expiry="1M", call_put="CALL",
                buy_sell="BUY", notional="1m", premium=0.0, premium_ccy=pair[3:])
    line.update(changes)
    return line


def test_bundle_builds_one_market_per_pair(markets):
    assert sorted(markets.pairs) == ["EURUSD", "USDJPY", "USDMXN"]
    assert sorted(markets.available()) == sorted(markets.pairs)
    assert markets.problems() == {}
    assert "eur/usd" in markets


def test_mixed_book_prices_each_line_on_its_own_pair(markets, payloads):
    book = [trade("USDJPY"), trade("EURUSD", strike="25d", call_put="PUT"),
            trade("USDMXN", expiry="3M"), trade("EURUSD", expiry="1W")]
    results, errors = pricer.price_lines(markets, book, MODS)
    assert errors == []
    assert [result["currency_pair"] for result in results] == ["USDJPY", "EURUSD", "USDMXN", "EURUSD"]
    for number, line in enumerate(book, start=1):
        alone = MarketContext.from_payload(payloads[line["currency_pair"]])
        expected = pricer.price_line(alone, number, pd.Series(line), MODS)[1]
        assert results[number - 1] == expected


def test_missing_pair_is_named_and_an_inverse_pair_gets_a_hint(markets):
    results, errors = pricer.price_lines(
        markets, [trade("GBPUSD"), trade("USDEUR"), trade("EURUSD")], MODS)
    assert len(results) == 1
    assert errors[0] == "Line 1: No market loaded for GBPUSD"
    assert errors[1] == "Line 2: No market loaded for USDEUR; EURUSD is loaded, enter the pair as EURUSD"


def test_broken_pair_fails_only_its_own_lines(payloads):
    stored = deepcopy(payloads)
    stored["USDJPY"]["spot"] = 0.0
    del stored["USDMXN"]["vol_data"]
    markets = MarketSet.from_payloads(stored, VALUATION)
    assert markets.available() == ["EURUSD"]
    assert set(markets.problems()) == {"USDJPY", "USDMXN"}
    results, errors = pricer.price_lines(
        markets, [trade("USDJPY"), trade("EURUSD"), trade("USDMXN")], MODS)
    assert [result["currency_pair"] for result in results] == ["EURUSD"]
    assert errors[0] == "Line 1: USDJPY Invalid market: Spot must be finite and positive"
    assert errors[1].startswith("Line 3: USDMXN Invalid market")


def test_market_saved_for_another_day_is_valued_as_of_the_set_date_when_rolling_is_off(
        payloads, monkeypatch):
    monkeypatch.setenv("VV_ROLL_MARKET", "0")
    stored = deepcopy(payloads)
    stored["USDJPY"]["valuation_date"] = "2026-10-14"
    markets = MarketSet.from_payloads(stored, VALUATION)
    assert sorted(markets.available()) == ["EURUSD", "USDJPY", "USDMXN"]
    assert markets.notes() == {"USDJPY": "USDJPY quotes were saved for 2026-10-14 "
                                         "and are valued as of 2026-10-15 "
                                         "(rolling is off: VV_ROLL_MARKET=0)"}
    results, errors = pricer.price_lines(markets, [trade("USDJPY")], MODS)
    assert errors == []
    # Same quotes as the correctly dated market, so the same price.
    expected, _ = pricer.price_lines(MarketSet.from_payloads(payloads, VALUATION),
                                     [trade("USDJPY")], MODS)
    assert results == expected


def test_market_saved_for_an_earlier_day_is_rolled(payloads):
    stored = deepcopy(payloads)
    stored["USDJPY"]["valuation_date"] = "2026-10-14"
    markets = MarketSet.from_payloads(stored, VALUATION)
    assert sorted(markets.available()) == ["EURUSD", "USDJPY", "USDMXN"]
    # Rolled only when first priced (a roll builds every pillar's smile).
    assert markets.rolls_when_priced("USDJPY") and not markets.rolls_when_priced("EURUSD")
    assert markets.notes()["USDJPY"] == ("USDJPY quotes of 2026-10-14, rolled to 2026-10-15 "
                                         "when priced")
    assert markets.spot("USDJPY") == stored["USDJPY"]["spot"]
    assert markets.rolls_when_priced("USDJPY")                     # a spot is no reason to roll
    context = markets.market("USDJPY")
    assert not markets.rolls_when_priced("USDJPY")
    note = markets.notes()["USDJPY"]
    assert note.startswith("USDJPY quotes of 2026-10-14 rolled to 2026-10-15")
    assert context.rolled_from == pd.Timestamp(2026, 10, 14) and not context.broker_strangle
    results, errors = pricer.price_lines(markets, [trade("USDJPY")], MODS)
    assert errors == [] and len(results) == 1


def test_a_market_that_cannot_roll_is_valued_as_before_and_says_why(payloads):
    stored = deepcopy(payloads)
    stored["USDJPY"]["valuation_date"] = "2026-10-14"
    stored["USDJPY"]["vol_data"][0]["tenor"] = ""                  # a pillar without a tenor
    markets = MarketSet.from_payloads(stored, VALUATION)
    assert "USDJPY" in markets.available()
    markets.market("USDJPY")
    assert "not rolled: a vol row has no tenor" in markets.notes()["USDJPY"]
    assert not hasattr(markets.market("USDJPY"), "rolled_from")


def test_context_added_for_another_date_is_not_silently_redated(payloads):
    context = MarketContext.from_payload(payloads["EURUSD"], valuation_date="2026-10-14")
    markets = MarketSet(VALUATION, [context])
    assert "valued on 2026-10-14, not on the pricing date 2026-10-15" in markets.problems()["EURUSD"]


def test_half_typed_live_pair_is_not_added(payloads):
    from app.market_context import build_context
    live = build_context("EUR", VALUATION, 1.1, 1e4, True, None, None)
    markets = MarketSet.from_payloads(payloads, VALUATION, live=live)
    assert sorted(markets.pairs) == ["EURUSD", "USDJPY", "USDMXN"]


def test_market_stored_under_the_wrong_pair_is_rejected(payloads):
    markets = MarketSet.from_payloads({"USDJPY": payloads["EURUSD"]}, VALUATION)
    assert markets.problems() == {"USDJPY": "market stored under USDJPY contains EURUSD"}


def test_live_edits_replace_the_stored_market(payloads):
    live_payload = deepcopy(payloads["EURUSD"])
    live_payload["spot"] = 1.2
    live = MarketContext.from_payload(live_payload)
    markets = MarketSet.from_payloads(payloads, VALUATION, live=live)
    assert markets.market("EURUSD") is live
    assert sorted(markets.available()) == ["EURUSD", "USDJPY", "USDMXN"]


def test_empty_set_reports_every_line():
    _, errors = pricer.price_lines(MarketSet(VALUATION), [trade("EURUSD")], MODS)
    assert errors == ["Line 1: No market loaded for EURUSD"]
    with pytest.raises(MarketContextError):
        MarketSet(VALUATION).market("EURUSD")


def test_hedges_book_one_forward_per_pair_and_expiry(markets):
    book = [trade("EURUSD"), trade("USDJPY"), trade("EURUSD", strike="25d")]
    results, _ = pricer.price_lines(markets, book, MODS)
    trades, errors = pricer.hedge_trades_for(markets, results)
    assert errors == []
    assert sorted(t["currency_pair"] for t in trades) == ["EURUSD", "USDJPY"]
    positions, position_errors = pricer.build_delta_hedges(markets, pd.DataFrame(book), MODS)
    assert position_errors == []
    assert sorted(p["currency_pair"] for p in positions) == ["EURUSD", "USDJPY"]


def test_strike_aliases_resolve_on_each_rows_own_pair(markets):
    rows = pd.DataFrame([trade("USDJPY", strike="25d"), trade("EURUSD", strike="25d")])
    resolved, changed, failures = pricer.resolve_strike_aliases(markets, rows)
    assert changed and failures == []
    assert float(resolved.at[0, "strike"]) > 100 > 2 > float(resolved.at[1, "strike"])


def test_switching_pairs_keeps_the_valuation_date(payloads, monkeypatch):
    import app.ui.market_block as block
    queued = []
    block._activate_market(payloads["USDJPY"], queued.append, date(2026, 10, 16))
    assert queued[0]["valuation_date"] == "2026-10-16"
    assert payloads["USDJPY"]["valuation_date"].startswith("2026-10-15")   # stored copy untouched
    block._activate_market(payloads["USDJPY"], queued.append)
    assert queued[1]["valuation_date"].startswith("2026-10-15")


def test_new_line_resolves_pair_tenor_and_delta_strike(markets):
    rows = pd.DataFrame([dict(currency_pair=None, strike="25d", expiry="1M", call_put="PUT",
                              buy_sell="BUY", notional="1m", premium=0.0, premium_ccy="USD")])
    resolved, changed, failures = pricer.resolve_line_inputs(markets, rows, "EURUSD")
    assert changed and failures == []
    line = resolved.iloc[0]
    assert line["currency_pair"] == "EURUSD"
    assert line["expiry"] == pd.Timestamp(markets.market("EURUSD").resolve_market_expiry("1M")).strftime("%Y-%m-%d")
    assert re.fullmatch(r"1\.\d{4}", line["strike"])
    # Priced as booked: the resolved line prices on the date it shows.
    results, errors = pricer.price_lines(markets, resolved, MODS)
    assert errors == [] and str(results[0]["expiry"])[:10] == line["expiry"]


def test_expiry_resolves_before_the_strike_can(markets):
    rows = pd.DataFrame([trade("USDJPY", strike="25d", expiry="ON", call_put=None),
                         trade("USDJPY", strike="25d", expiry="18nov26", call_put="CALL")])
    resolved, changed, failures = pricer.resolve_line_inputs(markets, rows)
    assert changed and failures == []
    assert resolved.at[0, "expiry"] == "2026-10-16" and resolved.at[0, "strike"] == "25d"
    assert resolved.at[1, "expiry"] == "2026-11-18"
    assert re.fullmatch(r"1\d\d\.\d{2}", resolved.at[1, "strike"])


def test_booked_lines_are_left_alone(markets):
    rows = pd.DataFrame([trade("EURUSD", strike="1.1200", expiry="2026-11-17")])
    resolved, changed, failures = pricer.resolve_line_inputs(markets, rows, "EURUSD")
    assert not changed and failures == []
    pd.testing.assert_frame_equal(resolved, rows)
    blank = pd.DataFrame([dict(currency_pair=None, strike=None, expiry=None, call_put=None)])
    assert pricer.resolve_line_inputs(markets, blank, "EURUSD")[1] is False


def test_complete_lines_that_cannot_resolve_are_reported(markets):
    rows = pd.DataFrame([trade("GBPUSD", strike="25d"),
                         trade("EURUSD", strike="1.12", expiry="2030-01-01x"),
                         trade("EURUSD", strike="25", expiry="1M")])        # half-typed alias
    resolved, changed, failures = pricer.resolve_line_inputs(markets, rows, "EURUSD")
    assert failures[0] == "Line 1: No market loaded for GBPUSD"
    assert failures[1].startswith("Line 2: ")
    assert len(failures) == 2
    assert resolved.at[2, "expiry"] != "1M" and resolved.at[2, "strike"] == "25"


def test_placeholder_line_from_the_blank_row_is_not_committed():
    """Committing it would reset the editor and drop the cell being typed."""
    from app.ui.portfolio_block import _only_placeholder_rows_added
    placeholder = {"edited_rows": {}, "deleted_rows": [],
                   "added_rows": [{"premium_ccy": "USD", "_remove": False}]}
    assert _only_placeholder_rows_added(placeholder)
    typed = dict(placeholder, added_rows=[{"premium_ccy": "USD", "strike": "25d"}])
    assert not _only_placeholder_rows_added(typed)
    mixed = dict(placeholder, edited_rows={0: {"notional": "2m"}})
    assert not _only_placeholder_rows_added(mixed)
    assert not _only_placeholder_rows_added({"edited_rows": {}, "added_rows": [], "deleted_rows": []})


def test_empty_strike_text_is_blank_not_an_alias(markets):
    rows = pd.DataFrame([trade("EURUSD", strike="nan", expiry="1M", call_put=None)])
    resolved, changed, failures = pricer.resolve_line_inputs(markets, rows, "EURUSD")
    assert failures == [] and resolved.at[0, "strike"] == "nan"
    assert changed   # the expiry still resolves


def test_new_grid_line_without_a_premium_prices(markets):
    for empty in (None, float("nan"), ""):
        results, errors = pricer.price_lines(markets, [trade("EURUSD", premium=empty)], MODS)
        assert errors == [] and results[0]["pnl_base"] == pytest.approx(results[0]["mtm_base"])
    _, errors = pricer.price_lines(markets, [trade("EURUSD", premium=-1.0)], MODS)
    assert errors == ["Line 1: Premium must be a finite non-negative amount"]


def test_a_rolled_market_is_kept_across_market_sets(payloads, monkeypatch):
    """Reruns build a new set each time; the same market rolled the same day rolls once."""
    import app.rolled_market as module
    from app.market_service import MarketService
    from app.smile_service import DiagnosticsCollector
    stored = deepcopy(payloads)
    stored["USDJPY"]["valuation_date"] = "2026-10-14"
    service, diagnostics = MarketService(DiagnosticsCollector()), DiagnosticsCollector()
    calls = []
    real = module.roll_context
    monkeypatch.setattr(module, "roll_context", lambda *args: calls.append(1) or real(*args))
    for _ in range(3):
        markets = MarketSet.from_payloads(stored, VALUATION, market_service=service,
                                          diagnostics=diagnostics)
        markets.market("USDJPY")
    assert len(calls) == 1


def test_with_spots_keeps_a_pair_to_roll_and_rolls_it_at_the_moved_spot(payloads):
    stored = deepcopy(payloads)
    stored["USDJPY"]["valuation_date"] = "2026-10-14"
    moved = MarketSet.from_payloads(stored, VALUATION).with_spots({"USDJPY": 151.0})
    assert moved.rolls_when_priced("USDJPY")
    assert moved.market("USDJPY").spot == 151.0
