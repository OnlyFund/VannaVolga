"""Session market layers: store < file < edits, with each pair's origin."""
import copy

import pytest

from app.market_layers import (
    FILE, MANUAL, STORE, describe, drop_store, mark_files, merge_store, origins,
    record_tables, unavailable,
)
from golden.app_harness import load_markets

pytestmark = pytest.mark.filterwarnings("ignore")
MARKETS = load_markets("MARKETS_20261015.json")         # EURUSD, USDJPY, USDMXN


def store_load(*pairs):
    return {pair: copy.deepcopy(MARKETS[pair]) for pair in pairs}


def edited(payload, spot):
    return dict(copy.deepcopy(payload), spot=spot)


def test_a_store_load_fills_the_library_with_store_pairs():
    library, layers = {}, {}
    notes = merge_store(library, layers, store_load("EURUSD", "USDJPY"), {}, 23)
    assert notes == [] and sorted(library) == ["EURUSD", "USDJPY"]
    assert layers["EURUSD"]["source"] == STORE and describe(layers["EURUSD"]) == "market 23"
    assert origins(layers) == {"EURUSD": 23, "USDJPY": 23}


def test_reruns_and_other_valuation_dates_are_not_edits():
    library, layers = {}, {}
    merge_store(library, layers, store_load("EURUSD"), {}, 23)
    same = dict(copy.deepcopy(MARKETS["EURUSD"]), valuation_date="2026-10-18")
    assert record_tables(layers, "EURUSD", same) is True
    assert not layers["EURUSD"]["edited"] and origins(layers) == {"EURUSD": 23}


def test_an_edit_is_flagged_kept_on_reload_and_clean_again_when_undone():
    library, layers = {}, {}
    merge_store(library, layers, store_load("EURUSD", "USDJPY"), {}, 23)
    record_tables(layers, "EURUSD", edited(MARKETS["EURUSD"], 1.2))
    assert layers["EURUSD"]["edited"] and describe(layers["EURUSD"]) == "edited, on market 23"
    assert origins(layers) == {"USDJPY": 23}               # edited: no shared disk cache
    library["EURUSD"] = edited(MARKETS["EURUSD"], 1.2)
    notes = merge_store(library, layers, store_load("EURUSD", "USDJPY"), {}, 24)
    assert notes == ["EURUSD kept as edited; not reloaded from market 24"]
    assert library["EURUSD"]["spot"] == 1.2                 # the edit survives
    assert layers["USDJPY"]["market_id"] == 24              # the clean pair reloads
    record_tables(layers, "EURUSD", copy.deepcopy(MARKETS["EURUSD"]))   # edit undone
    assert not layers["EURUSD"]["edited"]


def test_files_win_over_the_store():
    library, layers = {}, {}
    merge_store(library, layers, store_load("EURUSD"), {}, 23)
    uploaded = {"EURUSD": edited(MARKETS["EURUSD"], 1.15)}
    library.update(uploaded)
    mark_files(layers, uploaded, "EURUSD_desk.json")
    assert describe(layers["EURUSD"]) == "file EURUSD_desk.json"
    notes = merge_store(library, layers, store_load("EURUSD"), {}, 24)
    assert notes == ["EURUSD kept as from EURUSD_desk.json; not reloaded from market 24"]
    assert library["EURUSD"]["spot"] == 1.15 and origins(layers) == {}


def test_a_store_pair_the_new_market_cannot_build_becomes_unavailable():
    library, layers = {}, {}
    merge_store(library, layers, store_load("EURUSD", "USDJPY"), {}, 23)
    merge_store(library, layers, store_load("EURUSD"),
                {"USDJPY": "USDJPY: no JPY discount rate"}, 24)
    assert "USDJPY" not in library
    assert unavailable(layers) == {"USDJPY": "USDJPY: no JPY discount rate"}
    assert origins(layers) == {"EURUSD": 24}
    # Its stale tables, unedited, are kept out of pricing ...
    assert record_tables(layers, "USDJPY", copy.deepcopy(MARKETS["USDJPY"])) is False
    # ... but edited, they are the user's own market and do price.
    assert record_tables(layers, "USDJPY", edited(MARKETS["USDJPY"], 151.0)) is True
    assert unavailable(layers) == {}
    # And when a later market can build it again, it reloads cleanly.
    record_tables(layers, "USDJPY", copy.deepcopy(MARKETS["USDJPY"]))
    merge_store(library, layers, store_load("EURUSD", "USDJPY"), {}, 25)
    assert layers["USDJPY"]["market_id"] == 25 and unavailable(layers) == {}


def test_unavailable_pairs_never_seen_are_listed_and_typed_pairs_are_manual():
    library, layers = {}, {}
    merge_store(library, layers, store_load("EURUSD"), {"USDCZK": "USDCZK: no CZK rate"}, 23)
    assert unavailable(layers) == {"USDCZK": "USDCZK: no CZK rate"}
    assert record_tables(layers, "GBPUSD", edited(MARKETS["EURUSD"], 1.3)) is True
    assert layers["GBPUSD"]["source"] == MANUAL and describe(layers["GBPUSD"]) == "typed in"
    merge_store(library, layers, store_load("EURUSD"), {}, 24)
    assert layers["GBPUSD"]["source"] == MANUAL               # never replaced by the store


def test_no_usable_market_drops_clean_store_pairs_but_keeps_edits():
    """Moving the valuation date before every stored market must not keep
    pricing on the later market's data."""
    library, layers = {}, {}
    merge_store(library, layers, store_load("EURUSD", "USDJPY", "USDMXN"), {}, 23)
    record_tables(layers, "USDJPY", edited(MARKETS["USDJPY"], 151.0))
    library["USDJPY"] = edited(MARKETS["USDJPY"], 151.0)
    uploaded = {"USDMXN": copy.deepcopy(MARKETS["USDMXN"])}
    mark_files(layers, uploaded, "mxn.json")
    reason = "no market data on or before 2026-10-15"
    notes = drop_store(library, layers, reason)
    assert sorted(library) == ["USDJPY", "USDMXN"]               # edit and file kept
    assert unavailable(layers) == {"EURUSD": reason}
    assert notes == ["USDJPY kept as edited (on market 23), although no stored market "
                     "applies now: " + reason]
    assert origins(layers) == {}


def test_messages_shared_by_many_pairs_become_one_line():
    from app.market_layers import group_by_message, pair_list
    pairs = [f"USD{c}" for c in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH")]
    messages = {pair: "no market data on or before 2026-10-15" for pair in pairs}
    messages["USDCZK"] = "USDCZK: no CZK discount rate"
    grouped = dict((text, found) for found, text in group_by_message(messages))
    assert grouped["no market data on or before 2026-10-15"] == sorted(pairs)
    assert grouped["no CZK discount rate"] == ["USDCZK"]
    assert pair_list(pairs) == "USDAAA, USDBBB, USDCCC, USDDDD, USDEEE, USDFFF and 2 more"
    assert pair_list(["EURUSD"]) == "EURUSD"
