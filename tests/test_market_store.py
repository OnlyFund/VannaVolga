"""Single-pair and multi-pair market files."""
import json

import pandas as pd
import pytest

from app.market_store import (
    BUNDLE_FORMAT, bundle_payload, choose_active_pair, market_payload,
    read_market_file, validate_market,
)


def market(pair="EURUSD", atm=10.0, **overrides):
    payload = dict(
        currency_pair=pair, valuation_date="2026-10-02", spot=1.1,
        event_calendar=[],
        vol_data=[dict(tenor="1M", expiry="2026-11-02", atm=atm, rr_25=-.5,
                       bf_25=.2, rr_10=-1, bf_10=.6)],
        forward_points=[dict(tenor="1M", delivery_date="2026-11-04", fwd_pts=30,
                             term_discount=.04)],
    )
    payload.update(overrides)
    return payload


def test_single_pair_file_loads_as_before():
    markets, errors = read_market_file(json.dumps(market("eur/usd")).encode())
    assert list(markets) == ["EURUSD"] and not errors
    assert markets["EURUSD"]["currency_pair"] == "EURUSD"


def test_payload_round_trips_through_a_bundle():
    loaded = validate_market(market())
    single = market_payload("EURUSD", "2026-10-02", 1.1, 1e4, True,
                            loaded["vol_rows"], loaded["forward_rows"], loaded["event_rows"])
    bundle = bundle_payload({"EURUSD": single, "USDJPY": market("USDJPY", spot=150.)},
                            "2026-10-02")
    assert bundle["format"] == BUNDLE_FORMAT
    markets, errors = read_market_file(json.dumps(bundle))
    assert list(markets) == ["EURUSD", "USDJPY"] and not errors
    again = validate_market(markets["EURUSD"])
    pd.testing.assert_frame_equal(again["vol_rows"], loaded["vol_rows"])
    assert validate_market(markets["USDJPY"])["spot"] == 150.


def test_bad_pair_is_skipped_and_reported_without_losing_the_rest():
    bundle = bundle_payload({"EURUSD": market(), "GBPUSD": market("GBPUSD", atm=-1.)})
    markets, errors = read_market_file(json.dumps(bundle))
    assert list(markets) == ["EURUSD"]
    assert "ATM volatilities must be positive" in errors["GBPUSD"]


def test_file_without_a_valid_pair_fails_whole():
    with pytest.raises(ValueError, match="No valid market"):
        read_market_file(json.dumps({"markets": {"EURUSD": market(spot=-1)}}))
    with pytest.raises(ValueError, match="not valid JSON"):
        read_market_file(b"{not json")
    with pytest.raises(ValueError, match="Not a market file"):
        read_market_file(json.dumps({"hello": 1}))


def test_loose_shapes_fall_back_to_keys_and_bundle_defaults():
    entry = market()
    del entry["currency_pair"], entry["valuation_date"]
    # Bare {pair: market} mapping, a BOM, and the date inherited from the top.
    raw = ("﻿" + json.dumps({"valuation_date": "2026-10-02", "EURUSD": entry})).encode()
    markets, errors = read_market_file(raw)
    assert markets["EURUSD"]["currency_pair"] == "EURUSD" and not errors
    assert markets["EURUSD"]["valuation_date"] == "2026-10-02"
    # A list of self-describing markets.
    markets, _ = read_market_file({"markets": [market(), market("USDJPY")]})
    assert list(markets) == ["EURUSD", "USDJPY"]


def test_conflicting_or_duplicate_pairs_are_rejected():
    markets, errors = read_market_file({"markets": {
        "EURUSD": market(), "GBPUSD": market("EURUSD")}})
    assert list(markets) == ["EURUSD"]
    assert "Filed under GBPUSD but quotes EURUSD" in errors["GBPUSD"]
    markets, errors = read_market_file([market(atm=9.), market(atm=11.)])
    assert markets["EURUSD"]["vol_data"][0]["atm"] == 9.
    assert any("Duplicate" in error for error in errors.values())


def test_newer_bundle_version_is_refused():
    with pytest.raises(ValueError, match="newer"):
        read_market_file(dict(bundle_payload({"EURUSD": market()}), version=99))


def test_active_pair_prefers_the_current_one():
    assert choose_active_pair(["EURUSD", "USDJPY"], "usd/jpy") == "USDJPY"
    assert choose_active_pair(["EURUSD", "USDJPY"], "GBPUSD") == "EURUSD"
