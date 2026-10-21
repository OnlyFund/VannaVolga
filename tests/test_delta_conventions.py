"""Deltas: spot-equivalent in every total, each line in its pair's convention.

The app computes deltas spot-equivalent (``delta_reporting="spot"``), so
totals, buckets and ladder rows add up across tenors and pairs. Each line
also carries its forward delta (the forward amount that hedges it) and its
delta in the pair's quoted convention, marked S or F.
"""
from datetime import date

import pandas as pd
import pytest

import pricing.greeks as greeks
from app import pricer
from app.market_set import MarketSet
from app.ui.portfolio_block import _with_line_results
from app.ui.trades_block import pair_convention_rows
from golden.app_harness import load_markets
from market_data import Modifications
from pricing import greek_catalog
from pricing.greeks import DEFAULT_DELTA_CONVENTION, convention_deltas, delta_convention

VALUATION = date(2026, 10, 15)
MODS = Modifications(beta_atm=0.7, beta_rr=1.2, beta_bf=0.0)


@pytest.fixture(scope="module")
def markets():
    return MarketSet.from_payloads(load_markets("MARKETS_20261015.json"), VALUATION)


def line(pair="EURUSD", expiry="6M", strike="25d", call_put="PUT", premium_ccy="TERM"):
    return dict(currency_pair=pair, expiry=expiry, call_put=call_put, strike=strike,
                buy_sell="BUY", notional="10m", premium=0.0, premium_ccy=premium_ccy)


@pytest.fixture(scope="module")
def priced(markets):
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame(
        [line(), line("USDJPY", premium_ccy="BASE"), line("USDMXN")]))
    results, errors = pricer.price_lines(markets, rows, MODS)
    assert errors == []
    return results


@pytest.mark.parametrize("pair, expiry, expected", [
    ("EURUSD", "2026-11-17", "spot"),        # G10 up to 1Y: spot delta
    ("EURUSD", "2027-10-15", "spot"),
    ("EURUSD", "2027-12-15", "forward"),     # beyond the 1Y cutoff
    ("USDJPY", "2027-04-15", "spot"),
    ("USDMXN", "2026-11-17", "spot"),        # EM too: spot up to 1Y (as Bloomberg)
    ("USDMXN", "2027-12-15", "forward"),     # and forward beyond
    ("eur/usd", date(2026, 11, 17), "spot"),
])
def test_delta_convention_follows_the_pair_and_tenor(pair, expiry, expected):
    # Hyperion resolves it from the pair's calendars; the core reads it off the
    # market's carried inputs (it reads no calendar itself).
    from apeiron.hyperion.resolve import attach, resolve_inputs
    from market_data import MarketData
    assert resolve_inputs(pair, expiry, VALUATION).delta_convention == expected
    market = attach(MarketData(s=1.1, fwd_pts=10, v_atm=10, rr_25=0, rr_10=0, bf_25=0, bf_10=0,
                               currency_pair=pair), expiry, VALUATION)
    assert delta_convention(pair, expiry, VALUATION, market) == expected


def test_delta_convention_falls_back_to_forward():
    assert DEFAULT_DELTA_CONVENTION == "forward"
    assert delta_convention("XXXYYY", "2026-11-17", VALUATION) == "forward"
    assert delta_convention("EURUSD", "not a date", VALUATION) == "forward"


def test_lines_carry_spot_forward_and_pair_deltas(markets, priced):
    eurusd, usdjpy, usdmxn = priced
    for result in priced:
        market = pricer.market_for_pair(markets, result["currency_pair"]).market_for_expiry(
            result["expiry"])
        assert market.delta_reporting == "spot"
        factor = greeks.forward_delta_factor(market, result["expiry"], VALUATION)
        assert 0.9 < factor < 1.0
        for prefix in ("", "fixed_"):
            # The reported delta is spot-equivalent; forward = spot / factor.
            assert result[f"{prefix}spot_delta"] == result[f"{prefix}delta"]
            assert result[f"{prefix}fwd_delta"] == pytest.approx(
                result[f"{prefix}delta"] / factor, rel=1e-12)
    assert eurusd["delta_convention"] == usdjpy["delta_convention"] == "S"
    assert eurusd["pair_delta"] == eurusd["spot_delta"]
    # Every pair quotes spot delta up to 1Y, EM included.
    assert usdmxn["delta_convention"] == "S"
    assert usdmxn["pair_delta"] == usdmxn["spot_delta"]
    assert usdmxn["fixed_pair_delta"] == usdmxn["fixed_spot_delta"]


def test_forward_reporting_converts_the_other_way(markets, priced):
    """A market reporting forward delta (files, tests) gives the same values."""
    eurusd = priced[0]
    market = pricer.market_for_pair(markets, "EURUSD").market_for_expiry(eurusd["expiry"])
    forward = market.copy_with(delta_reporting="forward")
    row = {"currency_pair": "EURUSD"}
    values = convention_deltas(forward, row, eurusd["expiry"], VALUATION,
                               eurusd["fixed_fwd_delta"], eurusd["fwd_delta"])
    for key in ("spot_delta", "fixed_spot_delta", "fwd_delta", "pair_delta"):
        assert values[key] == pytest.approx(eurusd[key], rel=1e-12)


def test_convention_deltas_fail_safe(markets, monkeypatch):
    market = pricer.market_for_pair(markets, "EURUSD").market_for_expiry(date(2027, 4, 15))
    row = {"currency_pair": "EURUSD"}
    expiry = date(2027, 4, 15)
    # Legacy reporting or a bad delta: nothing, the reported delta stays.
    assert convention_deltas(market.copy_with(delta_reporting=None), row, expiry, VALUATION,
                             1.0, 1.0) == {}
    assert convention_deltas(market, row, expiry, VALUATION, None, 1.0) == {}
    assert convention_deltas(market, row, expiry, VALUATION, float("nan"), 1.0) == {}
    monkeypatch.setattr(greeks, "forward_delta_factor", lambda *a: 0.0)
    assert convention_deltas(market, row, expiry, VALUATION, 1.0, 1.0) == {}


def test_a_line_prices_when_its_deltas_cannot_be_converted(markets, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("boom")
    monkeypatch.setattr(greeks, "forward_delta_factor", broken)
    monkeypatch.setattr(pricer, "convention_deltas",
                        lambda *a, **k: greeks.convention_deltas(*a, **k))
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame([line()]))
    results, errors = pricer.price_lines(markets, rows, MODS)
    assert errors == [] and results[0]["delta"] is not None
    assert "pair_delta" not in results[0]
    # The Pricer grid then shows the reported (spot-equivalent) delta.
    shown = _with_line_results(rows.reset_index(drop=True), results)
    assert shown.at[0, "delta"] == f"{results[0]['delta']:,.0f}"


def test_hedges_are_forward_deltas_and_totals_spot_equivalent(markets, priced):
    hedges, errors = pricer.hedge_trades_for(markets, priced)
    assert errors == []
    by_pair = {hedge["currency_pair"]: hedge for hedge in hedges}
    for result in priced:
        hedge = by_pair[result["currency_pair"]]
        # The forward that flattens a line is minus its forward delta; its
        # spot-equivalent offset is minus its spot delta.
        assert hedge["base_amount"] == pytest.approx(-result["fwd_delta"], rel=1e-12)
        assert hedge["fwd_delta"] == hedge["base_amount"]
        assert hedge["spot_delta"] == pytest.approx(-result["spot_delta"], rel=1e-12)
        assert hedge["delta_offset"] == pytest.approx(-result["delta"], rel=1e-12)
    totals = pricer.aggregate_by_pair(priced, hedges, list(greek_catalog.CATALOG))
    # Hedges flatten the smile delta, in every convention (not fixed delta).
    for column in ("delta", "spot_delta", "fwd_delta"):
        assert totals[column].abs().max() < 1e-6, column
    unhedged = pricer.aggregate_by_pair(priced, (), ["delta", "spot_delta", "fwd_delta"])
    assert list(unhedged["delta"]) == pytest.approx(list(unhedged["spot_delta"]))


def test_line_tables_show_the_pair_convention(markets, priced):
    rows, _, _ = pricer.resolve_line_inputs(markets, pd.DataFrame(
        [line(), line("USDJPY", premium_ccy="BASE"), line("USDMXN")]))
    shown = _with_line_results(rows.reset_index(drop=True), priced)
    assert list(shown["delta_convention"]) == ["S", "S", "S"]
    # Each line shows its pair delta (spot up to 1Y for every pair now).
    assert shown.at[2, "delta"] == f"{priced[2]['pair_delta']:,.0f}"
    assert shown.at[2, "fixed_delta"] == f"{priced[2]['fixed_pair_delta']:,.0f}"
    assert shown.at[2, "delta"] == f"{priced[2]['spot_delta']:,.0f}"
    assert shown.at[0, "delta"] == f"{priced[0]['spot_delta']:,.0f}"
    items = pair_convention_rows(priced)
    assert items[2]["delta"] == priced[2]["pair_delta"] and items[0]["delta"] == priced[0]["delta"]
    assert pair_convention_rows([{"delta": 1.0}]) == [{"delta": 1.0}]


def test_spot_and_forward_delta_are_optional_picker_columns():
    assert "spot_delta" in greek_catalog.CATALOG and "fwd_delta" in greek_catalog.CATALOG
    assert "spot_delta" not in greek_catalog.BASE_GREEKS        # not shown by default
    assert "fwd_delta" not in greek_catalog.EXTRA_GREEKS        # no extra pricing
    assert greek_catalog.label("fixed_fwd_delta") == "Forward delta (sticky)"
