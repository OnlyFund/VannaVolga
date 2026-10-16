"""Depo curve tickers and curve construction (no Bloomberg call)."""
import pytest

from helper_functions.depo_tickers import TENORS, depo_tickers

get_spots = pytest.importorskip("get_spots")


def test_tickers_use_root_and_tenor_suffix_unless_special():
    usd = depo_tickers("USD")
    assert list(usd) == list(TENORS)
    assert usd["1W"] == "USOSFR1Z Curncy" and usd["1Y"] == "USOSFR1 Curncy"
    aud = depo_tickers("AUD")
    assert aud["1Y"] == "ADSWAP1Q Curncy" and aud["3M"] == "ADSOC Curncy"
    assert depo_tickers("PLN") == {"1M": "WIBO1M Index", "3M": "WIBO3M Index",
                                   "6M": "PZFR0CF curncy", "9M": "PZFR0FI curncy",
                                   "1Y": "PZFR0I1 curncy"}
    assert depo_tickers("NOK")["2W"] == "NKS2W Curncy"
    assert depo_tickers("DKK") == {}


def test_curve_is_decimal_and_backfilled():
    curve = get_spots.build_depo_curve({"1M": 3.74, "3M": 3.78, "6M": 4.245, "9M": None,
                                        "1Y": 4.775})
    assert list(curve) == list(TENORS)
    assert curve["1W"] == curve["2W"] == pytest.approx(0.0374)
    assert curve["2M"] == pytest.approx(0.0378)
    assert curve["9M"] == pytest.approx(0.04775)
    # Nothing after the last quote to bfill from: those tenors are left out.
    assert list(get_spots.build_depo_curve({"1M": 6.77})) == ["1W", "2W", "1M"]
    assert get_spots.build_depo_curve({}) == {}
