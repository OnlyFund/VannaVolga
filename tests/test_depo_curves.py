"""Depo curve tickers and curve construction (no Bloomberg call)."""
import pytest

from market_data.depo_tickers import TENORS, depo_tickers

get_spots = pytest.importorskip("market_data.bloomberg_fetch")


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


def test_curve_keeps_the_quoted_tenors_as_decimals():
    curve = get_spots.build_depo_curve({"1M": 3.74, "3M": 3.78, "6M": 4.245, "9M": None,
                                        "1Y": 4.775})
    assert curve == {"1M": 0.0374, "3M": 0.0378, "6M": 0.04245, "1Y": 0.04775}
    assert list(get_spots.build_depo_curve({"ON": 6.5, "1M": 6.77})) == ["ON", "1M"]
    assert get_spots.build_depo_curve({}) == {}


def test_vol_universe_and_surface():
    assert len(get_spots.G10_CROSSES) == 45 and len(set(get_spots.G10_CROSSES)) == 45
    assert len(get_spots.FX_UNIVERSE) == 65
    assert get_spots.vol_tickers[0] == "EURGBPVON Curncy"
    full = {(param, tenor): 1.0 for param in ("V", "25R", "10R", "25B", "10B")
            for tenor in ("1W", "1M")}
    del full[("10B", "1W")]
    surface = get_spots.build_vol_surface(full)
    assert list(surface) == ["1M"]          # 1W is missing its 10B quote: left out
    assert surface["1M"] == {"atm": 1.0, "rr_25": 1.0, "rr_10": 1.0, "bf_25": 1.0, "bf_10": 1.0}


def test_cnh_uses_cny_tickers_and_mxn_has_an_overnight_point():
    assert depo_tickers("CNH") == depo_tickers("CNY")
    assert depo_tickers("CNH")["1M"] == "CCSOOA Curncy"
    assert depo_tickers("MXN") == {"ON": "MXONBR Index", "1M": "MXIBTIIE Index"}


def test_implied_rates_invert_covered_interest_parity():
    from helper_functions.fx_conventions import spot_date
    from market_data.implied_rates import implied_rates
    from market_data.market_tables import _ql_date, tenor_dates
    spot, valuation = 18.4, "2026-10-16"
    usd = {tenor: 0.04 for tenor in TENORS}
    points = {}
    for tenor in TENORS:
        delivery = _ql_date(tenor_dates("USDMXN", valuation, tenor)[1])
        days = delivery - spot_date("USDMXN", _ql_date(valuation))
        forward = spot * (1 + 0.07 * days / 360) / (1 + 0.04 * days / 360)
        points[tenor] = (forward - spot) * 10000
    assert implied_rates("MXN", "USDMXN", valuation, spot, points, usd) == pytest.approx(
        {tenor: 0.07 for tenor in TENORS})
    assert implied_rates("USD", "USDMXN", valuation, spot, points,
                         {tenor: 0.07 for tenor in TENORS})["3M"] == pytest.approx(0.04)
