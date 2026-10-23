"""The Bloomberg benchmark's trades are fixed by their seed (tests/benchmark)."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "benchmark" / "bloomberg_vols.py"


def _module():
    spec = importlib.util.spec_from_file_location("bloomberg_vols", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_trades_are_the_same_every_run_and_match_the_file():
    module = _module()
    trades = module.generate()
    assert trades == module.generate() and len(trades) == module.COUNT
    stored = module.read()
    assert [row["case_id"] for row in stored] == [trade["case_id"] for trade in trades]
    for row, trade in zip(stored, trades):
        assert {key: row[key] for key in trade} == {key: str(value) for key, value in trade.items()}


def test_the_compare_sets_a_sold_option_on_bloombergs_footing(tmp_path):
    module = _module()
    path = tmp_path / "trades.csv"
    row = {column: "" for column in module.COLUMNS}
    row.update(case_id="B99", pair="EURUSD", buy_sell="SELL", strike_input="atm", expiry_input="1M",
               vol="7.0", premium_pct="-1.20", spot_delta_pct="-50.0",
               bbg_vol="7.1", bbg_premium_pct="1.25", bbg_delta_pct="50.5")
    module.write([row], path)
    (case, *_, vol, premium, delta), = module.compare(path, out=lambda line: None)
    assert case == "B99" and round(vol, 6) == -0.1
    assert round(premium, 6) == -0.05 and round(delta, 6) == -0.5
