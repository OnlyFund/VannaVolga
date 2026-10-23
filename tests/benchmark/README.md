# Benchmark: our prices against Bloomberg on the same market data

`market52_trades.csv` holds 30 random trades. They are drawn with a fixed seed
by `bloomberg_vols.py` and priced on stored market 52: the Bloomberg snapshot
of Friday 23 Oct 2026 at 12:53:32 London (07:53 New York), FX date 23 Oct.
The pricing uses the app's defaults (model `vv`, betas 0.7 / 0.5 / 0.0) and
each pair's own conventions.

The mix covers:
- majors, crosses and EM pairs;
- pillar tenors and broken dates;
- ATM, delta (10, 15, 25, 35) and fixed strikes (the forward ± up to 6%);
- calls and puts, bought and sold.

## How to check a trade on Bloomberg
1. Price the trade on Bloomberg's option pricing page (OVML) as of the
   snapshot (23 Oct 2026, 12:53 London), with the same pair, call or put, the
   resolved `strike` and `expiry`, and 1m of the base currency.
2. Fill in `bbg_vol`, `bbg_premium_pct` (% of the base notional) and
   `bbg_delta_pct` (spot delta, %). Use the values for one **bought** option,
   whatever `buy_sell` says.
3. Fill in `checked_on`, and `note` for anything unusual: a different delta
   convention, the premium currency, or a date that differs from ours.
4. Run `python tests/benchmark/bloomberg_vols.py --compare`. It lists each
   trade's vol difference in vol points, and the premium and delta
   differences.

Running `python tests/benchmark/bloomberg_vols.py` again re-prices the trades,
for example after a model change, and keeps the `bbg_` columns already filled
in.

Dates are YYYY-MM-DD. Excel rewrites them when it saves the file, so re-save
them in that format.
