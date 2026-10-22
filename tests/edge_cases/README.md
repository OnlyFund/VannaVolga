# Date edge cases: confirmed against the market

The date rules (`apeiron/gaia/market_dates`) are trusted only on cases the
market has confirmed. This directory holds those cases, and
`tests/test_date_edge_cases.py` enforces them.

## The files

- **`option_dates.csv`** has tenor cases: an option on `pair` traded on
  `trade_date` for `tenor`.
- **`typed_expiries.csv`** has typed expiries: an option on `pair` expiring
  on `typed_expiry`.
- **`rollover.csv`** has the FX date of a `moment` recorded in `timezone`
  (17:00 New York rollover); `new_york` shows the moment in New York time, and
  `market_fx_date` takes the market's answer.

Each row holds:
- **`expiry`, `delivery`:** the solver's answer, written when the case was
  proposed;
- **`market_expiry`, `market_delivery`:** the market's answer, filled in by
  hand (a blank field was not read);
- **`status`:** `confirmed` (the market's answer is filled in), `to_confirm`
  or `disputed` (the market and our calendar disagree; say why in `note`);
- **`source`, `checked_on`:** where and when the market was read, such as
  Bloomberg's option pricing page;
- **`priority`:** `1` is the next batch to confirm (one case per pair group and
  category), `2` the rest, and `0` the cases seeded already confirmed;
- **`category`:** why it is an edge case.

The categories:

| Category | Meaning |
|---|---|
| `closed_day_roll` | a day/week expiry lands on a day the pair is closed and moves |
| `expiry_on_holiday` | the expiry is a holiday in one currency (still a valid expiry) |
| `usd_intermediate` | a US-only holiday between expiry and delivery |
| `holiday_in_spot_window` | another holiday between expiry and delivery |
| `year_end` | delivery between 20 Dec and 10 Jan |
| `month_end` | a month tenor from a month-end spot (the end-of-month rule) |
| `rule_changed` | the rules before item 6 gave other dates |
| `one_closed`, `both_closed`, `weekend_after_holiday` | typed expiries on such days |
| `clock_change_gap`, `ordinary_week` | moments around 17:00 New York, with a 4- or 5-hour London gap |
| `friday_evening`, `saturday`, `sunday_evening`, `holiday_eve` | rollover moments around weekends and 25 Dec / 1 Jan |

## The practice

1. **Propose** cases whenever a date rule is added or changed:
   `python -m apeiron.gaia.market_dates.edge_cases`. Rows already in the files
   keep their market answers; new rows come in as `to_confirm`.
2. **Confirm** them on the market, starting with `priority` 1. Fill in
   `market_expiry`, `market_delivery`, `source` and `checked_on`, then set
   `status` to `confirmed`. When the market disagrees, set it to `disputed`
   and note why. The rule is then fixed and the case re-proposed.
3. **The test** holds the solver to every confirmed market answer, and to the
   stored answer on every row. A rule change that moves any case fails until
   the case is re-checked and its stored answer is updated.

Historical settlement dates fetched in bulk from Bloomberg
(`market_data/swap_dates.py`) are a separate reference:
`tests/fixtures/bbg_settlement_dates_20261022.csv`.
