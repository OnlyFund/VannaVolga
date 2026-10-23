# Apeiron architecture state

A record of where the library stands against its level rules, updated at each
migration step. The rules and the module map are in `apeiron/layers.py`;
`tests/test_architecture.py` enforces them, with today's crossings listed in
`apeiron/known_crossings.json` (that list may only shrink).

## Step 0: skeleton and boundary test (2026-10-21)

**Before:** the code had no levels. Modules live at the repository root, in
`pricing/`, `pricing_utils/`, `Greeks/`, `surface_models/`, `helper_functions/`,
`market_data/`, `app/` (services and UI together) and `research/`.

**After:**
- The `apeiron/` skeleton exists, with `atlas/`, `gaia/{time,market,book}/`,
  `hyperion/{cache}/`, `athena/`, `delphi/` and `olympus/`.
- Every module is placed in its level, without moving.
- The rules are checked on every test run, including imports inside functions.

No module moved and no code changed. The full test suite passes (1473: the 1468
before plus 5 architecture tests, 6 with the cycle check below), and the reference book prices identically
before and after.

### Modules per level (115 library modules)

| Level | Modules |
|---|---|
| atlas | 41 |
| gaia.time | 6 |
| gaia.market | 18 |
| gaia.book | 5 |
| hyperion | 23 |
| olympus | 17 |
| delphi | 13 |
| athena | 1 (the empty package) |

### Known crossings: 31

| From → to | Count | What is left to untangle |
|---|---|---|
| atlas → gaia.time | 14 | The core reads pair calendars and delivery dates (`fx_discounting`, `fx_conventions`). This is the Atlas untangling: dates, year fractions and discount factors become inputs. |
| gaia.market → gaia.time | 4 | Market data computes tenor dates and reads pair conventions. Conventions move to market; building a priceable market moves to Hyperion. |
| atlas → gaia.market | 3 | The core reads CSV vol curves (`market_data.vol_data`). |
| atlas → hyperion | 3 | Core modules reach the app's smile cache (`app.smile_service`); the cache hook must be injected, not imported. |
| gaia.market → hyperion | 2 | `market_tables` builds app markets; it moves to Hyperion. |
| gaia.market → atlas | 2 | `usd_spots` uses premium quote helpers; the `market_data` package re-exports the market objects. |
| hyperion → olympus | 1 | The spot ladder imports a UI helper (`app.ui.negatives`); the display part belongs in Olympus. |
| gaia.book → atlas | 1 | `Trade.to_position` builds pricing objects; it moves to Hyperion. |
| gaia.time → atlas | 1 | `fx_discounting` reads smile inputs. |

The full list is `apeiron/known_crossings.json`.

### Known import cycles (all from before Step 0)

Every module imports cleanly in a fresh interpreter, so none of these breaks
loading today. Each is still a knot to untangle. They are frozen in
`apeiron/known_cycles.json`; a new cycle fails the test suite, and this list may
only shrink.

- **At load time (1):** `app.ui` ↔ `app.ui.app`. The UI package's `__init__`
  imports the app page, which imports its sibling modules back through the
  package.
- **Through deferred imports inside functions (3 more):**
  - the smile cluster: `VannaVolga_ImpliedVol`, `app.smile_service`,
    `vanna_volga_helpers`, `resolved_smile_cores`, `smile_preparation`,
    `vanna_volga_sabr`, `wrapped_vol_resolver`. This is the core reaching into the
    app's smile cache;
  - `fx_discounting` ↔ `resolved_smile_inputs`;
  - `discount_curves` ↔ `implied_rates` ↔ `market_tables`.

The Atlas untangling and moving `market_tables` to Hyperion remove most of them.

Step 0 itself adds no cycle: the `apeiron` modules import no library code. The
architecture tests are now 6.

## Result cache in Hyperion (2026-10-21)

The first code placed straight in its final home is `apeiron/hyperion/cache/`:
- **`keys.py`:** a trade's key hashes its terms and version, its pair's market
  (every quote, the spot and any spot edit, the settings, the valuation date),
  the betas, the requested Greeks, the spot path and the code version.
- **`store.py`:** a bounded in-memory LRU over a SQLite file kept 24 hours
  (`data/local/result_cache.sqlite`, or `$VV_RESULT_CACHE`).
- **`pricing.py`:** `price_trades` (same signature and result as
  `app.trades.service.price_trades`) and `verify`.

The Portfolio and the Risk tab price the book through it. It respects the level
rules: Hyperion imports only Hyperion and Gaia. It adds no crossing and no cycle,
so the known lists are unchanged (31 crossings, 4 cycle groups).

## Atlas untangling, sub-step 1: resolved inputs carried on markets (2026-10-21)

**Before:** the core resolves delivery dates, year fractions and pair
conventions itself, through `fx_discounting` and `fx_conventions` (Gaia/time).

**After:**
- **`apeiron/atlas/resolved.py`:** `ResolvedInputs`, an immutable, self-checking
  record of the valuation, expiry and delivery dates, the year fraction and day
  count to delivery, the delta convention, the premium-adjusted flag and the ATM
  convention. Also `simple_discount_factor`, the same arithmetic as today. Rates
  are not part of the record, so bumps stay arithmetic.
- **`MarketData.resolved`:** an optional field, kept by every copy and bump. The
  calibration cache key leaves it out, since its dates are already in the key.
- **`apeiron/hyperion/resolve.py`:** fills it wherever Hyperion builds a market:
  per expiry (today's and theta's next day), per value date, and for the
  calibration pillars. It uses the very functions the core uses today. On any
  failure it stays `None` with a note, and the core resolves for itself as
  before.

Nothing reads it yet, so pricing is unchanged by construction.

**Validation:**
- 20 tests. Every carried value equals the core's own calculation for every
  fixture pair, pillar, broken and short expiry, theta's next-day market and
  value-dated deals.
- On the live market: 65 pairs and 1,782 markets, with 0 unresolved and 0
  mismatched.
- The full suite passes (1506), and the reference book prices identically.
- Crossings and cycles are unchanged (31 / 4). Sub-steps 2 and 3 switch the core
  to these inputs and remove the Atlas → time crossings.

## Atlas untangling, sub-step 2: the core discounts from the carried inputs (2026-10-21)

**Before:** the Greek calculators, pricing, vanna, the FX forward and the rate
bumps called the calendar module (`fx_discounting`) for every discount factor.

**After:**
- **`apeiron/atlas/discounting.py`:** `discount_kwargs` with the same signature
  and output. It computes the discount factor, carry rate and delta convention
  from `market.resolved` and the rate.
- **The carried record also covers:**
  - sticky theta's next weekday (year fraction and delta convention seen from
    that date);
  - the base currency's year fraction, for base-rate risk.
- **The rate bumps (`perturbed_market.bump_rate`)** use the carried year
  fractions.
- **A market without resolved inputs** (a standalone user's, or a request the
  record doesn't cover) gets the legacy `exp(-r·t)`, as the core already did
  without conventions. Such requests are counted (`FALLBACKS`).
- **Unit tests that build markets by hand** now resolve them through Hyperion,
  as the app does. Their expected values are unchanged.

**Validation:**
- Measured before removing the old path: golden and the app tests produce 0
  fallbacks.
- On the live market: 1,784 discount requests (today, theta's date, value-dated
  deals), every one identical to the calendar path.
- The full suite passes (1506). The reference book, with rho and phi, prices
  identically. The result cache equals fresh pricing.

**Crossings: 31 → 25.** These six are gone:
`Greeks.fixed_greeks`, `Greeks.smile_greeks`, `perturbed_market`,
`pricing.fx_forward`, `pricing.pricing` and `pricing.vanna`, each → `fx_discounting`.
Cycles: unchanged (4).

Left for sub-steps 3 and 4: the pair conventions (`pricing.greeks`,
`resolved_smile_inputs`, `VannaVolga_ImpliedVol`), Black-Scholes resolving its
own discount from `currency_pair`, and the date helpers.

## Atlas untangling, sub-step 3: conventions from the carried inputs (2026-10-21)

**Before:**
- The smile built its conventions from the pair tables and calendars
  (`resolved_smile_inputs.resolve_conventions`): delivery, discount factors,
  delta convention, premium adjustment, ATM convention and deposit basis.
- `pricing.greeks` looked up each line's delta convention and its
  premium-adjusted flag in the same tables.

**After:**
- **The carried record also has the pair's forward divisor.**
- **`resolve_conventions` builds from the carried inputs.** Without inputs
  matching the expiry and valuation, it reports the conventions as unavailable,
  and the smile falls back visibly, as it already did for any convention failure.
- **`pricing.greeks`** reads the delta convention and the premium-adjusted flag
  off the carried inputs. Without them it uses the default convention and the
  market's own premium-adjusted setting.
- **Hyperion resolves a market before its smile reads the conventions.**
- **Hand-built test markets** resolve through a test helper
  (`tests/resolution.py`), as the app does.

**Validation:**
- Measured with the old path kept as a counted fallback: golden and the app
  tests produce 0 fallbacks.
- A new test rebuilds the conventions with the old calendar logic. The new path
  equals it exactly for every fixture pair and expiry, today and on theta's date.
- On the live market: 1,650 smile convention sets, all identical.
- The full suite passes (1514). The reference book (105 trades, with rho and phi)
  prices identically. The result cache equals fresh pricing.

**Crossings: 25 → 22.** These three are gone: `pricing.greeks` → `fx_conventions`
and → `fx_discounting`, and `resolved_smile_inputs` → `fx_conventions`.

Left for sub-step 4:
- `VannaVolga_ImpliedVol` → `fx_conventions` (its demo block);
- Black-Scholes resolving its own discount from `currency_pair`;
- the date helpers (`_to_date`);
- `fx_discounting` → `resolved_smile_inputs`.

## Atlas untangling, sub-step 4: the core reads no calendar (2026-10-21)

**Before:**
- `BlackScholes` and the numba engine resolved their own convention discount
  from the calendar when given `currency_pair` and no discount factor.
- `wrapped_vol_resolver` and `resolved_smile_inputs` imported the date parser
  from `fx_discounting`.
- `VannaVolga_ImpliedVol`'s demo looked up EURUSD in the convention tables.
- The old `fx_discounting.discount_kwargs` (now only a test reference) called the
  smile inputs for the delta convention, which made a cycle.

**After:**
- **`currency_pair` is a label only in both pricers.** Convention discounting
  arrives as numbers (`term_discount_factor`, `carry_rate`) from
  `apeiron.atlas.discounting`; without them the pricers use the continuous
  factor, as with no pair.
- **`apeiron.atlas.dates.to_date`** parses dates exactly as the old helper did,
  QuantLib dates included, without importing QuantLib.
- **The demo** carries EURUSD's conventions as plain values. Its last two prints
  (premium and delta) now use the continuous factor, because it passes only a
  pair. Its vol and ATM strike are unchanged.
- **`fx_discounting.option_delta_convention`** gives the legacy reference its
  delta convention directly, the same rule Hyperion resolves.
- **Tests that priced with a pair alone** pass the resolved discount inputs
  (`tests/resolution.discounting`). The pair-resolution tests now check
  Hyperion's resolved inputs instead.

**Validation:**
- A probe on the self-resolving route found it used only by three test files:
  0 calls from the app, the book or production code.
- The full suite passes: 1535 (HEAD 1514, plus 21 new tests).
- Reference book (105 trades, rho and phi, market 38): 5,145 fields identical
  to HEAD.
- Live market 38: 1,784 discount requests and 1,650 smile convention sets,
  all identical, with no fallbacks.
- The result cache equals fresh pricing, and verification is empty.
- New tests check that the date parser equals the old one (every format and
  error case), that the legacy delta convention equals the resolved one, and
  that the core modules import no calendar or convention table.

**Crossings: 22 → 16. Cycle groups: 4 → 3** (`fx_discounting` ↔
`resolved_smile_inputs` is gone). Atlas now reads no calendar at all. The
remaining crossings belong to the Gaia and Hyperion moves (`vol_data`,
`market_tables`, `smile_service` and others).

## Item 6, step 1: the pillar date solver in Gaia time (2026-10-22)

**Before:** the FX date rules were spread over five modules:
- `fx_conventions` (pair calendar, spot date, option dates);
- `market_tables` (O/N pillar);
- `fx_discounting` (the delivery of an expiry);
- `curve_roll` (O/N expiry, spot date, short-end nodes);
- `spot_forwards` (value-date tokens and tenors).

Each built its own QuantLib calendar steps.

**After:**
- **`apeiron/gaia/time/pillars.py`, the pillar date solver.** It covers the spot
  date, next business day, roll back / forward, each tenor's expiry and
  delivery (O/N included), and the delivery of a given expiry. Dates in and
  out are `datetime.date`. A pair arrives as a `Settlement` (base, quote,
  settlement lag), so time reads no pair conventions; `fx_conventions.settlement(pair)`
  builds it.
- **`apeiron/gaia/time/calendars.py`** holds the currency calendar table
  (`fx_conventions.CCY` now takes its calendars from it), date coercion and
  the date records.
- **`fx_conventions` delegates** `joint_calendar`, `spot_date` and
  `option_dates` to the solver, with unchanged QuantLib signatures, and adds
  `pillar_dates`. The other callers listed above now use the solver too.
- **The rules are unchanged.** Later rule changes (week tenors from the
  valuation date, expiries skipping only weekends and 25 Dec / 1 Jan, the
  17:00 New York rollover) happen in this one place.

**Fail-safes:**
- **`apeiron/gaia/time/legacy_dates.py`** keeps the old rules verbatim, as the
  reference.
- **Fallback:** if a solver rule raises where the old rule does not, the old
  answer is used and the case is recorded in `pillars.FALLBACKS`. An error in
  both is raised.
- **Check mode:** with `VV_PILLAR_CHECK=1`, every answer is compared with the
  old rule and any difference is recorded in `pillars.MISMATCHES`. The solver's
  answer is kept. Both logs are bounded.

**Validation:**
- **New tests (`tests/test_pillar_dates.py`):** the solver and every caller
  equal the legacy rules for every pair, on every day from 1 Dec 2027 to 31 Mar
  2028, for 12 tenors. Value-date tokens are checked, and so are the fail-safes.
- **Against HEAD:** every caller's public date function was dumped on HEAD
  and on the new code: 65 pairs, 400 random dates from 2017 to 2032 plus
  Oct 2026 – Feb 2027, all tenors, by-expiry dates, short-end nodes and
  value-date tokens. 1,476,036 values, 0 differences, error messages included.
  Check mode recorded 0 mismatches and 0 fallbacks.
- **Currency table:** identical to HEAD (33 currencies: calendars, day counts,
  ranks, holidays 2017–2035).
- **Reference book:** priced on 10 valuation dates (markets 38 and 41,
  21 Oct – 26 Feb 2027): 26,586 fields identical to HEAD.
- **The app in headless Edge:** HEAD and the new code side by side. The
  Markets, Pricer, Portfolio and Risk grids are identical, with no exceptions.
- **The full suite passes:** 1741.

**Crossings: 16, cycle groups: 3** (unchanged; no new crossing or cycle).

## Item 6, step 2: Bloomberg settlement calendars and the market's spot rule (2026-10-22)

**Reference:** Bloomberg's historical settlement dates, fetched with
`market_data/swap_dates.py` (`SETTLE_DT` with `REFERENCE_DATE` on the spot,
forward and vol tickers). 21 pairs on 34 edge-case dates give 14,178 dates.
Option expiries (`OPT_EXPIRE_DT`) are live only in Bloomberg, so they are not
part of the reference.

**Before:** QuantLib 1.31 calendars and the joint-calendar spot rule.
**13.0%** of the reference dates differed. The causes:
- **US-only holidays:** they were treated as bad intermediate days.
- **Stale or missing QuantLib holidays:**
  - CNH missed 74 days in 2024–26;
  - PHP, COP, MYR and VND were weekends only;
  - IDR, TWD, INR, THB and SGD had gaps;
  - ILS had Friday closures.
- **Week tenors:** they rolled with Modified Following.

**After:**
- **`market_data/settlement_holidays.py`** fetches each currency's Bloomberg
  calendar code (`CALENDAR_CODE`) and non-settlement dates
  (`CALENDAR_NON_SETTLEMENT_DATES`) into the static, tracked list
  `market_data/calendars/settlement_holidays.csv`. The list covers 2005 to 5
  years ahead and is refreshed by hand. ARS and MYR end in 2030, THB in 2027,
  VND in 2026.
- **`apeiron/gaia/time/calendars.py`** builds every settlement calendar from
  that list. Outside a currency's coverage it uses QuantLib's holidays, and
  without the list it uses QuantLib's calendars whole (`SOURCE` says which).
  The old table stays as `QL_CALENDARS`.
- **The solver's rules (`pillars.py`):**
  - **Spot:** the settlement lag is counted on the non-USD calendar(s), so a
    US-only holiday is a good intermediate day for USD pairs and crosses. The
    spot day itself must be good in USD. MXN and BRL are excepted
    (`USD_BLOCKS_INTERMEDIATE`). ILS counts the lag on weekdays, so its own
    holidays don't block the intermediate day either (`LAG_ON_WEEKDAYS`,
    matching Bloomberg's USDILS spot on 708 of 708 days, 2015 to 2026).
  - **Tenors:** day and week tenors roll Following; month and year tenors
    roll Modified Following with the end-of-month rule.
  - **Expiry to delivery:** an expiry delivers on its spot date.
  - **Expiry days:** a weekday on which both currencies are closed moves to
    the next good day, even where Bloomberg expires on it (a house rule).
    Expiry overrides (`market_data/calendars/expiry_overrides.csv`, or
    `set_expiry_override` on the fly) mark a pair and day as an expiry day or
    not.
  - Tenor expiries (delivery less the lag) are unchanged: their rules are a
    later step.
- **Fail-safes:**
  - `legacy_dates` keeps the old rules on `QL_CALENDARS`;
  - check mode records exactly what the new rules move;
  - `UNCOVERED` records dates outside the Bloomberg coverage.

**Validation:**
- **Against Bloomberg:** **0 of 14,178** reference dates differ. The fixture
  `tests/fixtures/bbg_settlement_dates_20261022.csv` is a tracked test.
- **Reference book** on 10 valuation dates: 10 trades move. Seven expire on
  Fri 15 Jan 2027, so their delivery moves from 20 to 19 Jan (MLK is a good
  intermediate day). AUDUSD and EURAUD's 6M pillar moves from 26 to 27 Apr
  2027 (Bloomberg's AUD calendar has the ANZAC Day substitute). The largest
  MTM change is 1.3k on −683k. No fallbacks.
- **Golden, regenerated and reviewed:** only EURUSD and USDJPY move. Line 1
  expires on 15 Jan 2027, which is also the golden market's 3M pillar expiry,
  and line 7 is interpolated from that pillar (about 1e-8). Their hedges,
  totals and spot ladders move with them. Check mode traced every change to
  `delivery_for_expiry(2027-01-15)` and theta's roll of it (`2027-01-18`). The
  pillar's delivery now agrees with its tenor delivery (19 Jan); before, they
  disagreed. USDMXN golden is unchanged.
- **Two tests re-dated** because their premise followed the old spot rule: the
  USDZAR roll test (Columbus Day is now a good intermediate day) and the
  value-dated forward that delivers with the 15 Jan expiry.
- **Caches:** `SURFACE_CACHE_VERSION` 5 → 6 and the result `CACHE_VERSION`
  1 → 2, so nothing built on the old dates is reused.

## Gaia's time level renamed `market_dates` (2026-10-22)

`apeiron/gaia/time/` is now `apeiron/gaia/market_dates/` (level
`gaia.market_dates`). It is the home of every date rule: settlement calendars,
pillar and expiry dates, and later the expiry cut, the 17:00 New York rollover
and the variance days. Earlier sections keep the name they had at the time.
Crossings and cycles are unchanged.

## Item 6, step 3: option expiry rules (2026-10-22)

**Reference:** Bloomberg's option pricing page, checked by hand (historical
expiries are not available through the API).

**Before:** a tenor's expiry was its delivery less the settlement lag on the
full calendar (Preceding), week tenors included. A typed expiry rolled back to
a good business day, and forwards and options shared the same tenor delivery.

**After (`market_dates/pillars.py`):**
- **Expiry days:** a weekday on which at least one of the pair's two
  currencies is open. US-only, Japanese or UK holidays are valid expiries.
  25 Dec and 1 Jan are not for EURUSD (both currencies closed), but 25 Dec is
  for USDJPY (Tokyo is open).
- **O/N, day and week tenors:** expiry = trade date + tenor, moved to the next
  expiry day; delivery = the expiry's spot date.
- **Month and year tenors:** delivery = the forward delivery; expiry = the
  latest expiry day whose spot date is that delivery (`expiry_for_delivery`).
  From 2018 to Oct 2026, every pair and month tenor has one.
- **Typed expiries:** a weekend moves to the next weekday, and a weekday is
  taken as given (`typed_expiry`).
- **Forwards** keep spot + tenor through a separate `forward_delivery`. The
  forward table, implied rates and the sample market use it; option pillars
  use `tenor_dates`.

**Validation:**
- 26 tenor cases and 9 typed year-end cases on the pricing page, plus the
  USDJPY 3M whose delivery (29 Dec 2026) only a 25 Dec expiry reaches. All
  match, and all are tests now.
- The 14,178 historical Bloomberg dates still match. Bloomberg's vol tickers'
  `SETTLE_DT` for week tenors follows the forward convention, so those rows
  compare with `forward_delivery`.
- A property test over every pair checks each option expiry: it is an expiry
  day, its spot date is its delivery, and month and year expiries are the
  latest such day.
- The reference book on 10 valuation dates is unchanged, and so is golden.
- The full suite passes. Two tests now build their forward points on forward
  deliveries, the convention `implied_rates` inverts.

## Date edge cases confirmed against the market (2026-10-22)

`tests/edge_cases/` holds the option date edge cases that the market confirms
(`option_dates.csv` for tenors, `typed_expiries.csv` for typed expiries), with
a README describing the practice. `apeiron/gaia/market_dates/edge_cases.py`
proposes cases by category, a few per pair; rerunning it keeps the market
answers already filled in. `tests/test_date_edge_cases.py` enforces them.

The first set is 347 tenor and 143 typed cases. 36 are confirmed (the pricing
page checks above). 58 are marked `priority` 1 as the next batch to confirm.

## Item 6, step 3d: the 17:00 New York FX date rollover (2026-10-22)

- **`apeiron/gaia/market_dates/rollover.py`:**
  - `fx_date(moment, timezone)` converts the moment to New York time. From
    17:00 it is the next day, and a weekend becomes Monday. Holidays do not
    move the FX date: the spot and expiry rules apply to it.
  - `cutoff(day)` is the last moment of an FX day, in a given zone.
  - `fx_today()` is today's FX date.
- **The market store, schema 5:** each market records its `timezone`. The
  migration takes a backup first and runs all or nothing; earlier markets get
  `Europe/London`. The user's live store migrated on their own fetch at 09:54
  (backup `market_data.v4-backup-20261022_095449.sqlite`). Markets 1–43 are
  identical to the backup, and market 44 is recorded in `Europe/London`.
- **The market store, schema 6: the day's live pillar dates.**
  `market_data.live_dates` (Hyperion) fetches Bloomberg's dates for today
  with `bdp`:
  - the expiry from `OPT_EXPIRE_DT` on the ATM vol tickers;
  - the month and year delivery from `SETTLE_DT` on the outright forwards;
  - for ON and week tenors, delivery on the expiry's spot date.

  Each date is checked against the solver, and anything doubtful falls back
  with its reason. The fetch is refused outside New York's session, or when
  over 30% of tenors disagree. `--store` keeps every fetch: `date_fetches`,
  `pillar_dates` (with the solver's answer beside each) and
  `pillar_date_fallbacks`. `load_pillar_dates(fx_date)` reads the latest
  usable fetch, and `pillar_date_disputes()` lists where it differs from the
  solver.

  The migration only adds the tables, with a backup first. The user's store
  migrated on 23 Oct 2026 at 08:44 (backup
  `market_data.v5-backup-20261023_084458.sqlite`). Its 49 markets and event
  data are identical to before. The first fetch stored 585 dates and 40
  disputes. Nothing uses the stored dates yet (step 3).
- **The store keeps data only, so Gaia market imports no date rule.** Hyperion's
  loader (`app.store_markets`) picks the latest snapshot taken before 17:00
  New York on the valuation date, and attaches its FX date
  (`Snapshot.fx_date`). `Snapshot.date` is then the FX date.
- **The market picker** (Olympus) lists and links snapshots by FX date
  (`fx_date` column; `snapshot_filter` uses it when present), and "today" is
  the FX date.
- **Left for the Hyperion move:** `app.usd_spots` (Gaia market) still picks
  its snapshot by local date.
- **Edge cases:** `tests/edge_cases/rollover.csv` has 50 moments: London's
  4-hour clock-change gaps in 2026–27, ordinary weeks, Friday and Sunday
  evenings, Saturdays, and the evenings of 24 and 31 Dec. All are to be
  confirmed, 6 of them priority 1.
- **Validation:** the reference book on 10 valuation dates is unchanged
  (no stored snapshot is after 17:00 New York). The app runs in headless Edge
  on a migrated store copy with no exceptions.

## The trade store in memory (online app) (2026-10-22)

- **The online error:** "trade store is read-only: ... table schema_info
  already exists". Two sessions opened a new store at once: both saw an empty
  file, and the second failed creating the schema, so the store fell back to
  read-only. A schema script that fails now re-checks the version and
  carries on if another session just made it current.
- **In memory:** `SqliteTradeStore(MEMORY)` (`":memory:"`) is the same store
  on an in-memory SQLite database, with the same schema, triggers and checks,
  and no backups. The app uses it on Streamlit Cloud (when it runs from
  `/mount/src/`) or with `VV_TRADE_STORE=memory`. Booking works, the book
  lasts for the browser session, and a notice says it is not saved. Locally,
  an unreadable store still never becomes an empty book.
