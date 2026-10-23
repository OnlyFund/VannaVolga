# FXO Risk Analyzer roadmap

Last updated 2026-10-23. Decisions are recorded with the item they belong to.
Item numbers are stable references; the order of work is in **Priorities**.

## Design principles

- **Streamlit stays the UI**, run locally or on a simple host. No large service
  infrastructure; "services" means the engine is callable as plain Python functions.
- **Computation lives outside the GUI.** Pricing, Greeks, ladders, risk grids,
  portfolio valuation and caching are service modules without Streamlit; the UI
  only calls them and shows the results.
- **The UI validates pricing.** The services will later drive a strategy engine,
  a backtester and a portfolio optimizer.
- **Stored data is append-only and versioned** (trades, event calendars).
  Migrations take a backup first and run all or nothing; newer schemas are refused.
- **Date rules are confirmed against the market.** Every date rule change comes
  with edge cases in `tests/edge_cases/*.csv`, proposed by
  `python -m apeiron.gaia.market_dates.edge_cases` and confirmed by hand on
  Bloomberg's option pricing page (`priority` 1 first). The tests hold the
  solver to every confirmed answer.
- **A stored snapshot is static:** its valuation of a book is deterministic, so it
  is computed once and cached.

## Library architecture: Apeiron

The suite is **Apeiron** ("the boundless"), an FX option and volatility analytics
suite with research, pricing, risk and strategy outputs. It is one package with
six levels:

```
apeiron/
  atlas/      core numerical engine
  gaia/       data foundation: market_dates/, market/, book/
  hyperion/   engine room: analytics/, risk/, services/, cache/
  athena/     decision layer: strategy/, optimizer/, backtest/
  delphi/     research
  olympus/    user interface (Streamlit)
```

- **Atlas** (core) imports nothing else in the suite. It is pure numerics, usable
  by anyone: dates, year fractions, discount factors and conventions are inputs.
- **Gaia**'s three parts are each independent of one another and of Atlas:
  - **market_dates:** settlement calendars, holidays, pillar and expiry dates,
    cuts, rollover, variance days;
  - **market:** fetch, snapshot store, event calendars, discount curves and the
    static pair conventions;
  - **book:** a database of trades and lifecycle events.
- **Hyperion** combines Atlas and Gaia:
  - a priceable market from a snapshot and its dates;
  - pricing objects from trades;
  - lifecycle logic, ladders, grids and attribution;
  - the services entry point and the result cache.
- **Athena**, **Delphi** and **Olympus** consume the levels below them; none
  imports another. Whether they go only through Hyperion's services is decided
  when they are built.
### Migration approach

1. **Step 0:** create the `apeiron/` skeleton and an import-boundary test that
   enforces the rules above. Today's crossings are listed as known exceptions, and
   the list may only shrink.
2. **New work goes straight to its final home:**
   - the result cache → `hyperion/cache`;
   - the intraday cut and the pillar date solver → `gaia/market_dates`;
   - lifecycle logic → `hyperion`, and its event records → `gaia/book`.

   Until the old modules move, these may call them through marked transitional
   imports.
3. **Existing code moves one level at a time, between features:** Atlas first
   (remove the core's calendar dependency), then Gaia, Hyperion and Olympus. Old
   import paths keep working through re-exports, and pricing stays identical at
   every move.

## Done (recent)

- Spot and forward booking in the Pricer: value-dated deals priced at their own
  delivery date, ON/TN/SN points around spot, base or term amounts.
- No trade leaks to earlier valuation dates (the trade date is the booking date).
- Rate Greeks: rho in term-currency units, phi in base-currency units; never in
  USD, never summed across pairs.
- Black-Scholes (sticky strike) Greeks, the Bloomberg screen's: BS delta, BS delta
  incl. premium, BS gamma. Three spot regimes named: sticky delta (betas 0/0/0),
  smile (the Model tab's betas), sticky strike (close to betas 1/1/1, not beta ATM
  1 alone).
- Progress bars for the heavy blocks, through plain callbacks.
- Event calendar: named events, per-pair append-only versions with their save
  date. A market uses the pair's latest version saved on or before the valuation
  date, else its pinned event set, else none. CSV import and export.
- A vol Greek that cannot be computed is shown as unavailable instead of failing
  the line.
- A Refresh button in the market picker; collapsible Blotter amend/cancel.
- **Apeiron step 0:** the package skeleton, a level map of every module, and an
  import-boundary test (crossings and cycles may only shrink); its state is
  recorded step by step in `docs/ARCHITECTURE_STATE.md`.
- **Result cache** in `hyperion/cache`: each trade priced once per market state
  (in-memory LRU plus a 24-hour on-disk store, with a verify mode). Portfolio and
  Risk use it.
- **Atlas untangling** (four steps): markets carry their resolved dates and
  conventions, filled by Hyperion; the core discounts and reads conventions from
  them, and reads no calendar or convention table at all. Crossings 31 → 16.
- The Spot / forward grid has a Remove column.
- **Pillar date solver, step 1** (`apeiron/gaia/time/pillars.py`): every FX date
  rule in one place, the rules unchanged, with the old rules kept as a checked
  reference and fallback.
- **Pillar date solver, step 2:** Bloomberg settlement calendars (a static list from
  CDR, 2005 to 5 years ahead) and the market's spot rule (US-only holidays are good
  intermediate days, MXN/BRL excepted); week tenors roll Following. Matches all
  14,178 Bloomberg reference dates (21 pairs, 34 edge-case dates).
- **Pillar date solver, step 3:** option expiries. Expiry days are weekdays when
  either of the pair's currencies is open; day/week tenors expire on trade date +
  tenor; month/year tenors on the latest expiry day whose spot is the delivery;
  forwards keep spot + tenor. Checked on Bloomberg's pricing page (36 cases).
  Gaia's date level is now `market_dates`.
- **Pillar date solver, step 3d:** the FX date rolls at 17:00 New York (DST-aware):
  snapshots record their time zone (market store schema 5), the loader and the
  market picker date snapshots by FX date, and "today" is the FX date. Rollover
  edge cases in `tests/edge_cases/rollover.csv`.
- **Online booking:** on Streamlit Cloud the trade store is in memory (the session's
  book, not saved), so booking works without a trade database; two sessions
  opening a new store at once no longer leave it read-only.
- **Edge cases confirmed on Bloomberg** (2026-10-22/23): the tenor cases are
  bulk-checked (`market_data.bloomberg_edge_cases`) and stand at 308
  confirmed and 52 house rule, with none disputed. The house rule is that a
  weekday when both currencies are closed is no expiry day. Expiry overrides
  (`market_data/calendars/expiry_overrides.csv`, or
  `pillars.set_expiry_override` on the fly) change it for a pair and day.
- **Spot rules:** USDILS counts its spot lag on weekdays (Bloomberg, 708 of
  708 days). A cross takes the longer spot lag of its USD legs, so EURTRY
  and EURPHP are T+2.
- **Live pillar dates, steps 1–3** (`market_data.live_dates`): once a day,
  before 17:00 New York, Bloomberg's dates for today are fetched with `bdp`.
  - ON and week tenors take the expiry from `OPT_EXPIRE_DT` and deliver on
    its spot date.
  - Month and year tenors deliver on the outright forward's `SETTLE_DT`,
    with the expiry derived from that date. `OPT_EXPIRE_DT` is then a
    cross-check only. NZDUSD 1M (26 Nov → 30 Nov) is verified against the
    pricing page and bank pricing.

  Each date is checked against the solver, with fail-safes, and stored with
  the solver's answer beside it (market store schema 7). The daily
  `bloomberg_fetch` run stores the dates too. The day's markets use them, and
  a tenor booked that day takes them; expiry overrides win, and the solver
  is the fallback.

- **The rolled market** (`app.rolled_market`, 2026-10-23): a market valued on
  a later date is rolled to it, not reused with its old dates.
  - Pillars move to the new date's tenor dates, and today's pillar dates
    stay as interpolation nodes.
  - Each smile vol keeps today's forward variance everywhere.
  - Forward points are sticky tenor, and passed events drop out.
  - Only the pairs priced are rolled, and rolled markets are cached.

  **Theta is the date roll:** it is priced on the market rolled to the next
  business day, exactly (`tests/test_theta_date_roll.py`). Validated on the
  store's 22 Oct market rolled to 23 Oct (65 pairs): the dates match the
  actual 23 Oct market exactly.

## Priorities (2026-10-23)

**P1, now: the performance foundation, then finish item 6 (dates).**
1. Item 14, performance steps 0–2: a benchmark, profiling and timing budgets;
   quick wins; pre-warming after the daily fetch. Accuracy (rolling,
   calibration, theta on the rolled market) now costs real time, and the app
   stalled once on it (2026-10-23). The first profile on 3.7 is the baseline.
2. Item 14, performance step 3: Python 3.7 → 3.11, right after the baseline,
   so the optimising is done on the version that runs (and the same one as
   Streamlit Cloud).
3. Test-suite speed: tests run in parallel (`pytest-xdist`), a quick tier per
   change and the full suite before commits, calibrated fixtures shared. The
   full run takes 9–11 minutes.
4. Item 6, live pillar dates step 4: disputes with the solver go to
   `tests/edge_cases` for review, with a command. Small; it keeps the
   market-confirmed date practice going.
5. Item 6, trades store their delivery, and the delivery override on the
   pricing page. Booked month tenors deliver on the forward settlement date.
   Medium; a trade store schema change.
6. Item 6, wide validation: random past and future dates against Bloomberg's
   historical dates, plus the offline rateslib check (step 5).
7. A scheduled daily fetch (a Windows scheduled task before 17:00 New York),
   and a note in the app when today's live dates are missing. Live pillar
   dates exist only if the fetch runs; otherwise the solver is used silently.

**P2, next: the gate to the strategy stages.**
8. Continuous integration: the quick test tier on every push (GitHub
    Actions); later the place to build Rust wheels.
9. The date snapping back: with "Link to market data date" on, a future date
    returns to the market's date; say so in one line.
10. Item 4, the intraday cut (paused until item 6 is done).
11. Item 5, Apeiron: Gaia's move (after item 6), then Hyperion's (before item 8).
12. Item 8, the full option lifecycle: the gate for stages 4–6.
13. The pricing review backlog (VV-02 … VV-10), taken alongside items 8 and 15;
   VV-10 (reproducibility) goes with item 8.
14. Item 7, market data deployment and cross forward points.

**P3, later: features, models and UI.**
15. Items 9–13: vol compare, P&L attribution, surface analytics,
    cross-consistent spot moves, term-currency notional.
16. Item 14, performance steps 4–6: a process pool per pair; Numba and Rust
    kernels; fast and full accuracy modes.
17. Items 14a, 14b and 15: pure SABR, local vol (research), smile study.
18. Items 16–21: UI.
19. **Later**, as listed at the end.

## Paused

4. **Intraday time to the expiry cut** (New York / Tokyo; paused), built in `gaia/market_dates`.
   Decided:
   - the cuts are **TOK = 15:00 Tokyo** and **NYK = 10:00 New York** (clock
     changes included);
   - within **one week** of expiry, time to expiry is measured in **minutes** from
     the valuation moment to the cut; beyond one week, in whole days as today;
   - on the expiry day, once the cut has passed, the option is expired. A message
     shows when an expired option has not yet been exercised or lapsed in the book
     (the lifecycle, item 8, treats it).

   Open (to answer when resuming):
   - which pairs cut at TOK (proposal: USDJPY, the JPY crosses, KRW, TWD, CNH,
     IDR and other Asian pairs);
   - the valuation moment (proposal: the snapshot's time);
   - how minutes map to variance time (proposal: uniform within a day, times that
     day's variance-day weight);
   - whether theta rolls to the same clock time on the next business day
     (proposal: yes).

   Built with:
   - special cuts and fixings per pair (MXN, BRL, KRW, CEEMEA): cut time, fixing
     source and fixing date;
   - the valuation time of day (snapshot time vs valuation moment);
   - **sub-1-week options** (expiries before the first quoted pillar): Greeks,
     especially gamma, delta and theta, switch from whole days to the actual time
     to expiry. Mid priority; uses the same time machinery as the cut.

## Dates and data foundation

5. **Apeiron migration.** Atlas is done: it reads no calendar (crossings 31 → 16,
   cycle groups 4 → 3). The remaining levels move one at a time between features,
   with pricing identical at each move (state in `docs/ARCHITECTURE_STATE.md`):
   - **Gaia:** CSV vol curves (`vol_data`) out of the core and the smile code;
     pair conventions into market; tenor dates out of `market_tables`;
   - **Hyperion:** `market_tables` builds priceable markets, so it moves here;
     the smile cache hook (`smile_service`) is injected instead of imported by the
     core; `Trade.to_position` leaves the book;
   - **Olympus:** the spot ladder's display helper (`ui.negatives`) moves to the UI;
     the `app.ui` ↔ `app.ui.app` load-time cycle is removed.

   Suggested slots: Gaia after item 6, Hyperion before item 8, Olympus with item 17.
6. **Pillar date solver** (in `gaia/market_dates`; the solver, live dates and the
   rolled market are done, the rest is P1): one date service for O/N, days,
   weeks, months and years from any valuation date.
   - Delivery is spot plus tenor with **Modified Following** (a weekend or holiday
     in either currency moves to the next good day, back to the previous one if
     that crosses the month) and the **end-of-month rule** (spot on the month's
     last business day gives the target month's last business day).
   - Week tenors count from the valuation date. Expiries skip weekends and
     any weekday when both currencies are closed (a house rule, 2026-10-22;
     expiry overrides change it for a pair and day). Each currency settles on
     its own holiday calendar.
   - **One rolled market** for any later valuation date, used by both theta (the
     next business day) and valuing a snapshot on a later date. A test checks
     that they reconcile exactly. It is built in four steps:
     1. pillar dates are this solver's tenor dates for the new date;
     2. each pillar's vols are the forward vols from today's surface between the
        new date and the pillar, and the smile is calibrated on them;
     3. forward points stay per pillar, so a fixed-delivery forward slides down
        the curve;
     4. the portfolio is priced on that market.

     Rolling several days uses the forward surface to that date directly.

     **Built and validated (2026-10-23)**, theta included (see Done). The
     fail-safes:
     - a last pillar rolled past today's final quote extends the last
       segment per variance day;
     - a pillar whose forward variance can't be formed keeps its tenor smile;
     - forward points without tenors move by the days rolled;
     - a pair that can't roll is valued on its own dates, with the reason in
       the note;
     - `VV_ROLL_MARKET=0` turns rolling off, and `theta_vol_roll` = `forward`
       or `tenor` gives the earlier thetas for comparison.
   - The conventions:
     - forward points are **sticky tenor** (spot and rates unchanged, so
       fixed-delivery forwards roll down the curve);
     - vols keep **forward variance**: only the passing day's own variance drops
       out, so event days and weekends are handled correctly under variance-day
       weighting.

     Sticky-tenor vols remain available only as a comparison.
   - The trade date rolls at **17:00 New York**.
   - Answered: the USD intermediate-day rule (and MXN/BRL, ILS), T+1 pairs and
     crosses' lags, expiry for a delivery date, holiday-file gaps (Bloomberg's
     calendars). Still open: whether options use the SN point.
   - **Live pillar dates, still to do:**
     - **Step 4:** disputes between the live dates and the solver
       (`store.pillar_date_disputes`, `pillar_expiry_mismatches`) go to
       `tests/edge_cases` for review, with a command.
     - **Step 5:** an offline rateslib check, for personal use, in a separate
       Python 3.10 environment. It compares spot and tenor dates with ours.
   - **Trades store their delivery** (decided 2026-10-23): booking records the
     expiry and the delivery. A month tenor delivers on the forward settlement
     date; ON and week tenors deliver on the expiry's spot date. Pricing uses
     the stored delivery, and older trades derive it from their expiry. This
     is a trade store schema change (3 → 4), with a backup.
   - **Delivery override on the pricing page:** the solved delivery can be
     overwritten for edge cases and disputes with the market. The override is
     marked on the line, refused when it isn't after the expiry or isn't a
     business day for the pair, and stored on the booked trade.
   - **Validation:** random past and future dates and edge cases (month ends,
     holidays in either currency and USD-only, weekends, 25 Dec / 1 Jan, 28/29
     February and leap years, year ends, T+1 pairs, clock-change weeks) against
     Bloomberg historical settlement dates and an independent library, with a
     per-pair, per-tenor report of every difference.
7. Market data deployment; cross forward points triangulated from the USD legs
   when not quoted.

## Features

8. **Full FX option lifecycle** (gate for the strategy, backtest and optimizer stages):
   - **Expiry and exercise:** in-the-money options become deliveries, out-of-the-money
     ones lapse. Non-deliverable forwards and options are **cash settled** in USD
     against the fixing.
   - **Expiry report:** options expiring per date by trade and pair, with strike,
     distance from the market and in/out-of-the-money status. **Exercise** turns an
     option into a spot trade that replaces it (value date = delivery, rate = strike,
     trade date = expiry); **Expire** lapses it. The option is closed by a linked,
     recorded event, never deleted. Plain functions; the page only calls them.
   - **Cash ladder** of flows by currency and value date.
   - **Premium** paid at spot or deferred to delivery, each on its own value date.
   - **Rolls:** every date settles into USD; amounts are rolled with T/N and S/N
     swaps, not delivered. USD term (EURUSD): roll the base amount. USD base
     (USDJPY): roll the term amount. Crosses (EURJPY): roll the base with the
     cross's own points; the term residual goes to a USD book (USDJPY) and is rolled
     there. No spot conversion; only USD is realised. Open: hedging and netting of
     the rolled residual.
   - **FX swaps** booked as one ticket, split into two linked forward legs.
   - Amendments and cancels with effective dates, so past dates replay exactly.
9. Vol compare between two snapshots.
10. P&L attribution between snapshots (spot, vol, rates/points, time, residual).
11. Surface analytics: the surface by delta and tenor, forward vols, variance-day
   adjusted vols, term-structure and smile plots.
12. **Cross-consistent spot moves** in the Portfolio: spots stay triangulated
    (EURZAR = EURUSD x USDZAR). Moving a cross moves the USD pair of the
    lower-ranked currency in the FX hierarchy (EUR > GBP > AUD > NZD > USD > CAD >
    CHF > JPY > EM): EURZAR moves USDZAR, EURGBP moves GBPUSD. Several edits apply
    in the order made; moved pairs reprice along the smile; every move is listed.
13. Option notional in the term currency at booking.

## Performance and models

14. **Performance plan** (discussed 2026-10-23, to review when this item
    starts). Accuracy now costs real time:
    - rolling a pair took about 1.9 s, mostly smile preparation at each pillar;
    - theta needs a second, next-day market for every pair priced;
    - the rolled market's interpolation nodes double one rolled curve's rows;
    - the 203-trade book prices in 26 s cold, 0.2 s per extra line on a
      prepared pair, and 3.5 s on a cached rerun.

    The steps, in order:
    0. **Measure first.** A fixed benchmark: the 203-trade book, a 65-pair
       roll, a spot ladder and theta. A profile (cProfile or py-spy), and
       timing budgets as tests, so a change that makes the app stall fails a
       test first.
    1. **Avoid work.**
       - Prepare a curve once per market: parse the vol table once, and keep
         the converted curve and the prepared smiles.
       - Theta, decays and extra Greeks only when shown.
       - A per-line results cache keyed by (trade, market fingerprint,
         settings).
       - No pandas in hot loops.
       - Batch by (pair, expiry): one smile for every strike, and Greek
         bumps for all lines at once with numpy.
       - The spot ladder across all spots at once.
    2. **Pre-warm after the daily fetch.** A background job calibrates every
       pair's surface and its next-day rolled surface into the disk cache, so
       the app opens warm. Cache keys are content hashes, with a version bump
       whenever a model changes (as `SURFACE_CACHE_VERSION`).
    3. **Python 3.7 → 3.11/3.12.** 1.25–1.6× faster for pure Python, the same
       version as Streamlit Cloud, and it unlocks current Numba, PyO3 and
       rateslib. The golden files guard the numbers.
    4. **Parallel, per pair.**
       - A persistent process pool, since Windows starts processes slowly
         and Python's lock stops threads speeding up pure-Python code.
       - Each worker gets its market data passed in, grouped per pair.
       - Threads become useful once the hot code releases that lock (numpy,
         Numba with `nogil`, Rust).
    5. **Lower-level kernels, only for profiled hot spots.**
       - Numba first (already used: `black_scholes_numba`, `sabr_numba`): the
         VV smile, the delta-to-strike solve, implied vol, variance
         interpolation.
       - Then Rust (PyO3 and maturin) where Numba isn't enough, ideally a
         vectorised pricer over many strikes and expiries. It needs wheels
         for Windows and Linux (Streamlit Cloud), built in CI.
       - Every kernel keeps the Python implementation as its fallback and a
         parity check, held to the golden files.
    6. **Accuracy per use.**
       - A fast mode for interaction, and a full mode for end of day and
         reports, as the spot ladder's "optimized" mode already does.
       - For example, interactive theta from today's calibrated smiles with
         forward variance applied directly, and a full recalibration of the
         rolled market at end of day.
       - Every approximation is checked against the full version, with a
         stated tolerance.
       - Calibrations warm-start from the previous day's parameters.

    Longer term: Hyperion as a service process with its own cache, worker
    pool and pre-warming, and Streamlit as a thin client.

14a. **Pure SABR pricing model:** price with SABR (Hagan) vols per strike, calibrated
  to the pillars, as a third pricing model next to vv and vv-sabr. The existing
  SABR code is to be made compatible with the library (pure numerics in Atlas,
  inputs explicit), then validated against the pillars, against vv, and for
  arbitrage.

14b. **Experimental local vol:** Dupire local vol from the calibrated surface across
  tenors, priced by PDE or Monte Carlo. It needs a smooth, arbitrage-free surface,
  so it starts as research in Delphi and moves to Atlas only if it proves out.

15. Smile study: a smoother wing, butterfly negatives on steep smiles (HKD, CNH,
    TRY), vv-sabr gamma noise. The Bloomberg benchmark (`tests/benchmark`,
    2026-10-23) agrees to 0.02 vol points (median) at ATM and delta strikes,
    and its deltas match. The far wing is where it parts:
    - USDCHF 1W, 4.6% out of the money: +1.55 vol points against Bloomberg;
    - EM fixed strikes: USDTRY −0.25, USDILS +0.15, USDZAR −0.13.

    These are the wing cases to study first.

### Pricing review: tests and validation backlog

Recorded 2026-10-22 against revision `e867ce3`. These are points to test and
validate, not blanket confirmed defects or approved implementation changes.
Distinguish the legacy `VannaVolga` batch API from the wrapped smile API used
by the app; findings in one do not automatically apply to the other.

**VV-01 is closed as intended behaviour:** at expiry the option delivers to
spot, so its value is spot intrinsic. Do not change that pricing convention.
Exercise and the resulting spot trade remain part of lifecycle item 8.

- **VV-02 - Projection convergence:** stress the legacy batch projection on
  dense and irregular strike grids and steep/oscillating smiles. Measure final
  monotonicity and convexity residuals after the 100-iteration limit, including
  price-to-vol inversion error. Establish scale-aware acceptance tolerances
  and validate how non-convergence should be reported; the current iteration
  limit alone does not guarantee an arbitrage-free result.
- **VV-03 - Scalar/batch consistency:** quantify legacy scalar-versus-batch
  vol, price and Greek differences, separating five-pillar versus
  three-pillar construction from projection effects. Include grid-composition
  dependence and the actual consumer paths. Preserve scalar/batch equality
  in the wrapped API and decide the intended contract for the legacy API.
- **VV-04 - Discounting fallback:** test missing or mismatched carried
  resolved inputs, unsupported conventions and invalid discount factors.
  Compare settlement-date simple discounting with the legacy continuous
  expiry-date fallback. Validate normal app paths, standalone callers and
  risk bumps; determine which fallbacks are intentional and whether counters,
  per-result diagnostics or explicit failures provide adequate visibility.
- **VV-05 - Volatility fallback:** exercise pillar preparation, interpolation
  and wing failures, distinguishing invalid inputs from expected numerical
  limits. Validate legacy substitute pillars/boundary vols and wrapped
  fallback diagnostics, including strict mode. Quantify effects on prices
  and Greeks and check that degraded results are identifiable by consumers.
- **VV-06 - Batch strike validation:** cover zero, negative, NaN, infinite,
  empty, duplicate and unsorted strike inputs in legacy and wrapped APIs.
  Establish the input contract and require clear, consistent rejection of
  malformed inputs rather than downstream math-domain or indexing errors.
- **VV-07 - Gamma convention:** reconcile the docstring's fixed base discount
  factor with the implemented derivative of that factor under fixed forward
  points. Validate against finite differences of the reported delta for
  spot/forward, premium-adjusted/unadjusted, call/put and buy/sell cases;
  identify documentation corrections separately from any calculation defect.
- **VV-08 - Implied-volatility diagnostics:** test bracket endpoint roots,
  unattainable prices, non-finite evaluations and root-finder failures.
  Assess whether errors expose the strike, target price, bracket endpoints,
  endpoint prices/residuals and underlying cause without concealing failure.
- **VV-09 - Quoted-pillar preservation:** measure how much the legacy
  no-arbitrage projection moves included quoted anchors and reprices the
  original quotes, across representative and stressed markets. Separate this
  from wrapped preparation and validate its quote/premium matching too.
  Decide acceptable tolerances and the policy when quote preservation and
  arbitrage constraints conflict.
- **VV-10 - Historical reproducibility and provenance:** replay valuations
  using the existing market snapshots, trade versions and event/calendar
  inputs. Inventory which inputs and diagnostics are already persisted, then
  identify gaps in code revision, dependency versions, model/settings/betas,
  calendar overrides and fallback provenance. Validate cache invalidation
  across pricing changes; a manual cache version is not a full valuation
  audit record. Coordinate with the lifecycle and historical stages.

## UI

16. **Users and saved settings:** all users see all portfolios and trades; each
    user saves their own columns and order per tab, model and betas and surface
    parameters, with defaults as the fallback.
17. Layout: header into the sidebar, tabs at the top, column pickers top right.
    Separate trade input and booking from the outputs and Greeks in the Pricer,
    since they don't fit on one page:
    - the input grid stays on top;
    - one action row holds **Price portfolio**, the default book and
      counterparty, **Book delta hedges** and **Book ticked lines**;
    - the per-line results and Greeks move below that row, out of the input grid,
      followed by the By-pair totals.
18. Low priority: replace the Markets tab's "Currency pair" box with an
    always-visible **Active market** selector and a small **Add pair (manual
    market)** control. The active pair stays plain session state owned by no
    widget, so the loaders keep working.
19. Low priority: the market picker's **Latest** label always names the newest
    snapshot. The snapshot actually loaded gets its own caption ("Loaded: … a
    newer one exists").
20. Small fixes: first keystrokes lost in a new grid line; README.
21. Compact amounts in the strike grid and the bucket report: `1.25m` at 1 million
    or more in absolute value, `250k` from 1k to 999.999k, and plain below 1k.
    Rounding: 2 decimals for m (`1.23m`), up to 1 for k (`250k`, `12.5k`). The
    sign is kept (`-3.4m`, `-250k`). Only the display changes: values, sorting,
    totals and exports stay full precision.

## Later

- Low priority: **add non-settlement holidays on the fly** when our calendar and
  the market disagree (a dispute, a holiday announced late). A user override
  layered on the Bloomberg list (`settlement_holidays.csv`): add or remove a
  date for a currency, with who, when and why; append-only like the other
  stores; the date rules use it at once; a refresh of the Bloomberg list keeps
  the overrides and reports any it now agrees with.
  Expiry days already have one (2026-10-22): a weekday on which both
  currencies are closed moves to the next good day (house rule), and
  `market_data/calendars/expiry_overrides.csv` or
  `pillars.set_expiry_override` (on the fly, in memory) says where the market
  treats a pair and day differently. Still to do: setting one from the app,
  with who, when and why recorded.
- Rate-risk matrix by currency and tenor (where rho and phi aggregate across pairs).
- Event calendars by currency: first agree the scope and approach before building:
  - per currency, per pair, or both (currency events plus pair overrides);
  - how a pair's calendar is derived (same event counted once? weights per pair?);
  - crosses vs USD pairs;
  - moving the per-pair versions already saved;
  - the event source.
- A better audit trail on the event calendar page: who changed what, when and why;
  a diff between versions (events added, removed, re-weighted); and which version
  each market and valuation date used. Not high importance, but needs attention.
- Market data: more forward points and divisors for a few currencies, an event
  source, scheduled daily fetches, saving and comparing markets.
- Not now: discounting basis / CSA; bid/ask (arrives with the strategy engine's
  cost structure).

## Stages

1. Finish the engine and risk framework, validated in the UI.
2. Service layer.
3. History by date and batch speed.
4. Strategy engine.
5. Backtest engine.
6. Portfolio optimizer.

Stages 4 to 6 start only once prices are fully validated, edge cases included, and
the full option lifecycle (item 8) is complete.
