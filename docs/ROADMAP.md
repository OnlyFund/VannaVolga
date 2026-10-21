# FXO Risk Analyzer roadmap

Last updated 2026-10-20. Decisions are recorded with the item they belong to.

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
- **A stored snapshot is static:** its valuation of a book is deterministic, so it
  is computed once and cached.

## Library architecture: Apeiron

The suite is **Apeiron** ("the boundless"), an FX option and volatility analytics
suite with research, pricing, risk and strategy outputs. It is one package with
six levels:

```
apeiron/
  atlas/      core numerical engine
  gaia/       data foundation: time/, market/, book/
  hyperion/   engine room: analytics/, risk/, services/, cache/
  athena/     decision layer: strategy/, optimizer/, backtest/
  delphi/     research
  olympus/    user interface (Streamlit)
```

- **Atlas** (core) imports nothing else in the suite. It is pure numerics, usable
  by anyone: dates, year fractions, discount factors and conventions are inputs.
- **Gaia**'s three parts are each independent of one another and of Atlas:
  - **time:** calendars, holidays, pillar dates, cuts, rollover, variance days;
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
   - the intraday cut and the pillar date solver → `gaia/time`;
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

## Next

1. Small UI: a Refresh button in the market picker; collapsible Blotter
   amend/cancel sections.
2. **Step 0 of the Apeiron migration:** the package skeleton and the
   import-boundary test.
3. **Result caching (priority)**, built in `hyperion/cache`.
   - Key per trade: snapshot, trade and version, model settings, spot edits,
     requested Greeks and code version, so a change reprices only what it affects.
   - A bounded in-memory LRU, plus an on-disk store kept 24 hours (the portfolio
     summary per loaded snapshot for all currencies).
   - Different inputs give a different key; an unreadable cache means computing
     fresh; a check mode compares a sample against fresh results.
4. **Intraday time to the expiry cut** (New York / Tokyo), built in `gaia/time`, with:
   - special cuts and fixings per pair (MXN, BRL, KRW, CEEMEA): cut time, fixing
     source and fixing date;
   - the valuation time of day (snapshot time vs valuation moment);
   - **sub-1-week options** (expiries before the first quoted pillar): Greeks,
     especially gamma, delta and theta, switch from whole days to the actual time
     to expiry. Mid priority; uses the same time machinery as the cut.

## Dates and data foundation

5. **Atlas untangling:** the core engine takes dates, year fractions and
   discount factors as inputs, instead of reading pair calendars itself.
6. **Pillar date solver** (in `gaia/time`): one date service for O/N, days,
   weeks, months and years from any valuation date.
   - Delivery is spot plus tenor with **Modified Following** (a weekend or holiday
     in either currency moves to the next good day, back to the previous one if
     that crosses the month) and the **end-of-month rule** (spot on the month's
     last business day gives the target month's last business day).
   - Week tenors count from the valuation date. Expiries skip weekends and
     25 Dec / 1 Jan. Each currency settles on its own holiday calendar.
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
   - The conventions:
     - forward points are **sticky tenor** (spot and rates unchanged, so
       fixed-delivery forwards roll down the curve);
     - vols keep **forward variance**: only the passing day's own variance drops
       out, so event days and weekends are handled correctly under variance-day
       weighting.

     Sticky-tenor vols remain available only as a comparison.
   - The trade date rolls at **17:00 New York**.
   - Open: the USD intermediate-day rule for crosses' spot, T+1 pairs, expiry for
     a delivery date, holiday-file gaps (some currencies use the USD calendar),
     whether options use the SN point.
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

14. Performance after caching: spot-ladder market builds, repeated market builds
    per line, spline rebuilds per vol lookup, smile calibration per pair.
- **Pure SABR pricing model:** price with SABR (Hagan) vols per strike, calibrated
  to the pillars, as a third pricing model next to vv and vv-sabr. The existing
  SABR code is to be made compatible with the library (pure numerics in Atlas,
  inputs explicit), then validated against the pillars, against vv, and for
  arbitrage.
- **Experimental local vol:** Dupire local vol from the calibrated surface across
  tenors, priced by PDE or Monte Carlo. It needs a smooth, arbitrage-free surface,
  so it starts as research in Delphi and moves to Atlas only if it proves out.

15. Smile study: a smoother wing, butterfly negatives on steep smiles (HKD, CNH,
    TRY), vv-sabr gamma noise.

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

## Later

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
