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
