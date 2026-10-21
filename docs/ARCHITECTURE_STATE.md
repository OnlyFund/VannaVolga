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
