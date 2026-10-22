# Streamlit pricer

Run from the VannaVolga project root:

```bash
streamlit run app/streamlit_app.py
```

### Streamlit Community Cloud

Deploy with `app/streamlit_app.py` as the main file and **Python 3.11** in
Advanced settings. The root `requirements.txt` contains the app and pricing
dependencies, including Numba and QuantLib. Plotly and Seaborn are only needed
by the research scripts, not this app.

The entrypoint-local `app/requirements.txt` includes the root requirements and
takes precedence over the legacy root `Pipfile`/`Pipfile.lock`, which target
Python 3.7. Push both requirements files when deploying. Do not put competing
dependency manifests in `app/`. Select Python 3.11 explicitly; requirements
files do not select the interpreter. If an existing deployment uses another
Python version, delete and recreate that Cloud app with Python 3.11 selected.
Streamlit is pinned to 1.36.0 because the UI still calls
`st.experimental_rerun()`.

On Streamlit Cloud the trade store is **in memory**: booking, amending and
cancelling work as locally, but the book lasts only for the browser session
and is not saved (the app says so). Locally the store is
`data/local/trades.sqlite`; `VV_TRADE_STORE` names another file, or `memory`
for an in-memory store.

Bloomberg ingestion is a separate local workflow requiring Bloomberg access;
it is not installed by the app requirements and cannot use a desktop Bloomberg
session from Community Cloud. Supply market data separately. Cloud-local
SQLite databases and other generated files are not durable across redeploys.

The app shares one market surface across an editable portfolio table. Each row
contains a numeric strike, delta strike such as `25d`, or ATM alias
(`atm`, `atmf`, `atms`), expiry, call/put, buy/sell, and notional. Expiry can
be a case-insensitive surface pillar tenor (`1w`, `2w`, `1m`, ..., `1y`) or a
broken date in `mm/dd/yyyy` or `ddmonyy` form, such as `09/18/2025` or
`18sep25`. Tenors resolve from the valuation date to their pillar date. `atm` uses
the pair's automatic ATM convention, `atmf` uses forward ATM, and `atms` uses spot
ATM. Delta strikes are numerically verified on the selected model using the
pair/tenor's spot or forward delta and independent premium adjustment.
Resolved aliases are written into the strike input at pair-specific precision; numeric
strikes retain their entered value. Invalid or incomplete aliases remain in
the editor so they can be corrected. The
calculation output reports model volatility, MTM, and Greeks per option, then
aggregates the numeric results across all rows.

The Markets tab has editable Vol data, Forward points, and Event calendar
subtabs. Spot and currency pair sit above these tables; the forward divisor,
ATM/delta conventions and settlement lag are automatic, with a read-only summary.
Pairs accept spaces and slashes but are never inverted automatically.
Expiry and event dates display as MM/DD/YYYY; delivery uses the pair's joint
holiday calendar and settlement lag. Unknown conventions visibly fall back to
T+2 business days, dn ATM, premium-adjusted delta and the stored legacy divisor
(10000 if absent). Forward points and decimal SIMPLE term deposit rates
(`0.05 = 5%`) interpolate by delivery date only within the quoted range.
Vol quotes interpolate in calendar-time total variance, with explicit event
variance days added to the interpolation clock. Weekends use full calendar
days, with no automatic weekend variance discount. All three tables
feed portfolio pricing and the spot ladder.
Save market downloads a JSON file named `<currency_pair>_<valuation_date>.json`;
Load market accepts that file and restores the valuation date, spot, divisor,
and three market tables. Broker-strangle input mode is saved too; old files
without it default to broker mode. Saved rate numbers are preserved, not rescaled.

The app opts into `wrapped_vol_resolver` for pricing, fixed/smile Greeks,
perturbations, strike aliases and both ladder engines. Library callers remain
on legacy models unless explicitly opted in. Broker butterflies are converted
using the selected model **at each quoted expiry before interpolation**.
Disable "BF quotes are broker strangles" for existing two-vol butterflies.
Broken-date and bumped quotes are already two-vol and are not reconverted.
Preparation is reused across strikes, with input-sensitive bounded caches.

If any pillar cannot be prepared or converted, the whole curve coherently
retains the original raw quotes and uses the selected legacy model. VV-SABR
is **not** changed to VV when broker premiums cannot be matched.
Degraded preparation/evaluation (including Greek-only bumps) produces visible
deduplicated warnings, per-line status/reasons and ladder fallback summaries.
Malformed inputs still raise errors instead of silently returning an ATM guess.

Fixed Greeks use the unchanged sticky-delta surface. Smile Greeks apply the
configured ATM/RR/BF beta dynamics. Both Greek vectors are reported for every
option. Volatilities are percentage points, and gamma is quote-delta
sensitivity per 1% relative spot move. Vanna is the point-vega change for a 1%
relative spot move. All cash Greeks are reported in base currency, matching
the delta hedge, even though the underlying price is a term-currency amount.

### Trade premium currency and delta

Rho and phi are on-demand Greeks in the **rates** picker group, each per
+10bp (0.001 decimal rate), using central differences at fixed spot, strike
and dates. **Rho** is the sensitivity to the term currency's implied depo rate
(SOFR for USD), reported in the **term currency**; it moves forward points
consistently with FX parity while holding the base discount factor fixed.
**Phi** is the sensitivity to the base currency's implied depo rate (implied
by spot, forward and the term discount factor), reported in the **base
currency**, holding the term factor fixed.
Settlement mode uses simple rates to delivery on each currency's day count;
legacy mode uses continuous rates to expiry. Sticky-delta holds ATM/RR/BF
quotes unchanged and rebuilds pillars. Smile additionally applies the existing
betas to `log(F_bumped/F)`, using original-surface skews to shift quotes without
moving actual spot. This forward-driven beta response is an explicit modelling
convention, not an independently calibrated rate/vol correlation. A +10bp
sensitivity is the derivative times 0.001, not the exact one-sided scenario P&L.

The sidebar's **Delta convention** selects **Forward delta** (default) or
**Spot delta** for both delta columns, pair totals and spot-ladder reports.
Premium adjustment comes from `fx_conventions.get_pair().premium_adjusted`,
not the row's premium currency. Trade premium currency still controls cashflow
and premium reporting. The selector does not change smile pillars, resolved
strikes, implied volatility, option prices, or other option Greeks.

For forward `F`, spot `S`, term discount factor `D_term`, signed option value
`V` and the existing fixed-forward-points derivative `dV/dS`:

| Pair convention | Spot delta | Forward delta |
|---|---|---|
| Unadjusted | `(F/S) * dV/dS` | `(dV/dS) / D_term` |
| Premium-adjusted | `(F/S) * dV/dS - V/S` | `(dV/dS - V/F) / D_term` |

Thus `spot delta = D_base * forward delta`, with `D_base = D_term * F/S`.
Forward hedge sizing converts the selected delta back into forward notional;
its delta contribution uses the same reporting units, so switching the selector
does not change the hedge notional. Hedge theta uses that existing notional.

Library callers can opt in with `MarketData(delta_reporting="forward")` or
`"spot"`. The default `None` preserves the earlier trade-premium-adjusted raw
derivative behavior. Low-level `FixedDelta` and `SmileGreeks.delta()` remain
raw price-derivative APIs. Existing gamma calculations and their conventions
are unchanged by this delta-only selector.

Smile theta is the calendar-day price change to the next FX trading day:
normally one day, or three days from Friday to Monday (longer over trading
closures). Spot, strike, expiry and delivery stay fixed. Both volatility and
forward tenor curves roll, and the option's volatility is re-solved at its
fixed strike on the rolled smile, capturing movement along the skew as the
forward changes. Without explicit events, a flat tenor-volatility curve stays
flat, including below the first pillar. Explicit event weights redistribute
pillar variance over the remaining calendar days.
Risk-curve expiry and delivery coordinates translate by elapsed calendar days,
without settlement-calendar adjustment. A settlement holiday cannot suppress
forward roll-down, and saved expiry/delivery offsets are preserved. The
option's actual contractual delivery remains on its settlement calendar.

With a delta hedge enabled, pair totals and spot-ladder smile theta include
the existing forward hedge's rolled P&L at its original contract rate. The
forward uses the same undiscounted valuation as its MTM; its fixed theta is
zero when forward points are held unchanged. Theta is reported in the pair's
conventional premium currency.

### Smile spot-bump safeguards

Spot dynamics are measured on the unbumped reference smile. All ATM/RR/BF
shocks use the same signed `log(shifted_spot / reference_spot)` move; zero
move returns the original model surface without rebuilding quotes. Local
slopes are volatility points per unit log strike: RR slope is call25 slope
minus put25 slope, and BF slope is their average minus ATM slope. Both spot
perturbation entry points use the same reference surface.

Spots must be positive and finite; beta factors must be finite and non-negative.
Invalid inputs raise rather than being clipped or silently ignored. The existing
ATM factor interpretation and proportional wing scaling remain unchanged.
Principal wing/ATM volatility floors remain in place, with logging when reached;
all five reconstructed vols and returned model vols must be positive and finite.

Numerical smile-construction or repricing failures fall back to unchanged quotes
at the **requested shifted spot**, using the same pricing model and conventions.
Fallbacks log and emit `DegradedSmileWarning`; wrapped app markets also record
visible degradation diagnostics. A failed fallback propagates its error, and
programming errors are not caught. The fast legacy VV interpolation path can
fall back to pricing-model skew probes, with a logged warning.

The Spot ladder tab calculates an aggregated MTM and fixed/smile Greek vector
for each spot level. Enter the number of levels on each side of spot and the
spot increment as a percentage of spot (for example, `0.5` means `0.5%` per
step). Delta- and ATM-based strikes are resolved at the base spot and then
held fixed across the ladder, matching a spot-scenario analysis.

Application input resolvers live in `helper_functions/input_resolvers.py`.
Risk-report calculations and rendering are separated into `app/reports/`, with
the spot ladder implemented in `app/reports/spot_ladder.py`.
