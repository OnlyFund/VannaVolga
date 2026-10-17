# VannaVolga FX Volatility Analytics

This project is a research-oriented FX option analytics library. It builds
delta-quoted volatility smiles from ATM/RR/BF market quotes, prices options
with Vanna-Volga or SABR-enhanced models, and calculates fixed-surface and
spot-dependent smile risks.

## Scope and model architecture

The normal production path is:

1. `MarketData` stores spot, forward points, and the five market volatility
   quotes.
2. `helper_functions/VannaVolga_function.py` and
   `helper_functions/vanna_volga_helpers.py` normalize conventions, solve
   delta strikes, and construct the 10-delta/25-delta pillars.
3. `VannaVolga_ImpliedVol.VannaVolga` converts the three principal pillars
   into a Vanna-Volga price and inverts it to an implied volatility at the
   requested strike.
4. `surface_models/SmileExtension.py` applies spot-dependent ATM/RR/BF
   modifications before rebuilding the smile.
5. `pricing_utils/BlackScholes.py` provides the FX Black/Garman-Kohlhagen
   style pricing and analytic reference Greeks.
6. `Greeks/` provides finite-difference risks under fixed or modified smile
   dynamics.

`vanna_volga_sabr.py` is an optional model that calibrates SABR directly to
the five quoted pillars and uses SABR for wing behavior. The `research/SABR/`
directory also contains reference and experimental implementations; those
modules are not all production-ready.

## Market-data schema and units

`MarketData` fields are:

| Field | Meaning |
|---|---|
| `s` | Spot FX rate, quoted as domestic currency per foreign currency |
| `fwd_pts` | Forward points in market points |
| `term_discount` | Annualized continuously compounded term-currency rate; defaults to `0.05` |
| `forward_divisor` | Divisor converting forward points to rate units; normally `10000` |
| `v_atm` | ATM volatility in percentage points, so `6.50` means 6.50% |
| `rr_25`, `rr_10` | Risk reversals: call volatility minus put volatility |
| `bf_25`, `bf_10` | Butterflies: average wing volatility minus ATM volatility |
| `atm_conv` | ATM convention: `spot`, `atmf`/`forward`, or `dn`/`delta-neutral` |
| `deltaBase` | `False` for unadjusted delta, `True` for premium-adjusted delta (see also `delta_convention` for spot vs forward) |

The five quoted volatilities are reconstructed as:

```text
10P = ATM + BF10 - RR10 / 2
25P = ATM + BF25 - RR25 / 2
ATM = ATM
25C = ATM + BF25 + RR25 / 2
10C = ATM + BF10 + RR10 / 2
```

All volatility quote bumps use percentage points. A 10bp volatility move is
`0.10`, not `0.001`.

`MarketData` validates numeric market inputs when constructed: spot, the derived
forward rate, ATM vol, forward divisor, and central vol bump must be positive,
and quote values must be finite. ATM aliases are normalized at construction. SABR parameter mappings
are copied on input; the dataclass remains mutable because the app and legacy
callers update model options after construction. Use `copy_with()` when making
market quote perturbations. `from_vol_data()` uses the calendar already loaded
by its `VolData` instance.

## FX delta, forward, and ATM conventions

The forward is:

```text
F = S + fwd_pts / forward_divisor
```

The term discount factor is `exp(-term_discount * T)`. The base discount
factor is derived from FX forward parity as `D_base = D_term * F / S`. The
market editor stores the term rate alongside each forward-points quote and
interpolates both to the option delivery date.

For a call (`+`) or put (`-`), the signed forward quote deltas are:

```text
unadjusted:          +/- N(+/- d1)
premium-adjusted:    +/- (K/F) N(+/- d2)
```

`deltaBase=False` selects unadjusted delta and `deltaBase=True` selects
premium-adjusted delta. These conventions are used consistently by the
Black-Scholes reference pricer, delta-to-strike solvers, pillar construction,
and smile models.

### Spot versus forward delta

Premium adjustment is only *one* of two independent switches on an FX delta.
The second is whether the delta is quoted against a spot hedge or an outright
forward hedge:

```text
spot delta = D_base * forward delta
```

The two are routinely conflated because they are both called "the delta
convention". They answer different questions. Premium adjustment asks whether
base-currency premium already forms part of the base-currency exposure;
spot-versus-forward asks whether the hedge is a spot trade or a forward
settling on the option's delivery date.

`delta_convention` on `BlackScholes` and `black_scholes_values` selects
between `"spot"` (the default) and `"forward"`. The interbank market and
Bloomberg quote **spot** deltas out to roughly one year and forward deltas
beyond, so `"spot"` is the reporting default. The difference is not cosmetic:
at six months with a base rate near 3.4%, a forward delta overstates the spot
hedge by about 1.7% of notional, which is 0.9m on a 100m 50-delta EURUSD
option.

Gamma follows `delta_convention`, because gamma *is* the change in the delta
being reported. Note that `D_base = D_term * F / S` with the forward points
held fixed across a spot bump, so the base factor is itself spot-dependent and
the spot gamma carries the extra product-rule term
`-D_term * fwd_pts / S^2`. Holding rates fixed and holding forward points
fixed are incompatible; the ladder convention is to hold the points.

Vega, vanna and volga are deliberately **not** affected. They feed the
Vanna-Volga weights, so letting a delta *reporting* choice move them would
ripple into the smile construction and change quoted volatilities. For the
same reason every internal consumer — the delta-to-strike solvers, the pillar
construction and the Vanna-Volga calibration — pins itself to `"forward"`.
A consequence worth knowing is that a strike solved for "25 delta" reads back
as 25 delta only in the forward convention; in the reporting convention it
reads slightly lower, by the base discount factor.

### Premium quotation

The app has a **Delta convention** selector, defaulting to **Forward delta**,
with **Spot delta** as the alternative. `pricing.greeks.calculate_greeks` uses
the pair helper's premium-adjustment convention and reports
`spot delta = D_base * forward delta`. This reporting selector does not alter
smile pillars, strike aliases, prices or other option Greeks. Trade premium
currency still controls premium cashflows and quotation.

Library callers opt in using `MarketData.delta_reporting`. With the default
`None`, the earlier behavior remains: `dV/dS - V/S` for base-currency premium,
and `dV/dS` for term-currency premium, using fixed-forward-points derivatives.
Low-level finite-difference delta APIs still return raw price sensitivities.
See [the app documentation](app/README.md#trade-premium-currency-and-delta)
for formulas and consistent forward hedge sizing.

`BlackScholes.price()` returns a **cash amount in the term currency** for a
`Notional` expressed in the **base** currency. This is the canonical internal
unit: the vanna-volga solver, the finite-difference Greeks and the put/call
parity checks all rely on it, so it never changes with configuration.

To report a premium in a market convention use `premium(quote=...)`, which
rescales that value without altering the trade:

| `quote` | Meaning | Formula |
|---|---|---|
| `term_cash` | Term-currency cash (default) | `V` |
| `base_cash` | Base-currency cash | `V / S` |
| `term_pips` | Term pips per unit of base notional | `V / N * 1e4` |
| `base_pips` | Base pips per unit of term notional | `V / S / (N * K) * 1e4` |
| `pct_term` | Percent of the term notional | `V / (N * K)` |
| `pct_base` | Percent of the base notional | `V / S / N` |

For a 6m EURUSD call, `S = 1.1386`, `K = 1.1480`, `N = 1mm EUR`:

```text
term_cash  18131.9420 USD     term_pips  181.3194     pct_term  1.5794 %
base_cash  15924.7690 EUR     base_pips  138.7175     pct_base  1.5925 %
```

Two conventions are easy to get wrong. The term notional is `N * K`, **not**
`N * S` — percent-of-term is struck-notional based, and using spot overstates
it by `K / S` (here 0.8%). And cash-to-cash conversion uses **spot**, whereas
the pips and percent forms are struck-notional based and involve **K**.

The per-row `premium_ccy` field in the app is a separate concept: it selects
the currency the premium cashflow is *paid* in, and converts cash-to-cash at
spot. It does not change the delta or ATM convention.

#### In the app

The sidebar **Premium quote** selector sets the units of the per-line
`MTM (...)` and `P&L (...)` columns. It is **presentation only**: it does not
change the trade value, the delta convention, or the currency the premium
settles in. The conversions live in `helper_functions/premium_quotes.py`,
which is the single source of truth shared by `BlackScholes.premium()`, the
portfolio valuation layer and the UI.

Failure handling is layered so that a bad input degrades rather than breaks:

| Function | On bad input | Used by |
|---|---|---|
| `convert_premium_quote` | raises `ValueError` | engine, tests |
| `safe_convert_premium_quote` | returns `default` (`None`) | portfolio valuation |
| `format_premium_quote` | returns `"—"` | display |

`Pricable.price()` therefore always emits `mtm_<quote>`, `premium_<quote>` and
`pnl_<quote>` for all six conventions, using `None` where a conversion is not
well defined, so one malformed row cannot break a whole report.

Portfolio **aggregation** and the **spot ladder** remain in base-currency cash.
They sum across trades with different strikes, where pips and percent-of-
notional are not well defined, so quoting them in a struck-notional convention
would be meaningless rather than merely imprecise.

ATM aliases are normalized as follows:

| Aliases | Normalized convention |
|---|---|
| `spot` | Spot ATM |
| `atmf`, `forward` | Forward/ATMF |
| `dn`, `delta-neutral` | Delta-neutral ATM |

The delta-neutral ATM strike depends on the *delta* convention, because the
straddle whose deltas cancel is not the same strike under the two definitions:

```
unadjusted   (d1 = 0):  K = F * exp(+ sigma^2 T / 2)
prem-adjusted(d2 = 0):  K = F * exp(- sigma^2 T / 2)
```

The premium-adjusted delta-neutral strike is therefore always **lower**, by
approximately `F * sigma^2 * T`. For EURUSD at `F = 1.1470`, `sigma = 5.92%`,
this is about 10 pips at 3m, 20 pips at 6m and 49 pips at 1y. A strike that
moves when the delta convention is switched is expected behaviour, not an
error. Note that the convention is a property of the *quoted market data*
(for EURUSD the interbank convention is unadjusted, i.e. premium in USD), not
of the premium currency chosen for an individual trade; the per-row
`premium_ccy` field only controls the currency the premium is reported in.

Every other ATM convention (`spot`, `atmf`) is independent of the delta
convention. `get_atm_strike(..., delta_base=...)` takes the delta convention
directly and is the single source of truth; `get_dn_strike` is a thin wrapper
over it. `delta_base` defaults to `False` (unadjusted) and is ignored for
`spot` and `atmf`. The ATM smile pillar and the strike returned by the input
resolver are built from the same code path, so they agree exactly under all
six convention combinations.

A strike may also be entered as a delta alias such as `10d` or `25d`. A
delta-quoted strike is defined by the volatility *at that strike*: the
10-delta call strike is the strike whose delta is 10% when evaluated with the
10-delta call volatility, not with the ATM volatility. Volatility and strike
are therefore mutually dependent and are solved to a fixed point against the
smile. At a quoted pillar this converges onto the quote and reproduces the
market pillar strike exactly; between pillars it follows the interpolated
smile. Solving with the ATM volatility instead discards the butterfly and risk
reversal and pulls the wing strikes towards the money.

## Discounting and settlement

An FX option does not settle on its expiry date. It settles on its **delivery
date**, which is the expiry date rolled forward by the pair's spot lag over the
joint calendar of both currencies (plus USD for crosses). Most pairs are T+2;
`USDCAD` and `USDTRY` are T+1. Because the lag is in *business* days, the
calendar gap is frequently wider than the nominal lag: a 6m EURUSD option
expiring 2027-03-24 delivers on 2027-03-30, six calendar days later, and a
2026-12-24 expiry delivers on 2026-12-29 once Christmas and the weekend are
taken out.

Discounting therefore runs to the delivery date, at a **simple** money-market
rate, on the **term currency's own day-count basis** — ACT/360 for USD, EUR,
CHF, JPY and the Scandinavian currencies, ACT/365F for GBP, AUD, NZD, CAD and
others:

    DF = 1 / (1 + r * tau),    tau = daycount(valuation, delivery)

This replaces a continuously compounded `exp(-r * t)` run to expiry on an
ACT/365 clock. The two differences push in opposite directions — ACT/360
enlarges the year fraction while simple compounding raises the factor for a
given year fraction — so they partly offset and the net sign flips with tenor.
Against a 4.27% USD rate the convention factor is 6.7bp *lower* at 3m, 7.5bp
lower at 6m, and 0.9bp *higher* at 1Y. On a 6m EURUSD 1mm EUR call that is
roughly 14 USD of premium, or 0.14bp of notional.

### How it is applied

Discounting is driven entirely by the currency pair. `BlackScholes`,
`black_scholes_values` and `MarketData` all accept a `currency_pair`, and
supplying one is sufficient — there is no separate switch to enable. Passing
`settlement_discounting=False` on `MarketData` is an explicit opt-out that
restores the legacy continuous factor.

Every layer degrades rather than fails. When no pair is given, the pair is not
in the convention tables, the pair is quoted inverted (`USDEUR`), QuantLib is
not installed, or the dates do not define a discount, the pricers silently fall
back to `exp(-r * t)`. `safe_option_discount` never raises and `discount_kwargs`
returns an empty dict, which simply means "keep the legacy behaviour".

Swapping the discount factor is a *pure multiplicative rescale* of the price:
the base-currency factor is `DF_term * F / S`, so the price collapses to
`DF_term * (F*N(d1) - K*N(d2))`. Consequently the *forward* delta and its
gamma are completely unaffected, vega scales exactly with the factor, and the
volatility smile, the Vanna-Volga weights and every strike solver are
untouched. Only reported cash amounts move.

The reported *spot* delta does scale with the factor, because
`DF_base = DF_term * F / S` — both legs settle on the same delivery date, so a
change in the term discounting convention rescales the base factor by the same
ratio. That is intended, not a leak: it is the same hedge expressed against a
consistently discounted spot.

### Why a factor and a carry rate, not an equivalent rate

Theta consumes a *rate* rather than a factor, so it cannot be served by the
discount factor alone. The obvious shortcut — backing out an equivalent
continuously compounded rate `r_eq = -ln(DF)/t` and feeding that to the
existing code unchanged — reproduces the price exactly but corrupts theta at
short tenors. Because delivery sits a fixed couple of business days beyond
expiry, that lag dominates as expiry approaches: at a 4.27% quoted rate `r_eq`
reaches 6.8% at one week and 7.2% at one day.

The pricers are therefore given the factor and the true instantaneous decay
rate separately. For simple discounting,

    d(DF)/d(tau) = -r / (1 + r*tau)^2 = -DF * carry,    carry = r * DF = r / (1 + r*tau)

Unlike `r_eq`, this carry rate is bounded above by the quoted rate and tends to
it as expiry approaches, so it stays well conditioned at every tenor.
`equivalent_continuous_rate` exists in `helper_functions/fx_discounting.py` but
is documented as reporting-only for exactly this reason.

## Pricing and smile behavior

The Vanna-Volga engine uses three principal pillars (put wing, ATM, call
wing). RR/BF interpolation between 10-delta and 25-delta quotes is smooth,
avoiding a discontinuous switch between quote buckets. Single-strike queries
return the continuous Vanna-Volga solution; batch queries may apply a static
no-arbitrage projection across the complete strike grid.

Between the five quoted pillars, single-strike queries interpolate with a
natural cubic spline in log-strike. The previous implementation used a linear
interpolation, but the smile is convex between the quoted deltas, so a straight
chord sits systematically *above* the curve it is meant to approximate. The
error was worst midway between the ATM and 25-delta pillars -- the region most
options are actually struck in -- reaching about 5.8bp too high, and it
disappeared at the pillars themselves, so the quotes still reproduced exactly
and the bias was easy to miss. The spline reduces the maximum deviation from
the continuous Vanna-Volga solution to about 1.1bp.

A natural spline is used rather than the (slightly closer fitting) not-a-knot
variant because it is the only candidate tested whose second difference stays
strictly positive across the quoted range, so it cannot introduce a butterfly
arbitrage. PCHIP was also rejected: its monotonicity limiter suppresses exactly
the curvature being restored, leaving it barely better than linear. The
interpolation degrades to linear if SciPy is unavailable, if fewer than three
pillars are supplied, if any strike is non-positive, or if the spline returns a
non-finite value or one outside a band around the quoted volatilities.

The straight-line interpolation also flattened the *local* smile slope. Within
a segment the chord has a single constant gradient, so `dVol/dK` just inside
the ATM pillar was understated by roughly a factor of ten, and it jumped
discontinuously at every pillar. Any risk number derived by differentiating the
smile -- smile delta, vanna, the spot ladder -- inherited that error. The
spline restores a continuous, correctly scaled skew.

Outside the pillar range the smile is *extrapolated* with the continuous
Vanna-Volga solution, anchored to the boundary quote: the wing is shifted so it
meets the 10-delta pillar exactly. Anchoring matters because the Vanna-Volga put
anchor does not generally coincide with the 10-delta put pillar, so an
unanchored wing would leave a step of roughly 0.1 volatility points at the
boundary. All five quotes are still reproduced exactly.

Far outside the quoted range the option value underflows and the price
inversion stops being well posed; the Vanna-Volga wing also eventually turns
back on itself. In that region the extrapolation is held flat at the boundary
quote rather than being allowed to reverse direction or return a spurious
level. Note that a previous implementation used `np.interp` for the whole
strike axis, which clamps to the endpoint value outside the quoted range; every
out-of-range strike then returned exactly the 10-delta quote, giving a
perfectly flat wing.

The SABR extension keeps a frozen, validated calibration per base market/expiry/surface and reuses it for spot-ladder repricing and volga bumps. When the calibration residual or finite-value checks fail, the model falls back explicitly to the legacy Vanna-Volga path rather than silently producing a discontinuous surface. Single-strike SABR output uses a stable direct SABR quote instead of a local mini-smile projection; if that path is invalid, the current Vanna-Volga projection is used as the fallback.

SABR calibration least-squares fits `(alpha, rho, nu)` to the five quoted
pillars at fixed `beta`. That objective is not convex in `(rho, nu)`, so the
descent is seeded from a small deterministic grid of starting points and the
best result is kept. A single fixed seed stalls in a local minimum that fits a
far too flat smile, undershooting both 10-delta wings by around 0.25
volatility points and missing ATM outright. The calibration is cached per
instance so spot and volatility bumps do not repeat the search.

The smile extension uses:

- `beta_atm` to move ATM volatility with spot;
- `beta_rr` to move risk reversal with spot;
- `beta_bf` to move butterfly with spot.

These parameters describe a desk/model smile dynamic between unchanged
delta-quoted behavior and sticky-strike-like behavior. When
`keep_smile_ratio=True`, the 25-delta wing volatilities move proportionally
with ATM for the ATM-driven component. The 10-delta quotes are then inferred
from the 10/25-delta ratios. Subsequent explicit RR/BF beta adjustments may
change the final wing ratios by design.

Spot perturbations estimate skews on the unbumped reference smile and apply
the same signed log-spot move to ATM, RR, and BF. RR/BF slopes use log-strike
units consistently with ATM; zero spot move preserves the original surface.
Spots must be positive and finite and beta factors finite and non-negative.
Numerical failures warn and fall back to unchanged quotes at the requested
shifted spot, retaining the selected model; failed fallbacks propagate.
See [smile spot-bump safeguards](app/README.md#smile-spot-bump-safeguards)
for flooring, diagnostics, and input-validation details.

## Greek definitions

### Fixed Greeks

Fixed Greeks use a 100% sticky-delta market surface. ATM/RR/BF quotes remain
unchanged when spot moves, delta-defined pillar strikes move with spot, and a
fixed-strike option slides across the unchanged smile.

| Greek | Definition and units |
|---|---|
| Delta | `dPrice/dSpot` from a central spot bump |
| Gamma | `d(quote delta)/dSpot`, scaled for a 1% **relative** spot move |
| Vega | Price change per one volatility point under the selected quote bump |
| Vanna | Change in point vega for a 1% relative spot move |
| Volga | Price second difference per squared volatility-point bump |
| Theta | Rolled price change over one following business day |
| Rega | Price change for the configured RR quote bump |
| Sega | Price change for the configured BF quote bump |

#### Cash Greek currency

`BlackScholes.price()` returns a **term**-currency amount, but every cash Greek
is reported in **base** currency, the same currency as the delta hedge. Delta
gets this for free because `dPrice/dSpot` divides term currency by spot. Every
other Greek is built from a price difference, so it stays in term currency
unless it is explicitly divided by spot. Vega, theta, volga, rega and sega
therefore apply that conversion; gamma, vanna and delta are already base
currency by construction.

Two conventions follow from this and are pinned by
`tests/test_greek_conventions.py`:

- Gamma and vanna use a 1% *relative* spot move, so the spot derivative is
  multiplied by `S * 0.01`, not by `0.01`. Using an absolute move instead
  understates gamma by a factor of spot.
- Price-difference Greeks are divided by spot. Omitting this overstates vega
  and theta by a factor of spot.

The currency conversion is a *reporting unit*, not a risk, so it is held at
the valuation spot when differentiating. Vanna in particular must re-express
each bumped vega at the valuation spot; differentiating the conversion itself
adds a spurious `-vega/S` term that near the money is large enough to flip
vanna's sign.

#### Comparing vanna with other systems

Vanna is a cross derivative and is quoted two equivalent ways. This repo
reports the *vega change for a 1% relative spot move*; Bloomberg's `OVML`
reports the *delta change per volatility point*. They differ only by the
`0.01` scaling and by the term discount factor, since `delta()` is an
undiscounted forward delta while `vega()` is discounted:

```
vanna_here = 0.01 * (dDelta/dVolPoint) * term_discount_factor
```

A roughly 100x gap against Bloomberg's vanna is this convention, not an error.

### Smile Greeks

Smile Greeks first apply the configured spot-dependent ATM/RR/BF
modifications, rebuild the smile, extract volatility at the fixed strike, and
then reprice.

Smile theta is the price change over calendar days to the next FX trading
day, including three days from Friday to Monday. The app rolls both the
volatility and forward tenor curves, holding spot, strike and expiry fixed.
Volatility is re-solved on the rolled smile so the changing forward moves
the fixed strike along the skew. Total variance interpolates using calendar
days plus explicit event variance weights, without automatic weekend discounts.
Without events, the short end holds the first pillar's quoted volatilities.
Pair and spot-ladder totals include the existing delta hedge's forward-roll
P&L at its original contract rate, using the current undiscounted forward
valuation convention.

| Greek | Definition and units |
|---|---|
| Delta | Smile-adjusted `dPrice/dSpot` |
| Gamma | Smile-adjusted `d(quote delta)/dSpot`, per 1% relative spot move |
| Vega | Price sensitivity to a proportional five-point smile bump |
| Vanna | Finite-difference change in point vega for a 1% relative spot move |
| Volga | Change in point-vega for a 10bp proportional smile-vol bump |
| Rega | Price change for a 10bp RR bump |
| Sega | Price change for a 10bp BF bump |

For RR/BF risks, the 25-delta quote is bumped directly and the 10-delta quote
is bumped by the configured ratio. This reflects the market convention that
the 10/25-delta shape is approximately stable for a quote risk.

Quote delta is not automatically the same as the derivative of price with
respect to spot, especially for premium-adjusted delta. Do not compare quote
delta or quote-delta gamma with price delta or price gamma without matching
the numeraire and scaling conventions.

When the app's delta-hedge option is enabled, it adds grouped base-currency
forward positions to the portfolio. A positive forward amount receives base
currency. Its undiscounted term-currency MTM/P&L is
`(market forward - contract rate) * signed base amount`; base-currency MTM/P&L
is that amount divided by scenario spot. Each forward contributes its signed
base amount to both fixed and smile delta totals.

## Forward volatility and events

`market_data/vol_data.py` builds a forward volatility curve from total
variance. Weekdays contribute one variance day, Saturday contributes zero,
Sunday contributes 0.2, and configured events add variance days. The event
interval is `(start, end]`; the base interval is `[start, end)`.

Interpolation is linear in total variance, which implies piecewise-constant
forward variance between pillars. Quotes are annualized over calendar time, so
total variance is accumulated in calendar-annualized units (`quote**2 *
calendar days`) and annualized over the target's calendar days; the
event-weighted variance-day clock is only the abscissa the interpolation runs
against. Keeping the two measures separate is what lets the curve reproduce
each quote exactly at its own pillar while still shaping the curve between
pillars around weekends and events. The diagnostic forward-volatility columns
are bucket differences of total variance; they are not a full monotone-convex
Hagan-West implementation. Negative bucket forward variances issue a warning
and are floored to zero for those diagnostic columns. Interpolation rejects
invalid, duplicate, or non-increasing future expiry pillars rather than silently
choosing a bracket. A pillar on the valuation date is retained as the
zero-variance interpolation anchor.

Volatility and event CSVs are resolved from the current working directory
first, then from the bundled samples in `market_data/samples/`. Explicit paths retain
priority. Invalid required quote fields are dropped during CSV cleanup, but
the loader raises if no valid rows remain, reconstructed wing vols are not
positive, or expiry/variance times are ambiguous. Malformed nonnumeric event
weights warn and fall back to zero; missing event-calendar files raise an
error rather than silently disabling event adjustments.

## Market data store and how the app uses it

All market data lives in one SQLite file, `data/local/market_data.sqlite`
(`market_data/store.py`; `VV_MARKET_STORE` overrides the path). Nothing in
the app talks to Bloomberg: only `market_data.bloomberg_fetch` does, and a
test guards that the app never imports it.

### Market snapshots

`python -m market_data.bloomberg_fetch [YYYY-MM-DD] [--no-vols]` fetches one
valuation date (a one-day `bdh` window, not a historical bulk pull) and saves
it as one **market snapshot**:

| Field | Meaning |
|---|---|
| `market_id` | Running serial number |
| `market_source` | `bloomberg`, `import` (`manual` later) |
| `date_time` | When the data is from: the fetch time live; 23:59:59 for a past date |
| `fetched_at` | When it was pulled (audit trail) |
| `event_set_id` | The event calendar pinned to this market (see below) |

Its quotes are rows `(kind, name, tenor, value)`:

* `spot`: the 20 USD pairs (crosses are derived through USD);
* `depo`: quoted deposit/OIS curves per currency, decimal, 1W-1Y (`depo_tickers.py`;
  CNH uses the CNY curve, MXN is Banxico O/N plus 28-day TIIE, HUF BUBOR/FRA);
* `vol_atm`, `vol_rr_25`, `vol_rr_10`, `vol_bf_25`, `vol_bf_10`: ON-1Y, vol points
  (`vol_tickers.py`, `<PAIR><V|25R|10R|25B|10B><TENOR> Curncy`);
* `fwd_pts`: 1W-1Y in the pair's pips (`fwd_tickers.py`: `EUR1M Curncy` for
  EURUSD, `JPY12M Curncy` for USDJPY, `EURGBP1M Curncy` for crosses).

The universe (`market_data/universe.py`) is the 45 G10 crosses plus 20 EM pairs
(USD and EUR against SGD, HKD, ZAR, MXN, TRY, CNH, PLN, HUF, ILS, CZK). G10
butterflies are broker strangles; EM butterflies are smile strangles. A
store written by an older schema is backed up and migrated in one
transaction; a newer schema is refused.

**Rule:** a market can value its own date or later, never an earlier one: the
app uses the latest snapshot whose `date_time` is on or before the
valuation date (`app/store_markets.py`), and refuses future data.

### Event calendars

Events are kept apart from the quotes as versioned **event sets** of per-pair
rows `(pair, date, event, variance_days)`; several events on one date add up.
Each fetch pins the latest event set to the new market, so a market always
prices with the same events.

```bash
python -m market_data.events import calendar.csv   # columns: pair,date,event,variance_days
python -m market_data.events list
python -m market_data.events pin MARKET_ID EVENT_SET_ID --reason "..."
```

A CSV is validated whole: any bad row rejects it with every problem listed.

### Discount rates (`term_discount`)

`market_data/discount_curves.py` gives each market's term-currency rate:
SOFR for USD; for any other currency the rate implied by its USD pair's
forward points and SOFR (covered interest parity, the USD-collateralised
rate, consistent with the forwards). Fallbacks per currency: the quoted local
depo curve (basis not included), then the latest earlier snapshot within 5
business days (stale), then **unavailable**: the pair is not priced rather
than priced on a made-up rate. Each fetch prints a basis alert when
implied-minus-quoted moves more than 50bp (G10) / 200bp (EM).

### Markets in the app

* `market_data/market_tables.py` builds each pair's app market from a snapshot:
  vol and forward tables with expiry/delivery dates from the pair's
  conventions (`fx_conventions.option_dates`), discount rates, pinned events.
* The app loads the latest snapshot on or before the sidebar valuation date
  (`app/ui/store_sync.py`); the sidebar shows which market is used and any
  notes. Per pair, **edits win over an uploaded file, which wins over the
  store** (`app/market_layers.py`); the Markets tab shows each pair's source.
  Changing the date reloads clean store pairs and keeps files and edits. With
  no market for the date, clean store pairs become unavailable.
* **No store, sample market.** Without a store (no `VV_MARKET_STORE` and no
  `data/local/market_data.sqlite`: a fresh checkout or the Streamlit Cloud
  deploy) the app opens on an illustrative sample market for all 65 pairs,
  dated on the valuation date and kept in a temporary folder
  (`market_data/sample_market.py`). Its values are made up, not market data,
  and the sidebar says so. Download it with **Save all markets** on the
  Markets tab, edit the JSON and load it back with **Load market**.
  `python -m market_data.sample_market [YYYY-MM-DD] [--store PATH]` writes one
  into a store.
* Only the pairs in use are calibrated up front (the active pair, the Pricer
  lines' pairs, the selected Portfolio trades' pairs), and a pair only when
  its market changed (`app/calibration_plan.py`). Store pairs go through a
  disk cache, `data/local/calibration_cache/`, kept 24 hours per (market,
  pair, valuation date, settings, code version) (`app/calibration_cache.py`):
  a market is calibrated once and restored afterwards, also after a restart.

## Repository structure

```text
app/                              Streamlit risk app (markets, pricer, trades, reports)
market_data/                      Market data: model, stored snapshots and sources
  model.py                        MarketData quotes and Modifications (smile dynamics)
  store.py                        Market snapshots, quotes and event sets (SQLite)
  bloomberg_fetch.py              Fetch spots, depos, vols and forward points (Bloomberg)
  universe.py                     The 65 pairs and their butterfly convention
  depo_tickers.py, vol_tickers.py, fwd_tickers.py   Bloomberg tickers
  events.py                       Event calendar CSV import, list, re-pin
  discount_curves.py              SOFR / forward-implied rates with fallbacks
  implied_rates.py                Covered-interest-parity implied rates
  market_tables.py                App markets (vol, forward, events) from a snapshot
  sample_market.py                Illustrative sample market (no store, cloud deploy)
  vol_data.py, event_calendar.py  CSV vol curves and event variance-day calendar
  samples/                        Example vol, event calendar and market files
VannaVolga_ImpliedVol.py           Main Vanna-Volga implied-vol engine
vanna_volga_sabr.py                Five-pillar SABR/Vanna-Volga extension
pricing/, pricing_utils/          Pricing entry points; Black-Scholes and SABR routines
helper_functions/                 Strike, delta, interpolation, and convention helpers
surface_models/                   Spot-dependent smile reconstruction
Greeks/                           Fixed, smile, cross, decay, and weighted risks
research/                         Experimental scripts: SABR, variance swap, digitals,
                                  cross skew, Greek comparisons (not used by the app)
tests/                            Regression and golden tests
data/local/                       Runtime data, not committed: trades, market data
                                  store, calibration cache
```

## Running examples and tests

Run commands from the project directory:

```bash
python -m pytest -q
python -m pytest -q tests/test_vol_modeling_consistency.py
python -m pytest -q tests/test_forward_vol_curve.py
python -m market_data.bloomberg_fetch     # needs a Bloomberg terminal
python -m market_data.events list
```

Example scripts should be run as modules or with the project directory on
`PYTHONPATH`; importing them should not execute pricing or plotting examples.

## Numerical and implementation assumptions

- Dates use the repository's `year_fraction` helper and a 365-day year unless
  a method explicitly documents a business-day convention.
- Black-Scholes volatility inputs are percentage points at the public API.
- Forward points remain fixed during analytic spot sensitivities.
- Finite-difference Greeks depend on their bump sizes; results are not
  analytic derivatives of the full calibrated surface.
- Implied-volatility root failures are surfaced as exceptions rather than
  silently replaced with ATM volatility.
- No-arbitrage projection is applied only where documented, because projecting
  a changing local strike set can introduce discontinuities.
