import numpy as np
from numba import njit
from helper_functions.model_constants import INV_SQRT_TWO_PI
from helper_functions.helper_functions import year_fraction, cp_flag, pos_flag
from helper_functions.premium_quotes import PREMIUM_QUOTES, convert_premium_quote
from helper_functions.fx_discounting import safe_option_discount
from math import sqrt, log, erf

try:
    # ``ndtr`` is a compiled ufunc re-exported dynamically from
    # ``scipy.special._ufuncs``, so static analysers frequently report it as an
    # unresolved import even though it is present at runtime. Import it from
    # the private module first so the reference resolves, then fall back to the
    # public name, then to an erf-based implementation if SciPy is unavailable.
    from scipy.special._ufuncs import ndtr
except ImportError:  # pragma: no cover - exercised only on exotic SciPy builds
    try:
        from scipy.special import ndtr
    except ImportError:  # pragma: no cover - SciPy missing entirely
        _INV_SQRT_TWO = 1.0 / sqrt(2.0)
        # ``otypes`` is required so an empty input does not raise; np.vectorize
        # otherwise infers the dtype by calling the function on the first
        # element, which does not exist.
        _erf = np.vectorize(erf, otypes=[float])

        def ndtr(x):
            """Standard normal CDF, vectorised, matching SciPy's ``ndtr``."""
            x = np.asarray(x, dtype=float)
            result = 0.5 * (1.0 + _erf(x * _INV_SQRT_TWO))
            return result if result.ndim else result[()]




def d1(fwd: float, K: float, vol_decimal: float, t: float) -> float:
    if vol_decimal <= 0 or t <= 0:
        return 0.0
    return (log(fwd / K) + 0.5 * vol_decimal**2 * t) / (vol_decimal * sqrt(t))


@njit
def _vega_numba(pdf_d1, f, sqrt_t, nS, BuySell):
    # ``nS`` is Notional / S, which converts the term-currency vega amount
    # into base currency so vega matches the currency of delta.
    return BuySell * pdf_d1 * f * sqrt_t * 0.01 * nS

@njit
def _volga_numba(vega, d1, d2, vol_decimal):
    # Vega is quoted per volatility point; differentiate with respect to
    # volatility points rather than decimal volatility.
    return vega * d1 * d2 / (vol_decimal * 100.0)

@njit
def _vanna_numba(vega, f, S, d1, v, sqrt_t):
    # Change in base-currency point vega for a 1% relative spot move.
    #
    # The base-currency conversion (vega_term / S) is a *reporting unit*, not a
    # risk, so it is held at the valuation spot rather than differentiated.
    # Differentiating it would add a -vega/S term that has no risk content and
    # near the money is large enough to flip vanna's sign.
    d_vega_term_dS = vega * S * (1.0 / f - d1 / (f * v * sqrt_t))
    return d_vega_term_dS * 0.01


class BlackScholes:
    """Price an FX vanilla option using forward-based Black pricing.

    Public volatility inputs are percentage points. ``fwd_pts`` is in market
    points and converted with ``fwd_divisor``. ``term_discount`` is an
    annualized continuously compounded term-currency rate. FX forward parity
    ``Notional`` is a base-currency amount and prices are in term currency;
    forward delta follows the repository's quote convention. Use
    :meth:`premium` to re-express the price in the market quotation
    conventions (pips or percent, base or term).

    Cash Greeks (delta, gamma, vega, vanna, volga, theta) are reported in
    **base** currency, matching the market convention that quotes risk in the
    same currency as the delta hedge. Term-currency amounts are converted with
    ``Notional / S``. Gamma and vanna are scaled to a 1% *relative* spot move.
    """
    def __init__(self, S, K, fwd_pts, v, expiry_date, valuation_date,
                 CallPut, BuySell, Notional, deltaBase=True, fwd_divisor=10000,
                 term_discount=0.05, term_discount_factor=None, carry_rate=None,
                 currency_pair=None, delta_convention="spot"):
        self.S = np.asarray(S, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.fwd_pts = fwd_pts / fwd_divisor
        self.f = np.asarray(S + self.fwd_pts, dtype=float)
        self.v = np.asarray(v, dtype=float) / 100.0
        self.t = np.asarray(year_fraction(expiry_date, valuation_date), dtype=float)
        self.sqrt_t = np.sqrt(self.t)
        self.is_expired = bool(np.all(self.t == 0.0))
        self.term_discount = np.asarray(term_discount, dtype=float)
        self.expiry_date = expiry_date
        self.valuation_date = valuation_date
        self.currency_pair = currency_pair
        if delta_convention not in ("spot", "forward"):
            raise ValueError(
                "delta_convention must be 'spot' or 'forward', got %r"
                % (delta_convention,)
            )
        self.delta_convention = delta_convention

        # Discounting runs to the option's *delivery* date (expiry plus the
        # pair's spot lag on the joint calendar) at a *simple* rate on the term
        # currency's day-count basis. Resolving that needs the pair, so when no
        # pair is supplied -- and when the conventions are unavailable or the
        # pair is unknown -- fall back to the continuously compounded factor to
        # expiry on an ACT/365 clock.
        self.settlement_discounting = False
        if term_discount_factor is None:
            resolved = None
            if currency_pair is not None:
                resolved = safe_option_discount(
                    currency_pair, expiry_date, valuation_date, term_discount)
            if resolved is None:
                self.term_discount_factor = np.exp(-self.term_discount * self.t)
            else:
                self.term_discount_factor = np.asarray(
                    resolved.discount_factor, dtype=float)
                self.settlement_discounting = True
                self.delivery_date = resolved.delivery_date
                self.discount_year_fraction = resolved.year_fraction
                self.discount_day_count = resolved.day_count
                if carry_rate is None:
                    carry_rate = resolved.carry_rate
        else:
            factor = np.asarray(term_discount_factor, dtype=float)
            if not np.all(np.isfinite(factor)) or np.any(factor <= 0.0):
                raise ValueError("term_discount_factor must be positive and finite")
            self.term_discount_factor = factor
            self.settlement_discounting = True
        # Rate at which the discount factor decays. Under the library's
        # continuous default this is the quoted rate; under simple discounting
        # it is r / (1 + r * tau), which stays bounded by the quoted rate at
        # every tenor. Theta consumes a rate rather than a factor, so it must
        # be given the true carry rather than inferring one from the factor.
        if carry_rate is None:
            self.term_carry_rate = self.term_discount
        else:
            carry = np.asarray(carry_rate, dtype=float)
            if not np.all(np.isfinite(carry)):
                raise ValueError("carry_rate must be finite")
            self.term_carry_rate = carry
        # FX forward parity: F/S = D_base / D_term.
        self.base_discount_factor = self.term_discount_factor * self.f / self.S
        self.CallPut = cp_flag(CallPut)
        self.BuySell = pos_flag(BuySell)
        self.Notional = np.asarray(Notional, dtype=float)
        self.deltaBase = deltaBase

        self.nS = self.Notional / self.S
        self.v_sqrt_t = self.v * self.sqrt_t
        with np.errstate(divide='ignore', invalid='ignore'):
            self.d1 = np.where(self.v_sqrt_t > 0,
                               (np.log(self.f / self.K) + 0.5 * self.v ** 2 * self.t) / self.v_sqrt_t,
                               0.0)

            self.d2 = self.d1 - self.v_sqrt_t

        # ``scipy.stats.norm`` builds distribution objects and normalizes
        # arguments on every call. Direct special-function ufuncs avoid that
        # overhead for the many scalar Black-Scholes evaluations in VV.
        self.cdf_d1 = ndtr(self.CallPut * self.d1)
        self.cdf_d2 = ndtr(self.CallPut * self.d2)
        self.pdf_d1 = np.exp(-0.5 * self.d1 * self.d1) * INV_SQRT_TWO_PI
        self.pdf_d2 = np.exp(-0.5 * self.d2 * self.d2) * INV_SQRT_TWO_PI
        self.intrinsic = self.f * self.cdf_d1 - self.K * self.cdf_d2

    # ------------------- Pricing & Greeks -------------------
    def price(self):
        """Return the signed option value in term (quote) currency."""
        if self.is_expired:
            intrinsic = np.maximum(self.CallPut * (self.S - self.K), 0.0)
            return self.BuySell * self.Notional * intrinsic
        discounted_intrinsic = (
            self.S * self.base_discount_factor * self.cdf_d1
            - self.K * self.term_discount_factor * self.cdf_d2
        )
        return self.BuySell * self.CallPut * self.Notional * discounted_intrinsic

    #: Supported premium quotation styles, see :meth:`premium`.
    PREMIUM_QUOTES = PREMIUM_QUOTES

    def premium(self, quote="term_cash"):
        """Re-express :meth:`price` in a market premium quotation convention.

        ``price()`` is always a cash amount in the **term** currency for a
        ``Notional`` expressed in the **base** currency; this method only
        changes the units it is reported in, never the value of the trade.

        =============  ==================================  =========================
        ``quote``      Meaning                             Formula
        =============  ==================================  =========================
        ``term_cash``  Term-currency cash (the default)    ``V``
        ``base_cash``  Base-currency cash                  ``V / S``
        ``term_pips``  Term pips per unit of base notional ``V / N * 1e4``
        ``base_pips``  Base pips per unit of term notional ``V / S / (N * K) * 1e4``
        ``pct_term``   Percent of the term notional        ``V / (N * K)``
        ``pct_base``   Percent of the base notional        ``V / S / N``
        =============  ==================================  =========================

        Two conventions are easy to get wrong.  The term notional is
        ``N * K``, **not** ``N * S``: percent-of-term is struck-notional based,
        and using spot overstates it by ``K / S``.  Cash-to-cash conversion
        uses spot, whereas the pips and percent forms are struck-notional
        based and therefore involve ``K``.

        The sign convention of ``price()`` is preserved, so a sold option
        quotes a negative premium.  Conversions are shared with the portfolio
        layer via :mod:`helper_functions.premium_quotes`.
        """
        return convert_premium_quote(
            self.price(), quote, self.S, strike=self.K, notional=self.Notional
        )

    def delta(self):
        """Return an FX quote delta.

        Two independent choices are involved, and they are often conflated.

        *Premium adjustment* (``deltaBase``) asks whether the premium is paid
        in the base currency. If it is, the premium itself carries base-currency
        exposure that nets against the option's, so the hedge is smaller:

          - unadjusted:        ``N(d1)`` for calls, ``-N(-d1)`` for puts
          - premium-adjusted:  ``K/F*N(d2)`` for calls, ``-K/F*N(-d2)`` for puts

        *Spot versus forward* (``delta_convention``) asks what the hedge is
        denominated in. The expressions above are **forward** deltas -- the
        hedge in forward contracts to the option's delivery date. Hedging in
        the spot market instead requires discounting by the base-currency
        factor, because a spot trade settles immediately while the option's
        exposure is to the forward:

          ``spot delta = base discount factor * forward delta``

        The market -- and Bloomberg -- quotes **spot** delta for tenors out to
        roughly one year, which is why that is the default here. The difference
        is not cosmetic: at six months with a base rate near 3.4% the forward
        delta overstates the spot hedge by about 1.7% of notional, so a 50-delta
        EURUSD option on 100m reads roughly 0.9m too large.

        Longer-dated books conventionally switch to forward delta; pass
        ``delta_convention='forward'`` for that, and note that the strike
        solvers in :mod:`helper_functions.VannaVolga_function` pin the
        convention explicitly so that changing this default cannot silently
        move the quoted pillar strikes.

        The result is signed for puts and scaled by ``Notional``.
        """
        if self.is_expired:
            moneyness = self.CallPut * (self.S - self.K)
            delta = np.where(
                moneyness > 0.0,
                self.CallPut,
                np.where(moneyness == 0.0, 0.5 * self.CallPut, 0.0),
            )
            return self.BuySell * delta * self.Notional
        if self.deltaBase:
            delta = self.CallPut * self.K / self.f * self.cdf_d2
        else:
            delta = self.CallPut * self.cdf_d1
        if self.delta_convention == "spot":
            delta = delta * self.base_discount_factor
        return self.BuySell * delta * self.Notional

    def vega(self):
        """Return base-currency price sensitivity per one volatility point."""
        if self.is_expired:
            return np.zeros_like(self.S, dtype=float)
        return self.term_discount_factor * _vega_numba(
            self.pdf_d1, self.f, self.sqrt_t, self.nS, self.BuySell
        )

    def gamma(self):
        """Return base-currency delta sensitivity to a 1% relative spot move.

        This is the derivative of whatever :meth:`delta` reports, so it picks
        up the same spot/forward scaling. The base discount factor is held
        fixed across the bump, matching the market convention that a delta
        ladder re-prices the option, not the discount curve.
        """
        if self.is_expired:
            return np.zeros_like(self.S, dtype=float)
        if self.deltaBase:
            delta = self.CallPut * self.K / self.f * self.cdf_d2
            dd2_dS = 1.0 / (self.f * self.v * self.sqrt_t)
            d_delta_dS = self.Notional * (
                -self.CallPut * self.K * self.cdf_d2 / self.f**2
                + self.K * self.pdf_d2 * dd2_dS / self.f
            )
        else:
            d_delta_dS = (
                self.Notional * self.pdf_d1 / (self.f * self.v * self.sqrt_t)
            )
        if self.delta_convention == "spot":
            # Spot delta is ``base_DF * forward delta`` and the base factor is
            # itself spot-dependent here, since ``base_DF = term_DF * F / S``
            # with the forward points held fixed across the bump (the market
            # convention for a spot ladder). Differentiating the product keeps
            # gamma the exact derivative of the delta actually reported.
            forward_delta = (
                self.CallPut * self.K / self.f * self.cdf_d2
                if self.deltaBase else self.CallPut * self.cdf_d1
            )
            d_base_df_dS = -self.term_discount_factor * self.fwd_pts / self.S**2
            d_delta_dS = (
                d_delta_dS * self.base_discount_factor
                + self.Notional * forward_delta * d_base_df_dS
            )
        # Delta is already a base-currency amount, so scaling by ``S * 0.01``
        # gives the delta change for a 1% relative spot move.
        return self.BuySell * d_delta_dS * self.S * 0.01

    def volga(self):
        """Return the derivative of base-currency point vega per vol point."""
        if self.is_expired:
            return np.zeros_like(self.S, dtype=float)
        return _volga_numba(self.vega(), self.d1, self.d2, self.v)

    def vanna(self):
        """Return the base-currency point-vega change for a 1% spot move."""
        if self.is_expired:
            return np.zeros_like(self.S, dtype=float)
        return _vanna_numba(
            self.vega(), self.f, self.S, self.d1, self.v, self.sqrt_t
        )

    def theta(self):
        """Return one business day of base-currency decay (252-day year)."""
        if self.is_expired:
            return np.zeros_like(self.S, dtype=float)
        base_rate = self.term_carry_rate - np.log(self.f / self.S) / self.t
        calendar_theta = (
            -self.S * self.base_discount_factor * self.pdf_d1 * self.v / (2 * self.sqrt_t)
            + self.CallPut * (
                base_rate * self.S * self.base_discount_factor * self.cdf_d1
                - self.term_carry_rate * self.K * self.term_discount_factor * self.cdf_d2
            )
        )
        return self.BuySell * self.nS * calendar_theta / 252.0

    def d2V_dK2(self):
        """Return the discounted second strike derivative per unit notional.

        ``d2C/dK2 = D_term * n(d2) / (K * sigma * sqrt(t))``, identical for
        calls and puts (put-call parity is linear in ``K``). It is unsigned
        and unscaled by ``Notional``: it describes the market, not the trade.
        """
        if self.is_expired:
            raise ValueError("The strike density is undefined at expiry")
        return self.term_discount_factor * self.risk_neutral_density()

    def risk_neutral_density(self):
        """Return the Breeden-Litzenberger density of the spot at expiry.

        ``q(K) = (1 / D_term) * d2C/dK2 = n(d2) / (K * sigma * sqrt(t))``,
        the lognormal density at ``K`` under the term-currency forward
        measure. This uses this object's single (flat) volatility. For the
        density implied by a smile, use
        :func:`pricing_utils.breeden_litzenberger.smile_density`, which also
        captures the ``dsigma/dK`` terms a flat volatility leaves out.
        """
        if self.is_expired:
            raise ValueError("The strike density is undefined at expiry")
        return self.pdf_d2 / (self.K * self.v_sqrt_t)

    def vega_volga_risk(self):
        """Return the sum of point vega and point-volga."""
        return self.vega() + self.volga()
"""FX Black/Garman-Kohlhagen style pricing and analytic reference Greeks."""
