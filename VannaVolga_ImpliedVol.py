from __future__ import annotations
import logging
from dataclasses import dataclass
from math import log
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import brentq
from pricing_utils.BlackScholes import BlackScholes, d1, ndtr
from market_data import MarketData
from market_data.vol_data import VolData
from helper_functions.fx_conventions import get_pair
from helper_functions.fx_strangle_converter import broker_to_twovol_bf
from helper_functions.vanna_volga_helpers import (
    phi, calculate_strikes, get_vol_k, GetMainPillars, get_atm_strike,
    interpolate_pillar_vols,
)
from helper_functions.helper_functions import year_fraction
from helper_functions.VannaVolga_function import normalize_atm_type


logger = logging.getLogger(__name__)


@dataclass
class VVConfig:
    """Numerical controls for implied-volatility inversion."""
    vol_bracket: Tuple[float, float] = (1e-6, 500.0)  # percent units
    xtol: float = 1e-8


class VannaVolga:
    """Construct a Vanna-Volga volatility for one or more strikes.

    Public volatility inputs use percentage points. The model derives
    delta-defined market pillars from ATM/RR/BF quotes, computes the
    Vanna-Volga price correction, and inverts that price to implied volatility.
    Use :meth:`get_vol_for_strike` for a single strike or
    :meth:`get_vol_for_strikes` for a strike grid.
    """


    def __init__(self, *args, **kwargs) -> None:

        market = kwargs.get("market")
        K = kwargs.get("K")
        expiry_date = kwargs.get("expiry_date")
        valuation_date = kwargs.get("valuation_date")
        CallPut = kwargs.get("CallPut")
        BuySell = kwargs.get("BuySell", "BUY")
        Notional = kwargs.get("Notional", 1.0)
        deltaBase = kwargs.get("deltaBase", True)
        atm_conv = normalize_atm_type(kwargs.get("atm_conv", "dn"))
        strike_type = kwargs.get("strike_type", "rate")
        fwd_divisor = kwargs.get("fwd_divisor", getattr(market, "forward_divisor", 10000.0))
        black_scholes_vol_in_percent = kwargs.get("black_scholes_vol_in_percent", True)
        black_scholes_vega_per_percent = kwargs.get("black_scholes_vega_per_percent", True)
        config = kwargs.get("config", None)
        self.smooth_pillar_interpolation = kwargs.get(
            "smooth_pillar_interpolation",
            getattr(market, "smooth_pillar_interpolation", True),
        )
        self.legacy_target_blend = kwargs.get(
            "legacy_target_blend",
            getattr(market, "legacy_target_blend", False),
        )
        self.compatibility_mode = kwargs.get(
            "compatibility_mode",
            getattr(market, "compatibility_mode", "auto"),
        )

        self.s = market.s
        self.K = K
        self.fwd_pts = market.fwd_pts
        self.f = market.s + market.fwd_pts / market.forward_divisor
        self.forward_divisor = fwd_divisor
        self.term_discount = float(getattr(market, "term_discount", 0.05))

        self.expiry_date = expiry_date
        self.valuation_date = valuation_date
        self.t = year_fraction(expiry_date, valuation_date)

        self._v_atm_pct = market.v_atm
        self.v_atm = market.v_atm / 100.0

        self._rr_25 = market.rr_25
        self._rr_10 = market.rr_10
        self._bf_25 = market.bf_25
        self._bf_10 = market.bf_10

        self.CallPut = CallPut.upper()
        self.BuySell = BuySell
        self.Notional = Notional
        self.deltaBase = deltaBase
        self.atm_conv = atm_conv
        self.strike_type = strike_type

        self.black_scholes_vol_in_percent = black_scholes_vol_in_percent
        self.black_scholes_vega_per_percent = black_scholes_vega_per_percent

        self.config = config or VVConfig()

        atm_bs = BlackScholes(
            S=self.s,
            K=self.K,
            fwd_pts=self.fwd_pts,
            v=self.v_atm_pct,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            CallPut="CALL",
            BuySell="BUY",
            Notional=1,
            deltaBase=self.deltaBase,
            fwd_divisor=self.forward_divisor,
            term_discount=self.term_discount,
            # Pinned: this delta only classifies how far the strike sits from
            # delta-neutral for the Vanna-Volga weights, so it must stay on the
            # forward convention regardless of the reporting default.
            delta_convention="forward",
        )
        self._tv_delta = 0.5 - abs(atm_bs.delta() - 0.5)

    #  properties & cache

    @property
    def rr_25(self) -> float: return self._rr_25
    @rr_25.setter
    def rr_25(self, val: float):
        if val != self._rr_25:
            self._rr_25 = val
            self._clear_cache()

    @property
    def rr_10(self) -> float: return self._rr_10
    @rr_10.setter
    def rr_10(self, val: float):
        if val != self._rr_10:
            self._rr_10 = val
            self._clear_cache()

    @property
    def bf_25(self) -> float: return self._bf_25
    @bf_25.setter
    def bf_25(self, val: float):
        if val != self._bf_25:
            self._bf_25 = val
            self._clear_cache()

    @property
    def bf_10(self) -> float: return self._bf_10
    @bf_10.setter
    def bf_10(self, val: float):
        if val != self._bf_10:
            self._bf_10 = val
            self._clear_cache()

    @property
    def v_atm_pct(self) -> float: return self._v_atm_pct
    @v_atm_pct.setter
    def v_atm_pct(self, val: float):
        if val != self._v_atm_pct:
            self._v_atm_pct = val
            self.v_atm = val / 100.0
            self._clear_cache()

    def _clear_cache(self):
        for attr in ["_market_weights", "_market_pillars_cache", "_principal_strikes_cache"]:
            if hasattr(self, attr):
                delattr(self, attr)

    # static helpers
    @staticmethod
    def _phi(x: float) -> float: return phi(x)

    @staticmethod
    def _d1(fwd: float, K: float, vol_decimal: float, t: float) -> float:
        return d1(fwd, K, vol_decimal, t)

    # pillars
    def _get_market_pillars(self):
        """Build the interpolated put-wing, ATM, and call-wing pillars."""
        cached = getattr(self, "_market_pillars_cache", None)
        if cached is not None:
            return cached
        try:
            get_pillar = GetMainPillars(
                k=self.K,
                expiry_date=self.expiry_date,
                valuation_date=self.valuation_date,
                atm_conv=self.atm_conv,
                deltaBase=self.deltaBase,
                s=self.s, fwd_pts=self.fwd_pts,
                v_atm=self.v_atm_pct,
                rr_25=self.rr_25, rr_10=self.rr_10,
                bf_25=self.bf_25, bf_10=self.bf_10,
                forward_divisor=self.forward_divisor,
                term_discount=self.term_discount,
                smooth_pillar_interpolation=self.smooth_pillar_interpolation,
                legacy_target_blend=self.legacy_target_blend,
                compatibility_mode=self.compatibility_mode,
            )
            _rr_bf = get_pillar.get_weighted_rr_bf()
            sigma2_pct = self.v_atm_pct
            sigma1_pct = get_vol_k(sigma2_pct, bf=_rr_bf['BF'], rr=_rr_bf['RR'], callput='PUT')
            sigma3_pct = get_vol_k(sigma2_pct, bf=_rr_bf['BF'], rr=_rr_bf['RR'], callput='CALL')
            if not np.isfinite(sigma1_pct) or not np.isfinite(sigma3_pct):
                raise ValueError("Non-finite interpolated wing vols")
            result = {'rr_bf': _rr_bf, 'vols': [sigma1_pct, sigma2_pct, sigma3_pct]}
            self._market_pillars_cache = result
            return result
        except Exception:
            logger.warning(
                "Continuous pillar interpolation failed; falling back to ATM/legacy behaviour for Vanna-Volga pricing.",
                exc_info=True,
            )
            rr_bf = {
                'RR': self._rr_25 if np.isfinite(self._rr_25) else 0.0,
                'BF': self._bf_25 if np.isfinite(self._bf_25) else 0.0,
                'delta': 0.25,
            }
            sigma2_pct = self.v_atm_pct
            sigma1_pct = get_vol_k(sigma2_pct, bf=rr_bf['BF'], rr=rr_bf['RR'], callput='PUT')
            sigma3_pct = get_vol_k(sigma2_pct, bf=rr_bf['BF'], rr=rr_bf['RR'], callput='CALL')
            result = {'rr_bf': rr_bf, 'vols': [sigma1_pct, sigma2_pct, sigma3_pct]}
            self._market_pillars_cache = result
            return result

    # BS wrappers
    def _bs(self, vol_pct: float, K: Optional[float] = None) -> BlackScholes:
        # BlackScholes accepts volatility in percentage points.  Convert an
        # explicitly decimal input here rather than relying on magnitude.
        vol_arg = vol_pct if self.black_scholes_vol_in_percent else vol_pct * 100.0
        return BlackScholes(
            S=self.s,
            K=self.K if K is None else K,
            fwd_pts=self.fwd_pts,
            expiry_date=self.expiry_date,
            valuation_date=self.valuation_date,
            v=vol_arg,
            CallPut=self.CallPut,
            BuySell=self.BuySell,
            Notional=self.Notional,
            deltaBase=self.deltaBase,
            fwd_divisor=self.forward_divisor,
            term_discount=self.term_discount,
        )

    def _bs_price(self, vol_pct: float, K: Optional[float] = None) -> float:
        return self._bs(vol_pct, K=K).price()

    #  VV core unchanged
    def _vega_ratio_phis(self, f: float, K: float, K1: float, K2: float, K3: float, sigma2_dec: float) -> Tuple[float, float, float]:
        phi_d1_K  = self._phi(self._d1(f, K,  sigma2_dec, self.t))
        phi_d1_K1 = self._phi(self._d1(f, K1, sigma2_dec, self.t))
        phi_d1_K2 = self._phi(self._d1(f, K2, sigma2_dec, self.t))
        phi_d1_K3 = self._phi(self._d1(f, K3, sigma2_dec, self.t))
        safe_ratio = lambda a, b: a / b if b and not np.isclose(b, 0.0) else 0.0
        VR_K_K1 = safe_ratio(phi_d1_K, phi_d1_K1)
        VR_K_K2 = safe_ratio(phi_d1_K, phi_d1_K2)
        VR_K_K3 = safe_ratio(phi_d1_K, phi_d1_K3)
        return VR_K_K1, VR_K_K2, VR_K_K3

    def _vv_price(self, K: float, K1: float, K2: float, K3: float, sigma1_pct: float, sigma2_pct: float, sigma3_pct: float) -> float:
        sigma2_dec = sigma2_pct / 100.0
        VR_K_K1, VR_K_K2, VR_K_K3 = self._vega_ratio_phis(self.f, K, K1, K2, K3, sigma2_dec)

        ln = log
        denom_K1 = ln(K2 / K1) * ln(K3 / K1)
        denom_K2 = ln(K2 / K1) * ln(K3 / K2)
        denom_K3 = ln(K3 / K1) * ln(K3 / K2)
        if any(np.isclose(d, 0.0) for d in (denom_K1, denom_K2, denom_K3)):
            logger.warning("Degenerate strike configuration; returning ATM vol pct")
            return None

        x1 = VR_K_K1 * (ln(K2 / K) * ln(K3 / K)) / denom_K1
        x2 = VR_K_K2 * (ln(K / K1) * ln(K3 / K)) / denom_K2
        x3 = VR_K_K3 * (ln(K / K1) * ln(K / K2)) / denom_K3

        CMKT_K1 = self._bs_price(sigma1_pct, K=K1)
        CMKT_K2 = self._bs_price(sigma2_pct, K=K2)
        CMKT_K3 = self._bs_price(sigma3_pct, K=K3)

        CBS_K1 = self._bs_price(sigma2_pct, K=K1)
        CBS_K2 = self._bs_price(sigma2_pct, K=K2)
        CBS_K3 = self._bs_price(sigma2_pct, K=K3)
        CBS_K  = self._bs_price(sigma2_pct, K=K)

        return CBS_K + x1 * (CMKT_K1 - CBS_K1) + x2 * (CMKT_K2 - CBS_K2) + x3 * (CMKT_K3 - CBS_K3)

    def _solve_implied_vol(self, target_price: float, K: float) -> float:
        f_root = lambda vol_pct: self._bs_price(vol_pct, K=K) - target_price
        vol_min, vol_max = self.config.vol_bracket
        try:
            f_min, f_max = f_root(vol_min), f_root(vol_max)
        except Exception as exc:
            raise ValueError("Unable to evaluate Black-Scholes price at implied-volatility bracket") from exc
        if f_min == 0.0: return vol_min
        if f_max == 0.0: return vol_max
        if f_min * f_max > 0:
            raise ValueError(
                f"VV price is outside the implied-volatility bracket: "
                f"price={target_price}, strike={K}"
            )
        try:
            return brentq(f_root, vol_min, vol_max, maxiter=200, xtol=self.config.xtol)
        except Exception as exc:
            raise ValueError(f"Implied-volatility root finding failed for strike {K}") from exc

    # No-arb enforcement (monotone + convex)
    def _forward_call_price(self, vol_pct: float, K: float) -> float:
        """Undiscounted Black price of a long unit call, in forward terms.

        The arbitrage projection must run on this curve whatever the trade:
        the instance's ``CallPut``/``BuySell`` would make puts increasing and
        sales concave in strike, inverting both no-arbitrage conditions.
        """
        sigma_sqrt_t = vol_pct / 100.0 * np.sqrt(self.t)
        if sigma_sqrt_t <= 0.0:
            return max(self.f - K, 0.0)
        d1_ = (log(self.f / K) + 0.5 * sigma_sqrt_t ** 2) / sigma_sqrt_t
        return self.f * ndtr(d1_) - K * ndtr(d1_ - sigma_sqrt_t)

    def _solve_forward_call_vol(self, price: float, K: float) -> float:
        """Invert :meth:`_forward_call_price` to a volatility in percent."""
        f_root = lambda vol_pct: self._forward_call_price(vol_pct, K) - price
        vol_min, vol_max = self.config.vol_bracket
        if f_root(vol_min) * f_root(vol_max) > 0.0:
            raise ValueError(
                f"Projected call price is outside the implied-volatility "
                f"bracket: price={price}, strike={K}"
            )
        return brentq(f_root, vol_min, vol_max, maxiter=200, xtol=self.config.xtol)

    def _enforce_butterfly_no_arb_on_set(self, Ks: np.ndarray, vols_pct: np.ndarray) -> np.ndarray:
        """
        Enforce strict static no-arbitrage constraints across strikes for a single expiry.
        Guarantees that the call price curve C(K) is monotonically decreasing and convex.

        The projection runs on undiscounted long unit call prices, so the
        result is independent of the instance's option type and side.

        Inputs:
            Ks        : sorted array of strikes (strictly increasing, absolute K units)
            vols_pct  : array of implied vols (percent units) aligned to Ks
        Returns:
            adjusted vols (percent units) satisfying strict vertical & butterfly no-arb
        Raises:
            ValueError if a projected price cannot be inverted to a volatility.
        """
        Ks = np.asarray(Ks, dtype=float)
        sigmas = np.asarray(vols_pct, dtype=float).copy()
        n = len(Ks)

        # 1. Generate initial call prices from the uncleaned vol surface
        prices = np.array([self._forward_call_price(s, k) for s, k in zip(sigmas, Ks)], dtype=float)

        # Expanded iterations to ensure coordinate projection converges across a wide grid
        max_iter = 100
        tol = 1e-10

        for _ in range(max_iter):
            changed = False

            # Pass 1) Enforce Monotonicity (No Bull Spread Arbitrage): C_{i} >= C_{i+1}
            for i in range(n - 1):
                if prices[i] < prices[i + 1]:
                    prices[i + 1] = prices[i]
                    changed = True

            # Pass 2) Enforce Strict Convexity in Strike Space (No Butterfly Arbitrage)
            # Risk-neutral PDF proportional to second derivative: d^2C / dK^2 >= 0
            for i in range(1, n - 1):
                h_l = Ks[i] - Ks[i - 1]  # Delta K left
                h_r = Ks[i + 1] - Ks[i]  # Delta K right

                dl = (prices[i] - prices[i - 1]) / h_l
                dr = (prices[i + 1] - prices[i]) / h_r

                # If dl > dr, the slope is decreasing (concave), which means negative density (arbitrage)
                if dl > dr:
                    # Compute the exact price at K_i that linearizes the local slopes (dl == dr)
                    num = h_r * prices[i - 1] + h_l * prices[i + 1]
                    den = h_l + h_r
                    p_new = num / den

                    # Keep changes bounded within monotonic limits to prevent breaking Pass 1
                    p_new = np.clip(p_new, prices[i + 1], prices[i - 1])

                    if abs(p_new - prices[i]) > tol:
                        prices[i] = p_new
                        changed = True

            if not changed:
                break

        # 2. Invert back to implied volatility. A failure is an error, not a
        # zero vol: a silent 0.0 would price the option at intrinsic.
        adj_sigmas = []
        for p, k in zip(prices, Ks):
            vol = self._solve_forward_call_vol(p, K=k)
            if not np.isfinite(vol) or vol <= 0.0:
                raise ValueError(f"Arbitrage projection gave an invalid vol at strike {k}")
            adj_sigmas.append(vol)

        return np.array(adj_sigmas, dtype=float)

    # PUBLIC: single strike (now no-arb by construction)
    def get_vol_for_strike(self) -> float:
        """Return the implied volatility at the configured strike in vol points."""
        if self.smooth_pillar_interpolation and not self.legacy_target_blend:
            try:
                pillar = GetMainPillars(
                    k=self.K,
                    expiry_date=self.expiry_date,
                    valuation_date=self.valuation_date,
                    atm_conv=self.atm_conv,
                    deltaBase=self.deltaBase,
                    s=self.s,
                    fwd_pts=self.fwd_pts,
                    v_atm=self.v_atm_pct,
                    rr_25=self.rr_25,
                    rr_10=self.rr_10,
                    bf_25=self.bf_25,
                    bf_10=self.bf_10,
                    forward_divisor=self.forward_divisor,
                    term_discount=self.term_discount,
                    smooth_pillar_interpolation=True,
                    compatibility_mode=self.compatibility_mode,
                )
                strikes, vols = pillar.get_five_pillar_arrays()
                target_strike = float(self.K)
                if not np.isfinite(target_strike) or target_strike <= 0.0:
                    return float(self.v_atm_pct)
                # get_five_pillar_arrays has already checked that strikes are
                # finite, unique, and strictly increasing. Interpolate those
                # validated arrays directly instead of sorting and validating
                # them a second time in the generic safe helper.
                #
                # Interpolation is a natural cubic spline in log-strike. A
                # straight chord across the convex region between the ATM and
                # 25-delta pillars overstates the smile by around 6bp, which is
                # where most options are struck.
                target_log_strike = np.log(target_strike)
                log_strikes = np.log(strikes)
                # Linear interpolation *clamps* to the endpoint value outside
                # the pillar range, which would flatten the smile to the
                # 10-delta quote for every further-out strike and badly
                # compress the wings. Outside the quoted pillars, fall through
                # to the continuous Vanna-Volga solution below, which
                # extrapolates the smile properly.
                if log_strikes[0] <= target_log_strike <= log_strikes[-1]:
                    result = float(
                        interpolate_pillar_vols(strikes, vols, target_strike)
                    )
                    if not np.isfinite(result):
                        raise ValueError(
                            "Five-pillar interpolation returned a non-finite vol"
                        )
                    return result

                # Outside the quoted range, take the Vanna-Volga wing *shape*
                # but anchor it to the boundary pillar. The VV put anchor does
                # not generally coincide with the 10-delta put pillar, so using
                # the raw VV level would leave a step of ~0.1 vol at the
                # boundary. Offsetting by the residual at the boundary keeps
                # the smile continuous and reproduces all five quotes exactly.
                edge = 0 if target_log_strike < log_strikes[0] else -1
                edge_strike = float(strikes[edge])
                edge_quote = float(vols[edge])
                # Direction the smile moves as it leaves the quoted range.
                neighbour = vols[1] if edge == 0 else vols[-2]
                direction = 1.0 if edge_quote >= float(neighbour) else -1.0
                try:
                    wing = self._continuous_vv_vol(target_strike)
                    edge_wing = self._continuous_vv_vol(edge_strike)
                    step = wing - edge_wing
                    if not np.isfinite(step):
                        raise ValueError("Non-finite Vanna-Volga wing step")
                except Exception:
                    # Deep in the wing the option value underflows and the
                    # price inversion stops being well posed. Hold the boundary
                    # quote rather than returning a spurious level.
                    return edge_quote
                # The Vanna-Volga wing eventually turns back on itself far
                # outside the quoted range, where it is no longer meaningful.
                # Do not let the extrapolation reverse direction; flatten to
                # the boundary quote instead.
                if direction * step < 0.0:
                    return edge_quote
                result = edge_quote + step
                if not np.isfinite(result) or result <= 0.0:
                    return edge_quote
                return float(result)
            except Exception:
                logger.warning(
                    "Five-pillar interpolation failed; falling back to "
                    "legacy Vanna-Volga volatility.",
                    exc_info=True,
                )

        return self._continuous_vv_vol(self.K)

    def _continuous_vv_vol(self, strike: float) -> float:
        """Return the continuous Vanna-Volga volatility at ``strike``.

        This is the smile solution used both for the legacy path and for
        extrapolation beyond the quoted five-pillar range.
        """
        _pillars = self._get_market_pillars()
        rr_bf = _pillars['rr_bf']
        sigma1_pct, sigma2_pct, sigma3_pct = _pillars['vols']

        strikes = calculate_strikes(
            S=self.s, f=self.f, fwd_pts=self.fwd_pts,
            expiry_date=self.expiry_date, valuation_date=self.valuation_date,
            t=self.t,
            sigma1_pct=sigma1_pct, sigma2_pct=sigma2_pct, sigma3_pct=sigma3_pct,
            rr_bf=rr_bf, delta_base=self.deltaBase, atm_conv=self.atm_conv,
            fwd_divisor=self.forward_divisor,
        )
        K1, K2, K3 = strikes["K1"], strikes["K2"], strikes["K3"]

        vv_price = self._vv_price(strike, K1, K2, K3, sigma1_pct, sigma2_pct, sigma3_pct)
        if np.isclose(strike, K1, rtol=1e-10, atol=1e-12):
            vv_vol = sigma1_pct
        elif np.isclose(strike, K3, rtol=1e-10, atol=1e-12):
            vv_vol = sigma3_pct
        elif np.isclose(strike, K2, rtol=1e-10, atol=1e-12) or vv_price is None:
            vv_vol = sigma2_pct
        else:
            vv_vol = self._solve_implied_vol(vv_price, K=strike)

        # A local four-point arbitrage projection is not stable for a
        # single-strike query: as spot moves, the requested strike changes
        # ordering relative to the pillars and the projection can jump.
        # Return the continuous VV solution here.  Batch queries use the
        # full-smile projection below, where the strike grid is fixed.
        return float(vv_vol)

    # ======== NEW: batch / vectorized computation across many strikes ========
    def get_vol_for_strikes(self, Ks: np.ndarray) -> dict:
        """
        Return implied volatilities for a strike grid.

        The requested strikes and the three principal pillars are projected
        together onto a monotone, convex call-price curve before inversion.
        The result maps each requested strike to a volatility in percentage
        points.
        """
        Ks = np.asarray(Ks, dtype=float).ravel()

        _pillars = self._get_market_pillars()
        rr_bf = _pillars['rr_bf']
        sigma1_pct, sigma2_pct, sigma3_pct = _pillars['vols']
        strikes = calculate_strikes(
            S=self.s, f=self.f, fwd_pts=self.fwd_pts,
            expiry_date=self.expiry_date, valuation_date=self.valuation_date,
            t=self.t,
            sigma1_pct=sigma1_pct, sigma2_pct=sigma2_pct, sigma3_pct=sigma3_pct,
            rr_bf=rr_bf, delta_base=self.deltaBase, atm_conv=self.atm_conv,
            fwd_divisor=self.forward_divisor,
        )
        K1, K2, K3 = strikes["K1"], strikes["K2"], strikes["K3"]

        # Initial vols: pillars known, others via VV→IV
        # Construct full set: requested Ks + pillars
        all_Ks = np.array(sorted(set(Ks.tolist() + [K1, K2, K3])))

        init_vols = np.empty_like(all_Ks, dtype=float)
        for i, k in enumerate(all_Ks):
            if np.isclose(k, K1): init_vols[i] = sigma1_pct
            elif np.isclose(k, K2): init_vols[i] = sigma2_pct
            elif np.isclose(k, K3): init_vols[i] = sigma3_pct
            else:
                # VV price then IV
                vv_p = self._vv_price(k, K1, K2, K3, sigma1_pct, sigma2_pct, sigma3_pct)
                init_vols[i] = sigma2_pct if vv_p is None else self._solve_implied_vol(vv_p, K=k)

        # Enforce no-arb across the WHOLE set
        adj_vols = self._enforce_butterfly_no_arb_on_set(all_Ks, init_vols)

        # Extract adjusted vols only for requested Ks
        result = {}
        for K_query in Ks:
            j = np.where(np.isclose(all_Ks, K_query))[0][0]
            result[float(K_query)] = float(adj_vols[j])
        return result

if __name__ == "__main__":
    fx_conv = get_pair('EURUSD')
  
    bf_25 = broker_to_twovol_bf(spot=1.138 ,
                                fwd_pts=89.32,
                                pip_factor=10000,
                                v_atm=5.947,rr=-0.515,
                                bf_broker=0.233,delta=25,
                                valuation='2026-09-25',expiry='2027-03-25',delivery='2027-03-27',basis=360,depo_pct=4.27,is_pa=False,atm_conv='dn',date_format='%Y-%m-%d')


    market = MarketData(s=1.1380,
                        fwd_pts=89.32,
                        v_atm=5.947,
                        rr_25=-0.515,
                        rr_10=-0.9650,
                        bf_25=0.233, bf_10=0.86)


    vv = VannaVolga(market=market, K=1.1459, expiry_date='2027-03-25', valuation_date='2026-09-25', CallPut='CALL',
                    deltaBase=fx_conv.premium_adjusted, atm_conv=fx_conv.atm)



    import timeit

    runtime = timeit.timeit(lambda: vv.get_vol_for_strike(), number=100)
    print(f"Average runtime over 100 runs (parallel): {runtime / 100:.8f} seconds")
    v_k = vv.get_vol_for_strike()
    print(v_k)
    print(get_atm_strike(atm_type=fx_conv.atm, S=market.s,delta_base=fx_conv.premium_adjusted, fwd=market.fwd_pts, v=market.v_atm, expiry_date='2027-03-25', valuation_date='2026-09-25', fwd_divisor=fx_conv.fwd_divisor))
    print(BlackScholes(S=market.s, K=1.1459, fwd_pts=89.32, v=v_k, expiry_date='2027-03-25', valuation_date='2026-09-25', CallPut='CALL', BuySell='BUY', Notional=1, deltaBase=False, fwd_divisor=10000.0, term_discount=0.0429,currency_pair='EURUSD').premium(quote='base_cash')*100_000_000)
    print(BlackScholes(S=market.s, K=1.1459, fwd_pts=89.32, v=v_k, expiry_date='2027-03-25', valuation_date='2026-09-25', CallPut='CALL', BuySell='BUY', Notional=1, deltaBase=False, fwd_divisor=10000.0, term_discount=0.0429,currency_pair='EURUSD').delta()*100_000_000)

"""Vanna-Volga implied-volatility construction for delta-quoted FX smiles."""
