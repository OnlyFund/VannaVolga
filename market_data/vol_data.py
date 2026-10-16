import pandas as pd
import numpy as np
from pathlib import Path
from math import sqrt
from concurrent.futures import ThreadPoolExecutor
import warnings
from market_data.event_calendar import EventCalendar

PILLARS = ['PUT_10', 'PUT_25', 'ATM', 'CALL_25', 'CALL_10']


def _resolve_data_path(value, sibling_dir=None):
    """Resolve explicit paths first, then bundled project data for bare names."""
    path = Path(value).expanduser()
    if path.is_file():
        return path
    if not path.is_absolute():
        root = Path(__file__).resolve().parents[1]
        candidates = [root / path, root / path.name]
        if sibling_dir:
            candidates.append(root / sibling_dir / path.name)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Could not find data file {value!r}; checked the supplied path and bundled project data")


def get_vol_k(v_atm, bf, rr, callput):
    """Reconstruct a put or call wing volatility from ATM/RR/BF quotes."""
    cp = 1 if callput.upper() == "CALL" else -1
    return v_atm + bf + cp * rr / 2


class VolData:
    """Volatility data with event-adjusted broken-date interpolation.

    Quoted pillar vols are converted to total variance using the event-aware
    variance-day clock. Between pillars, total variance is interpolated in
    variance time and broken-date vols are annualized over calendar days.
    This is piecewise-linear total-variance interpolation, not the full
    monotone-convex Hagan-West algorithm. Negative bucket forward variances
    are reported and floored to zero when producing the diagnostic forward curve.
    """

    def __init__(self, data_file: str, valuation_date=None, calendar_csv='event_calendar.csv'):
        self.data_file = _resolve_data_path(data_file, "market_data/samples")
        self.df = pd.read_csv(self.data_file)
        required = {'tenor', 'expiry', 'v_atm', 'rr_25', 'bf_25', 'rr_10', 'bf_10'}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Volatility CSV is missing required columns: {', '.join(sorted(missing))}")
        self.df['expiry'] = pd.to_datetime(self.df['expiry'], errors='coerce')
        for col in ('v_atm', 'rr_25', 'bf_25', 'rr_10', 'bf_10'):
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        self.df = self.df.dropna(subset=['tenor', 'expiry', 'v_atm', 'rr_25', 'bf_25', 'rr_10', 'bf_10'])
        if self.df.empty:
            raise ValueError("Volatility data contains no valid rows")
        quote_values = self.df[['v_atm', 'rr_25', 'bf_25', 'rr_10', 'bf_10']].to_numpy(dtype=float)
        if not np.isfinite(quote_values).all():
            raise ValueError("Volatility quotes must all be finite")
        if (self.df['v_atm'] <= 0).any():
            raise ValueError("ATM volatilities must be greater than zero")
        if self.df['expiry'].duplicated().any():
            raise ValueError("Volatility CSV must have unique expiry dates")
        self.df.sort_values('expiry', inplace=True)
        self.df.set_index('tenor', inplace=True)

        self.valuation_date = pd.to_datetime(valuation_date).normalize() if valuation_date else None
        self.calendar = EventCalendar(calendar_csv)

        # Cache for vdays
        self._vdays_cache = {}

        # Build 5-point vols
        self.vols_5pt_df = self._compute_5pt_vols()

        if self.valuation_date is not None:
            # Add vdays and total variances
            self.add_vdays_and_total_vars(self.valuation_date)
            # Compute diagnostic bucket forwards from adjacent total variances.
            self._compute_bucket_forward_vols()

    def _get_vdays(self, start, end):
        key = (start, end)
        if key not in self._vdays_cache:
            self._vdays_cache[key] = self.calendar.get_var_days(start, end)
        return self._vdays_cache[key]

    def _compute_5pt_vols(self):
        df = self.df
        vols = pd.DataFrame({
            'PUT_10': get_vol_k(df['v_atm'], df['bf_10'], df['rr_10'], 'PUT'),
            'PUT_25': get_vol_k(df['v_atm'], df['bf_25'], df['rr_25'], 'PUT'),
            'ATM': df['v_atm'].values,
            'CALL_25': get_vol_k(df['v_atm'], df['bf_25'], df['rr_25'], 'CALL'),
            'CALL_10': get_vol_k(df['v_atm'], df['bf_10'], df['rr_10'], 'CALL'),
            'expiry': df['expiry']
        })
        wing_values = vols[PILLARS].to_numpy(dtype=float)
        if not np.isfinite(wing_values).all() or (wing_values <= 0).any():
            raise ValueError("Reconstructed 10D/25D pillar volatilities must be finite and positive")
        return vols

    def add_vdays_and_total_vars(self, valuation_date):
        vols = self.vols_5pt_df
        valuation_date = pd.to_datetime(valuation_date).normalize()
        if (vols['expiry'] < valuation_date).any():
            raise ValueError("All quoted expiries must be on or after valuation_date")
        vols['days'] = (vols['expiry'] - valuation_date).dt.days
        vols['vdays'] = vols['expiry'].apply(
            lambda d: self._get_vdays(valuation_date, pd.to_datetime(d))
        )
        if not np.isfinite(vols['vdays'].to_numpy(dtype=float)).all():
            raise ValueError("Computed variance days are not finite")
        if (vols['vdays'].diff().dropna() <= 0).any():
            raise ValueError("Expiry pillars must have strictly increasing variance days")
        if (vols['vdays'] < 0).any():
            raise ValueError("Expiry pillars must not precede the valuation date")
        for col in PILLARS:
            # Quotes are annualized over calendar time, so total variance is
            # accumulated in calendar-annualized units. The variance-day clock
            # is only the interpolation measure; see interpolate_vols.
            vols[f'total_var_{col}'] = (vols[col] ** 2) * vols['days']
        self.vols_5pt_df = vols

    def _compute_bucket_forward_vols(self):
        df = self.vols_5pt_df
        vdays = df['vdays'].values
        total_vars = df[[f'total_var_{p}' for p in PILLARS]].values

        fwd_vars = np.zeros_like(total_vars)
        fwd_vols = np.zeros_like(total_vars)

        for i in range(len(vdays)):
            previous_vday = 0.0 if i == 0 else vdays[i - 1]
            delta_vdays = max(vdays[i] - previous_vday, 1e-12)
            previous_var = 0.0 if i == 0 else total_vars[i - 1]
            raw_forward_variance = (total_vars[i] - previous_var) / delta_vdays
            if (raw_forward_variance < -1e-12).any():
                warnings.warn(
                    "Negative bucket forward variance found; flooring it to zero. "
                    "Check the input term structure for calendar arbitrage.",
                    RuntimeWarning,
                )
            fwd_vars[i] = np.maximum(raw_forward_variance, 0.0)
            fwd_vols[i] = np.sqrt(fwd_vars[i])

        for j, p in enumerate(PILLARS):
            df[f'FWDVOL_{p}'] = fwd_vols[:, j]

        self.vols_5pt_df = df

    def interpolate_vols(self, target_date):
        """
        Interpolate the five pillar vols at a target date.

        Total variance is linearly interpolated on the event-adjusted
        variance-day clock and annualized over elapsed calendar days.
        Returns the five pillar vols plus the accumulated variance days.
        """
        target_date = pd.to_datetime(target_date)
        if self.valuation_date is None:
            raise ValueError("valuation_date is required for expiry interpolation")
        vday_target = self._get_vdays(self.valuation_date, target_date)
        calendar_days_target = (target_date - self.valuation_date).days

        df = self.vols_5pt_df
        valid = df.dropna(subset=['expiry', 'vdays'])
        if len(valid) == 0:
            raise ValueError("No valid expiry pillars are available for interpolation")
        vdays_exp = valid['vdays'].values
        if vday_target <= 0:
            raise ValueError("target_date must be after valuation_date")
        if calendar_days_target <= 0:
            raise ValueError("target_date must be after valuation_date")
        if vday_target > vdays_exp[-1]:
            raise ValueError("target_date is outside the quoted expiry range")

        exact_idx = np.where(np.isclose(vdays_exp, vday_target))[0]
        if exact_idx.size:
            row = valid.iloc[exact_idx[0]]
            result = {p: row[p] for p in PILLARS}
            result['vday_to_expiry'] = vday_target
            result['target_date'] = target_date
            return pd.Series(result, name=target_date)

        # Include the valuation date as a zero-variance anchor. This supports
        # pre-first-pillar maturities and lets event variance enter only when
        # the maturity reaches the event date.
        if vday_target < vdays_exp[0]:
            idx = -1
            v0, v1 = 0.0, vdays_exp[0]
        else:
            idx = np.searchsorted(vdays_exp, vday_target) - 1
            if len(vdays_exp) < 2:
                raise ValueError("At least two expiry pillars are required for interpolation")
            idx = max(0, min(idx, len(vdays_exp) - 2))
            v0, v1 = vdays_exp[idx], vdays_exp[idx + 1]
        if v1 <= v0:
            raise ValueError("Expiry pillars have a non-increasing variance-time interval")

        result = {}
        for p in PILLARS:
            if idx == -1:
                tv0 = 0.0
                tv1 = valid.iloc[0][f'total_var_{p}']
            else:
                tv0 = valid.iloc[idx][f'total_var_{p}']
                tv1 = valid.iloc[idx + 1][f'total_var_{p}']

            # Linear interpolation in variance
            total_var_target = tv0 + (tv1 - tv0) * (vday_target - v0) / (v1 - v0)

            # Market vols are quoted over calendar maturities, so total variance
            # is held in calendar-annualized units and the broken-date quote is
            # annualized over elapsed calendar days. The event-weighted variance
            # clock only drives the interpolation weight, which produces the
            # intended pre-event discount when event variance days are not yet
            # included in the target maturity. Accumulating variance on the
            # variance-day clock here would rescale every quote by
            # sqrt(vdays / calendar days) instead of reproducing it at a pillar.
            vol_target = sqrt(total_var_target / calendar_days_target)

            result[p] = vol_target

        result['vday_to_expiry'] = vday_target
        result['target_date'] = target_date
        return pd.Series(result, name=target_date)

    def interpolate_vols_parallel(self, target_dates, max_workers=None):
        """
        Interpolate several target dates, optionally using worker threads.
        """
        if np.isscalar(target_dates):
            target_dates = [target_dates]

        target_dates = [pd.to_datetime(d) for d in target_dates]

        if max_workers and len(target_dates) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(self.interpolate_vols, target_dates))
        else:
            results = [self.interpolate_vols(d) for d in target_dates]

        return pd.DataFrame(results)

    def get_interpolated_atm_rr_bf(self, target_date):
        """Return interpolated ATM, RR, and BF quotes in MarketData units."""
        df = self.interpolate_vols(target_date).to_dict()
        v_atm = df['ATM']
        rr_25 = df['CALL_25'] - df['PUT_25']
        rr_10 = df['CALL_10'] - df['PUT_10']
        bf_25 = (df['CALL_25'] + df['PUT_25']) / 2 - v_atm
        bf_10 = (df['CALL_10'] + df['PUT_10']) / 2 - v_atm
        return dict(v_atm=v_atm, rr_25=rr_25, rr_10=rr_10,
                    bf_25=bf_25, bf_10=bf_10, vdays=df['vday_to_expiry'])


if __name__ == "__main__":
    vd = VolData('vol_data.csv', valuation_date='2025-08-29', calendar_csv='event_calendar.csv')

    # single interpolation
    interp = vd.interpolate_vols('2025-09-18')
    print("Interpolated 5pt (linear total variance):")
    print(interp)

    print("ATM/RR/BF + vdays:")
    print(vd.get_interpolated_atm_rr_bf('2025-09-18'))

    # batch example
    dates = ['2025-09-15', '2025-10-01', '2025-12-31']
    batch = vd.interpolate_vols_parallel(dates)
    print(batch)
