import pandas as pd
from pathlib import Path
from datetime import timedelta
import warnings

class EventCalendar:
    """
    Handle calendar-day and event weights for FX volatility calculations.

    Weekdays contribute 1.0, Saturday contributes 0.0, and Sunday contributes
    0.2 variance days. Event weights are additive and are counted on
    ``(start_date, end_date]`` while base weights use ``[start_date, end_date)``.
    """

    def __init__(self, csv_file: str, calendar_years_ahead: int = 5):
        path = Path(csv_file).expanduser()
        if not path.is_file() and not path.is_absolute():
            root = Path(__file__).resolve().parent
            for candidate in (root / path, root / path.name, root / "helper_functions" / path.name):
                if candidate.is_file():
                    path = candidate
                    break
        if not path.is_file():
            raise FileNotFoundError(f"Could not find event calendar {csv_file!r}")
        self.df = pd.read_csv(path)
        required = {'date', 'weight'}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Event calendar is missing required columns: {', '.join(sorted(missing))}")
        # Normalize all timestamps to date-only, timezone-naive values so event
        # dates align with the daily base calendar regardless of CSV timezone.
        self.df['date'] = (
            pd.to_datetime(self.df['date'], errors='raise', utc=True)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        original_weights = self.df['weight']
        parsed_weights = pd.to_numeric(original_weights, errors='coerce')
        malformed = parsed_weights.isna() & original_weights.notna() & (original_weights.astype(str).str.strip() != '')
        if malformed.any():
            warnings.warn("Malformed event weights were treated as zero", RuntimeWarning)
        self.df['weight'] = parsed_weights.fillna(0.0)
        if not self.df['weight'].map(lambda value: pd.notna(value) and abs(value) != float('inf')).all():
            raise ValueError("Event weights must be finite numbers")
        self.df.sort_values('date', inplace=True)
        self.df.set_index('date', inplace=True)

        # Event weights series (summed by date)
        self.event_weight_series = self.df['weight'].groupby(level=0).sum()

        # Base weekday weights: Mon-Fri=1, Sat=0, Sun=0.2
        weekday_weights = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.2], index=range(7))
        if self.df.empty:
            anchor = pd.Timestamp.today().normalize()
            self.event_weight_series = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
        else:
            anchor = self.df.index.min()
        start_date = min(anchor, pd.Timestamp.today().normalize()) - pd.Timedelta(days=365 * calendar_years_ahead)
        end_date = max(
            anchor if self.df.empty else self.df.index.max(),
            pd.Timestamp.today().normalize(),
        ) + pd.Timedelta(days=365 * calendar_years_ahead)
        all_days = pd.date_range(start_date, end_date, freq='D')

        self.base_weight_series = pd.Series(
            data=all_days.weekday.map(weekday_weights).values,
            index=all_days
        )
        self.base_weight_series.index = self.base_weight_series.index.tz_localize(None)

        # Cache for repeated var_days calculations
        self._vdays_cache = {}

    def get_var_days(self, start_date, end_date):
        """
        Return total variance days between two dates, including events.
        """
        start_date = pd.to_datetime(start_date, utc=True).tz_localize(None).normalize()
        end_date = pd.to_datetime(end_date, utc=True).tz_localize(None).normalize()
        if end_date < start_date:
            raise ValueError("end_date must not be before start_date")
        if end_date == start_date:
            return 0.0
        key = (start_date, end_date)

        if key in self._vdays_cache:
            return self._vdays_cache[key]

        # Base variance accrues over [start, end); event variance accrues over
        # (start, end]. This assigns an event to the first maturity that
        # covers it, while ensuring an event already observed on the valuation
        # date is not carried into future maturities.
        if start_date < self.base_weight_series.index.min() or end_date > self.base_weight_series.index.max() + pd.Timedelta(days=1):
            raise ValueError("Requested date range is outside the calendar horizon")
        base_slice = self.base_weight_series.loc[start_date:end_date - timedelta(days=1)]
        event_slice = self.event_weight_series.loc[
            start_date + timedelta(days=1):end_date
        ]
        total_vdays = base_slice.sum() + event_slice.sum()

        # Cache and return
        self._vdays_cache[key] = total_vdays
        return total_vdays

    def get_base_weights(self, dates):
        """
        Return base weekday weights for a list or range of dates.
        """
        dates = pd.to_datetime(dates, utc=True).tz_localize(None).normalize()
        return self.base_weight_series.reindex(dates, fill_value=0.0)

    def get_event_weights(self, dates):
        """
        Return event weights for a list or range of dates.
        """
        dates = pd.to_datetime(dates, utc=True).tz_localize(None).normalize()
        return self.event_weight_series.reindex(dates, fill_value=0.0)
"""Event-aware variance-day calendar used by forward volatility interpolation."""
