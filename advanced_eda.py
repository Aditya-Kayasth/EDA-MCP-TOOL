"""
advanced_eda.py
===============
Comprehensive Exploratory Data Analysis toolkit with Parquet I/O.
All results are returned as JSON-serialisable dicts — no print statements.

Datetime / Time-Series Diagnostics (exhaustive)
------------------------------------------------
- Timezone detection & cross-column consistency
- Inferred frequency (pd.infer_freq) with confidence
- Monotonicity (strictly increasing / decreasing / unsorted)
- Duplicate timestamp count + example values
- NaT (Not-a-Time) count separate from general nulls
- Gap analysis: min / max / mean / std gap, gap histogram
- Missing period count vs expected sequence
- Out-of-order timestamp count
- DST transition anomaly detection
- Sentinel / suspicious date detection (epoch zero, far-future, etc.)
- Future-date count (beyond analysis date)
- Distribution by: hour-of-day, day-of-week, month, quarter, year
- Weekday vs weekend proportion
- Business-hours proportion
- Autocorrelation at lags 1, 7, 14, 30 for paired numeric columns
- ADF stationarity p-value for paired numeric columns
- Rolling-mean stability check (coefficient of variation across windows)

General
-------
- All outputs: JSON-serialisable dict  (no DataFrames, no print statements)
- Custom JSON encoder handles np.nan, np.int*, np.float*, pd.Timestamp,
  pd.NaT, timedelta, Categorical, etc.
- run_full_report()  -> dict[str, Any]
- to_json()          -> str  (pretty-printed JSON string)
- save_json()        -> Path
- export_report_to_excel() -> Path
- Parquet I/O

Usage
-----
    from advanced_eda import AdvancedEDA

    eda = AdvancedEDA.from_parquet("dataset.parquet")
    report = eda.run_full_report()          # dict
    json_str = eda.to_json()               # str
    eda.save_json("report.json")           # Path
    eda.export_report_to_excel("out.xlsx") # Path
    eda.to_parquet("cleaned.parquet")      # Path
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Optional heavy dependency — graceful degradation
try:
    from statsmodels.tsa.stattools import adfuller as _adfuller
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


# =============================================================================
#  JSON Serialisation
# =============================================================================

class _EDAEncoder(json.JSONEncoder):
    """
    Converts every Python / NumPy / Pandas type that json.dumps cannot handle.
    NaN / Inf / NaT all become JSON null.
    """
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return None if pd.isnull(obj) else obj.isoformat()
        if obj is pd.NaT:
            return None
        if isinstance(obj, pd.Categorical):
            return list(obj)
        if isinstance(obj, pd.Series):
            return obj.where(pd.notnull(obj), other=None).tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "total_seconds"):
            return obj.total_seconds()
        return super().default(obj)


def _jsonable(obj: Any) -> Any:
    """Round-trip through the custom encoder to produce a pure-Python structure."""
    return json.loads(json.dumps(obj, cls=_EDAEncoder))


# =============================================================================
#  Private Helpers
# =============================================================================

def _safe_entropy(s: pd.Series) -> float:
    probs = s.value_counts(normalize=True)
    return float(scipy_stats.entropy(probs, base=2)) if len(probs) > 1 else 0.0


def _iqr_outlier_count(s: pd.Series) -> int:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    return int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())


def _zscore_outlier_count(s: pd.Series, threshold: float = 3.0) -> int:
    clean = s.dropna()
    return 0 if clean.std() == 0 else int((np.abs(scipy_stats.zscore(clean)) > threshold).sum())


def _quality_score(missing_pct: float, iqr_outliers: int, total: int, cardinality: int) -> float:
    score = 100.0
    score -= missing_pct * 0.40
    outlier_pct = (iqr_outliers / total * 100) if total > 0 else 0
    score -= min(outlier_pct * 2, 20)
    if cardinality == 1:
        score -= 20
    return max(round(score, 2), 0.0)


# Known sentinel dates that commonly indicate bad / default data
_SENTINEL_DATES: dict[str, pd.Timestamp] = {
    "unix_epoch":     pd.Timestamp("1970-01-01"),
    "excel_epoch":    pd.Timestamp("1899-12-30"),
    "far_future":     pd.Timestamp("9999-12-31"),
    "sql_server_min": pd.Timestamp("1753-01-01"),
    "zero_date":      pd.Timestamp("0001-01-01"),
    "y2k":            pd.Timestamp("2000-01-01"),
    "millennium":     pd.Timestamp("2001-01-01"),
}
_SENTINEL_WINDOW = pd.Timedelta(days=1)


def _detect_sentinels(s: pd.Series) -> dict[str, int]:
    result = {}
    for label, sentinel in _SENTINEL_DATES.items():
        try:
            count = int((abs(s - sentinel) <= _SENTINEL_WINDOW).sum())
        except Exception:
            count = 0
        if count:
            result[label] = count
    return result


def _dst_anomalies(s: pd.Series) -> int:
    """Count timestamps in DST spring-forward gaps (tz-aware only)."""
    if s.dt.tz is None:
        return 0
    try:
        import pytz
        tz = s.dt.tz
        count = 0
        for ts in s.dropna():
            try:
                ts.replace(tzinfo=None).replace(tzinfo=pytz.utc).astimezone(tz)
            except Exception:
                count += 1
        return count
    except ImportError:
        return 0


def _gap_histogram(gaps: pd.Series, n_bins: int = 8) -> list[dict]:
    if gaps.empty:
        return []
    seconds = gaps.dt.total_seconds()
    counts, edges = np.histogram(seconds, bins=n_bins)
    return [
        {
            "bin_start_s": round(float(edges[i]), 2),
            "bin_end_s":   round(float(edges[i + 1]), 2),
            "count":       int(counts[i]),
        }
        for i in range(len(counts))
    ]


def _adf_test(s: pd.Series) -> dict:
    """Augmented Dickey-Fuller stationarity test (requires statsmodels)."""
    if not _HAS_STATSMODELS:
        return {"available": False, "reason": "statsmodels not installed"}
    clean = s.dropna()
    if len(clean) < 20:
        return {"available": False, "reason": "fewer than 20 non-null observations"}
    try:
        res = _adfuller(clean, autolag="AIC")
        return {
            "available":        True,
            "adf_statistic":    round(float(res[0]), 6),
            "p_value":          round(float(res[1]), 6),
            "used_lags":        int(res[2]),
            "n_observations":   int(res[3]),
            "critical_values":  {k: round(v, 4) for k, v in res[4].items()},
            "is_stationary_5pct": bool(res[1] < 0.05),
        }
    except Exception as exc:
        return {"available": False, "reason": str(exc)}


def _rolling_cv(s: pd.Series, window: int = 30) -> Optional[float]:
    """Coefficient of variation of rolling mean — measures trend stability."""
    clean = s.dropna()
    if len(clean) < window * 2:
        return None
    roll_means = clean.rolling(window).mean().dropna()
    mean_val = roll_means.mean()
    if mean_val == 0:
        return None
    return round(float(roll_means.std() / abs(mean_val)), 6)


def _autocorr_lags(s: pd.Series, lags: list[int]) -> dict[str, Optional[float]]:
    clean = s.dropna()
    result = {}
    for lag in lags:
        if len(clean) > lag:
            val = clean.autocorr(lag=lag)
            result[f"lag_{lag}"] = round(float(val), 6) if not math.isnan(val) else None
        else:
            result[f"lag_{lag}"] = None
    return result


# =============================================================================
#  AdvancedEDA
# =============================================================================

class AdvancedEDA:
    """
    Comprehensive EDA toolkit for any structured Pandas DataFrame.

    All public analysis methods return JSON-serialisable Python dicts.
    No print statements are used anywhere in the class.

    Parameters
    ----------
    dataframe : pd.DataFrame
    name      : str  — Label used in report metadata.
    """

    def __init__(self, dataframe: pd.DataFrame, name: str = "dataset") -> None:
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        self.df = dataframe.copy()
        self.name = name
        self._report: dict[str, Any] = {}
        self._analysis_timestamp = datetime.now(timezone.utc).isoformat()

    # -------------------------------------------------------------------------
    #  Parquet I/O
    # -------------------------------------------------------------------------

    @classmethod
    def from_parquet(
        cls,
        path: Union[str, Path],
        name: Optional[str] = None,
        **kwargs,
    ) -> "AdvancedEDA":
        """Load a Parquet file and return an AdvancedEDA instance."""
        path = Path(path)
        df = pd.read_parquet(path, **kwargs)
        return cls(df, name=name or path.stem)

    def to_parquet(
        self,
        path: Union[str, Path],
        engine: str = "pyarrow",
        compression: str = "snappy",
        index: bool = False,
        **kwargs,
    ) -> Path:
        """Save the working DataFrame to Parquet. Returns resolved Path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(path, engine=engine, compression=compression, index=index, **kwargs)
        return path

    # -------------------------------------------------------------------------
    #  Dataset Overview
    # -------------------------------------------------------------------------

    def get_dataset_overview(self) -> dict:
        """High-level snapshot of the entire dataset."""
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_total = int(self.df.isnull().sum().sum())
        return _jsonable({
            "dataset_name":            self.name,
            "analysis_timestamp_utc":  self._analysis_timestamp,
            "rows":                    self.df.shape[0],
            "columns":                 self.df.shape[1],
            "total_cells":             total_cells,
            "column_type_counts": {
                "numeric":     int(len(self.df.select_dtypes(include="number").columns)),
                "categorical": int(len(self.df.select_dtypes(include=["object", "category"]).columns)),
                "datetime":    int(len(self.df.select_dtypes(include=["datetime", "datetimetz"]).columns)),
                "boolean":     int(len(self.df.select_dtypes(include="bool").columns)),
                "timedelta":   int(len(self.df.select_dtypes(include="timedelta").columns)),
            },
            "missing": {
                "total_missing_values":   missing_total,
                "missing_pct":            round(missing_total / total_cells * 100, 2) if total_cells else 0,
                "columns_with_missing":   int((self.df.isnull().sum() > 0).sum()),
                "rows_with_any_missing":  int(self.df.isnull().any(axis=1).sum()),
                "fully_null_columns":     int(self.df.isnull().all().sum()),
            },
            "duplicates": {
                "duplicate_rows":     int(self.df.duplicated().sum()),
                "duplicate_rows_pct": round(self.df.duplicated().mean() * 100, 2),
            },
            "column_health": {
                "constant_columns":     int((self.df.nunique() == 1).sum()),
                "fully_unique_columns": int((self.df.nunique() == len(self.df)).sum()),
            },
            "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / 1024 ** 2, 4),
            "column_names":    self.df.columns.tolist(),
        })

    # -------------------------------------------------------------------------
    #  Column Metadata
    # -------------------------------------------------------------------------

    def get_column_metadata(self) -> list[dict]:
        """Deep per-column metadata for every column. Returns list of dicts."""
        n = len(self.df)
        records = []

        for col in self.df.columns:
            s = self.df[col]
            dtype = str(s.dtype)
            missing = int(s.isnull().sum())
            missing_pct = round(missing / n * 100, 2) if n else 0
            unique = int(s.nunique())

            rec: dict[str, Any] = {
                "feature":        col,
                "dtype":          dtype,
                "dtype_kind":     s.dtype.kind,
                "missing_count":  missing,
                "missing_pct":    missing_pct,
                "cardinality":    unique,
                "is_constant":    unique == 1,
                "is_fully_unique": unique == n,
                "memory_bytes":   int(s.memory_usage(deep=True)),
                "quality_score":  None,
            }

            # Numeric
            if pd.api.types.is_numeric_dtype(s):
                clean = s.dropna()
                iqr_out = _iqr_outlier_count(clean) if len(clean) else 0
                z_out   = _zscore_outlier_count(clean) if len(clean) else 0
                q1 = float(clean.quantile(0.25)) if len(clean) else None
                q3 = float(clean.quantile(0.75)) if len(clean) else None
                iqr_v = (q3 - q1) if (q1 is not None and q3 is not None) else None
                mean_v = float(clean.mean()) if len(clean) else None
                std_v  = float(clean.std())  if len(clean) else None
                rec["numeric"] = _jsonable({
                    "mean":    round(mean_v, 6) if mean_v is not None else None,
                    "std_dev": round(std_v, 6)  if std_v  is not None else None,
                    "cv_pct":  round(std_v / abs(mean_v) * 100, 2)
                               if (mean_v and std_v and mean_v != 0) else None,
                    "min":    float(clean.min()) if len(clean) else None,
                    "p1":     float(clean.quantile(0.01)) if len(clean) else None,
                    "p5":     float(clean.quantile(0.05)) if len(clean) else None,
                    "p25":    q1,
                    "median": float(clean.median()) if len(clean) else None,
                    "p75":    q3,
                    "p95":    float(clean.quantile(0.95)) if len(clean) else None,
                    "p99":    float(clean.quantile(0.99)) if len(clean) else None,
                    "max":    float(clean.max()) if len(clean) else None,
                    "range":  float(clean.max() - clean.min()) if len(clean) else None,
                    "iqr":    round(iqr_v, 6) if iqr_v is not None else None,
                    "zeros":     int((clean == 0).sum()),
                    "negatives": int((clean < 0).sum()),
                    "skewness":  round(float(clean.skew()), 4) if len(clean) > 2 else None,
                    "kurtosis":  round(float(clean.kurtosis()), 4) if len(clean) > 3 else None,
                    "outliers_iqr":         iqr_out,
                    "outliers_iqr_pct":     round(iqr_out / n * 100, 2),
                    "outliers_zscore":      z_out,
                    "outliers_zscore_pct":  round(z_out / n * 100, 2),
                    "lower_fence_iqr":      round(q1 - 1.5 * iqr_v, 6)
                                            if (q1 is not None and iqr_v is not None) else None,
                    "upper_fence_iqr":      round(q3 + 1.5 * iqr_v, 6)
                                            if (q3 is not None and iqr_v is not None) else None,
                })
                rec["quality_score"] = _quality_score(missing_pct, iqr_out, n, unique)

            # Categorical / Object
            if pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
                vc = s.value_counts(dropna=True)
                top_freq = int(vc.iloc[0]) if len(vc) else None
                bot_freq = int(vc.iloc[-1]) if len(vc) else None
                str_lens = s.dropna().astype(str).str.len()
                rec["categorical"] = _jsonable({
                    "top_value":       vc.index[0] if len(vc) else None,
                    "top_frequency":   top_freq,
                    "top_pct":         round(top_freq / n * 100, 2) if top_freq else None,
                    "bottom_value":    vc.index[-1] if len(vc) else None,
                    "bottom_frequency": bot_freq,
                    "imbalance_ratio": round(top_freq / bot_freq, 2)
                                       if (top_freq and bot_freq and bot_freq > 0) else None,
                    "shannon_entropy_bits": round(_safe_entropy(s), 4),
                    "avg_string_length": round(float(str_lens.mean()), 2) if len(str_lens) else None,
                    "max_string_length": int(str_lens.max()) if len(str_lens) else None,
                    "min_string_length": int(str_lens.min()) if len(str_lens) else None,
                    "top_10_values": [
                        {"value": str(v), "count": int(c), "pct": round(c / n * 100, 2)}
                        for v, c in vc.head(10).items()
                    ],
                })
                rec["quality_score"] = _quality_score(missing_pct, 0, n, unique)

            # Boolean
            if pd.api.types.is_bool_dtype(s):
                vc = s.value_counts(dropna=True)
                rec["boolean"] = {
                    "true_count":  int(vc.get(True, 0)),
                    "false_count": int(vc.get(False, 0)),
                    "true_pct":    round(vc.get(True, 0) / n * 100, 2) if n else None,
                }
                rec["quality_score"] = _quality_score(missing_pct, 0, n, unique)

            if rec["quality_score"] is None:
                rec["quality_score"] = _quality_score(missing_pct, 0, n, unique)

            records.append(_jsonable(rec))

        return records

    # -------------------------------------------------------------------------
    #  Numerical Summary
    # -------------------------------------------------------------------------

    def get_numerical_summary(self) -> dict:
        """Extended descriptive statistics for all numeric columns."""
        num_df = self.df.select_dtypes(include="number")
        if num_df.empty:
            return {"available": False, "reason": "No numeric columns."}

        out: dict[str, Any] = {}
        for col in num_df.columns:
            s = num_df[col].dropna()
            out[col] = _jsonable({
                "count":          len(s),
                "mean":           float(s.mean())   if len(s) else None,
                "std":            float(s.std())    if len(s) else None,
                "min":            float(s.min())    if len(s) else None,
                "p25":            float(s.quantile(0.25)) if len(s) else None,
                "median":         float(s.median()) if len(s) else None,
                "p75":            float(s.quantile(0.75)) if len(s) else None,
                "max":            float(s.max())    if len(s) else None,
                "skewness":       float(s.skew())   if len(s) > 2 else None,
                "kurtosis":       float(s.kurtosis()) if len(s) > 3 else None,
                "iqr_outliers":   _iqr_outlier_count(s),
                "zscore_outliers": _zscore_outlier_count(s),
            })
        return out

    # -------------------------------------------------------------------------
    #  Categorical Summary
    # -------------------------------------------------------------------------

    def get_categorical_summary(self) -> dict:
        """Top-10 ranked frequency table for every categorical column."""
        cat_df = self.df.select_dtypes(include=["object", "category"])
        if cat_df.empty:
            return {"available": False, "reason": "No categorical columns."}

        n = len(self.df)
        out: dict[str, Any] = {}
        for col in cat_df.columns:
            vc = cat_df[col].value_counts(dropna=False).head(10)
            cumulative = 0
            rows = []
            for rank, (val, cnt) in enumerate(vc.items(), 1):
                cumulative += cnt
                rows.append({
                    "rank":           rank,
                    "value":          str(val) if pd.notnull(val) else None,
                    "count":          int(cnt),
                    "pct":            round(cnt / n * 100, 2),
                    "cumulative_pct": round(cumulative / n * 100, 2),
                })
            out[col] = rows
        return out

    # -------------------------------------------------------------------------
    #  Datetime / Time-Series Diagnostics  ← fully rewritten
    # -------------------------------------------------------------------------

    def get_datetime_summary(self) -> dict:
        """
        Exhaustive diagnostics for every datetime column plus cross-column checks.

        Sections per column
        -------------------
        basic_stats          min, max, range_days, NaT count, timezone
        frequency            pd.infer_freq, actual vs expected count
        ordering             monotonicity, out-of-order count
        duplicates           duplicate timestamp count + example values
        gaps                 min/max/mean/std gap, zero gaps, gap histogram,
                             missing period count relative to inferred frequency
        sentinel_dates       known bad-date markers (epoch, far-future, y2k …)
        future_dates         count of timestamps beyond the analysis date
        dst_anomalies        spring-forward non-existent time counts (tz-aware)
        distributions        by hour-of-day, day-of-week, month, quarter, year
        business_hours       weekday %, business-hours (09-17) %, weekend %
        time_series_analysis per paired numeric column:
                             autocorrelation at lags 1/7/14/30,
                             ADF stationarity test,
                             rolling-mean CV (stability)

        Cross-column section
        --------------------
        timezone_consistency  all datetime cols share the same tz (or are all naive)
        """
        dt_cols = self.df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()
        if not dt_cols:
            return {"available": False, "reason": "No datetime columns found."}

        now = pd.Timestamp.utcnow().tz_localize(None)
        num_cols = self.df.select_dtypes(include="number").columns.tolist()
        timezones_seen: list[Optional[str]] = []
        col_reports: dict[str, Any] = {}

        for col in dt_cols:
            s: pd.Series = self.df[col]
            s_clean = s.dropna()
            nat_count = int(s.isnull().sum())
            n = len(s)

            # Timezone
            tz_name: Optional[str] = (
                str(s_clean.dt.tz) if (len(s_clean) and s_clean.dt.tz) else None
            )
            timezones_seen.append(tz_name)

            # Naive copy for arithmetic
            s_naive = s_clean.dt.tz_localize(None) if tz_name else s_clean

            # ── Basic stats ───────────────────────────────────────────────
            ts_min = s_naive.min() if len(s_naive) else None
            ts_max = s_naive.max() if len(s_naive) else None
            range_days = (ts_max - ts_min).days if (ts_min is not None and ts_max is not None) else None

            # ── Inferred frequency ────────────────────────────────────────
            sorted_s = s_naive.sort_values().reset_index(drop=True)
            try:
                inferred_freq = pd.infer_freq(sorted_s)
            except Exception:
                inferred_freq = None

            expected_count: Optional[int] = None
            if inferred_freq and ts_min is not None and ts_max is not None:
                try:
                    expected_count = len(pd.date_range(ts_min, ts_max, freq=inferred_freq))
                except Exception:
                    pass

            # ── Ordering ──────────────────────────────────────────────────
            is_mono_inc = bool(s_naive.is_monotonic_increasing)
            is_mono_dec = bool(s_naive.is_monotonic_decreasing)
            out_of_order = int((s_naive.diff().dropna() < pd.Timedelta(0)).sum())

            # ── Duplicates ────────────────────────────────────────────────
            dup_mask = s_naive.duplicated(keep=False)
            dup_count = int(dup_mask.sum())
            dup_examples = [str(v) for v in s_naive[dup_mask].unique()[:5]]

            # ── Gap analysis ──────────────────────────────────────────────
            if len(sorted_s) >= 2:
                gaps = sorted_s.diff().dropna()
                gap_secs = gaps.dt.total_seconds()
                zero_gaps = int((gap_secs == 0).sum())
                actual_unique = len(s_naive.unique())
                missing_periods = (
                    max(expected_count - actual_unique, 0) if expected_count is not None else None
                )
                gap_stats: dict[str, Any] = _jsonable({
                    "min_gap_seconds":       float(gap_secs.min()),
                    "max_gap_seconds":       float(gap_secs.max()),
                    "mean_gap_seconds":      round(float(gap_secs.mean()), 3),
                    "std_gap_seconds":       round(float(gap_secs.std()), 3),
                    "median_gap_seconds":    round(float(gap_secs.median()), 3),
                    "zero_gaps":             zero_gaps,
                    "expected_period_count": expected_count,
                    "actual_unique_count":   actual_unique,
                    "missing_periods":       missing_periods,
                    "missing_periods_pct":   round(missing_periods / expected_count * 100, 2)
                                             if (missing_periods is not None and expected_count) else None,
                    "gap_histogram":         _gap_histogram(gaps),
                    "largest_gap_start":     str(sorted_s[gap_secs.idxmax() - 1])
                                             if len(gap_secs) else None,
                    "largest_gap_end":       str(sorted_s[gap_secs.idxmax()])
                                             if len(gap_secs) else None,
                })
            else:
                gap_stats = {"available": False, "reason": "Fewer than 2 non-null timestamps."}

            # ── Sentinel dates ────────────────────────────────────────────
            sentinels = _detect_sentinels(s_naive) if len(s_naive) else {}

            # ── Future dates ──────────────────────────────────────────────
            future_count = int((s_naive > now).sum()) if len(s_naive) else 0

            # ── DST anomalies ─────────────────────────────────────────────
            dst_count = _dst_anomalies(s_clean)

            # ── Distributions ─────────────────────────────────────────────
            def _freq_dist(series: pd.Series, attr: str) -> list[dict]:
                vc = getattr(series.dt, attr).value_counts().sort_index()
                total = vc.sum()
                return [
                    {"value": int(k), "count": int(v), "pct": round(v / total * 100, 2)}
                    for k, v in vc.items()
                ]

            distributions: dict[str, Any] = {}
            if len(s_naive):
                distributions = {
                    "by_hour":        _freq_dist(s_naive, "hour"),
                    "by_day_of_week": _freq_dist(s_naive, "dayofweek"),  # 0=Mon, 6=Sun
                    "by_month":       _freq_dist(s_naive, "month"),
                    "by_quarter":     _freq_dist(s_naive, "quarter"),
                    "by_year":        _freq_dist(s_naive, "year"),
                }

            # ── Business hours ────────────────────────────────────────────
            business_hours: dict[str, Any] = {}
            if len(s_naive):
                is_weekday  = s_naive.dt.dayofweek < 5
                is_bus_hour = s_naive.dt.hour.between(9, 16)
                business_hours = {
                    "weekday_pct":        round(float(is_weekday.mean() * 100), 2),
                    "weekend_pct":        round(float((~is_weekday).mean() * 100), 2),
                    "business_hours_pct": round(float((is_weekday & is_bus_hour).mean() * 100), 2),
                    "after_hours_pct":    round(float((is_weekday & ~is_bus_hour).mean() * 100), 2),
                }

            # ── Time-series analysis (paired with numeric columns) ─────────
            ts_analysis: dict[str, Any] = {}
            if num_cols and len(s_naive) > 20:
                sort_idx = s.sort_values().index
                for nc in num_cols:
                    paired = self.df.loc[sort_idx, nc]
                    ts_analysis[nc] = _jsonable({
                        "autocorrelation":     _autocorr_lags(paired, [1, 7, 14, 30]),
                        "rolling_cv_window30": _rolling_cv(paired, window=30),
                        "adf_stationarity":    _adf_test(paired),
                    })

            col_reports[col] = _jsonable({
                "basic_stats": {
                    "n_total":    n,
                    "n_non_null": int(len(s_clean)),
                    "nat_count":  nat_count,
                    "nat_pct":    round(nat_count / n * 100, 2) if n else 0,
                    "min":        str(ts_min) if ts_min is not None else None,
                    "max":        str(ts_max) if ts_max is not None else None,
                    "range_days": range_days,
                    "timezone":   tz_name,
                    "is_tz_aware": tz_name is not None,
                },
                "frequency": {
                    "inferred_freq":  inferred_freq,
                    "freq_confidence": "high" if inferred_freq else "none",
                    "actual_unique_count": int(len(s_naive.unique())),
                    "expected_count": expected_count,
                },
                "ordering": {
                    "is_monotonic_increasing": is_mono_inc,
                    "is_monotonic_decreasing": is_mono_dec,
                    "is_sorted":               is_mono_inc or is_mono_dec,
                    "out_of_order_count":      out_of_order,
                    "out_of_order_pct":        round(out_of_order / n * 100, 2) if n else 0,
                },
                "duplicates": {
                    "duplicate_timestamp_count": dup_count,
                    "duplicate_timestamp_pct":   round(dup_count / n * 100, 2) if n else 0,
                    "duplicate_examples":        dup_examples,
                },
                "gaps":           gap_stats,
                "sentinel_dates": sentinels,
                "future_dates": {
                    "count": future_count,
                    "pct":   round(future_count / n * 100, 2) if n else 0,
                },
                "dst_anomalies": {
                    "count": dst_count,
                    "applicable": tz_name is not None,
                },
                "distributions":         distributions,
                "business_hours":        business_hours,
                "time_series_analysis":  ts_analysis,
            })

        # Cross-column timezone consistency
        unique_tzs = list(set(timezones_seen))
        return {
            "cross_column": {
                "datetime_columns":    dt_cols,
                "timezones_present":   unique_tzs,
                "timezone_consistent": len(unique_tzs) == 1,
                "mixed_tz_warning":    len(unique_tzs) > 1,
            },
            "columns": col_reports,
        }

    # -------------------------------------------------------------------------
    #  Correlation Analysis
    # -------------------------------------------------------------------------

    def get_correlation_analysis(
        self,
        method: str = "pearson",
        high_corr_threshold: float = 0.85,
    ) -> dict:
        """Full correlation matrix + flagged high-correlation pairs."""
        num_df = self.df.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            return {"available": False, "reason": "Need ≥ 2 numeric columns."}

        corr = num_df.corr(method=method)
        corr_matrix = _jsonable(corr.round(4).to_dict())

        pairs = []
        cols = corr.columns.tolist()
        for i, c1 in enumerate(cols):
            for c2 in cols[i + 1:]:
                val = corr.loc[c1, c2]
                if abs(val) >= high_corr_threshold:
                    pairs.append({
                        "feature_a":       c1,
                        "feature_b":       c2,
                        "correlation":     round(float(val), 4),
                        "abs_correlation": round(abs(float(val)), 4),
                        "risk":            "High" if abs(val) >= 0.95 else "Moderate",
                    })
        pairs.sort(key=lambda x: -x["abs_correlation"])

        return {
            "method":                   method,
            "threshold":                high_corr_threshold,
            "matrix":                   corr_matrix,
            "high_correlation_pairs":   pairs,
            "high_correlation_count":   len(pairs),
        }

    # -------------------------------------------------------------------------
    #  Outlier Summary
    # -------------------------------------------------------------------------

    def get_outlier_summary(self) -> dict:
        """IQR and Z-score outlier report for all numeric columns."""
        num_df = self.df.select_dtypes(include="number")
        if num_df.empty:
            return {"available": False, "reason": "No numeric columns."}

        n = len(self.df)
        records = []
        for col in num_df.columns:
            s = num_df[col].dropna()
            if s.empty:
                continue
            q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
            iqr = q3 - q1
            iqr_out = _iqr_outlier_count(s)
            z_out   = _zscore_outlier_count(s)
            records.append(_jsonable({
                "feature":           col,
                "iqr_outliers":      iqr_out,
                "iqr_outlier_pct":   round(iqr_out / n * 100, 2),
                "zscore_outliers":   z_out,
                "zscore_outlier_pct": round(z_out / n * 100, 2),
                "lower_fence":       round(q1 - 1.5 * iqr, 4),
                "upper_fence":       round(q3 + 1.5 * iqr, 4),
                "severity":          "Critical" if iqr_out / n > 0.10 else
                                     "High"     if iqr_out / n > 0.05 else
                                     "Moderate" if iqr_out / n > 0.01 else "Low",
            }))

        records.sort(key=lambda x: -x["iqr_outliers"])
        return {"columns": records}

    # -------------------------------------------------------------------------
    #  Missing Value Report
    # -------------------------------------------------------------------------

    def get_missing_value_report(self) -> dict:
        """Sorted missing-value report with severity classification."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            return {"has_missing": False, "columns": []}

        n = len(self.df)
        records = []
        for col, cnt in missing.items():
            pct = round(cnt / n * 100, 2)
            sev = (
                "Critical"   if pct >= 50 else
                "High"       if pct >= 20 else
                "Moderate"   if pct >= 5  else
                "Low"        if pct >= 1  else
                "Negligible"
            )
            records.append({
                "feature":          col,
                "missing_count":    int(cnt),
                "missing_pct":      pct,
                "non_missing_count": n - int(cnt),
                "severity":         sev,
            })
        return {"has_missing": True, "columns": records}

    # -------------------------------------------------------------------------
    #  Data Quality Scorecard
    # -------------------------------------------------------------------------

    def get_data_quality_scorecard(self) -> dict:
        """Per-column quality scores (0–100) plus an overall dataset grade."""
        meta = self.get_column_metadata()
        scores = [
            {
                "feature":       r["feature"],
                "dtype":         r["dtype"],
                "missing_pct":   r["missing_pct"],
                "cardinality":   r["cardinality"],
                "quality_score": r["quality_score"],
            }
            for r in meta
        ]
        valid_scores = [r["quality_score"] for r in scores if r["quality_score"] is not None]
        overall = round(float(np.mean(valid_scores)), 2) if valid_scores else 0.0
        grade = next(g for t, g in [(90, "A"), (75, "B"), (60, "C"), (45, "D"), (0, "F")] if overall >= t)
        return {
            "overall_score": overall,
            "overall_grade": grade,
            "interpretation": {
                "A": "Excellent (≥90)",
                "B": "Good (75–89)",
                "C": "Fair (60–74)",
                "D": "Poor (45–59)",
                "F": "Critical (<45)",
            }[grade],
            "columns": scores,
        }

    # -------------------------------------------------------------------------
    #  Full Report
    # -------------------------------------------------------------------------

    def run_full_report(self, corr_method: str = "pearson") -> dict[str, Any]:
        """
        Execute all analysis sections and return a fully JSON-serialisable dict.

        Keys
        ----
        dataset_overview, column_metadata, numerical_summary,
        categorical_summary, datetime_summary, correlation_analysis,
        outlier_summary, missing_value_report, quality_scorecard
        """
        self._report = {
            "dataset_overview":     self.get_dataset_overview(),
            "column_metadata":      self.get_column_metadata(),
            "numerical_summary":    self.get_numerical_summary(),
            "categorical_summary":  self.get_categorical_summary(),
            "datetime_summary":     self.get_datetime_summary(),
            "correlation_analysis": self.get_correlation_analysis(method=corr_method),
            "outlier_summary":      self.get_outlier_summary(),
            "missing_value_report": self.get_missing_value_report(),
            "quality_scorecard":    self.get_data_quality_scorecard(),
        }
        return self._report

    # -------------------------------------------------------------------------
    #  Serialise / Export
    # -------------------------------------------------------------------------

    def to_json(self, indent: int = 2, corr_method: str = "pearson") -> str:
        """Return the full report as a pretty-printed JSON string."""
        if not self._report:
            self.run_full_report(corr_method=corr_method)
        return json.dumps(self._report, cls=_EDAEncoder, indent=indent, ensure_ascii=False)

    def save_json(
        self,
        path: Union[str, Path] = "eda_report.json",
        indent: int = 2,
        corr_method: str = "pearson",
    ) -> Path:
        """Write the full report JSON to disk. Returns resolved Path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(indent=indent, corr_method=corr_method), encoding="utf-8")
        return path

    def export_report_to_excel(
        self,
        path: Union[str, Path] = "eda_report.xlsx",
        corr_method: str = "pearson",
    ) -> Path:
        """Write a multi-sheet Excel workbook from the full report. Returns Path."""
        if not self._report:
            self.run_full_report(corr_method=corr_method)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        def _to_df(val: Any) -> pd.DataFrame:
            if isinstance(val, list):
                return pd.DataFrame(val)
            if isinstance(val, dict):
                flat = {
                    k: (json.dumps(v, cls=_EDAEncoder) if isinstance(v, (dict, list)) else v)
                    for k, v in val.items()
                }
                return pd.DataFrame([flat])
            return pd.DataFrame({"value": [str(val)]})

        sheet_map = {
            "Overview":     "dataset_overview",
            "Col Metadata": "column_metadata",
            "Numeric":      "numerical_summary",
            "Categorical":  "categorical_summary",
            "Datetime":     "datetime_summary",
            "Correlation":  "correlation_analysis",
            "Outliers":     "outlier_summary",
            "Missing":      "missing_value_report",
            "Quality":      "quality_scorecard",
        }

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet, key in sheet_map.items():
                data = self._report.get(key, {})
                try:
                    _to_df(data).to_excel(writer, sheet_name=sheet, index=False)
                except Exception:
                    pd.DataFrame({"raw": [json.dumps(data, cls=_EDAEncoder)]}).to_excel(
                        writer, sheet_name=sheet, index=False)

        return path

    # -------------------------------------------------------------------------
    #  Dunder
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AdvancedEDA(name='{self.name}', "
            f"shape={self.df.shape}, "
            f"memory_mb={round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)})"
        )


# =============================================================================
#  Demo  (python advanced_eda.py)
# =============================================================================

if __name__ == "__main__":
    import pprint

    np.random.seed(42)
    N = 500

    # Deliberately messy datetime to exercise every diagnostic
    base_ts = pd.date_range("2021-01-01", periods=N, freq="6h")
    ts_messy = base_ts.to_series().sample(frac=1, random_state=0)   # shuffled → unsorted
    ts_messy.iloc[10] = pd.Timestamp("1970-01-01")                   # epoch sentinel
    ts_messy.iloc[20] = pd.Timestamp("9999-12-31")                   # far-future sentinel
    # Duplicate a few timestamps
    ts_messy.iloc[30] = ts_messy.iloc[31]

    demo_df = pd.DataFrame({
        "event_time": ts_messy.values,
        "value":      np.random.normal(100, 15, N),
        "revenue":    np.random.exponential(500, N),
        "category":   np.random.choice(["A", "B", "C"], N, p=[0.7, 0.2, 0.1]),
        "region":     np.random.choice(["North", "South", "East", "West"], N),
        "is_active":  np.random.choice([True, False], N),
        "constant":   42,
    })
    # Inject NaT and NaN
    demo_df.loc[np.random.choice(demo_df.index, 30, replace=False), "event_time"] = pd.NaT
    demo_df.loc[np.random.choice(demo_df.index, 40, replace=False), "value"] = np.nan

    # Save → reload via Parquet
    demo_df.to_parquet("demo.parquet", index=False)
    eda = AdvancedEDA.from_parquet("demo.parquet", name="Demo")

    report = eda.run_full_report()

    # Inspect datetime section
    pprint.pprint(report["datetime_summary"], depth=3)

    # Save JSON & Excel reports
    eda.save_json("eda_report.json")
    eda.export_report_to_excel("eda_report.xlsx")

    # Re-save dataset
    eda.to_parquet("demo_v2.parquet")
