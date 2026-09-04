"""Differencing utilities for 4-hour aggregated energy time series.

Provides two usage styles:
    Style A  add_first_difference / add_seasonal_difference
             Appends new columns alongside originals in the same DataFrame.
             Use when you want to keep all representations together.

    Style B  create_differenced_view
             Returns a new DataFrame with the original columns replaced by
             their differenced values. Preferred for passing into test modules
             (Granger, optional PCMCI+ robustness runs) so downstream code
             keeps using the original column names.

Differencing periods for 4h aggregated data:
    first       shift=1   (4h change)
    daily       shift=6   (change vs same slot previous day)  ← default
    weekly      shift=42  (change vs same slot previous week)

See docs/differencing.md and docs/design_decisions.md for rationale.
"""

import pandas as pd


def add_first_difference(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Append first-difference columns (<col>_diff) for each column.

    Args:
        df: Input DataFrame.
        columns: Column names to difference.

    Returns:
        Copy of df with <col>_diff columns added. First row will be NaN.
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_diff"] = df[col].diff()
    return df


def add_seasonal_difference(
    df: pd.DataFrame,
    columns: list[str],
    period: int,
    suffix: str,
) -> pd.DataFrame:
    """Append seasonal-difference columns (<col>_<suffix>) for each column.

    Args:
        df: Input DataFrame.
        columns: Column names to difference.
        period: Lag in timesteps (e.g. 6 for daily at 4h, 42 for weekly at 4h).
        suffix: Suffix for the new column names (e.g. "sdiff_daily").

    Returns:
        Copy of df with <col>_<suffix> columns added. First `period` rows will be NaN.
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_{suffix}"] = df[col] - df[col].shift(period)
    return df


def create_differenced_view(
    df: pd.DataFrame,
    columns: list[str],
    method: str = "daily",
) -> pd.DataFrame:
    """Return a new DataFrame with columns replaced by their differenced values.

    Preferred over Style A when passing data into test modules that expect
    the original column names (e.g. Granger, robustness PCMCI+ runs).
    NaN rows introduced by the shift are dropped.

    Args:
        df: Input DataFrame.
        columns: Column names to replace with differenced values.
        method: One of "first" (shift=1), "daily" (shift=6), "weekly" (shift=42).

    Returns:
        Copy of df with the named columns replaced by differenced values,
        NaN rows dropped.

    Raises:
        ValueError: If method is not one of the supported options.
    """
    df = df.copy()

    if method == "first":
        for col in columns:
            df[col] = df[col].diff()
    elif method == "daily":
        for col in columns:
            df[col] = df[col] - df[col].shift(6)
    elif method == "weekly":
        for col in columns:
            df[col] = df[col] - df[col].shift(42)
    else:
        raise ValueError(f"Unknown differencing method: {method!r}. Use 'first', 'daily', or 'weekly'.")

    return df.dropna()
