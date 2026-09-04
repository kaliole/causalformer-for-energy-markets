"""Regression-based deseasonalization for 4-hour aggregated energy time series.

Removes intra-day (hour-of-day) and day-of-week seasonality by fitting a linear
model with dummy variables and taking the residuals. Designed for data that is
already aggregated to 4h resolution, where each row maps to exactly one of 6
daily time slots.

Two versions:
    v1 (include_month=False): hour_bin + day_of_week
        Removes intra-day and weekly patterns. Annual cycle remains.
    v2 (include_month=True):  hour_bin + day_of_week + month
        Also removes broad annual patterns (winter/summer level differences).
        Avoids over-parameterisation: 11 month dummies capture the annual cycle
        without fitting each week individually.

Requires a DatetimeIndex on input DataFrames.

Used by:
    prepare_causalformer_input.py  --deseasonalize (v1) / --deseasonalize-v2 (v2)
    src/validation/pcmci_validation.py
    src/validation/stability_tests.py   (bootstrap)
    src/validation/falsification_tests.py  (phase randomization)
"""

import numpy as np
import pandas as pd


def deseasonalize_column(df: pd.DataFrame, col: str,
                         include_month: bool = False) -> pd.DataFrame:
    """Remove intra-day and day-of-week seasonality from one column.

    Fits a linear model with hour-of-day bins (6 bins at 4h resolution,
    computed as hour // 4) and day-of-week dummies using OLS. Optionally
    adds month-of-year dummies to also remove broad annual patterns.
    Stores the residual as <col>_ds without modifying the original column.

    Args:
        df: DataFrame with DatetimeIndex and a column named col.
        col: Name of the column to deseasonalize.
        include_month: If True, add month-of-year dummies (v2). Default False (v1).

    Returns:
        Copy of df with an additional <col>_ds column.
    """
    df = df.copy()

    hour_bin = df.index.hour // 4
    dow = df.index.dayofweek

    hour_dummies = pd.get_dummies(hour_bin, prefix="h", drop_first=True)
    dow_dummies = pd.get_dummies(dow, prefix="d", drop_first=True)

    parts = [hour_dummies, dow_dummies]
    if include_month:
        month_dummies = pd.get_dummies(df.index.month, prefix="m", drop_first=True)
        parts.append(month_dummies)

    # Intercept is required: without it, the reference category (slot 0 / Monday)
    # gets seasonal=0 instead of its true group mean, leaving it un-deseasonalized.
    X = pd.concat(parts, axis=1).astype(float)
    X.insert(0, "const", 1.0)
    X = X.values
    y = df[col].values

    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    seasonal = X @ beta

    df[f"{col}_ds"] = y - seasonal
    return df


def deseasonalize_dataframe(df: pd.DataFrame, columns: list[str],
                            include_month: bool = False) -> pd.DataFrame:
    """Deseasonalize multiple columns, adding a <col>_ds column for each.

    Applies deseasonalize_column sequentially. Original columns are preserved
    alongside the new _ds columns.

    Args:
        df: DataFrame with DatetimeIndex.
        columns: Column names to deseasonalize.
        include_month: If True, also remove month-of-year effects (v2).

    Returns:
        Copy of df with <col>_ds columns added for each entry in columns.
    """
    result = df.copy()
    for col in columns:
        result = deseasonalize_column(result, col, include_month=include_month)
    return result
