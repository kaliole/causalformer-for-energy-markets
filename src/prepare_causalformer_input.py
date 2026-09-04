"""Prepare the DE-LU dataset for CausalFormer input.

CausalFormer expects a plain CSV with:
  - A header row of variable names
  - No timestamp column — only numeric values
  - Each column is one time series variable
  - Each row is one timestep
  - Shape: [num_timesteps, num_variables]

The data loader internally applies MinMaxScaler((0.5, 1.0)) and constructs
sliding windows, so we do NOT normalize or window the data here.

Modes:
  (default):                        Raw hourly values. Output: causalformer_input.csv
  --diff:                           First-differenced values x(t) - x(t-1). Output: causalformer_input_diff.csv
  --aggregate N:                    Temporal aggregation to N-hourly means. Output: causalformer_input_aggNh.csv
  --aggregate N --diff:             Aggregated then first-differenced. Output: causalformer_input_aggNh_diff.csv
  --aggregate N --deseasonalize:    Aggregated + deseasonalized v1 (hour_bin + dow). Output: causalformer_input_aggNh_ds.csv
  --aggregate N --deseasonalize-v2: Aggregated + deseasonalized v2 (hour_bin + dow + month). Output: causalformer_input_aggNh_ds_v2.csv

Usage:
    python src/prepare_causalformer_input.py
    python src/prepare_causalformer_input.py --diff
    python src/prepare_causalformer_input.py --aggregate 4
    python src/prepare_causalformer_input.py --aggregate 4 --diff
    python src/prepare_causalformer_input.py --aggregate 4 --deseasonalize
    python src/prepare_causalformer_input.py --aggregate 4 --deseasonalize-v2
"""

import argparse
import pathlib

import numpy as np
import pandas as pd

from preprocessing.deseasonalize import deseasonalize_dataframe

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "delu_dataset.csv"
OUTPUT_CSV_LEVELS = PROJECT_ROOT / "data" / "processed" / "causalformer_input.csv"
OUTPUT_CSV_DIFF = PROJECT_ROOT / "data" / "processed" / "causalformer_input_diff.csv"

# Column order for the CausalFormer input.
# This order determines variable indices in the discovered causal graph.
# Weather variables first (exogenous), then market variables (endogenous).
COLUMNS = [
    "temperature",
    "wind_speed",
    "solar_radiation",
    "price",
    "load",
    "wind_generation",
    "solar_generation",
]


def load_dataset() -> pd.DataFrame:
    """Load the merged DE-LU dataset."""
    df = pd.read_csv(INPUT_CSV, parse_dates=["timestamp"], index_col="timestamp")
    print(f"Loaded: {INPUT_CSV}")
    print(f"  Shape:   {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Period:  {df.index.min()} to {df.index.max()}")
    return df


def validate_input(df: pd.DataFrame) -> None:
    """Check the input data before conversion."""
    errors = []

    missing_cols = set(COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")

    n_nan = df[COLUMNS].isna().sum().sum()
    if n_nan > 0:
        errors.append(f"NaN values: {n_nan}")

    if errors:
        raise ValueError("Input validation failed:\n" + "\n".join(f"  - {e}" for e in errors))


def apply_aggregation(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Aggregate hourly data into N-hourly means.

    Groups consecutive hours into non-overlapping windows of size N and
    computes the mean within each window. This reduces the temporal
    resolution to match the timescale at which causal effects operate
    in energy markets, weakening the autoregressive shortcut that
    dominates at hourly resolution.

    For window=4: 17,520 hourly rows → 4,380 four-hourly rows.
    """
    n_rows = len(df)
    # Drop trailing rows that don't fill a complete window
    n_complete = (n_rows // window) * window
    n_dropped = n_rows - n_complete
    df_trimmed = df.iloc[:n_complete]

    # Group into non-overlapping windows and average
    group_idx = np.arange(n_complete) // window
    df_agg = df_trimmed[COLUMNS].groupby(group_idx).mean()
    # Restore DatetimeIndex (groupby replaces it with integers).
    # Each aggregated row corresponds to the first hour of its window.
    df_agg.index = df_trimmed.index[::window]

    print(f"\n  Temporal aggregation applied (window={window}h):")
    print(f"    Rows: {n_rows} → {len(df_agg)} ({window}-hourly means)")
    if n_dropped > 0:
        print(f"    Last {n_dropped} hours dropped (incomplete window)")
    print(f"    Each row is the mean of {window} consecutive hourly values")
    return df_agg


def apply_differencing(df: pd.DataFrame) -> pd.DataFrame:
    """Compute first differences: x(t) - x(t-1) for all columns.

    Removes the dominant autoregressive signal so that CausalFormer
    can learn cross-variable causal structure instead of just copying
    the previous value.

    Drops the first row (no previous value to difference against),
    resulting in 17,519 rows instead of 17,520.
    """
    df_diff = df[COLUMNS].diff().iloc[1:]  # drop first NaN row
    print(f"\n  First-differencing applied:")
    print(f"    Rows: {len(df)} → {len(df_diff)} (first row dropped)")
    print(f"    Each value is now x(t) - x(t-1)")
    return df_diff


def apply_deseasonalization(df: pd.DataFrame, include_month: bool = False) -> pd.DataFrame:
    """Remove seasonality from all variables using OLS dummy regression.

    v1 (include_month=False): hour_bin + day_of_week
    v2 (include_month=True):  hour_bin + day_of_week + month_of_year

    Returns a DataFrame with the original column names containing only the
    deseasonalized residuals. Must be called after apply_aggregation so the
    DatetimeIndex is present (see docs/design_decisions.md).
    """
    df_ds = deseasonalize_dataframe(df, COLUMNS, include_month=include_month)

    result = pd.DataFrame(index=df_ds.index)
    for col in COLUMNS:
        result[col] = df_ds[f"{col}_ds"]

    version = "v2 (hour_bin + dow + month)" if include_month else "v1 (hour_bin + dow)"
    print(f"\n  Deseasonalization applied ({version}):")
    print(f"    Rows: {len(result)} (unchanged — no lag introduced)")
    print(f"    Output uses original column names (residuals only)")
    return result


def convert_and_save(df: pd.DataFrame, output_csv: pathlib.Path, mode: str) -> None:
    """Drop timestamp, reorder columns, and save as CausalFormer input CSV."""
    df_out = df[COLUMNS].copy()
    data = df_out.values.astype(np.float32)

    print(f"\nConverted array ({mode}):")
    print(f"  Shape:       {data.shape} (timesteps × variables)")
    print(f"  Variables:   {COLUMNS}")
    print(f"  Value range: [{data.min():.2f}, {data.max():.2f}]")
    print(f"  Mean:        {data.mean():.4f}")
    print(f"  Std:         {data.std():.4f}")

    df_out.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")


def main() -> None:
    """Run the conversion pipeline."""
    parser = argparse.ArgumentParser(description="Prepare CausalFormer input")
    parser.add_argument("--diff", action="store_true",
                        help="Apply first-differencing x(t) - x(t-1)")
    parser.add_argument("--aggregate", type=int, default=None, metavar="N",
                        help="Aggregate to N-hourly means (e.g., 4 for 4-hourly)")
    parser.add_argument("--deseasonalize", action="store_true",
                        help="v1: remove hour-of-day and day-of-week seasonality (requires --aggregate)")
    parser.add_argument("--deseasonalize-v2", action="store_true", dest="deseasonalize_v2",
                        help="v2: remove hour-of-day, day-of-week, and month-of-year seasonality (requires --aggregate)")
    args = parser.parse_args()

    if args.deseasonalize and args.deseasonalize_v2:
        parser.error("--deseasonalize and --deseasonalize-v2 are mutually exclusive")
    if (args.deseasonalize or args.deseasonalize_v2) and not args.aggregate:
        parser.error("--deseasonalize / --deseasonalize-v2 require --aggregate (deseasonalization needs a DatetimeIndex)")

    df = load_dataset()
    validate_input(df)

    if args.aggregate:
        window = args.aggregate
        df = apply_aggregation(df, window)
        if args.diff:
            df = apply_differencing(df)
            output_csv = PROJECT_ROOT / "data" / "processed" / f"causalformer_input_agg{window}h_diff.csv"
            convert_and_save(df, output_csv, mode=f"{window}h-aggregated-first-differenced")
        elif args.deseasonalize:
            df = apply_deseasonalization(df, include_month=False)
            output_csv = PROJECT_ROOT / "data" / "processed" / f"causalformer_input_agg{window}h_ds.csv"
            convert_and_save(df, output_csv, mode=f"{window}h-aggregated-deseasonalized-v1")
        elif args.deseasonalize_v2:
            df = apply_deseasonalization(df, include_month=True)
            output_csv = PROJECT_ROOT / "data" / "processed" / f"causalformer_input_agg{window}h_ds_v2.csv"
            convert_and_save(df, output_csv, mode=f"{window}h-aggregated-deseasonalized-v2")
        else:
            output_csv = PROJECT_ROOT / "data" / "processed" / f"causalformer_input_agg{window}h.csv"
            convert_and_save(df, output_csv, mode=f"{window}h-aggregated")
    elif args.diff:
        df = apply_differencing(df)
        convert_and_save(df, OUTPUT_CSV_DIFF, mode="first-differenced")
    else:
        convert_and_save(df, OUTPUT_CSV_LEVELS, mode="levels")

    print("Done. Ready for CausalFormer.")


if __name__ == "__main__":
    main()
