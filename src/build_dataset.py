"""Build the merged DE-LU dataset for WP1.

Combines ERA5 weather variables and SMARD electricity market variables
into a single hourly dataset for 2019–2021.

ERA5 (weather, spatial mean over the DE-LU bounding box [55.06°N, 5.87°E, 47.27°N, 15.04°E]):
  - temperature (°C)
  - wind_speed (m/s)
  - solar_radiation (W/m²)

SMARD (electricity market, DE / DE-LU):
  - price (EUR/MWh)
  - load (MWh)
  - wind (MWh) — onshore + offshore combined
  - solar (MWh)

Output: data/processed/delu_dataset.csv
  - 26,304 rows (1096 days × 24 hours, 2020 is a leap year)
  - 7 columns + timestamp index

Usage:
    python src/build_dataset.py
"""

import pathlib

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
ERA5_DIR = PROJECT_ROOT / "data" / "raw" / "era5" / "csv"
SMARD_DIR = PROJECT_ROOT / "data" / "raw" / "smard" / "csv"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "delu_dataset.csv"

# ---------------------------------------------------------------------------
# Source files → column names
# ---------------------------------------------------------------------------

SOURCES = {
    # ERA5 weather
    ERA5_DIR / "era5_temperature.csv": "temperature",
    ERA5_DIR / "era5_wind_speed.csv": "wind_speed",
    ERA5_DIR / "era5_solar_radiation.csv": "solar_radiation",
    # SMARD electricity market
    SMARD_DIR / "smard_price.csv": "price",
    SMARD_DIR / "smard_load.csv": "load",
    SMARD_DIR / "smard_wind.csv": "wind_generation",
    SMARD_DIR / "smard_solar.csv": "solar_generation",
}


# ---------------------------------------------------------------------------
# Load and merge
# ---------------------------------------------------------------------------

def load_sources() -> pd.DataFrame:
    """Load all source CSVs and merge on timestamp."""
    dfs = []
    for path, col_name in SOURCES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing: {path}")
        df = pd.read_csv(path, parse_dates=["timestamp"])
        df = df.set_index("timestamp").rename(columns={"value": col_name})
        dfs.append(df)
        print(f"  Loaded {path.name} → {col_name}: {len(df)} rows")

    merged = pd.concat(dfs, axis=1)
    return merged


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def run_checks(df: pd.DataFrame) -> None:
    """Run quality checks and print diagnostics."""
    print(f"\nShape: {df.shape}")
    print(f"Period: {df.index.min()} to {df.index.max()}")
    print(f"Frequency: {pd.infer_freq(df.index)}")

    # Missing values
    missing = df.isna().sum()
    total_missing = missing.sum()
    print(f"\nMissing values per column:")
    for col, n in missing.items():
        status = "OK" if n == 0 else f"MISSING {n}"
        print(f"  {col}: {status}")

    if total_missing > 0:
        print(f"\n  Total missing: {total_missing}")
        # Forward-fill small gaps (up to 3 hours)
        before = df.isna().sum().sum()
        df.ffill(limit=3, inplace=True)
        after = df.isna().sum().sum()
        filled = before - after
        print(f"  Forward-filled {filled} values (limit=3h)")
        if after > 0:
            print(f"  Still missing after ffill: {after}")
            # Report which rows still have gaps
            gap_rows = df[df.isna().any(axis=1)]
            print(f"  Rows with remaining gaps: {len(gap_rows)}")
            print(gap_rows.head())

    # Time gaps
    deltas = df.index.to_series().diff().dropna()
    unique_deltas = deltas.value_counts()
    if len(unique_deltas) == 1 and unique_deltas.index[0] == pd.Timedelta("1h"):
        print("\nTime gaps: OK (all 1h)")
    else:
        print("\nTime gaps: WARNING — irregular intervals found:")
        print(unique_deltas)

    # Expected row count
    expected = (365 + 366 + 365) * 24  # 2019 + 2020 (leap) + 2021
    actual = len(df)
    if actual == expected:
        print(f"\nRow count: OK ({actual} = 1096 days × 24h)")
    else:
        print(f"\nRow count: WARNING — expected {expected}, got {actual} "
              f"(diff: {actual - expected})")

    # Basic sanity checks
    print("\nSanity checks:")
    checks = [
        ("temperature", -40, 50, "°C"),
        ("wind_speed", 0, 40, "m/s"),
        ("solar_radiation", 0, 1200, "W/m²"),
        ("price", -500, 1000, "EUR/MWh"),
        ("load", 10000, 90000, "MWh"),
        ("wind_generation", 0, 60000, "MWh"),
        ("solar_generation", 0, 50000, "MWh"),
    ]
    for col, lo, hi, unit in checks:
        if col not in df.columns:
            continue
        vmin, vmax = df[col].min(), df[col].max()
        ok = vmin >= lo and vmax <= hi
        status = "OK" if ok else "CHECK"
        print(f"  {col}: [{vmin:.1f}, {vmax:.1f}] {unit} — {status}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Build and save the merged dataset."""
    print("Loading source files...")
    df = load_sources()

    print("\nRunning quality checks...")
    run_checks(df)

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")
    print(f"  {len(df)} rows × {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Quick summary stats
    print("\nColumn statistics:")
    print(df.describe().round(2).to_string())


if __name__ == "__main__":
    main()
