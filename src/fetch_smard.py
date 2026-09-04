"""Download electricity market data from SMARD.de (Bundesnetzagentur).

Downloads hourly data for Germany, 2019–2021:
  - Day-ahead prices (DE-LU bidding zone)
  - Total electricity load (DE)
  - Wind generation (onshore + offshore, combined)
  - Solar generation (DE)

No API key required. Raw data saved to data/raw/smard/csv/ as
two-column CSVs (timestamp, value).

Usage:
    python src/fetch_smard.py
    python src/fetch_smard.py --variable price
    python src/fetch_smard.py --variable wind solar
"""

import argparse
import pathlib
import time

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.smard.de/app/chart_data"
RESOLUTION = "hour"

TARGET_START = pd.Timestamp("2019-01-01", tz="UTC")
TARGET_END = pd.Timestamp("2022-01-01", tz="UTC")  # exclusive

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "smard"
CSV_DIR = RAW_DIR / "csv"

# SMARD filter IDs and regions
VARIABLES = {
    "price": {
        "filter_id": 4169,
        "region": "DE-LU",
        "unit": "EUR/MWh",
        "description": "Day-ahead electricity price",
    },
    "load": {
        "filter_id": 410,
        "region": "DE",
        "unit": "MWh",
        "description": "Total electricity load",
    },
    "wind_onshore": {
        "filter_id": 4067,
        "region": "DE",
        "unit": "MWh",
        "description": "Wind onshore generation",
    },
    "wind_offshore": {
        "filter_id": 1225,
        "region": "DE",
        "unit": "MWh",
        "description": "Wind offshore generation",
    },
    "solar": {
        "filter_id": 4068,
        "region": "DE",
        "unit": "MWh",
        "description": "Solar generation",
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_chunk_timestamps(filter_id: int, region: str) -> list[int]:
    """Fetch the available chunk timestamps for a given filter/region."""
    url = f"{BASE_URL}/{filter_id}/{region}/index_{RESOLUTION}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["timestamps"]


def fetch_chunk(filter_id: int, region: str, ts: int) -> list[list]:
    """Fetch a single data chunk and return its series."""
    url = (f"{BASE_URL}/{filter_id}/{region}/"
           f"{filter_id}_{region}_{RESOLUTION}_{ts}.json")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["series"]


def fetch_variable(filter_id: int, region: str, **_kwargs) -> pd.DataFrame:
    """Fetch all chunks that overlap with the target window."""
    timestamps = get_chunk_timestamps(filter_id, region)

    start_ms = int(TARGET_START.timestamp() * 1000)
    end_ms = int(TARGET_END.timestamp() * 1000)

    # Select chunks that could contain data in our window
    relevant = sorted([t for t in timestamps if t < end_ms])
    relevant = [t for i, t in enumerate(relevant)
                if i + 1 >= len(relevant) or relevant[i + 1] > start_ms]

    # Include the chunk starting just before our window
    all_before = [t for t in timestamps if t <= start_ms]
    if all_before:
        first_chunk = max(all_before)
        if first_chunk not in relevant:
            relevant.insert(0, first_chunk)
    relevant = sorted(set(relevant))

    print(f"    Downloading {len(relevant)} weekly chunks...")
    all_rows = []
    for ts in relevant:
        rows = fetch_chunk(filter_id, region, ts)
        all_rows.extend(rows)
        time.sleep(0.1)  # Be polite to the API

    df = pd.DataFrame(all_rows, columns=["timestamp_ms", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    df = df.drop(columns=["timestamp_ms"])
    df = df.dropna(subset=["value"])
    df = df[(df["timestamp"] >= TARGET_START) & (df["timestamp"] < TARGET_END)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "value"]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Fetch all variables and save as CSVs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", nargs="*", default=None,
                        help="Specific variables to download (default: all)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    var_names = args.variable if args.variable else list(VARIABLES.keys())

    for var_name in var_names:
        if var_name not in VARIABLES:
            print(f"Unknown variable: {var_name}. "
                  f"Options: {list(VARIABLES.keys())}")
            continue

        var_config = VARIABLES[var_name]
        csv_path = CSV_DIR / f"smard_{var_name}.csv"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"[{var_name}] Already exists: {len(df)} rows → {csv_path.name}")
            continue

        print(f"[{var_name}] {var_config['description']} "
              f"(filter {var_config['filter_id']})...")
        df = fetch_variable(**var_config)
        df.to_csv(csv_path, index=False)
        print(f"  → {csv_path.name}: {len(df)} rows, "
              f"{df['timestamp'].min()} to {df['timestamp'].max()}")

    # Derive combined wind = onshore + offshore
    on_path = CSV_DIR / "smard_wind_onshore.csv"
    off_path = CSV_DIR / "smard_wind_offshore.csv"
    combined_path = CSV_DIR / "smard_wind.csv"

    if on_path.exists() and off_path.exists() and not combined_path.exists():
        print("\n[wind] Combining onshore + offshore...")
        on = pd.read_csv(on_path, parse_dates=["timestamp"])
        off = pd.read_csv(off_path, parse_dates=["timestamp"])
        merged = pd.merge(on, off, on="timestamp", suffixes=("_on", "_off"))
        merged["value"] = merged["value_on"] + merged["value_off"]
        merged = merged[["timestamp", "value"]]
        merged.to_csv(combined_path, index=False)
        print(f"  → {combined_path.name}: {len(merged)} rows, "
              f"mean={merged['value'].mean():.0f} MWh")

    # Summary
    print("\nSummary:")
    for f in sorted(CSV_DIR.glob("smard_*.csv")):
        df = pd.read_csv(f)
        print(f"  {f.name}: {len(df)} rows")

    print("\nDone.")


if __name__ == "__main__":
    main()
