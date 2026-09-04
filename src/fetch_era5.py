"""Download ERA5 reanalysis data for the DE-LU energy thesis.

Downloads hourly data for the DE-LU bidding zone (bounding box), 2019–2021:
  - 2m temperature
  - 10m u-component of wind
  - 10m v-component of wind
  - Surface solar radiation downwards

The bounding box covers Germany and Luxembourg [55.06°N, 5.87°E, 47.27°N, 15.04°E].
Each variable is spatially averaged (simple mean across all grid points)
to produce one value per hour, matching the bidding-zone level of SMARD data.

Wind speed is computed per grid point (sqrt(u² + v²)) before spatial averaging,
to avoid cancellation of opposing wind directions.

Raw downloads (zip/NetCDF) are saved to data/raw/era5/.
Processed CSVs (timestamp, value) are saved to data/raw/era5/csv/.

Usage:
    python src/fetch_era5.py
    python src/fetch_era5.py --variable temperature
    python src/fetch_era5.py --variable wind_u wind_v
"""

import argparse
import pathlib
import zipfile

import cdsapi
import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# DE-LU bounding box: [north, west, south, east]
AREA = [55.06, 5.87, 47.27, 15.04]
YEARS = [2019, 2020, 2021]
DATASET = "reanalysis-era5-single-levels"

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5"
CSV_DIR = RAW_DIR / "csv"

# Each variable: our name → (ERA5 API name, NetCDF variable name, unit, conversion function)
VARIABLES = {
    "temperature": {
        "api_name": "2m_temperature",
        "nc_var": "t2m",
        "unit": "°C",
        "convert": lambda x: x - 273.15,  # Kelvin → Celsius
        "valid_range": (-40, 50),
    },
    "wind_u": {
        "api_name": "10m_u_component_of_wind",
        "nc_var": "u10",
        "unit": "m/s",
        "convert": None,
        "valid_range": (-30, 30),
    },
    "wind_v": {
        "api_name": "10m_v_component_of_wind",
        "nc_var": "v10",
        "unit": "m/s",
        "convert": None,
        "valid_range": (-30, 30),
    },
    "solar_radiation": {
        "api_name": "surface_solar_radiation_downwards",
        "nc_var": "ssrd",
        "unit": "W/m²",
        "convert": lambda x: x / 3600,  # J/m² per hour → W/m²
        "valid_range": (0, 1200),
    },
}

# Expected total rows: sum of hours across all years
EXPECTED_ROWS = sum(366 * 24 if y % 4 == 0 else 365 * 24 for y in YEARS)

# Minimum expected grid points for the Germany bounding box at 0.25° resolution
MIN_GRID_POINTS = 500

# Valid ranges for derived variables (not in VARIABLES dict)
DERIVED_RANGES = {
    "wind_speed": {"valid_range": (0, 40), "unit": "m/s"},
}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_variable(client: cdsapi.Client, var_name: str) -> list[pathlib.Path]:
    """Download one ERA5 variable (one file per year) and return paths."""
    var_config = VARIABLES[var_name]
    all_months = [f"{m:02d}" for m in range(1, 13)]
    all_days = [f"{d:02d}" for d in range(1, 32)]
    all_hours = [f"{h:02d}:00" for h in range(24)]

    file_paths = []
    for year in YEARS:
        # Check for either .nc or .zip (API may return either format)
        nc_path = RAW_DIR / f"era5_{var_name}_{year}.nc"
        zip_path = RAW_DIR / f"era5_{var_name}_{year}.zip"

        if nc_path.exists():
            print(f"  Already downloaded: {nc_path}")
            file_paths.append(nc_path)
            continue
        if zip_path.exists():
            print(f"  Already downloaded: {zip_path}")
            file_paths.append(zip_path)
            continue

        request = {
            "product_type": ["reanalysis"],
            "variable": [var_config["api_name"]],
            "year": [str(year)],
            "month": all_months,
            "day": all_days,
            "time": all_hours,
            "area": AREA,
            "data_format": "netcdf",
        }

        # Download — save as .nc since this dataset typically returns NetCDF directly
        download_path = nc_path
        print(f"  Requesting {var_config['api_name']} for {year} from CDS API...")
        client.retrieve(DATASET, request).download(str(download_path))
        print(f"  Saved: {download_path}")
        file_paths.append(download_path)

    return file_paths


# ---------------------------------------------------------------------------
# Extract and convert
# ---------------------------------------------------------------------------

def open_netcdf(file_path: pathlib.Path) -> xr.Dataset:
    """Open a NetCDF file, extracting from zip first if needed.

    Detects the actual file type regardless of extension, since the CDS API
    may return either a raw NetCDF or a zip containing one.
    """
    if zipfile.is_zipfile(file_path):
        with zipfile.ZipFile(file_path) as z:
            nc_files = [f for f in z.namelist() if f.endswith(".nc")]
            if not nc_files:
                raise ValueError(f"No .nc file in {file_path}. Contents: {z.namelist()}")
            if len(nc_files) > 1:
                raise ValueError(f"Multiple .nc files in {file_path}: {nc_files}")
            z.extract(nc_files[0], RAW_DIR)
            file_path = RAW_DIR / nc_files[0]
    return xr.open_dataset(file_path)


def open_multi_netcdf(file_paths: list[pathlib.Path]) -> xr.Dataset:
    """Open and concatenate multiple NetCDF files along the time dimension."""
    datasets = [open_netcdf(p) for p in file_paths]
    if len(datasets) == 1:
        return datasets[0]
    # Find the time dimension name
    time_dim = "valid_time" if "valid_time" in datasets[0].dims else "time"
    return xr.concat(datasets, dim=time_dim)


def validate_output(df: pd.DataFrame, var_name: str, n_grid_points: int) -> None:
    """Validate the processed output before writing to CSV.

    Checks:
    1. Row count matches expected hours for the configured years
    2. No NaN values after spatial averaging
    3. Grid point count is plausible for the Germany bounding box
    4. All values fall within the physically plausible range
    5. All time steps are exactly 1 hour apart (no gaps or duplicates)
    """
    errors = []

    # 1. Row count
    if len(df) != EXPECTED_ROWS:
        errors.append(
            f"Row count: expected {EXPECTED_ROWS}, got {len(df)} "
            f"(diff: {len(df) - EXPECTED_ROWS})"
        )

    # 2. NaN values
    n_nan = df["value"].isna().sum()
    if n_nan > 0:
        errors.append(f"NaN values: {n_nan} found")

    # 3. Grid point count
    if n_grid_points < MIN_GRID_POINTS:
        errors.append(
            f"Grid points: only {n_grid_points} (expected >= {MIN_GRID_POINTS}). "
            f"Bounding box may not have been applied correctly."
        )

    # 4. Value range — check both downloaded and derived variables
    range_config = VARIABLES.get(var_name) or DERIVED_RANGES.get(var_name)
    if range_config and "valid_range" in range_config:
        lo, hi = range_config["valid_range"]
        vmin, vmax = df["value"].min(), df["value"].max()
        if vmin < lo or vmax > hi:
            errors.append(
                f"Value range: [{vmin:.2f}, {vmax:.2f}] {range_config['unit']} "
                f"outside expected [{lo}, {hi}]"
            )

    # 5. Timestamp continuity
    if len(df) > 1:
        deltas = df["timestamp"].diff().dropna()
        expected_delta = pd.Timedelta("1h")
        bad_deltas = deltas[deltas != expected_delta]
        if len(bad_deltas) > 0:
            errors.append(
                f"Time gaps: {len(bad_deltas)} irregular intervals found. "
                f"Unique deltas: {deltas.value_counts().to_dict()}"
            )

    # Report results
    if errors:
        msg = f"Validation FAILED for {var_name}:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ValueError(msg)

    print(f"  Validation OK: {len(df)} rows, no NaN, {n_grid_points} grid points, "
          f"values in range, no time gaps")


def extract_to_csv(var_name: str, file_paths: list[pathlib.Path]) -> pathlib.Path:
    """Open NetCDF files, spatially average, and convert to a two-column CSV."""
    var_config = VARIABLES[var_name]
    csv_path = CSV_DIR / f"era5_{var_name}.csv"

    ds = open_multi_netcdf(file_paths)
    nc_var = var_config["nc_var"]

    if nc_var not in ds:
        available = list(ds.data_vars)
        raise ValueError(f"Variable '{nc_var}' not in NetCDF. Available: {available}")

    # Spatial mean: average across all grid points per timestep
    da = ds[nc_var]
    time_dim = "valid_time" if "valid_time" in da.dims else "time"
    spatial_dims = [d for d in da.dims if d != time_dim]
    da_mean = da.mean(dim=spatial_dims)

    n_points = 1
    for d in spatial_dims:
        n_points *= len(ds[d])
    print(f"  Averaging {n_points} grid points per timestep")

    values = da_mean.values.astype(float)

    if var_config["convert"] is not None:
        values = var_config["convert"](values)

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(da_mean[time_dim].values, utc=True),
        "value": values,
    })
    out = out.sort_values("timestamp").reset_index(drop=True)

    validate_output(out, var_name, n_points)

    out.to_csv(csv_path, index=False)

    print(f"  → {csv_path}: {len(out)} rows, "
          f"{out['timestamp'].min()} to {out['timestamp'].max()}, "
          f"mean={out['value'].mean():.2f} {var_config['unit']}")

    return csv_path


# ---------------------------------------------------------------------------
# Wind speed (derived)
# ---------------------------------------------------------------------------

def compute_wind_speed(u_paths: list[pathlib.Path], v_paths: list[pathlib.Path]) -> None:
    """Compute wind speed per grid point, then spatially average.

    wind_speed = mean_over_grid( sqrt(u² + v²) )

    This is done from the raw NetCDF files, not the already-averaged CSVs,
    because averaging u and v separately before computing speed would
    underestimate it (opposing directions cancel in the mean).
    """
    out_path = CSV_DIR / "era5_wind_speed.csv"

    ds_u = open_multi_netcdf(u_paths)
    ds_v = open_multi_netcdf(v_paths)

    # Compute speed per grid point, then average spatially
    speed = np.sqrt(ds_u["u10"] ** 2 + ds_v["v10"] ** 2)
    time_dim = "valid_time" if "valid_time" in speed.dims else "time"
    spatial_dims = [d for d in speed.dims if d != time_dim]
    speed_mean = speed.mean(dim=spatial_dims)

    n_points = 1
    for d in spatial_dims:
        n_points *= len(ds_u[d])

    out = pd.DataFrame({
        "timestamp": pd.to_datetime(speed_mean[time_dim].values, utc=True),
        "value": speed_mean.values.astype(float),
    })
    out = out.sort_values("timestamp").reset_index(drop=True)

    # Wind speed has its own valid range: always >= 0, and bounded above
    validate_output(out, "wind_speed", n_points)

    out.to_csv(out_path, index=False)
    print(f"  → {out_path}: {len(out)} rows, mean={out['value'].mean():.2f} m/s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download and process all ERA5 variables."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--variable", nargs="*", default=None,
                        help="Specific variables to download (default: all)")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    var_names = args.variable if args.variable else list(VARIABLES.keys())
    client = cdsapi.Client()

    downloaded = {}  # var_name → list of file paths
    for var_name in var_names:
        if var_name not in VARIABLES:
            print(f"Unknown variable: {var_name}. Options: {list(VARIABLES.keys())}")
            continue

        print(f"\n[{var_name}]")
        paths = download_variable(client, var_name)
        downloaded[var_name] = paths
        extract_to_csv(var_name, paths)

    # Derive wind speed if both components are available
    if "wind_u" in downloaded and "wind_v" in downloaded:
        print("\n[wind_speed] (derived)")
        compute_wind_speed(downloaded["wind_u"], downloaded["wind_v"])

    print("\nDone.")


if __name__ == "__main__":
    main()
