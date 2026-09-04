"""Granger causality triangulation for WP5 validation pipeline.

Runs pairwise Granger causality tests on all candidate CausalFormer
cross-variable edges as an additional validation signal.

Data design decisions:
- Input: causalformer_input_agg4h.csv (original 4h-aggregated data)
- Preprocessing: daily differencing via create_differenced_view(method="daily")
  (shift=6 at 4h resolution, i.e. x(t) − x(t−24h)). Daily differencing
  removes the 24h seasonal cycle while preserving cross-variable structure,
  and satisfies the stationarity requirement of Granger tests better than
  the raw levels.
- Test: statsmodels grangercausalitytests, ssr_chi2test p-value
- Max lag: 24 (matches CausalFormer time_step / PCMCI+ max_lag)
- Alpha: 0.05 (standard significance threshold)

For each CausalFormer cross-variable edge (cause → effect):
- Run grangercausalitytests(data[[effect, cause]], maxlag=max_lag)
- Record the minimum p-value across all tested lags (best_lag = argmin p-value)
- granger_supported = True if min p-value < alpha

Output:
    results/metrics/granger_triangulation.csv — one row per CausalFormer
        cross-variable edge, columns: cause, effect, best_lag,
        granger_pvalue, granger_supported

Usage:
    uv run python src/validation/granger.py
    uv run python src/validation/granger.py --edges results/edges/delu_edges_raw.csv
    uv run python src/validation/granger.py --alpha 0.01 --max-lag 24
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests

# Allow imports from src/ whether run directly or imported as a module
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from preprocessing.differencing import create_differenced_view

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "causalformer_input_agg4h.csv"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
DEFAULT_EDGES = PROJECT_ROOT / "results" / "edges" / "delu_edges_raw.csv"

DEFAULT_ALPHA = 0.05
DEFAULT_MAX_LAG = 24


def load_data(data_path: pathlib.Path) -> tuple[pd.DataFrame, list[str]]:
    """Load and daily-difference the DE-LU dataset.

    Applies daily differencing (shift=6 at 4h resolution) via
    create_differenced_view to satisfy the stationarity requirement of
    Granger causality tests. NaN rows from the shift are dropped.

    Returns:
        data: DataFrame of daily-differenced values with original column names
        var_names: list of variable names in column order
    """
    df = pd.read_csv(data_path)
    var_names = list(df.columns)
    data = create_differenced_view(df, var_names, method="daily")
    return data, var_names


def run_granger_tests(
    cf_edges: pd.DataFrame,
    data: pd.DataFrame,
    var_names: list[str],
    max_lag: int,
    alpha: float,
) -> pd.DataFrame:
    """Run pairwise Granger causality tests for each CausalFormer cross-variable edge.

    For each edge (cause → effect), runs grangercausalitytests with the
    ssr_chi2test statistic up to max_lag. Records the minimum p-value across
    all tested lags.

    Args:
        cf_edges: CausalFormer edge DataFrame with columns cause, effect, lag
        data: daily-differenced DataFrame with original column names
        var_names: list of variable names (used for membership checks)
        max_lag: maximum lag to pass to grangercausalitytests
        alpha: significance threshold for granger_supported

    Returns:
        DataFrame with columns: cause, effect, best_lag, granger_pvalue,
        granger_supported — one row per cross-variable CausalFormer edge
    """
    rows = []
    for _, row in cf_edges.iterrows():
        cause, effect = row["cause"], row["effect"]

        if cause == effect:
            continue

        if cause not in data.columns or effect not in data.columns:
            print(f"  Warning: unknown variable in edge {cause} → {effect}, skipping")
            continue

        # grangercausalitytests expects column 0 = dependent (effect), column 1 = cause
        xy = data[[effect, cause]].values

        try:
            results = grangercausalitytests(xy, maxlag=max_lag, verbose=False)
        except Exception as e:
            print(f"  Warning: Granger test failed for {cause} → {effect}: {e}")
            rows.append({
                "cause": cause,
                "effect": effect,
                "best_lag": np.nan,
                "granger_pvalue": np.nan,
                "granger_supported": False,
            })
            continue

        # ssr_chi2test returns (chi2_stat, pvalue, df) — index 1 is the p-value
        pvals = {lag: results[lag][0]["ssr_chi2test"][1] for lag in results}
        best_lag = min(pvals, key=pvals.get)
        min_pval = pvals[best_lag]

        rows.append({
            "cause": cause,
            "effect": effect,
            "best_lag": int(best_lag),
            "granger_pvalue": round(float(min_pval), 6),
            "granger_supported": bool(min_pval < alpha),
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Run Granger causality triangulation and save results."""
    parser = argparse.ArgumentParser(
        description="WP5: Granger causality triangulation of CausalFormer edges"
    )
    parser.add_argument("--edges", type=str, default=str(DEFAULT_EDGES),
                        help="CausalFormer edge CSV to triangulate (default: delu_edges_raw.csv)")
    parser.add_argument("--data", type=str, default=str(DATA_PATH),
                        help="Input dataset CSV (default: causalformer_input_agg4h.csv)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Significance threshold (default: 0.05)")
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG,
                        help="Maximum lag to test (default: 32, matches CausalFormer time_step)")
    args = parser.parse_args()

    edges_path = pathlib.Path(args.edges)
    data_path = pathlib.Path(args.data)

    print(f"\nWP5 — Granger Causality Triangulation")
    print(f"Dataset : {data_path.name} (daily-differenced at 4h resolution)")
    print(f"Edges   : {edges_path.name}")
    print(f"Max lag : {args.max_lag} steps × 4h = {args.max_lag * 4}h")
    print(f"Alpha   : {args.alpha}")
    print("=" * 60)

    # Load
    data, var_names = load_data(data_path)
    print(f"Data shape (after differencing): {data.shape}  ({data.shape[0] * 4}h of data)")
    print(f"Variables  : {var_names}")

    cf_edges = pd.read_csv(edges_path)
    cross_edges = cf_edges[cf_edges["cause"] != cf_edges["effect"]]
    print(f"\nCausalFormer edges : {len(cf_edges)} total, "
          f"{len(cross_edges)} cross-variable, "
          f"{len(cf_edges) - len(cross_edges)} self-loops (excluded)")

    # Run Granger tests
    print("\nRunning Granger causality tests...")
    results = run_granger_tests(cf_edges, data, var_names, args.max_lag, args.alpha)

    n_supported = int(results["granger_supported"].sum())
    n_tested = len(results)
    print(f"CausalFormer edges supported by Granger: {n_supported}/{n_tested} cross-variable edges")

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = METRICS_DIR / "granger_triangulation.csv"
    results.to_csv(out_path, index=False)
    print(f"\nGranger triangulation saved to: {out_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("Granger triangulation results (cross-variable edges only):")
    print(f"{'='*60}")
    if not results.empty:
        print(results.to_string(index=False))


if __name__ == "__main__":
    main()
