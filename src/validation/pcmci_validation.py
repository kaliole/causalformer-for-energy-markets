"""PCMCI+ triangulation for WP5 validation pipeline.

Runs PCMCI+ independently on the DE-LU dataset, then checks which CausalFormer
candidate edges are supported by conditional independence testing.

Design decisions (see docs/methods_notes.md):
- Dataset: causalformer_input_agg4h.csv — same aggregation as CausalFormer, lags are comparable
- Max lag: 24 — matches CausalFormer's time_step (receptive field)
- CI test: ParCorr — linear, fast, appropriate for ~4000 timesteps at 4h aggregation
- Alpha: 0.05 — standard significance threshold
- tau_min=1 — skip contemporaneous edges; CausalFormer only produces lagged edges
- Self-loops excluded — PCMCI+ absorbs autoregression into conditioning (see methods_notes.md)

Outputs:
    results/metrics/pcmci_triangulation.csv  — one row per CausalFormer cross-variable edge
    results/metrics/pcmci_all_edges.csv      — full PCMCI+ significant edge set

Usage:
    uv run python src/validation/pcmci_validation.py
    uv run python src/validation/pcmci_validation.py --edges results/edges/delu_edges_raw.csv
    uv run python src/validation/pcmci_validation.py --alpha 0.01 --max-lag 24
"""

import argparse
import pathlib

import numpy as np
import pandas as pd
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "causalformer_input_agg4h_ds.csv"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
DEFAULT_EDGES = PROJECT_ROOT / "results" / "edges" / "delu_edges_raw.csv"

DEFAULT_ALPHA = 0.05
DEFAULT_MAX_LAG = 24


def load_data(data_path: pathlib.Path) -> tuple[np.ndarray, list[str]]:
    """Load the DE-LU dataset from CSV.

    Returns:
        data: float array of shape (T, N) — T timesteps, N variables
        var_names: list of variable names in column order
    """
    df = pd.read_csv(data_path)
    return df.values.astype(float), list(df.columns)


def run_pcmciplus(
    data: np.ndarray,
    var_names: list[str],
    max_lag: int,
    alpha: float,
) -> dict:
    """Run PCMCI+ with ParCorr on the dataset.

    Args:
        data: array of shape (T, N)
        var_names: variable names for the N columns
        max_lag: maximum lag to test (tau_max)
        alpha: significance level for the PC step (pc_alpha)

    Returns:
        tigramite results dict with keys p_matrix, val_matrix, graph, etc.
        p_matrix shape: (N, N, tau_max+1) — p_matrix[effect, cause, lag]
        val_matrix shape: (N, N, tau_max+1) — partial correlation values
    """
    dataframe = pp.DataFrame(data, var_names=var_names)
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=1)
    results = pcmci.run_pcmciplus(tau_min=1, tau_max=max_lag, pc_alpha=alpha)
    return results


def parse_pcmci_edges(
    results: dict,
    var_names: list[str],
    alpha: float,
) -> pd.DataFrame:
    """Extract all significant PCMCI+ edges (cross-variable only) into a DataFrame.

    Args:
        results: tigramite results dict
        var_names: variable names
        alpha: significance threshold for reporting edges

    Returns:
        DataFrame with columns: cause, effect, lag, pcmci_pvalue, pcmci_val
    """
    p_matrix = results["p_matrix"]
    val_matrix = results["val_matrix"]
    n_vars = len(var_names)
    rows = []
    for effect_idx in range(n_vars):
        for cause_idx in range(n_vars):
            if cause_idx == effect_idx:
                continue  # self-loops excluded
            for lag in range(1, p_matrix.shape[2]):
                pval = float(p_matrix[effect_idx, cause_idx, lag])
                val = float(val_matrix[effect_idx, cause_idx, lag])
                if pval < alpha:
                    rows.append({
                        "cause": var_names[cause_idx],
                        "effect": var_names[effect_idx],
                        "lag": lag,
                        "pcmci_pvalue": round(pval, 6),
                        "pcmci_val": round(val, 6),
                    })
    return pd.DataFrame(rows)


def triangulate(
    cf_edges: pd.DataFrame,
    p_matrix: np.ndarray,
    val_matrix: np.ndarray,
    var_names: list[str],
    alpha: float,
) -> pd.DataFrame:
    """Look up PCMCI+ support for each CausalFormer cross-variable edge.

    Self-loops (cause == effect) are skipped. Lag-0 edges are flagged as
    not testable (PCMCI+ uses tau_min=1).

    Args:
        cf_edges: DataFrame with columns cause, effect, lag from CausalFormer
        p_matrix: shape (N, N, tau_max+1), p_matrix[effect_idx, cause_idx, lag]
        val_matrix: shape (N, N, tau_max+1), partial correlation values
        var_names: variable names in index order
        alpha: significance threshold

    Returns:
        DataFrame with one row per cross-variable CausalFormer edge, with
        columns: cause, effect, lag, pcmci_pvalue, pcmci_val, pcmci_supported
    """
    name_to_idx = {name: i for i, name in enumerate(var_names)}
    rows = []
    for _, row in cf_edges.iterrows():
        cause, effect, lag = row["cause"], row["effect"], int(row["lag"])

        if cause == effect:
            continue  # self-loops not comparable with PCMCI+

        if cause not in name_to_idx or effect not in name_to_idx:
            print(f"  Warning: unknown variable in edge {cause} → {effect}, skipping")
            continue

        cause_idx = name_to_idx[cause]
        effect_idx = name_to_idx[effect]

        if lag == 0:
            # Contemporaneous edges: not tested (tau_min=1)
            rows.append({
                "cause": cause,
                "effect": effect,
                "lag": lag,
                "pcmci_pvalue": np.nan,
                "pcmci_val": np.nan,
                "pcmci_supported": False,
                "note": "lag=0 not tested (tau_min=1)",
            })
            continue

        if lag >= p_matrix.shape[2]:
            # Edge lag exceeds max_lag — outside tested range
            rows.append({
                "cause": cause,
                "effect": effect,
                "lag": lag,
                "pcmci_pvalue": np.nan,
                "pcmci_val": np.nan,
                "pcmci_supported": False,
                "note": f"lag={lag} exceeds max_lag={p_matrix.shape[2]-1}",
            })
            continue

        pval = float(p_matrix[effect_idx, cause_idx, lag])
        val = float(val_matrix[effect_idx, cause_idx, lag])
        rows.append({
            "cause": cause,
            "effect": effect,
            "lag": lag,
            "pcmci_pvalue": round(pval, 6),
            "pcmci_val": round(val, 6),
            "pcmci_supported": bool(pval < alpha),
            "note": "",
        })

    return pd.DataFrame(rows)


def main() -> None:
    """Run PCMCI+ triangulation and save results."""
    parser = argparse.ArgumentParser(description="WP5: PCMCI+ triangulation of CausalFormer edges")
    parser.add_argument("--edges", type=str, default=str(DEFAULT_EDGES),
                        help="CausalFormer edge CSV to triangulate (default: delu_edges_raw.csv)")
    parser.add_argument("--data", type=str, default=str(DATA_PATH),
                        help="Input dataset CSV (default: causalformer_input_agg4h_ds.csv — deseasonalized)")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="Significance threshold (default: 0.05)")
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG,
                        help="Maximum lag to test in PCMCI+ (default: 32)")
    args = parser.parse_args()

    edges_path = pathlib.Path(args.edges)
    data_path = pathlib.Path(args.data)

    print(f"\nWP5 — PCMCI+ Triangulation")
    print(f"Dataset : {data_path.name}")
    print(f"Edges   : {edges_path.name}")
    print(f"Max lag : {args.max_lag} steps × 4h = {args.max_lag * 4}h")
    print(f"Alpha   : {args.alpha}")
    print("=" * 60)

    # Load
    data, var_names = load_data(data_path)
    print(f"Data shape : {data.shape}  ({data.shape[0] * 4}h of data)")
    print(f"Variables  : {var_names}")

    cf_edges = pd.read_csv(edges_path)
    cross_edges = cf_edges[cf_edges["cause"] != cf_edges["effect"]]
    print(f"\nCausalFormer edges : {len(cf_edges)} total, "
          f"{len(cross_edges)} cross-variable, "
          f"{len(cf_edges) - len(cross_edges)} self-loops (excluded)")

    # Run PCMCI+
    print("\nRunning PCMCI+ (this may take several minutes)...")
    results = run_pcmciplus(data, var_names, args.max_lag, args.alpha)

    # Full PCMCI+ edge set
    pcmci_all = parse_pcmci_edges(results, var_names, args.alpha)
    print(f"\nPCMCI+ significant edges (cross-variable): {len(pcmci_all)}")

    # Triangulate CausalFormer edges
    triangulated = triangulate(
        cf_edges, results["p_matrix"], results["val_matrix"], var_names, args.alpha
    )
    n_supported = int(triangulated["pcmci_supported"].sum())
    n_testable = int(triangulated["pcmci_pvalue"].notna().sum())
    print(f"CausalFormer edges supported by PCMCI+: {n_supported}/{n_testable} testable cross-variable edges")

    # Save
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    tri_path = METRICS_DIR / "pcmci_triangulation.csv"
    triangulated.to_csv(tri_path, index=False)
    print(f"\nTriangulation table saved to : {tri_path}")

    all_path = METRICS_DIR / "pcmci_all_edges.csv"
    pcmci_all.to_csv(all_path, index=False)
    print(f"Full PCMCI+ edges saved to   : {all_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("Triangulation results (cross-variable edges only):")
    print(f"{'='*60}")
    if not triangulated.empty:
        print(triangulated.to_string(index=False))


if __name__ == "__main__":
    main()
