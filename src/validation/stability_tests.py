"""Stability and sensitivity tests for WP5 validation pipeline.

Tests whether CausalFormer's discovered edges are robust across different
conditions. Two tests are implemented:

4.1 Regime Robustness:
    Re-run CausalFormer on winter and summer subsets separately.
    Keep only edges that appear in both regimes.
    Why: checks whether edges reflect stable market relationships or are
    driven by one specific seasonal regime.
    Winter = December–February, Summer = June–August.

4.2 Temporal Stability (Block Bootstrap):
    Resample contiguous temporal blocks (2-week blocks = 84 timesteps at 4h)
    with replacement to create 10 bootstrap datasets of the same length.
    Re-run CausalFormer on each. Report per-edge stability scores
    (fraction of runs the edge appeared in). Edges with score >= 0.6 are stable.
    Why: time series are not i.i.d. — block resampling preserves autocorrelation
    and seasonal structure that simple shuffling would destroy.

Note: each CausalFormer run takes 1-2h. This module runs 12 total
(2 regime + 10 bootstrap). Designed for the final clean run overnight.

Usage:
    uv run python src/validation/stability_tests.py
    uv run python src/validation/stability_tests.py --test regime
    uv run python src/validation/stability_tests.py --test bootstrap
    uv run python src/validation/stability_tests.py --edges results/edges/delu_edges_raw.csv
"""

import argparse
import json
import os
import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CAUSALFORMER_DIR = PROJECT_ROOT / "CausalFormer"
AGG4H_DATA = PROJECT_ROOT / "data" / "processed" / "causalformer_input_agg4h_diff.csv"
AGG4H_DS_DATA = PROJECT_ROOT / "data" / "processed" / "causalformer_input_agg4h_ds.csv"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
DEFAULT_EDGES = PROJECT_ROOT / "results" / "edges" / "delu_edges_raw.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config_wp4.json"

COLUMNS = ["temperature", "wind_speed", "solar_radiation", "price",
           "load", "wind_generation", "solar_generation"]

WINTER_MONTHS = [12, 1, 2]
SUMMER_MONTHS = [6, 7, 8]

BLOCK_SIZE = 84      # 2 weeks × 6 timesteps/day = 84 timesteps at 4h aggregation
N_BOOTSTRAP = 10
STABILITY_THRESHOLD = 0.6  # edge must appear in >= 60% of runs to be considered stable
SEED = 42


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------

def load_regime_splits() -> dict[str, pd.DataFrame]:
    """Split the already-aggregated 4h dataset into winter and summer subsets.

    Loads causalformer_input_agg4h.csv and reconstructs timestamps from the
    known start date (2019-01-01) and 4h frequency. Filters by month.

    Returns:
        Dict with keys 'winter' and 'summer', each a DataFrame of shape
        (~540, 7), columns in COLUMNS order.
    """
    df = pd.read_csv(AGG4H_DATA)
    timestamps = pd.date_range(start="2019-01-01", periods=len(df), freq="4h")
    df.index = timestamps

    splits = {}
    for name, months in [("winter", WINTER_MONTHS), ("summer", SUMMER_MONTHS)]:
        subset = df[df.index.month.isin(months)].reset_index(drop=True)
        splits[name] = subset
        print(f"  {name}: {len(subset)} 4h rows (months {months})")
    return splits


def generate_block_bootstrap(df: pd.DataFrame, block_size: int = BLOCK_SIZE,
                              n_samples: int = N_BOOTSTRAP,
                              seed: int = SEED) -> list[pd.DataFrame]:
    """Generate block bootstrap samples by resampling contiguous blocks.

    Samples ceil(T/block_size) blocks with replacement and concatenates
    them, trimming to the original length T. This preserves autocorrelation
    and seasonal structure within each block.

    Args:
        df: input dataset, shape (T, N)
        block_size: number of consecutive timesteps per block (84 = 2 weeks at 4h)
        n_samples: number of bootstrap datasets to generate
        seed: random seed for reproducibility

    Returns:
        List of n_samples DataFrames each of shape (T, N).
    """
    rng = np.random.default_rng(seed)
    T = len(df)
    n_blocks_needed = int(np.ceil(T / block_size))
    max_start = T - block_size
    samples = []
    for i in range(n_samples):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        chunks = [df.iloc[s:s + block_size] for s in starts]
        sample = pd.concat(chunks, ignore_index=True).iloc[:T]
        samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# CausalFormer runner
# ---------------------------------------------------------------------------

def run_causalformer_on(df: pd.DataFrame, base_config_path: pathlib.Path,
                        label: str = "") -> list[tuple[int, int, int]]:
    """Save df to a temp file and run CausalFormer train + interpret on it.

    Args:
        df: dataset to run on (already in CausalFormer column order)
        base_config_path: base config to use for training
        label: human-readable label for logging (e.g. 'winter', 'bootstrap_3')

    Returns:
        List of (cause_idx, effect_idx, lag) tuples.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from run_causalformer import train, interpret, set_seed, setup_causalformer_imports

    # Save data to temp file inside CausalFormer/data/ so relative paths work
    tmp_data = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix=f"stability_{label}_",
        delete=False, dir=str(CAUSALFORMER_DIR / "data"),
    )
    df.to_csv(tmp_data.name, index=False)
    tmp_data.close()

    # Create temp config pointing to this file
    with open(base_config_path) as f:
        config = json.load(f)
    config["data_loader"]["args"]["data_dir"] = str(tmp_data.name)

    tmp_config = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"stability_{label}_",
        delete=False, dir=str(CAUSALFORMER_DIR),
    )
    json.dump(config, tmp_config, indent=4)
    tmp_config.close()

    try:
        os.chdir(PROJECT_ROOT)
        set_seed(SEED)
        setup_causalformer_imports()
        model_dir = train(tmp_config.name)
        edges = interpret(model_dir, tmp_config.name, bigdata=True)
    finally:
        try:
            os.unlink(tmp_data.name)
            os.unlink(tmp_config.name)
        except OSError:
            pass

    return edges


# ---------------------------------------------------------------------------
# Edge aggregation
# ---------------------------------------------------------------------------

def edges_to_pairs(edges: list[tuple[int, int, int]],
                   var_names: list[str]) -> set[tuple[str, str]]:
    """Convert edge tuples to a set of (cause, effect) name pairs, excluding self-loops."""
    return {
        (var_names[c], var_names[e])
        for c, e, _ in edges
        if c != e
    }


def compute_stability_scores(all_edge_sets: list[set[tuple[str, str]]],
                              real_pairs: set[tuple[str, str]]) -> pd.DataFrame:
    """Compute per-edge stability score across multiple runs.

    Args:
        all_edge_sets: list of (cause, effect) pair sets, one per run
        real_pairs: (cause, effect) pairs from the real WP4 run

    Returns:
        DataFrame with columns: cause, effect, stability_score, n_appearances, stable
        Rows cover all edges that appeared in at least one run.
    """
    n_runs = len(all_edge_sets)
    all_pairs = set().union(*all_edge_sets)

    rows = []
    for cause, effect in sorted(all_pairs):
        count = sum(1 for s in all_edge_sets if (cause, effect) in s)
        score = count / n_runs
        rows.append({
            "cause": cause,
            "effect": effect,
            "stability_score": round(score, 2),
            "n_appearances": count,
            "n_runs": n_runs,
            "in_real_edges": (cause, effect) in real_pairs,
            "stable": score >= STABILITY_THRESHOLD,
        })

    if not rows:
        return pd.DataFrame(columns=["cause", "effect", "stability_score", "n_appearances", "n_runs", "in_real_edges", "stable"])
    return pd.DataFrame(rows).sort_values("stability_score", ascending=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run stability tests and save results."""
    parser = argparse.ArgumentParser(description="WP5: Stability tests for CausalFormer edges")
    parser.add_argument("--edges", type=str, default=str(DEFAULT_EDGES),
                        help="Real CausalFormer edge CSV (default: delu_edges_raw.csv)")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Base CausalFormer config for stability runs")
    parser.add_argument("--test", type=str, default="both",
                        choices=["regime", "bootstrap", "both"],
                        help="Which stability test to run (default: both)")
    args = parser.parse_args()

    edges_path = pathlib.Path(args.edges)
    config_path = pathlib.Path(args.config)

    print(f"\nWP5 — Stability Tests")
    print(f"Real edges : {edges_path.name}")
    print(f"Test(s)    : {args.test}")
    print("=" * 60)

    real_edges = pd.read_csv(edges_path)
    var_names = COLUMNS
    real_pairs = {
        (row["cause"], row["effect"])
        for _, row in real_edges.iterrows()
        if row["cause"] != row["effect"]
    }
    print(f"Real cross-variable edges: {len(real_pairs)}")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    tests_to_run = ["regime", "bootstrap"] if args.test == "both" else [args.test]

    # -----------------------------------------------------------------------
    # 4.1 Regime Robustness
    # -----------------------------------------------------------------------
    if "regime" in tests_to_run:
        print(f"\n{'='*60}")
        print("4.1 Regime Robustness (winter vs summer)")
        print(f"{'='*60}")

        splits = load_regime_splits()
        regime_edge_sets = {}

        for regime, df_regime in splits.items():
            print(f"\nRunning CausalFormer on {regime} subset ({len(df_regime)} timesteps)...")
            edge_tuples = run_causalformer_on(df_regime, config_path, label=regime)
            regime_edge_sets[regime] = edges_to_pairs(edge_tuples, var_names)
            print(f"  {regime}: {len(regime_edge_sets[regime])} cross-variable edges found")

        # Intersection: edges present in both regimes
        robust_edges = regime_edge_sets["winter"] & regime_edge_sets["summer"]
        winter_only = regime_edge_sets["winter"] - regime_edge_sets["summer"]
        summer_only = regime_edge_sets["summer"] - regime_edge_sets["winter"]

        print(f"\nResults:")
        print(f"  Winter edges     : {len(regime_edge_sets['winter'])}")
        print(f"  Summer edges     : {len(regime_edge_sets['summer'])}")
        print(f"  Both (robust)    : {len(robust_edges)}")
        print(f"  Winter only      : {len(winter_only)}")
        print(f"  Summer only      : {len(summer_only)}")

        # Build result DataFrame
        all_regime_pairs = regime_edge_sets["winter"] | regime_edge_sets["summer"]
        regime_rows = []
        for cause, effect in sorted(all_regime_pairs):
            regime_rows.append({
                "cause": cause,
                "effect": effect,
                "in_winter": (cause, effect) in regime_edge_sets["winter"],
                "in_summer": (cause, effect) in regime_edge_sets["summer"],
                "in_real_edges": (cause, effect) in real_pairs,
                "regime_robust": (cause, effect) in robust_edges,
            })
        regime_df = pd.DataFrame(regime_rows).sort_values(
            ["regime_robust", "in_real_edges"], ascending=False
        )

        out_path = METRICS_DIR / "stability_regime.csv"
        regime_df.to_csv(out_path, index=False)
        print(f"\nRegime results saved to: {out_path}")

        if robust_edges:
            print("\nRegime-robust edges:")
            for cause, effect in sorted(robust_edges):
                marker = " ← in real WP4 edges" if (cause, effect) in real_pairs else ""
                print(f"  {cause} → {effect}{marker}")
        else:
            print("\nNo edges survived both regimes.")

    # -----------------------------------------------------------------------
    # 4.2 Temporal Stability (Block Bootstrap)
    # -----------------------------------------------------------------------
    if "bootstrap" in tests_to_run:
        print(f"\n{'='*60}")
        print(f"4.2 Temporal Stability (block bootstrap)")
        print(f"Block size: {BLOCK_SIZE} timesteps ({BLOCK_SIZE * 4}h = 2 weeks)")
        print(f"Samples:    {N_BOOTSTRAP}")
        print(f"Threshold:  {STABILITY_THRESHOLD:.0%}")
        print(f"{'='*60}")

        df_agg = pd.read_csv(AGG4H_DS_DATA)
        bootstrap_samples = generate_block_bootstrap(df_agg, BLOCK_SIZE, N_BOOTSTRAP, SEED)
        bootstrap_edge_sets = []

        for i, sample in enumerate(bootstrap_samples):
            print(f"\nRunning CausalFormer on bootstrap sample {i+1}/{N_BOOTSTRAP}...")
            edge_tuples = run_causalformer_on(sample, config_path, label=f"bootstrap_{i}")
            bootstrap_edge_sets.append(edges_to_pairs(edge_tuples, var_names))
            print(f"  Sample {i+1}: {len(bootstrap_edge_sets[-1])} cross-variable edges found")

        scores_df = compute_stability_scores(bootstrap_edge_sets, real_pairs)

        n_stable = scores_df["stable"].sum()
        n_stable_real = (scores_df["stable"] & scores_df["in_real_edges"]).sum()
        print(f"\nResults:")
        print(f"  Stable edges (score >= {STABILITY_THRESHOLD:.0%})   : {n_stable}")
        print(f"  Stable edges also in real WP4 run : {n_stable_real}")

        out_path = METRICS_DIR / "stability_bootstrap.csv"
        scores_df.to_csv(out_path, index=False)
        print(f"\nBootstrap stability scores saved to: {out_path}")
        print(f"\n{'='*60}")
        print("Bootstrap stability scores (top edges):")
        print(f"{'='*60}")
        print(scores_df.to_string(index=False))


if __name__ == "__main__":
    main()
