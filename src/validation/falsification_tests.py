"""Falsification tests for WP5 validation pipeline.

Tests whether CausalFormer's discovered edges are spurious by re-running the model
on corrupted versions of the data. Real causal edges should disappear when the
temporal structure is broken; spurious ones will persist.

Two corruption methods:
- Time-shift placebo: shift all non-price variables forward by 500 steps,
  breaking temporal alignment between price and all other variables.
  Any edge that survives cannot reflect a real causal relationship.
- Phase randomization: randomize Fourier phases of all variables, preserving
  each variable's marginal distribution and autocorrelation but destroying
  all cross-series correlations. Should yield near-zero cross-variable edges.

Pass criterion: near-zero overlap between real WP4 edges and placebo edges.

Note: this module re-runs CausalFormer (train + interpret) on corrupted data,
which takes 1-2 hours per test. Designed to run as part of the final clean run,
not during development.

Usage:
    uv run python src/validation/falsification_tests.py
    uv run python src/validation/falsification_tests.py --edges results/edges/delu_edges_raw.csv
    uv run python src/validation/falsification_tests.py --test timeshift
    uv run python src/validation/falsification_tests.py --test phaserand
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
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "causalformer_input_agg4h_diff.csv"
DS_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "causalformer_input_agg4h_ds.csv"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
DEFAULT_EDGES = PROJECT_ROOT / "results" / "edges" / "delu_edges_raw.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config_wp4.json"

TIMESHIFT_STEPS = 500   # 500 × 4h = 83 days — well beyond any real causal lag
PRICE_COL = "price"
SEED = 42


# ---------------------------------------------------------------------------
# Corruption functions
# ---------------------------------------------------------------------------

def apply_timeshift(df: pd.DataFrame, shift: int = TIMESHIFT_STEPS,
                    target_col: str = PRICE_COL) -> pd.DataFrame:
    """Shift all non-target variables forward by `shift` steps.

    Creates a dataset where the target variable (price) is from timesteps
    0 to T-shift, and all other variables are from timesteps shift to T.
    The two groups are no longer from the same time period, so any edges
    found between them are by definition spurious.

    Args:
        df: input dataset with headers, shape (T, N)
        shift: number of timesteps to shift non-target variables forward
        target_col: the variable to keep anchored in time (default: price)

    Returns:
        Corrupted DataFrame of shape (T-shift, N), column order preserved.
    """
    n = len(df)
    target = df[[target_col]].iloc[:n - shift].reset_index(drop=True)
    others = df.drop(columns=[target_col]).iloc[shift:].reset_index(drop=True)
    corrupted = pd.concat([target, others], axis=1)
    return corrupted[df.columns]  # restore original column order


def apply_phase_randomization(df: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    """Randomize Fourier phases of all variables independently.

    Preserves each variable's marginal distribution and autocorrelation
    structure (amplitude spectrum) but destroys all cross-series correlations.
    Running causal discovery on this data should yield near-zero edges.

    Args:
        df: input dataset with headers, shape (T, N)
        seed: random seed for reproducibility

    Returns:
        Phase-randomized DataFrame of same shape, same column order.
    """
    rng = np.random.default_rng(seed)
    result = df.copy()
    for col in df.columns:
        x = df[col].values
        fft = np.fft.rfft(x)
        # Randomize phases, keep amplitudes intact
        random_phases = rng.uniform(0, 2 * np.pi, len(fft))
        fft_randomized = np.abs(fft) * np.exp(1j * random_phases)
        # Keep DC component (the mean) unchanged to preserve the variable's level
        fft_randomized[0] = fft[0]
        result[col] = np.fft.irfft(fft_randomized, n=len(x))
    return result


# ---------------------------------------------------------------------------
# CausalFormer runner on corrupted data
# ---------------------------------------------------------------------------

def make_temp_config(base_config_path: pathlib.Path,
                     corrupted_data_path: pathlib.Path) -> str:
    """Create a temporary config pointing to the corrupted dataset.

    Returns path to the temp config file (caller is responsible for cleanup).
    """
    with open(base_config_path) as f:
        config = json.load(f)

    # Use absolute path so CausalFormer can find it after cd'ing into its dir
    config["data_loader"]["args"]["data_dir"] = str(corrupted_data_path)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="falsification_",
        delete=False, dir=str(CAUSALFORMER_DIR),
    )
    json.dump(config, tmp, indent=4)
    tmp.close()
    return tmp.name


def run_causalformer_on(corrupted_df: pd.DataFrame, var_names: list[str],
                        base_config_path: pathlib.Path) -> list[tuple[int, int, int]]:
    """Save corrupted data to a temp file and run CausalFormer on it.

    Reuses train() and interpret() from run_causalformer.py.

    Args:
        corrupted_df: the corrupted dataset (time-shifted or phase-randomized)
        var_names: variable names in column order
        base_config_path: path to the base config to use for training

    Returns:
        List of (cause_idx, effect_idx, lag) tuples from CausalFormer.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from run_causalformer import train, interpret, set_seed, setup_causalformer_imports

    # Save corrupted data to a temp CSV inside CausalFormer/ so relative paths work
    tmp_data = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="falsification_data_",
        delete=False, dir=str(CAUSALFORMER_DIR / "data"),
    )
    corrupted_df.to_csv(tmp_data.name, index=False)
    tmp_data.close()

    config_path = make_temp_config(base_config_path, pathlib.Path(tmp_data.name))

    try:
        os.chdir(PROJECT_ROOT)
        set_seed(SEED)
        setup_causalformer_imports()
        model_dir = train(config_path)
        edges = interpret(model_dir, config_path, bigdata=True)
    finally:
        try:
            os.unlink(tmp_data.name)
            os.unlink(config_path)
        except OSError:
            pass

    return edges


# ---------------------------------------------------------------------------
# Edge comparison
# ---------------------------------------------------------------------------

def compare_edges(real_edges: pd.DataFrame,
                  placebo_edges: list[tuple[int, int, int]],
                  var_names: list[str]) -> dict:
    """Compare real CausalFormer edges to placebo edges.

    Checks for overlapping directed pairs (cause, effect) regardless of lag,
    and for exact matches (cause, effect, lag).

    Args:
        real_edges: DataFrame with columns cause, effect, lag from WP4
        placebo_edges: list of (cause_idx, effect_idx, lag) from the corrupted run
        var_names: variable names in index order

    Returns:
        Dict with counts and lists of overlapping edges.
    """
    # Convert placebo edge indices to names, exclude self-loops
    placebo_rows = [
        {"cause": var_names[c], "effect": var_names[e], "lag": lag}
        for c, e, lag in placebo_edges if c != e
    ]
    placebo_df = pd.DataFrame(placebo_rows) if placebo_rows else pd.DataFrame(
        columns=["cause", "effect", "lag"]
    )

    # Cross-variable real edges only
    real_cross = real_edges[real_edges["cause"] != real_edges["effect"]].copy()

    # Pair-level overlap (cause, effect — any lag)
    real_pairs = set(zip(real_cross["cause"], real_cross["effect"]))
    placebo_pairs = set(zip(placebo_df["cause"], placebo_df["effect"])) if not placebo_df.empty else set()
    pair_overlap = real_pairs & placebo_pairs

    # Exact overlap (cause, effect, lag)
    real_triples = set(zip(real_cross["cause"], real_cross["effect"], real_cross["lag"]))
    placebo_triples = set(zip(placebo_df["cause"], placebo_df["effect"], placebo_df["lag"])) if not placebo_df.empty else set()
    exact_overlap = real_triples & placebo_triples

    return {
        "n_real_edges": len(real_cross),
        "n_placebo_edges": len(placebo_df),
        "n_pair_overlap": len(pair_overlap),
        "n_exact_overlap": len(exact_overlap),
        "pair_overlap_rate": len(pair_overlap) / len(real_pairs) if real_pairs else 0.0,
        "overlapping_pairs": sorted(pair_overlap),
        "overlapping_triples": sorted(exact_overlap),
        "placebo_edges": placebo_df,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run falsification tests and save results."""
    parser = argparse.ArgumentParser(description="WP5: Falsification tests for CausalFormer edges")
    parser.add_argument("--edges", type=str, default=str(DEFAULT_EDGES),
                        help="Real CausalFormer edge CSV (default: delu_edges_raw.csv)")
    parser.add_argument("--data", type=str, default=str(DATA_PATH),
                        help="Dataset for time-shift placebo (default: causalformer_input_agg4h.csv — original)")
    parser.add_argument("--data-ds", type=str, default=str(DS_DATA_PATH),
                        help="Dataset for phase randomization (default: causalformer_input_agg4h_ds.csv — deseasonalized)")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Base CausalFormer config to use for falsification runs")
    parser.add_argument("--test", type=str, default="both",
                        choices=["timeshift", "phaserand", "both"],
                        help="Which falsification test to run (default: both)")
    parser.add_argument("--shift", type=int, default=TIMESHIFT_STEPS,
                        help=f"Steps to shift for time-shift placebo (default: {TIMESHIFT_STEPS})")
    args = parser.parse_args()

    edges_path = pathlib.Path(args.edges)
    data_path = pathlib.Path(args.data)
    ds_data_path = pathlib.Path(args.data_ds)
    config_path = pathlib.Path(args.config)

    print(f"\nWP5 — Falsification Tests")
    print(f"Real edges      : {edges_path.name}")
    print(f"Time-shift data : {data_path.name} (original)")
    print(f"Phase-rand data : {ds_data_path.name} (deseasonalized)")
    print(f"Test(s)         : {args.test}")
    print("=" * 60)

    # Load real edges — var_names come from the original column order
    real_edges = pd.read_csv(edges_path)
    df_orig = pd.read_csv(data_path)
    df_ds = pd.read_csv(ds_data_path)
    var_names = list(df_orig.columns)
    n_cross = len(real_edges[real_edges["cause"] != real_edges["effect"]])
    print(f"Real CausalFormer edges: {len(real_edges)} total, {n_cross} cross-variable")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    tests_to_run = ["timeshift", "phaserand"] if args.test == "both" else [args.test]

    for test in tests_to_run:
        print(f"\n{'='*60}")
        if test == "timeshift":
            print(f"Time-Shift Placebo (shift={args.shift} steps × 4h = {args.shift*4}h)")
            print(f"{'='*60}")
            corrupted_df = apply_timeshift(df_orig, shift=args.shift)
            out_name = "falsification_timeshift.csv"
        else:
            print("Phase Randomization (on deseasonalized data)")
            print(f"{'='*60}")
            corrupted_df = apply_phase_randomization(df_ds, seed=SEED)
            out_name = "falsification_phaserand.csv"

        print(f"Corrupted data shape: {corrupted_df.shape}")
        print("Running CausalFormer on corrupted data...")

        placebo_edge_tuples = run_causalformer_on(corrupted_df, var_names, config_path)
        result = compare_edges(real_edges, placebo_edge_tuples, var_names)

        # Report
        print(f"\nResults:")
        print(f"  Real cross-variable edges  : {result['n_real_edges']}")
        print(f"  Placebo edges found        : {result['n_placebo_edges']}")
        print(f"  Pair-level overlap         : {result['n_pair_overlap']} "
              f"({result['pair_overlap_rate']:.1%} of real edges survived)")
        print(f"  Exact overlap (same lag)   : {result['n_exact_overlap']}")

        if result["overlapping_pairs"]:
            print(f"\n  Surviving edges (suspicious):")
            for cause, effect in result["overlapping_pairs"]:
                print(f"    {cause} → {effect}")
        else:
            print("\n  No edges survived — falsification passed.")

        # Save placebo edge list
        out_path = METRICS_DIR / out_name
        result["placebo_edges"].to_csv(out_path, index=False)
        print(f"\nPlacebo edges saved to: {out_path}")


if __name__ == "__main__":
    main()
