"""Run CausalFormer on synthetic benchmark datasets and evaluate against ground truth.

WP3 synthetic validation: loops over 4 DAG structures × 10 replicates,
trains CausalFormer, extracts edges, and evaluates precision/recall/F1/SHD.

Usage:
    python src/run_synthetic_experiments.py
    python src/run_synthetic_experiments.py --structures diamond,fork --replicates 3
"""

import argparse
import json
import os
import pathlib
import sys
import tempfile
import warnings

# Suppress pandas FutureWarning from CausalFormer's vendored util.py
warnings.filterwarnings("ignore", category=FutureWarning, module="utils.util")

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CAUSALFORMER_DIR = PROJECT_ROOT / "CausalFormer"
CONFIGS_DIR = PROJECT_ROOT / "configs"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

SEED = 42

# Map each structure to its base config
STRUCTURE_CONFIG = {
    "diamond": CONFIGS_DIR / "config_wp3_diamond_mediator.json",
    "mediator": CONFIGS_DIR / "config_wp3_diamond_mediator.json",
    "fork": CONFIGS_DIR / "config_wp3_v_fork.json",
    "v": CONFIGS_DIR / "config_wp3_v_fork.json",
}

ALL_STRUCTURES = ["diamond", "fork", "mediator", "v"]


def make_config(base_config_path: pathlib.Path, structure: str, replicate: int) -> str:
    """Create a temp config JSON with the data_dir pointing to the right replicate."""
    with open(base_config_path) as f:
        config = json.load(f)

    config["data_loader"]["args"]["data_dir"] = f"data/basic/{structure}/data_{replicate}.csv"

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix=f"wp3_{structure}_{replicate}_",
        delete=False, dir=str(CAUSALFORMER_DIR),
    )
    json.dump(config, tmp, indent=4)
    tmp.close()
    return tmp.name


def run_single(structure: str, replicate: int, seed: int = SEED) -> dict:
    """Train, interpret, and evaluate a single structure/replicate."""
    from run_causalformer import train, interpret, set_seed, setup_causalformer_imports
    from validation.evaluate_edges import evaluate

    base_config = STRUCTURE_CONFIG[structure]
    config_path = make_config(base_config, structure, replicate)

    gt_path = str(CAUSALFORMER_DIR / "data" / "basic" / structure / "groundtruth.csv")
    data_path = CAUSALFORMER_DIR / "data" / "basic" / structure / f"data_{replicate}.csv"
    num_vars = len(pd.read_csv(data_path, nrows=0).columns)
    receptive_field = 16  # time_step from config

    try:
        # Reset CWD and seed for each run
        os.chdir(PROJECT_ROOT)
        set_seed(seed)
        setup_causalformer_imports()

        model_dir = train(config_path)
        edges = interpret(model_dir, config_path, bigdata=False)
        metrics = evaluate(edges, gt_path, num_vars, receptive_field)
    finally:
        # Clean up temp config
        try:
            os.unlink(config_path)
        except OSError:
            pass

    metrics["structure"] = structure
    metrics["replicate"] = replicate
    metrics["num_vars"] = num_vars
    return metrics


def main() -> None:
    """Run synthetic experiments and save results."""
    parser = argparse.ArgumentParser(description="WP3: Synthetic validation of CausalFormer")
    parser.add_argument("--structures", type=str, default=",".join(ALL_STRUCTURES),
                        help="Comma-separated structures to run (default: all)")
    parser.add_argument("--replicates", type=int, default=10,
                        help="Number of replicates per structure (default: 10)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    structures = [s.strip() for s in args.structures.split(",")]
    n_replicates = args.replicates

    for s in structures:
        if s not in STRUCTURE_CONFIG:
            parser.error(f"Unknown structure: {s}. Choose from {ALL_STRUCTURES}")

    # Add src/ to path so validation module is importable
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    print(f"\nWP3 Synthetic Validation")
    print(f"Seed: {args.seed}")
    print(f"Structures: {structures}")
    print(f"Replicates: {n_replicates}")
    print(f"Total runs: {len(structures) * n_replicates}")
    print("=" * 60)

    results = []
    for structure in structures:
        for rep in range(n_replicates):
            print(f"\n{'='*60}")
            print(f"Running: {structure} replicate {rep}")
            print(f"{'='*60}")
            metrics = run_single(structure, rep, seed=args.seed)
            results.append(metrics)
            print(f"  -> F1={metrics['f1']}, F1'={metrics['f1_ext']}, SHD={metrics['shd']}, "
                  f"self={metrics['num_pred_self']}/{metrics['num_gt_self']}, "
                  f"nonself={metrics['num_pred_nonself']}/{metrics['num_gt_nonself']}")

    # Collect and save
    df = pd.DataFrame(results)
    col_order = [
        "structure", "replicate", "num_vars", "num_gt_edges", "num_predicted_edges",
        "num_gt_self", "num_gt_nonself", "num_pred_self", "num_pred_nonself",
        "precision", "recall", "f1", "precision_ext", "recall_ext", "f1_ext",
        "shd", "pod",
    ]
    df = df[col_order]

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    seed_suffix = f"_seed{args.seed}" if args.seed != SEED else ""
    out_path = METRICS_DIR / f"synthetic_results{seed_suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")

    # Summary table
    print(f"\n{'='*60}")
    print("Summary (mean ± std per structure)")
    print(f"{'='*60}")
    metric_cols = ["precision", "recall", "f1", "precision_ext", "recall_ext", "f1_ext", "shd", "pod"]
    summary = df.groupby("structure")[metric_cols].agg(["mean", "std"])
    for structure in structures:
        print(f"\n{structure}:")
        for col in metric_cols:
            mean = summary.loc[structure, (col, "mean")]
            std = summary.loc[structure, (col, "std")]
            print(f"  {col:18s}: {mean:.3f} ± {std:.3f}")


if __name__ == "__main__":
    main()
