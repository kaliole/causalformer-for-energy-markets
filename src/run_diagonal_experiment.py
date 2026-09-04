"""Compare three CausalFormer attention-diagonal treatments.

Variants:
  baseline   — standard run, no diagonal intervention
  soft_diag  — soft L2 penalty on attention mask diagonal (diag_lam=0.1)
  hard_diag  — hard block: diagonal zeroed + gradient hook prevents learning

Each variant uses the fast config (100 epochs, early_stop=10) on the original
4h-aggregated data. Edges and a comparison summary are saved to results/.

Run from project root:
    uv run python src/run_diagonal_experiment.py
"""

import json
import pathlib
import sys
import os

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
EDGES_DIR = PROJECT_ROOT / "results" / "edges"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

VARIANTS = [
    {
        "name": "baseline",
        "config": str(PROJECT_ROOT / "configs" / "config_wp2_agg4h_diff_fast.json"),
    },
    {
        "name": "soft_diag",
        "config": str(PROJECT_ROOT / "configs" / "config_wp2_agg4h_diff_soft_diag_fast.json"),
    },
    {
        "name": "hard_diag",
        "config": str(PROJECT_ROOT / "configs" / "config_wp2_agg4h_diff_hard_diag_fast.json"),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_columns(config_path: str) -> list[str]:
    """Read variable names from the data CSV referenced in the config."""
    with open(config_path) as f:
        cfg = json.load(f)
    data_dir = cfg["data_loader"]["args"]["data_dir"]
    # Paths in config are relative to CausalFormer/, resolve from there
    from run_causalformer import CAUSALFORMER_DIR
    csv_path = CAUSALFORMER_DIR / data_dir
    return list(pd.read_csv(csv_path, nrows=0).columns)


def _classify_edges(edges: list[tuple[int, int, int]]) -> tuple[list, list]:
    """Split edges into (self_loops, cross_variable) lists."""
    self_loops = [(c, e, l) for c, e, l in edges if c == e]
    cross = [(c, e, l) for c, e, l in edges if c != e]
    return self_loops, cross


def _edges_to_df(edges: list[tuple[int, int, int]], columns: list[str]) -> pd.DataFrame:
    """Convert edge tuples to a labelled DataFrame."""
    rows = [{"cause": columns[c], "effect": columns[e], "lag": l,
             "cause_idx": c, "effect_idx": e} for c, e, l in edges]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run all three variants and produce comparison summary."""
    # Import run_causalformer from src/ — must be on path
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from run_causalformer import setup_causalformer_imports, set_seed, SEED, train, interpret, save_edges

    set_seed(SEED)
    setup_causalformer_imports()

    columns = _get_columns(VARIANTS[0]["config"])
    EDGES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    comparison_rows = []

    for variant in VARIANTS:
        name = variant["name"]
        config_path = variant["config"]

        print(f"\n{'='*60}")
        print(f"VARIANT: {name}")
        print(f"{'='*60}")

        set_seed(SEED)
        model_dir = train(config_path)
        edges = interpret(model_dir, config_path, bigdata=True)

        self_loops, cross = _classify_edges(edges)

        prefix = f"edges_diag_{name}_"
        edge_path = save_edges(edges, columns, config_path, prefix=prefix)

        edge_strings = [f"{columns[c]}→{columns[e]}(lag={l})" for c, e, l in edges]

        comparison_rows.append({
            "variant": name,
            "n_edges": len(edges),
            "n_self_loops": len(self_loops),
            "n_cross_variable": len(cross),
            "edges": "; ".join(edge_strings) if edge_strings else "",
        })

        print(f"\n[{name}] {len(edges)} edges: {len(self_loops)} self-loops, {len(cross)} cross-variable")

    # Save comparison summary
    comparison = pd.DataFrame(comparison_rows)
    out_path = METRICS_DIR / "diagonal_experiment_comparison.csv"
    comparison.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(comparison[["variant", "n_edges", "n_self_loops", "n_cross_variable"]].to_string(index=False))
    print(f"\nFull comparison saved to: {out_path}")


if __name__ == "__main__":
    main()
