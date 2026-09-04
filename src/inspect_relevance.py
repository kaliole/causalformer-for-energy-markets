"""Inspect raw RRP relevance scores from a trained CausalFormer model.

Extracts the attention relevance matrix (relA) to see the actual causal
signal strengths before k-means clustering discards everything below
the autoregressive edge.

Usage:
    python src/inspect_relevance.py --model-dir "CausalFormer/saved/models/WP2 DE-LU Causal Discovery/0331_153008"
"""

import argparse
import json
import os
import pathlib
import sys
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CAUSALFORMER_DIR = PROJECT_ROOT / "CausalFormer"

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def main() -> None:
    """Extract and display the raw relevance scores."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()

    model_dir = str(pathlib.Path(args.model_dir).resolve())

    # Force CPU in config (MPS has RRP compatibility issues)
    config_json_path = pathlib.Path(model_dir) / "config.json"
    with open(config_json_path) as f:
        config_data = json.load(f)
    config_data["n_gpu"] = 0
    with open(config_json_path, "w") as f:
        json.dump(config_data, f, indent=4)

    # Setup CausalFormer imports
    os.chdir(CAUSALFORMER_DIR)
    sys.path.insert(0, str(CAUSALFORMER_DIR))

    import interpret as cf_interpret
    from explainer.explainer import RRP
    from utils import prepare_device

    load_args = argparse.Namespace(device=None)
    model, config, data_loader = cf_interpret.load_model(model_dir, load_args)

    device, _ = prepare_device(config['n_gpu'])
    columns = list(data_loader.df_data.columns)
    series_num = data_loader.series_num

    # Prepare data (bigdata=True: average all samples)
    data = [t[0] for t in data_loader.dataset]
    data = torch.tensor(np.array(data), dtype=torch.float).to(device)
    data = data.mean(0).unsqueeze(0)

    # Run RRP for each target variable
    attribution_generator = RRP(model)
    relA = []
    for i in range(series_num):
        rel_a, _ = attribution_generator.generate_RRP(data_loader.batch_size, data, i)
        relA.append(rel_a.detach().cpu().numpy()[i])

    # Build and display the full relevance matrix
    relA_matrix = np.array(relA)  # shape: [effect, cause]
    df = pd.DataFrame(relA_matrix, index=columns, columns=columns)
    df.index.name = "effect ←"
    df.columns.name = "cause →"

    print("\n" + "=" * 70)
    print("Raw attention relevance scores (relA)")
    print("Rows = effect (target), Columns = cause (source)")
    print("=" * 70)
    print(df.to_string(float_format=lambda x: f"{x:.6f}"))

    # Show each row normalized to percentages for easier reading
    print("\n" + "=" * 70)
    print("Normalized relevance (% of total per target variable)")
    print("=" * 70)
    row_sums = df.sum(axis=1)
    df_pct = df.div(row_sums, axis=0) * 100
    print(df_pct.to_string(float_format=lambda x: f"{x:.1f}%"))

    # Highlight: for each target, rank the sources
    print("\n" + "=" * 70)
    print("Ranked sources per target variable")
    print("=" * 70)
    for target in columns:
        scores = df.loc[target].sort_values(ascending=False)
        print(f"\n  {target} ← ")
        for source, score in scores.items():
            pct = score / scores.sum() * 100
            marker = " ★" if source != target and pct > 5 else ""
            print(f"    {source:20s}  {score:.6f}  ({pct:.1f}%){marker}")


if __name__ == "__main__":
    main()
