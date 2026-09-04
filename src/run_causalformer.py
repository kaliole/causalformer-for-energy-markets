"""Run CausalFormer training and causal discovery on the DE-LU dataset.

Pipeline:
  1. Train the CausalFormer model (predict future values from past)
  2. Run RRP (Regression Relevance Propagation) to extract causal attributions
  3. Use k-means clustering to identify causal edges
  4. Save discovered edges to results/edges/

This script must be run from the project root:
    python src/run_causalformer.py
    python src/run_causalformer.py --config configs/config_wp2.json

It internally changes directory to CausalFormer/ because CausalFormer's
imports are relative to its own directory.
"""

import argparse
import json
import os
import pathlib
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
CAUSALFORMER_DIR = PROJECT_ROOT / "CausalFormer"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config_wp2.json"
EDGES_DIR = PROJECT_ROOT / "results" / "edges"

SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def setup_causalformer_imports() -> None:
    """Add CausalFormer to sys.path and cd into it.

    CausalFormer uses relative imports (e.g., `from base import BaseDataLoader`)
    that only work when the CWD is the CausalFormer directory.
    """
    os.chdir(CAUSALFORMER_DIR)
    sys.path.insert(0, str(CAUSALFORMER_DIR))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config_path: str) -> str:
    """Train the CausalFormer model. Returns the path to the saved model directory."""
    from parse_config import ConfigParser

    # Create a unique run ID so we can find the saved model
    run_id = datetime.now().strftime(r"%m%d_%H%M%S")

    # Build config the same way CausalFormer's train.py does
    args_dict = {
        "name": None,
        "config": config_path,
        "resume": None,
        "device": None,
    }
    config = ConfigParser.from_args(args=args_dict, run_id=run_id)

    # Import and run training
    import train as cf_train
    cf_train.main(config)

    # Return the save directory so we can load the model for interpretation
    save_dir = str(config.save_dir)
    print(f"\nModel saved to: {save_dir}")
    return save_dir


# ---------------------------------------------------------------------------
# Interpretation (RRP + edge extraction)
# ---------------------------------------------------------------------------

def interpret(model_dir: str, config_path: str, bigdata: bool = True,
              explainer_m: int | None = None, explainer_n: int | None = None) -> list[tuple[int, int, int]]:
    """Run RRP attribution and extract causal edges.

    Args:
        model_dir: Path to the saved model directory (contains model_best.pth).
        config_path: Path to the config JSON.
        bigdata: If True, average all samples before RRP (needed for large datasets
                 to fit in memory). With 17,520 timesteps this should be True.

    Returns:
        List of (cause_idx, effect_idx, lag) tuples.
    """
    import interpret as cf_interpret

    # Force CPU for interpretation: CausalFormer's RRP uses .type(tensor.type())
    # which returns 'torch.mps.FloatTensor' on Apple Silicon — a string that MPS
    # doesn't recognise as a valid type. CPU avoids this. Interpretation is a single
    # pass so CPU is fast enough.
    args = argparse.Namespace(device=None)
    config_json_path = pathlib.Path(model_dir) / "config.json"
    with open(config_json_path) as f:
        config_override = json.load(f)
    config_override["n_gpu"] = 0  # force CPU
    with open(config_json_path, "w") as f:
        json.dump(config_override, f, indent=4)

    model, config, data_loader = cf_interpret.load_model(model_dir, args)

    # Run RRP attribution and edge extraction directly (CausalFormer's
    # interpret.main() doesn't return the edges, only logs them).
    from explainer.explainer import RRP
    from copy import deepcopy
    from utils import prepare_device

    device, _ = prepare_device(config['n_gpu'])
    columns = list(data_loader.df_data.columns)
    data = [t[0] for t in data_loader.dataset]
    data = torch.tensor(np.array(data), dtype=torch.float).to(device)
    if bigdata:
        data = data.mean(0).unsqueeze(0)

    attribution_generator = RRP(model)
    relA, relK = [], []
    for i in range(data_loader.series_num):
        print(f"  RRP for variable {i+1}/{data_loader.series_num}: {columns[i]}")
        rel_a, rel_k = attribution_generator.generate_RRP(data_loader.batch_size, data, i)
        relA.append(rel_a.detach().cpu().numpy()[i])
        relk_align = deepcopy(rel_k.detach().cpu().numpy()[:, i, -1, :])
        relk_align[i, :] = rel_k.detach().cpu().numpy()[i, i, -2, :]
        relK.append(relk_align)

    m = explainer_m if explainer_m is not None else config['explainer']['m']
    n = explainer_n if explainer_n is not None else config['explainer']['n']
    print(f"  Clustering: m={m} (top clusters), n={n} (total clusters)")
    edges = cf_interpret.analyze(relA, relK, m, n, config['data_loader']['args']['time_step'])

    for e in edges:
        print(f"  {columns[e[0]]} → {columns[e[1]]} (lag={e[2]}h)")

    return edges


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_edges(edges: list[tuple[int, int, int]], columns: list[str], config_path: str,
               prefix: str | None = None) -> pathlib.Path:
    """Save discovered causal edges to a CSV file.

    Output columns: cause, effect, lag, cause_idx, effect_idx
    """
    EDGES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if prefix is None:
        prefix = f"edges_{pathlib.Path(config_path).stem}_"
    out_path = EDGES_DIR / f"{prefix}{timestamp}.csv"

    rows = []
    for cause_idx, effect_idx, lag in edges:
        rows.append({
            "cause": columns[cause_idx],
            "effect": columns[effect_idx],
            "lag": lag,
            "cause_idx": cause_idx,
            "effect_idx": effect_idx,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"Discovered {len(edges)} causal edges:")
    print(f"{'='*60}")
    for _, row in df.iterrows():
        print(f"  {row['cause']} → {row['effect']} (lag={row['lag']}h)")
    print(f"\nSaved to: {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the full CausalFormer pipeline: train → interpret → save edges."""
    parser = argparse.ArgumentParser(description="Run CausalFormer on DE-LU dataset")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="Path to config JSON")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training and load from --model-dir")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Path to saved model directory (for --skip-train)")
    parser.add_argument("--no-bigdata", action="store_true",
                        help="Disable bigdata mode (uses more memory)")
    parser.add_argument("--explainer-m", type=int, default=None,
                        help="Override explainer m (top clusters to select)")
    parser.add_argument("--explainer-n", type=int, default=None,
                        help="Override explainer n (total clusters)")
    args = parser.parse_args()

    config_path = str(pathlib.Path(args.config).resolve())

    # Read column names from the config's data_dir
    with open(config_path) as f:
        config_json = json.load(f)
    data_csv = config_json["data_loader"]["args"]["data_dir"]
    # Resolve relative to CausalFormer/ (where we'll cd to)
    data_csv_path = CAUSALFORMER_DIR / data_csv
    columns = list(pd.read_csv(data_csv_path, nrows=0).columns)
    print(f"Variables ({len(columns)}): {columns}")

    # Resolve model_dir to absolute path BEFORE cd'ing into CausalFormer/
    if args.model_dir:
        model_dir_resolved = str(pathlib.Path(args.model_dir).resolve())
    else:
        model_dir_resolved = None

    set_seed(SEED)
    setup_causalformer_imports()

    # Step 1: Train
    if args.skip_train:
        if not model_dir_resolved:
            parser.error("--skip-train requires --model-dir")
        model_dir = model_dir_resolved
        print(f"Skipping training, loading from: {model_dir}")
    else:
        print("\n" + "="*60)
        print("Step 1: Training CausalFormer")
        print("="*60)
        model_dir = train(config_path)

    # Step 2: Interpret
    print("\n" + "="*60)
    print("Step 2: RRP Attribution & Edge Extraction")
    print("="*60)
    bigdata = not args.no_bigdata
    edges = interpret(model_dir, config_path, bigdata=bigdata,
                      explainer_m=args.explainer_m, explainer_n=args.explainer_n)

    # Step 3: Save
    if edges:
        save_edges(edges, columns, config_path)
    else:
        print("\nNo causal edges discovered.")


if __name__ == "__main__":
    main()
