"""Evaluate predicted causal edges against ground truth.

Computes precision, recall, F1 (direct and extended), SHD, PoD, and self-loop counts.
Uses CausalFormer's getextendeddelays() for indirect-path evaluation.
Includes self-loops, matching the paper's evaluation methodology.
"""

import pathlib
import sys

import numpy as np
import pandas as pd

# Make CausalFormer's evaluator importable
_CF_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "CausalFormer"
sys.path.insert(0, str(_CF_DIR))
from evaluator.evaluator import getextendeddelays


def load_ground_truth(gt_path: str) -> list[tuple[int, int, int]]:
    """Load ground truth edges from CSV (cause, effect, delay — no header)."""
    df = pd.read_csv(gt_path, header=None, names=["cause", "effect", "delay"])
    return list(df.itertuples(index=False, name=None))


def evaluate(
    predicted_edges: list[tuple[int, int, int]],
    gt_path: str,
    num_vars: int,
    receptive_field: int,
) -> dict:
    """Evaluate predicted edges against ground truth (including self-loops).

    Args:
        predicted_edges: List of (cause_idx, effect_idx, lag) tuples.
        gt_path: Path to groundtruth.csv (cause, effect, delay — no header).
        num_vars: Number of variables in the dataset.
        receptive_field: CausalFormer's time_step (receptive field size).

    Returns:
        Dict with precision, recall, f1, precision_ext, recall_ext, f1_ext,
        shd, pod, and counts (including self-loop diagnostics).
    """
    gt_edges = load_ground_truth(gt_path)
    columns = [f"V{i+1}" for i in range(num_vars)]

    # Build predicted causes dict: effect -> list of causes (including self-loops)
    pred_causes = {k: [] for k in range(num_vars)}
    pred_delays = {}
    for cause, effect, lag in predicted_edges:
        pred_causes[effect].append(cause)
        pred_delays[(effect, cause)] = lag

    # Extended GT via CausalFormer's evaluator (includes self-loops)
    extendedgtdelays, readgt, extendedreadgt = getextendeddelays(gt_path, columns)

    # --- Direct evaluation (matching CausalFormer's evaluator.evaluate) ---
    tp_direct = fp_direct = 0
    tp_ext = fp_ext = 0
    tp_pairs_ext = []
    for effect in readgt:
        for cause in pred_causes[effect]:
            # Direct
            if cause in readgt[effect]:
                tp_direct += 1
            else:
                fp_direct += 1
            # Extended (includes indirect paths)
            if cause in extendedreadgt[effect]:
                tp_ext += 1
                tp_pairs_ext.append((effect, cause))
            else:
                fp_ext += 1

    fn = 0
    for effect in readgt:
        for cause in readgt[effect]:
            if cause not in pred_causes[effect]:
                fn += 1

    # Direct F1
    prec = tp_direct / (tp_direct + fp_direct) if (tp_direct + fp_direct) > 0 else 0.0
    rec = tp_direct / (tp_direct + fn) if (tp_direct + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # Extended F1 (includes indirect paths)
    prec_ext = tp_ext / (tp_ext + fp_ext) if (tp_ext + fp_ext) > 0 else 0.0
    rec_ext = tp_ext / (tp_ext + fn) if (tp_ext + fn) > 0 else 0.0
    f1_ext = 2 * prec_ext * rec_ext / (prec_ext + rec_ext) if (prec_ext + rec_ext) > 0 else 0.0

    # --- SHD (Structural Hamming Distance) — non-self-loop edges only ---
    gt_adj = np.zeros((num_vars, num_vars), dtype=int)
    for cause, effect, _ in gt_edges:
        if cause != effect:
            gt_adj[cause, effect] = 1

    pred_adj = np.zeros((num_vars, num_vars), dtype=int)
    for cause, effect, _ in predicted_edges:
        if cause != effect:
            pred_adj[cause, effect] = 1

    additions = int(np.sum((pred_adj == 1) & (gt_adj == 0) & (gt_adj.T == 0)))
    deletions = int(np.sum((pred_adj == 0) & (gt_adj == 1)))
    reversals = int(np.sum((pred_adj == 1) & (gt_adj == 0) & (gt_adj.T == 1)))
    shd = additions + deletions + reversals

    # --- PoD (Probability of Detection for lag accuracy) ---
    correct_delays = 0
    total_tp_with_delay = 0
    for effect, cause in tp_pairs_ext:
        if (effect, cause) in extendedgtdelays and (effect, cause) in pred_delays:
            gt_delays_list = extendedgtdelays[(effect, cause)]
            pred_delay = pred_delays[(effect, cause)]
            for d in gt_delays_list:
                if d <= receptive_field:
                    total_tp_with_delay += 1
                    if d == pred_delay:
                        correct_delays += 1
    pod = correct_delays / total_tp_with_delay if total_tp_with_delay > 0 else 0.0

    # --- Self-loop diagnostics ---
    num_gt_self = sum(1 for c, e, _ in gt_edges if c == e)
    num_gt_nonself = sum(1 for c, e, _ in gt_edges if c != e)
    num_pred_self = sum(1 for c, e, _ in predicted_edges if c == e)
    num_pred_nonself = sum(1 for c, e, _ in predicted_edges if c != e)

    return {
        "num_gt_edges": num_gt_self + num_gt_nonself,
        "num_predicted_edges": len(predicted_edges),
        "num_gt_self": num_gt_self,
        "num_gt_nonself": num_gt_nonself,
        "num_pred_self": num_pred_self,
        "num_pred_nonself": num_pred_nonself,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "precision_ext": round(prec_ext, 4),
        "recall_ext": round(rec_ext, 4),
        "f1_ext": round(f1_ext, 4),
        "shd": shd,
        "pod": round(pod, 4),
    }
