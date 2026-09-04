"""Edge acceptance for WP5 validation pipeline.

Combines all validation evidence into a single table and applies a two-tier
acceptance rule to determine which CausalFormer edges are validated.

Inputs (all from earlier WP5 modules):
    results/edges/delu_edges_raw.csv          — CausalFormer candidate edges (WP4)
    results/metrics/pcmci_triangulation.csv   — PCMCI+ support per edge
    results/metrics/falsification_timeshift.csv — placebo edges from time-shift run
    results/metrics/falsification_phaserand.csv — placebo edges from phase-rand run
    results/metrics/stability_regime.csv      — regime robustness per edge pair
    results/metrics/stability_bootstrap.csv   — bootstrap stability scores per edge pair
    results/metrics/granger_triangulation.csv — Granger causality support per edge (optional)

Output:
    results/edges/delu_validated_edges.csv    — full evidence table with acceptance flags

Acceptance rule (see docs/methods_notes.md for full explanation):

    Tier 1 — accepted:
        survived_timeshift = False
        AND survived_phaserand = False

    Tier 2 — strongly_accepted:
        Tier 1 AND (regime_robust = True OR bootstrap_stability_score >= 0.60)

    Note: PCMCI+ is retained as an informational signal but is not a gate.
    Granger causality serves as the baseline validation method (all 17 WP4 edges
    supported). PCMCI+ is excluded from the gate because it runs on deseasonalized
    data while CausalFormer trains on first-differenced data, making direct
    triangulation unreliable.

Usage:
    uv run python src/validation/edge_acceptance.py
    uv run python src/validation/edge_acceptance.py --edges results/edges/delu_edges_raw.csv
"""

import argparse
import pathlib

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
EDGES_DIR = PROJECT_ROOT / "results" / "edges"
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

DEFAULT_EDGES = EDGES_DIR / "delu_edges_raw.csv"
BOOTSTRAP_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_edges(path: pathlib.Path) -> pd.DataFrame:
    """Load CausalFormer candidate edges, excluding self-loops."""
    df = pd.read_csv(path)
    cross = df[df["cause"] != df["effect"]].copy().reset_index(drop=True)
    n_self = len(df) - len(cross)
    print(f"  CausalFormer edges: {len(df)} total, {n_self} self-loops excluded, "
          f"{len(cross)} cross-variable edges to validate")
    return cross


def load_pcmci(path: pathlib.Path) -> pd.DataFrame | None:
    """Load PCMCI+ triangulation results."""
    if not path.exists():
        print(f"  PCMCI+ triangulation: not found ({path.name}) — skipping")
        return None
    df = pd.read_csv(path)
    # Keep only the columns we need for merging
    cols = ["cause", "effect", "lag", "pcmci_pvalue", "pcmci_val", "pcmci_supported"]
    return df[[c for c in cols if c in df.columns]]


def load_falsification_pairs(path: pathlib.Path, label: str) -> set[tuple[str, str]]:
    """Load a falsification result CSV and return the set of (cause, effect) pairs found."""
    if not path.exists():
        print(f"  Falsification ({label}): not found ({path.name}) — treating as no surviving edges")
        return set()
    df = pd.read_csv(path)
    if df.empty:
        return set()
    return set(zip(df["cause"], df["effect"]))


def load_regime(path: pathlib.Path) -> pd.DataFrame | None:
    """Load regime robustness results."""
    if not path.exists():
        print(f"  Regime robustness: not found ({path.name}) — skipping")
        return None
    df = pd.read_csv(path)
    cols = ["cause", "effect", "in_winter", "in_summer", "regime_robust"]
    return df[[c for c in cols if c in df.columns]]


def load_bootstrap(path: pathlib.Path) -> pd.DataFrame | None:
    """Load bootstrap stability scores."""
    if not path.exists():
        print(f"  Bootstrap stability: not found ({path.name}) — skipping")
        return None
    df = pd.read_csv(path)
    cols = ["cause", "effect", "stability_score", "n_appearances", "n_runs", "stable"]
    return df[[c for c in cols if c in df.columns]]


def load_granger(path: pathlib.Path) -> pd.DataFrame | None:
    """Load Granger causality triangulation results."""
    if not path.exists():
        print(f"  Granger triangulation: not found ({path.name}) — skipping")
        return None
    df = pd.read_csv(path)
    cols = ["cause", "effect", "best_lag", "granger_pvalue", "granger_supported"]
    return df[[c for c in cols if c in df.columns]]


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_evidence_table(
    edges: pd.DataFrame,
    pcmci: pd.DataFrame | None,
    timeshift_pairs: set[tuple[str, str]],
    phaserand_pairs: set[tuple[str, str]],
    regime: pd.DataFrame | None,
    bootstrap: pd.DataFrame | None,
    granger: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge all validation evidence onto the CausalFormer edge list.

    Args:
        edges: cross-variable CausalFormer edges (cause, effect, lag, ...)
        pcmci: PCMCI+ triangulation results or None
        timeshift_pairs: (cause, effect) pairs that survived the time-shift placebo
        phaserand_pairs: (cause, effect) pairs that survived phase randomization
        regime: regime robustness results or None
        bootstrap: bootstrap stability scores or None
        granger: Granger causality triangulation results or None

    Returns:
        Evidence table with one row per CausalFormer cross-variable edge,
        all validation columns attached, and acceptance flags applied.
    """
    df = edges[["cause", "effect", "lag"]].copy()

    # --- PCMCI+ ---
    if pcmci is not None:
        df = df.merge(pcmci, on=["cause", "effect", "lag"], how="left")
        df["pcmci_supported"] = df["pcmci_supported"].fillna(False)
    else:
        df["pcmci_pvalue"] = float("nan")
        df["pcmci_val"] = float("nan")
        df["pcmci_supported"] = False

    # --- Falsification ---
    df["survived_timeshift"] = df.apply(
        lambda r: (r["cause"], r["effect"]) in timeshift_pairs, axis=1
    )
    df["survived_phaserand"] = df.apply(
        lambda r: (r["cause"], r["effect"]) in phaserand_pairs, axis=1
    )

    # --- Regime robustness ---
    if regime is not None:
        df = df.merge(regime[["cause", "effect", "regime_robust"]],
                      on=["cause", "effect"], how="left")
        df["regime_robust"] = df["regime_robust"].fillna(False)
    else:
        df["regime_robust"] = float("nan")

    # --- Bootstrap stability ---
    if bootstrap is not None:
        df = df.merge(
            bootstrap[["cause", "effect", "stability_score", "n_appearances", "n_runs"]],
            on=["cause", "effect"], how="left",
        )
        df["stability_score"] = df["stability_score"].fillna(0.0)
    else:
        df["stability_score"] = float("nan")
        df["n_appearances"] = float("nan")
        df["n_runs"] = float("nan")

    # --- Granger causality ---
    if granger is not None:
        df = df.merge(granger[["cause", "effect", "granger_pvalue", "granger_supported"]],
                      on=["cause", "effect"], how="left")
        df["granger_supported"] = df["granger_supported"].fillna(False)
    else:
        df["granger_pvalue"] = float("nan")
        df["granger_supported"] = False

    return df


def apply_acceptance_rules(df: pd.DataFrame,
                            bootstrap_threshold: float = BOOTSTRAP_THRESHOLD) -> pd.DataFrame:
    """Apply the two-tier acceptance rule to the evidence table.

    Tier 1 — accepted:
        NOT survived_timeshift AND NOT survived_phaserand

    Tier 2 — strongly_accepted:
        Tier 1 AND (regime_robust OR stability_score >= threshold)

    Returns:
        DataFrame with 'accepted' and 'strongly_accepted' columns added.
    """
    tier1 = (
        ~df["survived_timeshift"].astype(bool)
        & ~df["survived_phaserand"].astype(bool)
    )

    # For Tier 2, treat NaN stability/regime as False
    regime_ok = df["regime_robust"].fillna(False).infer_objects(copy=False).astype(bool)
    stability_ok = df["stability_score"].fillna(0.0).infer_objects(copy=False) >= bootstrap_threshold

    tier2 = tier1 & (regime_ok | stability_ok)

    df["accepted"] = tier1
    df["strongly_accepted"] = tier2
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Build the evidence table, apply acceptance rules, and save results."""
    parser = argparse.ArgumentParser(description="WP5: Edge acceptance — combine validation evidence")
    parser.add_argument("--edges", type=str, default=str(DEFAULT_EDGES),
                        help="CausalFormer raw edge CSV (default: delu_edges_raw.csv)")
    args = parser.parse_args()

    edges_path = pathlib.Path(args.edges)

    print(f"\nWP5 — Edge Acceptance")
    print(f"Candidate edges : {edges_path.name}")
    print(f"Bootstrap threshold : {BOOTSTRAP_THRESHOLD:.0%}")
    print("=" * 60)
    print("Loading validation inputs...")

    # Load all inputs
    edges = load_edges(edges_path)
    pcmci = load_pcmci(METRICS_DIR / "pcmci_triangulation.csv")
    timeshift_pairs = load_falsification_pairs(
        METRICS_DIR / "falsification_timeshift.csv", "time-shift"
    )
    phaserand_pairs = load_falsification_pairs(
        METRICS_DIR / "falsification_phaserand.csv", "phase-rand"
    )
    regime = load_regime(METRICS_DIR / "stability_regime.csv")
    bootstrap = load_bootstrap(METRICS_DIR / "stability_bootstrap.csv")
    granger = load_granger(METRICS_DIR / "granger_triangulation.csv")

    # Build and score
    print("\nBuilding evidence table...")
    evidence = build_evidence_table(
        edges, pcmci, timeshift_pairs, phaserand_pairs, regime, bootstrap, granger
    )
    evidence = apply_acceptance_rules(evidence, BOOTSTRAP_THRESHOLD)

    # Summary
    n_total = len(evidence)
    n_tier1 = int(evidence["accepted"].sum())
    n_tier2 = int(evidence["strongly_accepted"].sum())

    print(f"\n{'='*60}")
    print(f"Results ({n_total} candidate cross-variable edges):")
    print(f"{'='*60}")
    print(f"  Tier 1 — accepted        : {n_tier1} / {n_total}")
    print(f"  Tier 2 — strongly accepted: {n_tier2} / {n_total}")

    if n_tier1 > 0:
        print(f"\nTier 1 edges (accepted):")
        accepted = evidence[evidence["accepted"]]
        for _, row in accepted.iterrows():
            tier2_marker = " [strongly accepted]" if row["strongly_accepted"] else ""
            print(f"  {row['cause']} → {row['effect']} (lag={row['lag']}) "
                  f"pcmci_p={row['pcmci_pvalue']:.4f} "
                  f"stability={row['stability_score']:.2f}{tier2_marker}")
    else:
        print("\n  No edges passed Tier 1.")

    # Save
    EDGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EDGES_DIR / "delu_validated_edges.csv"
    evidence.to_csv(out_path, index=False)
    print(f"\nFull evidence table saved to: {out_path}")


if __name__ == "__main__":
    main()
