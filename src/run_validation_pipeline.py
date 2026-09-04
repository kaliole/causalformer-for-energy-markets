"""WP5 Validation Pipeline — orchestrator.

Runs all validation modules in sequence on the CausalFormer candidate edges:

    1.  PCMCI+ triangulation      — CI-based cross-check
    2.  Falsification tests       — time-shift placebo + phase randomization
    2b. Granger causality         — additional pairwise test on daily-differenced data
    3.  Stability tests           — regime robustness + block bootstrap
    4.  Edge acceptance           — combine evidence, apply two-tier rule

Each module saves its own outputs to results/metrics/ or results/edges/.
The final output is results/edges/delu_validated_edges.csv.

Note: steps 2 and 3 re-run CausalFormer multiple times (14 runs total).
This script is designed to run overnight as part of the final clean run.

Usage:
    uv run python src/run_validation_pipeline.py
    uv run python src/run_validation_pipeline.py --edges results/edges/delu_edges_raw.csv
    uv run python src/run_validation_pipeline.py --skip-falsification
    uv run python src/run_validation_pipeline.py --skip-stability
    uv run python src/run_validation_pipeline.py --skip-granger
"""

import argparse
import pathlib
import sys
import time

import pandas as pd

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
EDGES_DIR = PROJECT_ROOT / "results" / "edges"

DEFAULT_EDGES = EDGES_DIR / "delu_edges_raw.csv"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config_wp4.json"
DEFAULT_DATA = PROJECT_ROOT / "data" / "processed" / "causalformer_input_agg4h.csv"

# Add src/ to path so validation modules are importable
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def fmt_duration(seconds: float) -> str:
    """Format a duration in seconds as a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def step(label: str) -> float:
    """Print a step header and return the current time for duration tracking."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    return time.time()


def done(t0: float) -> None:
    """Print step completion with elapsed time."""
    print(f"  Done ({fmt_duration(time.time() - t0)})")


def main() -> None:
    """Run the full WP5 validation pipeline."""
    parser = argparse.ArgumentParser(description="WP5: Full validation pipeline")
    parser.add_argument("--edges", type=str, default=str(DEFAULT_EDGES),
                        help="CausalFormer raw edge CSV (default: delu_edges_raw.csv)")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG),
                        help="CausalFormer config for falsification/stability runs")
    parser.add_argument("--data", type=str, default=str(DEFAULT_DATA),
                        help="Input dataset CSV (default: causalformer_input_agg4h.csv)")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="PCMCI+ significance threshold (default: 0.05)")
    parser.add_argument("--max-lag", type=int, default=32,
                        help="PCMCI+ max lag (default: 32, matches CausalFormer time_step)")
    parser.add_argument("--skip-falsification", action="store_true",
                        help="Skip falsification tests (saves ~4h compute)")
    parser.add_argument("--skip-granger", action="store_true",
                        help="Skip Granger causality triangulation (fast, <1 min)")
    parser.add_argument("--skip-stability", action="store_true",
                        help="Skip stability tests (saves ~20h compute)")
    args = parser.parse_args()

    edges_path = pathlib.Path(args.edges).resolve()
    config_path = pathlib.Path(args.config).resolve()
    data_path = pathlib.Path(args.data).resolve()

    pipeline_start = time.time()
    print(f"\nWP5 Validation Pipeline")
    print(f"Edges  : {edges_path.name}")
    print(f"Data   : {data_path.name}")
    print(f"Config : {config_path.name}")
    print(f"Alpha  : {args.alpha}  |  Max lag : {args.max_lag}")
    if args.skip_falsification:
        print("  [skipping falsification tests]")
    if args.skip_granger:
        print("  [skipping Granger causality tests]")
    if args.skip_stability:
        print("  [skipping stability tests]")

    # -----------------------------------------------------------------------
    # Step 1 — PCMCI+ triangulation
    # -----------------------------------------------------------------------
    t0 = step("Step 1 / 4 — PCMCI+ Triangulation")

    from validation.pcmci_validation import load_data, run_pcmciplus, parse_pcmci_edges, triangulate

    data, var_names = load_data(data_path)
    cf_edges = pd.read_csv(edges_path)
    print(f"Data: {data.shape}  |  CausalFormer edges: {len(cf_edges)}")

    print("Running PCMCI+...")
    pcmci_results = run_pcmciplus(data, var_names, args.max_lag, args.alpha)

    pcmci_all = parse_pcmci_edges(pcmci_results, var_names, args.alpha)
    pcmci_all.to_csv(METRICS_DIR / "pcmci_all_edges.csv", index=False)
    print(f"PCMCI+ found {len(pcmci_all)} significant cross-variable edges")

    triangulated = triangulate(
        cf_edges, pcmci_results["p_matrix"], pcmci_results["val_matrix"],
        var_names, args.alpha
    )
    triangulated.to_csv(METRICS_DIR / "pcmci_triangulation.csv", index=False)
    n_supported = int(triangulated["pcmci_supported"].sum())
    n_testable = int(triangulated["pcmci_pvalue"].notna().sum())
    print(f"CausalFormer edges supported by PCMCI+: {n_supported}/{n_testable} testable")

    done(t0)

    # -----------------------------------------------------------------------
    # Step 2 — Falsification tests
    # -----------------------------------------------------------------------
    if not args.skip_falsification:
        t0 = step("Step 2 / 4 — Falsification Tests")

        from validation.falsification_tests import (
            apply_timeshift, apply_phase_randomization,
            run_causalformer_on as cf_run_falsification,
            compare_edges,
        )

        df_raw = pd.read_csv(data_path)

        for test_name, corrupt_fn in [
            ("timeshift", lambda df: apply_timeshift(df)),
            ("phaserand", lambda df: apply_phase_randomization(df)),
        ]:
            print(f"\nRunning {test_name} falsification...")
            corrupted = corrupt_fn(df_raw)
            placebo_tuples = cf_run_falsification(corrupted, var_names, config_path)
            result = compare_edges(cf_edges, placebo_tuples, var_names)

            out_path = METRICS_DIR / f"falsification_{test_name}.csv"
            result["placebo_edges"].to_csv(out_path, index=False)

            print(f"  Placebo edges found: {result['n_placebo_edges']}")
            print(f"  Surviving real edges: {result['n_pair_overlap']} "
                  f"({result['pair_overlap_rate']:.1%})")

        done(t0)
    else:
        print("\n[Step 2 skipped — falsification tests]")

    # -----------------------------------------------------------------------
    # Step 2b — Granger causality triangulation
    # -----------------------------------------------------------------------
    if not args.skip_granger:
        t0 = step("Step 2b — Granger Causality Triangulation")

        from validation.granger import load_data as load_granger_data, run_granger_tests

        print("Loading and daily-differencing data for Granger tests...")
        granger_data, granger_var_names = load_granger_data(data_path)
        print(f"Data shape (after differencing): {granger_data.shape}")

        print("Running Granger causality tests...")
        granger_results = run_granger_tests(
            cf_edges, granger_data, granger_var_names, args.max_lag, args.alpha
        )

        granger_out = METRICS_DIR / "granger_triangulation.csv"
        granger_results.to_csv(granger_out, index=False)

        n_gr_supported = int(granger_results["granger_supported"].sum())
        n_gr_tested = len(granger_results)
        print(f"  Granger-supported edges: {n_gr_supported}/{n_gr_tested} cross-variable edges")

        done(t0)
    else:
        print("\n[Step 2b skipped — Granger causality tests]")

    # -----------------------------------------------------------------------
    # Step 3 — Stability tests
    # -----------------------------------------------------------------------
    if not args.skip_stability:
        t0 = step("Step 3 / 4 — Stability Tests")

        from validation.stability_tests import (
            load_regime_splits, generate_block_bootstrap,
            run_causalformer_on as cf_run_stability,
            compute_stability_scores, edges_to_pairs,
            BLOCK_SIZE, N_BOOTSTRAP, STABILITY_THRESHOLD, COLUMNS,
        )

        real_pairs = {
            (r["cause"], r["effect"])
            for _, r in cf_edges.iterrows()
            if r["cause"] != r["effect"]
        }

        # 4.1 Regime robustness
        print("\n4.1 Regime Robustness (winter vs summer)...")
        splits = load_regime_splits()
        regime_edge_sets = {}
        for regime_name, df_regime in splits.items():
            print(f"  Running CausalFormer on {regime_name} subset...")
            tuples = cf_run_stability(df_regime, config_path, label=regime_name)
            regime_edge_sets[regime_name] = edges_to_pairs(tuples, COLUMNS)

        robust = regime_edge_sets["winter"] & regime_edge_sets["summer"]
        all_regime_pairs = regime_edge_sets["winter"] | regime_edge_sets["summer"]
        regime_rows = [
            {
                "cause": c, "effect": e,
                "in_winter": (c, e) in regime_edge_sets["winter"],
                "in_summer": (c, e) in regime_edge_sets["summer"],
                "in_real_edges": (c, e) in real_pairs,
                "regime_robust": (c, e) in robust,
            }
            for c, e in sorted(all_regime_pairs)
        ]
        regime_df = pd.DataFrame(regime_rows)
        regime_df.to_csv(METRICS_DIR / "stability_regime.csv", index=False)
        print(f"  Regime-robust edges: {len(robust)}")

        # 4.2 Block bootstrap
        print(f"\n4.2 Block Bootstrap ({N_BOOTSTRAP} samples, block={BLOCK_SIZE} steps)...")
        df_agg = pd.read_csv(data_path)
        bootstrap_samples = generate_block_bootstrap(df_agg, BLOCK_SIZE, N_BOOTSTRAP)
        bootstrap_edge_sets = []
        for i, sample in enumerate(bootstrap_samples):
            print(f"  Bootstrap sample {i+1}/{N_BOOTSTRAP}...")
            tuples = cf_run_stability(sample, config_path, label=f"bootstrap_{i}")
            bootstrap_edge_sets.append(edges_to_pairs(tuples, COLUMNS))

        scores_df = compute_stability_scores(bootstrap_edge_sets, real_pairs)
        scores_df.to_csv(METRICS_DIR / "stability_bootstrap.csv", index=False)
        n_stable = int(scores_df["stable"].sum())
        print(f"  Stable edges (>= {STABILITY_THRESHOLD:.0%}): {n_stable}")

        done(t0)
    else:
        print("\n[Step 3 skipped — stability tests]")

    # -----------------------------------------------------------------------
    # Step 4 — Edge acceptance
    # -----------------------------------------------------------------------
    t0 = step("Step 4 / 4 — Edge Acceptance")

    from validation.edge_acceptance import (
        load_edges, load_pcmci, load_falsification_pairs,
        load_regime, load_bootstrap, load_granger,
        build_evidence_table, apply_acceptance_rules,
        BOOTSTRAP_THRESHOLD,
    )

    edges = load_edges(edges_path)
    pcmci_tri = load_pcmci(METRICS_DIR / "pcmci_triangulation.csv")
    timeshift_pairs = load_falsification_pairs(
        METRICS_DIR / "falsification_timeshift.csv", "time-shift"
    )
    phaserand_pairs = load_falsification_pairs(
        METRICS_DIR / "falsification_phaserand.csv", "phase-rand"
    )
    regime_results = load_regime(METRICS_DIR / "stability_regime.csv")
    bootstrap_results = load_bootstrap(METRICS_DIR / "stability_bootstrap.csv")
    granger_tri = load_granger(METRICS_DIR / "granger_triangulation.csv")

    evidence = build_evidence_table(
        edges, pcmci_tri, timeshift_pairs, phaserand_pairs,
        regime_results, bootstrap_results, granger_tri,
    )
    evidence = apply_acceptance_rules(evidence, BOOTSTRAP_THRESHOLD)
    evidence.to_csv(EDGES_DIR / "delu_validated_edges.csv", index=False)

    n_tier1 = int(evidence["accepted"].sum())
    n_tier2 = int(evidence["strongly_accepted"].sum())
    print(f"  Tier 1 (accepted)         : {n_tier1} / {len(evidence)}")
    print(f"  Tier 2 (strongly accepted): {n_tier2} / {len(evidence)}")

    done(t0)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"WP5 complete  ({fmt_duration(time.time() - pipeline_start)} total)")
    print(f"{'='*60}")
    print(f"  results/metrics/pcmci_triangulation.csv")
    print(f"  results/metrics/pcmci_all_edges.csv")
    if not args.skip_falsification:
        print(f"  results/metrics/falsification_timeshift.csv")
        print(f"  results/metrics/falsification_phaserand.csv")
    if not args.skip_granger:
        print(f"  results/metrics/granger_triangulation.csv")
    if not args.skip_stability:
        print(f"  results/metrics/stability_regime.csv")
        print(f"  results/metrics/stability_bootstrap.csv")
    print(f"  results/edges/delu_validated_edges.csv  ← final validated edge set")


if __name__ == "__main__":
    main()
