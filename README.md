# Causal Transformers for Energy Markets

Discovering Causal Dependencies in European Electricity Time Series

Master's thesis, Humboldt-Universität zu Berlin, Institut für Wirtschaftsinformatik.

| | |
|---|---|
| **Type** | Master's thesis (Information Systems, M. Sc.) |
| **Author** | Katharina Lenz |
| **First examiner** | Prof. Dr. Stefan Lessmann |
| **Second examiner** | Prof. Dr. Jan Mendling |
| **Full text** | [`thesis/main.pdf`](thesis/main.pdf) |

**Keywords:** causal discovery, transformer models, electricity markets, time series, CausalFormer, causal validation

## Summary

Transformer-based causal discovery methods can propose directed dependencies between
time series, but their outputs are candidate hypotheses rather than established causal
facts. This thesis asks how far such proposals can be trusted and applies them to the
German-Luxembourg (DE-LU) electricity market.

The work uses CausalFormer, a transformer that derives directed lagged edges through
relevance propagation, and then subjects every proposed edge to a multi-stage validation
framework built from independent statistical tools. The thesis is guided by two research
questions.

- **RQ1** To what extent are CausalFormer-proposed edges statistically credible and robust under independent validation?
- **RQ2** Which proposed dependencies receive support in the DE-LU data, and how can they be interpreted?

**Main results.** On synthetic benchmarks CausalFormer reached an edge-recovery F1 of
0.684 (seed 42, 40 runs). On real DE-LU data it proposed 24 edges, of which 17 are
cross-variable. Under the validation framework only a small subset survived. Five edges
were accepted (Tier 1) and three of those were strongly accepted (Tier 2). Notably only
1 of 17 cross-variable edges was supported by PCMCI+, while Granger causality supported
all 17, which shows how strongly the conclusion depends on the validation method. The
overall answer to RQ1 is cautious. Transformer-derived relevance is not causal evidence
on its own and needs independent confirmation.

Accepted edges (see `figures/accepted_graph.pdf`):

- **Tier 2 (strongly accepted):** temperature to load, wind_speed to wind_generation, wind_generation to wind_speed
- **Tier 1 (accepted):** solar_generation to temperature, solar_radiation to price

## Data

- DE-LU electricity market, 2019-2021, aggregated to a 4-hour resolution.
- Market variables (load, price, wind and solar generation) from ENTSO-E via SMARD.
- Weather variables (temperature, wind speed, solar radiation) from ERA5 reanalysis.

Raw downloads live in `data/raw/`. The aligned modelling dataset is
`data/processed/delu_dataset.csv`. The three model-ready inputs used by CausalFormer are
stored alongside it and can also be regenerated from it with
`src/prepare_causalformer_input.py`.

- `causalformer_input_agg4h.csv` — 4-hour means, used for the seasonal regime test
- `causalformer_input_agg4h_ds.csv` — deseasonalized, used for PCMCI+, bootstrap and phase randomization
- `causalformer_input_agg4h_diff.csv` — first-differenced, used for the main CausalFormer run, Granger and the time-shift test

## Setup

Python 3.10 or newer. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

CausalFormer is vendored as a git submodule with its own dependencies.

```bash
git submodule update --init --recursive
# see CausalFormer/requirements.txt for its environment
```

## Reproducing the results

All pipeline code is in `src/` as modular Python scripts. Random seed is 42 throughout.

| Step | Script |
|---|---|
| Fetch source data | `src/fetch_smard.py`, `src/fetch_era5.py` |
| Build aligned dataset | `src/build_dataset.py` |
| Prepare CausalFormer input | `src/prepare_causalformer_input.py` |
| Synthetic benchmark (F1) | `src/run_synthetic_experiments.py` |
| Causal discovery on DE-LU | `src/run_causalformer.py` (config `configs/config_wp4.json`) |
| Validation pipeline | `src/run_validation_pipeline.py` |

The validation pipeline runs PCMCI+, Granger causality, two falsification tests
(time-shift and phase randomization), and two stability tests (seasonal regime and
bootstrap), then applies the two-tier acceptance rule. Its modules are in
`src/validation/`.

## Results and figures

Result tables are in `results/`.

- `results/metrics/synthetic_results.csv` — synthetic benchmark (F1 = 0.684)
- `results/edges/delu_edges_raw.csv` — 24 candidate edges from CausalFormer
- `results/edges/delu_validated_edges.csv` — full validation evidence table

The three thesis figures are regenerated from these CSVs by scripts in `src/figures/`.

```bash
uv run python src/figures/plot_candidate_graph.py   # figures/candidate_graph.pdf
uv run python src/figures/plot_evidence_heatmap.py  # figures/evidence_heatmap.pdf
uv run python src/figures/plot_accepted_graph.py    # figures/accepted_graph.pdf
```

## Thesis document

The full thesis is available as [`thesis/main.pdf`](thesis/main.pdf).

## Repository layout

```
src/          Pipeline code (data, discovery, validation, figures)
CausalFormer/ Vendored CausalFormer model (git submodule)
configs/      Experiment configurations
data/         Raw downloads and the processed dataset
results/      Edge and metric CSVs
figures/      Thesis figures
thesis/       Full thesis (PDF)
```
