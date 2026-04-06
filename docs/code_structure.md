# Code Structure Specification

## Goal

This document defines the required project structure for implementing the paper:

**Proxy-Robust Latent Reliability Correction under Noisy Timing**

The purpose is to ensure that:
- all code modules are logically separated,
- all intermediate artifacts are reproducible,
- all experiments can be rerun consistently,
- benchmark, ablation, and mechanism analysis are cleanly isolated.

---

# 1. Top-Level Directory Layout

Project root should look like this:

project_root/
│
├── configs/
├── data/
├── features/
├── models/
├── training/
├── evaluation/
├── experiments/
├── analysis/
├── outputs/
├── docs/
├── scripts/
└── main.py

---

# 2. Directory Responsibilities

## 2.1 `configs/`

Purpose:
- configuration files
- experiment settings
- model hyperparameters
- split definitions

Recommended files:
- `base.yaml`
- `benchmark.yaml`
- `ablation.yaml`
- `mechanism.yaml`
- `robustness.yaml`

---

## 2.2 `data/`

Purpose:
- raw and processed tabular data
- event-level panels
- train/val/test split files

Recommended structure:

data/
├── raw/
├── processed/
├── panels/
└── splits/

Examples:
- `raw/raw_events.csv`
- `processed/event_panel.parquet`
- `splits/time_split_v1.csv`

---

## 2.3 `features/`

Purpose:
- feature extraction
- feature loading
- feature transformation

Recommended modules:
- `market_features.py`
- `control_features.py`
- `ecc_text_features.py`
- `ecc_audio_features.py`
- `proxy_features.py`
- `feature_registry.py`

---

## 2.4 `models/`

Purpose:
- define reusable model objects

Recommended modules:
- `market_prior_model.py`
- `ecc_residual_model.py`
- `proxy_noise_model.py`
- `minimax_gate.py`
- `abstention.py`

---

## 2.5 `training/`

Purpose:
- training logic for each stage

Recommended modules:
- `train_market_prior.py`
- `train_ecc_residual.py`
- `train_proxy_noise.py`
- `fit_full_pipeline.py`

---

## 2.6 `evaluation/`

Purpose:
- metric computation
- result aggregation
- selective prediction metrics

Recommended modules:
- `regression_metrics.py`
- `selective_metrics.py`
- `bootstrap.py`
- `significance_tests.py`

---

## 2.7 `experiments/`

Purpose:
- run predefined experiments

Recommended modules:
- `run_benchmarks.py`
- `run_ablations.py`
- `run_mechanism.py`
- `run_robustness.py`

---

## 2.8 `analysis/`

Purpose:
- plots
- tables
- diagnostics

Recommended modules:
- `make_main_table.py`
- `make_ablation_table.py`
- `make_mechanism_plots.py`
- `make_risk_coverage_plot.py`
- `proxy_diagnostics.py`

---

## 2.9 `outputs/`

Purpose:
- save experiment artifacts

Recommended structure:

outputs/
├── models/
├── predictions/
├── metrics/
├── plots/
├── tables/
└── logs/

Examples:
- `outputs/models/market_prior_xgb.pkl`
- `outputs/predictions/benchmark_finbert_test.csv`
- `outputs/metrics/main_results.csv`
- `outputs/plots/risk_coverage.png`

---

## 2.10 `docs/`

Purpose:
- all control markdown files

Required files:
- `project_north_star.md`
- `training_pipeline.md`
- `benchmark_spec.md`
- `evaluation_protocol.md`
- `experiment_matrix.md`
- `data_contract.md`
- `code_structure.md`
- `paper_alignment.md`

---

## 2.11 `scripts/`

Purpose:
- shell entry points
- reproducible run commands

Examples:
- `run_main.sh`
- `run_benchmarks.sh`
- `run_ablation.sh`
- `run_all_tables.sh`

---

# 3. Required Core Pipeline

The implementation MUST follow this order:

1. Load processed event-level panel
2. Build train/val/test split
3. Train market prior model
4. Compute residual target
5. Train ECC residual model
6. Compute ECC residual error
7. Train proxy noise model
8. Estimate signal variance `tau^2`
9. Compute shrinkage gate `alpha`
10. Produce final prediction
11. Evaluate
12. Save artifacts

This order MUST NOT be violated.

---

# 4. Core Python Modules

## 4.1 `models/market_prior_model.py`

Purpose:
Train the strong baseline using:
- market features
- control features

Required interface:

    class MarketPriorModel:
        def fit(self, X, y): ...
        def predict(self, X): ...
        def save(self, path): ...
        def load(self, path): ...

Output:
- predicted `mu_hat`

---

## 4.2 `models/ecc_residual_model.py`

Purpose:
Train ECC-based residual predictor

Input:
- ECC features
- residual target `r_tilde`

Required interface:

    class ECCResidualModel:
        def fit(self, X_ecc, residual): ...
        def predict(self, X_ecc): ...
        def save(self, path): ...
        def load(self, path): ...

Output:
- predicted `z`

---

## 4.3 `models/proxy_noise_model.py`

Purpose:
Map A4-like proxies to estimated error variance

Input:
- proxy features
- squared ECC error `u`

Required interface:

    class ProxyNoiseModel:
        def fit(self, X_proxy, u): ...
        def predict(self, X_proxy): ...
        def save(self, path): ...
        def load(self, path): ...

Output:
- estimated `sigma2`

Notes:
- should support monotone mapping
- may use isotonic regression or monotone boosting

---

## 4.4 `models/minimax_gate.py`

Purpose:
Compute minimax shrinkage coefficient

Required interface:

    def compute_gate(tau2, sigma2):
        return tau2 / (tau2 + sigma2)

Optional robust version:

    def compute_gate_upper(tau2, sigma2_upper):
        return tau2 / (tau2 + sigma2_upper)

Output:
- `alpha`

---

## 4.5 `models/abstention.py`

Purpose:
Apply abstention rule

Required interface:

    def apply_abstention(mu_hat, z_hat, alpha, sigma2, threshold):
        # if sigma2 > threshold, fallback to mu_hat
        ...

Output:
- final `y_hat`
- accept flag

---

# 5. Training Modules

## 5.1 `training/train_market_prior.py`

Responsibilities:
- load train/val split
- fit market prior model
- save model
- save predictions for train/val/test

Output files:
- `outputs/models/market_prior_*.pkl`
- `outputs/predictions/market_prior_train.csv`
- `outputs/predictions/market_prior_val.csv`
- `outputs/predictions/market_prior_test.csv`

---

## 5.2 `training/train_ecc_residual.py`

Responsibilities:
- read `mu_hat` predictions
- compute residual `r_tilde`
- fit ECC residual model
- save `z` predictions

Output files:
- `outputs/models/ecc_residual_*.pkl`
- `outputs/predictions/ecc_residual_train.csv`
- `outputs/predictions/ecc_residual_val.csv`
- `outputs/predictions/ecc_residual_test.csv`

---

## 5.3 `training/train_proxy_noise.py`

Responsibilities:
- read residual target and `z`
- compute `u = (r_tilde - z)^2`
- fit proxy-to-noise model
- save `sigma2` predictions

Output files:
- `outputs/models/proxy_noise_*.pkl`
- `outputs/predictions/proxy_sigma2_train.csv`
- `outputs/predictions/proxy_sigma2_val.csv`
- `outputs/predictions/proxy_sigma2_test.csv`

---

## 5.4 `training/fit_full_pipeline.py`

Responsibilities:
- orchestrate the whole pipeline
- train all stages in sequence
- estimate `tau^2`
- compute `alpha`
- produce final `y_hat`
- save all results

Output files:
- `outputs/predictions/final_main_test.csv`
- `outputs/metrics/main_metrics.json`

---

# 6. Benchmark Runner

## `experiments/run_benchmarks.py`

Purpose:
Run all external baselines

Must support:
- Market only
- Market + controls
- TF-IDF + Elastic Net
- FinBERT pooled
- MR-QA
- Text+audio fusion
- Ours

Expected output:
- one CSV per model
- one aggregated benchmark table

Output files:
- `outputs/metrics/benchmark_results.csv`
- `outputs/tables/benchmark_table.tex`

---

# 7. Ablation Runner

## `experiments/run_ablations.py`

Purpose:
Run only internal variants of our method

Variants:
- No gate
- Observed gate
- Minimax gate
- Minimax + abstention
- Proxy permutation
- Monotone vs unconstrained

Expected output:
- `outputs/metrics/ablation_results.csv`
- `outputs/tables/ablation_table.tex`

---

# 8. Mechanism Runner

## `experiments/run_mechanism.py`

Purpose:
Run mechanism analysis supporting the reliability story

Must support:
- reliability bucket vs ECC error
- proxy permutation test
- proxy noise injection
- audio uplift by reliability bucket

Output files:
- `outputs/metrics/mechanism_results.csv`
- `outputs/plots/reliability_buckets.png`
- `outputs/plots/proxy_noise_injection.png`

---

# 9. Robustness Runner

## `experiments/run_robustness.py`

Purpose:
Run non-core but important checks

Must support:
- time window variation
- sample variation
- temporal split variation

Output files:
- `outputs/metrics/robustness_results.csv`
- `outputs/tables/robustness_table.tex`

---

# 10. Main Program Entry

## `main.py`

Purpose:
Single command entry point

Must support modes:

    python main.py --mode benchmark
    python main.py --mode ablation
    python main.py --mode mechanism
    python main.py --mode robustness
    python main.py --mode full

Behavior:
- load config
- call correct experiment runner
- save outputs
- log metadata

---

# 11. Output Naming Convention

Every output file must include:
- experiment group
- model name
- split version
- date or run id

Example:
- `benchmark_finbert_split_v1_run01.csv`
- `ablation_minimax_split_v1_run01.csv`

---

# 12. Logging Requirements

Every run must save:
- config snapshot
- feature list
- split version
- random seed
- model hyperparameters
- date/time
- git hash if available

Log path:
- `outputs/logs/`

---

# 13. Reproducibility Requirements

- fixed random seed
- deterministic split file
- no hidden manual operations
- every figure must come from a script
- every table must be regenerable from saved metrics

---

# 14. Anti-Pattern Rules

The AI must NOT:
- merge benchmark and ablation logic
- train on test data
- use future information
- silently change target definition
- silently change sample filter
- build new experimental branches without approval

---

# 15. Minimal Execution Order

To reproduce the main paper:

1. Prepare processed panel
2. Run benchmark suite
3. Run main method
4. Run ablations
5. Run mechanism experiments
6. Generate tables and plots

---

# END