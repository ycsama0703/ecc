# Modeling Environment Recovery 2026-03-14

## Purpose

This note records the minimal Python environment that was reconstructed on the current server to rerun the core post-migration ICAIF baselines after the `scheduled_hour_et` panel fix.

Scope of this environment:
- `scripts/run_identity_classical_baselines.py`
- `scripts/run_target_variant_experiments.py`
- `scripts/run_regime_residual_experiments.py`
- `scripts/run_regime_subset_residual_experiments.py`

This is a minimal rerun environment, not yet a full end-to-end reproduction environment for every script in the repo.

Repo-managed minimal package file now available:
- `requirements-modeling-min.txt`

## Current Working Interpreter

- Virtual environment path: `/home/exouser/.venvs/icaif_ecc_news_attention`
- Interpreter path: `/home/exouser/.venvs/icaif_ecc_news_attention/bin/python`

## Verified Package Versions

- `numpy 2.4.3`
- `scipy 1.17.1`
- `scikit-learn 1.8.0`

Transitive packages installed with this environment:
- `joblib 1.5.3`
- `threadpoolctl 3.6.0`

## Rebuild Commands

```bash
mkdir -p /home/exouser/.venvs
/home/exouser/.local/bin/uv venv /home/exouser/.venvs/icaif_ecc_news_attention
/home/exouser/.local/bin/uv pip install \
  --python /home/exouser/.venvs/icaif_ecc_news_attention/bin/python \
  -r requirements-modeling-min.txt
```

## Verified Rerun Commands

```bash
/home/exouser/.venvs/icaif_ecc_news_attention/bin/python scripts/run_identity_classical_baselines.py \
  --panel-csv results/panel_real/event_modeling_panel.csv \
  --output-dir results/identity_classical_baselines_real

/home/exouser/.venvs/icaif_ecc_news_attention/bin/python scripts/run_target_variant_experiments.py \
  --panel-csv results/panel_real/event_modeling_panel.csv \
  --features-csv results/features_real/event_text_audio_features.csv \
  --audio-csv results/audio_real/event_real_audio_features.csv \
  --qa-csv results/qa_benchmark_features_real/qa_benchmark_features.csv \
  --output-dir results/target_variant_experiments_real

/home/exouser/.venvs/icaif_ecc_news_attention/bin/python scripts/run_regime_residual_experiments.py \
  --panel-csv results/panel_real/event_modeling_panel.csv \
  --features-csv results/features_real/event_text_audio_features.csv \
  --audio-csv results/audio_real/event_real_audio_features.csv \
  --qa-csv results/qa_benchmark_features_real/qa_benchmark_features.csv \
  --output-dir results/regime_residual_experiments_real

/home/exouser/.venvs/icaif_ecc_news_attention/bin/python scripts/run_regime_subset_residual_experiments.py \
  --panel-csv results/panel_real/event_modeling_panel.csv \
  --features-csv results/features_real/event_text_audio_features.csv \
  --audio-csv results/audio_real/event_real_audio_features.csv \
  --qa-csv results/qa_benchmark_features_real/qa_benchmark_features.csv \
  --output-dir results/regime_subset_experiments_real
```

## Current Limitation

The repo still does not contain a checked-in environment file such as:
- `requirements.txt`
- `pyproject.toml`
- `environment.yml`

So this recovery note is a stopgap rather than a final reproducibility solution.

Recommended next cleanup:
- add any additional dependencies needed by the full audio-feature and QC pipelines
- promote the minimal package file into a fuller end-to-end environment spec if the full pipeline needs to be reproduced on a fresh machine
