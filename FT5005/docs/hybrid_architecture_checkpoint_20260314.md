# Hybrid Architecture Checkpoint 2026-03-14

This note records the first architecture-upgrade round after the `qav2` feature refresh.

## Goal

Test whether stronger model structure can recover incremental ECC value even when single-model ridge baselines still trail the strongest market-side benchmark.

## Script and outputs

- Script: `scripts/run_hybrid_architecture_experiments.py`
- Output folder: `results/hybrid_architecture_qav2_real/`

Saved result files:
- `hybrid_architecture_shock_minus_pre_all_regimes_all_html.json`
- `hybrid_architecture_shock_minus_pre_all_regimes_exclude-fail.json`
- `hybrid_architecture_shock_minus_pre_after_hours-pre_market_all_html.json`

## Methods tried

Base experts:
- `market_controls_ridge`
- `market_controls_hgbr`
- `ecc_text_ridge`
- `full_ridge`

Hybrid upgrades:
- global blend of `market_controls_ridge` and `full_ridge`
- regime-gated blend of `market_controls_ridge` and `full_ridge`
- positive linear stack over base-expert residual predictions
- gated `HistGradientBoostingRegressor` stack using base-expert residuals plus a small gate-feature bundle

## Test-side results

| Slice | Strong market ridge | Full ridge | Regime-gated blend | Positive stack | Gated stack HGBR |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full sample, all HTML rows | 0.909 | 0.884 | 0.908 | 0.904 | 0.287 |
| Full sample, exclude html_integrity_flag=fail | 0.911 | 0.878 | 0.910 | 0.905 | 0.308 |
| Off-hours only, all HTML rows | 0.909 | 0.892 | 0.906 | 0.904 | 0.285 |

Additional findings:
- `market_controls_hgbr` remains far below the ridge market baseline on every slice, around `0.495` to `0.512` test `R^2`.
- `ecc_text_ridge` stays weak or unstable:
  - about `0.095` on the full all-HTML sample,
  - about `0.139` on off-hours,
  - about `-0.005` on the clean full sample.
- the positive stack mostly collapses to the `full_ridge` expert:
  - on the full sample it assigns weight about `0.935` to `full_ridge` and `0` to the other experts.

## Interpretation

- The best hybrid result still does not beat the strongest market-only residual baseline.
- The regime-gated blend is the closest competitor, but it remains slightly below the market ridge on every tested slice.
- The nonlinear gated stack overfits badly despite looking more "SOTA-like" architecturally.
- The tree-based market expert also underperforms the simpler ridge market baseline.

## Current decision

These experiments strengthen the current research reading:
- architecture upgrades alone are not enough to unlock robust ECC incrementality,
- the bottleneck is still signal quality and supervision,
- so the next serious modeling push should prioritize:
  - transferred pair-level `Q&A` quality or evasion labels,
  - cleaner benchmark-adjusted targets,
  - or a public-data benchmark pivot if stronger ECC labels still fail.

In short:
- keep the hybrid script as a reusable benchmark,
- but do not spend the next cycle stacking more complex experts on top of the same weak ECC supervision.
