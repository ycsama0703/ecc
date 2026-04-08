## Current Hybrid PREC Report

This note summarizes the latest method change and the full benchmark table for
the current strongest version:

- panel / split source: selective Experiment A data
- residual feature mode: `ecc_plus_market_controls`
- target coverage setting: `0.80`
- main outputs: [h80_main](/mnt/d/NUS/Nus_coursework/SEM2/FT5005/ecc/results/h80_main)
- benchmark outputs: [h80_bench](/mnt/d/NUS/Nus_coursework/SEM2/FT5005/ecc/results/h80_bench)

### What Changed

The original PREC pipeline used:
- `mu = f(market + controls)`
- `z = f(ECC only)`
- `sigma2 = g(proxy)`
- final prediction: `mu + A * alpha * z`

The hybrid version changes only the residual predictor:
- `mu = f(market + controls)` stays unchanged
- `z = f(ECC + market_controls)`
- `sigma2`, `alpha`, and `A` still follow the same proxy / gate / abstention design

This means strong market/control variables are now available both:
- in the prior `mu`
- in the correction `z`

### Why This Matters

The previous selective A result suggested:
- the `PREC` framework itself was not broken
- but the residual predictor `z` was too weak

After adding market/control features to `z`, the family comparison changed
dramatically:

| Family Model | Full-set R² | Full-set MAE | Coverage | Accepted-set R² | Accepted-set MSE | Gain over prior |
|---|---:|---:|---:|---:|---:|---:|
| `prior_only` | 0.1106 | 0.000947 | 1.0000 | 0.1106 | 3.3843e-06 | 0 |
| `prior_plus_z_no_gate` | 0.9390 | 0.000316 | 1.0000 | 0.9390 | 2.3211e-07 | 3.1522e-06 |
| `prior_plus_alpha_z_gate_only` | 0.9475 | 0.000282 | 1.0000 | 0.9475 | 1.9971e-07 | 3.1846e-06 |
| `prec_selective` | 0.9313 | 0.000313 | 0.9467 | 0.9523 | 1.8692e-07 | 3.1229e-06 |

Interpretation:
- the hybrid residual `z` became very strong
- gate-only is the strongest point-prediction variant
- the selective version remains strong, with slightly lower full-set R² but better accepted-set precision

### Main Result Comparison

Reference baseline from the earlier selective A version:

| Version | Full-set R² | Full-set MAE | Coverage | Accepted-set R² | Accepted-set MSE | AURC |
|---|---:|---:|---:|---:|---:|---:|
| Original selective A (`ecc_only`) | 0.1172 | 0.000904 | 0.7333 | 0.1942 | 1.7926e-06 | 1.3590e-06 |
| Hybrid PREC (`ecc_plus_market_controls`, cov80) | 0.9313 | 0.000313 | 0.9467 | 0.9523 | 1.8692e-07 | 2.0908e-07 |

This is the clearest current conclusion:
- the large improvement comes from making `z` much stronger
- not from changing the prior or changing the proxy/gate formulas

### Full Benchmark Table

Source:
- [comprehensive_results_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv](/mnt/d/NUS/Nus_coursework/SEM2/FT5005/ecc/outputs/exp_a_sel_hybrid_cov80/metrics/comprehensive_results_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv)

| Group | Model | Full-set R² | Full-set MSE | Full-set MAE | nMAE | Relative MAE Improvement | Spearman | Coverage | Accepted-set R² | Accepted-set MSE | AURC | Gain over prior |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| market_baseline | `market_only` | 0.1089 | 3.3906e-06 | 0.000950 | 0.4872 | -0.0039 | 0.8489 | 1.0000 | 0.1089 | 3.3906e-06 |  | -6.3e-09 |
| market_baseline | `market_plus_controls` | 0.1106 | 3.3843e-06 | 0.000947 | 0.4853 | 0.0000 | 0.8583 | 1.0000 | 0.1106 | 3.3843e-06 |  | 0 |
| machine_learning | `xgboost_market_controls` | 0.1106 | 3.3843e-06 | 0.000947 | 0.4853 | 0.0000 | 0.8583 | 1.0000 | 0.1106 | 3.3843e-06 |  | 0 |
| machine_learning | `lightgbm_market_controls` | 0.1840 | 3.1048e-06 | 0.000837 | 0.4291 | 0.1159 | 0.8614 | 1.0000 | 0.1840 | 3.1048e-06 |  | 2.7950e-07 |
| machine_learning | `random_forest_market_controls` | 0.8930 | 4.0710e-07 | 0.000207 | 0.1060 | 0.7815 | 0.9062 | 1.0000 | 0.8930 | 4.0710e-07 |  | 2.9772e-06 |
| ecc_baseline | `tfidf_elasticnet` | -0.0307 | 3.9219e-06 | 0.001089 | 0.5585 | -0.1508 |  | 1.0000 | -0.0307 | 3.9219e-06 |  | -5.3760e-07 |
| ecc_baseline | `compact_qa_baseline` | 0.0044 | 3.7884e-06 | 0.001063 | 0.5447 | -0.1224 | 0.1432 | 1.0000 | 0.0044 | 3.7884e-06 |  | -4.0410e-07 |
| ecc_baseline | `finbert_pooled` | 0.0651 | 3.5572e-06 | 0.001063 | 0.5451 | -0.1231 | 0.4286 | 1.0000 | 0.0651 | 3.5572e-06 |  | -1.7300e-07 |
| prior_correction | `prior_only` | 0.1106 | 3.3843e-06 | 0.000947 | 0.4853 | 0.0000 | 0.8583 | 1.0000 | 0.1106 | 3.3843e-06 |  | 0 |
| prior_correction | `prior_plus_z_no_gate` | 0.9390 | 2.3210e-07 | 0.000316 | 0.1622 | 0.6658 | 0.6780 | 1.0000 | 0.9390 | 2.3210e-07 |  | 3.1522e-06 |
| prior_correction | `prior_plus_alpha_z_gate_only` | 0.9475 | 1.9970e-07 | 0.000282 | 0.1445 | 0.7022 | 0.6818 | 1.0000 | 0.9475 | 1.9970e-07 |  | 3.1846e-06 |
| prior_correction | `prec_selective` | 0.9313 | 2.6130e-07 | 0.000313 | 0.1607 | 0.6689 | 0.7139 | 0.9467 | 0.9523 | 1.8690e-07 | 2.0910e-07 | 3.1229e-06 |

### Benchmark Interpretation

#### 1. Market baselines
- `market_plus_controls` is still the fair reference baseline
- `xgboost_market_controls` is effectively the same model family and gives the same result

#### 2. Machine-learning group
- `lightgbm_market_controls` is a genuinely stronger nonlinear baseline than the default market prior
- `random_forest_market_controls` is extremely strong on this task

Current interpretation of Random Forest:
- it is now fair in feature scope
- but it should still be described as a very strong nonlinear market/control benchmark
- it is not a selective reliability-aware method

#### 3. ECC baselines
- all text/ECC-only baselines remain much weaker than the hybrid correction family
- this suggests the main gain is not from raw text alone

#### 4. Prior + correction family
- `mu + z` is already excellent
- `mu + alpha z` is slightly better
- `mu + A alpha z` trades a small amount of full-set accuracy for selective behavior and better accepted-set precision

### Coverage Sweep

Source:
- [selective_summary_by_target_coverage_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv](/mnt/d/NUS/Nus_coursework/SEM2/FT5005/ecc/outputs/exp_a_sel_hybrid_cov80/metrics/selective_summary_by_target_coverage_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv)

Important rows:

| Target Coverage | Test Realized Coverage | Test Full-set R² | Test Accepted-set R² | Test Accepted-set MSE |
|---|---:|---:|---:|---:|
| 0.20 | 0.0800 | 0.1182 | -1.0695 | 1.0435e-07 |
| 0.40 | 0.3200 | 0.1332 | 0.5707 | 2.8758e-07 |
| 0.80 | 0.9467 | 0.9313 | 0.9523 | 1.8692e-07 |
| unconstrained | 0.9467 | 0.9313 | 0.9523 | 1.8692e-07 |

Interpretation:
- for the hybrid model, low-coverage thresholds are too conservative
- the useful operating regime is near `0.80`
- once coverage is relaxed, PREC keeps most of the strong hybrid residual benefit

### Practical Bottom Line

At this point the most defensible current statement is:

1. The original weak result was largely due to a weak residual correction model.
2. Adding strong market/control signals to the residual layer changes the picture completely.
3. The best current version is:
   - `ecc_plus_market_controls`
   - target coverage `0.80`
4. In this setting, `PREC` becomes competitive with the strongest benchmarks and clearly dominates all ECC-only baselines.
5. `RandomForest` is still the strongest overall benchmark, but it solves a different problem class:
   - full-set point prediction
   - no reliability-aware abstention

### Recommended Files

- Main report:
  - [main_metrics_a_sel_hybrid_cov80_v1_main_hybrid80_01.json](/mnt/d/NUS/Nus_coursework/SEM2/FT5005/ecc/outputs/exp_a_sel_hybrid_cov80/metrics/main_metrics_a_sel_hybrid_cov80_v1_main_hybrid80_01.json)
- Family breakdown:
  - [prior_correction_family_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv](/mnt/d/NUS/Nus_coursework/SEM2/FT5005/ecc/outputs/exp_a_sel_hybrid_cov80/metrics/prior_correction_family_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv)
- Full benchmark table:
  - [comprehensive_results_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv](/mnt/d/NUS/Nus_coursework/SEM2/FT5005/ecc/outputs/exp_a_sel_hybrid_cov80/metrics/comprehensive_results_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv)
- Coverage sweep:
  - [selective_summary_by_target_coverage_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv](/mnt/d/NUS/Nus_coursework/SEM2/FT5005/ecc/outputs/exp_a_sel_hybrid_cov80/metrics/selective_summary_by_target_coverage_a_sel_hybrid_cov80_v1_main_hybrid80_01.csv)
