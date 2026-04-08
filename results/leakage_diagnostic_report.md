# Leakage Diagnostic Report

## 1. Time Leakage
- post-start features present: YES
- count flagged as `TIME_LEAKAGE`: 47
- key flagged features: RV_post_60m, a4_kept_rows_for_duration, alignment_score, audio_completeness, call_duration_min, call_duration_sec, call_end_datetime, post_call_60m_bar_count, post_call_60m_volume_sum, post_call_60m_vw_rv, post_window_first_bar, post_window_last_bar, proxy_quality_mean, qa_embedding_0, qa_embedding_1, qa_embedding_2, qa_embedding_3, qa_embedding_4, qa_embedding_5, qa_embedding_6
- severity: HIGH

## 2. Target Leakage
- high-correlation features vs `RV_post` or `y`: 6
- most suspicious features: RV_post_60m, post_call_60m_vw_rv, RV_pre_60m, pre_60m_rv, pre_call_volatility, pre_60m_vw_rv

## 3. Residual Structure
- `R2_z_only`: 0.9366
- overlap count between `mu` and `z`: 20
- overlapping features: analyst_eps_norm_num_est, analyst_eps_norm_std, analyst_net_income_num_est, analyst_net_income_std, analyst_revenue_num_est, analyst_revenue_std, call_duration_min, ebitda_surprise_pct, eps_gaap_surprise_pct, firm_size, historical_volatility, pre_60m_rv, pre_60m_volume_sum, pre_60m_vw_rv, revenue_surprise_pct, scheduled_hour_et, sector, within_call_rv, within_call_volume_sum, within_call_vw_rv
- residual collapse: YES

## 4. Feature Ablation
- `R2_A (pre_call + ECC)`: 0.9899
- `R2_B (pre_call + within_call)`: 0.9368
- `R2_C (pre_call + post_call)`: 0.9999
- strongest abnormal jump source: post_call_features

## 5. Data Split
- split type: time-based split
- train firms: 24
- val firms: 22
- test firms: 24
- train/test overlap count: 24
- train/val overlap count: 22
- val/test overlap count: 22
- potential panel leakage: YES

## 6. ML Behavior
| model | r2 | mae | mse |
| --- | --- | --- | --- |
| ElasticNet | 0.9575804589341431 | 0.00022958052889675813 | 1.614100748562534e-07 |
| RandomForest | 0.8927375768116429 | 0.0002081182357178489 | 4.0814292943944936e-07 |
| LightGBM | 0.6155780335135859 | 0.0005376469714379915 | 1.4627593045060873e-06 |

## 7. Final Conclusion
- leakage present: YES
- most likely source: within_call market features and duplicated market/control usage in both mu and z
- severity: HIGH
- remove immediately: within_call_rv, within_call_vw_rv, within_call_volume_sum, call_duration_min, RV_post_60m, post_call_60m_vw_rv, post_call_60m_volume_sum

## Final Answer
- Current `0.93+` R² is mainly from: information leakage / near-leakage from within-call and post-call features, amplified by residual duplication
- Primary leakage class: TIME_LEAKAGE via within_call/post_call features plus RESIDUAL_COLLAPSE
- Features to remove from the main experiment: within_call_rv, within_call_vw_rv, within_call_volume_sum, call_duration_min, RV_post_60m, post_call_60m_vw_rv, post_call_60m_volume_sum

## Notes
- Under a strict pre-call forecasting definition, `within_call_*`, transcript-derived features, and A4 alignment features all use post-start information.
- The hybrid residual additionally reuses `market + controls` in both `mu` and `z`, which breaks clean residual separation even if feature scope is aligned.