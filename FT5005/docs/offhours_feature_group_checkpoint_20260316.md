# Off-Hours Feature-Group Checkpoint 2026-03-16

## 1. Why This Checkpoint Exists

The corrected off-hours headline result is already strong, but one key question remained open:

- does the best corrected result actually contain incremental ECC information beyond `market + controls`,
- or is it mostly a market-state result that only looks "ECC-driven" because the feature bundle is mixed together?

This checkpoint reruns the corrected residual-on-prior benchmark in a stricter feature-group ladder so we can separate:

- pre-call market state,
- within-call market state,
- analyst and scheduling controls,
- transcript-structure features from `A1/A2`,
- alignment and observability features from `A4`,
- and `Q&A` semantics from `qna_lsa`.

## 2. Experimental Design

Locked setting:

- target: `shock_minus_pre`
- formulation: same-ticker prior plus residual ridge
- feature table: corrected panel plus `qav2`
- main slices:
  - pooled `pre_market + after_hours`
  - clean pooled `pre_market + after_hours` with `html_integrity_flag=fail` removed
  - `after_hours` only
  - clean `after_hours` only
  - `pre_market` only

Feature ladder:

- `prior_only`
- `pre_call_market_only`
- `market_only`
- `market_plus_controls`
- `market_controls_plus_a1_a2`
- `market_controls_plus_a4`
- `market_controls_plus_ecc_structure`
- `market_controls_plus_ecc_structure_plus_qna_lsa`

New script:

- `scripts/run_offhours_feature_group_ladder.py`

Primary result files:

- `results/offhours_feature_group_ladder_corrected_qav2_real/offhours_feature_group_ladder_summary.json`
- `results/offhours_feature_group_ladder_corrected_qav2_clean_real/offhours_feature_group_ladder_summary.json`
- `results/offhours_feature_group_ladder_corrected_qav2_after_hours_real/offhours_feature_group_ladder_summary.json`
- `results/offhours_feature_group_ladder_corrected_qav2_after_hours_clean_real/offhours_feature_group_ladder_summary.json`
- `results/offhours_feature_group_ladder_corrected_qav2_pre_market_real/offhours_feature_group_ladder_summary.json`

## 3. Main Findings

### 3.1 Pooled off-hours is still mostly market-dominated

Corrected all-HTML pooled off-hours:

- `prior_only`: `R^2 ≈ 0.191`
- `pre_call_market_only`: `R^2 ≈ 0.902`
- `market_only`: `R^2 ≈ 0.907`
- `market_plus_controls`: `R^2 ≈ 0.909`
- `market_controls_plus_ecc_structure`: `R^2 ≈ 0.912`

Corrected clean pooled off-hours:

- `prior_only`: `R^2 ≈ 0.198`
- `pre_call_market_only`: `R^2 ≈ 0.899`
- `market_only`: `R^2 ≈ 0.905`
- `market_plus_controls`: `R^2 ≈ 0.911`
- `market_controls_plus_ecc_structure`: `R^2 ≈ 0.913`

Interpretation:

- the pooled off-hours result is overwhelmingly explained by market-state and simple control variables,
- the incremental gain from adding the current ECC-structure block is numerically small,
- and the paired permutation tests do not show a decisive pooled off-hours win for `ecc_structure` over `market_plus_controls`.

This means the pooled off-hours result is still scientifically useful, but it should be presented as:

- a strong corrected forecasting benchmark,
- not as proof that ECC structure robustly dominates market-aware baselines in every off-hours slice.

### 3.2 `pre_market` is not the right main story

Corrected `pre_market` only:

- `prior_only`: `R^2 ≈ -0.787`
- `market_only`: `R^2 ≈ -0.234`
- `market_plus_controls`: `R^2 ≈ -0.043`
- `market_controls_plus_a4`: `R^2 ≈ -0.026`
- `market_controls_plus_ecc_structure`: `R^2 ≈ -0.046`

Interpretation:

- `pre_market` improves substantially relative to the prior,
- but it remains noisy and too weak to support a clean headline contribution claim.

The paper should therefore not treat `pre_market` and `after_hours` as equally informative.

### 3.3 The new positive signal concentrates in `after_hours`

Corrected `after_hours` all-HTML:

- `prior_only`: `R^2 ≈ 0.035`
- `pre_call_market_only`: `R^2 ≈ 0.918`
- `market_plus_controls`: `R^2 ≈ 0.893`
- `market_controls_plus_ecc_structure`: `R^2 ≈ 0.868`
- `market_controls_plus_ecc_structure_plus_qna_lsa`: `R^2 ≈ 0.926`

Important reading:

- the pooled off-hours story hides strong regime heterogeneity,
- in noisy all-HTML `after_hours`, richer `Q&A` semantics can rescue the ECC-augmented model,
- and `qna_lsa` significantly improves over the all-HTML `after_hours` `ecc_structure` model:
  - permutation `mse p ≈ 0.0028`
  - permutation `mae p ≈ 0.006`

This is not yet a universal "text always helps" claim.
It is a regime-specific positive result:

- when the sample is narrowed to `after_hours`,
- and the ECC structure block is present,
- `Q&A` semantics become meaningfully helpful on the noisier all-HTML slice.

### 3.4 Clean `after_hours` reveals the clearest integrity-driven ECC increment

Corrected clean `after_hours`:

- `prior_only`: `R^2 ≈ 0.035`
- `pre_call_market_only`: `R^2 ≈ 0.917`
- `market_only`: `R^2 ≈ 0.902`
- `market_plus_controls`: `R^2 ≈ 0.875`
- `market_controls_plus_a4`: `R^2 ≈ 0.909`
- `market_controls_plus_ecc_structure`: `R^2 ≈ 0.879`
- `market_controls_plus_ecc_structure_plus_qna_lsa`: `R^2 ≈ 0.899`

Most important positive result:

- `market_controls_plus_a4` significantly beats `market_plus_controls` on clean `after_hours`:
  - permutation `mse p ≈ 0.0075`
  - permutation `mae p ≈ 0.0073`

Interpretation:

- the most credible incremental ECC value is currently not "all ECC features together",
- it is specifically the `A4`-derived alignment and observability block in clean `after_hours`.

This is a much stronger paper story than a generic multimodal claim because it is directly limitation-driven:

- scheduled time is noisy,
- transcript quality varies,
- and once low-integrity rows are removed,
- alignment or observability features become the clearest incremental gain over the market-aware baseline.

## 4. What This Changes In The Paper Story

The project should now be described more precisely.

Not:

- "off-hours pooled multimodal ECC features strongly beat all baselines"

Closer to:

- pooled off-hours forecasting is strong but mostly market-dominated,
- `pre_market` is a weaker and noisier regime,
- the most credible incremental ECC contribution lives in clean `after_hours`,
- where `A4` observability or alignment structure significantly improves over `market + controls`,
- and `Q&A` semantics help most on noisier all-HTML `after_hours`.

## 5. Best Current Contribution Package

If we were writing the paper from the latest evidence, the cleanest contribution package would be:

1. Reformulate the task as corrected `after_hours` or off-hours `shock_minus_pre` prediction under noisy scheduled timing.
2. Evaluate everything against same-ticker priors and strong market-aware baselines.
3. Show that pooled off-hours gains can be overstated if regime mixing is ignored.
4. Show that clean `after_hours` gains are most credible when they come from `A4` observability or alignment structure.
5. Treat `pre_market` weakness and pooled-feature negative results as part of the scientific message, not as noise to hide.

## 6. Recommended Next Step

The next research step should not be "add a bigger architecture."

It should be one of these:

- strengthen the `A4` pathway and build a cleaner `after_hours` main model around observability and alignment quality,
- or build a small `after_hours` text-plus-`A4` extension explicitly targeted at the subset where `Q&A` semantics already help.

That would give the project a sharper limitation-driven novelty claim and a more defensible empirical story.
