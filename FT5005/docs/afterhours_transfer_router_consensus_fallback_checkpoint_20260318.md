# After-Hours Transfer Router Consensus Fallback Checkpoint 2026-03-18

## Purpose

The agreement-based consensus checkpoint showed that requiring agreement between the retained hybrid tree and hybrid-plus-text logistic routes can improve pooled temporal robustness.

The next natural question was:

- if agreement is helpful,
- **what should the model fall back to when the two routers disagree?**

That is a much more targeted question than expanding expert families or router complexity again.

## Design

New script:

- `scripts/run_afterhours_transfer_router_consensus_fallback_benchmark.py`

Source inputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`

Same temporal confirmation windows:

- `val2020_test_post2020`
- `val2021_test_post2021`
- `val2022_test_post2022`

Common rule:

- if tree and logistic agree, use the agreed expert

Disagreement fallback families compared:

1. `pre_call_market_only`
2. retained semantic+audio backbone
3. validation-selected expert
4. pair-tree output
5. plus-text logistic output
6. average of tree and logistic on disagreement

Outputs:

- `results/afterhours_transfer_router_consensus_fallback_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_router_consensus_fallback_summary.json`
- `results/afterhours_transfer_router_consensus_fallback_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_router_consensus_fallback_overview.csv`
- `results/afterhours_transfer_router_consensus_fallback_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_router_consensus_fallback_predictions.csv`

## Main Findings

### 1. The best pooled temporal route is now **agreement + pre-call-market fallback**

Across all temporal confirmation windows combined:

- pre-call market only `≈ 0.997886`
- retained semantic+audio expert `≈ 0.997878`
- selected expert `≈ 0.997879`
- pair tree `≈ 0.997901`
- plus-text logistic `≈ 0.997899`
- agreement + retained fallback `≈ 0.997902`
- agreement + disagreement average `≈ 0.997903`
- **agreement + pre-call-market fallback `≈ 0.997919`**

So the strongest pooled temporal route in the current transfer-confirmation family is now:

- **agreement-triggered abstention to `pre_call_market_only`**

### 2. This route is stronger than the other fallback families, not just the old baselines

Pooled paired tests:

- versus pair tree:
  - `p(MSE) ≈ 0.049`
- versus disagreement average:
  - `p(MSE) ≈ 0.0495`
- versus agreement + retained semantic fallback:
  - `p(MSE) ≈ 0.068`

It still does **not** significantly beat raw `pre_call_market_only` itself:

- versus pre-call market only:
  - `p(MSE) ≈ 0.197`

So the right interpretation is:

- when the routers agree, there is useful transfer-side incremental signal,
- but when they disagree, the safest fallback is the strongest market baseline rather than a weaker transfer route.

### 3. The fallback benchmark clarifies why the pre-call market baseline remains so hard to beat

The best fallback family only sends about `10%` of pooled events to the `Q&A` expert:

- agreement + pre-call fallback QA share `≈ 0.101`

This is much lower than the more aggressive disagreement policies:

- selected fallback QA share `≈ 0.202`
- tree fallback QA share `≈ 0.272`
- logistic fallback QA share `≈ 0.249`

So the best current temporal route is not the one that pushes harder into transfer-side deviations.

It is the one that **abstains more aggressively** when the reliability evidence is mixed.

### 4. Split-by-split picture

By split:

- `val2020_test_post2020`
  - fallback-to-pre `≈ 0.996682`
  - best among the current fallback families
- `val2021_test_post2021`
  - fallback-to-pre `≈ 0.998823`
  - very close to raw `pre_call_market_only ≈ 0.998826`
  - clearly above agreement + retained fallback and above pair tree
- `val2022_test_post2022`
  - fallback-to-pre `≈ 0.998640`
  - slightly above plus-text logistic `≈ 0.998636`
  - slightly above disagreement average `≈ 0.998639`

So this fallback family beats:

- retained semantic fallback in `3 / 3` windows
- pair tree in `3 / 3` windows
- plus-text logistic in `3 / 3` windows
- pre-call market only in `2 / 3` windows

## Updated Interpretation

This checkpoint sharpens the transfer-side contribution again:

- the value is no longer just “routing among experts”
- it is now more clearly **reliability-aware abstention**

The strongest temporal route currently says:

- use transfer-side agreement when both retained router views support it,
- otherwise abstain to the strongest market baseline.

That is a cleaner, more defensible method idea than continuing to add expert families.

## Practical Consequence

The safe repo conclusion is now:

1. keep the fixed-split semantic headline unchanged,
2. keep the transfer-side hybrid tree and hybrid-plus-text logistic routes as the two key router views,
3. treat **agreement-triggered abstention to `pre_call_market_only`** as the strongest current pooled temporal transfer route,
4. interpret the gain as evidence for **reliability-aware abstention**, not just stronger routing,
5. and still avoid overclaiming because this route does not yet significantly beat raw `pre_call_market_only`.

So the transfer-side story is getting cleaner:

- direct complementary-expert expansion still looks weak,
- router agreement matters,
- and the safest disagreement behavior is to fall back to the strongest market baseline.
