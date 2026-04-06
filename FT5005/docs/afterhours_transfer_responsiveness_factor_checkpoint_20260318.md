# After-Hours Transfer Responsiveness-Factor Checkpoint 2026-03-18

## Purpose

After the agreement-signal benchmark, the next clean research question was no longer:

- can we keep expanding upstream feature families,
- or can we keep tuning the router shell around the same proxies?

The better question is:

## can the current compact upstream quality signals be **distilled into one learnable responsiveness factor**?

This is a useful step because it tests a more disciplined method idea:

- fewer hand-built feature blocks,
- one compact latent,
- clearer interpretation,
- and an easier transfer story if it works.

So this checkpoint is deliberately about **distillation**, not more module piling.

## Design

New script:

- `scripts/run_afterhours_transfer_responsiveness_factor_benchmark.py`

Inputs:

- temporal router outputs:
  - `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`
- compact side inputs:
  - `results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv`
  - `results/features_real/event_text_audio_features.csv`
  - `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`
  - `results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv`
- reference predictions from the prior agreement-signal benchmark:
  - `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_agreement_signal_benchmark_test_predictions.csv`

Coverage remains complete on the temporal benchmark rows:

- `257 / 257` rows have all side inputs

The shell stays fixed:

- disagreement still falls back to `pre_call_market_only`
- only agreement events are refined

What changes is the upstream representation.

### Distilled factor families

1. `responsiveness_core`
   - `qa_pair_answer_forward_rate_mean`
   - `qa_evasive_proxy_share`
   - `qa_bench_coverage_mean`
   - `qa_bench_direct_early_score_mean`
   - `qa_bench_evasion_score_mean`
   - `qa_pair_count`
2. `responsiveness_plus_observability`
   - `responsiveness_core`
   - `a4_strict_row_share`
3. `directness_coverage_core`
   - `qa_pair_answer_forward_rate_mean`
   - `qa_bench_coverage_mean`
   - `qa_bench_direct_early_score_mean`
   - `qa_bench_evasion_score_mean`
   - `qa_bench_direct_answer_share`
4. `observability_directness_core`
   - `a4_strict_row_share`
   - `a4_strict_high_conf_share`
   - `qa_pair_count`
   - `qa_bench_coverage_mean`
   - `qa_bench_direct_early_score_mean`
   - `qa_bench_evasion_score_mean`

### Factor builders

For each family, the benchmark fits a single latent with:

- `pca1` (unsupervised one-component PCA)
- `pls1` (supervised one-component PLS)

### Route variants

Each latent is tested in two compact ways:

1. `factor_only`
   - the factor alone decides whether to use the agreed expert or fall back to market
2. `geometry_plus_factor`
   - current geometry gate plus one distilled factor

Temporal protocol:

- train on `val2020_test_post2020` agreement events
- tune on `val2021_test_post2021` agreement events
- refit on `2020 + 2021` agreement events
- test on full `val2022_test_post2022`

Outputs:

- `results/afterhours_transfer_responsiveness_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_responsiveness_factor_benchmark_summary.json`
- `results/afterhours_transfer_responsiveness_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_responsiveness_factor_benchmark_overview.csv`
- `results/afterhours_transfer_responsiveness_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_responsiveness_factor_benchmark_tuning.csv`
- `results/afterhours_transfer_responsiveness_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_responsiveness_factor_benchmark_factor_loadings.csv`
- `results/afterhours_transfer_responsiveness_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_responsiveness_factor_benchmark_test_predictions.csv`

## Main Findings

### 1. A single learnable responsiveness factor can recover almost all of the earlier multi-feature upstream signal

Best distilled route:

- **`responsiveness_plus_observability__pls1__geometry_plus_factor ≈ 0.998639314`**

References:

- `geometry_only ≈ 0.998638784`
- prior best `geometry_plus_hybrid ≈ 0.998638845`
- **`agreement_pre_only_abstention ≈ 0.998640168`**

So the distilled route:

- slightly edges the earlier `geometry_plus_hybrid` route numerically,
- stays essentially tied with the current geometry-led family,
- but still does **not** beat hard abstention.

Paired comparison:

- best distilled route vs `geometry_plus_hybrid`: `p(MSE) ≈ 0.506`
- best distilled route vs `geometry_only`: `p(MSE) ≈ 0.751`
- best distilled route vs hard abstention: `p(MSE) ≈ 0.753`

Interpretation:

- the current upstream quality proxies do contain a coherent low-dimensional direction,
- but that direction is still too weak to replace the simpler abstention rule.

### 2. Distillation is cleaner than raw quality-only families

Best `factor_only` route:

- **`directness_coverage_core__pca1__factor_only ≈ 0.998559`**

Earlier raw quality-only references:

- `lite_quality ≈ 0.998482`
- `hybrid_quality ≈ 0.998470`

So compressing the upstream quality block into one factor is actually more effective than using those raw quality families directly.

But the ceiling is still limited:

- `directness_coverage_core__pca1__factor_only` stays well below the geometry-led routes,
- and is not a robust win over `pre_call_market_only`.

So factor distillation helps **discipline** the signal, but does not solve the main transfer problem on its own.

### 3. The best latent is interpretable

Top loadings for the best route (`responsiveness_plus_observability__pls1__geometry_plus_factor`):

- `qa_pair_answer_forward_rate_mean` with a strong negative weight
- `qa_evasive_proxy_share` with a positive weight
- `qa_bench_direct_early_score_mean` with a positive weight
- `qa_bench_evasion_score_mean` with a negative weight
- `a4_strict_row_share` with a modest positive weight
- `qa_pair_count` with a modest negative weight
- `qa_bench_coverage_mean` with a modest positive weight

That is a useful scientific result because it shows the latent is not arbitrary:

- it is centered on a compact **responsiveness / directness / coverage / evasion** axis,
- with a smaller observability contribution,
- rather than a diffuse pile of unrelated features.

### 4. Observability helps only marginally inside the latent

The pure `responsiveness_core` PLS route reaches:

- `responsiveness_core__pls1__geometry_plus_factor ≈ 0.998639212`

while the observability-augmented version reaches:

- `responsiveness_plus_observability__pls1__geometry_plus_factor ≈ 0.998639314`

So adding `a4_strict_row_share` helps only slightly.

This matters because it suggests:

- the main distilled upstream direction is still mostly **Q&A responsiveness quality**,
- while observability remains supportive but not dominant inside this compact latent.

## Updated Interpretation

This checkpoint is a good narrowing result.

### What we now know

1. The current upstream quality proxies can be compressed into a single interpretable responsiveness factor.
2. That single-factor route nearly reproduces the best earlier multi-feature upstream result.
3. Factor distillation is therefore a cleaner representation than continuing to grow raw quality families.
4. But even the best distilled factor route still does **not** beat hard abstention.

### What that means

The current transfer bottleneck is probably **not**:

- lack of another small calibration layer,
- lack of another router tweak,
- or lack of a better way to recombine the same compact proxies.

Instead, the current evidence suggests:

## if we stay with the present upstream proxy pool, one distilled responsiveness factor is enough.

Further progress probably needs a **genuinely stronger transferable signal source**, not a larger pile of similar proxies.

## Practical Consequence

This is exactly the kind of result we wanted at this stage of research:

- it reduces method sprawl,
- it keeps interpretability,
- it improves the internal discipline of the transfer story,
- and it tells us we do **not** need to keep expanding upstream feature families built from the same ingredients.

So the current best reading is:

- **hard abstention remains the strongest compact transfer route**,
- but if we revisit the existing upstream quality block, the cleanest representation is now a **single learnable responsiveness factor**, not another wider handcrafted family.
