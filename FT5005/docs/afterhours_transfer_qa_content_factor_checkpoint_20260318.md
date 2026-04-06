# After-Hours Transfer QA-Content Factor Checkpoint 2026-03-18

## Purpose

After the responsiveness-factor checkpoint, the next research question was:

## do the **richer QA benchmark content features** contain a stronger transferable signal than the earlier responsiveness proxy block?

This is a meaningful next step because the earlier agreement-signal and responsiveness-factor checkpoints mainly worked with:

- directness / evasion / coverage style proxies,
- observability proxies,
- and geometry around the existing experts.

But the benchmark QA feature table also contains a second, underused family of features that are more content-grounded:

- specificity,
- overlap / fidelity,
- topic drift,
- numeric mismatch,
- justification,
- attribution,
- subjectivity,
- and question complexity.

So this checkpoint asks whether a **new upstream semantic-content source** can help the transfer story more than recombining the old proxy set.

## Design

New script:

- `scripts/run_afterhours_transfer_qa_content_factor_benchmark.py`

Inputs:

- temporal router outputs:
  - `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`
- panel observability inputs:
  - `results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv`
- full QA benchmark feature table:
  - `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`
- references:
  - `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_agreement_signal_benchmark_test_predictions.csv`
  - `results/afterhours_transfer_responsiveness_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_responsiveness_factor_benchmark_test_predictions.csv`

Coverage is complete:

- `257 / 257` temporal rows have both panel and full QA benchmark inputs

The shell stays fixed:

- disagreement still falls back to `pre_call_market_only`
- only agreement events are refined

So the only change is the upstream latent source.

### Factor families

1. `specificity_fidelity_core`
   - answer specificity
   - question specificity
   - specificity gap
   - early overlap
   - late overlap share
   - topic drift share
   - restatement share
   - numeric mismatch share
2. `commitment_attribution_core`
   - internal action rate
   - external attribution rate
   - justification rate
   - subjective rate
   - certainty rate
   - forward rate
   - hedge rate
   - delay share
3. `content_accountability_core`
   - question complexity
   - numeric rate
   - numeric mismatch share
   - justification rate
   - external attribution rate
   - subjective rate
4. `specificity_directness_hybrid`
   - specificity / fidelity terms
   - direct-early score
   - evasion score
   - coverage
5. `specificity_plus_observability`
   - specificity / fidelity terms
   - `a4_strict_row_share`

### Factor builders and route variants

Each family is compressed into one latent with:

- `pca1`
- `pls1`

and then evaluated as:

- `factor_only`
- `geometry_plus_factor`

Temporal protocol stays the same:

- train on `2020` agreement events
- tune on `2021` agreement events
- refit on `2020 + 2021`
- test on full `2022`

Outputs:

- `results/afterhours_transfer_qa_content_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_qa_content_factor_benchmark_summary.json`
- `results/afterhours_transfer_qa_content_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_qa_content_factor_benchmark_overview.csv`
- `results/afterhours_transfer_qa_content_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_qa_content_factor_benchmark_tuning.csv`
- `results/afterhours_transfer_qa_content_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_qa_content_factor_benchmark_factor_loadings.csv`
- `results/afterhours_transfer_qa_content_factor_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_qa_content_factor_benchmark_test_predictions.csv`

## Main Findings

### 1. The richer QA content source gives the strongest standalone factor-only route so far

Best `factor_only` route:

- **`content_accountability_core__pca1__factor_only ≈ 0.998592197`**

Comparison points:

- `pre_call_market_only ≈ 0.998482062`
- previous best distilled responsiveness `factor_only ≈ 0.998558665`

So this richer QA-content family gives the **highest factor-only compact route currently seen in the repo**.

That is important because it shows the content/accountability block is not redundant with the earlier responsiveness block.

But we should stay honest:

- vs `pre_call_market_only`: `p(MSE) ≈ 0.229`
- vs previous best responsiveness `factor_only`: `p(MSE) ≈ 0.355`

So the improvement is real numerically, but not yet statistically firm.

### 2. Once geometry is available, the new content factor does not add incremental value

Best overall route in this checkpoint:

- **`content_accountability_core__pca1__geometry_plus_factor ≈ 0.998638784`**

References:

- `geometry_only ≈ 0.998638784`
- previous best responsiveness `geometry_plus_factor ≈ 0.998639314`
- `geometry_plus_hybrid ≈ 0.998638845`
- **`hard abstention ≈ 0.998640168`**

So the content factor does **not** improve the main geometry-led transfer route.

In fact, the best content geometry-plus-factor route lands exactly at the current `geometry_only` score and remains below the earlier responsiveness geometry-plus-factor route.

Paired comparison:

- best content `geometry_plus_factor` vs `geometry_only`: `p(MSE) = 1.0`
- best content `geometry_plus_factor` vs previous best responsiveness `geometry_plus_factor`: `p(MSE) ≈ 0.751`
- best content `geometry_plus_factor` vs hard abstention: `p(MSE) ≈ 0.450`

So this new signal source helps most as a **standalone compact factor**, not as a stronger trust-modulation layer.

### 3. The best standalone latent is interpretable and qualitatively different from the responsiveness factor

Top loadings for `content_accountability_core__pca1__factor_only`:

- `qa_bench_question_complexity_mean` (negative)
- `qa_bench_justification_rate_mean` (negative)
- `qa_bench_subjective_rate_mean` (positive)
- `qa_bench_external_attr_rate_mean` (positive)
- `qa_bench_numeric_rate_mean` (positive)
- `qa_bench_numeric_mismatch_share` (small positive)

This is useful because it suggests the richer QA content table captures a different axis than the earlier responsiveness latent.

The earlier factor was mostly about:

- directness,
- evasion,
- coverage,
- forwardness.

This new factor is more about:

- content accountability,
- justification vs attribution,
- subjective framing,
- numeric texture,
- and question complexity.

So the repo now has evidence for **two distinct low-dimensional QA axes**, not just one.

### 4. The new factor behaves more like a compressed complementary expert than a better gate

This is probably the most important scientific reading.

The new QA-content factor:

- strengthens the standalone `factor_only` route,
- but does not improve the geometry-led agreement route.

That pattern is different from what we would expect if the factor were primarily a better **trust signal**.

Instead, it looks more like:

## a compact complementary expert signal

that can sometimes help on its own,
but is not the missing ingredient for improving the current abstention shell.

## Updated Interpretation

This checkpoint is a helpful refinement of the transfer story.

### What we now know

1. The full QA benchmark table contains a richer semantic-content source than the earlier responsiveness proxy block alone.
2. That richer source improves the strongest `factor_only` compact route numerically.
3. But it still does not improve the main geometry-led route and does not beat hard abstention.
4. So richer QA content is scientifically useful, but not yet the decisive upstream signal for the current transfer method.

### What that means

The current transfer bottleneck is still not best described as:

- “we forgot one more handcrafted content feature”,
- or “we only need a slightly richer agreement latent”.

Instead, the evidence now suggests a more precise split:

- **responsiveness-style latents** are the cleaner trust / agreement-side representation,
- **content-accountability latents** are the cleaner standalone complementary signal,
- but neither currently breaks the hard-abstention ceiling.

## Practical Consequence

This is another good research narrowing step.

It says we should not just keep widening the same upstream block.

Instead, we now have a clearer map:

- keep the transfer headline with **hard abstention**,
- keep a distilled responsiveness factor as the cleanest summary of the old proxy block,
- and treat richer QA-content factors as promising **complementary expert-style** signals rather than as the next main gating story.
