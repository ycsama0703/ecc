# After-Hours Transfer Factor-Expert Integration Checkpoint 2026-03-18

## Purpose

After the QA-content-factor checkpoint, the next research question was:

## can the new distilled factor signals work as a **minimal complementary expert**, rather than only as standalone compact latents?

This is the cleanest follow-up because the last two checkpoints suggested a natural split:

- the **responsiveness** latent is the cleaner trust-style summary of the earlier directness / coverage / evasion proxy block,
- the **content-accountability** latent is the strongest standalone compact factor seen so far,
- but neither improved the current hard-abstention shell once geometry was available.

So instead of adding another routing layer or another wide feature family, this checkpoint asks a tighter question:

## if we build the smallest possible direct factor experts, do they help as transfer experts?

## Design

New script:

- `scripts/run_afterhours_transfer_factor_expert_integration.py`

Inputs:

- clean aligned `after_hours` panel:
  - `results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv`
- event-level side features:
  - `results/features_real/event_text_audio_features.csv`
- QA benchmark table:
  - `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`
- role-aware aligned-audio event table, used to keep the matched transfer subset identical to the current transfer shell:
  - `results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv`
- hard-abstention reference predictions:
  - `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_agreement_signal_benchmark_test_predictions.csv`
  - `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_agreement_signal_benchmark_summary.json`

Coverage is complete on this matched subset:

- `172 / 172` clean aligned `after_hours` rows have all side inputs

The benchmark stays intentionally compact.

### Distilled experts

1. `residual_pre_call_market_only`
2. `residual_pre_call_market_plus_a4_plus_responsiveness_factor_observability_gate`
3. `residual_pre_call_market_plus_a4_plus_content_accountability_factor_observability_gate`

The two factor experts use:

- `pre_call_market`
- `A4`
- one one-dimensional distilled factor
- the same simple observability gate on `a4_strict_row_share`

The integration layer is deliberately minimal:

- per held-out ticker, choose the best expert on validation `R²`
- then score that selected expert on the held-out latest-window test

This yields:

- `validation_selected_compact_factor_expert`

Outputs:

- `results/afterhours_transfer_factor_expert_integration_role_aware_audio_real/afterhours_transfer_factor_expert_integration_summary.json`
- `results/afterhours_transfer_factor_expert_integration_role_aware_audio_real/afterhours_transfer_factor_expert_integration_predictions.csv`
- `results/afterhours_transfer_factor_expert_integration_role_aware_audio_real/afterhours_transfer_factor_expert_integration_selection.csv`
- `results/afterhours_transfer_factor_expert_integration_role_aware_audio_real/afterhours_transfer_factor_expert_integration_factor_loadings.csv`

## Main Findings

### 1. The direct distilled factor experts are materially weaker than both the market baseline and hard abstention

Held-out overall metrics:

- `residual_pre_call_market_only ≈ 0.998482062`
- `responsiveness factor expert ≈ 0.997590632`
- `content-accountability factor expert ≈ 0.997559927`
- `hard abstention ≈ 0.998640168`

So turning the distilled factors into direct compact experts is clearly not enough.

Both factor experts land well below:

- the simple market baseline,
- and the current best transfer-shell route.

This is useful because it says the distilled factors are not automatically strong just because they are compact and interpretable.

### 2. Validation-selected factor-expert integration still fails on the held-out latest window

The validation-selected route reaches:

- **`validation_selected_compact_factor_expert ≈ 0.997580267`**

That is below:

- `pre_call_market_only ≈ 0.998482062`
- `hard abstention ≈ 0.998640168`

Directional paired comparisons are also unfavorable:

- selected vs `pre_call_market_only`: `p(MSE) ≈ 0.0895`
- selected vs hard abstention: `p(MSE) ≈ 0.0608`

So while the gap is not conventionally significant in this small matched transfer slice, it is consistently in the wrong direction.

### 3. The failure mode is classic validation over-selection

Selection counts:

- responsiveness factor expert: `5`
- content-accountability factor expert: `4`
- `pre_call_market_only`: `0`

So the validation layer chooses a distilled factor expert for **every** held-out ticker.

But on held-out test, the selected route only beats `pre_call_market_only` on a minority of names:

- clear wins: `AMZN`, `MSFT`, `NKE`
- clear losses: `AAPL`, `AMGN`, `CSCO`, `DIS`, `IBM`, `NVDA`

This is the important scientific pattern:

## the distilled factor experts look attractive on validation, but they do not transfer robustly to the later unseen-ticker test window

That is exactly the kind of result we want to surface honestly, because it stops us from turning a neat latent into an unjustified new module.

### 4. The two distilled factors remain scientifically useful, but not yet as direct transfer experts

This checkpoint does **not** erase the value of the earlier factor work.

We still learned that:

- there is a coherent **responsiveness / directness / coverage / evasion** axis,
- and a distinct **content-accountability / attribution / numeric texture** axis.

But this benchmark shows a sharper limit:

## those axes currently work better as explanatory compact summaries than as direct complementary experts

That is a healthy narrowing result.

## Updated Interpretation

This checkpoint clarifies the transfer story in a useful way.

### What we now know

1. The new distilled latents are real and interpretable.
2. But converting them into minimal direct experts does not improve transfer.
3. Validation-selected factor-expert integration overfits the late-window validation slice rather than generalizing.
4. So the repo still does **not** support promoting these distilled factors into the main transfer shell as standalone experts.

### What that means

The project should keep the current hierarchy:

- **hard abstention** remains the strongest compact transfer-side route,
- the **responsiveness** factor remains the cleanest trust-style explanatory latent,
- the **content-accountability** factor remains the strongest standalone complementary compact signal,
- but neither should yet be elevated into a new main expert branch.

So the next real gain still has to come from:

- a stronger upstream transferable signal source,
- not from another direct factor-expert integration trick.
