# After-Hours Transfer Role-Semantic Expert Checkpoint 2026-03-20

## Purpose

After the role-text signal benchmark, the next research question was:

## can the stronger standalone `question_role_lsa` signal replace pooled `qna_lsa` as the semantic core of the matched transfer expert?

This is the right follow-up because the previous checkpoint showed:

- `question_role_lsa` is the strongest standalone compact transfer signal currently in the repo,
- but it did **not** improve the geometry-led controller.

So the remaining clean test is:

## maybe question-role semantics are not a better controller, but they could still be a better **expert core**

## Design

New script:

- `scripts/run_afterhours_transfer_role_semantic_expert_benchmark.py`

Inputs:

- clean aligned `after_hours` panel:
  - `results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv`
- event-level text features:
  - `results/features_real/event_text_audio_features.csv`
- old audio view:
  - `results/audio_real/event_real_audio_features.csv`
- role-aware aligned audio view:
  - `results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv`
- QA benchmark table:
  - `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`
- hard-abstention reference:
  - `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/...`

Coverage is complete on the matched transfer subset:

- `172 / 172` rows have all side inputs

Split protocol stays the same as the existing matched unseen-ticker transfer shell:

- train on non-held-out tickers through `2021`
- validate on non-held-out tickers in `2022`
- test on held-out tickers after `2022`

### Experts compared

1. `residual_pre_call_market_only`
2. `pre_call_market + A4 + qna_lsa`
3. `pre_call_market + A4 + question_role_lsa`
4. `pre_call_market + A4 + answer_role_lsa`
5. `pre_call_market + A4 + qna_lsa + aligned_audio_svd`
6. `pre_call_market + A4 + question_role_lsa + aligned_audio_svd`

All non-pre models use the same simple observability gate on `a4_strict_row_share`.

To keep the integration disciplined, the only validation-selected route is:

- `validation_selected_role_semantic_core`

selected from just:

- `pre_call_market_only`
- `qna_lsa + aligned_audio`
- `question_role_lsa + aligned_audio`

Outputs:

- `results/afterhours_transfer_role_semantic_expert_benchmark_role_aware_audio_real/afterhours_transfer_role_semantic_expert_benchmark_summary.json`
- `results/afterhours_transfer_role_semantic_expert_benchmark_role_aware_audio_real/afterhours_transfer_role_semantic_expert_benchmark_predictions.csv`
- `results/afterhours_transfer_role_semantic_expert_benchmark_role_aware_audio_real/afterhours_transfer_role_semantic_expert_benchmark_selection.csv`

## Main Findings

### 1. Question-role semantics do not improve the matched transfer expert when used as a direct semantic replacement

Held-out overall metrics:

- `pre_call_market_only ≈ 0.998482062`
- `A4 + qna_lsa ≈ 0.997706362`
- `A4 + question_role_lsa ≈ 0.997562102`
- `A4 + answer_role_lsa ≈ 0.997589376`
- `A4 + qna_lsa + aligned_audio ≈ 0.997785330`
- `A4 + question_role_lsa + aligned_audio ≈ 0.997581610`
- `hard abstention ≈ 0.998640168`

So the main result is negative but clear:

## question-role semantics are stronger as a standalone compact signal than as a direct replacement expert core

This is the important distinction.

### 2. The existing pooled-Q&A semantic core still beats the role-specific replacement inside the expert shell

Direct paired comparisons:

- `question_role_lsa` expert vs `qna_lsa` expert:
  - overall direction is worse
  - `p(MSE) ≈ 0.481`
- `question_role_lsa + aligned_audio` vs `qna_lsa + aligned_audio`:
  - overall direction is also worse
  - `p(MSE) ≈ 0.469`

So while `question_role_lsa` was the best standalone route in the previous benchmark, that advantage does **not** survive once it is forced into the current direct transfer-expert shell.

### 3. Conservative selection over the two semantic cores still fails

Validation-selected counts:

- `qna_lsa + aligned_audio`: `4`
- `question_role_lsa + aligned_audio`: `3`
- `pre_call_market_only`: `2`

Held-out selected result:

- **`validation_selected_role_semantic_core ≈ 0.997569753`**

That is below:

- `pre_call_market_only ≈ 0.998482062`
- `qna_lsa + aligned_audio ≈ 0.997785330`
- `hard abstention ≈ 0.998640168`

And against hard abstention the result is clearly unfavorable:

- `p(MSE) ≈ 0.00175`

So even the more disciplined two-core selection still overfits validation and does not generalize.

### 4. The negative result is still scientifically useful

This checkpoint resolves an important ambiguity from the previous role-text result.

We now know that:

- `question_role_lsa` is a stronger **standalone compact signal**
- but not a stronger **semantic expert core**

That means the current transfer hierarchy should stay:

- hard abstention as the strongest method,
- pooled-Q&A semantic expert as the better direct expert shell,
- question-role text as a promising complementary signal source that still needs a different integration strategy.

## Updated Interpretation

This checkpoint narrows the research picture further.

### What we now know

1. Low-rank analyst-question semantics are real and stronger than earlier handcrafted standalone factors.
2. But they do not replace pooled `qna_lsa` inside the current transfer expert shell.
3. Conservative validation selection over semantic cores still overfits and degrades late-window held-out performance.
4. So the role-text result should remain a **standalone complementary-signal finding**, not a justification for swapping out the current semantic expert.

### What that means

The next useful research move is not:

- more validation selection over semantic cores,
- nor simply replacing pooled `qna_lsa` with question-only text.

The next useful move would have to be a more principled way to:

- preserve the standalone transferability of analyst-question semantics,
- without destabilizing the current expert shell or weakening hard abstention.
