# After-Hours Role-Semantic Mainline Checkpoint 2026-03-20

## Purpose

After the recent transfer-side role-text results, the next clean research question was:

## if analyst-question text is the strongest standalone compact transfer signal, does it also strengthen the fixed-split clean `after_hours` headline line?

This matters because the main paper-safe contribution is still the fixed-split semantic story:

- clean `after_hours`
- `shock_minus_pre`
- `A4` observability / alignment
- compact pooled `Q&A` semantics

So before we push role-specific text any further on the transfer side, we should test whether it actually improves the mainline itself.

## Design

New script:

- `scripts/run_afterhours_role_semantic_mainline_benchmark.py`

Inputs:

- clean aligned `after_hours` panel:
  - `results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv`
- event-level text features:
  - `results/features_real/event_text_audio_features.csv`
- old audio feature table (used only for the joined row builder):
  - `results/audio_real/event_real_audio_features.csv`
- QA benchmark table:
  - `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`

Coverage is complete on the fixed-split clean `after_hours` subset:

- split sizes: train `89`, val `23`, test `60`
- `172 / 172` rows have pooled `Q&A` text, `question_text`, and `answer_text`

Role-specific text is built using one shared `TF-IDF + LSA(8)` basis over question and answer text.

The benchmark compares three kinds of role-semantic use:

1. **direct replacement**
   - replace pooled `qna_lsa` with `question_role_lsa` or `answer_role_lsa`
2. **raw stacking**
   - add question-role or answer-role semantics on top of pooled `qna_lsa`
3. **orthogonalized stacking**
   - residualize question-role or answer-role semantics against pooled `qna_lsa`
   - then add only the role-specific residual signal

The residualization step is intentionally compact and learnable:

- fit a multi-output ridge from pooled `qna_lsa` to role text
- tune the ridge on validation reconstruction MSE
- keep only the residual role-specific component

Outputs:

- `results/afterhours_role_semantic_mainline_benchmark_clean_real/afterhours_role_semantic_mainline_benchmark_summary.json`
- `results/afterhours_role_semantic_mainline_benchmark_clean_real/afterhours_role_semantic_mainline_benchmark_overview.csv`
- `results/afterhours_role_semantic_mainline_benchmark_clean_real/afterhours_role_semantic_mainline_benchmark_predictions.csv`
- `results/afterhours_role_semantic_mainline_benchmark_clean_real/afterhours_role_semantic_mainline_component_terms.csv`

## Main Findings

### 1. Pooled `Q&A` semantics still dominate the fixed-split semantic headline

Core clean `after_hours` results on the held-out test split:

- `prior_only ≈ 0.0351`
- `pre_call_market_only ≈ 0.9174`
- `pre_call_market + A4 + qna_lsa ≈ 0.9271`
- `pre_call_market + controls + A4 + qna_lsa ≈ 0.9347`

These are exactly the current mainline reference points, and they remain the strongest compact semantic results in this checkpoint.

### 2. Question-role and answer-role text are both worse as direct replacements

Direct replacement results:

- `pre_call_market + A4 + question_role_lsa ≈ 0.8997`
- `pre_call_market + A4 + answer_role_lsa ≈ 0.9142`
- `pre_call_market + controls + A4 + question_role_lsa ≈ 0.9082`
- `pre_call_market + controls + A4 + answer_role_lsa ≈ 0.9099`

So the transfer-side `question_role_lsa` strength does **not** carry over as a better fixed-split semantic core.

## role-specific text is not a replacement for pooled `Q&A` semantics on the mainline

### 3. Raw question-role stacking also hurts the mainline

Stacking question-role text on top of pooled `qna_lsa` is consistently worse than leaving the pooled semantic block alone:

- `pre_call_market + A4 + qna_lsa ≈ 0.9271`
- `+ question_role_lsa ≈ 0.9220`
  - `p(MSE) ≈ 0.00325`
- `pre_call_market + controls + A4 + qna_lsa ≈ 0.9347`
- `+ question_role_lsa ≈ 0.9294`
  - `p(MSE) ≈ 0.00425`

So even when we keep the mainline pooled semantic block intact, adding question-role text degrades the fixed-split headline.

### 4. Answer-role stacking is closer to neutral, but still not an upgrade

Answer-role stacking is less damaging than question-role stacking, but still does not produce a meaningful gain:

- `pre_call_market + A4 + qna_lsa + answer_role_lsa ≈ 0.9267`
  - vs pooled `qna_lsa` baseline: `p(MSE) ≈ 0.845`
- `pre_call_market + controls + A4 + qna_lsa + answer_role_lsa ≈ 0.9337`
  - vs pooled `qna_lsa` baseline: `p(MSE) ≈ 0.69175`

So answer-role text is not clearly harmful, but it is also not an additive win.

### 5. Orthogonalizing question-role text against pooled `Q&A` does not rescue it

This was the most important follow-up in the benchmark.

If raw question-role stacking hurts because it is redundant with pooled `qna_lsa`, then maybe the unique analyst-question component should help once we explicitly remove the pooled `Q&A` part.

That does **not** happen.

Residualized role-semantic results:

- `pre_call_market + A4 + qna_lsa + question_role_resid ≈ 0.9049`
  - `p(MSE) ≈ 0.0355`
- `pre_call_market + controls + A4 + qna_lsa + question_role_resid ≈ 0.8877`
  - `p(MSE) ≈ 0.016`

Residualized answer-role is also worse:

- `pre_call_market + A4 + qna_lsa + answer_role_resid ≈ 0.8952`
- `pre_call_market + controls + A4 + qna_lsa + answer_role_resid ≈ 0.9034`

This gives a sharper result than the raw stacking benchmark alone:

## the role-specific component that is unique beyond pooled `Q&A` is not helping the fixed-split headline either

### 6. The residualization metadata is still scientifically informative

Best residualization ridge fits:

- question-role from pooled `qna_lsa`:
  - best alpha `= 0.01`
  - validation reconstruction MSE `≈ 0.2404`
- answer-role from pooled `qna_lsa`:
  - best alpha `= 0.01`
  - validation reconstruction MSE `≈ 0.0378`

This means:

- analyst-question text is much less reconstructable from pooled `Q&A` than answer text is
- so there really is a distinct analyst-question semantic axis here
- but that unique axis still does **not** improve the clean fixed-split mainline target

That is an important nuance:

## question-role semantics are distinct, but not mainline-additive

## Updated Interpretation

This checkpoint materially sharpens the overall story.

### What we now know

1. Pooled `qna_lsa` remains the best compact semantic core for the clean fixed-split `after_hours` headline.
2. Question-role text is not a better semantic replacement on the mainline.
3. Raw question-role stacking hurts the mainline even when pooled `qna_lsa` is retained.
4. Orthogonalized question-role residuals hurt even more, so the unique analyst-question component is not the missing fixed-split gain.
5. Answer-role text is closer to neutral, but still not a convincing mainline upgrade.

### What that means

The recent role-text finding should now be interpreted much more narrowly:

- `question_role_lsa` remains a **strong standalone compact transfer signal**
- but it is **not** a better fixed-split semantic core
- and it is **not** a reliable additive upgrade to the fixed-split headline, even after careful orthogonalization

So the project story is now cleaner:

- pooled `Q&A` semantics stay in the main fixed-split headline
- analyst-question semantics stay as a complementary research signal, mostly relevant for transfer-side interpretation rather than for replacing the core paper-safe semantic block
