# After-Hours Transfer Pair-Tail Question Non-Structural Encoding Checkpoint 2026-03-22

## Purpose

The signature-taxonomy checkpoint showed a very clear split:

- the compact shared core is entirely `clarify+quant`
- the broader semantic tail contains both:
  - a useful operational / quantitative branch
  - and a noisier structural / strategic branch

That naturally suggests a very compact next test:

## if we explicitly remove structural-probe vocabulary from the hardest question,
## can we keep the useful local semantic lift while suppressing the noisier branch?

This is a clean follow-up because it:

- keeps the same local-object focus (`tail_top1_question_text`)
- keeps the same compact LSA-style representation
- does **not** add a new shell, expert, or router

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark.py`

Inputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`
- `results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv`
- `results/afterhours_transfer_pair_tail_question_encoding_benchmark_real/afterhours_transfer_pair_tail_question_encoding_benchmark_test_predictions.csv`

The benchmark:

1. starts from the hardest-question text
2. removes only the existing `structural_probe` pattern family
   - `platform`
   - `assets`
   - `content`
   - `sports`
   - `portfolio`
   - `services`
   - `distribution`
   - `channel`
3. re-runs compact question encodings on the masked text:
   - `question_mask_struct_lsa4_bi`
   - `question_mask_struct_lsa8_bi`
   - `question_mask_struct_lsa4_uni`
4. also checks geometry-plus versions to confirm whether the old shell changes the result

Outputs:

- `results/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_real/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_summary.json`
- `results/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_real/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_family_overview.csv`
- `results/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_real/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_component_terms.csv`
- `results/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_real/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_test_predictions.csv`

## Main Findings

### 1. Structural masking preserves the best hardest-question route exactly

Best family:

- `question_mask_struct_lsa4_bi`

Held-out latest window:

- `question_mask_struct_lsa4_bi ≈ 0.99865468`
- `question_lsa4_bi ≈ 0.99865468`
- `hard_abstention ≈ 0.99864017`
- `geometry_only ≈ 0.99863878`

And relative to the existing raw hardest-question route:

- `p(MSE)` vs `question_lsa4_bi = 1.0`
- all pooled metric differences vs `question_lsa4_bi` are exactly `0`

So the most important result of this checkpoint is:

## removing structural-probe vocabulary does **not** reduce the best local hardest-question signal at all

### 2. This is a useful confirmation, even though it is not an improvement

The masked route does **not** beat the raw route.
But that is not the interesting part.

The interesting part is:

## the useful held-out lift survives intact after structural masking

That means the current best hardest-question route is already getting its value from the **non-structural** part of the question.

This directly supports the previous taxonomy result.

### 3. Bigger or looser masked encodings are worse

Other masked variants:

- `question_mask_struct_lsa8_bi ≈ 0.99857873`
- `question_mask_struct_lsa4_uni ≈ 0.99847473`

So this is not a case where structural masking unlocks a need for a larger or more flexible encoding.

The compact answer remains:

## if we stay in this non-structural local-question regime,
## the same compact `LSA(4)` bi-gram representation is still the right scale

### 4. Geometry still does not help

Geometry-plus variants remain lower:

- `geometry_plus_question_mask_struct_lsa8_bi ≈ 0.99863867`
- `geometry_plus_question_mask_struct_lsa4_bi ≈ 0.99863851`
- `geometry_plus_question_mask_struct_lsa4_uni ≈ 0.99863790`

So this result does not revive the geometry shell.
It reinforces the same pattern we have seen several times:

- the useful signal is living in the compact local question representation itself
- not in adding more control-shell complexity around it

### 5. The current hardest-question signal is therefore robustly non-structural

Taken together with the taxonomy checkpoint:

- the shared compact core is entirely `clarify+quant`
- the raw strongest route survives intact after masking structural-probe vocabulary

This is a strong research confirmation that:

## the retained upside in the hardest-question route is already coming from
## clarificatory / operational / quantitative wording,
## not from structural / strategic probe language

## Updated Interpretation

This is a small but very clean confirmation checkpoint.

### What we now know

1. The best hardest-question route does **not** need structural-probe vocabulary.
2. The useful local signal survives exactly after structural masking.
3. Larger masked encodings are worse, so the current compact scale still looks right.
4. Geometry still does not add value on top of this signal.

### What that means

The signature-taxonomy story now gets a stronger causal reading:

- the compact shared core is `clarify+quant`
- the useful tail beyond it is still operational / quantitative
- the structural / strategic branch is not carrying the held-out value of the best hardest-question route

So the next step should not be “add back more structure.”
The more interesting direction is:

## can we build a slightly richer but still compact **non-structural** local representation
## that extends recall into the operational / quantitative sem-only tail,
## while continuing to avoid the structural / strategic branch?
