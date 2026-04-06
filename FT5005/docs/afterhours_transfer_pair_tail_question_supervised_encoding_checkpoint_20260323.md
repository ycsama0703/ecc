# After-Hours Transfer Pair-Tail Question Supervised Encoding Checkpoint 2026-03-23

## Purpose

After the non-structural multiview benchmark, the story became even narrower:

- the useful local signal is non-structural
- it is question-centric
- and adding more local views does not help

The next clean research question was therefore:

## if we stay on the hardest question itself,
## can a small **learnable supervised text subspace** beat the current unsupervised `LSA(4)` route?

This is a reasonable next test because it is:

- compact
- learnable
- still tied to the same local object
- and does not add any new shell or routing logic

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_supervised_encoding_benchmark.py`

Inputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`
- `results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv`
- `results/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_real/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_test_predictions.csv`

The benchmark:

1. keeps the same structurally masked hardest-question text
2. builds TF-IDF bi-gram features
3. learns a compact supervised text subspace with PLS:
   - `question_mask_struct_pls1_bi`
   - `question_mask_struct_pls2_bi`
   - `question_mask_struct_pls4_bi`
4. uses the same agreement-side gain-prediction protocol on top

Outputs:

- `results/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_real/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_summary.json`
- `results/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_real/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_family_overview.csv`
- `results/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_real/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_component_terms.csv`
- `results/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_real/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_test_predictions.csv`

## Main Findings

### 1. The supervised question encodings do not beat the current unsupervised route

Held-out latest window:

- `question_mask_struct_pls4_bi ≈ 0.99858590`
- `question_mask_struct_pls2_bi ≈ 0.99856058`
- `question_mask_struct_pls1_bi ≈ 0.99855619`

References:

- `question_mask_struct_lsa4_bi ≈ 0.99865468`
- `question_lsa4_bi ≈ 0.99865468`
- `hard_abstention ≈ 0.99864017`

So the clean result is:

## the current compact unsupervised hardest-question representation still wins

### 2. Even the best supervised variant falls below hard abstention

Best supervised route:

- `question_mask_struct_pls4_bi`

But relative to hard abstention:

- `p(MSE) ≈ 0.5665`
- mean MSE gain is negative

Relative to the current best masked question route:

- `p(MSE) ≈ 0.305`
- mean MSE gain is again negative

So this is not “close but slightly noisier.”
It is a real negative result.

### 3. The supervised text factors look more idiosyncratic than the stronger unsupervised route

The leading PLS terms are things like:

- `medical and`
- `about linearity`
- `retail margin`
- `mean why`
- `picked medical`

These are much less stable-looking than the broader clarificatory / quantitative patterns that survived the earlier checkpoints.

This suggests:

## at the current sample size,
## supervised text compression is drifting toward small train-window-specific phrase pockets

instead of recovering a cleaner transferable local question axis.

### 4. This is a useful methodological boundary

The current hardest-question line has now been tested in three increasingly “learnable” directions:

1. compact lexical factors
2. non-structural masking
3. supervised PLS text subspaces

The first two helped explain and stabilize the signal.
This third one did **not**.

That is valuable, because it shows the next step should not be:

- “make the question representation more supervised”

at least not at this small matched-sample scale.

## Updated Interpretation

This checkpoint strengthens the current story by ruling out another tempting path.

### What we now know

1. The strongest local route remains the compact unsupervised masked hardest-question representation.
2. A small supervised question subspace does not improve it.
3. The supervised variants appear to overfit to idiosyncratic phrase pockets rather than learning a more transferable local question axis.

### What that means

The current best reading is:

- the useful local question signal is real
- it is compact
- it is non-structural
- but at this scale it is better captured by a **stable unsupervised representation** than by a more aggressively supervised compact text encoder

So the next step should not be more supervision on the same tiny question object.
If we continue this line, the more promising direction is likely:

- either a better unsupervised / weakly structured question representation
- or a broader data regime for learning the supervised version safely
