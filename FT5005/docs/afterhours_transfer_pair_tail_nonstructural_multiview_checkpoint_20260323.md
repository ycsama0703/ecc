# After-Hours Transfer Pair-Tail Non-Structural Multiview Checkpoint 2026-03-23

## Purpose

The non-structural encoding checkpoint showed something surprisingly clean:

- if we remove structural-probe vocabulary from the hardest question,
- the strongest local route stays **exactly unchanged**

That leaves a natural next question:

## if the useful tail is really non-structural and operational / quantitative,
## can a slightly richer but still compact **local multiview** representation help?

The most conservative way to test that is:

- keep the same local top-1 hardest pair
- keep structural masking
- but allow a small second view from:
  - the hardest answer
  - or the hardest `Q&A` pair text

This is still much more disciplined than adding a new shell or router.

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_nonstructural_multiview_benchmark.py`

Inputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`
- `results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv`
- `results/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_real/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_test_predictions.csv`

Views used after structural masking:

- hardest question
- hardest answer
- hardest top-1 `Q&A` pair text

Families compared:

- `question_mask_struct_lsa4_bi`
- `answer_mask_struct_lsa4_bi`
- `qa_mask_struct_lsa4_bi`
- `question_plus_answer_mask_struct_lsa4_bi`
- `question_plus_qa_mask_struct_lsa4_bi`

All stay in the same compact agreement-side transfer protocol with `LSA(4)` bi-gram encodings.

Outputs:

- `results/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_real/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_summary.json`
- `results/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_real/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_family_overview.csv`
- `results/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_real/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_component_terms.csv`
- `results/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_real/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_test_predictions.csv`

## Main Findings

### 1. The single hardest-question view still wins

Held-out latest window:

- `question_mask_struct_lsa4_bi ≈ 0.99865468`
- `qa_mask_struct_lsa4_bi ≈ 0.99855719`
- `answer_mask_struct_lsa4_bi ≈ 0.99855053`
- `question_plus_qa_mask_struct_lsa4_bi ≈ 0.99852148`
- `question_plus_answer_mask_struct_lsa4_bi ≈ 0.99850934`

So the main result is very simple:

## even after structural masking,
## the hardest-question view remains the cleanest local object

### 2. Adding answer or pair text hurts rather than helps

This is the key negative result:

- the answer view alone is worse
- the `Q&A` pair view alone is worse
- concatenating them with the strongest question view is **also** worse

So the hoped-for “slightly richer multiview local representation” does not materialize here.

Instead:

## the extra local views reintroduce noise faster than they add useful recall

### 3. The best multiview route does not even match hard abstention's edge

Best route still remains:

- `question_mask_struct_lsa4_bi`
  - `p(MSE)` vs hard `≈ 0.04425`

But all multiview combinations fall back below that:

- `question_plus_qa_mask_struct_lsa4_bi`
  - `≈ 0.99852148`
- `question_plus_answer_mask_struct_lsa4_bi`
  - `≈ 0.99850934`

So the single-view non-structural hardest question remains the only local route in this family that still preserves the small held-out edge over hard abstention.

### 4. This strengthens the interpretation of the local signal

Taken together with the last two checkpoints:

1. the useful hardest-question signal is already non-structural
2. it does not need structural-probe vocabulary
3. it also does not benefit from pulling in the paired answer or top-1 `Q&A` text

That means the current local signal is even more specific than before:

## it lives in the wording of the hardest analyst question itself,
## not in a broader local pair-text fusion

## Updated Interpretation

This is a valuable narrowing checkpoint.

### What we now know

1. The strongest local route remains the masked hardest-question view.
2. The answer and paired-`Q&A` views do not help, even in a compact multiview setup.
3. The current useful local signal is therefore best understood as:
   - non-structural
   - question-centric
   - compact

### What that means

The next step should not be broader local fusion.
If we keep pushing this line, the right direction is more likely:

- a **better compact representation of the hardest question itself**

rather than:

- adding answer-side or pair-side local views

So this checkpoint keeps the story disciplined:

- the retained local transfer signal is narrower than we hoped,
- but also cleaner and better defined than before.
