# After-Hours Transfer Pair-Tail Question-Encoding Checkpoint 2026-03-22

## Purpose

The pair-tail text checkpoint produced the first small held-out latest-window win over hard abstention:

- `tail_question_top1_lsa ≈ 0.99865468`
- `hard_abstention ≈ 0.99864017`

That made the next question very important:

## is this new gain actually a robust local-question signal, or is it just an artifact of one particular LSA encoding?

This is exactly the kind of follow-up we want:

- no new shell complexity,
- no broader module pile-up,
- just a confirmation benchmark on the same underlying object:
  - the top-1 hardest analyst question.

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_encoding_benchmark.py`

The benchmark fixes the same transfer protocol:

- train `val2020_test_post2020`
- validate `val2021_test_post2021`
- test `val2022_test_post2022`
- disagreement rows still fallback to `pre_call_market_only`
- only agreement rows are learnably refined

It then compares compact encoding variants of the same local text field `tail_top1_question_text`:

1. `question_lsa4_bi`
   - current baseline
   - no stopword removal
   - TF-IDF bi-gram LSA with `4` components
2. `question_stop_lsa2_bi`
   - English stopword removal
   - TF-IDF bi-gram LSA with `2` components
3. `question_stop_lsa4_bi`
   - English stopword removal
   - TF-IDF bi-gram LSA with `4` components
4. `question_stop_lsa8_bi`
   - English stopword removal
   - TF-IDF bi-gram LSA with `8` components

For each encoding, the benchmark compares:

- standalone local-question route
- `geometry_plus_<encoding>` route

Outputs:

- `results/afterhours_transfer_pair_tail_question_encoding_benchmark_real/afterhours_transfer_pair_tail_question_encoding_benchmark_summary.json`
- `results/afterhours_transfer_pair_tail_question_encoding_benchmark_real/afterhours_transfer_pair_tail_question_encoding_benchmark_family_overview.csv`
- `results/afterhours_transfer_pair_tail_question_encoding_benchmark_real/afterhours_transfer_pair_tail_question_encoding_benchmark_tuning.csv`
- `results/afterhours_transfer_pair_tail_question_encoding_benchmark_real/afterhours_transfer_pair_tail_question_encoding_benchmark_test_predictions.csv`
- `results/afterhours_transfer_pair_tail_question_encoding_benchmark_real/afterhours_transfer_pair_tail_question_encoding_component_terms.csv`

## Main Findings

### 1. The original no-stopword bi-gram LSA encoding remains the best route

Held-out latest-window scores:

- `question_lsa4_bi ≈ 0.99865468`
- `hard_abstention ≈ 0.99864017`
- `geometry_only ≈ 0.99863878`
- `question_stop_lsa8_bi ≈ 0.99859079`
- `question_stop_lsa4_bi ≈ 0.99855614`
- `question_stop_lsa2_bi ≈ 0.99851083`

So the earlier win is not because we accidentally left performance on the table inside the same local-question object.
The current encoding is still the best one in this compact family.

### 2. Cleaner stopword-removed encodings are more interpretable, but they clearly lose the predictive lift

The stopword-removed components become much more topical. Their leading terms look like:

- `supply chain`
- `advertising`
- `growth`
- `little bit`
- `quarter`

By contrast, the current best encoding keeps more conversational / framing-heavy components.

## this means the useful signal is not just topic content
## it is tied to the local phrasing / framing structure of the hardest analyst question

That is a meaningful scientific update.
The strongest local-question signal looks more like an **interaction-form / discourse-style** cue than a clean topical semantic axis.

### 3. Adding geometry still does not help

The strongest geometry-coupled variants stay at or below the old geometry ceiling:

- `geometry_plus_question_stop_lsa2_bi ≈ 0.99863878`
- `geometry_plus_question_lsa4_bi ≈ 0.99863851`
- `geometry_plus_question_stop_lsa8_bi ≈ 0.99863727`
- `geometry_plus_question_stop_lsa4_bi ≈ 0.99863723`

So this checkpoint reinforces the same message as before:

## the new signal is useful as a narrow standalone local semantic veto,
## not as a broader geometry-shell upgrade

### 4. The new local-question result survives this compact-encoding check, but it also becomes easier to interpret cautiously

This benchmark does **not** overturn the previous pair-tail text result.
Instead it sharpens it:

- the win is real enough that nearby compact encodings do not replace it,
- but cleaner content-focused encodings do worse,
- so the mechanism is probably more subtle than “better topic semantics”.

That suggests a more careful description going forward:

## the strongest new transfer signal is the local semantic-framing structure of the hardest analyst question

rather than simply “question topic semantics”.

## Updated Interpretation

This is a good confirmation checkpoint because it narrows the meaning of the new result.

### What we now know

1. The local hardest-question signal survives a compact encoding benchmark.
2. The original `LSA(4)` bi-gram version remains the best current encoding.
3. More content-cleaned, stopword-removed encodings are easier to read but consistently weaker.
4. Geometry still does not improve this line.

### What that means

The next step should stay disciplined:

- **not** more shell complexity,
- **not** wider feature unions,
- but a careful follow-up on local hardest-question **framing / interaction-form** evidence and whether it can be confirmed across additional confirmation slices.

So the current reading is:

## the pair-tail hardest-question result is real enough to survive an encoding check,
## but its useful content appears to live in local analyst-question framing rather than in a cleaner topical embedding alone
