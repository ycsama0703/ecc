# After-Hours Transfer Pair-Tail Question-Framing Factor Checkpoint 2026-03-22

## Purpose

The new hardest-question research line has now reached a natural next test:

- the local hardest-question semantic route is real,
- its gain survives an encoding check,
- and diagnostics suggest a small clarificatory / model-building framing pocket.

That raises the obvious compact-method question:

## can this local framing pocket be compressed into a small interpretable feature/factor route that reproduces the semantic gain?

This is exactly the kind of disciplined follow-up we want:

- no extra shell complexity,
- no new routing logic,
- just a compact learnable approximation test.

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_framing_factor_benchmark.py`

The benchmark keeps the same transfer protocol fixed:

- train `val2020_test_post2020`
- validate `val2021_test_post2021`
- test `val2022_test_post2022`
- disagreement rows still fallback to `pre_call_market_only`
- only agreement rows are learnably refined

It builds compact framing features from `tail_top1_question_text` using the same surface-form diagnostics already identified:

### framing families

1. `framing_core`
   - `token_count`
   - `numeric_token_share`
   - `first_person_share`
   - `second_person_share`
   - `hedge_share`
   - `followup_marker_count`

2. `framing_structure`
   - the above plus:
   - `sentence_like_count`
   - `question_mark_count`
   - `modal_share`

### route variants

For each family, the benchmark compares:

- direct feature route
- `PCA1` factor route
- `geometry_plus_factor` route

Outputs:

- `results/afterhours_transfer_pair_tail_question_framing_factor_benchmark_real/afterhours_transfer_pair_tail_question_framing_factor_benchmark_summary.json`
- `results/afterhours_transfer_pair_tail_question_framing_factor_benchmark_real/afterhours_transfer_pair_tail_question_framing_factor_benchmark_family_overview.csv`
- `results/afterhours_transfer_pair_tail_question_framing_factor_benchmark_real/afterhours_transfer_pair_tail_question_framing_factor_benchmark_tuning.csv`
- `results/afterhours_transfer_pair_tail_question_framing_factor_benchmark_real/afterhours_transfer_pair_tail_question_framing_factor_benchmark_test_predictions.csv`
- `results/afterhours_transfer_pair_tail_question_framing_factor_benchmark_real/afterhours_transfer_pair_tail_question_framing_factor_benchmark_loadings.csv`

## Main Findings

### 1. The compact framing routes do not reproduce the hardest-question semantic gain

Held-out latest-window scores:

- `geometry_only ≈ 0.99863878`
- `geometry_plus_framing_core_factor_pca1 ≈ 0.99863878`
- `geometry_plus_framing_structure_factor_pca1 ≈ 0.99863697`
- `framing_structure_factor_pca1 ≈ 0.99862733`
- `framing_structure_direct ≈ 0.99852090`
- `framing_core_direct ≈ 0.99850684`
- `framing_core_factor_pca1 ≈ 0.99846908`

Reference routes remain:

- `hard_abstention ≈ 0.99864017`
- `question_lsa4_bi ≈ 0.99865468`

So the answer is clear:

## the compact framing block does not recover the semantic route
## and it does not beat the current geometry / hard-abstention shell

### 2. The best compact framing route just collapses back to the geometry ceiling

The best compact route is:

- `geometry_plus_framing_core_factor_pca1 ≈ 0.99863878`

which is effectively tied with:

- `geometry_only ≈ 0.99863878`

This means the framing factor is not adding new usable lift once the geometry controller is already present.

### 3. Direct framing features look attractive on validation, but do not transfer cleanly

Validation scores:

- `framing_structure_direct val R^2 ≈ 0.997389`
- `framing_core_direct val R^2 ≈ 0.997417`

These are actually a bit higher than the `geometry_only` validation score.
But on held-out latest-window test they fall clearly below even the geometry baseline.

So this is a useful warning sign:

## the compact framing summary is easy to overfit on validation,
## but it does not transfer as a stable standalone controller

### 4. The framing factors are interpretable, but still too lossy

The strongest `framing_core_factor_pca1` loadings are:

- `first_person_share`
- `hedge_share`
- `token_count`
- `numeric_token_share`
- `second_person_share`

The strongest `framing_structure_factor_pca1` loadings are:

- `sentence_like_count`
- `token_count`
- `question_mark_count`
- `first_person_share`
- `numeric_token_share`

So the factorization is sensible and interpretable.
But that is exactly why the result is informative:

## the surface-form framing pocket is real,
## yet these compact aggregates still lose too much of what the semantic route is capturing

### 5. This clarifies the status of the new signal

At this point we have a sharper separation:

- the hardest-question semantic route is real and structured,
- the local framing pocket is interpretable,
- but a small handcrafted / factorized framing summary does not replace the richer semantic representation.

That means the current signal is not merely a few obvious surface cues in disguise.

## Updated Interpretation

This is a strong negative-result checkpoint in the good sense.

### What we now know

1. The local framing pocket is interpretable, but not easily reducible to a tiny surface-form feature block.
2. Compact framing factors do not reproduce the strongest hardest-question semantic route.
3. Direct framing features show a validation temptation but fail to transfer robustly.
4. The semantic route still contains information beyond these simple framing summaries.

### What that means

The current best reading is:

## hardest-question framing matters, but it is not adequately captured by a tiny handcrafted surface-form factor

That keeps the research story disciplined:

- we are not pretending the diagnostics alone are a new model,
- we tested the compact approximation directly,
- and the result says the semantic route is richer than a small surface summary.

So the next step should not be “turn the framing diagnostics into a new rule”.
Instead, it should focus on whether there is a slightly richer but still compact representation of local question framing that preserves transferability without collapsing into overfit.
