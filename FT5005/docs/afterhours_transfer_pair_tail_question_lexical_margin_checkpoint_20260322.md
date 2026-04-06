# After-Hours Transfer Pair-Tail Question Lexical Margin Checkpoint 2026-03-22

## Purpose

After the lexical confirmation result, the next very natural question was:

## can we keep the compact clarificatory core,
## but filter away its weak lexical-only tail with a tiny learned confidence margin?

This is the cleanest possible follow-up to the previous checkpoint:

- no new shell,
- no new feature family,
- no extra routing logic,
- just a conservative threshold on the already learned compact lexical gain signal.

If this worked, it would have been a strong sign that the compact core is not only interpretable, but also stable enough to support a minimal learned tail filter.

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_lexical_margin_benchmark.py`

The benchmark fixes the same compact route as before:

- family: `clarify_modeling_lex`
- representation: `PCA1` factor
- train `val2020_test_post2020`
- validate `val2021_test_post2021`
- refit on `2020 + 2021`
- test on `val2022_test_post2022`

The only new degree of freedom is a **conservative margin threshold** on the compact factor's predicted gain signal.

Decision rule:

- baseline compact route: use agreed prediction if `signal > 0`
- conservative margin route: use agreed prediction if `signal > threshold`
- only thresholds `<= 0` are allowed, so the route can only become **more conservative**, not more aggressive

Outputs:

- `results/afterhours_transfer_pair_tail_question_lexical_margin_benchmark_real/afterhours_transfer_pair_tail_question_lexical_margin_benchmark_summary.json`
- `results/afterhours_transfer_pair_tail_question_lexical_margin_benchmark_real/afterhours_transfer_pair_tail_question_lexical_margin_benchmark_overview.csv`
- `results/afterhours_transfer_pair_tail_question_lexical_margin_benchmark_real/afterhours_transfer_pair_tail_question_lexical_margin_benchmark_tuning.csv`
- `results/afterhours_transfer_pair_tail_question_lexical_margin_benchmark_real/afterhours_transfer_pair_tail_question_lexical_margin_benchmark_test_predictions.csv`

## Main Findings

### 1. Validation learns no conservative veto at all

Best validation setting:

- `alpha = 0.01`
- `threshold = 0.0`
- validation `R^2 ≈ 0.99716231`
- validation use-agreed share `= 1.0`
- validation veto share `= 0.0`

This is the most important result of the whole checkpoint.

## the earlier windows provide no learnable evidence for a conservative lexical tail filter

In other words:

- the compact lexical factor does **not** learn a stable negative-margin region on validation,
- so the natural margin-tuning procedure simply collapses back to the original zero-threshold route.

### 2. The test-side margin route is exactly identical to the base compact factor

Held-out latest-window test:

- base compact factor:
  - `R^2 ≈ 0.99864268`
  - veto rows `= 7`
- margin route:
  - `R^2 ≈ 0.99864268`
  - veto rows `= 7`
- `p(MSE)` margin vs base factor `= 1.0`

So the conservative margin route does not improve, degrade, or alter the compact route at all.
It simply reproduces it.

### 3. This is a useful negative result, not a dead end

The failure here is informative because it narrows the story in a disciplined way.

The previous lexical-confirmation result already showed:

- the compact factor captures a clean shared-veto core,
- but its lexical-only tail is weak.

This new margin benchmark now adds:

## that weak tail is not learnably separable from earlier temporal windows by a tiny confidence threshold

So the lexical-only tail is not just “one more threshold away” from becoming a clean method.

### 4. The compact clarificatory core is real, but still temporally narrow

Taken together, the lexical-pattern benchmark, lexical-confirmation checkpoint, and this margin result now imply:

1. there is a real compact clarificatory / model-building core,
2. it recovers the precision-oriented shared-veto subset,
3. but it does not yet show stable enough earlier-window behavior to support a learned conservative tail filter.

That means the current compact signal is still better described as:

- a **research-confirmed local mechanism**,
- not yet a stable new transfer-side method block.

## Updated Interpretation

This is a clean and valuable negative-result checkpoint.

### What we now know

1. The compact `clarify_modeling_lex` factor is real.
2. Its clean value is concentrated in the shared-veto core with the richer semantic route.
3. But a tiny learned margin threshold cannot use earlier windows to separate the good compact veto cases from the weak lexical-only tail.
4. So the compact core is still not stable enough to be hardened into a new minimal standalone method.

### What that means

The current best hierarchy becomes even clearer:

- `hard_abstention` remains the safest transfer-side mainline,
- `question_lsa4_bi` remains the stronger richer exploratory route,
- `clarify_modeling_lex_factor_pca1` remains the best compact confirmation of the same mechanism,
- but **margin-tuned tail filtering does not yet add anything learnable**.

So the next step should not be “tune more thresholds”.
Instead, the more valuable direction is to understand **why the compact core only appears late**, and whether the shared-veto pocket itself has a sharper event taxonomy that is stable enough to learn across time.
