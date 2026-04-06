# After-Hours Transfer Pair-Tail Question Lexical-Pattern Checkpoint 2026-03-22

## Purpose

The new hardest-question line has already established four things:

1. the top-1 hardest analyst-question text is a real transfer-side signal,
2. the best encoding still uses the original no-stopword bi-gram `LSA(4)` setup,
3. the gain comes only through a narrow agreement-veto pocket,
4. tiny surface-form framing factors do **not** recover that gain.

So the next disciplined question is:

## can a slightly richer but still compact lexical-pattern bundle recover part of the hardest-question signal without reverting to the full semantic route?

This is exactly the kind of follow-up we want at this stage:

- still compact,
- still interpretable,
- still learnable,
- no new routing shell,
- no feature-sprawl beyond the local hardest-question text.

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_lexical_pattern_benchmark.py`

The benchmark keeps the same transfer protocol fixed:

- train `val2020_test_post2020`
- validate `val2021_test_post2021`
- refit on `2020 + 2021`
- test on `val2022_test_post2022`
- disagreement rows still fallback to `pre_call_market_only`
- only agreement rows are learnably refined

### lexical-pattern families

1. `clarify_modeling_lex`
   - patterns such as:
     - `wondering`
     - `wondering if`
     - `if you could`
     - `help us understand`
     - `help us think`
     - `give us color`
     - `talk about`
     - `walk us through`
     - `frame` / `model`

2. `quant_bridge_lex`
   - patterns such as:
     - `around`
     - `million`
     - `billion`
     - `percent`
     - `basis points`
     - `sequential`
     - `quarter`
     - `cadence`
     - `bridge`
     - `trajectory`
     - `outlook`
     - `visibility`

3. `structural_probe_lex`
   - a deliberately cautious contrast family with:
     - `platform`
     - `assets`
     - `content`
     - `sports`
     - `portfolio`
     - `services`
     - `distribution`
     - `channel`

4. `clarify_quant_lex`
   - union of the clarificatory/model-building and quant/bridge bundles

### route variants

For each family, the benchmark compares:

- direct lexical-count route
- `PCA1` factor route
- `geometry_plus_factor` route

Outputs:

- `results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_summary.json`
- `results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_family_overview.csv`
- `results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_tuning.csv`
- `results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_test_predictions.csv`
- `results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_loadings.csv`
- `results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_pattern_catalog.csv`
- `results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_activation_summary.csv`

## Main Findings

### 1. A compact clarificatory lexical factor is the first small interpretable block to edge above hard abstention

Held-out latest-window scores:

- `question_lsa4_bi ≈ 0.99865468`
- `clarify_modeling_lex_factor_pca1 ≈ 0.99864268`
- `hard_abstention ≈ 0.99864017`
- `geometry_only ≈ 0.99863878`
- `quant_bridge_lex ≈ 0.99861000`
- `clarify_quant_lex ≈ 0.99859188`
- `clarify_modeling_lex ≈ 0.99849471`

So the new result is real but narrow:

## a very small clarificatory/model-building lexical factor can recover a little of the hardest-question gain
## but it still remains below the full hardest-question semantic route

Paired tests for the best compact route (`clarify_modeling_lex_factor_pca1`):

- vs `hard_abstention`: `p(MSE) ≈ 0.325`
- vs `geometry_only`: `p(MSE) ≈ 0.1635`
- vs `question_lsa4_bi`: `p(MSE) ≈ 0.12675`

So this is a promising compact positive result, but it is still exploratory rather than a stable upgrade claim.

### 2. The gain comes from a tiny subset of agreement events

Relative to hard abstention, the best lexical-factor route changes only:

- `5 / 60` latest-window events in total
- all `5` are agreement events
- no disagreement behavior changes

Net result on those changed rows:

- `4` positive changes
- `1` negative change
- total net gain `≈ 1.69e-09`

That means this compact route behaves exactly like the broader hardest-question line suggested:

## not as a new global controller,
## but as a tiny local veto refinement

### 3. The compact lexical factor partially recovers the semantic veto set

Of the `5` rows where the compact lexical factor differs from hard abstention:

- `4` overlap with the richer `question_lsa4_bi` veto set
- only `1` row is lexical-only (`IBM_2023_Q4`), and it is the lone negative change

This is scientifically useful because it says:

## the compact lexical factor is not inventing a different story
## it is compressing part of the same local hardest-question mechanism,
## but less completely than the richer semantic route

### 4. The successful compact factor is specifically clarificatory / model-building

The strongest refit loadings for `clarify_modeling_lex_factor_pca1` are:

- `wondering`
- `wondering if`
- `if you could`
- `help us understand`
- `help us think`

with smaller opposite-side weight on:

- `give_color`
- `walk_through`
- generic `can_you`

This is a very nice confirmation of the earlier diagnostics:

## the compact recoverable part of the signal is the clarificatory / model-building framing,
## not just arbitrary topic content

### 5. Direct lexical bundles do not transfer nearly as well as the factorized version

The direct lexical-count routes are much weaker:

- `clarify_modeling_lex ≈ 0.99849471`
- `clarify_quant_lex ≈ 0.99859188`

while the best factor route is:

- `clarify_modeling_lex_factor_pca1 ≈ 0.99864268`

So this is another clean methodological lesson:

## the useful compact signal is not raw lexical-count piling
## it is the low-dimensional shared axis inside the clarificatory lexical block

### 6. Geometry-plus-factor does not help further

Just like several earlier upstream experiments, once geometry is already present:

- `geometry_plus_clarify_modeling_lex_factor_pca1 ≈ 0.99863878`

which is effectively tied with:

- `geometry_only ≈ 0.99863878`

So the compact lexical factor is best read as:

- a standalone compact veto-style signal,
- not a geometry-shell upgrade.

## Updated Interpretation

This is a genuinely useful research checkpoint.

### What we now know

1. The hardest-question signal still cannot be reduced to tiny surface-form framing statistics alone.
2. But it **can** now be partially compressed into a very small clarificatory/model-building lexical factor.
3. That compact factor only recovers a subset of the richer semantic veto behavior.
4. The full `question_lsa4_bi` route remains stronger than the compact lexical version.
5. The compact signal is most naturally interpreted as a local clarificatory framing axis, not as a new broad routing layer.

### What that means

The current best reading is:

## there is now a compact recoverable core inside the hardest-question signal,
## and that core is clarificatory / model-building analyst framing

That is a better result than the earlier tiny framing-factor test, because it shows the pocket is not purely opaque.
But it also keeps the story honest:

- the recoverable compact core is only partial,
- the gain remains small,
- and the richer hardest-question semantic route still carries additional information.

So the next step should not be “declare victory with a hand-built lexical rule”.
Instead, it should confirm how stable this compact clarificatory factor is across slices and whether it consistently tracks the same narrow agreement-veto pocket as the richer semantic route.
