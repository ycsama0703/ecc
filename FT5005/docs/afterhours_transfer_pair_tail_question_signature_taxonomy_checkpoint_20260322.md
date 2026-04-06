# After-Hours Transfer Pair-Tail Question Signature Taxonomy Checkpoint 2026-03-22

## Purpose

The shared-core diagnostic clarified that the compact lexical factor recovers a tiny clean subset of the richer hardest-question semantic route.

The next research question was:

## what is the **taxonomy** of that broader semantic tail?

More concretely:

- is the shared core just “clarificatory language,” or something more specific?
- what kinds of questions sit in the `sem_only_veto` tail?
- does that tail decompose into cleaner subfamilies that might explain why the compact lexical factor is high-precision but low-recall?

This checkpoint stays deliberately compact:

- no new model shell
- no new routing layer
- just a structured taxonomy over the existing lexical families

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_signature_taxonomy.py`

Inputs:

- `results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_test_predictions.csv`
- `results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv`

Each active veto row is labeled by which lexical families are present in the hardest question:

- `clarify`
- `quant`
- `structural`

and then compressed into a compact signature such as:

- `clarify+quant`
- `clarify+structural`
- `quant`
- `structural`

Outputs:

- `results/afterhours_transfer_pair_tail_question_signature_taxonomy_real/afterhours_transfer_pair_tail_question_signature_taxonomy_summary.json`
- `results/afterhours_transfer_pair_tail_question_signature_taxonomy_real/afterhours_transfer_pair_tail_question_signature_taxonomy_rows.csv`
- `results/afterhours_transfer_pair_tail_question_signature_taxonomy_real/afterhours_transfer_pair_tail_question_signature_taxonomy_cell_summary.csv`
- `results/afterhours_transfer_pair_tail_question_signature_taxonomy_real/afterhours_transfer_pair_tail_question_signature_taxonomy_counts.csv`
- `results/afterhours_transfer_pair_tail_question_signature_taxonomy_real/afterhours_transfer_pair_tail_question_signature_taxonomy_year_counts.csv`

## Main Findings

### 1. The compact shared core is not generic clarificatory language

It is much more specific:

## every single `shared_veto` row is `clarify+quant`

Counts:

- `shared_veto`
  - `clarify+quant = 4 / 4`
- `sem_only_veto`
  - `clarify+structural = 3 / 9`
  - `clarify+quant = 2 / 9`
  - `clarify = 1 / 9`
  - `quant = 1 / 9`
  - `quant+structural = 1 / 9`
  - `structural = 1 / 9`
- `lex_only_veto`
  - one each of `clarify`, `clarify+quant`, and `clarify+structural`

So the clean compact core is not merely “clarify”:

## it is a very specific **clarify + quantitative bridge** pocket

### 2. That `clarify+quant` shared core is the cleanest subtype in the whole taxonomy

For `shared_veto :: clarify+quant`:

- `n = 4`
- years: `2023,2024`
- lexical net MSE gain vs hard `≈ 2.42e-09`
- semantic net MSE gain vs hard `≈ 2.42e-09`
- both win shares vs hard `= 1.0`

Representative events:

- `IBM_2023_Q2`
- `NKE_2023_Q1`
- `AMZN_2024_Q1`
- `MSFT_2024_Q3`

These are exactly the rows where:

- clarificatory/model-building phrasing is present
- quantitative or bridge-like content is also present
- and the compact lexical factor matches the richer semantic route perfectly

### 3. The richer semantic tail is heterogeneous, not monolithic

The `sem_only_veto` tail breaks into at least two meaningful branches.

#### Branch A: operational / quantitative bridge tail

- `clarify+quant` (`n = 2`)
  - semantic net gain vs hard `≈ 2.92e-09`
  - semantic win share `= 1.0`
- `quant` (`n = 1`)
  - semantic net gain vs hard `≈ 3.43e-09`
  - semantic win share `= 1.0`

These rows look closer to:

- operational follow-up
- numeric bridge questions
- capacity / spending / trajectory clarification

Examples:

- `NVDA_2023_Q1`
- `AAPL_2023_Q4`
- `MSFT_2025_Q1`

#### Branch B: structural / strategic probe tail

- `clarify+structural` (`n = 3`)
  - semantic net gain vs hard `≈ 1.63e-09`
  - semantic win share `≈ 0.333`
- `structural` (`n = 1`)
  - semantic net gain vs hard `≈ -6.05e-10`
- `quant+structural` (`n = 1`)
  - semantic net gain vs hard `≈ -1.61e-10`

These rows are much noisier and more topic-contaminated.
They include the Disney / ESPN / asset-separation style questions that had already started to show up in earlier diagnostics.

So the richer semantic tail is not “one extra thing.”
It is a mix of:

- a useful operational / quantitative extension,
- plus a noisier structural / strategic branch.

### 4. The compact factor is therefore high-precision partly because it avoids the structural branch

This is the most important interpretation from the checkpoint.

The compact lexical factor does **not** cover the whole semantic tail.
But what it does cover is exactly the subtype that is:

- clarificatory,
- quantitative,
- and locally model-building.

Meanwhile, the broader semantic route reaches into:

- operational quantitative follow-ups that the compact factor misses,
- but also a noisier structural / strategic tail.

That is why the current hierarchy makes sense:

- compact lexical factor = cleaner precision subset
- richer semantic route = larger recall with some mixed-quality tail

### 5. The `2025` tail is telling

Year counts:

- `2023`
  - shared `clarify+quant = 2`
  - sem-only mostly mixed and structural-heavy
- `2024`
  - shared `clarify+quant = 2`
  - sem-only splits across `clarify` and `clarify+structural`
- `2025`
  - no shared-core rows
  - only one sem-only row, and it is `clarify+quant`

So the compact shared core disappears in `2025`,
but the semantic tail does **not** disappear entirely.
Instead, the remaining tail is a more semantically varied operational / quantitative question that the tiny lexical factor does not recover.

This is a cleaner explanation for the earlier temporal-emergence puzzle:

## the compact factor is not “late-window unstable” in a random way;
## it is tuned to the clean clarify+quant core,
## while the last surviving tail in `2025` is already beyond that compact signature

## Updated Interpretation

This checkpoint sharpens the hardest-question story in a very useful way.

### What we now know

1. The compact lexical core is specifically a **clarify+quant** pocket.
2. The richer semantic route contains that pocket entirely.
3. Beyond it, the semantic tail splits into:
   - a promising operational / quantitative bridge branch
   - and a noisier structural / strategic branch
4. The compact factor's precision comes partly from refusing to model that noisier structural branch.
5. The remaining `2025` tail is consistent with this story: it still looks operational / quantitative, but not compactly lexical enough for the current factor.

### What that means

The current transfer-side hierarchy becomes more interpretable:

- `hard_abstention` remains the safest mainline
- `question_lsa4_bi` remains the stronger richer exploratory route
- `clarify_modeling_lex_factor_pca1` remains the best compact confirmation
- and this taxonomy now explains **why** the compact factor is high-precision but low-recall

The most promising next step is therefore **not** more threshold tuning.
It is to test whether a slightly richer but still compact local representation can extend the compact core from:

- pure `clarify+quant`

toward:

- the useful operational / quantitative `sem_only` tail

without also opening the noisier structural / strategic branch.
