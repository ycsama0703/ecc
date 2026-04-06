# After-Hours Transfer Pair-Tail Question Shared-Core Checkpoint 2026-03-22

## Purpose

After the lexical-margin negative result, the next useful research question was no longer “can we tune the compact factor a bit more?”

It was:

## what exactly is the compact clarificatory core,
## and why does it appear only in a narrow late-window subset?

The goal of this checkpoint is therefore diagnostic rather than method-building:

- separate the richer hardest-question semantic veto set into:
  - `shared_veto`
  - `sem_only_veto`
  - `lex_only_veto`
- characterize the text and `Q&A` structure of those subsets
- understand the current temporal-emergence story without adding a new shell or rule

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_shared_core_diagnostics.py`

Inputs:

- `results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_test_predictions.csv`
- `results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv`

The diagnostic:

1. reconstructs the latest held-out overlap groups
2. attaches hardest-question text plus local pair-tail `Q&A` features
3. compares:
   - group-level gains vs `hard_abstention`
   - year distribution
   - text statistics
   - clarificatory lexical-pattern activations
   - distinctive ngrams for `shared_veto` vs `sem_only_veto`

Outputs:

- `results/afterhours_transfer_pair_tail_question_shared_core_diagnostics_real/afterhours_transfer_pair_tail_question_shared_core_summary.json`
- `results/afterhours_transfer_pair_tail_question_shared_core_diagnostics_real/afterhours_transfer_pair_tail_question_shared_core_rows.csv`
- `results/afterhours_transfer_pair_tail_question_shared_core_diagnostics_real/afterhours_transfer_pair_tail_question_shared_core_year_counts.csv`
- `results/afterhours_transfer_pair_tail_question_shared_core_diagnostics_real/afterhours_transfer_pair_tail_question_shared_core_activation_summary.csv`

## Main Findings

### 1. The compact shared core is tiny and disappears by 2025

Overlap counts on the held-out latest window:

- `shared_veto = 4`
- `sem_only_veto = 9`
- `lex_only_veto = 3`

Year breakdown:

- `2023`: shared `2`, sem-only `6`, lex-only `1`
- `2024`: shared `2`, sem-only `2`, lex-only `2`
- `2025`: shared `0`, sem-only `1`, lex-only `0`

So the compact clarificatory core is **not** a broad late-window rule.
It is a very small subset that appears in `2023–2024` inside the latest held-out window, while the richer semantic route still carries an extra `2025` tail.

### 2. The shared core is the cleanest part of the richer semantic route

For `shared_veto`:

- lexical net MSE gain vs hard `≈ 2.42e-09`
- semantic net MSE gain vs hard `≈ 2.42e-09`
- lexical win share vs hard `= 1.0`
- semantic win share vs hard `= 1.0`

For `sem_only_veto`:

- lexical net gain `= 0`
- semantic net gain vs hard `≈ 7.36e-09`
- semantic win share vs hard `≈ 0.556`

For `lex_only_veto`:

- lexical net gain vs hard `≈ -7.31e-10`

So this checkpoint strengthens the earlier lexical-confirmation result:

## the compact factor recovers the cleanest precision core,
## while the richer semantic route keeps extra recall that is real but noticeably less pure

### 3. The shared core aligns strongly with clarificatory / model-building language

The clearest lexical-pattern difference between `shared_veto` and `sem_only_veto` is:

- `wondering`
- `wondering if`
- `if you could`

Activation summary for `shared_veto`:

- `clarify_modeling__wondering`: mean `1.25`, nonzero share `0.75`
- `clarify_modeling__wondering_if`: mean `1.0`, nonzero share `0.75`
- `clarify_modeling__if_you_could`: mean `1.0`, nonzero share `0.75`

By contrast, these features are effectively absent from `sem_only_veto`.

Representative shared-core events:

- `IBM_2023_Q2`: “wondering if you can share some thoughts...”
- `NKE_2023_Q1`: “I was just wondering if you -- to clarify...”
- `AMZN_2024_Q1`: “Talk just about where we are in terms of...”
- `MSFT_2024_Q3`: “I'd love to hone in a little bit...”

This is the cleanest evidence so far that the compact factor is picking up a **clarificatory / model-building analyst framing core**, not generic topic content.

### 4. The shared core is more numerically grounded than the sem-only tail

Mean text stats:

- `shared_veto` numeric-token share `≈ 0.0314`
- `sem_only_veto` numeric-token share `≈ 0.0070`

It also has slightly higher hedge share:

- `shared_veto ≈ 0.0342`
- `sem_only_veto ≈ 0.0270`

At the same time, the sem-only tail is denser and broader on the `Q&A` side:

- mean `qa_pair_count`
  - `shared_veto ≈ 4.5`
  - `sem_only_veto ≈ 6.22`

So the compact core does **not** simply correspond to the hardest or densest local interaction.
Instead, it looks like a narrower subset where the hardest question is phrased in a more explicit clarificatory / modeling style.

### 5. The sem-only tail is broader, more mixed, and more topic-contaminated

`sem_only_veto` still contains real upside:

- `AAPL_2023_Q4`
- `NVDA_2023_Q1`
- `AMZN_2024_Q3`
- `MSFT_2025_Q1`

But it also contains noisier or topic-contaminated rows, especially around:

- Disney / ESPN / asset-separation style strategic probes

This is consistent with the current story:

- the richer semantic route keeps more recall
- but that extra recall is less compressible into a clean compact lexical mechanism

## Updated Interpretation

This checkpoint clarifies the current hierarchy a lot.

### What we now know

1. The compact `clarify_modeling_lex` factor captures a **real shared core**.
2. That shared core is dominated by:
   - clarificatory / model-building framing
   - “wondering if” / “if you could” style phrasing
   - stronger local numeric grounding
3. The richer `question_lsa4_bi` route still adds extra recall beyond that core.
4. But that extra recall is broader, less pure, and harder to compress.
5. The compact core is absent in the small `2025` tail, so it is still too temporally narrow to be promoted into a standalone method block.

### What that means

The current transfer-side hierarchy becomes even sharper:

- `hard_abstention` remains the safest mainline
- `question_lsa4_bi` remains the stronger richer exploratory local-semantic route
- `clarify_modeling_lex_factor_pca1` remains the best compact confirmation of the same mechanism
- and this new shared-core diagnostic explains **what part** of the richer route is currently most trustworthy and most interpretable

So the next step should not be more threshold tuning.
The more valuable direction is to keep characterizing the **late shared core vs broader semantic tail**, and only then ask whether a more expressive but still compact local representation can bridge that gap without bloating the method.
