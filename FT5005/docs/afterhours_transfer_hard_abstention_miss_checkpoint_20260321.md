# After-Hours Transfer Hard-Abstention Miss Diagnostics Checkpoint 2026-03-21

## Purpose

The latest question-role diagnostics already narrowed the role-text line quite aggressively:

- analyst-question semantics are real but sparse,
- they are not a better transfer controller,
- and their only distinctive routing behavior is a small, noisy agreement-veto policy.

That raises the next cleaner research question:

## where does the current strongest transfer shell still fail?

The goal here is not to add a new model.
It is to understand whether the remaining hard cases of the current hard-abstention shell have a recognizable profile in the existing observability, `Q&A`, and surprise features.

## Design

New script:

- `scripts/run_afterhours_transfer_hard_abstention_miss_diagnostics.py`

Inputs:

- question-role gate diagnostics events:
  - `results/afterhours_transfer_question_role_gate_diagnostics_lsa4_real/afterhours_transfer_question_role_gate_diagnostics_events.csv`
- clean `after_hours` panel side inputs:
  - `results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv`
- event text / QA interaction features:
  - `results/features_real/event_text_audio_features.csv`
- QA benchmark features:
  - `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`

The script works on the held-out latest-window transfer rows and:

1. computes hard-abstention squared error for every event,
2. defines the top miss subset as the top quartile of hard-abstention squared error,
3. compares that top-miss subset against the remaining events,
4. reports the largest standardized feature shifts, state enrichment, ticker concentration, dominant semantic-component pockets, and top event-level failures.

Outputs:

- `results/afterhours_transfer_hard_abstention_miss_diagnostics_lsa4_real/afterhours_transfer_hard_abstention_miss_diagnostics_summary.json`
- `results/afterhours_transfer_hard_abstention_miss_diagnostics_lsa4_real/afterhours_transfer_hard_abstention_miss_feature_shifts.csv`
- `results/afterhours_transfer_hard_abstention_miss_diagnostics_lsa4_real/afterhours_transfer_hard_abstention_miss_states.csv`
- `results/afterhours_transfer_hard_abstention_miss_diagnostics_lsa4_real/afterhours_transfer_hard_abstention_miss_components.csv`
- `results/afterhours_transfer_hard_abstention_miss_diagnostics_lsa4_real/afterhours_transfer_hard_abstention_miss_tickers.csv`
- `results/afterhours_transfer_hard_abstention_miss_diagnostics_lsa4_real/afterhours_transfer_hard_abstention_miss_events.csv`

## Main Findings

### 1. The remaining hard-abstention misses are real and sharply concentrated

Held-out latest-window counts:

- total events: `60`
- top miss events: `15`
- rest: `45`
- top-miss threshold (`hard_abstention_se`): `≈ 4.91e-09`

Error magnitude gap:

- top-miss mean hard-abstention squared error: `≈ 5.83e-08`
- rest mean hard-abstention squared error: `≈ 9.47e-10`
- top-miss mean absolute error: `≈ 1.90e-04`
- rest mean absolute error: `≈ 2.64e-05`

So the current shell is not just making a few random mistakes.
Its residual failures define a very distinct high-error pocket.

### 2. The hardest misses skew toward dense, difficult `Q&A` under slightly weaker observability

Largest standardized shifts between the top-miss quartile and the rest include:

- higher `qa_pair_count` (`≈ 8.53` vs `6.18`, effect `≈ +0.746`)
- lower `a4_strict_row_share` (`≈ 0.667` vs `0.693`, effect `≈ -0.602`)
- lower `qa_multi_part_question_share` (`≈ 0.614` vs `0.735`, effect `≈ -0.590`)
- lower `qa_bench_direct_early_score_mean` (`≈ 0.659` vs `0.760`, effect `≈ -0.580`)
- lower `a4_strict_high_conf_share` (`≈ 0.940` vs `0.949`, effect `≈ -0.574`)
- lower `qa_bench_coverage_mean` (`≈ 0.178` vs `0.194`, effect `≈ -0.572`)
- higher `qa_bench_evasion_score_mean` (`≈ 0.254` vs `0.220`, effect `≈ +0.570`)
- lower `qa_bench_direct_answer_share` (`≈ 0.744` vs `0.831`, effect `≈ -0.558`)
- higher `qa_pair_low_overlap_share` (effect `≈ +0.539`)
- higher `qa_pair_answer_forward_rate_mean` (effect `≈ +0.539`)

This is the cleanest empirical pattern in the checkpoint:

## the hardest remaining shell misses look like dense `Q&A` exchanges with weaker observability, weaker directness / coverage, and more evasive or low-overlap answers

That is a much tighter research direction than continuing to squeeze more value out of role-text routing.

### 3. Surprise is present, but only as a secondary stress marker

The miss quartile also shows somewhat larger fundamentals surprise:

- `revenue_surprise_pct`: effect `≈ +0.386`
- `abs_revenue_surprise_pct`: effect `≈ +0.323`

So surprise may matter, but it is not the first-order pattern.
The stronger and more stable signature is still answerability / evasion under weaker observability.

### 4. These failures are shell-wide, not just a question-role veto artifact

Top-miss state composition:

- `agreement_keep_agreed`: `5` top misses, `24` rest
- `agreement_veto_to_pre`: `6` top misses, `9` rest
- `disagreement_auto_pre`: `4` top misses, `12` rest

Within-state top-miss rates:

- keep-agreed: `≈ 0.172`
- veto-to-pre: `≈ 0.400`
- disagreement-auto-pre: `≈ 0.250`

The veto subset is enriched, but the hard cases are clearly **not confined** to role-text behavior.

## the next signal search should target the broader hard-abstention miss profile, not just the role-text veto subset

### 5. The miss pocket is semantically and cross-sectionally concentrated

Ticker concentration among top misses:

- `AMGN`: `4 / 5` latest-window events are top misses
- `CSCO`: `3 / 9`
- `AAPL`: `2 / 5`
- `NKE`: `2 / 8`
- `NVDA`: `2 / 8`

Dominant semantic-component pockets with repeated hard misses include:

- `data center / computing / data / center / nvidia`
- `high performance / ... / compute`
- generic management-phrasing buckets

Representative high-error events include:

- `CSCO_2023_Q2`
- `AMGN_2024_Q2`
- `AMZN_2023_Q1`
- `AMGN_2024_Q3`
- `NVDA_2024_Q1`
- `NVDA_2024_Q2`
- `CSCO_2024_Q2`
- `NKE_2025_Q2`

So the miss structure is not random noise; it lives in recognizable semantic and ticker pockets.

## Updated Interpretation

This checkpoint is useful because it shifts the transfer research agenda in a cleaner direction.

### What we now know

1. The current hard-abstention shell is still the strongest compact transfer method in the repo.
2. Its remaining misses are concentrated enough to be studied as a distinct failure pocket.
3. That pocket is characterized most clearly by:
   - denser `Q&A`,
   - weaker observability,
   - lower directness / coverage,
   - higher evasion and low-overlap answering,
   - and only secondarily by larger surprise.
4. Those misses appear across agreement, veto, and disagreement states, so the next step should not be framed as “fix role-text routing.”

### What that means

The next compact research direction should be:

- **not** more role-text veto logic,
- **not** another router family sweep,
- but a cleaner search for a learnable upstream signal around:
  - answerability,
  - evasion,
  - `Q&A` density / difficulty,
  - observability,
  - and possibly surprise as a secondary modifier.

That keeps the project aligned with the current scientific story:
compact, interpretable, and transferable signals first — not more shell complexity.
