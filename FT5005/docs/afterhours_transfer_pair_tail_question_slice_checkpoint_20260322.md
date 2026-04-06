# After-Hours Transfer Pair-Tail Question-Slice Confirmation Checkpoint 2026-03-22

## Purpose

The pair-tail text and encoding checkpoints together established that:

- the top-1 hardest analyst-question text is the first local semantic refinement in the current transfer line to edge past hard abstention,
- and that gain survives a compact encoding benchmark,
- but it still looks small, narrow, and more like local framing than broad topic semantics.

That makes the next confirmation question straightforward:

## where does this new local-question signal actually help?

This is an important step for the research story because it tells us whether the gain lines up with the shell’s known hard pockets, or whether it is just an unstructured local accident.

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_slice_confirmation.py`

Inputs:

- latest held-out predictions from `results/afterhours_transfer_pair_tail_text_benchmark_lsa4_real/afterhours_transfer_pair_tail_text_benchmark_test_predictions.csv`
- clean `after_hours` panel
- event-level text / timing features
- `Q&A` benchmark features

The script joins all `60 / 60` held-out latest-window events with side inputs and defines a compact set of confirmation slices:

1. `agreement_veto`
2. `agreement_keep`
3. `disagreement`
4. `hard_miss_top_quartile`
5. `high_qna_density`
6. `low_directness_high_evasion`
7. `weaker_observability`
8. `dense_evasive_weak_obs`

Key medians used for slice construction:

- `qa_pair_count median = 6.0`
- `qa_bench_direct_early_score_mean median ≈ 0.7586`
- `qa_bench_evasion_score_mean median ≈ 0.2227`
- `a4_strict_row_share median ≈ 0.6935`

Outputs:

- `results/afterhours_transfer_pair_tail_question_slice_confirmation_real/afterhours_transfer_pair_tail_question_slice_confirmation_summary.json`
- `results/afterhours_transfer_pair_tail_question_slice_confirmation_real/afterhours_transfer_pair_tail_question_slice_confirmation_overview.csv`

## Main Findings

### 1. The gain still comes entirely from the agreement-veto subset

This confirms the earlier mechanism cleanly.

Slice summary:

- `agreement_veto`: `n = 20`
- `agreement_keep`: `n = 24`
- `disagreement`: `n = 16`

Relative to hard abstention:

- `agreement_keep` gain: exactly `0`
- `disagreement` gain: exactly `0`
- `agreement_veto` net gain: `≈ 9.78e-09`
- `agreement_veto` win share: `9 / 20 = 0.45`
- `agreement_veto` `p(MSE) ≈ 0.047`

## the local-question route is not a general transfer upgrade
## it is a better agreement-veto policy over a narrow subset

That is actually a healthy result, because it tells us precisely what this signal is doing.

### 2. The gain is meaningfully aligned with dense and difficult `Q&A`, but only partially

The strongest broader confirmation slices are:

- `high_qna_density` (`n = 33`)
  - route-minus-hard `R^2 ≈ +3.995e-05`
  - share of overall net gain `≈ 0.806`
- `low_directness_high_evasion` (`n = 26`)
  - route-minus-hard `R^2 ≈ +7.88e-06`
  - share of overall net gain `≈ 0.438`
- `weaker_observability` (`n = 30`)
  - route-minus-hard `R^2 ≈ +4.62e-06`
  - share of overall net gain `≈ 0.271`

This is important because it matches the earlier hard-abstention miss diagnostics:

- denser `Q&A`
- less direct answers
- more evasive answering
- somewhat weaker observability

## so the new local-question signal is not arbitrary
## it points in the same direction as the shell’s known difficult pockets

### 3. But it is not simply a detector of the hardest overall shell failures

The `hard_miss_top_quartile` slice does improve slightly:

- `n = 15`
- route-minus-hard `R^2 ≈ +1.00e-05`
- share of overall net gain `≈ 0.351`

But it is clearly **not** where most of the gain comes from.

This is an important nuance.
The new signal is related to difficult `Q&A`, but it does not act like a generic “hardest-case detector”.
It seems to capture a more specific local framing pocket inside the broader difficult region.

### 4. The densest combined hard slice stays flat

For the intersection slice `dense_evasive_weak_obs`:

- `n = 14`
- route-minus-hard `R^2 = 0`
- net gain vs hard `= 0`

That means the local-question route does **not** simply activate on the most extreme version of the earlier answerability story.

This is useful scientifically because it prevents an overly simple interpretation.
The signal is related to density / evasion / observability, but it is not reducible to a single “most difficult combined slice”.

### 5. The event pockets still look coherent rather than random

Largest positive events vs hard abstention remain:

- `AAPL_2023_Q4`
- `NVDA_2023_Q1`
- `AMZN_2024_Q3`
- `AMZN_2024_Q1`
- `NKE_2023_Q1`

Largest negative events remain:

- `DIS_2023_Q2`
- `DIS_2023_Q3`
- `NKE_2023_Q2`
- `DIS_2023_Q4`

Across years in the held-out latest window, the gain is not confined to a single post-2024 or 2025 pocket:

- `2023` net gain `≈ 6.05e-09`
- `2024` net gain `≈ 3.47e-09`
- `2025` net gain `≈ 2.71e-10`

So this is still a concentrated signal, but it is not just one tiny end-of-sample artifact.

## Updated Interpretation

This confirmation checkpoint sharpens the story in a productive way.

### What we now know

1. The hardest-question route helps only through a narrow agreement-veto subset.
2. That subset is meaningfully aligned with denser, lower-directness, more evasive `Q&A` pockets.
3. But it is not equivalent to the hardest global shell-failure slice.
4. The gain is concentrated yet temporally spread across both `2023` and `2024` within the held-out window.

### What that means

The current best reading is:

## hardest-question framing is a narrow local veto signal that partly tracks difficult analyst-interaction pockets, but it is not a generic hardest-case detector

That is a good place to be research-wise:

- it confirms the signal is real and structured,
- it keeps the method story compact,
- and it tells us the next step should focus on **better characterizing the local framing pocket**, not on adding more shell complexity.
