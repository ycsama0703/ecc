# After-Hours Transfer Q&A-Signal Checkpoint 2026-03-17

## Purpose

After the transfer reliability-gate search, the next question was narrower:

- if the simple `a4_strict_row_share` gate is already the right retained transfer mechanism,
- can stronger event-level `Q&A` quality or evasion signals improve the signal **inside** that gate?

This checkpoint tests that directly on the same matched transfer slice instead of changing the gate again.

## Design

New script:

- `scripts/run_afterhours_transfer_qa_signal_benchmark.py`

Run setting:

- clean `after_hours`
- unseen-ticker evaluation
- low-rank transfer semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- fixed simple observability gate on `a4_strict_row_share`

Compared transfer branches:

1. retained low-rank semantic line: `pre_call_market + A4 + qna_lsa`
2. compact weak-label `Q&A` quality core
3. compressed `qa_benchmark v2` block
4. semantic plus each weak-label block
5. semantic plus role-aware audio
6. semantic plus weak-label block plus role-aware audio

Outputs:

- `results/afterhours_transfer_qa_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_qa_signal_summary.json`
- `results/afterhours_transfer_qa_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_qa_signal_predictions.csv`
- `results/afterhours_transfer_qa_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_qa_signal_thresholds.csv`

## Main Findings

### 1. The retained best transfer branch does not change

Overall matched unseen-ticker `R^2` still ranks as:

- **observability-gated `qna_lsa + role_aware_audio_svd8` ≈ `0.998527`**
- `pre_call_market_only ≈ 0.998482`
- observability-gated `qna_lsa` alone `≈ 0.998434`
- observability-gated `qa_benchmark_svd` alone `≈ 0.998414`

So the current retained best matched transfer extension is still:

- low-rank `qna_lsa`
- plus role-aware audio
- under the simple local `A4` observability gate

### 2. Event-level weak-label `Q&A` quality blocks do not replace the retained semantic line

The two direct quality/evasion replacements are weaker:

- gated `qa_quality_core ≈ 0.997896`
- gated `qa_benchmark_svd ≈ 0.998414`
- gated `qna_lsa ≈ 0.998434`

So even on this transfer-friendly matched subset, the current weak-label event-level quality/evasion summaries do **not** beat the retained low-rank semantic text block.

### 3. Adding weak-label `Q&A` blocks on top of the retained semantic or semantic+audio line hurts

Adding the quality/evasion blocks to the current transfer line moves performance down rather than up:

- gated `qna_lsa + qa_quality_core ≈ 0.998144`
- gated `qna_lsa + qa_benchmark_svd ≈ 0.998342`
- gated `qna_lsa + qa_quality_core + role_aware_audio ≈ 0.998219`
- gated `qna_lsa + qa_benchmark_svd + role_aware_audio ≈ 0.998308`
- retained gated `qna_lsa + role_aware_audio ≈ 0.998527`

The strongest deterioration is also statistically clearer than the earlier tiny positive audio hint:

- versus the retained best branch, adding `qa_quality_core + audio` gives paired `p(MSE) ≈ 0.001`
- adding `qa_quality_core` without audio gives paired `p(MSE) ≈ 0.004`
- adding `qa_benchmark_svd + audio` gives paired `p(MSE) ≈ 0.014`

So the weak-label additions are not just “not better”; several are measurably worse on this slice.

### 4. There is some cross-ticker complementarity, but not a robust upgrade

One nuance is still worth keeping in mind:

- gated `qa_benchmark_svd` beats `pre_call_market_only` on `5 / 9` held-out tickers
- and beats the retained gated semantic+audio branch on `6 / 9` held-out tickers

But its pooled score still falls below both because the losing tickers hurt more than the winning tickers help.

So the current weak-label `Q&A` block may contain **heterogeneous local signal**, but the simple direct add-on path is not a credible overall improvement.

## Updated Interpretation

This checkpoint sharpens the transfer-side story again.

What works best is still:

- a simple observability gate,
- low-rank `Q&A` semantics,
- and role-aware aligned audio as a small exploratory extension.

What does **not** currently work is:

- replacing that line with heuristic event-level `Q&A` quality/evasion summaries,
- or stacking those summaries on top of the retained transfer branch.

So the next novelty move should not be “more heuristic event-level `qa_benchmark` stacking.”

It should be something narrower and stronger, such as:

1. transferred pair-level directness / evasion / answer-quality supervision,
2. or a very selective expert-routing rule that only uses those signals where they validate cleanly.

For now, the safe repo conclusion is:

- **keep the simple gate, keep the retained low-rank semantic+audio branch, and do not promote the current weak-label `Q&A` add-ons into the main transfer story.**
