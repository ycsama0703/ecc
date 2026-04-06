# After-Hours Transfer Pair-Tail Question-Framing Diagnostics Checkpoint 2026-03-22

## Purpose

The latest pair-tail research line now says:

- the top-1 hardest analyst-question text is the first local semantic refinement to edge past hard abstention,
- the effect survives an encoding check,
- and it operates entirely through a narrow agreement-veto subset.

That naturally raises a deeper interpretation question:

## what kind of local analyst-question framing lives inside that veto pocket?

This is a diagnostic step, not a new method proposal.
Its purpose is to characterize the local pocket more clearly without adding new routing complexity or inventing new rules.

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_framing_diagnostics.py`

Inputs:

- latest held-out predictions from `results/afterhours_transfer_pair_tail_text_benchmark_lsa4_real/afterhours_transfer_pair_tail_text_benchmark_test_predictions.csv`
- hardest-question text views from `results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv`

The script focuses only on the `agreement_veto` rows for the current hardest-question route and splits them into:

- positive veto rows (`mse_gain_vs_hard > 0`)
- negative veto rows (`mse_gain_vs_hard < 0`)
- flat veto rows (`mse_gain_vs_hard = 0`)

It then reports:

1. compact surface-form statistics
2. distinctive uni/bi/tri-grams for positive vs negative veto questions
3. top positive / negative event examples

Outputs:

- `results/afterhours_transfer_pair_tail_question_framing_diagnostics_real/afterhours_transfer_pair_tail_question_framing_summary.json`
- `results/afterhours_transfer_pair_tail_question_framing_diagnostics_real/afterhours_transfer_pair_tail_question_framing_rows.csv`
- `results/afterhours_transfer_pair_tail_question_framing_diagnostics_real/afterhours_transfer_pair_tail_question_framing_positive_ngrams.csv`
- `results/afterhours_transfer_pair_tail_question_framing_diagnostics_real/afterhours_transfer_pair_tail_question_framing_negative_ngrams.csv`

## Main Findings

### 1. The active veto pocket is very small and mixed

Inside the `agreement_veto` slice:

- total veto rows: `20`
- positive veto rows: `9`
- negative veto rows: `4`
- flat veto rows: `7`

So this remains a very small local pocket.
That means the resulting framing interpretation should be treated as a **diagnostic clue**, not a stable new rule.

### 2. Positive veto questions look more interactive, clarificatory, and numerically grounded

Compared with negative veto rows, positive veto rows show:

- higher `first_person_share` (`≈ 0.030` vs `≈ 0.020`)
- higher `second_person_share` (`≈ 0.039` vs `≈ 0.035`)
- higher `hedge_share` (`≈ 0.035` vs `≈ 0.016`)
- higher `followup_marker_count` (`≈ 0.556` vs `≈ 0.500`)
- substantially higher `numeric_token_share` (`≈ 0.019` vs `≈ 0.004`)
- shorter questions on average (`≈ 103` tokens vs `≈ 134`)

The most distinctive positive-side ngrams include:

- `wondering`
- `wondering if`
- `if you could`
- `i think`
- `color`
- `you guys`
- `around`
- `million`

Taken together, these look less like pure topic markers and more like:

## follow-up / clarification / modeling language
## where the analyst is asking management to help frame or quantify the situation

### 3. Negative veto questions look more strategic-structural, but this is partly topic contaminated

The negative-side ngrams are dominated by:

- `disney`
- `platform`
- `sports`
- `assets`
- `content`
- `the next`
- `u s`

This is still informative, but it also carries an obvious warning:

## the negative pocket is partly contaminated by company/topic concentration

In other words, the local framing difference is not cleanly separable from issuer/topic identity on the negative side.
So this diagnostic should be interpreted cautiously.

### 4. The event examples support a clarificatory / model-building reading

Largest positive examples include:

- `AAPL_2023_Q4`
- `NVDA_2023_Q1`
- `AMZN_2024_Q3`
- `AMZN_2024_Q1`
- `NKE_2023_Q1`

Their hardest questions often contain language like:

- `follow-up`
- `how do I think about ...`
- `can you give color ...`
- `wondering if ...`
- multi-part but still explanatory / framing-oriented prompts

By contrast, negative examples such as:

- `DIS_2023_Q2`
- `DIS_2023_Q3`
- `DIS_2023_Q4`
- `NKE_2023_Q2`

look more like strategic-asset or policy-choice probes, often around:

- integration choices
- platform structure
- asset separation
- channel / promotion state

So the current local pocket seems closer to:

## analyst attempts to refine an internal operating model

than to broad strategic or asset-portfolio interrogation.

## Updated Interpretation

This diagnostic helps the story mature without over-claiming.

### What we now know

1. The new hardest-question signal still lives in a very small veto pocket.
2. Positive veto questions look more clarificatory, interactive, and numerically grounded.
3. Negative veto questions look more strategic-structural, but that side is partly topic-contaminated.
4. So the signal is better described as a **local analyst-framing pocket** than as generic topic semantics.

### What that means

The current best reading is:

## the hardest-question signal is most promising when analysts ask management to help quantify, frame, or reconcile a difficult local operating picture

That is a useful research result because it keeps the story coherent:

- no new shell complexity,
- no blind feature piling,
- and a more interpretable explanation of what the new transfer-side signal is actually picking up.

At the same time, the sample is still small enough that this should remain a **diagnostic interpretation**, not a promoted new rule or method contribution.
