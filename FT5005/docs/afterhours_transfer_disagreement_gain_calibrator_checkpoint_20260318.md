# After-Hours Transfer Disagreement Gain Calibrator Checkpoint 2026-03-18

## Purpose

After the disagreement-slice diagnostic, the next methodological question was not:

- add another hand-built rule,
- or keep branching into more router families.

The more research-clean question is:

## can the small latest-window `Q&A` pocket be captured by a **compact learnable model**?

That is a better test because it is:

- less rule-based,
- more interpretable,
- more transferable,
- and more faithful to a coherent scientific story.

## Design

New script:

- `scripts/run_afterhours_transfer_disagreement_gain_calibrator.py`

Input source:

- `results/afterhours_transfer_disagreement_slice_diagnostics_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_slice_event_rows.csv`

Model idea:

- fit a **compact ridge gain regressor** on disagreement events only
- predict `qa_vs_pre_mse_gain`
- choose `Q&A` only when the predicted gain is positive, otherwise keep `pre_call_market_only`

Feature set is intentionally small and interpretable:

- `qa_minus_pre_pred`
- `sem_minus_pre_pred`
- `sem_minus_qa_pred`
- `abs_sem_minus_qa_pred`
- `tree_choose_qa`
- `logistic_choose_qa`

Temporal protocol:

- train on `val2020_test_post2020`
- tune `alpha` on `val2021_test_post2021`
- refit on `2020 + 2021`
- test on `val2022_test_post2022`

Outputs:

- `results/afterhours_transfer_disagreement_gain_calibrator_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_gain_calibrator_summary.json`
- `results/afterhours_transfer_disagreement_gain_calibrator_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_gain_calibrator_tuning.csv`
- `results/afterhours_transfer_disagreement_gain_calibrator_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_gain_calibrator_test_predictions.csv`
- `results/afterhours_transfer_disagreement_gain_calibrator_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_gain_calibrator_coefficients.csv`

## Main Findings

### 1. The compact learnable calibrator does **not** recover the latest `Q&A` pocket

Selected alpha:

- `0.01`

Validation (`2021`) result:

- compact calibrator `≈ 0.998896`
- `pre_call_market_only ≈ 0.998896`
- `qa_benchmark_svd ≈ 0.998895`

So even at validation time the learnable calibrator does not beat the conservative market fallback.

### 2. On the held-out latest disagreement slice, it collapses back to pure market fallback

Held-out `2022` disagreement slice:

- `pre_call_market_only ≈ 0.999042`
- `qa_benchmark_svd ≈ 0.999079`
- **compact learnable calibrator `≈ 0.999042`**

The calibrator chooses `Q&A` on:

- **`0 / 16` test events**

So the learnable model simply reverts to the market baseline on the latest disagreement slice.

### 3. This is actually an informative negative result

The latest disagreement diagnostic had already shown:

- the `Q&A` edge is tiny,
- concentrated,
- and localized.

This new benchmark sharpens that further:

## the latest `Q&A` pocket is not learnable / transferable from earlier disagreement geometry

At least not with a compact, interpretable, temporally sensible model.

That is valuable because it means we should **not** respond by hard-coding a new disagreement exception rule.

### 4. The learned coefficients are simple and interpretable

Largest-magnitude coefficients:

- `qa_minus_pre_pred`
- `sem_minus_qa_pred`
- `sem_minus_pre_pred`
- `abs_sem_minus_qa_pred`

Router identity features (`tree_choose_qa`, `logistic_choose_qa`) carry much smaller weight.

So even the learnable model says the decision is mostly driven by:

- prediction geometry between the experts,
- not by which router happened to vote for `Q&A`.

That is a cleaner and more transferable finding than another pocket-specific rule.

## Updated Interpretation

This checkpoint makes the transfer-side story cleaner in an important way.

### What we now know

1. The latest disagreement slice does contain a small `Q&A`-friendly pocket.
2. That pocket is real but tiny and concentrated.
3. A compact learnable gain calibrator trained on earlier windows does **not** recover it.
4. Therefore the pocket is currently **not a stable transferable signal**.

So the correct scientific reaction is:

- **do not overfit to the pocket**
- **do not promote it into a new global rule**

## Practical Consequence

This is a strong negative result in a good sense.

It supports the cleaner story:

- keep **reliability-aware abstention** as the main transfer-side extension,
- keep `pre_call_market_only` as the safest global fallback on disagreement,
- and do not replace it with a new `Q&A` exception rule unless a later, broader, more learnable signal appears.

So this benchmark is very aligned with a good research story:

- fewer ad hoc rules,
- fewer piled modules,
- more compact learning,
- and honest acknowledgement of what currently transfers and what does not.

## Best Next Step

Given this result, the most coherent next move is **not** more disagreement heuristics.

The next high-value step is now:

1. either build a compact paper-facing scorecard for the current fixed-split + transfer hierarchy,
2. or look for a stronger, more transferable supervision source rather than squeezing the current pocket harder.

Right now, the safer choice is still to strengthen the reporting and keep the transfer-side claim conservative.
