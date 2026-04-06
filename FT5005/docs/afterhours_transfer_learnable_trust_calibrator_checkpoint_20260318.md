# After-Hours Transfer Learnable Trust Calibrator Checkpoint 2026-03-18

## Purpose

After the disagreement gain-calibrator result, the next natural question was broader:

- not just whether a tiny disagreement pocket can be learned,
- but whether the whole **agreement / abstention** transfer story can be replaced by a compact learnable trust model.

That is a cleaner research question because it asks whether we can move from:

- a hand-coded abstention rule

to

- a small, interpretable, temporally trained trust calibrator built only on prediction geometry.

If that works, it would be a more learnable and potentially more transferable transfer-side method.
If it fails, that is also valuable, because it tells us the current hard abstention rule is not just a placeholder.

## Design

New script:

- `scripts/run_afterhours_transfer_learnable_trust_calibrator.py`

Source inputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`

We build a compact transfer candidate from the two strongest temporal router families:

1. if pair-tree and plus-text logistic **agree**, use the agreed expert
2. if they **disagree**, use the average of the two router outputs

That gives a smooth candidate route:

- `agreement_avg_transfer_candidate`

We then compare it with the current best conservative rule:

- `agreement_pre_only_abstention`

which uses the agreed expert only under agreement and otherwise falls back to `pre_call_market_only`.

Two compact learnable calibrators are tested:

1. **gain gate**
   - predict whether the candidate beats `pre_call_market_only`
   - choose candidate only when predicted gain is positive
2. **soft trust**
   - predict a trust weight `λ ∈ [0, 1]`
   - interpolate between `pre_call_market_only` and the candidate correction

Feature set is deliberately compact and geometry-only:

- `agreement`
- `candidate_minus_pre_pred`
- `qa_minus_pre_pred`
- `sem_minus_pre_pred`
- `selected_minus_pre_pred`
- `tree_minus_pre_pred`
- `logistic_minus_pre_pred`
- `abs_pair_minus_logistic_pred`

Temporal protocol:

- train on `val2020_test_post2020`
- tune on `val2021_test_post2021`
- refit on `2020 + 2021`
- test on `val2022_test_post2022`

Outputs:

- `results/afterhours_transfer_learnable_trust_calibrator_role_aware_audio_lsa4_real/afterhours_transfer_learnable_trust_calibrator_summary.json`
- `results/afterhours_transfer_learnable_trust_calibrator_role_aware_audio_lsa4_real/afterhours_transfer_learnable_trust_calibrator_tuning.csv`
- `results/afterhours_transfer_learnable_trust_calibrator_role_aware_audio_lsa4_real/afterhours_transfer_learnable_trust_calibrator_test_predictions.csv`
- `results/afterhours_transfer_learnable_trust_calibrator_role_aware_audio_lsa4_real/afterhours_transfer_learnable_trust_calibrator_gate_coefficients.csv`
- `results/afterhours_transfer_learnable_trust_calibrator_role_aware_audio_lsa4_real/afterhours_transfer_learnable_trust_calibrator_soft_coefficients.csv`

## Main Findings

### 1. Both compact learnable calibrators look mildly promising on the `2021` tuning window

Validation (`2021`) reference:

- `pre_call_market_only ≈ 0.998826`
- `agreement_avg_transfer_candidate ≈ 0.998807`
- `agreement_pre_only_abstention ≈ 0.998823`

Best learnable validation scores:

- gain gate (`alpha = 100`) `≈ 0.998841`
- soft trust (`alpha = 10`) `≈ 0.998837`

So on the middle window, both compact learnable variants look slightly better than the conservative hard abstention rule.

### 2. But on the held-out latest window, hard abstention still wins

Held-out `2022` result:

- `pre_call_market_only ≈ 0.998482`
- `agreement_avg_transfer_candidate ≈ 0.998639`
- **`agreement_pre_only_abstention ≈ 0.998640`**
- compact gain gate `≈ 0.998637`
- compact soft trust `≈ 0.998537`

So the broader learnable replacement story does **not** beat the current hard abstention rule.

### 3. The gain gate gets close, but does not improve on hard abstention

The gain gate is the cleaner of the two learnable variants:

- it keeps candidate usage at about `0.60` on the held-out latest split
- it lands almost exactly on top of the smooth transfer candidate
- but it still sits slightly below hard abstention

Paired test against hard abstention:

- `p(MSE) ≈ 0.221`

So there is no evidence that the learnable gain gate is a more reliable replacement for the current abstention rule.

### 4. The soft trust model is more learnable in spirit, but worse in practice

Held-out latest split:

- soft trust `≈ 0.998537`
- hard abstention `≈ 0.998640`

The soft trust model stays conservative:

- mean predicted `λ ≈ 0.138`
- max predicted `λ ≈ 0.511`

But that soft interpolation appears to **wash out** the useful late-window correction rather than sharpen it.

So the clean message here is:

- a softer continuous trust blend is not currently a better transfer strategy than hard abstention.

### 5. The learned models are still interpretable

The gain gate puts its largest weight on:

- `abs_pair_minus_logistic_pred`
- `tree_minus_pre_pred`
- `qa_minus_pre_pred`
- `candidate_minus_pre_pred`

The soft trust model puts its largest weight on:

- `tree_minus_pre_pred`
- `selected_minus_pre_pred`
- `candidate_minus_pre_pred`
- `sem_minus_pre_pred`

So even this broader learnable benchmark still says the important object is:

- **prediction geometry between experts and the market baseline**

not another large feature pile or another deep routing stack.

## Updated Interpretation

This is a useful negative result in a strong sense.

### What it adds

1. We now tested a broader, less rule-based, more learnable alternative to hard abstention.
2. That alternative can look slightly better on an intermediate window.
3. But on the held-out latest window, it still does **not** beat the current hard abstention rule.
4. Softer trust blending is currently worse still.

### What stays true

The best current transfer-side reading remains:

- use transfer correction when reliability evidence is strong,
- otherwise stay conservative,
- and do not assume that a more learnable-looking trust layer automatically transfers better.

So the current repo evidence still supports:

## reliability-aware abstention is the best compact transfer-side method we currently have

not because we failed to try a learnable alternative,

but because the compact learnable alternative does not yet beat it on the held-out late window.

## Practical Consequence

This checkpoint is well aligned with a disciplined research direction:

- fewer ad hoc fixes,
- no blind method stacking,
- explicit tests of learnability,
- and honest retention of the simplest strategy that still holds up best.

So the safest next move is **not** to keep adding new routing machinery.

The more coherent next move is:

1. keep the current abstention logic as the transfer-side anchor,
2. keep looking for a stronger compact transferable signal source,
3. and only promote a learnable trust layer if it eventually wins on held-out time windows rather than just on tuning windows.
