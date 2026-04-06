# After-Hours Transfer Agreement Refinement Checkpoint 2026-03-18

## Purpose

The last learnable-trust checkpoint asked whether the whole hard abstention rule could be replaced by a compact learnable trust layer.

That broader replacement did **not** beat hard abstention on the held-out latest window.

So the next cleaner question is narrower:

## if disagreement already falls back to market, can the **agreement side alone** be improved by a compact learnable refinement?

This is a better-posed test because it keeps the current conservative structure fixed:

- disagreement still abstains to `pre_call_market_only`
- only agreement events are learnably refined

That makes the benchmark:

- more interpretable,
- more targeted,
- and less vulnerable to method sprawl.

## Design

New script:

- `scripts/run_afterhours_transfer_agreement_refinement.py`

Source inputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`

Agreement path:

- if pair-tree and plus-text logistic agree, use the agreed transfer expert
- otherwise keep the current disagreement fallback:
  - `pre_call_market_only`

So this checkpoint only learns whether an **agreement event** should trust the agreed expert or stay with the market baseline.

Two compact agreement refinements are tested:

1. **agreement gain gate**
   - predict whether the agreed expert beats `pre_call_market_only`
   - use the agreed expert only when predicted gain is positive
2. **agreement soft trust**
   - predict a trust weight `λ ∈ [0, 1]`
   - interpolate between `pre_call_market_only` and the agreed expert

Feature set is still small and geometry-only:

- `agreed_choose_qa`
- `agreed_minus_pre_pred`
- `qa_minus_pre_pred`
- `sem_minus_pre_pred`
- `selected_minus_pre_pred`
- `pair_minus_pre_pred`
- `logistic_minus_pre_pred`
- `abs_pair_minus_logistic_pred`

Temporal protocol:

- train on `val2020_test_post2020` agreement events
- tune on `val2021_test_post2021` agreement events
- refit on `2020 + 2021` agreement events
- test on full `val2022_test_post2022`, with disagreement still fixed to `pre_call_market_only`

Outputs:

- `results/afterhours_transfer_agreement_refinement_role_aware_audio_lsa4_real/afterhours_transfer_agreement_refinement_summary.json`
- `results/afterhours_transfer_agreement_refinement_role_aware_audio_lsa4_real/afterhours_transfer_agreement_refinement_tuning.csv`
- `results/afterhours_transfer_agreement_refinement_role_aware_audio_lsa4_real/afterhours_transfer_agreement_refinement_test_predictions.csv`
- `results/afterhours_transfer_agreement_refinement_role_aware_audio_lsa4_real/afterhours_transfer_agreement_refinement_gain_coefficients.csv`
- `results/afterhours_transfer_agreement_refinement_role_aware_audio_lsa4_real/afterhours_transfer_agreement_refinement_soft_coefficients.csv`

## Main Findings

### 1. Agreement refinement looks a little better on the `2021` tuning window

Validation agreement subset:

- `pre_call_market_only ≈ 0.997191`
- `agreement_supported_pred ≈ 0.997162`
- best agreement gain gate `≈ 0.997331`
- best agreement soft trust `≈ 0.997200`

So a learnable agreement-side refinement does look somewhat promising on the intermediate window.

### 2. But on the held-out latest agreement subset, the raw agreed expert is already basically as good as it gets

Held-out `2022` agreement subset:

- `pre_call_market_only ≈ 0.998418`
- **`agreement_supported_pred ≈ 0.998591`**
- agreement gain gate `≈ 0.998590`
- agreement soft trust `≈ 0.998453`

So the gain gate nearly matches the agreed expert, but does **not** improve it.
The soft trust version is clearly worse.

### 3. On the full latest window, hard abstention still remains best

Held-out full `2022` split:

- `pre_call_market_only ≈ 0.998482`
- `agreement_supported_pred ≈ 0.998624`
- **`agreement_pre_only_abstention ≈ 0.998640`**
- agreement gain gate `≈ 0.998639`
- agreement soft trust `≈ 0.998514`

So even this more focused learnable refinement still does **not** beat the current hard abstention route.

The gain gate gets very close:

- it uses the agreed expert on about `0.50` of latest agreement events
- but it remains slightly below hard abstention
- paired `p(MSE)` vs hard abstention `≈ 0.450`

### 4. The soft trust variant again under-trusts the useful correction

Held-out latest agreement subset:

- mean predicted `λ ≈ 0.118`
- optimal mean `λ ≈ 0.285`

So the soft trust model stays conservative, but ends up shrinking the useful agreement correction too much.

That is consistent with the broader trust-calibrator checkpoint:

- soft blending currently looks less effective than keeping a hard conservative decision boundary.

### 5. The learned agreement model is still simple and interpretable

Largest gain-gate coefficients:

- `pair_minus_pre_pred`
- `logistic_minus_pre_pred`
- `agreed_minus_pre_pred`
- `sem_minus_pre_pred`

The agreement-side geometry is doing almost all the work.

This is useful because it again points to:

- **prediction geometry around the market baseline**

as the stable object,

not a need for larger routing stacks or more feature families.

## Updated Interpretation

This checkpoint is another good negative result.

### What it adds

1. We did not just test a broad learnable trust replacement.
2. We also tested the narrower and more methodologically fair version:
   - keep disagreement abstention fixed,
   - refine only agreement events.
3. Even then, the learnable agreement refinement still does **not** beat hard abstention on the held-out latest split.

### What it means

The current transfer-side story gets cleaner again:

- disagreement already wants conservative market fallback
- agreement already contains the useful transfer lift
- and a compact learnable agreement-side refinement does not yet improve on simply trusting the agreement route

So the best current transfer-side interpretation remains:

## reliability-aware abstention is still the strongest compact transfer method in the repo

not because we failed to try learnable refinements,

but because the targeted learnable refinement still does not outperform the simpler abstention rule.

## Practical Consequence

This result argues against spending the next round on more router refinement.

The more coherent next move is:

1. stop squeezing more performance out of the current transfer routing shell,
2. keep the current abstention logic as the anchor,
3. and search instead for a **stronger compact transferable signal source** that could materially change the geometry upstream.
