# After-Hours Transfer Router Consensus Checkpoint 2026-03-18

## Purpose

The recent transfer-router checkpoints left us with a clean tension:

- the shallow hybrid tree is the more interpretable route,
- the hybrid-plus-text logistic route has the higher upside on later windows,
- but neither one is clearly dominant across all temporal confirmation splits.

After the complementary-expert benchmark also came back negative, the next useful question became:

- can we get a **more temporally stable transfer route**
- by requiring agreement between the two retained router families,
- rather than by building more complex experts?

## Design

New script:

- `scripts/run_afterhours_transfer_router_consensus_confirmation.py`

Source inputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`

Retained source routers:

1. `hybrid_pair_bench` conservative tree
2. `hybrid_plus_text` conservative logistic

Same temporal confirmation windows:

- `val2020_test_post2020`
- `val2021_test_post2021`
- `val2022_test_post2022`

Benchmarked new consensus routes:

1. **agreement fallback to selected expert**
   - if tree and logistic agree on the expert, use that expert
   - otherwise fall back to `validation_selected_transfer_expert`
2. **agreement with semantic backbone**
   - if tree and logistic agree on the expert, use that expert
   - otherwise fall back to the retained semantic+audio expert

Outputs:

- `results/afterhours_transfer_router_consensus_confirmation_role_aware_audio_lsa4_real/afterhours_transfer_router_consensus_summary.json`
- `results/afterhours_transfer_router_consensus_confirmation_role_aware_audio_lsa4_real/afterhours_transfer_router_consensus_overview.csv`
- `results/afterhours_transfer_router_consensus_confirmation_role_aware_audio_lsa4_real/afterhours_transfer_router_consensus_predictions.csv`

## Main Findings

### 1. The semantic-backbone consensus is the strongest **pooled temporal** route so far

Across all three temporal confirmation windows combined:

- pre-call market only `≈ 0.997886`
- retained semantic+audio expert `≈ 0.997878`
- selected expert `≈ 0.997879`
- pair tree `≈ 0.997901`
- plus-text logistic `≈ 0.997899`
- agreement fallback to selected `≈ 0.997899`
- **agreement with semantic backbone `≈ 0.997902`**

So the best pooled temporal result in this transfer-confirmation family is now:

- **agreement-based consensus with the retained semantic+audio backbone**

### 2. The consensus route improves the weaker transfer baselines more consistently than it beats the best single router

Pooled paired tests:

- consensus semantic-backbone vs selected expert:
  - `p(MSE) ≈ 0.029`
- consensus semantic-backbone vs retained semantic+audio expert:
  - `p(MSE) ≈ 0.0168`

But it does **not** clearly beat the top single routers:

- versus pair tree:
  - `p(MSE) ≈ 0.844`
- versus plus-text logistic:
  - `p(MSE) ≈ 0.941`

So this route is not a new huge accuracy jump.

Its value is different:

- it gives the best pooled temporal robustness,
- while avoiding dependence on either tree-only or logistic-only behavior.

### 3. The semantic-backbone consensus is much more conservative than the late-window best logistic route

Across all temporal confirmation rows:

- agreement rate `≈ 0.681`
- semantic-backbone override rate `≈ 0.101`
- semantic-backbone QA share `≈ 0.101`

So this route only departs from the retained semantic+audio backbone on about `10%` of events.

That is exactly why it is interesting:

- it is not a heavy new model,
- it is a **small, agreement-triggered correction layer** on top of the retained semantic branch.

### 4. Split-by-split behavior is cleaner than the single-router story

By split:

- `val2020_test_post2020`
  - consensus semantic-backbone `≈ 0.996681`
  - beats pre-call market only `≈ 0.996679`
  - beats retained semantic+audio `≈ 0.996678`
- `val2021_test_post2021`
  - consensus semantic-backbone `≈ 0.998790`
  - still below pre-call market only `≈ 0.998826`
  - but above retained semantic+audio `≈ 0.998781`
- `val2022_test_post2022`
  - consensus semantic-backbone `≈ 0.998618`
  - below tree `≈ 0.998624` and logistic `≈ 0.998636`
  - but still above retained semantic+audio `≈ 0.998527`

So the consensus route:

- beats the selected expert in `3 / 3` splits,
- beats the retained semantic+audio expert in `3 / 3` splits,
- beats pre-call market only in `2 / 3` splits,
- but is not the per-split winner against the best individual router in the latest window.

## Updated Interpretation

This checkpoint sharpens the transfer story in a useful way:

- hybrid supervision still matters,
- but one part of its value is now clearly **agreement-based reliability control**,
- not just individual router sophistication.

The main method pattern is becoming:

- keep the retained semantic+audio branch as the backbone,
- allow stronger transfer deviations only when multiple reliability-aware router views agree.

That is a cleaner and more defensible extension than continuing to widen expert families.

## Practical Consequence

The safe repo conclusion is now:

1. keep the fixed-split semantic headline unchanged,
2. keep the retained semantic+audio branch as the transfer backbone,
3. keep the hybrid tree and hybrid-plus-text logistic routes as the two core router families,
4. add the **agreement-based semantic-backbone consensus** as the strongest current pooled temporal transfer route,
5. and still avoid overclaiming because it does not significantly beat the best single-router alternatives or the pre-call market baseline.

So the transfer-side story is now stronger and cleaner:

- routing remains useful,
- direct complementary-expert expansion still looks weak,
- and the most promising robustness move is a **small agreement-triggered correction layer** rather than more model sprawl.
