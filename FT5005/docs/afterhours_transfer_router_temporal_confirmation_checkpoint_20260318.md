# After-Hours Transfer Router Temporal Confirmation Checkpoint 2026-03-18

## Purpose

The previous checkpoints established two retained transfer-router candidates on the matched clean `after_hours` aligned-audio subset:

1. the most interpretable route:
   - `hybrid_pair_bench` conservative shallow tree
2. the strongest pooled matched route:
   - `hybrid_plus_text` conservative logistic with `C = 2`

The next question was not whether we could tune them a little more, but whether these gains survive a more honest **temporal confirmation** check.

## Design

New script:

- `scripts/run_afterhours_transfer_router_temporal_confirmation.py`

Retained modeling setting:

- clean `after_hours`
- unseen-ticker evaluation
- low-rank semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- same matched aligned-audio subset as the recent transfer-router checkpoints

Compared routes:

1. `hybrid_pair_bench` conservative tree
   - depth `2`
   - minimum leaf fraction `0.08`
2. `hybrid_plus_text` conservative logistic
   - `C = 2`

Temporal confirmation splits:

- `val2020_test_post2020`
  - train `<= 2019`
  - validate on `2020`
  - test on `2021+`
- `val2021_test_post2021`
  - train `<= 2020`
  - validate on `2021`
  - test on `2022+`
- `val2022_test_post2022`
  - train `<= 2021`
  - validate on `2022`
  - test on `2023+`

Outputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/afterhours_transfer_router_temporal_confirmation_overview.csv`
- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/afterhours_transfer_router_temporal_confirmation_summary.json`

## Main Findings

### 1. The strongest matched transfer route does **not** dominate every temporal split

The tuned matched benchmark had made `hybrid_plus_text` logistic look like the best pooled transfer route.

That is still true for the latest split:

- `val2022_test_post2022`
  - pre-call market only `≈ 0.998482`
  - retained semantic+audio expert `≈ 0.998527`
  - selected expert `≈ 0.998546`
  - hybrid pair tree `≈ 0.998624`
  - **hybrid plus-text logistic `≈ 0.998636`**

But the earlier temporal confirmation split looks different:

- `val2020_test_post2020`
  - pre-call market only `≈ 0.996679`
  - retained semantic+audio expert `≈ 0.996678`
  - hybrid pair tree `≈ 0.996676`
  - **hybrid plus-text logistic `≈ 0.996636`**

So the best late-period matched result does not automatically back-propagate into an across-time headline claim.

### 2. The hybrid routes still help relative to the retained transfer experts in the later windows

The middle temporal split is the cleanest positive confirmation for routing **within** the transfer extension:

- `val2021_test_post2021`
  - retained semantic+audio expert `≈ 0.998781`
  - selected expert `≈ 0.998782`
  - hybrid pair tree `≈ 0.998790`
  - **hybrid plus-text logistic `≈ 0.998821`**

And the logistic route is materially better than the retained transfer experts on that split:

- versus retained semantic+audio expert:
  - paired `p(MSE) = 0.0`
- versus selected expert:
  - paired `p(MSE) = 0.0`

But even there, it still does **not** beat the hardest baseline:

- pre-call market only `≈ 0.998826`

So the safer reading is:

- hybrid routing can improve the **transfer extension**
- but it still does not uniformly overturn the strongest pre-call market baseline across temporal windows

### 3. The shallow tree is the more conservative and more stable route

Across the three temporal confirmation splits:

- hybrid pair tree beats the selected expert in `3 / 3` splits
- hybrid pair tree beats the retained semantic+audio expert in `2 / 3` splits
- hybrid plus-text logistic beats the selected expert in `2 / 3` splits
- hybrid plus-text logistic beats the retained semantic+audio expert in `2 / 3` splits

So the tree route remains the more stable interpretable choice, while the logistic route remains the higher-upside but more time-sensitive pooled route.

### 4. Pre-call market-only remains the hardest temporal confirmation baseline

This is the main integrity check from the benchmark.

Across the three temporal confirmation windows:

- pre-call market only is best in the earliest split,
- pre-call market only is still slightly best in the middle split,
- only the latest split reproduces the matched-period transfer-router ordering where the hybrid routes move clearly ahead.

That means the paper-safe interpretation should remain conservative:

- the transfer-router idea is promising,
- the hybrid supervision pattern is real,
- but the strongest claim is still about **improving the transfer-side extension relative to weaker transfer experts**, not about fully replacing the strongest pre-call market baseline across time.

## Updated Interpretation

This checkpoint is valuable because it sharpens the research story rather than just adding another score:

- hybrid supervision is still the right signal source for transfer routing,
- the shallow tree is still the cleanest interpretable route,
- the tuned plus-text logistic route is still the strongest **late-period matched** route,
- but the whole transfer-router story is **temporally sensitive**.

So the evidence is now stronger and more honest at the same time:

- the router is not just a one-split curiosity,
- but it is also not yet a uniformly dominant temporal model.

## Practical Consequence

The safe repo conclusion is now:

1. keep the fixed-split semantic headline unchanged,
2. keep hybrid supervision as the transfer-router signal source,
3. keep the shallow hybrid pair tree as the most interpretable transfer route,
4. keep the tuned hybrid-plus-text logistic route as the best late-period matched transfer route,
5. and explicitly state that the transfer-router extension still does **not** consistently beat `pre_call_market_only` under broader temporal confirmation.

So the next extension step should not be more router family expansion.

Instead, the right next gate is:

- either confirm the same idea on a broader or less matched slice,
- or move toward a stronger signal source rather than a more elaborate router.
