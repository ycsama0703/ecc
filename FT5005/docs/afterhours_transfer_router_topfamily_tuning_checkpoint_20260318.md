# After-Hours Transfer Router Top-Family Tuning Checkpoint 2026-03-18

## Purpose

The signal-family benchmark showed two promising transfer-router directions:

1. the most interpretable shallow-tree route:
   - `hybrid_pair_bench`
2. the best pooled logistic route:
   - `hybrid_plus_text`

The next question was:

- are these results fragile to router hyperparameters,
- or do the top families remain strong under a small but explicit tuning sweep?

## Design

New script:

- `scripts/run_afterhours_transfer_router_topfamily_tuning.py`

It tunes the two strongest families from the previous checkpoint:

### A. `hybrid_pair_bench` shallow tree

Swept:

- tree depth: `1, 2, 3, 4`
- minimum leaf fraction: `0.05, 0.08, 0.12, 0.16`

### B. `hybrid_plus_text` conservative logistic router

Swept:

- logistic `C`: `0.1, 0.25, 0.5, 1, 2, 4, 8`

Retained evaluation setting:

- clean `after_hours`
- unseen-ticker evaluation
- low-rank semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- same matched aligned-audio subset as the recent transfer checkpoints

Outputs:

- `results/afterhours_transfer_router_topfamily_tuning_role_aware_audio_lsa4_real/afterhours_transfer_router_topfamily_tuning_summary.json`
- `results/afterhours_transfer_router_topfamily_tuning_role_aware_audio_lsa4_real/afterhours_transfer_router_topfamily_tuning_overview.csv`
- plus one subdirectory per tuned configuration with the underlying conservative-router outputs

## Main Findings

### 1. The best pooled result in the repo now comes from the tuned hybrid-plus-text logistic router

Best pooled matched unseen-ticker result:

- **`hybrid_plus_text` conservative logistic with `C = 2`**
- `R^2 ≈ 0.998636`
- `RMSE ≈ 1.238e-4`

That improves on the earlier untuned hybrid-plus-text logistic result:

- earlier `C = 1` result `≈ 0.998628`
- tuned `C = 2` result `≈ 0.998636`

So the strongest current pooled transfer-side routing result in the repo is now:

- **hybrid-plus-text conservative logistic, `C = 2`**

### 2. The hybrid pair-plus-benchmark tree is very stable

The `hybrid_pair_bench` shallow tree is not just good — it is stable.

Depth `2` is best:

- `R^2 ≈ 0.998624`
- paired `p(MSE)` vs retained semantic+audio expert `≈ 0.052`

But depth `3` and `4` stay almost identical:

- `R^2 ≈ 0.998624`

And the minimum leaf fraction barely matters in practice.

So the tree result is not depending on a delicate leaf-size choice.

That is helpful because it makes the tree route easier to defend as a real method pattern rather than a tuning accident.

### 3. The logistic family has a clearer optimum

The `hybrid_plus_text` logistic route improves steadily as `C` rises from `0.1` to `2`,
then softens again at `4` and `8`.

Best logistic settings:

- `C = 2` gives the best pooled result
- `C = 4` is close but slightly weaker
- lower `C` values are consistently worse

So the logistic route has a clear retained configuration now:

- **`hybrid_plus_text` with `C = 2`**

### 4. The current best tree and best logistic tell a coherent story

The tuning sweep reinforces the same broader interpretation:

- the best tree still comes from hybrid pair-plus-benchmark supervision
- the best pooled logistic still comes from the hybrid-plus-text family

So the ranking is stable:

- pair-only is too weak
- benchmark-only is too weak
- hybrid supervision remains the right transfer-side signal source

## Updated Interpretation

This checkpoint improves the current transfer extension in two different ways:

1. it slightly raises the best pooled result,
2. and it shows that the strongest families are not purely one-off tuning artifacts.

That makes the transfer-side idea more mature:

- **conservative routing**
- guided by **hybrid reliability supervision**
- with a stable shallow-tree route and a stronger pooled logistic route

## Practical Consequence

The safe repo conclusion is now:

1. keep the fixed-split semantic headline unchanged,
2. keep hybrid supervision as the transfer-router signal source,
3. treat `hybrid_pair_bench` tree as the cleanest interpretable route,
4. treat `hybrid_plus_text` logistic with `C = 2` as the strongest pooled matched transfer result,
5. and still avoid overclaiming because the tuned gains remain exploratory rather than fully robust submission-ready evidence.

So the transfer extension path is now materially sharper:

- **hybrid supervised conservative routing** is the strongest current direction,
- and the tuned `C = 2` logistic route is the best pooled result we have on this matched transfer subset.
