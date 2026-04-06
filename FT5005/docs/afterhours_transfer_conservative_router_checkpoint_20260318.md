# After-Hours Transfer Conservative-Router Checkpoint 2026-03-18

## Purpose

The previous event-router checkpoint improved the matched unseen-ticker transfer subset, but it still made some avoidable mistakes:

- the best pooled result came from an event-level router,
- but several held-out tickers were still hurt because the router overrode the fold-level expert choice too aggressively.

So the next question was:

- can we keep the same compact reliability-aware routing idea, but make it **conservative** by defaulting to the fold-level selected expert and overriding it only when the event-level router is confident?

## Design

New script:

- `scripts/run_afterhours_transfer_conservative_router.py`

Retained benchmark setting:

- clean `after_hours`
- unseen-ticker evaluation
- low-rank semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- same matched aligned-audio subset as the recent transfer checkpoints

Retained fold-level default:

- first choose the best expert on validation among:
  - gated `qna_lsa + role_aware_audio`
  - gated `qa_benchmark_svd`
  - `pre_call_market_only`
- then treat that selected expert as the **default** for the held-out ticker

New conservative step:

- fit an event-level router on validation rows,
- but only override the selected expert when the router probability crosses a tuned confidence threshold

Compared conservative routers:

1. logistic conservative override
2. shallow tree conservative override

Retained compact reliability feature set:

- `a4_strict_row_share`
- `a4_strict_high_conf_share`
- `qa_pair_count`
- `qa_bench_direct_answer_share`
- `qa_bench_evasion_score_mean`
- `qa_bench_coverage_mean`
- `aligned_audio__aligned_audio_sentence_count`

Plus compact derived disagreement / activation terms:

- gate activation indicators,
- each expert prediction,
- signed prediction gap,
- absolute prediction gap.

Outputs:

- `results/afterhours_transfer_conservative_router_lite_role_aware_audio_lsa4_real/afterhours_transfer_conservative_router_summary.json`
- `results/afterhours_transfer_conservative_router_lite_role_aware_audio_lsa4_real/afterhours_transfer_conservative_router_predictions.csv`
- `results/afterhours_transfer_conservative_router_lite_role_aware_audio_lsa4_real/afterhours_transfer_conservative_router_details.csv`

## Main Findings

### 1. Conservative tree override is now the best pooled transfer router in the repo

Overall matched unseen-ticker `R^2`:

- retained gated semantic+audio expert `≈ 0.998527`
- validation-selected expert `≈ 0.998546`
- previous non-conservative event tree `≈ 0.998567`
- logistic conservative override `≈ 0.998553`
- **tree conservative override `≈ 0.998623`**

RMSE tells the same story:

- validation-selected expert `≈ 1.278e-4`
- previous non-conservative event tree `≈ 1.269e-4`
- **tree conservative override `≈ 1.244e-4`**

So the strongest current matched transfer result in this routing family is:

- **conservative shallow-tree override on top of the selected transfer expert**

### 2. The conservative framing matters more than simply adding a router

The improvement over the earlier non-conservative tree is meaningful in pooled terms:

- previous event tree `≈ 0.998567`
- **conservative event tree `≈ 0.998623`**

The reason is intuitive:

- do **not** abandon the fold-level expert choice by default,
- only override it when the event-level evidence is strong enough.

That keeps the local transfer gains from routing while reducing some of the avoidable event-level switches.

### 3. The gain is still exploratory rather than statistically secure

Even though this is the best pooled transfer router so far, the lift is still not statistically convincing:

- versus validation-selected expert, paired `p(MSE) ≈ 0.318`
- versus retained gated semantic+audio expert, paired `p(MSE) ≈ 0.165`

So the result is directionally encouraging, but still not strong enough to be promoted to the paper headline.

### 4. The tree keeps the same interpretable story

The median held-out ticker `R^2` also improves slightly:

- validation-selected expert `≈ 0.995916`
- **conservative tree override `≈ 0.995940`**

The repeated tree features stay concentrated on intuitive reliability signals:

- expert disagreement
- `A4` observability / confidence
- `Q&A` coverage / evasion

So the next-step story is still:

- **reliability-aware routing across complementary experts**

rather than:

- bigger feature stacks
- more gate complexity
- or heavier sequence modeling

## Updated Interpretation

This checkpoint makes the transfer extension story cleaner.

The repo now has a progressively stronger sequence of results:

1. simple observability gating beats ungated transfer semantics
2. fold-level expert choice beats brute-force stacking
3. non-conservative event routing slightly beats fold-level choice
4. **conservative event routing** is the best pooled result in the current routing family

That is a coherent method idea:

- trust the default selected expert,
- and only switch when observability, reliability, and expert disagreement jointly justify it.

## Practical Consequence

The safe repo conclusion is now:

1. keep the fixed-split semantic headline unchanged,
2. keep the conservative tree router as the strongest current **exploratory transfer extension**,
3. avoid claiming a robust publishable gain yet because the lift remains non-significant,
4. and if we continue this line, the best next step is stronger supervision for the router rather than a larger feature pile.

So the main contribution still does **not** change, but the extension path is getting clearer:

- **conservative observability-aware / reliability-aware routing across complementary transfer experts**.
