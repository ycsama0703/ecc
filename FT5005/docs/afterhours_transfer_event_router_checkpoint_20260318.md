# After-Hours Transfer Event-Router Checkpoint 2026-03-18

## Purpose

The previous transfer expert-selection checkpoint showed a small but coherent hint:

- the retained gated `qna_lsa + role_aware_audio` expert is still the default best transfer branch,
- but a complementary gated `qa_benchmark_svd` expert helps on several held-out tickers,
- and simple fold-level expert choice beats positive stacking.

That raises the next question:

- can we do better with a **small event-level router** that chooses between the two gated experts using compact reliability features, instead of one expert per ticker?

## Design

New script:

- `scripts/run_afterhours_transfer_event_router.py`

Retained benchmark setting:

- clean `after_hours`
- unseen-ticker evaluation
- low-rank semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- same matched aligned-audio subset as the recent transfer checkpoints

Compared transfer experts:

1. retained gated `qna_lsa + role_aware_audio`
2. complementary gated `qa_benchmark_svd`

Compared routers:

1. retained semantic+audio expert as-is
2. previous validation-selected expert choice
3. event-level logistic router on the two gated experts
4. event-level shallow decision-tree router on the two gated experts

Retained compact reliability feature set:

- `a4_strict_row_share`
- `a4_strict_high_conf_share`
- `qa_pair_count`
- `qa_bench_direct_answer_share`
- `qa_bench_evasion_score_mean`
- `qa_bench_coverage_mean`
- `aligned_audio__aligned_audio_sentence_count`

The router also sees simple engineered disagreement / activation terms derived from the two experts:

- gate activation indicators,
- each expert prediction,
- signed prediction gap,
- absolute prediction gap.

Outputs:

- `results/afterhours_transfer_event_router_lite_role_aware_audio_lsa4_real/afterhours_transfer_event_router_summary.json`
- `results/afterhours_transfer_event_router_lite_role_aware_audio_lsa4_real/afterhours_transfer_event_router_predictions.csv`
- `results/afterhours_transfer_event_router_lite_role_aware_audio_lsa4_real/afterhours_transfer_event_router_details.csv`

## Main Findings

### 1. The shallow event tree is the best pooled router in this family

Overall matched unseen-ticker `R^2`:

- `pre_call_market_only ≈ 0.998482`
- retained gated semantic+audio expert `≈ 0.998527`
- validation-selected expert `≈ 0.998546`
- event logistic router `≈ 0.998556`
- **event tree router `≈ 0.998567`**

RMSE tells the same story:

- retained gated semantic+audio expert `≈ 1.286e-4`
- validation-selected expert `≈ 1.278e-4`
- event logistic router `≈ 1.274e-4`
- **event tree router `≈ 1.269e-4`**

So the best transfer result in this routing family is now:

- a shallow event-level tree,
- over the two retained gated experts,
- using compact reliability features rather than a larger stacked branch.

### 2. The gain is still small and not statistically convincing

Even though the tree is now the best pooled router in this family, the lift is still modest:

- retained gated semantic+audio expert `≈ 0.998527`
- validation-selected expert `≈ 0.998546`
- **event tree router `≈ 0.998567`**

Paired significance stays weak:

- versus retained semantic+audio expert, paired `p(MSE) ≈ 0.683`
- versus validation-selected expert, paired `p(MSE) ≈ 0.829`

So this is a coherent routing refinement, not yet a publishable upgrade claim by itself.

### 3. The routing story is more interpretable than generic stacking

The shallow event tree improves the median held-out ticker `R^2` slightly as well:

- validation-selected expert `≈ 0.995916`
- **event tree router `≈ 0.995940`**

Its feature-usage counts are concentrated on a small set of intuitive signals:

- expert disagreement: `abs_sem_minus_qa_pred`
- `A4` observability / confidence: `a4_strict_row_share`, `a4_strict_high_conf_share`
- `Q&A` reliability: `qa_bench_coverage_mean`, `qa_bench_evasion_score_mean`

So the useful routing information does **not** look like “more raw features are always better.”

It looks more like:

- trust the semantic+audio expert by default,
- but reroute toward the `qa_benchmark_svd` expert when observability, `Q&A` reliability, and expert disagreement line up in the right way.

### 4. The improvement is heterogeneous across held-out tickers

The tree helps meaningfully on some held-out names:

- `AAPL`
- `AMGN`
- `IBM`
- `NKE`

It is flat or nearly flat on:

- `AMZN`
- `CSCO`

And it still gives back some performance on:

- `DIS`
- `MSFT`
- `NVDA`

That mixed pattern explains why the pooled result is better but still not statistically secure.

## Updated Interpretation

This checkpoint strengthens the current transfer-side method story.

The repo now has evidence for three increasingly structured routing readings:

1. simple observability gating beats ungated transfer semantics,
2. fold-level selective expert choice beats brute-force stacking,
3. and a shallow event-level router over compact reliability features is the strongest pooled result in this routing family.

That is a cleaner and more defensible idea than:

- more weak-label feature stacking,
- more gate complexity,
- or a larger sequence model.

## Practical Consequence

The safe repo conclusion is now:

1. keep the retained gated semantic+audio expert as the default transfer extension,
2. keep the shallow event-tree router as the strongest current **exploratory routing extension**,
3. do not turn this into a headline claim yet because the gain is still tiny and non-significant,
4. and if we continue this line, the most promising next step is a **more principled transfer router with stronger supervision**, not brute-force feature growth.

So the main contribution still does **not** change, but the repo now has a clearer novel extension path:

- observability-aware, reliability-aware routing across complementary transfer experts.
