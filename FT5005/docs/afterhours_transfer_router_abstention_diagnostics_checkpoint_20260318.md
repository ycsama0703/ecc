# After-Hours Transfer Router Abstention Diagnostics Checkpoint 2026-03-18

## Purpose

The consensus-fallback benchmark already suggested a cleaner transfer-side story:

- when the retained hybrid tree and hybrid-plus-text logistic routes agree, their shared transfer signal may be useful,
- but when they disagree, the safest pooled temporal fallback is `pre_call_market_only`.

The missing diagnostic was to verify that directly.

This checkpoint asks:

1. **Where does the transfer lift actually live?**
2. **Are agreement events the part where the agreed transfer expert helps?**
3. **Are disagreement events really the part where abstaining back to the market baseline is safest?**

## Design

New script:

- `scripts/run_afterhours_transfer_router_abstention_diagnostics.py`

Source inputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`

Same temporal confirmation windows:

- `val2020_test_post2020`
- `val2021_test_post2021`
- `val2022_test_post2022`

Subsets diagnosed:

1. `agreement`
   - tree and logistic choose the same transfer-side expert
2. `agreement_semantic`
   - agreement subset where both choose the retained semantic+audio expert
3. `agreement_qa`
   - agreement subset where both choose the `qa_benchmark_svd` expert
4. `disagreement`
   - tree and logistic disagree, so disagreement fallback behavior is the real question

Outputs:

- `results/afterhours_transfer_router_abstention_diagnostics_role_aware_audio_lsa4_real/afterhours_transfer_router_abstention_diagnostics_summary.json`
- `results/afterhours_transfer_router_abstention_diagnostics_role_aware_audio_lsa4_real/afterhours_transfer_router_abstention_diagnostics_overview.csv`
- `results/afterhours_transfer_router_abstention_diagnostics_role_aware_audio_lsa4_real/afterhours_transfer_router_abstention_diagnostics_predictions.csv`

## Main Findings

### 1. The pooled agreement subset is where the transfer-side lift lives

Pooled over all three temporal confirmation windows, the agreement subset contains:

- `175 / 257` events
- agreement share `≈ 0.681`

On this subset:

- `pre_call_market_only ≈ 0.997974`
- retained semantic+audio expert `≈ 0.997992`
- selected expert `≈ 0.997999`
- **agreement-supported prediction `≈ 0.998037`**

So the best pooled agreement-subset route is exactly the transfer-side route implied by both routers.

Paired pooled tests on the agreement subset show:

- versus retained semantic+audio expert:
  - `p(MSE) ≈ 0.0158`
- versus selected expert:
  - `p(MSE) ≈ 0.0485`
- versus `pre_call_market_only`:
  - `p(MSE) ≈ 0.204`

So the strongest reading is:

- agreement events are where the hybrid routers recover a cleaner transfer-side improvement,
- especially relative to weaker transfer baselines,
- though this still does **not** become a hard win over raw `pre_call_market_only` by itself.

### 2. The agreement subset splits cleanly into two interpretable regimes

#### Agreement-semantic subset

Pooled:

- size `149`
- share `≈ 0.580`
- selected / retained semantic route `≈ 0.998117`
- `pre_call_market_only ≈ 0.998058`

So when both routers agree on the retained semantic backbone, the semantic branch is indeed the right expert more often than not.

#### Agreement-QA subset

Pooled:

- size `26`
- share `≈ 0.101`
- `qa_benchmark_svd` expert `≈ 0.994953`
- `pre_call_market_only ≈ 0.994713`
- retained semantic expert `≈ 0.993352`

So when both routers agree on the `Q&A` expert, that slice does look like a real `Q&A`-favored pocket rather than noise.

This is useful because it means the abstention story is not just “use the market baseline everywhere.”

It is closer to:

- trust semantic when both retained router views agree on semantic,
- trust `Q&A` when both agree on `Q&A`,
- and abstain when the reliability evidence is mixed.

### 3. The pooled disagreement subset still favors the market baseline

Pooled disagreement subset:

- `82 / 257` events
- disagreement share `≈ 0.319`

On this subset:

- **`pre_call_market_only ≈ 0.997757`**
- disagreement average `≈ 0.997723`
- retained semantic+audio expert `≈ 0.997720`
- pair tree `≈ 0.997719`
- selected expert `≈ 0.997715`
- plus-text logistic `≈ 0.997715`
- `qa_benchmark_svd` expert `≈ 0.997714`

So the pooled disagreement subset is exactly where abstaining to the market baseline remains safest.

Directional pooled tests support that reading:

- versus retained semantic+audio expert:
  - `p(MSE) ≈ 0.0825`
  - `p(MAE) ≈ 0.037`
- versus selected expert:
  - `p(MSE) ≈ 0.0573`
  - `p(MAE) ≈ 0.022`
- versus pair tree:
  - `p(MSE) ≈ 0.0685`
  - `p(MAE) ≈ 0.0313`
- versus disagreement average:
  - `p(MSE) ≈ 0.0573`
  - `p(MAE) ≈ 0.0275`

This is still not a fully hard significance story in MSE terms, but the direction is very consistent and is materially cleaner than expanding the transfer model again.

### 4. Split-by-split reading is mostly coherent, with one useful caveat

By split:

- `val2020_test_post2020`
  - agreement subset best = agreement-supported route `≈ 0.997716`
  - disagreement subset best = `pre_call_market_only ≈ 0.994262`
- `val2021_test_post2021`
  - agreement subset best = `pre_call_market_only ≈ 0.997191`
  - disagreement subset best = `pre_call_market_only ≈ 0.998896`
- `val2022_test_post2022`
  - agreement subset best = agreement-supported route `≈ 0.998591`
  - disagreement subset best = `qa_benchmark_svd ≈ 0.999079`

So the pooled story is strong but not perfectly uniform:

- the agreement subset behaves as hoped in `2020` and `2022`, but not in `2021`
- the disagreement subset favors `pre_call_market_only` in `2020` and `2021`, but the `Q&A` expert edges it out in the latest split

That means the right conclusion is still **reliability-aware abstention with temporal sensitivity**, not a universal rule.

## Updated Interpretation

This diagnostic makes the transfer-side method story cleaner:

- the gain is not coming from more aggressive expert switching everywhere,
- it is coming from a relatively small set of events where multiple router views agree on the same transfer-side correction,
- while the disagreement slice is still mostly a case for abstaining back to the strongest market baseline.

So the safest contribution framing is now:

- **agreement reveals where the transfer signal is credible**
- **disagreement reveals where abstention is safer**

That is a more precise and more defensible statement than the earlier, looser “routing helps” interpretation.

## Practical Consequence

The repo-safe takeaway is now:

1. keep the fixed-split headline unchanged,
2. keep the transfer-side hybrid tree and hybrid-plus-text logistic routes as the two retained router views,
3. keep `agreement + pre_call_market_only fallback` as the best pooled temporal transfer route,
4. interpret the gain as evidence for **reliability-aware abstention**,
5. and use this new diagnostic as the reason that interpretation is now evidence-backed rather than just heuristic.

This is a better place to pause expansion of router families and instead tighten the paper-quality transfer narrative around:

- agreement-supported transfer correction,
- disagreement-triggered abstention,
- and the remaining temporal sensitivity that still limits how hard we should claim the result.
