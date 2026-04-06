# After-Hours Observability-Gated Audio Transfer Checkpoint 2026-03-17

## Purpose

The previous two transfer-side checkpoints established:

- low-rank semantic transfer (`lsa=4`) is better than larger semantic bottlenecks,
- a simple `A4` observability gate improves the transfer semantic line,
- role-aware aligned audio gives a tiny positive hint on the same matched clean `after_hours` sample.

This checkpoint asks the next coherent question:

- if transfer-side semantics should only be trusted under good `A4` observability,
- should the role-aware audio branch be activated under the same logic rather than used uniformly?

## Design

New script:

- `scripts/run_afterhours_observability_gated_audio_unseen_ticker.py`

Setting:

- clean `after_hours`
- unseen-ticker evaluation
- semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- gate feature: `a4_strict_row_share`

Compared models on the matched aligned-audio sample:

- `pre_call_market_only`
- `pre_call_market + A4 + qna_lsa`
- observability-gated semantic line
- `pre_call_market + A4 + qna_lsa + role_aware_audio_svd8`
- observability-gated semantic+audio line

The gate threshold is chosen on validation separately for:

1. semantic branch versus `pre_call_market_only`
2. semantic+audio branch versus `pre_call_market_only`

Outputs:

- `results/afterhours_observability_gated_role_aware_audio_unseen_ticker_lsa4_real/afterhours_observability_gated_audio_unseen_ticker_summary.json`
- `results/afterhours_observability_gated_role_aware_audio_unseen_ticker_lsa4_real/afterhours_observability_gated_audio_unseen_ticker_predictions.csv`
- `results/afterhours_observability_gated_role_aware_audio_unseen_ticker_lsa4_real/afterhours_observability_gated_audio_unseen_ticker_thresholds.csv`

## Main Findings

### 1. Gating helps the role-aware audio branch more than using it everywhere

On the matched unseen-ticker test sample:

- `pre_call_market_only ≈ 0.998482`
- ungated semantic line `≈ 0.998370`
- ungated semantic + role-aware audio `≈ 0.998434`
- observability-gated semantic line `≈ 0.998434`
- observability-gated semantic + role-aware audio `≈ 0.998527`

Median ticker `R^2`:

- `pre_call_market_only ≈ 0.995427`
- ungated semantic + role-aware audio `≈ 0.995907`
- gated semantic + role-aware audio `≈ 0.995916`

So the gated role-aware audio line becomes the best overall transfer-side model on this matched sample.

### 2. This is the first transfer extension here that slightly exceeds the pre-call market baseline

The key numerical point is narrow but real on this sample:

- `pre_call_market_only ≈ 0.998482`
- gated semantic + role-aware audio `≈ 0.998527`

The absolute lift is tiny.

But it is the first current transfer-side extension in this branch that moves above the strongest plain pre-call market model rather than only narrowing the gap.

### 3. The gate becomes slightly more conservative when audio is added

Mean test activation rate:

- gated semantic: about `0.799`
- gated semantic + role-aware audio: about `0.784`

This is a useful interpretation change:

- transfer-side audio should not be treated as a uniform add-on,
- it looks most plausible when both semantic content and audio are only trusted under stronger `A4` observability.

Examples from the by-ticker readout:

- `NVDA`: about `0.999701 -> 0.999747`
- `MSFT`: about `0.94568 -> 0.94956`
- `AMGN`: about `0.72559 -> 0.72904`

But it does not help everywhere:

- `AAPL` remains worse than `pre_call_market_only`
- `AMZN` also stays below `pre_call_market_only`
- several firms are unchanged because the gate stays fully active

## Significance and Caution

This is still a small-signal result, not a locked headline claim.

Paired pooled unseen-ticker tests show:

- gated semantic+audio versus ungated semantic+audio:
  - positive mean `R^2` gain
  - bootstrap `R^2` gain CI still crosses zero
  - permutation `p(MSE) ≈ 0.123`
- gated semantic+audio versus `pre_call_market_only`:
  - tiny positive overall `R^2` difference
  - permutation `p(MSE) ≈ 0.752`

So the right reading is:

- promising,
- coherent with the observability story,
- but not yet statistically convincing.

## Updated Interpretation

The transfer-side story is now cleaner than before:

1. low-rank semantics are the transferable semantic baseline,
2. observability gating improves that baseline,
3. role-aware audio helps most when it is also gated by observability,
4. the best current transfer extension is therefore not raw audio or ungated semantic fusion,
5. it is a conservative observability-gated semantic+audio branch.

This is a more defensible research direction than returning to high-dimensional sequence structure.

## Recommended Next Step

The next credible move is one of:

1. replace heuristic `qna_lsa` with stronger transferred `Q&A` quality or evasion supervision while keeping the same gate,
2. test the gated role-aware audio branch on harder or more adversarial subsets,
3. convert the gate from a threshold rule into a very small learned reliability model, but only if it stays interpretable and prior-aware.
