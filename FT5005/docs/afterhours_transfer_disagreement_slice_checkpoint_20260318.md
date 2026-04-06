# After-Hours Transfer Disagreement Slice Checkpoint 2026-03-18

## Purpose

The new paper-alignment note recommends tightening the current transfer-side claim by diagnosing the clearest remaining exception:

- in the latest temporal window,
- the disagreement slice no longer has `pre_call_market_only` as the best model,
- and the `qa_benchmark_svd` expert edges it out slightly.

That matters because the current transfer-side story is now built around **reliability-aware abstention**.

So this checkpoint asks:

1. is that latest-window `Q&A` edge broad or fragile?
2. is it tied to one disagreement direction only, or both?
3. does it overturn the abstention story, or just qualify it?

## Design

New script:

- `scripts/run_afterhours_transfer_disagreement_slice_diagnostics.py`

Source inputs:

- `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`

Outputs:

- `results/afterhours_transfer_disagreement_slice_diagnostics_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_slice_summary.json`
- `results/afterhours_transfer_disagreement_slice_diagnostics_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_slice_pattern_overview.csv`
- `results/afterhours_transfer_disagreement_slice_diagnostics_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_slice_event_rows.csv`
- `results/afterhours_transfer_disagreement_slice_diagnostics_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_slice_latest_ticker_gain.csv`

The diagnostic keeps only **disagreement** events and analyzes them from three angles:

1. pooled disagreement behavior,
2. disagreement-direction behavior,
3. and the latest-window event/ticker concentration behind the small `Q&A` edge.

Disagreement directions:

- `tree_sem_logistic_qa`
  - tree prefers the retained semantic expert while logistic prefers the `Q&A` expert
- `tree_qa_logistic_sem`
  - tree prefers the `Q&A` expert while logistic prefers the retained semantic expert

## Main Findings

### 1. The pooled disagreement conclusion does **not** change

Across all disagreement events pooled:

- `pre_call_market_only ≈ 0.997757`
- `qa_benchmark_svd ≈ 0.997714`
- retained semantic+audio expert `≈ 0.997720`

So the disagreement slice still supports the same global conclusion:

- **`pre_call_market_only` remains the best pooled fallback**

This means the latest-window `Q&A` edge does **not** overturn the abstention story at the pooled level.

### 2. Both disagreement directions still favor `pre_call_market_only` when pooled

Pooled by disagreement direction:

#### `tree_sem_logistic_qa` (`n = 38`)
- `pre_call_market_only ≈ 0.998877`
- `qa_benchmark_svd ≈ 0.998817`
- retained semantic+audio expert `≈ 0.998824`

#### `tree_qa_logistic_sem` (`n = 44`)
- `pre_call_market_only ≈ 0.994192`
- `qa_benchmark_svd ≈ 0.994176`
- retained semantic+audio expert `≈ 0.994182`

So the global disagreement story is not being broken by one router mode only.

Instead, both disagreement directions still lean toward `pre_call_market_only` once earlier windows are included.

### 3. The latest-window reversal is real, but very small and highly concentrated

For `val2022_test_post2022`, disagreement size is only `16` events.

On this slice:

- `qa_benchmark_svd` is best with `R² ≈ 0.999079`
- `pre_call_market_only ≈ 0.999042`

But the edge is fragile:

- `qa` beats `pre` on only `7` events
- `pre` beats `qa` on `6` events
- `3` are ties

Net `qa` advantage over `pre` in summed event-level MSE is only:

- `≈ 1.90e-09`

So this is **not** a broad disagreement-regime inversion.
It is a tiny late-window edge created by a nearly balanced event mix.

### 4. The latest-window `Q&A` edge is concentrated in a few events and tickers

Gross positive `qa` gain in the latest disagreement slice is highly concentrated:

- top positive event share of total positive gain `≈ 0.595`
- top 3 positive events share `≈ 0.857`

The single biggest positive event is:

- `CSCO_2025_Q1`

Ticker-level net `qa` gains in the latest disagreement slice are concentrated in:

- `CSCO ≈ +1.26e-09`
- `MSFT ≈ +7.61e-10`
- `NKE ≈ +3.11e-10`

while some names still lean the other way:

- `DIS ≈ -4.31e-10`

So the latest-window `Q&A` edge is not broad evidence that disagreement should generally route to `Q&A`.
It is much more consistent with a **small concentrated pocket**.

### 5. The latest-window exception is not one-sided by disagreement direction

In the latest window:

#### `tree_sem_logistic_qa` (`n = 7`)
- `qa_benchmark_svd ≈ 0.998858`
- `pre_call_market_only ≈ 0.998848`

#### `tree_qa_logistic_sem` (`n = 9`)
- `qa_benchmark_svd ≈ 0.999056`
- `pre_call_market_only ≈ 0.998908`

So the late-window `Q&A` edge appears in **both** disagreement directions.

That is useful because it means the exception is better read as:

- a small late-window disagreement pocket where `Q&A` happened to be slightly more helpful,

not:

- a failure of one specific retained router family.

## Updated Interpretation

This checkpoint keeps the transfer-side contribution honest while making it more precise.

### What stays true

- the strongest pooled temporal transfer route is still:
  - **agreement-triggered abstention to `pre_call_market_only`**
- the disagreement slice is still best understood as:
  - **fallback territory**

### What gets added

- the latest temporal disagreement slice contains a **tiny, concentrated, late-window `Q&A`-friendly pocket**
- that pocket is:
  - small,
  - event-concentrated,
  - ticker-concentrated,
  - and not strong enough to replace the current pooled abstention rule

So the right update is not:

- abandon the abstention story

It is:

- keep the abstention story,
- but explicitly record that the latest disagreement slice contains a fragile `Q&A`-positive sub-pocket that may be worth a later selective calibration study.

## Practical Consequence

The best current research posture is now:

1. keep **reliability-aware abstention** as the main transfer-side extension,
2. do **not** replace disagreement fallback with a global `Q&A` rule,
3. interpret the latest-window `Q&A` edge as a concentrated exception rather than a structural reversal,
4. and use this checkpoint to justify why the next research step should be:
   - either a more selective late-window disagreement calibration,
   - or paper-facing scorecard/reporting rather than more generic router expansion.

So this diagnostic makes the current transfer story stronger, not weaker:

- the exception is real,
- but it is also bounded,
- which means the repo still supports a conservative transfer-side contribution rather than another round of model sprawl.
