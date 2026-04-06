# After-Hours Transfer Agreement Signal Benchmark Checkpoint 2026-03-18

## Purpose

After the agreement-refinement checkpoint, the next question was no longer:

- can we keep tuning the router shell,
- or can we keep stacking more calibration layers on top of the same geometry?

The cleaner next question is:

## can a **stronger compact upstream signal family** improve agreement-side transfer refinement?

This is a better research move because it shifts the focus away from:

- more routing mechanics

and toward:

- whether observability / `Q&A` quality / role-aware alignment signals can materially improve the trust decision upstream.

## Design

New script:

- `scripts/run_afterhours_transfer_agreement_signal_benchmark.py`

Inputs:

- temporal router outputs:
  - `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`
- compact upstream feature assets:
  - `results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv`
  - `results/features_real/event_text_audio_features.csv`
  - `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`
  - `results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv`

Coverage is complete on the temporal benchmark rows:

- `257 / 257` temporal rows have all side inputs

So this checkpoint is not limited by missing side-feature joins.

The shell stays fixed:

- disagreement still falls back to `pre_call_market_only`
- only agreement events are refined

Model family is also kept fixed:

- one compact **agreement gain gate**
- trained to predict whether the agreed expert should replace the market baseline

What changes is only the upstream feature family.

Compared families:

1. `geometry_only`
   - the compact prediction-geometry family from the previous agreement-refinement checkpoint
2. `lite_quality`
   - observability + core `Q&A` benchmark + aligned-audio coverage
3. `hybrid_quality`
   - the strongest compact upstream family suggested by the earlier router signal benchmark
4. `geometry_plus_hybrid`
   - geometry plus the hybrid upstream signal block

Temporal protocol:

- train on `val2020_test_post2020` agreement events
- tune on `val2021_test_post2021` agreement events
- refit on `2020 + 2021` agreement events
- test on full `val2022_test_post2022`

Outputs:

- `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_agreement_signal_benchmark_summary.json`
- `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_agreement_signal_benchmark_overview.csv`
- `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_agreement_signal_benchmark_tuning.csv`
- `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_agreement_signal_benchmark_test_predictions.csv`

## Main Findings

### 1. Quality-only upstream families do not beat the current geometry baseline

Held-out full `2022` split:

- `geometry_only ≈ 0.998639`
- `lite_quality ≈ 0.998482`
- `hybrid_quality ≈ 0.998470`

So the compact upstream quality families on their own are **not** stronger than the existing geometry-led agreement gate.

That is already informative:

- simply swapping out prediction geometry for compact observability / `Q&A` / aligned-audio quality signals is not enough.

### 2. Adding the best upstream quality family to geometry helps only trivially

Best family:

- **`geometry_plus_hybrid ≈ 0.998639`**

Direct comparison:

- `geometry_plus_hybrid ≈ 0.998638845`
- `geometry_only ≈ 0.998638784`

So the gain is only a tiny nudge.

Paired comparison:

- best family vs geometry-only: `p(MSE) = 1.0`

So the upstream hybrid block is directionally compatible with the geometry gate,
but does **not** provide a robust new transfer win.

### 3. Hard abstention still remains the best route overall

Held-out full `2022` split:

- `pre_call_market_only ≈ 0.998482`
- `agreement_supported_pred ≈ 0.998624`
- **`agreement_pre_only_abstention ≈ 0.998640`**
- `geometry_plus_hybrid ≈ 0.998639`

Paired comparison:

- best family vs hard abstention: `p(MSE) ≈ 0.507`

So even after moving upstream and testing a stronger compact signal family,
the repo still does **not** have a cleaner agreement-side replacement for the current abstention rule.

### 4. The best family is still interpretable

Top coefficients in `geometry_plus_hybrid` include:

- `qa_pair_answer_forward_rate_mean`
- `qa_evasive_proxy_share`
- `qa_bench_coverage_mean`
- `pair_minus_pre_pred`
- `logistic_minus_pre_pred`
- `agreed_minus_pre_pred`

This is useful because it shows:

- the relevant upstream direction is not random feature piling
- it is still concentrated in a small set of interpretable `Q&A` quality / coverage terms plus prediction geometry

But at the moment that direction is only **marginally supportive**, not decisive.

## Updated Interpretation

This checkpoint helps us rule out another tempting path.

### What we now know

1. The current transfer bottleneck is not just “missing more compact upstream features”.
2. Compact quality-only feature families are weaker than geometry-only agreement gating.
3. Geometry plus hybrid upstream quality is slightly better than geometry alone, but only trivially.
4. Even the best upstream feature family still does **not** beat hard abstention.

### What that means

The next useful research step is probably **not**:

- more agreement feature-family tuning,
- more small calibration variants,
- or more reweighted versions of the current router shell.

Instead, this benchmark suggests:

## the project likely needs a genuinely stronger upstream transferable signal source,

not just a better combination of the current compact observability / `Q&A` proxy blocks.

## Practical Consequence

This is another good narrowing result.

It keeps the story disciplined:

- hard abstention still stands,
- geometry still matters most,
- compact upstream quality signals are scientifically useful but not yet strong enough,
- and the next research move should search for a stronger transferable signal source rather than another round of routing-shell optimization.
