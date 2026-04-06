# After-Hours Transfer Reliability-Gate Search Checkpoint 2026-03-17

## Purpose

After the observability-gated role-aware audio checkpoint, the next obvious question was:

- is the current simple `a4_strict_row_share` gate just a weak heuristic,
- or does a more flexible reliability gate actually work better on the matched unseen-ticker audio subset?

This checkpoint benchmarks that directly.

## Design

New script:

- `scripts/run_afterhours_transfer_reliability_gate_search.py`

Matched sample and task:

- clean `after_hours`
- unseen-ticker evaluation
- low-rank semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- branch under study: `pre_call_market + A4 + qna_lsa + role_aware_audio_svd8`

Compared gate families:

1. no gate
2. simple local gate on `a4_strict_row_share`
3. local per-ticker single-feature search over a broader candidate set
4. shared pooled logistic gate over compact reliability features
5. shared pooled conjunctive gate over two interpretable features

Outputs:

- `results/afterhours_transfer_reliability_gate_search_role_aware_audio_lsa4_real/afterhours_transfer_reliability_gate_search_summary.json`
- `results/afterhours_transfer_reliability_gate_search_role_aware_audio_lsa4_real/afterhours_transfer_reliability_gate_search_predictions.csv`
- `results/afterhours_transfer_reliability_gate_search_role_aware_audio_lsa4_real/afterhours_transfer_reliability_gate_search_gate_details.csv`

## Main Findings

### 1. The current simple `A4` gate is still the best tested gate family

Overall matched unseen-ticker `R^2`:

- `pre_call_market_only ≈ 0.998482`
- ungated semantic+audio branch `≈ 0.998434`
- **simple local `a4_strict_row_share` gate `≈ 0.998527`**
- local single-feature search `≈ 0.998475`
- shared logistic gate `≈ 0.998503`
- shared conjunctive gate `≈ 0.998441`

Median ticker `R^2` tells the same story:

- ungated branch `≈ 0.995907`
- **simple local gate `≈ 0.995916`**
- local single-feature search `≈ 0.995428`
- shared logistic gate `≈ 0.995847`
- shared conjunctive gate `≈ 0.995235`

So the best current transfer-side result remains the simplest one:

- local per-ticker thresholding on `a4_strict_row_share`

### 2. Richer local gate search overfits quickly

Allowing each ticker to search over multiple features and directions does **not** help.

It often picks unstable rules such as:

- `a4_kept_rows_for_duration <= threshold`
- or ticker-specific reversals on broad coverage

Even though these rules can look good on validation, the pooled test result falls back to about `0.998475`, below both:

- the simple local `a4_strict_row_share` gate
- and even the stronger shared logistic gate

So wider local feature search is not a credible upgrade path here.

### 3. A pooled learned reliability gate is more stable than local search, but still not better than the simple gate

The best pooled learned gate uses a small logistic classifier over compact reliability features.

Its learned coefficients are dominated by:

- `a4_broad_row_share` positive
- `a4_strict_row_share` negative

while the other candidate features shrink effectively to zero.

Test result:

- shared logistic gate `≈ 0.998503`

This is better than:

- ungated semantic+audio `≈ 0.998434`
- local single-feature search `≈ 0.998475`

But it still does **not** beat the simple local `a4_strict_row_share` gate at `≈ 0.998527`.

### 4. Shared conjunctive gates are not worth keeping

The best pooled conjunctive gate is roughly:

- `a4_strict_row_share >= 0.6408`
- and `a4_broad_row_share >= 0.9607`

But its test result is only:

- `≈ 0.998441`

So the conjunctive gate adds complexity without improving the transfer branch.

## Updated Interpretation

This checkpoint is useful because it sharpens the method claim.

The current best transfer-side reliability mechanism is **not**:

- broader local gate search,
- pooled logistic gating,
- or two-feature conjunctions.

It is still the simplest interpretable rule:

- trust the low-rank semantic+role-aware-audio branch only when `A4` strict observability is high.

That is scientifically cleaner than a more flexible but unstable gate.

## Practical Consequence

The next step should not be “more gate complexity.”

The better next move is:

1. keep the current simple observability gate fixed,
2. improve the transferable semantic content inside the gated branch,
3. or replace heuristic `qna_lsa` with stronger transferable `Q&A` quality or evasion supervision.

So this checkpoint supports a more disciplined rule:

- **hold the gate simple, and upgrade the signal instead of the gate.**
