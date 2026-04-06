# After-Hours Observability-Gated Semantics Checkpoint 2026-03-17

## Purpose

The current clean `after_hours` semantic story has a sharp split:

- fixed temporal holdout prefers richer `Q&A` semantics (`lsa=64`),
- ticker-held-out transfer prefers a much smaller bottleneck (`lsa=4`),
- but even then `pre_call_market_only` still beats the semantic line overall.

This checkpoint tests one targeted idea that stays close to the existing scientific story:

- only trust the semantic increment when `A4` observability is strong.

Concretely, I used `a4_strict_row_share` as a simple event-level gate for the `pre_call_market + A4 + qna_lsa` line.

## Design

New script:

- `scripts/run_afterhours_observability_gated_semantics.py`

Setting:

- regime: clean `after_hours`
- target: `shock_minus_pre`
- gate feature: `a4_strict_row_share`
- gate rule: on validation, choose a threshold over deciles and then
  - use `pre_call_market + A4 + qna_lsa` when `a4_strict_row_share >= threshold`
  - otherwise fall back to `pre_call_market_only`

Both sides were tested in one reproducible run:

1. fixed temporal split with `qna_lsa=64`
2. unseen-ticker split with `qna_lsa=4`

Outputs:

- `results/afterhours_observability_gated_semantics_clean_real/afterhours_observability_gated_summary.json`
- `results/afterhours_observability_gated_semantics_clean_real/afterhours_observability_gated_fixed_predictions.csv`
- `results/afterhours_observability_gated_semantics_clean_real/afterhours_observability_gated_unseen_predictions.csv`
- `results/afterhours_observability_gated_semantics_clean_real/afterhours_observability_gated_unseen_thresholds.csv`

## Main Findings

### 1. Fixed split does not need the gate

Fixed split (`lsa=64`):

- `pre_call_market_only ≈ 0.9174`
- `pre_call_market + A4 + qna_lsa ≈ 0.9271`
- observability-gated semantic line: `≈ 0.9271`

Selected validation threshold:

- `a4_strict_row_share = 0.5814`

Test activation rate:

- `1.0`

So on the fixed temporal split, the gate simply leaves the semantic line active for all test events.

Interpretation:

- within-panel temporal generalisation already trusts the semantic increment,
- so observability gating does not change the fixed-split headline.

### 2. Unseen-ticker transfer benefits from conservative semantic activation

Ticker-held-out transfer (`lsa=4`):

- `pre_call_market_only ≈ 0.99848`
- `pre_call_market + A4 + qna_lsa ≈ 0.99837`
- observability-gated semantic line `≈ 0.99843`

Median ticker `R^2`:

- `pre_call_market_only ≈ 0.99543`
- `pre_call_market + A4 + qna_lsa ≈ 0.99519`
- observability-gated semantic line `≈ 0.99534`

Mean unseen test activation rate:

- about `0.799`

So the gate does two useful things on the harder transfer split:

- it improves over the ungated semantic line,
- and it recovers much of the gap back toward `pre_call_market_only`.

### 3. The effect is small but directionally coherent

Paired comparison on the unseen-ticker pooled test set:

- versus ungated semantic line:
  - mean `R^2` gain is positive but small
  - bootstrap `R^2` gain CI still crosses zero
  - permutation `p(MSE) ≈ 0.110`
- versus `pre_call_market_only`:
  - gated line is still slightly worse overall
  - permutation `p(MSE) ≈ 0.615`

So this is not yet a strong new headline result.

But it is a cleaner and more interpretable transfer-side idea than adding more sequence depth or another unconstrained modality.

## Per-Ticker Reading

The gate mainly helps on the firms where the transfer semantic block was previously too aggressive under lower observability.

Examples from the saved by-ticker table:

- `NVDA` improves from about `0.99966` to about `0.99970`
- `CSCO` improves from about `0.99519` to about `0.99534`
- `AMGN` improves from about `0.712` to about `0.726`

It does not help everywhere:

- some firms remain fully activated and unchanged,
- some already preferred the ungated semantic line,
- and the best overall held-out-ticker baseline is still the plain pre-call market model.

## Updated Interpretation

This is the most useful new semantic-transfer checkpoint so far because it sharpens the mechanism:

1. richer semantics still matter on the fixed split,
2. cross-firm transfer should not trust the semantic increment uniformly,
3. a simple `A4` observability gate recovers part of the transfer loss without adding a heavier model class.

So the safe current story becomes:

- fixed split headline: keep `A4 + compact Q&A semantics`
- transfer-side refinement: semantic value is conditional on observability quality
- still-open limitation: even the gated transfer line does not yet beat `pre_call_market_only`

## Recommended Next Step

The next credible extension is not more sequence structure.

It is one of:

1. combine the observability gate with the tiny transfer-side role-aware audio hint,
2. replace heuristic `qna_lsa` with stronger transferable `Q&A` quality or evasion supervision,
3. test whether the gate helps more clearly on more adversarial subsets such as lower-coverage or noisier calls.
