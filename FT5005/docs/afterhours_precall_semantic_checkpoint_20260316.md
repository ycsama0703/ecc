# After-Hours Strict Pre-Call Semantic Checkpoint 2026-03-16

## 1. Purpose

The current project needed one tighter checkpoint around the most defensible contribution line:

- `after_hours`
- `shock_minus_pre`
- evaluated against a strong pre-call market baseline
- with incremental ECC value defined narrowly as coarse `A4` observability plus compact `Q&A` semantics

The fixed-split ladder already showed that this line works on the temporal holdout.

What remained unresolved was whether that same semantic increment survives the harder ticker-held-out stress test.

## 2. Experimental Design

Locked setting:

- regime: `after_hours`
- target: `shock_minus_pre`
- formulation:
  - fixed split: same-ticker prior plus residual ridge
  - unseen-ticker split: global prior plus residual ridge because the held-out ticker has no in-sample history
- slices:
  - all-HTML
  - clean `exclude html_integrity_flag=fail`

Main scripts:

- `scripts/run_afterhours_precall_semantic_ladder.py`
- `scripts/analyze_offhours_shock_robustness.py`
- `scripts/run_afterhours_precall_unseen_ticker.py`

Primary result files:

- `results/afterhours_precall_semantic_ladder_real/afterhours_precall_semantic_ladder_summary.json`
- `results/afterhours_precall_semantic_ladder_clean_real/afterhours_precall_semantic_ladder_summary.json`
- `results/afterhours_precall_semantic_robustness_real/offhours_shock_robustness_summary.json`
- `results/afterhours_precall_semantic_robustness_clean_real/offhours_shock_robustness_summary.json`
- `results/afterhours_precall_unseen_ticker_real/afterhours_precall_unseen_ticker_summary.json`
- `results/afterhours_precall_unseen_ticker_clean_real/afterhours_precall_unseen_ticker_summary.json`

Compared model families:

- `prior_only`
- `pre_call_market_only`
- `pre_call_market_plus_controls`
- `pre_call_market_plus_a4`
- `pre_call_market_plus_a4_plus_qna_lsa`
- `pre_call_market_plus_controls_plus_a4`
- `pre_call_market_plus_controls_plus_a4_plus_qna_lsa`

## 3. Fixed-Split Findings

### 3.1 The strict baseline is strong and interpretable

All-HTML `after_hours`:

- `pre_call_market_only`: `R^2 ≈ 0.918`
- `pre_call_market_plus_a4`: `R^2 ≈ 0.898`
- `pre_call_market_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.919`
- `pre_call_market_plus_controls_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.924`

Clean `after_hours`:

- `pre_call_market_only`: `R^2 ≈ 0.917`
- `pre_call_market_plus_a4`: `R^2 ≈ 0.901`
- `pre_call_market_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.927`
- `pre_call_market_plus_controls_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.935`

Interpretation:

- the pre-call market baseline is already very strong,
- `A4` alone does not reliably improve it,
- but `A4 + qna_lsa` does produce a real incremental gain on the fixed temporal split,
- especially on the clean slice.

### 3.2 The fixed-split result is not a one-year fluke

Clean robustness for `pre_call_market_plus_controls_plus_a4_plus_qna_lsa`:

- overall test `R^2 ≈ 0.935`
- `2023`: `R^2 ≈ 0.822`
- `2024`: `R^2 ≈ 0.946`
- `2025`: `R^2 ≈ 0.965`

The same clean model also survives leave-one-ticker-out removal inside the fixed split:

- minimum leave-one-ticker-out `R^2` gain over prior: about `0.637`
- median gain: about `0.906`

So the fixed-split semantic result is stable enough to treat as real within-panel evidence.

## 4. Unseen-Ticker Findings

### 4.1 The harder cross-firm test changes the ranking

All-HTML unseen-ticker:

- `pre_call_market_only`: `R^2 ≈ 0.9985`
- `pre_call_market_plus_controls`: `R^2 ≈ 0.9984`
- `pre_call_market_plus_a4`: `R^2 ≈ 0.9982`
- `pre_call_market_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.9950`
- `pre_call_market_plus_controls_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.9951`

Clean unseen-ticker:

- `pre_call_market_only`: `R^2 ≈ 0.9985`
- `pre_call_market_plus_controls`: `R^2 ≈ 0.9984`
- `pre_call_market_plus_a4`: `R^2 ≈ 0.9982`
- `pre_call_market_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.9962`
- `pre_call_market_plus_controls_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.9937`

Median ticker `R^2` tells the same story:

- all-HTML:
  - `pre_call_market_only ≈ 0.995`
  - `pre_call_market_plus_a4_plus_qna_lsa ≈ 0.990`
  - `pre_call_market_plus_controls_plus_a4_plus_qna_lsa ≈ 0.982`
- clean:
  - `pre_call_market_only ≈ 0.995`
  - `pre_call_market_plus_a4_plus_qna_lsa ≈ 0.994`
  - `pre_call_market_plus_controls_plus_a4_plus_qna_lsa ≈ 0.969`

### 4.2 The semantic signal is not yet cross-firm transferable

The unseen-ticker result is the most important new limitation:

- once the test ticker is entirely absent from train and validation,
- the simplest pre-call market model is best,
- `A4 + qna_lsa` no longer gives the incremental gain seen on the fixed split,
- and the richer semantic stack can degrade materially on some firms.

Concrete examples on the clean unseen-ticker slice:

- `MSFT`:
  - `pre_call_market_only ≈ 0.941`
  - `pre_call_market_plus_a4_plus_qna_lsa ≈ 0.565`
  - `pre_call_market_plus_controls_plus_a4_plus_qna_lsa ≈ -0.528`
- `AMGN`:
  - `pre_call_market_only ≈ 0.752`
  - `pre_call_market_plus_a4_plus_qna_lsa ≈ 0.640`
  - `pre_call_market_plus_controls_plus_a4_plus_qna_lsa ≈ 0.264`

This means the current semantic contribution should not be framed as universal cross-firm transfer.

### 4.3 A smaller semantic bottleneck helps materially

To test whether the transfer failure is partly a representation-size issue, I also reran the clean unseen-ticker stress with smaller `qna_lsa` bottlenecks.

Clean unseen-ticker with lower-rank `Q&A` semantics:

- `lsa=8`:
  - `pre_call_market_plus_a4_plus_qna_lsa ≈ 0.99825`
  - median ticker `R^2 ≈ 0.99534`
- `lsa=16`:
  - `pre_call_market_plus_a4_plus_qna_lsa ≈ 0.99781`
  - median ticker `R^2 ≈ 0.99305`
- `lsa=32`:
  - `pre_call_market_plus_a4_plus_qna_lsa ≈ 0.99743`
  - median ticker `R^2 ≈ 0.99258`
- baseline `lsa=64`:
  - `pre_call_market_plus_a4_plus_qna_lsa ≈ 0.99621`
  - median ticker `R^2 ≈ 0.99372`

This does not overturn the main limitation, because `pre_call_market_only` is still best at about `0.99848`.

But it does sharpen the diagnosis:

- the transfer problem is not simply "semantic signal is useless,"
- it is at least partly a variance and representation-size problem,
- and low-rank semantic summaries are a more coherent next step than richer sequence structure.

### 4.4 Fixed-split and unseen-ticker prefer different bottlenecks

I then ran a consolidated clean sweep over `lsa = 4, 8, 16, 32, 64` for both:

- the strict fixed temporal split
- the stricter unseen-ticker split

Result file:

- `results/afterhours_precall_bottleneck_sweep_clean_real/afterhours_precall_bottleneck_sweep_summary.json`

Key readout:

- fixed split `pre_call_market + a4 + qna_lsa`
  - `lsa=4`: `R^2 ≈ 0.906`
  - `lsa=8`: `R^2 ≈ 0.877`
  - `lsa=16`: `R^2 ≈ 0.877`
  - `lsa=32`: `R^2 ≈ 0.909`
  - `lsa=64`: `R^2 ≈ 0.927`
- unseen-ticker `pre_call_market + a4 + qna_lsa`
  - `lsa=4`: `R^2 ≈ 0.99837`
  - `lsa=8`: `R^2 ≈ 0.99825`
  - `lsa=16`: `R^2 ≈ 0.99781`
  - `lsa=32`: `R^2 ≈ 0.99743`
  - `lsa=64`: `R^2 ≈ 0.99621`

So the current tradeoff is sharp:

- `lsa=64` is best for the fixed-split headline result,
- `lsa=4` is best for ticker-held-out transfer,
- and intermediate sizes do not dominate both settings at once.

That is a useful scientific constraint, not just an engineering nuisance:

- richer semantics help within-panel temporal generalisation,
- but lower-rank semantics are more transferable across firms,
- which suggests the current semantic block mixes transferable content with firm-specific variance.

## 5. Current Best Scientific Reading

The strongest defensible contribution is now:

1. on clean `after_hours`,
2. relative to a strong pre-call market baseline,
3. coarse `A4` observability and compact `Q&A` semantics add value on the temporal holdout,
4. but that semantic increment is still mostly within-panel and ticker-specific rather than cross-firm transferable.

This is a stronger and more honest paper story than a broader claim that "multimodal semantics generalize across firms."

## 6. What Should Happen Next

The next modeling step should target transferability rather than raw holdout score.

Most coherent options:

- keep `lsa=64` for the main fixed-split benchmark but treat low-rank semantics like `lsa=4` as the transfer-oriented variant,
- compress `Q&A` semantics further and regularize toward lower-variance firm-invariant signals,
- replace current unsupervised `qna_lsa` with transferred pair-level labels or benchmark-style supervision,
- keep the strict `pre_call_market_only` model as the external-validity baseline that every semantic extension must beat.

The wrong next step would be:

- adding more high-dimensional sequence structure,
- because the current evidence already shows that complexity is not the main bottleneck.
