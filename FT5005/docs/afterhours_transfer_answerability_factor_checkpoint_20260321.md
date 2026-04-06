# After-Hours Transfer Answerability-Factor Checkpoint 2026-03-21

## Purpose

The new hard-abstention miss diagnostics gave us a much cleaner target than another router sweep.
The remaining shell failures looked most like:

- dense `Q&A`,
- lower directness / coverage,
- higher evasion or low-overlap answering,
- slightly weaker observability,
- and only secondarily larger surprise.

That raises the next compact method question:

## can a single learnable answerability-style factor improve agreement-side abstention?

This is intentionally a small and interpretable test.
Instead of adding another routing shell, the goal is to see whether the newly diagnosed failure profile can be compressed into a one-factor signal that actually helps decide when an agreed transfer correction should be kept or vetoed.

## Design

New script:

- `scripts/run_afterhours_transfer_answerability_factor_benchmark.py`

Protocol:

- reuse the temporal transfer setup:
  - train `val2020_test_post2020`
  - validate `val2021_test_post2021`
  - test `val2022_test_post2022`
- disagreement rows still fall back to `pre_call_market_only`
- only agreement rows are learnably refined

Feature families:

1. `answerability_core`
   - `qa_pair_count`
   - `qa_pair_low_overlap_share`
   - `qa_pair_answer_forward_rate_mean`
   - `qa_multi_part_question_share`
   - `qa_bench_direct_answer_share`
   - `qa_bench_direct_early_score_mean`
   - `qa_bench_coverage_mean`
   - `qa_bench_evasion_score_mean`
   - `qa_bench_high_evasion_share`
2. `answerability_plus_observability`
   - the same core plus:
   - `a4_strict_row_share`
   - `a4_strict_high_conf_share`
   - `a4_median_match_score`
3. `answerability_observability_surprise`
   - the same plus:
   - `revenue_surprise_pct`
   - `abs_revenue_surprise_pct`
   - `abs_eps_gaap_surprise_pct`

Factor builders:

- `PCA1`
- `PLS1`

Route variants:

- `factor_only`
- `geometry_plus_factor`

Coverage is complete:

- temporal rows: `257 / 257`
- train agreement rows: `80`
- validation agreement rows: `51`
- test agreement rows: `44`

Outputs:

- `results/afterhours_transfer_answerability_factor_benchmark_lsa4_real/afterhours_transfer_answerability_factor_benchmark_summary.json`
- `results/afterhours_transfer_answerability_factor_benchmark_lsa4_real/afterhours_transfer_answerability_factor_benchmark_tuning.csv`
- `results/afterhours_transfer_answerability_factor_benchmark_lsa4_real/afterhours_transfer_answerability_factor_benchmark_test_predictions.csv`
- `results/afterhours_transfer_answerability_factor_benchmark_lsa4_real/afterhours_transfer_answerability_factor_benchmark_scores.csv`
- `results/afterhours_transfer_answerability_factor_benchmark_lsa4_real/afterhours_transfer_answerability_factor_benchmark_loadings.csv`
- `results/afterhours_transfer_answerability_factor_benchmark_lsa4_real/afterhours_transfer_answerability_factor_benchmark_coefficients.csv`

## Main Findings

### 1. The best validation route is compact, but still only nudges the existing shell

Best validation configuration:

- family: `answerability_core`
- factor: `PCA1`
- route: `geometry_plus_factor`
- ridge `alpha = 10`
- validation `R^2 ≈ 0.9988408`

The strongest competitors on validation are essentially tied:

- `answerability_plus_observability + geometry_plus_factor`
- `answerability_observability_surprise + geometry_plus_factor`

So the first useful lesson is already clear:

## the new signal is dominated by the answerability core itself; extra observability and surprise terms do not produce a cleaner validation win

### 2. Held-out latest-window performance still does not beat hard abstention

Held-out latest-window test scores:

- `answerability_factor_route ≈ 0.99863849`
- `hard_abstention ≈ 0.99864017`
- `geometry_only ≈ 0.99863878`
- `geometry_plus_hybrid ≈ 0.99863884`
- `pre_call_market_only ≈ 0.99848206`

Paired tests:

- vs hard abstention: `p(MSE) ≈ 0.3865`
- vs geometry only: `p(MSE) ≈ 0.6335`
- vs geometry plus hybrid: `p(MSE) ≈ 0.8758`

So this route is not bad — it stays near the top shell family — but it still does **not** become the new best transfer method.

## hard abstention remains the strongest compact transfer route in the repo

### 3. The learned factor is interpretable and does align with the miss-diagnostics story

The best factor loadings are very clean:

- negative on `qa_bench_evasion_score_mean`
- positive on `qa_bench_direct_early_score_mean`
- positive on `qa_bench_direct_answer_share`
- negative on `qa_bench_high_evasion_share`
- positive on `qa_bench_coverage_mean`
- negative on `qa_pair_low_overlap_share`
- smaller negative contributions from `qa_pair_count`

So the factor is not random.
It is approximately a:

## directness / coverage / low-evasion answerability axis

That is scientifically useful because it confirms the miss-diagnostics interpretation with a learnable one-factor construction.

### 4. But the route still does not target the latest-window hard misses well enough

On the held-out latest window:

- the route keeps the agreed expert on `29 / 44` agreement rows
- it vetoes to `pre_call_market_only` on `15 / 44`

So it ends up producing a veto policy very similar in size to the earlier question-role veto family.
But that veto is still not well targeted:

- only `3 / 15` vetoed agreement events fall inside the top hard-miss quartile from the new shell-miss diagnostics

So even though the factor is interpretable, it is still not isolating the right high-risk agreement cases reliably enough to beat hard abstention.

### 5. Surprise still looks secondary rather than central

The validation leaderboard is also telling:

- `answerability_core`, `answerability_plus_observability`, and `answerability_observability_surprise`
  are effectively tied at the top when paired with geometry

That means the current gain is not being driven by surprise.
If surprise matters, it is only as a weak modifier on top of a stronger answerability signal.

## Updated Interpretation

This checkpoint is a good research result even though it is not a new headline win.

### What we now know

1. The shell-miss diagnostics were pointing at a real learnable structure.
2. That structure compresses cleanly into a one-factor answerability axis.
3. The factor is interpretable and transferable enough to stay competitive with the best non-abstention agreement refinements.
4. But it still does **not** beat hard abstention on the held-out latest window.
5. Adding observability or surprise to the factor family does not materially improve the result.

### What that means

The right takeaway is narrow and useful:

- the next signal search should stay focused on answerability / evasion,
- but it should not keep expanding the current family sideways,
- because the current one-factor answerability signal already captures most of what this proxy pool can offer.

So this checkpoint strengthens the current research direction in a disciplined way:

- the failure profile is real,
- a compact learnable answerability factor exists,
- but the current proxy pool is still not enough to displace hard abstention as the best transfer-side method.
