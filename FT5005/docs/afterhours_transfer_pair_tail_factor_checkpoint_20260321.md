# After-Hours Transfer Pair-Tail Factor Checkpoint 2026-03-21

## Purpose

The answerability-factor checkpoint already showed that the current shell-miss profile is learnable:

- dense, lower-directness, more evasive `Q&A` does define a compact latent,
- but that average-style factor still did not beat hard abstention.

That raises the next narrower question:

## is the remaining signal really about local worst-case `Q&A` pairs rather than event-level averages?

This is a clean follow-up because the miss diagnostics suggested concentrated hard pockets rather than broad, smooth shifts.
So instead of adding a new shell, this checkpoint asks whether a compact **pair-tail** representation of the same transcript can do better.

## Design

New scripts:

- `scripts/build_qa_pair_tail_features.py`
- `scripts/run_afterhours_transfer_pair_tail_factor_benchmark.py`

### Step 1. Build raw pair-tail features

From the clean `after_hours` panel and raw `A1` JSON transcripts, the feature builder constructs event-level summaries of local worst-case `Q&A` behavior:

- tail evasion (`max`, `top2 mean`, `high-evasion share`)
- tail directness / coverage (`min`, `bottom2 mean`)
- tail severity (`max`, `top2 mean`), where severity combines:
  - question complexity,
  - evasion,
  - low coverage,
  - weak direct-early answering
- dispersion and first-two-pair summaries
- nonresponse / numeric-mismatch / short-evasive shares

Feature outputs:

- `results/qa_pair_tail_features_real/qa_pair_tail_features.csv`
- `results/qa_pair_tail_features_real/qa_pair_tail_pair_metrics.csv`
- `results/qa_pair_tail_features_real/qa_pair_tail_features_summary.json`

Coverage summary:

- events: `172`
- events with QA pairs: `172`
- average pair count: `≈ 7.41`
- average max evasion: `≈ 0.454`
- average top-2 severity: `≈ 2.743`

### Step 2. Benchmark compact pair-tail factors

Transfer protocol stays unchanged:

- train `val2020_test_post2020`
- validate `val2021_test_post2021`
- test `val2022_test_post2022`
- disagreement rows still fall back to `pre_call_market_only`
- only agreement rows are learnably refined

Factor families:

1. `pair_tail_core`
2. `pair_tail_dispersion`
3. `pair_tail_with_observability`

Factor builders:

- `PCA1`
- `PLS1`

Route variants:

- `factor_only`
- `geometry_plus_factor`

Benchmark outputs:

- `results/afterhours_transfer_pair_tail_factor_benchmark_lsa4_real/afterhours_transfer_pair_tail_factor_benchmark_summary.json`
- `results/afterhours_transfer_pair_tail_factor_benchmark_lsa4_real/afterhours_transfer_pair_tail_factor_benchmark_tuning.csv`
- `results/afterhours_transfer_pair_tail_factor_benchmark_lsa4_real/afterhours_transfer_pair_tail_factor_benchmark_test_predictions.csv`
- `results/afterhours_transfer_pair_tail_factor_benchmark_lsa4_real/afterhours_transfer_pair_tail_factor_benchmark_scores.csv`
- `results/afterhours_transfer_pair_tail_factor_benchmark_lsa4_real/afterhours_transfer_pair_tail_factor_benchmark_loadings.csv`
- `results/afterhours_transfer_pair_tail_factor_benchmark_lsa4_real/afterhours_transfer_pair_tail_factor_benchmark_coefficients.csv`

## Main Findings

### 1. The best validation route comes from the local tail core, not from broader dispersion or observability add-ons

Best validation configuration:

- family: `pair_tail_core`
- method: `PCA1`
- route: `geometry_plus_factor`
- `alpha = 10`
- validation `R^2 ≈ 0.99884079`

Importantly, this is essentially tied with the broader tail families.
So the main learnable content is already captured by the local worst-pair core itself.

## local worst-case answerability structure matters more than extra tail-dispersion or observability add-ons

### 2. Held-out latest-window performance still does not beat hard abstention

Held-out latest-window test scores:

- `pair_tail_factor_route ≈ 0.99863855`
- `hard_abstention ≈ 0.99864017`
- `geometry_only ≈ 0.99863878`
- previous `answerability_factor_route ≈ 0.99863849`
- `pre_call_market_only ≈ 0.99848206`

Paired tests:

- vs hard abstention: `p(MSE) ≈ 0.3865`
- vs geometry only: `p(MSE) ≈ 0.6860`
- vs previous answerability factor route: effectively tied (`p(MSE) = 1.0`)

So the pair-tail route is slightly better than the previous average-style answerability factor, but only marginally and not in a statistically meaningful way.

## hard abstention still remains the strongest compact transfer route in the repo

### 3. The learned factor is very interpretable

The best factor loadings are exactly what the miss-diagnostics story would predict:

- negative on `qa_tail_max_evasion`
- negative on `qa_tail_top2_evasion_mean`
- positive on `qa_tail_bottom2_direct_early_mean`
- positive on `qa_tail_min_direct_early`
- positive on `qa_tail_bottom2_coverage_mean`
- positive on `qa_tail_min_coverage`
- negative on `qa_tail_severity_max`
- negative on `qa_tail_top2_severity_mean`

So the factor is a clean local answerability axis:

## events with less evasive worst-case pairs and stronger bottom-tail coverage / directness are more trustworthy

That is a useful scientific result, because it says the earlier answerability story was not just an averaging artifact.
A local worst-pair structure really is there.

### 4. But the route still does not target the shell’s hardest agreement cases well enough

On the latest held-out window:

- the route keeps the agreed expert on `29 / 44` agreement rows
- it vetoes to `pre_call_market_only` on `15 / 44`

And among those vetoes:

- only `3 / 15` overlap the hard-abstention top-miss quartile from the previous miss-diagnostics checkpoint

So even though the representation changed from event averages to local pair tails, the route still is not isolating the right high-risk agreement cases reliably enough.

### 5. The new signal is real, but still stays inside the same ceiling as the previous proxy pool

This is the most important summary point.
The pair-tail route is:

- better grounded in the actual local `Q&A` structure,
- slightly cleaner conceptually,
- and very slightly better numerically than the previous answerability factor,

but it still lands in the same narrow band just below hard abstention.

That means the current proxy family is being refined, not fundamentally expanded.

## Updated Interpretation

This checkpoint is a good research result because it clarifies the structure of the remaining signal.

### What we now know

1. The shell-miss profile is not only an event-average phenomenon.
2. A compact local worst-pair answerability factor is real and interpretable.
3. The strongest part of that factor is the tail core itself, not extra dispersion or observability add-ons.
4. But even this more local representation still does not beat hard abstention.

### What that means

The next step should stay disciplined:

- **not** more shell complexity,
- **not** more variations on the same aggregate proxy pool,
- but a search for genuinely new upstream evidence that can identify the hard agreement failures more directly.

The pair-tail result is therefore a useful narrowing step:

- local answerability structure matters,
- but the current handcrafted proxy family still does not overturn the hard-abstention mainline.
