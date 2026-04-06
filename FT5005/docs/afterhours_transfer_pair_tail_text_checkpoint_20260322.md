# After-Hours Transfer Pair-Tail Text Checkpoint 2026-03-22

## Purpose

The pair-tail factor checkpoint established two things:

- the remaining hard-abstention miss profile is genuinely local rather than just event-average,
- but the current handcrafted answerability proxy pool still stays just below the hard-abstention ceiling.

That makes the next disciplined question very natural:

## if the hardest local `Q&A` pair really matters, does its **semantic content** carry a stronger transfer signal than the current handcrafted pair-tail factors?

This is a clean follow-up because it does **not** add a more complex shell.
It only asks whether the highest-severity local question/answer text exposes new upstream evidence that the current factorized proxy family misses.

## Design

New scripts:

- `scripts/build_qa_pair_tail_text_views.py`
- `scripts/run_afterhours_transfer_pair_tail_text_benchmark.py`

### Step 1. Build hardest-pair text views

From the clean `after_hours` panel and raw `A1` transcripts, the builder recovers the top-severity `Q&A` pairs per event and writes compact text views for:

- top-1 hardest question
- top-1 hardest answer
- top-1 hardest `Q&A` pair
- top-2 hardest `Q&A` pair

Outputs:

- `results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv`
- `results/qa_pair_tail_text_views_real/qa_pair_tail_text_top_pairs.csv`
- `results/qa_pair_tail_text_views_real/qa_pair_tail_text_views_summary.json`

Coverage summary:

- events: `172`
- events with top-1 text: `172 / 172`
- average pair count: `≈ 7.41`
- average top-1 severity: `≈ 3.082`
- average top-2 severity mean: `≈ 2.743`

### Step 2. Benchmark local semantic text families

Transfer protocol stays fixed:

- train `val2020_test_post2020`
- validate `val2021_test_post2021`
- test `val2022_test_post2022`
- disagreement rows still fallback to `pre_call_market_only`
- only agreement rows are learnably refined

Benchmarked local text families:

1. `tail_question_top1_lsa`
2. `tail_answer_top1_lsa`
3. `tail_qa_top1_lsa`
4. `tail_qa_top2_lsa`

For each family, the benchmark compares:

- standalone local text route
- `geometry_plus_<family>` route

Outputs:

- `results/afterhours_transfer_pair_tail_text_benchmark_lsa4_real/afterhours_transfer_pair_tail_text_benchmark_summary.json`
- `results/afterhours_transfer_pair_tail_text_benchmark_lsa4_real/afterhours_transfer_pair_tail_text_benchmark_family_overview.csv`
- `results/afterhours_transfer_pair_tail_text_benchmark_lsa4_real/afterhours_transfer_pair_tail_text_benchmark_tuning.csv`
- `results/afterhours_transfer_pair_tail_text_benchmark_lsa4_real/afterhours_transfer_pair_tail_text_benchmark_test_predictions.csv`
- `results/afterhours_transfer_pair_tail_text_benchmark_lsa4_real/afterhours_transfer_pair_tail_text_component_terms.csv`

## Main Findings

### 1. The single best family is the top-1 hardest **question** text, not answer text, not whole-pair text, and not geometry-plus-text

Held-out latest-window test scores:

- `tail_question_top1_lsa ≈ 0.99865468`
- `hard_abstention ≈ 0.99864017`
- `geometry_only ≈ 0.99863878`
- `pair_tail_factor_route ≈ 0.99863855`
- `tail_qa_top1_lsa ≈ 0.99855719`
- `tail_answer_top1_lsa ≈ 0.99855101`
- `tail_qa_top2_lsa ≈ 0.99850821`

Paired tests for the best family (`tail_question_top1_lsa`):

- vs hard abstention: `p(MSE) ≈ 0.04425`
- vs geometry only: `p(MSE) ≈ 0.0210`
- vs pair-tail factor route: `p(MSE) ≈ 0.0175`

## this is the first compact transfer-side refinement in the current local-signal line that actually edges past hard abstention on the held-out latest window

At the same time, this needs to be read honestly:

- the gain is small,
- it comes from a family benchmark rather than a pre-registered single shot,
- and versus `pre_call_market_only` directly the gain is still only borderline (`p(MSE) ≈ 0.098`).

So this is a **promising exploratory win**, not yet a new stable headline.

### 2. The gain is highly specific: local hardest-question semantics help, but geometry-plus-text does not

All `geometry_plus_tail_*` variants remain slightly below `geometry_only` and below hard abstention:

- `geometry_plus_tail_qa_top1_lsa ≈ 0.99863857`
- `geometry_plus_tail_answer_top1_lsa ≈ 0.99863855`
- `geometry_plus_tail_question_top1_lsa ≈ 0.99863851`
- `geometry_plus_tail_qa_top2_lsa ≈ 0.99863831`

This is scientifically useful because it says the new signal is **not** “more geometry plus some local text”.
Instead, the useful information seems to be the **standalone semantic content of the hardest analyst question itself**.

## the strongest new signal is narrow and role-specific, not another broader shell upgrade

### 3. Mechanically, the improvement comes entirely from a better agreement-veto subset

On the latest held-out window (`60` events total):

- disagreement rows: `16`
- agreement rows kept on the agreed expert: `24`
- agreement rows vetoed to `pre_call_market_only`: `20`

Relative to hard abstention:

- the `24` keep-agreed rows are identical to hard abstention
- the entire difference comes from the `20` veto rows

On those veto rows:

- wins vs hard abstention: `9 / 20`
- net MSE gain: `≈ 9.78e-09`
- total positive gain: `≈ 1.09e-08`
- total negative gain: `≈ -1.16e-09`

So the new family is not learning a richer controller everywhere.
It is learning a **more effective local semantic veto** over a subset of agreement events.

### 4. The improvement is concentrated but not random

Largest positive latest-window events vs hard abstention:

- `AAPL_2023_Q4`
- `NVDA_2023_Q1`
- `AMZN_2024_Q3`
- `AMZN_2024_Q1`
- `NKE_2023_Q1`

Largest negative events vs hard abstention:

- `DIS_2023_Q2`
- `DIS_2023_Q3`
- `NKE_2023_Q2`
- `DIS_2023_Q4`

Overlap with the earlier hard-abstention top-miss quartile is:

- `4 / 20` veto rows

That is still not especially high, so this route is **not** a clean detector of all hardest failures.
But it is slightly better targeted than the earlier factor-style vetoes, and it is clearly not just random noise.

### 5. This result updates the transfer story in an important way

Until now, the compact answerability / pair-tail line kept refining the same ceiling just below hard abstention.
This checkpoint is the first clear sign that a **genuinely new upstream signal source** may help:

- not broader event averages,
- not more shell complexity,
- not richer geometry stacking,
- but the local semantic content of the **hardest analyst question**.

That fits the broader project story well:

- analyst-question text has already been the strongest standalone role-text signal,
- but pooled question-role text did not integrate cleanly as a controller or direct expert,
- whereas **local hardest-question text** now appears to isolate a more useful transfer-side pocket.

## Updated Interpretation

This is a strong research checkpoint because it changes the status of the transfer-side signal search.

### What we now know

1. The current hard-abstention shell still remains the safest mainline transfer method.
2. The local answerability proxy family was real but previously stuck just below that ceiling.
3. The first local semantic family to break that ceiling is the **top-1 hardest analyst-question text**.
4. The gain comes through a focused agreement-veto mechanism rather than a broad shell rewrite.

### What that means

The next step should stay disciplined:

- **not** more global shell complexity,
- **not** broad feature piling,
- but a careful follow-up on whether local hardest-question semantics hold up under confirmation / robustness checks and whether they can be summarized more cleanly than the current LSA family.

So the current reading is:

## local hardest-question semantics are the first genuinely new upstream transfer signal to beat hard abstention on the held-out latest window, but the effect is still small and exploratory and must be confirmed carefully
