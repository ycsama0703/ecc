# After-Hours Transfer Expert-Selection Checkpoint 2026-03-18

## Purpose

The previous transfer `Q&A` signal benchmark showed two things at once:

- the retained gated `qna_lsa + role_aware_audio` branch is still the strongest pooled transfer branch,
- but the gated `qa_benchmark_svd` expert still wins on several held-out tickers even though it loses overall.

So the next practical question was:

- can we turn that complementarity into a cleaner transfer result with **selective expert routing**, instead of stacking more raw features into one branch?

## Design

New script:

- `scripts/run_afterhours_transfer_expert_selection.py`

Run setting:

- clean `after_hours`
- unseen-ticker evaluation
- low-rank semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- same matched aligned-audio subset as the recent transfer checkpoints

Compared experts:

1. `pre_call_market_only`
2. gated `qa_benchmark_svd` expert
3. retained gated `qna_lsa + role_aware_audio` expert

Compared routing / combination rules:

1. keep the retained gated semantic+audio expert as-is
2. validation-selected expert choice among the three experts above
3. positive linear stack on the three gated experts

Outputs:

- `results/afterhours_transfer_expert_selection_role_aware_audio_lsa4_real/afterhours_transfer_expert_selection_summary.json`
- `results/afterhours_transfer_expert_selection_role_aware_audio_lsa4_real/afterhours_transfer_expert_selection_predictions.csv`
- `results/afterhours_transfer_expert_selection_role_aware_audio_lsa4_real/afterhours_transfer_expert_selection_details.csv`

## Main Findings

### 1. Validation-selected expert choice gives the best pooled result in this family

Overall matched unseen-ticker `R^2`:

- gated `qna_lsa + role_aware_audio ≈ 0.998527`
- **validation-selected expert ≈ 0.998546**
- `pre_call_market_only ≈ 0.998482`
- positive stack on gated experts `≈ 0.998501`
- gated `qa_benchmark_svd` expert `≈ 0.998414`

RMSE tells the same story:

- retained gated semantic+audio expert `≈ 1.286e-4`
- **validation-selected expert `≈ 1.278e-4`**

So the best result in this routing family is the simplest one:

- choose one expert on validation,
- then keep that expert on the held-out ticker test fold.

### 2. The routing signal is sparse and interpretable

The validation-selected expert chooses:

- retained gated semantic+audio expert on `6 / 9` folds
- gated `qa_benchmark_svd` expert on `3 / 9` folds
- `pre_call_market_only` on `0 / 9` folds

The folds that switch to the `qa_benchmark_svd` expert are:

- `AMZN`
- `DIS`
- `MSFT`

So the complementary `qa_benchmark_svd` expert is not globally better, but it is locally useful often enough to justify a selective-routing reading.

### 3. The gain over the retained best expert is still small and not statistically convincing

Even though the validation-selected expert is the best pooled model in this family, the gain remains tiny:

- retained gated semantic+audio expert `≈ 0.998527`
- validation-selected expert `≈ 0.998546`

Paired significance versus the retained best expert stays weak:

- paired `p(MSE) ≈ 0.185`

So this is a coherent refinement, but not yet a publishable upgrade claim by itself.

### 4. Positive stacking is not the retained route

The positive stack looks acceptable on pooled `R^2`:

- positive stack `≈ 0.998501`

But it is still below the retained gated semantic+audio expert on `R^2`, and its paired MAE comparison is meaningfully worse:

- versus retained gated semantic+audio expert, paired `p(MAE) ≈ 0.00025`

So the stack is not a stable improvement and should not be kept as the preferred route.

## Updated Interpretation

This checkpoint is useful because it clarifies where the remaining transfer headroom may come from.

What seems to matter is not:

- adding more heuristic `Q&A` features into one larger branch,
- or generic positive stacking across all experts.

What looks more promising is:

- **selective use of complementary experts**,
- where the retained semantic+audio expert remains the default,
- but a compact `qa_benchmark_svd` expert can occasionally be the better transfer-side choice.

That is a better next-step story than “just stack everything.”

## Practical Consequence

The safe repo conclusion is now:

1. keep the retained gated semantic+audio expert as the default transfer extension,
2. keep the validation-selected expert result as an exploratory routing hint,
3. do not retain the positive stack,
4. and if we continue this line, the next stronger method idea should be a **more principled expert router** rather than more raw feature stacking.

So the main contribution does **not** change yet, but the repo now has one more coherent extension path:

- modest, selective transfer expert routing beats brute-force stacking.
