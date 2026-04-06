# After-Hours Transfer Complementary Expert Checkpoint 2026-03-18

## Purpose

The temporal confirmation checkpoint made one thing clear:

- the current transfer-router story is real enough to keep,
- but the next bottleneck is no longer router complexity,
- it is the **quality of the complementary expert** that competes with the retained semantic+audio branch.

Within the current repo snapshot, the clean `after_hours` aligned-audio slice is already the same `172`-event slice used by the recent transfer checkpoints, so there was no meaningfully broader matched slice to unlock immediately.

That made the next practical question:

- can we build a **stronger complementary transfer expert**
- than the current gated `qa_benchmark_svd` branch?

## Design

New script:

- `scripts/run_afterhours_transfer_complementary_expert_benchmark.py`

Retained setting:

- clean `after_hours`
- unseen-ticker evaluation
- low-rank semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- same 172-row matched clean `after_hours` slice

Retained anchor expert:

- `pre_call_market + A4 + qna_lsa + aligned_audio_svd`
- with the same simple observability gate

Benchmarked complementary experts:

1. current baseline complementary expert
   - `qa_benchmark_svd`
2. pair-only direct expert
   - `qa_pair_core`
3. hybrid direct expert
   - `hybrid_pair_bench`
4. hybrid direct expert with small text structure
   - `hybrid_plus_text`
5. hybrid direct expert plus aligned audio
   - `hybrid_pair_bench + aligned_audio`
6. hybrid-plus-text direct expert plus aligned audio
   - `hybrid_plus_text + aligned_audio`

For each family, we built:

- the gated complementary expert itself
- and a validation-selected transfer route choosing among:
  - `pre_call_market_only`
  - retained semantic+audio expert
  - that complementary expert

Outputs:

- `results/afterhours_transfer_complementary_expert_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_complementary_expert_summary.json`
- `results/afterhours_transfer_complementary_expert_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_complementary_expert_predictions.csv`
- `results/afterhours_transfer_complementary_expert_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_complementary_expert_selection.csv`

## Main Findings

### 1. The old `qa_benchmark_svd` complementary route is still the best one

Best validation-selected complementary route:

- **`validation_selected_transfer_expert__qa_benchmark_svd ≈ 0.998546`**

That remains better than all new complementary-expert families:

- `qa_pair_core` selected route `≈ 0.998528`
- `hybrid_pair_bench` selected route `≈ 0.998529`
- `hybrid_plus_text` selected route `≈ 0.998510`
- `hybrid_pair_bench + audio` selected route `≈ 0.998521`
- `hybrid_plus_text + audio` selected route `≈ 0.998522`

So the repo does **not** currently support the claim that hybrid supervision becomes stronger simply by turning it into a direct complementary expert.

### 2. Pair-only is too weak as a direct complementary expert

The pair-only direct expert underperforms both the retained semantic+audio branch and the old `qa_benchmark_svd` route:

- gated `qa_pair_core` branch `≈ 0.998373`
- selected `qa_pair_core` route `≈ 0.998528`

It is chosen in only `1 / 9` held-out ticker folds.

So pair-level `Q&A` behaviour remains useful as a **routing / reliability** signal, but not yet as a standalone complementary expert block.

### 3. Hybrid direct experts do not beat the old selected route either

The hybrid direct experts are better than pair-only, but still not enough:

- gated `hybrid_pair_bench` branch `≈ 0.998398`
- selected `hybrid_pair_bench` route `≈ 0.998529`
- gated `hybrid_plus_text` branch `≈ 0.998412`
- selected `hybrid_plus_text` route `≈ 0.998510`

The selected-route comparison against the old `qa_benchmark_svd` route is still negative:

- `hybrid_pair_bench` selected vs old selected:
  - paired `p(MSE) ≈ 0.658`
- `hybrid_plus_text` selected vs old selected:
  - paired `p(MSE) ≈ 0.421`

So hybrid supervision still looks strongest **inside the router**, not as a direct alternative expert family.

### 4. Adding aligned audio to the hybrid complementary experts does not rescue them

Audio-augmented hybrid complementary experts do not improve the picture:

- selected `hybrid_pair_bench + audio` route `≈ 0.998521`
- selected `hybrid_plus_text + audio` route `≈ 0.998522`

And the `hybrid_plus_text + audio` selected route is actually worse than the retained semantic+audio expert:

- versus retained semantic+audio expert:
  - paired `p(MSE) ≈ 0.031`

So simply injecting aligned audio into these direct hybrid expert families is not the right next move.

### 5. The retained semantic+audio expert is still the backbone

Selection counts make the picture especially clear:

- old `qa_benchmark_svd` selected route:
  - retained semantic+audio expert chosen in `6 / 9` folds
  - `qa_benchmark_svd` chosen in `3 / 9` folds
- new hybrid families:
  - usually choose the retained semantic+audio expert in `7 / 9` or `8 / 9` folds
  - the new complementary expert is only used in `1–2 / 9` folds

So the new complementary experts are not yet competitive enough to displace the current retained branch very often.

## Updated Interpretation

This checkpoint gives a useful negative result that sharpens the story:

- hybrid supervision is still valuable,
- but its current value is **decision support for routing**,
- not a drop-in stronger complementary expert representation.

That is actually helpful for the paper narrative because it narrows the method claim:

- the gain is coming from **reliability-aware routing**
- rather than from a new direct hybrid expert that dominates the old `qa_benchmark_svd` branch.

## Practical Consequence

The safe repo conclusion is now:

1. keep the fixed-split semantic headline unchanged,
2. keep the retained semantic+audio branch as the transfer-side backbone,
3. keep the old `qa_benchmark_svd` route as the strongest current complementary expert,
4. keep hybrid supervision primarily as a **routing signal family** rather than a direct expert family,
5. and avoid spending more time on direct hybrid-expert expansion unless a genuinely stronger supervision source is available.

So the next worthwhile move is not another direct expert family sweep.

The better next gate is:

- either a cleaner new supervision source,
- or stronger confirmation of the retained conservative-router story,
- rather than trying to replace the current complementary expert with slightly richer handcrafted blocks.
