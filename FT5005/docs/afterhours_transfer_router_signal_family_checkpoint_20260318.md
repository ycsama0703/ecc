# After-Hours Transfer Router Signal-Family Checkpoint 2026-03-18

## Purpose

The conservative transfer-router checkpoint improved the matched unseen-ticker transfer subset by making event-level routing more cautious.

The next question was:

- **which supervision signals actually make that conservative router better?**

In particular, we wanted to test whether the next gain comes from:

1. pair-level `Q&A` behavior features,
2. benchmark-style directness / evasion features,
3. or a compact hybrid of both.

## Design

New script:

- `scripts/run_afterhours_transfer_router_signal_family_benchmark.py`

It reuses the conservative transfer-router benchmark and compares five signal families on the same matched transfer slice:

1. `lite_baseline`
   - the previously retained compact router features
2. `pair_core`
   - pair-level `Q&A` behavior signals
3. `bench_directness`
   - benchmark-style directness / evasion signals
4. `hybrid_pair_bench`
   - pair-level plus benchmark-style reliability signals
5. `hybrid_plus_text`
   - the hybrid family above plus a small amount of compact event text structure

Retained evaluation setting:

- clean `after_hours`
- unseen-ticker evaluation
- low-rank semantic bottleneck `lsa=4`
- role-aware aligned audio compressed to `SVD(8)`
- same matched aligned-audio subset as the recent transfer checkpoints

Outputs:

- `results/afterhours_transfer_router_signal_family_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_router_signal_family_summary.json`
- `results/afterhours_transfer_router_signal_family_benchmark_role_aware_audio_lsa4_real/afterhours_transfer_router_signal_family_overview.csv`
- plus one subdirectory per family with the underlying conservative-router outputs

## Main Findings

### 1. Hybrid supervision clearly beats pair-only or benchmark-only routing

Pure family results are weaker:

- `pair_core` conservative tree `≈ 0.998603`
- `bench_directness` conservative tree `≈ 0.998624`

The strongest families are the hybrids:

- `hybrid_pair_bench` conservative tree `≈ 0.998624`
- `hybrid_plus_text` conservative logistic `≈ 0.998628`

So the next transfer gain does **not** come from pair-only supervision or benchmark-only supervision alone.

It comes from:

- **combining pair-level behavior, benchmark-style directness/evasion, and compact contextual text structure**

inside the conservative routing scaffold.

### 2. Best pooled result now comes from the hybrid-plus-text conservative logistic router

Best pooled matched unseen-ticker `R^2` in this family search:

- previous lite conservative tree `≈ 0.998623`
- **hybrid-plus-text conservative logistic `≈ 0.998628`**

That is now the highest pooled transfer-side routing result in the repo on this matched aligned-audio subset.

### 3. Best interpretable tree comes from the hybrid pair-plus-benchmark family

Among shallow trees, the best family is:

- **`hybrid_pair_bench` conservative tree `≈ 0.998624`**

That route is especially useful because it keeps the method story more interpretable:

- pair-level answer behavior
- benchmark-style directness / evasion
- conservative routing

all contribute without needing a heavier model class.

### 4. The gain is still exploratory, but the tree result is the most suggestive so far

The most interesting significance readout in this family search is:

- `hybrid_pair_bench` conservative tree versus retained gated semantic+audio expert:
  - paired `p(MSE) ≈ 0.052`

That is still **not** strong enough to claim a robust improvement, especially given the family search itself.

But it is notably closer to a convincing transfer-side gain than the earlier router checkpoints.

### 5. The ranking of families sharpens the method story

The signal-family benchmark now supports a much cleaner interpretation:

- pair-only supervision is not enough,
- benchmark-only supervision is not enough,
- but **hybrid reliability supervision** is the strongest current route.

That means the promising idea is not just:

- “add more `Q&A` features,”

but rather:

- **use complementary pair-level behavior and benchmark-style reliability signals to decide when transfer-side expert switching should be trusted.**

## Updated Interpretation

This checkpoint strengthens the transfer-side research direction in two ways.

First, it improves the best matched transfer result a little more.

Second, it makes the method idea more explicit:

- the right transfer extension is not raw multimodal growth,
- and not more gate complexity,
- but **conservative routing under hybrid reliability supervision**.

That is a better story for novelty and contribution quality.

## Practical Consequence

The safe repo conclusion is now:

1. keep the fixed-split semantic headline unchanged,
2. keep the conservative router as the transfer-side scaffold,
3. treat **hybrid supervision families** as the strongest current signal source for that router,
4. regard the best pooled logistic result and the best shallow-tree result as exploratory rather than final claims,
5. and if we continue this line, the next best step is to make the supervision cleaner and more explicit rather than merely broader.

So the extension path now looks more concrete:

- **conservative transfer routing + hybrid pair-level / benchmark-style reliability supervision**.
