## Role-Aware Audio Checkpoint

Note:
- this 2026-03-16 checkpoint used the earlier matched semantic setting with `qna_lsa lsa_components=16`
- for the later fixed-split-aligned rerun under `lsa=64`, see `docs/afterhours_audio_lsa64_recheck_20260317.md`

This note records the first role-aware aligned-audio extension after the successful `aligned_audio_svd8` checkpoint on clean `after_hours`.

### Motivation

The previous audio checkpoint established two things:

1. raw sentence-aligned acoustic aggregates are too noisy;
2. low-rank compressed aligned audio can improve the clean `after_hours` `market + controls` and `market + controls + A4` lines.

But that earlier result still did not help the `A4 + qna_lsa` semantic branch.

The natural next question was:

- can we restrict the audio view toward `question / answer / presenter` segments so that compressed audio aligns better with the `Q&A` storyline?

### New assets

- New script:
  - `scripts/build_role_aware_aligned_acoustic_features.py`
- Output root:
  - `results/role_aware_aligned_audio_afterhours_clean_real/`
- Main event table:
  - `results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv`

### Mapping reality check

The role-aware mapping is noisy and sparse:

- `172` events
- `53,873` sentence rows
- mean mapped-row share about `0.182`
- role counts:
  - `answer = 4,169`
  - `question = 827`
  - `presenter = 4,421`
  - `operator = 224`
  - `unknown = 44,232`

So this is not a clean fully supervised role-aligned asset. It is a weakly aligned, low-coverage extension.

### Benchmark setup

- Reused:
  - `scripts/run_afterhours_audio_upgrade_benchmark.py`
- Aligned audio block:
  - role-aware winsorized aggregates
  - compressed to `SVD(8)`
- Effective input size:
  - `353` role-aware aligned-audio inputs
  - compressed to `8` factors
  - explained variance ratio sum about `0.611`

### Main results

#### Controls-only line

- `pre_call_market + controls = 0.919`
- `+ generic aligned_audio_svd8 = 0.925`
- `+ role_aware_aligned_audio_svd8 = 0.924`

So role-aware audio does not beat the simpler generic compressed audio on the plain controls-only line.

#### A4 line

- `pre_call_market + controls + A4 = 0.897`
- `+ generic aligned_audio_svd8 = 0.920`
- `+ role_aware_aligned_audio_svd8 = 0.930`

This is the strongest result in the role-aware branch:

- delta versus no-audio `A4` line: about `+0.033`
- permutation `p(MSE) = 0.0025`
- permutation `p(MAE) = 0.008`

#### A4 + qna_lsa line

- `pre_call_market + controls + A4 + qna_lsa = 0.880`
- `+ generic aligned_audio_svd8 = 0.877`
- `+ role_aware_aligned_audio_svd8 = 0.886`

This is the first audio result that improves the semantic branch rather than hurting it.

The direct comparison versus the earlier generic compressed aligned audio is especially important:

- role-aware `A4 + qna_lsa + audio_svd8 = 0.886`
- generic `A4 + qna_lsa + audio_svd8 = 0.877`
- paired bootstrap mean `R^2` gain about `+0.0096`
- bootstrap CI stays positive
- permutation `p(MSE) = 0.0055`

So even though role-aware mapping coverage is low, it still improves the semantic audio branch relative to the generic compressed audio representation.

### Interpretation

This is the cleanest current audio story:

1. Whole-call raw audio is too noisy.
2. Whole-call compressed aligned audio is useful on the market and `A4` lines.
3. Role-aware compressed aligned audio is the first audio variant that helps the `A4 + qna_lsa` semantic branch.

That means the current audio contribution can now be stated as a limitation-driven progression:

- sentence-aligned extraction
- robust event aggregation
- low-rank compression
- role-aware restriction for the semantic branch

### Recommended next step

The best next audio experiment is no longer generic raw feature expansion.

It is:

- keep generic `aligned_audio_svd8` for the plain `market + controls` branch
- use role-aware `aligned_audio_svd8` for the `A4 + qna_lsa` branch
- do not prioritize `qa`-only or `answer`-only variants as the main branch

### Follow-up subset sweep

After this checkpoint, a narrower sweep was run with:

- `qa`-only compressed audio
- `answer`-only compressed audio
- merged `whole-call + role-aware` compressed audio

The result was clear:

- `qa`-only and `answer`-only both underperform the current role-aware mixed branch
- merged `whole-call + role-aware` compression also underperforms the role-aware mixed branch on the semantic line

Key comparison on `A4 + qna_lsa + audio_svd8`:

- whole-call compressed audio: `0.877`
- role-aware mixed compressed audio: `0.886`
- `qa`-only compressed audio: `0.876`
- `answer`-only compressed audio: `0.873`
- merged whole-call plus role-aware compressed audio: `0.879`

Direct paired comparison versus the current role-aware mixed branch:

- whole-call compressed audio is worse with permutation `p(MSE) = 0.0055`
- merged whole-call plus role-aware compressed audio is also worse with permutation `p(MSE) = 0.00025`

So the current best audio choice for the semantic branch is not the narrowest subset and not the broadest union.
It is the compressed role-aware mixed branch built from weak `question / answer / presenter` alignment.
