# After-Hours Audio LSA64 Recheck 2026-03-17

## Purpose

The 2026-03-16 audio notes were informative, but they were not aligned to the now-locked fixed-split semantic bottleneck.

The key issue was:

- the earlier audio benchmark used `qna_lsa` with `lsa_components=16`
- the later strict semantic sweep showed the clean fixed-split mainline is best at `lsa=64`

So the main question became:

- does compressed aligned audio still help once the semantic branch is rerun at the current fixed-split best `lsa=64`?

## Design

Matched clean `after_hours` sample:

- `172` joined rows
- split `train=89`, `val=23`, `test=60`
- target: `shock_minus_pre`
- baseline frame: same-ticker prior plus residual ridge

Reruns:

1. generic aligned-audio benchmark
   - output: `results/afterhours_audio_upgrade_benchmark_winsor_svd8_lsa64_real/`
2. role-aware mixed aligned-audio benchmark
   - output: `results/afterhours_role_aware_audio_upgrade_benchmark_svd8_lsa64_real/`

Shared semantic setting:

- `lsa_components=64`
- aligned audio compressed to `SVD(8)`

## Main Findings

### 1. Generic compressed aligned audio still helps the controls and `A4` lines

Generic rerun:

- `pre_call_market + controls = 0.919`
- `+ generic aligned_audio_svd8 = 0.925`
- `pre_call_market + controls + A4 = 0.897`
- `+ generic aligned_audio_svd8 = 0.920`

So the earlier generic audio message survives:

- compressed aligned audio is still a useful compact factor block on the plain controls line
- and still a useful improvement on the `A4` line

### 2. The best semantic baseline is much stronger once `lsa=64` is restored

With the fixed-split best semantic bottleneck:

- `pre_call_market + controls + A4 + qna_lsa = 0.935`

This is much stronger than the earlier `lsa=16` semantic baseline (`0.880`), which means the earlier role-aware semantic comparison was not on the now-preferred mainline.

### 3. Neither audio variant improves the current `lsa=64` semantic mainline

At `lsa=64`:

- generic compressed audio:
  - `A4 + qna_lsa = 0.935`
  - `A4 + qna_lsa + generic aligned_audio_svd8 = 0.927`
- role-aware mixed compressed audio:
  - `A4 + qna_lsa + role_aware_aligned_audio_svd8 = 0.901`

So the corrected reading is:

- generic compressed audio is only slightly below the best semantic baseline
- role-aware compressed audio drops materially below it
- neither should currently be claimed as an improvement to the strongest fixed-split semantic branch

### 4. Role-aware audio still looks best on the `A4` line, not on the semantic line

At `lsa=64`:

- generic `A4 + audio_svd8 = 0.920`
- role-aware mixed `A4 + audio_svd8 = 0.930`

So role-aware restriction still looks useful on the `A4` branch.

But that advantage does **not** carry over to the current best `A4 + qna_lsa` semantic setting.

## Updated Interpretation

This rerun corrects the previous audio storyline.

The strongest current audio reading is now:

1. compressed aligned audio remains useful on:
   - the plain controls line
   - the `A4` line
2. role-aware compressed audio is still strongest on the `A4` line
3. but the earlier claim that role-aware audio improves the semantic `A4 + qna_lsa` branch does **not** survive once the semantic bottleneck is aligned to the current fixed-split best `lsa=64`

So the safe current paper claim is narrower:

- audio is a credible supporting modality on the controls and `A4` branches
- it is not yet a stable add-on to the best fixed-split semantic branch

## Best Next Step

If the audio line is continued, the next comparison should be:

- not more role subsets,
- but semantic-aligned reruns under:
  - fixed-split `lsa=64`
  - and transfer-friendly `lsa=4`

That is the cleanest way to test whether audio can help without mixing together incompatible semantic bottlenecks.
