# After-Hours Audio Transfer Checkpoint 2026-03-17

## Purpose

After aligning the fixed-split audio benchmark to the current semantic mainline, the next question was:

- can aligned audio help the harder ticker-held-out `after_hours` semantic line when the `Q&A` bottleneck is set to the transfer-friendly `lsa=4` setting?

This is the most relevant audio test if the project wants a transfer-aware extension rather than another same-panel fixed-split gain.

## Design

New script:

- `scripts/run_afterhours_audio_unseen_ticker.py`

Reruns on clean `after_hours`:

1. generic aligned audio
   - `results/afterhours_audio_unseen_ticker_winsor_svd8_lsa4_real/`
2. role-aware mixed aligned audio
   - `results/afterhours_role_aware_audio_unseen_ticker_svd8_lsa4_real/`

Shared setting:

- ticker-held-out evaluation
- clean `exclude html_integrity_flag=fail`
- semantic bottleneck `lsa=4`
- aligned audio compressed to `SVD(8)`

## Main Findings

### 1. The hardest baseline still wins overall

Overall clean unseen-ticker result:

- `pre_call_market_only ≈ 0.99848`

That remains the strongest overall model in this transfer setting.

So audio still does **not** overturn the main transfer limitation.

### 2. Generic aligned audio does not improve the transfer semantic line

Generic unseen-ticker audio:

- `pre_call_market + A4 + qna_lsa ≈ 0.99837`
- `+ generic aligned_audio_svd8 ≈ 0.99832`

Median ticker `R^2`:

- without audio: about `0.99519`
- with generic audio: about `0.99551`

Interpretation:

- overall it is slightly worse,
- median ticker is slightly better,
- but the effect is tiny and not a stable improvement.

### 3. Role-aware audio gives only a very small non-robust lift on the transfer semantic line

Role-aware unseen-ticker audio:

- `pre_call_market + A4 + qna_lsa ≈ 0.99837`
- `+ role_aware_aligned_audio_svd8 ≈ 0.99843`

Median ticker `R^2`:

- without audio: about `0.99519`
- with role-aware audio: about `0.99591`

So role-aware audio is directionally better than the no-audio semantic line here.

But the gain is extremely small and still sits below `pre_call_market_only ≈ 0.99848`.

That means this is, at best, a weak transfer-side hint rather than a new headline result.

### 4. The controls-plus-semantic audio line is even less convincing

For the richer control stack:

- `pre_call_market + controls + A4 + qna_lsa ≈ 0.99806`
- `+ role_aware audio ≈ 0.99810`

The overall number ticks up slightly, but the median ticker metric weakens.

So the richer transfer line does not give a clean stable audio win either.

## Updated Interpretation

The transfer-aware audio reading is now:

1. audio still does not beat the best ticker-held-out pre-call market baseline
2. generic aligned audio does not give a robust transfer gain
3. role-aware aligned audio may slightly stabilize the low-rank semantic line
4. but that lift is too small and too fragile to support a strong audio-transfer claim

So the safe current story is:

- audio remains a useful supporting modality on fixed-split controls / `A4` lines
- transfer-aware audio is currently exploratory, not yet publishable as a core contribution

## Best Next Step

If we keep the audio branch alive, the next credible route is not more view-splitting.

It is one of:

- supervised or prior-aware audio selection on top of the transfer-friendly semantic bottleneck
- or a cleaner statement that audio is an auxiliary fixed-split modality while the main transferable contribution remains semantic bottleneck control rather than audio
