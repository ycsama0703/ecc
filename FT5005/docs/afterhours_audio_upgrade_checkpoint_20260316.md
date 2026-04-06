## After-Hours Audio Upgrade Checkpoint

This note records the first real benchmark round for the upgraded `A3 + A4` aligned acoustic pipeline on the clean `after_hours` line.

### Assets built

- `scripts/build_a4_aligned_acoustic_features.py` was used to extract sentence-level `openSMILE eGeMAPSv02 Functionals` features on the clean `after_hours` subset.
- The production pilot asset is:
  - `results/audio_sentence_aligned_afterhours_clean_parallel_real/event_aligned_acoustic_features.csv`
- Build summary:
  - `172` clean `after_hours` events
  - `53,873` strict sentence rows
  - `0` extraction failures
  - `88` sentence-level acoustic fields
  - event-level robust aggregation built from sentence-level features

### Benchmark setup

- Script:
  - `scripts/run_afterhours_audio_upgrade_benchmark.py`
- Sample:
  - clean `after_hours` only
  - matched rows with `panel + features + old audio proxy + aligned audio`
  - split sizes: train `89`, val `23`, test `60`
- Target:
  - `shock_minus_pre`
- Baseline family:
  - same-ticker residual forecasting on top of the current pre-call market baseline

### Main result

The first raw aligned-audio result was not good enough:

- `pre_call_market + controls`: test `R^2 = 0.919`
- `+ old audio proxy`: test `R^2 = 0.836`
- `+ raw aligned audio (winsor_mean)`: test `R^2 = 0.888`
- `+ raw aligned audio + A4`: test `R^2 = 0.891`

So the straightforward "replace old proxy with raw aligned acoustic aggregates" story is false on this sample.

### What fixed it

The bottleneck was not the aligned audio source itself, but the high-dimensional representation.

When the `92` aligned acoustic inputs (`winsor_mean + summary`) were compressed to `SVD(8)`:

- `pre_call_market + controls + aligned_audio_svd8`: test `R^2 = 0.925`
- this beats:
  - `pre_call_market + controls = 0.919`
  - `pre_call_market + controls + raw aligned audio = 0.888`
  - `pre_call_market + controls + old audio proxy = 0.836`
- paired testing versus the old audio proxy is strong:
  - bootstrap mean `R^2` gain about `+0.108`
  - permutation `p(MSE) = 0.0025`
  - permutation `p(MAE) = 0.00075`
- paired testing versus raw aligned audio is also strong:
  - bootstrap mean `R^2` gain about `+0.046`
  - permutation `p(MSE) = 0.00275`

This is the clearest evidence so far that the upgraded audio line is viable, but only after dimensionality control.

### A4-centered line

The compressed aligned-audio result is also useful on the A4-centered branch:

- `pre_call_market + controls + A4 = 0.897`
- `+ raw aligned audio = 0.891`
- `+ aligned_audio_svd8 = 0.920`
- paired test for `A4` branch:
  - delta `R^2` about `+0.023`
  - permutation `p(MSE) = 0.0475`

This is important because it means the current paper story can now say:

- raw aligned acoustic aggregates are too noisy,
- but low-rank aligned acoustic factors do add value on top of the clean `after_hours` `A4` line.

### What did not work

- `duration_weighted_mean` alone is materially worse than `winsor_mean`
- combining `winsor_mean + duration_weighted_mean` is also worse
- adding raw aligned audio on top of `A4 + qna_lsa` still hurts
- even compressed aligned audio does not improve the current `A4 + qna_lsa` line:
  - `A4 + qna_lsa = 0.880`
  - `A4 + qna_lsa + aligned_audio_svd8 = 0.877`

So the current best reading is not "audio helps everywhere." It is:

- audio helps most as a compact low-rank factor block on the `market + controls` and `market + controls + A4` branches,
- but it does not currently strengthen the semantic `A4 + qna_lsa` branch.

### Interpretation

The updated audio story is now much stronger and cleaner:

1. The old call-level chunk proxy is not the right final audio representation.
2. Raw sentence-aligned acoustic aggregates are also too high-variance for the current `after_hours` sample.
3. Low-rank aligned acoustic factors are the first audio representation that clearly beats both the old proxy and the raw aligned aggregates.
4. The defensible contribution is therefore not "we used audio", but:
   - `A4`-aligned acoustic extraction,
   - robust aggregation,
   - and low-rank compression that turns noisy sentence-level acoustics into a useful event-level signal.

### Recommended next step

The next audio branch should focus on:

- keeping `winsor_mean` as the retained raw aggregate
- treating `SVD(8)` style compression as the default aligned-audio representation
- testing whether role-aware acoustic compression or Q&A-only acoustic compression can help the semantic branch without reintroducing high-dimensional variance
