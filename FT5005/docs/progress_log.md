# Progress Log

## 2026-03-14

### Migration recovery audit
- Re-scanned the current server after the storage migration.
- Confirmed the canonical project root is `/media/volume/dataset/xma8/work/icaif_ecc_news_attention`.
- Confirmed `/ocean/projects/cis250100p/xma8/icaif_ecc_news_attention` is a compatibility symlink to that canonical path.
- Confirmed `/home/exouser/ACM_ICAIF.txt` is only a historical planning note and should not be treated as the authoritative project workspace.
- Added `docs/server_migration_audit_20260314.md` to record the current server inventory, canonical paths, and newly identified research risks.

### Critical implementation finding
- Found a regime-labeling bug in `scripts/build_modeling_panel.py`:
  - `scheduled_hour_et` had been truncated to an integer hour,
  - so `09:30` events were incorrectly labeled as `pre_market` in downstream regime scripts because those scripts use the cutoff `hour < 9.5`.
- Fixed the panel builder so `scheduled_hour_et` now preserves fractional hours such as `8.5`, `9.5`, and `17.0`.
- Rebuilt `results/panel_real/event_modeling_panel.csv` from the corrected panel script.

### Measured impact of the timing fix
- Old regime counts on the `553`-event panel were:
  - `after_hours=204`,
  - `pre_market=293`,
  - `market_hours=56`.
- Corrected regime counts are now:
  - `after_hours=204`,
  - `pre_market=273`,
  - `market_hours=76`.
- `20` events changed regime under the corrected logic.
- All currently identified changed events are `GS` calls scheduled at `09:30`.
- The previously reported off-hours split of `train=248`, `val=82`, `test=167` should now be treated as stale; the corrected off-hours subset is `train=239`, `val=78`, `test=160`.

### Interpretation update
- The migration did not uncover missing project files; it uncovered a stronger foundation issue:
  - a key regime field used by the strongest reported result was discretized too aggressively.
- Reconstructed a local modeling environment at `/home/exouser/.venvs/icaif_ecc_news_attention` with:
  - `numpy 2.4.3`,
  - `scipy 1.17.1`,
  - `scikit-learn 1.8.0`.
- Added `requirements-modeling-min.txt` as the first repo-managed minimal environment file for the core rerun stack.
- Re-ran the most affected outputs after the panel fix:
  - `results/identity_classical_baselines_real/identity_classical_baseline_summary.json`,
  - `results/target_variant_experiments_real/target_variant_summary.json`,
  - `results/regime_residual_experiments_real/regime_residual_summary_shock_minus_pre.json`,
  - `results/regime_subset_experiments_real/regime_subset_summary_shock_minus_pre_after_hours-pre_market.json`.
- Updated quantitative reading:
  - the full-sample target redesign result is essentially unchanged,
  - `shock_minus_pre` still reaches about `0.890` test `R^2` with `residual structured + extra + qna_lsa + qa_benchmark`,
  - the corrected off-hours subset remains very strong but drops slightly from about `0.901` to about `0.897`,
  - the strongest off-hours corrected split is now `train=239`, `val=78`, `test=160`.
- Most important storyline change after the rerun:
  - the old claim that a simple global residual model is clearly safer than regime-specific fitting no longer holds cleanly.
  - On the corrected panel for `shock_minus_pre`, the global residual model reaches about `0.842` overall test `R^2`, while the regime-specific and hybrid selections reach about `0.858`.
  - The corrected regime tradeoff is now more nuanced:
    1. after-hours improves under regime-specific fitting,
    2. market-hours improves dramatically,
    3. pre-market weakens.
- Added `scripts/run_signal_decomposition_benchmarks.py` to explicitly separate:
  - market-state signal,
  - non-market ECC text and timing signal,
  - audio signal,
  - and combined bundles.
- This new decomposition result is the strongest current integrity finding in the project:
  - on full-sample `shock_minus_pre`, `market_only` already reaches about `0.904` test `R^2`,
  - `market_plus_controls` reaches about `0.909`,
  - `ecc_text_timing_only` is about `-0.022`,
  - `ecc_text_timing_plus_audio` is about `-0.176`,
  - `market_controls_plus_ecc_text_timing` is about `0.891`,
  - `market_controls_plus_ecc_plus_audio` is about `0.898`.
- The corrected off-hours subset shows the same basic pattern:
  - `market_only` is about `0.907`,
  - `market_plus_controls` is about `0.909`,
  - `ecc_text_timing_only` is about `-0.076`,
  - `market_controls_plus_ecc_text_timing` is about `0.898`,
  - `market_controls_plus_ecc_plus_audio` is about `0.907`.
- Interpretation after this benchmark:
  - the current high `shock_minus_pre` performance is primarily market-driven,
  - ECC features do not yet show stable incremental out-of-sample gains beyond the market-only residual benchmark,
  - and current audio features do not rescue that gap.
- Added a strict-clean sensitivity version of the decomposition benchmark by excluding `html_integrity_flag=fail`.
- Clean-sample result:
  - the core conclusion survives,
  - full-sample clean split becomes `train=244`, `val=74`, `test=156`,
  - off-hours clean split becomes `train=212`, `val=62`, `test=137`,
  - clean `market_plus_controls` still reaches about `0.911` test `R^2` on full-sample `shock_minus_pre`,
  - clean `ecc_text_timing_only` remains negative at about `-0.074`,
  - clean off-hours `market_plus_controls` remains about `0.911`,
  - clean off-hours `ecc_text_timing_only` remains negative at about `-0.054`.
- Updated reading after the clean-sample test:
  - the current market-dominant finding is not an artifact of the `html_integrity_flag=fail` rows,
  - so the next bottleneck is genuinely feature-design and research framing, not only transcript cleanliness.
- Added `docs/sota_novelty_flexible_roadmap_20260314.md` to formalize the fallback paths if the current DJ30 route stalls.
- Current recommended branching rule:
  - stay on the current path only if transferred `Q&A` quality or evasion signals beat the market-only benchmark on a defensible slice,
  - otherwise pivot toward a more reproducible public-data benchmark using SEC and open earnings-call resources.
- The migration audit also sharpened several remaining methodological gaps:
  1. strongest `shock_minus_pre` results are now clearly shown to be dominated by within-call market features,
  2. audio is still evaluated with coarse call-level chunk summaries rather than `A4`-aligned features,
  3. `79` rows with `html_integrity_flag=fail` still remain in the modeling sample,
  4. the repo still lacks a fuller end-to-end installable environment file even though the minimal rerun stack is now documented,
  5. current evaluation is still based mainly on one temporal split.

### `Q&A` v2 checkpoint and interpretation reset
- Expanded `scripts/build_qa_benchmark_features.py` into a richer `Q&A` v2 weak-label layer with added certainty, justification, temporal framing, attribution, drift, and evasion-style features.
- Wrote the refreshed feature table to `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`.
- Re-ran the most relevant `shock_minus_pre` stacks with `qav2` outputs:
  - `results/target_variant_experiments_qav2_real/target_variant_summary.json`
  - `results/regime_subset_experiments_qav2_real/regime_subset_summary_shock_minus_pre_after_hours-pre_market.json`
  - `results/regime_residual_experiments_qav2_real/regime_residual_summary_shock_minus_pre.json`
  - `results/signal_decomposition_qav2_real/`
- Updated reading from the decomposition reruns:
  - full-sample `ecc_text_timing_only` improves from about `-0.022` under `qav1` to about `0.095` under `qav2`,
  - off-hours `ecc_text_timing_only` improves from about `-0.076` to about `0.139` on the all-HTML slice,
  - but clean-sample robustness remains weak, with full-sample `ecc_text_timing_only` near `-0.005` and off-hours `ecc_text_timing_only` near `-0.085`,
  - and the best market baselines still dominate, with full-sample `market_only` about `0.904` and `market_plus_controls` about `0.909`.
- Updated reading from the headline reruns:
  - `qav2` does not improve the strongest mixed full-sample `shock_minus_pre` model, which drops from about `0.890` to about `0.884`,
  - `qav2` also does not improve the strongest off-hours mixed model, which drops from about `0.897` to about `0.891`,
  - but `qav2` does improve the global residual regime model from about `0.842` to about `0.882`,
  - while simultaneously weakening the regime-specific fit from about `0.858` to about `0.799`.
- Interpretation reset after the `qav2` reruns:
  - richer heuristic `Q&A` semantics can recover some event-specific ECC signal,
  - but they still do not deliver a robust incremental win over the strongest market-only residual benchmarks,
  - so `qav2` should be treated as a partial signal-recovery step rather than a paper-ready breakthrough.
- Added `scripts/build_qav2_checkpoint_report.py` plus:
  - `docs/qna_signal_checkpoint_20260314.md`
  - `results/research_checkpoints_real/qna_signal_checkpoint_20260314.json`
- The checkpoint report is now the recommended source for the current `qav1 -> qav2` comparison and for the gating rule on the next modeling branch.

### Hybrid architecture upgrade round
- Added `scripts/run_hybrid_architecture_experiments.py` to test whether stronger architecture can recover ECC incrementality on `shock_minus_pre`.
- New method families in that script:
  - nonlinear `market_controls_hgbr`,
  - global blend of market and full experts,
  - regime-gated blend of market and full experts,
  - positive linear stacking over base-expert residual predictions,
  - gated `HGBR` stacking with a small validation-time gate feature bundle.
- Added a defensive runtime fix inside the script so these small experiments default to single-threaded BLAS behavior instead of spawning hundreds of threads.
- Saved current result bundles to:
  - `results/hybrid_architecture_qav2_real/hybrid_architecture_shock_minus_pre_all_regimes_all_html.json`
  - `results/hybrid_architecture_qav2_real/hybrid_architecture_shock_minus_pre_all_regimes_exclude-fail.json`
  - `results/hybrid_architecture_qav2_real/hybrid_architecture_shock_minus_pre_after_hours-pre_market_all_html.json`
- Main result:
  - no tested architecture beats the strong market-side ridge baseline.
- Test-side highlights:
  - full-sample all-HTML `market_controls_ridge` remains about `0.909`,
  - full-sample all-HTML `regime_gated_market_full_blend` reaches about `0.908`,
  - full-sample clean `market_controls_ridge` remains about `0.911`,
  - full-sample clean `regime_gated_market_full_blend` reaches about `0.910`,
  - off-hours all-HTML `market_controls_ridge` remains about `0.909`,
  - off-hours all-HTML `regime_gated_market_full_blend` reaches about `0.906`.
- Negative but useful architecture findings:
  - `market_controls_hgbr` is much worse than ridge on every slice, only about `0.495` to `0.512` test `R^2`,
  - the positive stack mostly collapses back onto the `full_ridge` expert,
  - the gated `HGBR` stack overfits badly and falls to about `0.285` to `0.308` test `R^2`.
- Updated interpretation:
  - the current bottleneck is still signal and supervision quality rather than lack of model complexity,
  - so the next serious method jump should focus on stronger transferred pair-level `Q&A` labels or a cleaner benchmark path,
  - not on stacking more complex experts over the same weak ECC supervision.
- Added `docs/hybrid_architecture_checkpoint_20260314.md` as the current one-stop summary for this architecture round.

## 2026-03-16

### Corrected off-hours mainline closure
- Re-ran the fixed `off-hours + shock_minus_pre` ablation ladder on the corrected panel using the `qav2` feature table.
- Extended `scripts/run_offhours_shock_ablations.py`, `scripts/run_unseen_ticker_stress_test.py`, and `scripts/run_prior_gated_residual.py` with `--exclude-html-flags` so the corrected mainline can be stress-tested on a strict-clean slice.
- New corrected all-HTML main result:
  - split `train=239`, `val=78`, `test=160`
  - `prior_only` test `R^2 ≈ 0.191`
  - `residual_structured_only` test `R^2 ≈ 0.912`
  - `residual_structured_plus_extra` test `R^2 ≈ 0.871`
  - `residual_structured_plus_extra_plus_qna_lsa` test `R^2 ≈ 0.901`
  - `residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark` test `R^2 ≈ 0.891`
  - `...plus_audio` test `R^2 ≈ 0.901`
- New corrected clean-sample result after excluding `html_integrity_flag=fail`:
  - split `train=212`, `val=62`, `test=137`
  - `prior_only` test `R^2 ≈ 0.198`
  - `residual_structured_only` test `R^2 ≈ 0.913`
  - `residual_structured_plus_extra` test `R^2 ≈ 0.862`
  - `residual_structured_plus_extra_plus_qna_lsa` test `R^2 ≈ 0.907`
  - `residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark` test `R^2 ≈ 0.903`
  - `...plus_audio` test `R^2 ≈ 0.867`
- Updated significance reading on the corrected all-HTML split:
  - `structured_only` still strongly beats `prior_only`
  - adding the current `extra` block hurts significantly on holdout test
  - `Q&A LSA` significantly recovers a large part of that loss relative to `structured + extra`
  - but `structured_only` versus `structured + extra + qna_lsa` is not statistically decisive
  - `qa_benchmark` and `audio` still do not produce a robust incremental gain on top of the best corrected off-hours core

### Corrected robustness and harder generalisation
- Re-ran off-hours robustness analysis on the corrected all-HTML and corrected clean slices.
- Corrected all-HTML `residual_structured_only` remains strong:
  - overall test `R^2 ≈ 0.912`
  - by year:
    - `2023 ≈ 0.798`
    - `2024 ≈ 0.930`
    - `2025 ≈ 0.935`
  - by regime:
    - `after_hours ≈ 0.917`
    - `pre_market ≈ 0.714`
  - concentration:
    - top-name SSE gain share remains high at about `0.790`
    - but minimum leave-one-ticker-out `R^2` gain versus `prior_only` remains positive at about `0.446`
- Corrected clean `residual_structured_only` is similarly stable:
  - overall test `R^2 ≈ 0.913`
  - by year:
    - `2023 ≈ 0.807`
    - `2024 ≈ 0.923`
    - `2025 ≈ 0.957`
  - by regime:
    - `after_hours ≈ 0.915`
    - `pre_market ≈ 0.726`
  - minimum leave-one-ticker-out `R^2` gain remains positive at about `0.464`
- Re-ran the harder unseen-ticker stress test on the corrected all-HTML split:
  - `prior_only` overall test `R^2 ≈ -0.056`
  - `residual_structured_only` overall test `R^2 ≈ 0.991`
  - median ticker-level `R^2` for `residual_structured_only` is about `0.580`
- Updated reading:
  - the off-hours structured core is now much closer to a solid corrected-panel headline result,
  - the result is still concentrated in a small set of names,
  - but it is no longer fair to describe it as merely a pre-fix pilot artifact

### Contribution framing adjustment
- Re-ran the prior-gated residual prototype on the corrected off-hours split.
- Negative result survives:
  - corrected `structured` ridge remains about `0.912` test `R^2`
  - corrected `structured` gated version drops to about `0.806`
  - corrected `structured + extra + qna_lsa` ridge remains about `0.901`
  - corresponding gated version drops to about `0.885`
- This strengthens the paper positioning:
  - the contribution is not "we found a fancier architecture that wins,"
  - it is closer to:
    1. target and sample redesign under noisy timing,
    2. prior-aware and market-aware integrity evaluation,
    3. a robust corrected-panel finding that simple finance-aware structured ECC signals already explain off-hours volatility shock well,
    4. with richer `Q&A` semantics as secondary, partially incremental evidence.
- Added `docs/solid_storyline_contribution_memo_20260316.md` as the new paper-story memo for this corrected-panel stage.

### Feature-group closure inside off-hours
- Added `scripts/run_offhours_feature_group_ladder.py` to separate corrected off-hours signal into:
  - `pre_call_market`
  - `market`
  - `market + controls`
  - `A1/A2` transcript structure
  - `A4` observability or alignment structure
  - `Q&A` semantics via `qna_lsa`
- Ran the new ladder on:
  - pooled corrected off-hours all-HTML
  - pooled corrected off-hours clean
  - corrected `after_hours` all-HTML
  - corrected `after_hours` clean
  - corrected `pre_market` all-HTML
- New pooled reading:
  - corrected pooled off-hours remains strongly market-dominated,
  - `pre_call_market_only` already reaches about `0.902` all-HTML and `0.899` clean,
  - `market_plus_controls` reaches about `0.909` all-HTML and `0.911` clean,
  - adding the current `ecc_structure` block only lifts this to about `0.912` to `0.913`,
  - and that pooled incremental gain is not statistically decisive versus `market_plus_controls`.
- New regime-heterogeneity finding:
  - `pre_market` remains much weaker and should not be treated as equally strong evidence,
  - the strongest new positive ECC-specific signal is concentrated in `after_hours`.
- New corrected `after_hours` all-HTML result:
  - `pre_call_market_only` remains very strong at about `0.918`,
  - `market_plus_controls` drops to about `0.893`,
  - full `ecc_structure` alone is not enough there,
  - but `market_controls_plus_ecc_structure_plus_qna_lsa` jumps to about `0.926`,
  - and `qna_lsa` significantly improves over the all-HTML `after_hours` `ecc_structure` model with permutation `mse p ≈ 0.0028`, `mae p ≈ 0.006`.
- New corrected clean `after_hours` result:
  - `market_plus_controls` is about `0.875`,
  - `market_controls_plus_a4` improves to about `0.909`,
  - and this `A4` gain is significant with permutation `mse p ≈ 0.0075`, `mae p ≈ 0.0073`.
- Updated interpretation:
  - the best pooled off-hours claim is still mainly about strong corrected forecasting under honest priors and market baselines,
  - but the strongest limitation-driven novelty now lives in clean `after_hours`,
  - where `A4` observability or alignment structure adds a credible incremental gain,
  - while `Q&A` semantics appear most useful on noisier all-HTML `after_hours`.
- Added `docs/offhours_feature_group_checkpoint_20260316.md` as the focused checkpoint memo for this refined story.

### After-hours `A4` extension check
- Added `scripts/run_afterhours_a4_extensions.py` to test what actually helps on top of the useful coarse `A4` block inside `after_hours`.
- Compared:
  - `market + controls`
  - `market + controls + A4`
  - `market + controls + A4 + qna_lsa`
  - `market + controls + A4 + role_sequence`
  - `market + controls + A4 + weak_sequence`
  - and the corresponding `A4 + qna_lsa + sequence` hybrids
- New positive result:
  - all-HTML `after_hours`:
    - `market + controls + A4 + qna_lsa` reaches about `0.927`
  - clean `after_hours`:
    - `market + controls + A4 + qna_lsa` reaches about `0.935`
    - versus about `0.909` for `market + controls + A4`
    - clean permutation comparison gives `mse p ≈ 0.037`, `mae p ≈ 0.058`
- New negative but highly useful result:
  - adding either strict role-aware sequence bins or weak strict sequence bins on top of `A4` hurts materially on both all-HTML and clean `after_hours`
  - clean `A4 -> A4 + role_sequence` drops from about `0.909` to about `0.841`
  - clean `A4 -> A4 + weak_sequence` drops from about `0.909` to about `0.866`
  - both degradations are significant
- Strongest negative extension result:
  - even after adding `qna_lsa`, sequence still hurts
  - clean `A4 + qna_lsa` is about `0.935`
  - `A4 + qna_lsa + role_sequence` falls to about `0.839`
  - `A4 + qna_lsa + weak_sequence` falls to about `0.886`
- Updated interpretation:
  - the right refinement path is not higher-dimensional sequence modeling,
  - it is a compact `after_hours` model centered on `market + controls + A4 + Q&A semantics`,
  - which is much more publishable than a vague claim that "richer sequence structure should help eventually".
- Added `docs/afterhours_a4_extension_checkpoint_20260316.md` as the checkpoint memo for this extension round.

### Harder stress and stricter pre-call reading
- Ran a robustness pass for the new `after_hours` extension line using temporary output outside the full data volume because the project disk is currently full.
- New robustness reading on the fixed `after_hours` split:
  - `market + controls`, `market + controls + A4`, and `market + controls + A4 + qna_lsa` all remain strong by year and under leave-one-ticker-out removal,
  - but ticker concentration remains extreme, with roughly `82%` to `84%` of SSE gain still concentrated in `NVDA`.
- New unseen-ticker stress result for the `after_hours` extension line:
  - with the held-out ticker removed from train and validation, `market + controls` is strongest,
  - `A4` and especially `A4 + qna_lsa` do not beat that market-heavy unseen-ticker baseline.
- Updated interpretation from the unseen-ticker stress:
  - the compact `A4 + qna_lsa` line is promising for same-panel temporal generalisation,
  - but it is not yet a cross-firm transferable semantic signal in the harder ticker-held-out setup,
  - so the contribution should stay with honest within-panel incremental value rather than universal cross-firm semantic transfer.
- To test a stricter and more interpretable ECC increment, added `scripts/run_afterhours_precall_semantic_ladder.py` and evaluated a pre-call-only market baseline.
- New pre-call-only `after_hours` finding:
  - all-HTML:
    - `pre_call_market_only ≈ 0.918`
    - `pre_call_market + A4 + qna_lsa ≈ 0.919`
    - `pre_call_market + controls + A4 + qna_lsa ≈ 0.924`
  - clean:
    - `pre_call_market_only ≈ 0.917`
    - `pre_call_market + A4 + qna_lsa ≈ 0.927`
    - `pre_call_market + controls + A4 + qna_lsa ≈ 0.935`
- Most important strict-baseline result:
  - on the clean slice, `pre_call_market + A4` alone does not beat `pre_call_market_only`,
  - but `pre_call_market + A4 + qna_lsa` does,
  - with permutation support around `mse p ≈ 0.002`, `mae p ≈ 0.0028`,
  - and the same pattern remains when controls are included.
- Updated interpretation:
  - the most defensible current ECC increment is no longer "A4 helps everywhere,"
  - it is closer to:
    1. clean `after_hours`,
    2. relative to a pre-call market baseline,
    3. coarse `A4` observability plus compact `Q&A` semantics together add a real incremental gain.
- Added `scripts/run_afterhours_precall_unseen_ticker.py` to carry that same strict `pre-call` semantic line into the harder ticker-held-out setting.
- New strict `pre-call` unseen-ticker result:
  - all-HTML:
    - `pre_call_market_only ≈ 0.9985`
    - `pre_call_market + a4 + qna_lsa ≈ 0.9950`
    - `pre_call_market + controls + a4 + qna_lsa ≈ 0.9951`
  - clean:
    - `pre_call_market_only ≈ 0.9985`
    - `pre_call_market + a4 + qna_lsa ≈ 0.9962`
    - `pre_call_market + controls + a4 + qna_lsa ≈ 0.9937`
- Median ticker reading shows the same limitation:
  - all-HTML:
    - `pre_call_market_only ≈ 0.995`
    - `pre_call_market + a4 + qna_lsa ≈ 0.990`
    - `pre_call_market + controls + a4 + qna_lsa ≈ 0.982`
  - clean:
    - `pre_call_market_only ≈ 0.995`
    - `pre_call_market + a4 + qna_lsa ≈ 0.994`
    - `pre_call_market + controls + a4 + qna_lsa ≈ 0.969`
- Important ticker-level failure modes now visible on the clean unseen-ticker split:
  - `MSFT` drops from about `0.941` under `pre_call_market_only` to about `0.565` with `a4 + qna_lsa`, and to about `-0.528` with the richer control stack.
  - `AMGN` drops from about `0.752` under `pre_call_market_only` to about `0.640` with `a4 + qna_lsa`, and to about `0.264` with the richer control stack.
- Updated interpretation after the strict unseen-ticker test:
  - the fixed-split `pre-call + A4 + Q&A` result is still the cleanest within-panel contribution,
  - but the semantic increment is not yet cross-firm transferable,
  - so the next improvement target is transferability of the semantic block, not a higher fixed-split score.
- Ran a quick clean unseen-ticker bottleneck sweep on the strict pre-call line:
  - `lsa=8` lifts `pre_call_market + a4 + qna_lsa` to about `0.99825` overall test `R^2` with median ticker `R^2 ≈ 0.9953`,
  - versus about `0.9962` and median `≈ 0.9937` under the current `lsa=64`,
  - while `pre_call_market_only` still remains best at about `0.99848`.
- Updated diagnosis from the bottleneck sweep:
  - the unseen-ticker semantic failure is not purely "no transferable signal",
  - it is at least partly a representation-size and variance problem,
  - so the next sensible direction is lower-rank or supervised semantic transfer, not more feature dimensionality.
- Added `scripts/run_afterhours_precall_bottleneck_sweep.py` to sweep the clean strict pre-call mainline across `lsa = 4, 8, 16, 32, 64` under both fixed-split and unseen-ticker evaluation.
- Consolidated sweep result:
  - fixed split `pre_call_market + a4 + qna_lsa` is best at `lsa=64` with test `R^2 ≈ 0.927`,
  - unseen-ticker `pre_call_market + a4 + qna_lsa` is best at `lsa=4` with test `R^2 ≈ 0.99837`,
  - `lsa=32` partially closes the gap on both sides but still does not dominate both settings.
- Updated interpretation from the clean sweep:
  - the semantic block now shows a clear accuracy-versus-transferability tradeoff,
  - higher-rank semantics help the main temporal holdout result,
  - lower-rank semantics help cross-firm transfer,
  - so the next real research question is how to separate transferable `Q&A` content from firm-specific variance.
- Audited a collaborator-provided `openSMILE` demo archive for `A3 + A4` sentence-level acoustic extraction.
- The demo's useful advance over our current audio path is real:
  - it extracts `eGeMAPSv02` functionals at the `sentence_id` level using `A4` start and end times,
  - which is much closer to the paper story than our existing call-level chunk summary script.
- The demo is not production-ready as received:
  - it mixes notebook-only exploration with dead `wav2vec` code,
  - uses silent `pass` on extraction failures,
  - and outputs a mergeable sentence table but not a project-standard schema with `event_key`, QC flags, and event-level summaries.
- Added `scripts/build_a4_aligned_acoustic_features.py` to engineeringize that idea inside the repo:
  - inputs: `panel`, `A3`, and `a4_row_qc`,
  - supports `strict`, `broad`, or `all` QC slices,
  - writes sentence-level aligned acoustic features,
  - writes event-level robust aggregates including plain means, medians, winsorized means, and duration-weighted means,
  - records extraction failures instead of discarding them silently.
- Updated interpretation of audio-improvement space:
  - the biggest near-term upgrade is not a larger deep audio encoder,
  - it is replacing coarse call-level proxies with `A4`-aligned acoustic functionals that can be attached to sentence-aware and role-aware ECC experiments.
- Added `docs/afterhours_precall_semantic_checkpoint_20260316.md` as the consolidated checkpoint for the strict pre-call semantic line.

## 2026-03-12

### Data audit
- Re-checked the shared DJ30 package after download and extraction.
- Confirmed the working local path is `data/raw/dj30_shared`.
- Confirmed the package includes `Readme.docx`, the Compustat manuals, analyst tables, and all four ECC subfolders.
- Confirmed the current observed coverage is `A1=2146`, `A2=842`, `A3=796`, `A4=670`, and `D=25`.
- Confirmed `D` is missing `CRM`, `CVX`, `PG`, `TRV`, and `V`.
- Confirmed `D` includes extended-hours bars, which keeps within-call and immediate post-call targets feasible.

### Quality control
- Built and ran the initial manifest and QC scripts on the real package.
- Flagged `A2` HTML integrity via visible-length heuristics and group-relative size ratios.
- Treated `A4` as noisy alignment supervision rather than gold-standard timing, following Kelvin and Yu Ta's guidance.
- Current `A2` QC snapshot is `pass=722`, `fail=114`, `warn=6`.
- Current `A4` summary confirms a usable clean subset after filtering, with `670` event-level rows in the QC output.

### Event targets and panel
- Built event-level intraday targets from `A2 + A4 + D`.
- Recovered `8` malformed `A2` scheduled timestamps from `A1` headlines.
- Current target table has `553` event rows with usable intraday outcomes.
- Built the first joined modeling panel from targets, analyst tables, transcript stats, and QC summaries.
- The current panel also has `553` rows and is the working base for all downstream baselines.

### Baseline status
- Ran a first structured ridge baseline on `post_call_60m_rv`.
- Current split is `train=281`, `val=90`, `test=182`.
- Validation `R^2` is positive, but test `R^2` is negative, which is consistent with time drift and the limited structured feature set.
- This is the main reason the next step shifts toward richer text and timing features instead of more ad hoc structured controls.

### Modeling adjustment
- We currently assume no intraday benchmark ETF data is available because that question did not receive a response.
- We therefore keep the pilot target on raw intraday volatility proxies rather than market-adjusted abnormal returns.
- We also avoid depending on precomputed audio embeddings because the shared package only includes a small and incomplete set.
- The current practical modeling path is:
  1. build event-level transcript and timing features from `A1` and `A4`,
  2. add lightweight audio proxies from `A3`,
  3. run sparse text baselines and combined dense+sparse baselines,
  4. use those results to decide whether full audio embedding extraction is worth the additional engineering cost.

### Immediate next steps
- Build event-level text fields and timing or audio-proxy features for all `553` panel events.
- Run `TF-IDF + ridge` baselines on `Q&A` and full transcript text.
- Compare `structured only`, `text only`, and `structured + text + timing/audio-proxy` models.
- Use the results to refine the main paper framing and decide whether to invest in heavier audio feature generation.

### Richer feature layer
- Built `results/features_real/event_text_audio_features.csv` with `62` event-level fields plus reusable text blobs.
- The new table covers all `553` panel events and includes:
  - full transcript, `Q&A`, question-only, answer-only, and presenter-only text,
  - lexical rates for guidance, uncertainty, positive, and negative terms,
  - timing density, gap, and span features derived from `A4`,
  - lightweight audio proxies derived from `A3` file size and aligned call span.
- Current feature summary:
  - mean full transcript length is about `9,451` words,
  - mean `Q&A` length is about `5,599` words,
  - mean strict aligned span is about `3,807` seconds,
  - all `553` current panel events have both text and audio files available.

### First text and combined baselines
- Ran `TF-IDF + ridge` baselines on `post_call_60m_rv` using the same time split as the structured baseline.
- Pure text baselines are weak:
  - `Q&A tfidf only` validation `R^2` is about `-0.037`,
  - `full transcript tfidf only` validation `R^2` is about `-0.039`.
- Combining raw `Q&A tfidf` with the existing structured panel does not help much by itself.
- The first useful gain appears when adding the new event-level timing and audio-proxy features:
  - `structured only` validation `R^2` is about `0.519`,
  - `structured + Q&A tfidf + extra event features` validation `R^2` improves to about `0.533`.
- Test `R^2` remains negative, but the combined model improves from about `-0.247` to about `-0.223`, which is still directionally useful at this stage.

### Updated interpretation
- Naive bag-of-words text alone is not enough for this target on the current `553`-event DJ30 panel.
- The current evidence is more supportive of a finance-style story in which timing quality, `Q&A` structure, uncertainty language, and delivery proxies matter more than raw topical words.
- The next adjustment should therefore prioritize:
  1. temporally safer validation and robustness checks,
  2. more finance-aware text aggregation beyond plain TF-IDF,
  3. lightweight, reproducible audio features before any heavier multimodal model.

### Dialogue and integrity extension
- Added a dialogue-aware feature layer on top of the event table:
  - `Q&A` pair overlap,
  - low-overlap share,
  - hedge, assertive, and forward-looking answer rates,
  - multi-part question share,
  - presenter-versus-`Q&A` uncertainty and guidance gaps,
  - an event-level evasiveness proxy.
- Re-ran the combined baseline after adding those features.
- Updated result:
  - `structured + Q&A tfidf + extra event features` validation `R^2` improved further to about `0.535`,
  - the corresponding test `R^2` improved to about `-0.216`.
- The top dense signals are now more consistent with the intended finance story: uncertainty, negativity, presenter-versus-`Q&A` contrast, and `Q&A`-pair behavior all appear in the coefficient ranking.

### Scheduled-time robustness
- Built and ran a scheduled-time offset sweep for `-5`, `-2`, `0`, `+2`, and `+5` minutes.
- For `post_call_60m_rv`, the `-2` minute target remains very close to the base version with correlation about `0.984`.
- For `within_call_rv`, the `-2` minute target remains even closer with correlation about `0.995`.
- The median absolute difference is still zero for some small offsets because many events keep the same 5-minute bins.
- Larger shifts create a small but important set of unstable outliers, so timing robustness should remain a core integrity section in the paper rather than a footnote.

### Storyline adjustment
- The main story should now be tightened to:
  - role-aware `Q&A`,
  - timing-aware volatility prediction,
  - noisy ECC timing as an explicitly tested constraint.
- This is stronger than a generic multimodal transcript model and better aligned with recent ECC `Q&A` benchmark work.

### Role-aware sequence baseline
- Built a first fixed-length sequence table from `A4` timed rows plus fuzzy sequential mapping into `A1` component roles.
- Current mapping quality is usable but not yet clean:
  - mean strict mapped-row share is about `0.50`,
  - mean broad mapped-row share is about `0.45`,
  - high-confidence row share is still low.
- First sequence baseline results:
  - `strict_sequence_only` remains weak,
  - `structured + strict_sequence` improves test `R^2` to about `-0.223`,
  - `broad_sequence_only` is unstable and currently not trustworthy.
- Interpretation:
  - role-aware sequence information does contain signal,
  - the strict subset is clearly safer than the broad subset,
  - current fuzzy component mapping is still too noisy for the sequence model to replace the best event-level baseline.

### Hybrid strict-sequence result
- Added a hybrid comparison that merges the best current dense event-level features with the strict sequence bins.
- Result:
  - `structured + extra` validation `R^2` stays around `0.535`,
  - `structured + extra + strict sequence` improves test `R^2` further to about `-0.207`,
  - but validation `R^2` falls to about `0.474`.
- A PCA-compressed strict sequence version does not recover that validation loss.
- Current judgment:
  - sequence information is probably real,
  - but the present `A4 -> A1` fuzzy mapping is still too unstable for a clean headline claim,
  - so the safest current paper positioning is still event-level finance-aware modeling first, strict sequence as an extension.

### Alternative sequence routes
- Tried two additional sequence routes so the project is not over-dependent on one mapping design.

Weak section or role route:
- Built a weakly supervised `A4`-only section and role tagger as a fallback.
- It runs reliably, but it performs worse than the current strict fuzzy role mapping.
- This means the fallback is useful as a sanity check, not as a stronger main model.

Sentence-level `A1` alignment route:
- Built a sentence-level `A1` alignment prototype intended to improve the coarse component mapping.
- Full all-event runs are computationally expensive, so I first checked it on a `120`-event sample.
- On that sample, mean strict mapped-row share is only about `0.19`, much lower than the current strict fuzzy mapping.
- The corresponding sample baseline also does not improve over the sample structured baseline.
- Current judgment:
  - this sentence-aligned route is not promising enough in its current form,
  - and it should not be prioritised ahead of the finance-aware event model.

### SOTA workflow review and gap analysis
- Added a full workflow-level SOTA memo in `docs/sota_workflow_gap_analysis.md`.
- The review confirms that the mainline should stay event-level and finance-aware rather than jumping to a giant multimodal foundation model.
- The most important newly identified integrity risk is identity confounding:
  - recent 2025 ACL evidence shows that earnings-call volatility models can be dominated by same-company priors,
  - so same-ticker and historical-volatility baselines now need to be treated as must-have controls rather than optional extras.
- The highest-ROI next improvements are now ordered as:
  1. identity-aware baselines and integrity checks,
  2. real audio features,
  3. `Q&A` weak labels from recent benchmarks such as `SubjECTive-QA` and `EvasionBench`,
  4. stronger market-side numeric baselines,
  5. sequence modeling only after labels improve.

### Identity-aware and classical baselines
- Added `scripts/run_identity_classical_baselines.py`.
- Ran a first hard-baseline suite on the `553`-event panel.
- Main result:
  - `same-ticker expanding mean` is currently the strongest hard test baseline with test `R^2` about `0.112`,
  - `OLS structured` on the raw target reaches test `R^2` about `0.075`,
  - `HAR-style event OLS` reaches test `R^2` about `0.033`.
- This materially changes the interpretation of the project:
  - the most relevant benchmark is now no longer the global mean or sparse text,
  - it is the same-ticker historical volatility prior.
- The simple recursive `AR(1)` route is unstable on this panel and should not be treated as the main classical baseline.

### Real audio features and dense multimodal ablations
- Added `scripts/build_real_audio_features.py` and `scripts/run_dense_multimodal_ablation_baselines.py`.
- Extracted real call-level audio features for all `553` matched events using short sampled chunks from the A3 mp3 files.
- The first real-audio result is negative but useful:
  - `audio-only` is weak,
  - `structured + real audio` does not materially beat the structured baseline,
  - so audio currently does not justify a strong standalone multimodal claim.
- Added a denser semantic text baseline using `TF-IDF + TruncatedSVD` as a stronger semantic compression baseline than raw sparse `TF-IDF`.
- These dense semantic text variants slightly improve the test side relative to the earlier sparse text baselines, but they still weaken validation stability.
- Current best result in this new dense multimodal family is `structured + extra + qna_lsa + real audio` with test `R^2` about `-0.208`, but validation falls to about `0.404`.

### Updated interpretation after the new baselines
- The project is now more honest and more difficult:
  - same-ticker history is a real and strong benchmark,
  - current text and audio additions do not yet beat it out of sample.
- This means the next research question is sharper:
  - can dialogue-aware, role-aware, or benchmark-transferred `Q&A` features beat the same-ticker volatility prior,
  - not merely beat a weak text or mean baseline.

### Benchmark-inspired `Q&A` weak labels
- Added `scripts/build_qa_benchmark_features.py` to construct a new event-level feature block from `A1` question-answer pairs.
- The new block includes:
  - question and answer specificity,
  - early versus late overlap,
  - direct-answer share,
  - non-response share,
  - hedge and subjectivity rates,
  - numeric-answer rates,
  - question complexity.
- The extracted table covers all `553` joined panel events, with `552` events containing at least one `Q&A` pair and an average of about `9.63` pairs per event.

### Dense `Q&A` benchmark ablations
- Extended the dense ablation pipeline to include the new `qa_benchmark` block alongside structured, extra, dense semantic `Q&A`, and real audio bundles.
- Main result:
  - `structured + extra` remains the dense baseline reference at test `R^2` about `-0.216`,
  - `structured + extra + qa_benchmark` improves validation `R^2` to about `0.582`, but test `R^2` stays about `-0.216`,
  - `structured + extra + qna_lsa + qa_benchmark + real audio` reaches test `R^2` about `-0.208`, but still does not beat the same-ticker prior.
- Interpretation:
  - the benchmark-inspired `Q&A` block is scientifically meaningful and validation-helpful,
  - but it is not yet the missing ingredient that produces a clean out-of-sample win.

### Prior-as-feature versus residual-on-prior
- Added `scripts/run_prior_augmented_tabular_baselines.py` to test whether the same-ticker prior should simply be fed into the dense event model as another input.
- Result:
  - `prior + structured + extra` improves the earlier negative test result to about `-0.160`,
  - adding `qa_benchmark` or dense `Q&A LSA` does not improve the test side beyond that.
- However, this is still clearly worse than the raw same-ticker prior itself, whose test `R^2` remains about `0.112`.
- This means "prior as just another feature" is not the right formulation for the current panel.

- Added `scripts/run_prior_residual_ridge_baselines.py` to test the more defensible formulation:
  - first use the same-ticker expanding mean as the base forecast,
  - then train the ECC model only on the residual.
- Result:
  - `prior only` test `R^2` stays about `0.112`,
  - `residual structured + extra` test `R^2` is about `0.037`,
  - `residual structured + extra + qna_lsa + qa_benchmark` test `R^2` is about `0.008`.
- Interpretation:
  - residual-on-prior is a better and more honest experimental design than prior-as-input,
  - but the current ECC feature stack still does not beat the same-ticker expanding-mean baseline on holdout test.

### Practical storyline adjustment
- The main scientific hurdle is now extremely clear:
  - not "can ECC features beat a weak mean baseline",
  - but "can ECC features add stable event-specific value beyond same-ticker prior volatility?"
- The current answer is:
  - yes on validation and interpretability,
  - not yet on final holdout test.
- That is still useful because it narrows the next method search:
  - stronger benchmark-transferred `Q&A` signals,
  - better residual modeling,
  - or a better point-in-time fusion with the prior,
  - rather than more generic text or audio complexity.

### Fixed off-hours shock setting
- Locked the main pilot to:
  - `pre_market + after_hours`,
  - `shock_minus_pre = post_call_60m_rv - pre_60m_rv`,
  - residual-on-prior evaluation.
- Ran a strict ablation ladder with paired bootstrap and paired permutation tests.
- Main result:
  - `prior only` test `R^2` is about `0.196`,
  - `residual structured only` reaches about `0.916`,
  - adding `extra`, `qa_benchmark`, or `audio` does not improve on that holdout result.
- This materially changes the modeling strategy:
  - the paper should now prioritize disciplined incremental-value testing,
  - not larger multimodal unions.

### Off-hours robustness diagnostics
- Added a robustness pass specifically for the fixed main setting.
- Year-wise results for `residual_structured_only` remain strong:
  - `2023` test `R^2 ≈ 0.804`,
  - `2024` test `R^2 ≈ 0.933`,
  - `2025` test `R^2 ≈ 0.938`.
- Regime-wise results remain positive in both off-hours buckets:
  - `after_hours` test `R^2 ≈ 0.919`,
  - `pre_market` test `R^2 ≈ 0.727`.
- Added gain-concentration and leave-one-ticker-out checks against `prior_only`.
- Important integrity finding:
  - the improvement over `prior_only` is highly concentrated,
  - roughly `78.7%` of the SSE gain comes from `NVDA`,
  - but the model still beats `prior_only` after removing any single ticker, with minimum leave-one-ticker-out `R^2` gain still about `0.453`.
- Current reading:
  - the result is not a single-name artifact,
  - but ticker concentration is real and must be reported explicitly.

### Prior-gated residual prototype
- Built a first prior-gated residual prototype as a candidate method contribution.
- The prototype learns:
  - a residual branch,
  - multiplied by a learned sigmoid gate on the feature vector.
- Result:
  - validation `R^2` sometimes improves over ridge,
  - but holdout test does not.
- On the main `structured` bundle:
  - ridge test `R^2 ≈ 0.916`,
  - gated test `R^2 ≈ 0.906`.
- On `structured + extra + qna_lsa`:
  - ridge test `R^2 ≈ 0.907`,
  - gated test `R^2 ≈ 0.897`.
- Learned gate values are nearly constant around `0.15`, which suggests the prototype is mainly shrinking the residual branch rather than learning meaningful event-specific gating.
- Current judgment:
  - this is a useful negative result,
  - but it does not yet constitute a publishable method improvement.

### Harder unseen-ticker stress test
- Added a ticker-held-out stress test on the fixed `off-hours + shock_minus_pre` setting.
- Design:
  - remove one ticker entirely from train and validation,
  - train only on other tickers,
  - evaluate only on that unseen ticker in `2023+`.
- Because same-ticker history is unavailable in this setup, the baseline becomes a global train mean rather than a same-ticker prior.
- Result:
  - overall `prior_only` test `R^2 ≈ -0.053`,
  - overall `residual_structured_only` test `R^2 ≈ 0.991`,
  - median ticker-level `R^2` for `residual_structured_only` is about `0.534`.
- Important nuance:
  - cross-ticker generalisation is clearly not collapsing,
  - but it is still heterogeneous, with several individual tickers showing negative ticker-level `R^2`.
- Current reading:
  - the main off-hours shock result is not just a same-company memorisation artifact,
  - and the event-level structured design appears to encode a more transferable relationship than the earlier raw-target setup.

### Target redesign experiments
- Added `scripts/run_target_variant_experiments.py` to test whether the raw target itself is the real bottleneck.
- Compared four targets under the same residual-on-prior setup:
  - raw `post_call_60m_rv`,
  - `log_post_over_pre`,
  - `log_post_over_within`,
  - `shock_minus_pre = post_call_60m_rv - pre_60m_rv`.
- The result is decisive:
  - raw level target remains hard, with the same-ticker prior still best,
  - normalized or shock-style targets become much more learnable,
  - `shock_minus_pre` is the strongest by far.
- Main numbers:
  - raw target: prior-only test `R^2` about `0.112`, best residual model still below that
  - `log_post_over_pre`: residual test `R^2` about `0.588`
  - `log_post_over_within`: residual test `R^2` about `0.270`
  - `shock_minus_pre`: residual structured-plus-extra test `R^2` about `0.879`, and adding dense `Q&A` text plus benchmark-inspired `Q&A` signals lifts this to about `0.890`
- This is the biggest conceptual improvement so far:
  - the ECC feature stack may not be best suited to predicting absolute volatility level,
  - but it is much better suited to predicting incremental volatility shock relative to the pre-call state.

### Timing-regime experiments
- Added `scripts/run_regime_residual_experiments.py` to test whether the shock target behaves differently for pre-market, market-hours, and after-hours calls.
- Using the strong `shock_minus_pre` target:
  - the global residual model already strongly beats the prior overall,
  - and it is especially strong for after-hours events.
- Test-side regime results for the global residual model:
  - after-hours `R^2` about `0.889`,
  - pre-market `R^2` about `0.446`,
  - market-hours `R^2` still negative and unstable.
- A fully regime-specific fit overfits:
  - it improves validation,
  - but it weakens overall holdout test relative to the simpler global model.
- Interpretation:
  - regime matters a lot scientifically,
  - but the current sample is not large enough to justify fully separate models for every regime.

### Off-hours subset experiments
- Added `scripts/run_regime_subset_residual_experiments.py` to test the cleanest subset suggested by the regime analysis.
- Restricting to `pre_market + after_hours` and keeping the `shock_minus_pre` target produces the strongest result in the project so far.
- On that subset:
  - split sizes are `train=248`, `val=82`, `test=167`
  - prior-only test `R^2` is about `0.196`
  - residual dense test `R^2` is about `0.837`
  - residual dense plus `Q&A LSA` test `R^2` reaches about `0.901`
- This is the first setting where the project clearly has a strong and defensible out-of-sample result rather than only a promising internal direction.

### Updated main adjustment
- The best current paper direction is no longer:
  - predict raw `post_call_60m_rv` for all calls.
- It is now much closer to:
  - predict off-hours post-call volatility shock,
  - relative to the pre-call baseline,
  - using finance-aware event-level ECC features and dense `Q&A` semantics.
- This is both more financially interpretable and much more empirically successful on the current DJ30 pilot.

### Strict off-hours ablation and significance tests
- Added `scripts/run_offhours_shock_ablations.py` to lock the setting to:
  - `pre_market + after_hours`
  - target = `shock_minus_pre`
  - residual-on-prior forecasting
- This script now produces:
  - a full ablation ladder,
  - test-set predictions for every model,
  - paired bootstrap confidence intervals,
  - paired sign-permutation p-values for error reductions.

### Main ablation result
- The fixed-setting ablation produced an important correction to the current intuition:
  - the best holdout model is not the most complex one.
- Test results:
  - `prior only`: `R^2 ≈ 0.196`
  - `residual structured only`: `R^2 ≈ 0.916`
  - `residual structured + extra`: `R^2 ≈ 0.874`
  - `residual structured + extra + qna_lsa`: `R^2 ≈ 0.907`
  - `residual structured + extra + qna_lsa + qa_benchmark`: `R^2 ≈ 0.900`
  - `... + audio`: `R^2 ≈ 0.889`
- So the current test ranking is:
  1. structured-only residual,
  2. structured + extra + `Q&A LSA`,
  3. structured + extra + `Q&A LSA` + `qa_benchmark`,
  4. the heavier audio-augmented model,
  5. prior only far behind.

### Significance reading
- `structured only` vs `prior only`:
  - very strong improvement,
  - bootstrap CI for RMSE difference is strictly positive,
  - permutation p-values for both MSE and MAE gains are near zero.
- `structured only` vs `structured + extra`:
  - adding the current extra feature block hurts significantly on test.
- `structured + extra` vs `structured + extra + qna_lsa`:
  - dense `Q&A` semantics recover a significant part of that damage.
- `structured only` vs `structured + extra + qna_lsa`:
  - the difference is not statistically decisive on test.
- `qa_benchmark` on top of `structured + extra + qna_lsa`:
  - no robust test-side gain.
- `audio` on top of that stack:
  - also no robust test-side gain.

### Reflection and adjustment
- This is exactly why the stricter ablation mattered:
  - once the target and timing subset are correctly specified,
  - the remaining problem is not "how do we add more modalities",
  - but "which signals are truly necessary and robust".
- The current answer is:
  - target design and sample design matter more than piling on new feature families,
  - finance-aware structured event signals carry most of the value,
  - dense `Q&A` semantics are useful but not yet dominant,
  - the current `qa_benchmark` and audio branches should be treated as secondary evidence, not the headline.

### Publishability gap review
- Added a formal submission-gap review in `docs/publishability_gap_checklist.md`.
- The updated judgment is explicit:
  - current version is a strong research pilot,
  - but still not ready for a serious ACL- or ICAIF-level submission.
- The largest remaining blockers are now clearly documented:
  - DJ30-scale external-validity limits,
  - insufficient method novelty,
  - target-selection skepticism risk,
  - and a multimodal headline that is still weaker than the event-design result itself.
- `docs/research_plan.md` and `docs/implementation_plan.md` were also rewritten so the project documents no longer point back to the earlier raw-volatility or generic multimodal framing.

### After-hours aligned-audio upgrade checkpoint
- Built a real clean `after_hours` aligned-audio asset under:
  - `results/audio_sentence_aligned_afterhours_clean_parallel_real/`
  - `172` events
  - `53,873` strict sentence rows
  - `0` extraction failures
- Added:
  - `scripts/run_afterhours_audio_upgrade_benchmark.py`
  - `docs/afterhours_audio_upgrade_checkpoint_20260316.md`
- On the matched clean `after_hours` sample (`train=89`, `val=23`, `test=60`):
  - `pre_call_market + controls` is about `0.919`
  - old call-level audio proxy is weak at about `0.836`
  - raw aligned acoustic `winsor_mean` improves over the old proxy to about `0.888`, but still loses to the no-audio control baseline
  - raw `duration_weighted_mean` and dual-aggregation variants are worse
- The important positive result is low-rank compression:
  - `winsor_mean -> SVD(8)` raises `pre_call_market + controls + audio` to about `0.925`
  - this beats no-audio controls (`0.919`), raw aligned audio (`0.888`), and old audio proxy (`0.836`)
  - permutation tests versus old proxy and raw aligned audio are significant on MSE
- On the `A4` branch:
  - `pre_call_market + controls + A4` is about `0.897`
  - `+ raw aligned audio` is worse at about `0.891`
  - `+ aligned_audio_svd8` improves to about `0.920`
  - paired MSE permutation test versus `A4`-only is about `p = 0.0475`
- On the `A4 + qna_lsa` branch:
  - aligned audio still does not help, even after compression
  - current interpretation: aligned audio is now a viable compact factor block for the market and `A4` lines, but not yet a proven semantic add-on.

### Role-aware audio checkpoint
- Added:
  - `scripts/build_role_aware_aligned_acoustic_features.py`
  - `docs/role_aware_audio_checkpoint_20260316.md`
- Built `results/role_aware_aligned_audio_afterhours_clean_real/` from the clean `after_hours` sentence-audio asset:
  - `172` events
  - `53,873` sentence rows
  - mean mapped-row share only about `0.182`
  - role counts are sparse, especially `question`
- Raw role-aware audio is still too noisy, but compressed role-aware audio is useful:
  - `pre_call_market + controls + A4 + role_aware_audio_svd8` reaches about `0.930`
  - this beats `A4`-only (`0.897`) and generic aligned-audio `svd8` (`0.920`)
  - paired MSE permutation test versus `A4`-only is about `p = 0.0025`
- The more important research result is on the semantic branch:
  - `A4 + qna_lsa + generic aligned_audio_svd8` is about `0.877`
  - `A4 + qna_lsa + role_aware_aligned_audio_svd8` improves to about `0.886`
  - direct paired comparison versus generic compressed audio gives `p(MSE) = 0.0055`
- Current interpretation:
  - generic compressed aligned audio is still the best plain market/audio factor block,
  - but role-aware compressed audio is the first audio representation that actually helps the `A4 + qna_lsa` storyline.
- Follow-up narrowing confirms the role-aware mixed branch is the right retained version:
  - `qa`-only compressed audio underperforms the role-aware mixed branch on the semantic line
  - `answer`-only compressed audio also underperforms
  - merged `whole-call + role-aware` compressed audio underperforms too
  - current keep/drop rule:
  - keep generic whole-call `aligned_audio_svd8` for the plain market branch
  - keep role-aware mixed `aligned_audio_svd8` for the `A4 + qna_lsa` branch
  - drop `qa`-only, `answer`-only, and merged whole-call-plus-role-aware variants from the main path

## 2026-03-17

### Repo-scope takeover and checkpoint hygiene
- Added `docs/repo_operating_protocol.md` to lock future work to this ICAIF repository only and require frequent git checkpoints.
- Updated `.gitignore` so large audio caches, sentence-level aligned-audio tables, and temporary panel subsets stay local while compact summaries remain repo-trackable.
- Committed and pushed the pending 2026-03-16 checkpoint batch to GitHub so the main after-hours extension round is no longer trapped in an uncommitted local working tree.

### Audio semantic-line recheck under the current fixed-split bottleneck
- The earlier audio benchmark used `qna_lsa lsa_components=16`, but the strict clean fixed-split semantic sweep later showed the preferred setting is `lsa=64`.
- To remove that mismatch, reran:
  - `results/afterhours_audio_upgrade_benchmark_winsor_svd8_lsa64_real/`
  - `results/afterhours_role_aware_audio_upgrade_benchmark_svd8_lsa64_real/`
- Generic compressed aligned audio still helps on the non-semantic lines:
  - `pre_call_market + controls` goes from about `0.919` to `0.925`
  - `pre_call_market + controls + A4` goes from about `0.897` to `0.920`
- Role-aware compressed audio is still strongest on the `A4` line:
  - `pre_call_market + controls + A4 + role_aware_audio_svd8 ≈ 0.930`
- But the semantic-line reading changes materially once `lsa=64` is restored:
  - `pre_call_market + controls + A4 + qna_lsa ≈ 0.935`
  - `+ generic aligned_audio_svd8 ≈ 0.927`
  - `+ role_aware_aligned_audio_svd8 ≈ 0.901`
- Updated interpretation:
  - audio remains a useful supporting modality on the controls and `A4` lines,
  - but it is not yet a stable improvement to the strongest fixed-split semantic branch,
  - so the earlier role-aware semantic gain should be treated as conditional on the old `lsa=16` matched setup rather than as the current mainline claim.
- Added `docs/afterhours_audio_lsa64_recheck_20260317.md` as the corrected checkpoint for this aligned comparison.

### Audio transfer recheck on the low-rank unseen-ticker line
- Added `scripts/run_afterhours_audio_unseen_ticker.py` to test whether aligned audio helps the transfer-friendly ticker-held-out semantic line.
- Reran clean unseen-ticker `after_hours` audio with `lsa=4` under:
  - generic aligned audio
  - role-aware mixed aligned audio
- New reading:
  - `pre_call_market_only ≈ 0.99848` still remains the strongest overall held-out-ticker model
  - generic audio does not improve the transfer semantic line:
    - `pre_call_market + A4 + qna_lsa ≈ 0.99837`
    - `+ generic aligned_audio_svd8 ≈ 0.99832`
  - role-aware audio gives only a tiny directional lift on that same low-rank semantic line:
    - `pre_call_market + A4 + qna_lsa ≈ 0.99837`
    - `+ role_aware aligned_audio_svd8 ≈ 0.99843`
    - median ticker `R^2` also nudges up from about `0.99519` to about `0.99591`
  - but that lift is still below `pre_call_market_only`, and it is too small to treat as a robust audio-transfer result
- Updated interpretation:
  - audio still looks strongest as a fixed-split supporting modality on the controls and `A4` lines,
  - while transfer-aware audio remains exploratory rather than mainline evidence.
- Added `docs/afterhours_audio_transfer_checkpoint_20260317.md` as the transfer-side checkpoint for this round.

### Observability-gated semantic transfer checkpoint
- Added `scripts/run_afterhours_observability_gated_semantics.py` to test a narrower transfer-side idea: only trust the `A4 + qna_lsa` semantic increment when `A4` observability is strong.
- Clean `after_hours` run saved under `results/afterhours_observability_gated_semantics_clean_real/`.
- Fixed temporal split result with `lsa=64` is unchanged:
  - `pre_call_market_only ≈ 0.9174`
  - `pre_call_market + A4 + qna_lsa ≈ 0.9271`
  - observability-gated semantic line also `≈ 0.9271`
  - validation chooses `a4_strict_row_share ≈ 0.5814`, and the test activation rate is `1.0`, so the fixed split simply keeps the semantic branch on.
- Ticker-held-out transfer with `lsa=4` is the more useful readout:
  - `pre_call_market_only ≈ 0.99848`
  - `pre_call_market + A4 + qna_lsa ≈ 0.99837`
  - observability-gated semantic line improves to `≈ 0.99843`
  - median ticker `R^2` also lifts from about `0.99519` to about `0.99534`
  - mean test activation rate is about `0.799`, so around one-fifth of held-out events now fall back to the safer pre-call market baseline.
- Updated interpretation:
  - the fixed-split headline still belongs to `A4 + compact Q&A semantics`,
  - but the transfer-side semantic increment should be treated as conditional on observability rather than uniformly trusted,
  - and this simple gate is a cleaner next-step idea than returning to heavier sequence structure.
- The effect is still too small to call a new headline result because:
  - it remains slightly below `pre_call_market_only`,
  - pooled unseen paired `p(MSE)` versus the ungated semantic line is about `0.110`,
  - and pooled unseen paired `p(MSE)` versus `pre_call_market_only` is about `0.615`.
- Added `docs/afterhours_observability_gated_semantics_checkpoint_20260317.md` as the narrative checkpoint for this transfer-side refinement.

### Observability-gated role-aware audio transfer checkpoint
- Added `scripts/run_afterhours_observability_gated_audio_unseen_ticker.py` to test whether the low-rank transfer-side role-aware audio hint becomes more useful when it is activated only under strong `A4` observability.
- Ran the matched clean `after_hours` unseen-ticker audio subset under:
  - `results/afterhours_observability_gated_role_aware_audio_unseen_ticker_lsa4_real/`
  - `lsa=4` semantics
  - role-aware aligned audio compressed to `SVD(8)`
  - gate feature `a4_strict_row_share`
- Main readout on the matched unseen-ticker sample:
  - `pre_call_market_only ≈ 0.998482`
  - ungated semantic line `≈ 0.998370`
  - ungated semantic + role-aware audio `≈ 0.998434`
  - observability-gated semantic line `≈ 0.998434`
  - observability-gated semantic + role-aware audio `≈ 0.998527`
- Median ticker `R^2` also edges up:
  - ungated semantic + role-aware audio `≈ 0.995907`
  - gated semantic + role-aware audio `≈ 0.995916`
- The audio gate is slightly more conservative than the semantic-only gate:
  - mean test activation rate is about `0.784` versus about `0.799` for the semantic gate
  - this is consistent with the idea that transfer-side audio should only be trusted under stronger observability than the text-only semantic increment.
- Updated interpretation:
  - this is the first current transfer-side extension on the matched aligned-audio subset that slightly exceeds `pre_call_market_only`,
  - but the lift is tiny and still not statistically convincing (`p(MSE) ≈ 0.752` versus `pre_call_market_only`),
  - so it should be treated as an exploratory but scientifically coherent extension rather than as the headline result.
- Added `docs/afterhours_observability_gated_audio_transfer_checkpoint_20260317.md` as the checkpoint note for this audio-gated transfer branch.

### Transfer reliability-gate family search
- Added `scripts/run_afterhours_transfer_reliability_gate_search.py` to benchmark whether richer transfer-side reliability gates actually improve on the current simple local `A4` observability rule for the matched role-aware audio branch.
- Ran the search under `results/afterhours_transfer_reliability_gate_search_role_aware_audio_lsa4_real/`.
- Compared on the matched clean `after_hours` unseen-ticker audio subset:
  - ungated semantic + role-aware audio,
  - simple local `a4_strict_row_share` gate,
  - local per-ticker single-feature search over a broader candidate set,
  - shared pooled logistic gate over compact reliability features,
  - shared pooled conjunctive gate.
- Overall test `R^2` ranking is now clear:
  - `pre_call_market_only ≈ 0.998482`
  - ungated semantic + role-aware audio `≈ 0.998434`
  - simple local `a4_strict_row_share` gate `≈ 0.998527`
  - shared logistic gate `≈ 0.998503`
  - local single-feature search `≈ 0.998475`
  - shared conjunctive gate `≈ 0.998441`
- Median ticker `R^2` gives the same ordering, with the simple local `A4` gate still best among the transfer-gated audio variants.
- The broader conclusion is useful: the next gain is unlikely to come from more gate complexity.
  - local multi-feature search overfits,
  - shared logistic gating is more stable but still slightly weaker than the simple local `A4` gate,
  - conjunctive gates add complexity without benefit.
- So the repo should now treat the current simple observability gate as the retained transfer mechanism, and focus next on improving the transferable signal inside that gate rather than making the gate itself more flexible.
- Added `docs/afterhours_transfer_reliability_gate_search_checkpoint_20260317.md` as the checkpoint note for this gate-family benchmark.

### Transfer-side `Q&A` signal benchmark under the fixed gate
- Added `scripts/run_afterhours_transfer_qa_signal_benchmark.py` to test whether stronger event-level weak-label `Q&A` quality or evasion blocks actually improve the retained matched transfer branch once the simple `A4` observability gate is held fixed.
- Ran the matched clean `after_hours` unseen-ticker benchmark under `results/afterhours_transfer_qa_signal_benchmark_role_aware_audio_lsa4_real/`.
- Compared under the same role-aware aligned-audio subset and `lsa=4` semantic setting:
  - retained low-rank semantic line,
  - compact weak-label `qa_quality_core`,
  - compressed `qa_benchmark_svd`,
  - semantic plus each weak-label block,
  - and semantic plus weak-label block plus role-aware audio.
- Overall pooled test `R^2` says the retained best branch does not change:
  - gated `qna_lsa + role_aware_audio ≈ 0.998527`
  - `pre_call_market_only ≈ 0.998482`
  - gated `qna_lsa ≈ 0.998434`
  - gated `qa_benchmark_svd ≈ 0.998414`
  - gated `qna_lsa + qa_benchmark_svd ≈ 0.998342`
  - gated `qna_lsa + qa_benchmark_svd + role_aware_audio ≈ 0.998308`
  - gated `qna_lsa + qa_quality_core + role_aware_audio ≈ 0.998219`
- So the current weak-label event-level `Q&A` blocks do not replace the retained low-rank semantic line, and stacking them on top of the retained semantic or semantic+audio branch mostly makes the pooled transfer result worse.
- There is still some local heterogeneity worth noting:
  - gated `qa_benchmark_svd` beats `pre_call_market_only` on `5 / 9` held-out tickers,
  - and beats the retained gated semantic+audio branch on `6 / 9` held-out tickers,
  - but its losses on the remaining tickers are larger, so the pooled result stays weaker.
- Updated interpretation:
  - the repo should keep the simple gate,
  - keep the retained low-rank semantic + role-aware-audio transfer branch,
  - and treat the current heuristic weak-label `Q&A` additions as non-retained side evidence rather than as a new main extension.
- Added `docs/afterhours_transfer_qa_signal_checkpoint_20260317.md` as the checkpoint note for this transfer-side `Q&A` signal benchmark.

### Transfer expert-selection benchmark
- Added `scripts/run_afterhours_transfer_expert_selection.py` to test whether the complementary gated transfer experts can be used more selectively instead of simply stacking more weak-label features into a single branch.
- Ran the matched clean `after_hours` unseen-ticker benchmark under `results/afterhours_transfer_expert_selection_role_aware_audio_lsa4_real/`.
- Compared three transfer experts on the same matched `lsa=4` subset:
  - `pre_call_market_only`,
  - gated `qa_benchmark_svd`,
  - retained gated `qna_lsa + role_aware_audio`.
- Then benchmarked two combination rules on top of them:
  - validation-selected expert choice,
  - positive linear stack on the gated experts.
- Main pooled result:
  - retained gated semantic+audio expert `≈ 0.998527`
  - validation-selected expert `≈ 0.998546`
  - `pre_call_market_only ≈ 0.998482`
  - positive stack `≈ 0.998501`
- The validation-selected router picks:
  - retained gated semantic+audio expert on `6 / 9` folds,
  - gated `qa_benchmark_svd` expert on `3 / 9` folds,
  - and never falls back to `pre_call_market_only`.
- So the repo now has a coherent routing readout:
  - the complementary `qa_benchmark_svd` expert is not strong enough to replace the retained semantic+audio expert,
  - but it is useful often enough that selective expert choice slightly improves the matched transfer slice.
- That gain is still small and not statistically convincing versus the retained expert (`p(MSE) ≈ 0.185`), so it should be treated as an exploratory routing hint, not a new headline claim.
- The positive stack is not retained because it stays below the retained gated semantic+audio expert on pooled `R^2` and also worsens paired MAE.
- Added `docs/afterhours_transfer_expert_selection_checkpoint_20260318.md` as the checkpoint note for this routing benchmark.

### Transfer event-router benchmark
- Added `scripts/run_afterhours_transfer_event_router.py` to test whether the complementary gated transfer experts can be routed at the **event** level rather than only by held-out ticker.
- Ran the matched clean `after_hours` unseen-ticker benchmark under `results/afterhours_transfer_event_router_lite_role_aware_audio_lsa4_real/`.
- Kept the same aligned-audio subset and `lsa=4` transfer setting, but compared four routing levels:
  - retained gated semantic+audio expert,
  - previous validation-selected expert choice,
  - event-level logistic router,
  - event-level shallow tree router.
- The retained compact reliability feature set for the router was:
  - `a4_strict_row_share`,
  - `a4_strict_high_conf_share`,
  - `qa_pair_count`,
  - `qa_bench_direct_answer_share`,
  - `qa_bench_evasion_score_mean`,
  - `qa_bench_coverage_mean`,
  - `aligned_audio__aligned_audio_sentence_count`,
  - plus simple gate-activation and expert-disagreement terms.
- Main pooled result:
  - retained gated semantic+audio expert `≈ 0.998527`
  - validation-selected expert `≈ 0.998546`
  - event logistic router `≈ 0.998556`
  - event tree router `≈ 0.998567`
- The shallow event tree is therefore the strongest pooled routing extension on this matched transfer subset, and it also nudges the median held-out ticker `R^2` up to about `0.995940`.
- But the gain is still small and not statistically convincing:
  - versus retained gated semantic+audio expert, paired `p(MSE) ≈ 0.683`
  - versus validation-selected expert, paired `p(MSE) ≈ 0.829`
- The useful routing features remain interpretable rather than sprawling:
  - repeated tree usage falls on expert disagreement, `A4` observability / confidence, and `Q&A` coverage / evasion features.
- So the repo now has a stronger extension readout:
  - event-level reliability-aware routing beats brute-force stacking,
  - and slightly beats fold-level expert choice,
  - but it still remains an exploratory transfer extension rather than the paper headline.
- Added `docs/afterhours_transfer_event_router_checkpoint_20260318.md` as the checkpoint note for this routing benchmark.

### Conservative transfer-router benchmark
- Added `scripts/run_afterhours_transfer_conservative_router.py` to make the transfer router more conservative:
  - keep the fold-level selected expert as the default,
  - and only allow event-level overrides when router confidence is strong enough on validation.
- Ran the matched clean `after_hours` unseen-ticker benchmark under `results/afterhours_transfer_conservative_router_lite_role_aware_audio_lsa4_real/`.
- Retained the same aligned-audio matched subset and `lsa=4` transfer setting.
- Compared:
  - retained gated semantic+audio expert,
  - validation-selected expert,
  - conservative logistic override,
  - conservative shallow-tree override.
- Main pooled result:
  - retained gated semantic+audio expert `≈ 0.998527`
  - validation-selected expert `≈ 0.998546`
  - conservative logistic override `≈ 0.998553`
  - conservative tree override `≈ 0.998623`
- So the strongest current transfer routing result in the repo is now the **conservative shallow-tree override**.
- That gain is still exploratory rather than statistically secure:
  - versus validation-selected expert, paired `p(MSE) ≈ 0.318`
  - versus retained gated semantic+audio expert, paired `p(MSE) ≈ 0.165`
- The median held-out ticker `R^2` also edges up to about `0.995940`.
- The story remains coherent:
  - do not replace the selected expert by default,
  - only override when observability / reliability and expert-disagreement signals are strong enough.
- Repeated tree usage is still concentrated on intuitive features:
  - expert disagreement,
  - `A4` confidence / observability,
  - and `Q&A` coverage / evasion.
- A small local sweep over tree depth and broader feature sets did not change the retained choice:
  - the compact feature set with depth `2` (or equivalent depth `3`) stayed best,
  - while the broader full-feature router was weaker.
- Added `docs/afterhours_transfer_conservative_router_checkpoint_20260318.md` as the checkpoint note for this conservative routing benchmark.

### Transfer router signal-family benchmark
- Added `scripts/run_afterhours_transfer_router_signal_family_benchmark.py` to test which supervision families actually make the conservative transfer router better.
- Ran the matched clean `after_hours` unseen-ticker family benchmark under `results/afterhours_transfer_router_signal_family_benchmark_role_aware_audio_lsa4_real/`.
- Compared five conservative-router signal families:
  - `lite_baseline`,
  - `pair_core`,
  - `bench_directness`,
  - `hybrid_pair_bench`,
  - `hybrid_plus_text`.
- Main finding:
  - pair-only and benchmark-only families are each weaker than the hybrid families.
- Best shallow-tree family:
  - `hybrid_pair_bench` conservative tree `≈ 0.998624`
  - paired `p(MSE)` versus retained semantic+audio expert `≈ 0.052`
- Best pooled logistic family:
  - `hybrid_plus_text` conservative logistic `≈ 0.998628`
- So the current best transfer-side routing signal is **hybrid supervision**:
  - pair-level `Q&A` behavior,
  - benchmark-style directness / evasion,
  - and, for the best pooled logistic route, a small amount of compact text structure.
- That sharpens the method story:
  - the next gain is not coming from more raw feature stacking or heavier sequence models,
  - it is coming from better reliability supervision inside the conservative routing scaffold.
- Safe interpretation:
  - the best pooled result is now the hybrid-plus-text conservative logistic route,
  - the cleanest interpretable tree route is the hybrid pair-plus-benchmark family,
  - both remain exploratory rather than final claims.
- Added `docs/afterhours_transfer_router_signal_family_checkpoint_20260318.md` as the checkpoint note for this signal-family benchmark.

### Transfer router top-family tuning benchmark
- Added `scripts/run_afterhours_transfer_router_topfamily_tuning.py` to tune only the two strongest current hybrid router families instead of continuing to widen the family search.
- Ran the tuning benchmark under `results/afterhours_transfer_router_topfamily_tuning_role_aware_audio_lsa4_real/`.
- Tuned:
  - `hybrid_pair_bench` shallow-tree depth and minimum leaf fraction,
  - `hybrid_plus_text` logistic regularization `C`.
- Main outcomes:
  - the `hybrid_pair_bench` tree is very stable,
  - depth `2` is best but depths `3` and `4` are almost identical,
  - minimum leaf fraction barely changes the result,
  - best retained tree remains around `0.998624` with paired `p(MSE)` vs retained semantic+audio expert around `0.052`.
- The logistic family shows a clearer optimum:
  - the best pooled matched transfer result is now `hybrid_plus_text` conservative logistic with `C = 2`,
  - `R^2 ≈ 0.998636`,
  - which improves on the earlier untuned `C = 1` result.
- So the repo now has a cleaner split between:
  - the strongest interpretable route: stable `hybrid_pair_bench` shallow tree,
  - and the strongest pooled route: tuned `hybrid_plus_text` conservative logistic.
- This keeps the transfer-side story coherent:
  - hybrid supervision is real,
  - the tree route is stable rather than leaf-fragile,
  - and the logistic route can still be sharpened a little with modest regularization tuning.
- Added `docs/afterhours_transfer_router_topfamily_tuning_checkpoint_20260318.md` as the checkpoint note for this top-family tuning benchmark.

### Transfer router temporal confirmation benchmark
- Added `scripts/run_afterhours_transfer_router_temporal_confirmation.py` to test whether the two strongest retained hybrid transfer-router routes still hold up under broader temporal confirmation windows instead of just the latest matched split.
- Ran the temporal confirmation benchmark under `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`.
- Retained the same core transfer setting:
  - clean `after_hours`,
  - unseen-ticker evaluation,
  - `lsa=4`,
  - role-aware aligned audio `SVD(8)`,
  - matched aligned-audio subset.
- Compared the two retained hybrid routes across three rolling temporal windows:
  - `val2020_test_post2020`,
  - `val2021_test_post2021`,
  - `val2022_test_post2022`.
- Retained candidate routes:
  - `hybrid_pair_bench` conservative shallow tree,
  - tuned `hybrid_plus_text` conservative logistic with `C = 2`.
- Main result:
  - the later-window ordering still survives in `val2022_test_post2022`,
  - but it does **not** dominate every temporal split.
- Split-level picture:
  - earliest split `val2020_test_post2020`:
    - `pre_call_market_only ≈ 0.996679`
    - retained semantic+audio expert `≈ 0.996678`
    - hybrid pair tree `≈ 0.996676`
    - hybrid plus-text logistic `≈ 0.996636`
  - middle split `val2021_test_post2021`:
    - `pre_call_market_only ≈ 0.998826`
    - retained semantic+audio expert `≈ 0.998781`
    - selected expert `≈ 0.998782`
    - hybrid pair tree `≈ 0.998790`
    - hybrid plus-text logistic `≈ 0.998821`
  - latest split `val2022_test_post2022`:
    - `pre_call_market_only ≈ 0.998482`
    - retained semantic+audio expert `≈ 0.998527`
    - selected expert `≈ 0.998546`
    - hybrid pair tree `≈ 0.998624`
    - hybrid plus-text logistic `≈ 0.998636`
- The tree route remains the more stable interpretable route:
  - beats the selected expert in all `3/3` temporal windows,
  - beats the retained semantic+audio expert in `2/3` windows.
- The logistic route remains the higher-upside but more time-sensitive route:
  - beats selected in `2/3` windows,
  - beats retained in `2/3` windows,
  - but falls behind badly in the earliest confirmation split.
- The middle split gives the strongest within-extension confirmation:
  - logistic beats retained semantic+audio expert with paired `p(MSE) = 0.0`
  - logistic beats selected expert with paired `p(MSE) = 0.0`
  - yet still does not beat `pre_call_market_only`
- So the safest update is:
  - hybrid supervision remains the right transfer-router signal source,
  - the shallow tree is still the cleanest interpretable route,
  - the tuned hybrid-plus-text logistic route is still the strongest **late-period matched** route,
  - but the whole transfer-router extension is temporally sensitive and still does not consistently beat `pre_call_market_only` across broader time windows.
- Added `docs/afterhours_transfer_router_temporal_confirmation_checkpoint_20260318.md` as the checkpoint note for this temporal confirmation benchmark.

### Transfer complementary-expert benchmark
- Added `scripts/run_afterhours_transfer_complementary_expert_benchmark.py` to test the more important next question after temporal confirmation:
  - not whether the router can become more complex,
  - but whether we can build a **stronger complementary expert** than the current `qa_benchmark_svd` branch.
- The natural broader matched slice was not actually available in the current repo snapshot:
  - the clean `after_hours` aligned-audio subset already covers the full `172` clean `after_hours` rows with all current side inputs.
- So this checkpoint shifted to the stronger-signal-source question under the same retained transfer setting:
  - clean `after_hours`,
  - unseen-ticker evaluation,
  - `lsa=4`,
  - role-aware aligned audio `SVD(8)`,
  - same 172-row matched slice.
- Benchmarked complementary experts:
  - current `qa_benchmark_svd`,
  - `qa_pair_core`,
  - `hybrid_pair_bench`,
  - `hybrid_plus_text`,
  - `hybrid_pair_bench + aligned_audio`,
  - `hybrid_plus_text + aligned_audio`.
- For each family, the validation-selected route chose among:
  - `pre_call_market_only`,
  - retained semantic+audio expert,
  - that complementary expert.
- Main result:
  - the old `qa_benchmark_svd` complementary route is still the best one.
- Best selected route remains:
  - `validation_selected_transfer_expert__qa_benchmark_svd ≈ 0.998546`
- New families all stay below that:
  - `qa_pair_core` selected `≈ 0.998528`
  - `hybrid_pair_bench` selected `≈ 0.998529`
  - `hybrid_plus_text` selected `≈ 0.998510`
  - `hybrid_pair_bench + audio` selected `≈ 0.998521`
  - `hybrid_plus_text + audio` selected `≈ 0.998522`
- Selection counts make the bottleneck especially clear:
  - old `qa_benchmark_svd` route chooses retained semantic+audio expert in `6/9` folds and the `qa_benchmark_svd` expert in `3/9` folds,
  - the richer families usually choose the retained semantic+audio expert in `7/9` or `8/9` folds,
  - and the new complementary expert only wins `1–2/9` folds.
- So pair-level and hybrid direct experts are currently not competitive enough to replace the old complementary expert family.
- Audio does not rescue those direct hybrid experts:
  - `hybrid_plus_text + audio` selected route is actually worse than the retained semantic+audio expert,
  - paired `p(MSE) ≈ 0.031`.
- The safe interpretation is now sharper:
  - hybrid supervision is still useful,
  - but right now it is more useful as a **routing / reliability** signal family than as a direct standalone complementary expert family.
- Added `docs/afterhours_transfer_complementary_expert_checkpoint_20260318.md` as the checkpoint note for this complementary-expert benchmark.

### Transfer router consensus benchmark
- Added `scripts/run_afterhours_transfer_router_consensus_confirmation.py` to test a cleaner robustness idea built directly on top of the temporal confirmation outputs:
  - instead of expanding router families again,
  - require agreement between the retained hybrid tree and hybrid-plus-text logistic routes before making a stronger transfer deviation.
- Built two agreement-based consensus routes from the temporal confirmation outputs under `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`:
  - agreement fallback to the selected expert,
  - agreement fallback to the retained semantic+audio backbone.
- Wrote the new summary under `results/afterhours_transfer_router_consensus_confirmation_role_aware_audio_lsa4_real/`.
- Main pooled temporal result:
  - pre-call market only `≈ 0.997886`
  - retained semantic+audio expert `≈ 0.997878`
  - selected expert `≈ 0.997879`
  - pair tree `≈ 0.997901`
  - plus-text logistic `≈ 0.997899`
  - agreement fallback to selected `≈ 0.997899`
  - **agreement semantic-backbone consensus `≈ 0.997902`**
- So the strongest pooled temporal route in the current transfer-confirmation family is now:
  - **agreement-triggered semantic-backbone consensus**
- Importantly, that gain is meaningful relative to the weaker transfer baselines:
  - versus selected expert, pooled paired `p(MSE) ≈ 0.029`
  - versus retained semantic+audio expert, pooled paired `p(MSE) ≈ 0.0168`
- But it still does **not** significantly beat the best single routers:
  - versus pair tree, `p(MSE) ≈ 0.844`
  - versus plus-text logistic, `p(MSE) ≈ 0.941`
- The pattern is still useful because it is very conservative:
  - agreement rate `≈ 0.681`
  - semantic-backbone override rate only `≈ 0.101`
  - semantic-backbone QA share only `≈ 0.101`
- Split-level reading:
  - beats the selected expert in `3/3` temporal windows,
  - beats the retained semantic+audio expert in `3/3` windows,
  - beats pre-call market only in `2/3` windows,
  - but is not the late-window best single route because the tuned logistic remains slightly stronger there.
- This gives the transfer-side story a cleaner robustness extension:
  - keep the retained semantic+audio branch as the backbone,
  - and only allow stronger deviations when multiple hybrid reliability-aware routes agree.
- Added `docs/afterhours_transfer_router_consensus_checkpoint_20260318.md` as the checkpoint note for this agreement-based transfer consensus benchmark.

### Transfer router consensus fallback benchmark
- Added `scripts/run_afterhours_transfer_router_consensus_fallback_benchmark.py` to answer the next sharper question after the consensus checkpoint:
  - once the two retained hybrid router families agree or disagree,
  - **what is the best disagreement fallback policy?**
- Reused the temporal confirmation outputs under `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`.
- Compared six disagreement fallback policies under the same agreement trigger:
  - fall back to `pre_call_market_only`,
  - fall back to the retained semantic+audio backbone,
  - fall back to the selected expert,
  - fall back to the pair tree,
  - fall back to the plus-text logistic route,
  - or average tree and logistic on disagreement.
- Wrote the benchmark under `results/afterhours_transfer_router_consensus_fallback_benchmark_role_aware_audio_lsa4_real/`.
- Main pooled temporal result:
  - pre-call market only `≈ 0.997886`
  - pair tree `≈ 0.997901`
  - plus-text logistic `≈ 0.997899`
  - agreement + retained fallback `≈ 0.997902`
  - agreement + disagreement average `≈ 0.997903`
  - **agreement + pre-call-market fallback `≈ 0.997919`**
- So the strongest pooled temporal route in the current transfer-confirmation family is now:
  - **agreement-triggered abstention to `pre_call_market_only`**
- This route also beats the other fallback families more cleanly than expected:
  - versus pair tree, pooled paired `p(MSE) ≈ 0.049`
  - versus disagreement average, pooled paired `p(MSE) ≈ 0.0495`
  - versus agreement + retained semantic fallback, pooled paired `p(MSE) ≈ 0.068`
- It still does **not** significantly beat raw `pre_call_market_only` itself:
  - pooled paired `p(MSE) ≈ 0.197`
- So the safest updated interpretation is:
  - the transfer-side value is increasingly about **reliability-aware abstention**,
  - not just about more elaborate routing or richer expert families.
- The pooled QA-share pattern supports that reading:
  - agreement + pre-call fallback QA share `≈ 0.101`
  - selected fallback QA share `≈ 0.202`
  - tree fallback QA share `≈ 0.272`
  - logistic fallback QA share `≈ 0.249`
- That means the best current temporal route is the most conservative one:
  - use the transfer-side expert only when both retained router views agree,
  - otherwise abstain back to the strongest market baseline.
- Added `docs/afterhours_transfer_router_consensus_fallback_checkpoint_20260318.md` as the checkpoint note for this consensus fallback benchmark.

### Transfer router abstention diagnostics
- Added `scripts/run_afterhours_transfer_router_abstention_diagnostics.py` to directly test the core question left open by the consensus-fallback checkpoint:
  - does the transfer-side lift really concentrate on router-agreement events,
  - and is the disagreement slice really where abstention to `pre_call_market_only` is safest?
- Reused the temporal confirmation outputs under `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/` and split the pooled temporal test rows into four interpretable subsets:
  - `agreement`
  - `agreement_semantic`
  - `agreement_qa`
  - `disagreement`
- Wrote the diagnostics under `results/afterhours_transfer_router_abstention_diagnostics_role_aware_audio_lsa4_real/`.
- Pooled agreement subset (`175 / 257`, share `≈ 0.681`) result:
  - `pre_call_market_only ≈ 0.997974`
  - retained semantic+audio expert `≈ 0.997992`
  - selected expert `≈ 0.997999`
  - **agreement-supported prediction `≈ 0.998037`**
- So the pooled transfer lift really does concentrate on agreement events.
- The agreement subset also decomposes in a useful way:
  - agreement-semantic slice (`149`, share `≈ 0.580`) is best under the retained / selected semantic branch (`≈ 0.998117`)
  - agreement-`Q&A` slice (`26`, share `≈ 0.101`) is best under the `qa_benchmark_svd` expert (`≈ 0.994953`)
- That means the best reading is now more precise than “routing helps”:
  - when both retained router views agree on semantic, semantic is the right expert,
  - when both agree on `Q&A`, `Q&A` is the right expert,
  - and disagreement is the real abstention problem.
- Pooled disagreement subset (`82 / 257`, share `≈ 0.319`) result:
  - **`pre_call_market_only ≈ 0.997757`**
  - disagreement average `≈ 0.997723`
  - retained semantic+audio expert `≈ 0.997720`
  - pair tree `≈ 0.997719`
  - selected expert `≈ 0.997715`
  - plus-text logistic `≈ 0.997715`
  - `qa_benchmark_svd` expert `≈ 0.997714`
- Directional pooled tests on the disagreement subset also lean the same way:
  - versus retained semantic+audio expert: `p(MSE) ≈ 0.0825`, `p(MAE) ≈ 0.037`
  - versus selected expert: `p(MSE) ≈ 0.0573`, `p(MAE) ≈ 0.022`
  - versus pair tree: `p(MSE) ≈ 0.0685`, `p(MAE) ≈ 0.0313`
  - versus disagreement average: `p(MSE) ≈ 0.0573`, `p(MAE) ≈ 0.0275`
- So the pooled disagreement slice still looks most consistent with **abstain back to the market baseline**.
- Split-level nuance still matters:
  - the agreement story behaves as hoped in `2020` and `2022`, but not in `2021`
  - the disagreement story favors `pre_call_market_only` in `2020` and `2021`, while the latest split has a small `qa_benchmark_svd` edge
- This keeps the safe transfer-side conclusion honest:
  - the method story is now best framed as **reliability-aware abstention with temporal sensitivity**,
  - not as a universal claim that the transfer router always beats the market baseline.
- Added `docs/afterhours_transfer_router_abstention_diagnostics_checkpoint_20260318.md` as the checkpoint note for this new agreement/disagreement diagnostic.

### Project-state and paper alignment snapshot
- Added `docs/project_state_paper_alignment_20260318.md` as a repo-wide alignment note before the next research round.
- Purpose:
  - separate the strongest overall benchmark result from the strongest ECC-increment claim,
  - separate the fixed-split headline from the transfer-side extension,
  - and lock the next research priorities to the cleanest current paper story.
- Safe current hierarchy recorded in that note:
  - strongest overall benchmark = corrected pooled off-hours benchmark
  - strongest paper-safe ECC increment = clean `after_hours` + `A4 + compact Q&A semantics`
  - strongest current transfer-side method extension = **reliability-aware abstention**
- Safe next-step research guidance recorded there:
  - first tighten the temporal-sensitivity exceptions in the abstention story,
  - then convert the current evidence into a paper-facing scorecard,
  - and only then decide whether a new deeper branch is warranted.

### Transfer disagreement-slice diagnostics
- Added `scripts/run_afterhours_transfer_disagreement_slice_diagnostics.py` to explain the clearest remaining exception inside the abstention story:
  - why the latest temporal disagreement slice shows a tiny `qa_benchmark_svd` edge over `pre_call_market_only`.
- Reused the temporal confirmation outputs and isolated only disagreement events.
- Compared two disagreement directions:
  - `tree_sem_logistic_qa`
  - `tree_qa_logistic_sem`
- Wrote the outputs under `results/afterhours_transfer_disagreement_slice_diagnostics_role_aware_audio_lsa4_real/`.
- Main pooled disagreement conclusion does **not** change:
  - `pre_call_market_only ≈ 0.997757`
  - `qa_benchmark_svd ≈ 0.997714`
  - retained semantic+audio expert `≈ 0.997720`
- Both disagreement directions still favor `pre_call_market_only` when pooled:
  - `tree_sem_logistic_qa`: `0.998877` vs `qa ≈ 0.998817`
  - `tree_qa_logistic_sem`: `0.994192` vs `qa ≈ 0.994176`
- So the latest-window reversal is **not** a global inversion of the disagreement story.
- The latest-window disagreement slice (`n = 16`) is real but fragile:
  - `qa` beats `pre` on `7` events
  - `pre` beats `qa` on `6` events
  - `3` are ties
  - net `qa` advantage in summed event-level MSE is only `≈ 1.90e-09`
- The latest-window `Q&A` edge is highly concentrated:
  - top positive event share of positive `qa` gain `≈ 0.595`
  - top 3 positive events share `≈ 0.857`
  - biggest positive event is `CSCO_2025_Q1`
  - ticker-level positive concentration is mainly `CSCO`, `MSFT`, and `NKE`
- So the safest update is:
  - keep **reliability-aware abstention** as the main transfer-side extension,
  - do **not** replace disagreement fallback with a global `Q&A` rule,
  - and treat the latest-window `Q&A` edge as a small concentrated exception worth later selective calibration, not a structural reversal.
- Added `docs/afterhours_transfer_disagreement_slice_checkpoint_20260318.md` as the checkpoint note for this diagnostic.

### Transfer disagreement gain calibrator
- Added `scripts/run_afterhours_transfer_disagreement_gain_calibrator.py` to test a less rule-based, more interpretable next step after the disagreement-slice diagnostic.
- Instead of adding a new fallback heuristic, the script fits a **compact ridge gain regressor** over only six prediction-geometry features:
  - `qa_minus_pre_pred`
  - `sem_minus_pre_pred`
  - `sem_minus_qa_pred`
  - `abs_sem_minus_qa_pred`
  - `tree_choose_qa`
  - `logistic_choose_qa`
- Temporal protocol:
  - train on `2020` disagreement events
  - tune on `2021` disagreement events
  - refit on `2020 + 2021`
  - test on `2022` disagreement events
- Wrote the outputs under `results/afterhours_transfer_disagreement_gain_calibrator_role_aware_audio_lsa4_real/`.
- Main result:
  - validation calibrator `≈ 0.998896`, which only matches `pre_call_market_only`
  - held-out `2022` calibrator `≈ 0.999042`, exactly matching `pre_call_market_only`
  - raw `qa_benchmark_svd` on that same latest disagreement slice remains slightly higher at `≈ 0.999079`
- Most important behavioral result:
  - the compact learnable calibrator chooses `Q&A` on **`0 / 16`** latest disagreement events
- So the late-window `Q&A` pocket is currently:
  - real,
  - but not recoverable from earlier disagreement geometry by a compact learnable model,
  - which means it is **not yet a stable transferable signal**.
- The learned coefficients are also clean:
  - the main weight falls on prediction-geometry terms such as `qa_minus_pre_pred` and `sem_minus_qa_pred`
  - router identity votes carry much smaller weight
- This is a good negative result for the paper story:
  - it argues against adding a new `Q&A` exception rule,
  - and supports keeping **reliability-aware abstention** as the main transfer-side extension.
- Added `docs/afterhours_transfer_disagreement_gain_calibrator_checkpoint_20260318.md` as the checkpoint note for this learnable compact calibrator benchmark.

### Transfer learnable trust calibrator
- Added `scripts/run_afterhours_transfer_learnable_trust_calibrator.py` to test a broader and less rule-based follow-up question:
  - can the current agreement-based abstention logic itself be replaced by a compact learnable trust model?
- The script builds a smooth transfer candidate from the two strongest temporal router families:
  - if pair-tree and plus-text logistic agree, use the agreed expert
  - otherwise use the average of the two router outputs
- It then benchmarks two compact geometry-only calibrators against the current hard abstention rule:
  - a **gain gate** that predicts whether the candidate should replace `pre_call_market_only`
  - and a **soft trust** ridge model that predicts a blending weight between market baseline and transfer candidate
- Feature set is intentionally compact and interpretable:
  - `agreement`
  - `candidate_minus_pre_pred`
  - `qa_minus_pre_pred`
  - `sem_minus_pre_pred`
  - `selected_minus_pre_pred`
  - `tree_minus_pre_pred`
  - `logistic_minus_pre_pred`
  - `abs_pair_minus_logistic_pred`
- Temporal protocol:
  - train on `2020`
  - tune on `2021`
  - refit on `2020 + 2021`
  - test on `2022`
- Wrote the outputs under `results/afterhours_transfer_learnable_trust_calibrator_role_aware_audio_lsa4_real/`.
- Main result:
  - validation looks mildly encouraging:
    - gain gate `≈ 0.998841`
    - soft trust `≈ 0.998837`
    - hard abstention `≈ 0.998823`
  - but on the held-out latest window the current hard abstention rule still wins:
    - `pre_call_market_only ≈ 0.998482`
    - transfer candidate `≈ 0.998639`
    - **hard abstention `≈ 0.998640`**
    - gain gate `≈ 0.998637`
    - soft trust `≈ 0.998537`
- Interpretation:
  - the compact learnable gain gate comes close, but does not beat hard abstention
  - the softer continuous trust blend is worse still
  - so a broader learnable trust replacement does **not** currently improve on the repo's best conservative transfer rule
- This is a useful negative result:
  - it tests a more learnable and more interpretable alternative rather than adding more heuristic logic,
  - but still lands on the same conclusion:
  - **reliability-aware abstention remains the best compact transfer-side method currently supported by held-out evidence**
- Added `docs/afterhours_transfer_learnable_trust_calibrator_checkpoint_20260318.md` as the checkpoint note for this broader learnable-trust benchmark.

### Transfer agreement refinement
- Added `scripts/run_afterhours_transfer_agreement_refinement.py` to ask an even narrower, cleaner follow-up question:
  - if disagreement already falls back to `pre_call_market_only`, can the **agreement side alone** be improved by a compact learnable refinement?
- The benchmark keeps the current conservative abstention shell fixed:
  - disagreement events still use `pre_call_market_only`
  - only agreement events are learnably refined
- Two compact agreement-side calibrators were tested:
  - a gain gate that decides whether to trust the agreed expert
  - and a soft trust interpolator between the market baseline and the agreed expert
- Feature set is still intentionally compact and geometry-only:
  - `agreed_choose_qa`
  - `agreed_minus_pre_pred`
  - `qa_minus_pre_pred`
  - `sem_minus_pre_pred`
  - `selected_minus_pre_pred`
  - `pair_minus_pre_pred`
  - `logistic_minus_pre_pred`
  - `abs_pair_minus_logistic_pred`
- Temporal protocol:
  - train on `2020` agreement events
  - tune on `2021` agreement events
  - refit on `2020 + 2021` agreement events
  - test on full `2022`, with disagreement still fixed to market fallback
- Wrote the outputs under `results/afterhours_transfer_agreement_refinement_role_aware_audio_lsa4_real/`.
- Main result:
  - validation agreement refinement looks mildly promising:
    - best gain gate `≈ 0.997331`
    - best soft trust `≈ 0.997200`
    - compared with `pre_call_market_only ≈ 0.997191`
  - but on the held-out latest agreement subset:
    - `agreement_supported_pred ≈ 0.998591`
    - gain gate `≈ 0.998590`
    - soft trust `≈ 0.998453`
  - and on the full latest split:
    - `pre_call_market_only ≈ 0.998482`
    - `agreement_supported_pred ≈ 0.998624`
    - **`agreement_pre_only_abstention ≈ 0.998640`**
    - agreement gain gate `≈ 0.998639`
    - agreement soft trust `≈ 0.998514`
- Interpretation:
  - the targeted agreement gain gate comes very close, but still does **not** beat hard abstention
  - the softer trust blend again under-trusts the useful correction and is clearly worse
  - so even a narrower learnable refinement of the strongest transfer subset does not yet improve on the current simple abstention rule
- This is another good negative result:
  - it suggests the next research step should not be more router refinement,
  - but a search for a stronger **upstream transferable signal source**
- Added `docs/afterhours_transfer_agreement_refinement_checkpoint_20260318.md` as the checkpoint note for this agreement-side learnable refinement benchmark.

### Transfer agreement signal benchmark
- Added `scripts/run_afterhours_transfer_agreement_signal_benchmark.py` to push on the next cleaner research question after the agreement-refinement result:
  - can a **stronger compact upstream signal family** improve agreement-side transfer refinement better than more router-shell tuning?
- The benchmark keeps the shell fixed:
  - disagreement still falls back to `pre_call_market_only`
  - only agreement events are refined
  - model family stays a compact agreement gain gate
- Reused the temporal router outputs and merged them with all compact upstream side inputs:
  - `results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv`
  - `results/features_real/event_text_audio_features.csv`
  - `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`
  - `results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv`
- Coverage is complete on the temporal benchmark rows:
  - `257 / 257` rows have all side inputs
- Compared four compact feature families:
  - `geometry_only`
  - `lite_quality`
  - `hybrid_quality`
  - `geometry_plus_hybrid`
- Temporal protocol:
  - train on `2020` agreement events
  - tune on `2021` agreement events
  - refit on `2020 + 2021`
  - test on full `2022`
- Wrote the outputs under `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/`.
- Main result:
  - quality-only families underperform:
    - `lite_quality ≈ 0.998482`
    - `hybrid_quality ≈ 0.998470`
  - `geometry_only ≈ 0.998639`
  - best family is `geometry_plus_hybrid ≈ 0.998639`
  - but that best family only nudges past geometry-only by a tiny amount and does **not** beat hard abstention:
    - `agreement_pre_only_abstention ≈ 0.998640`
    - best family vs geometry-only: `p(MSE) = 1.0`
    - best family vs hard abstention: `p(MSE) ≈ 0.507`
- Interpretation:
  - compact upstream quality signals are scientifically relevant and interpretable,
  - but they are **not yet strong enough** to replace the current geometry-led abstention story
  - the bottleneck now looks less like “better routing” and more like “find a genuinely stronger transferable upstream signal source”
- Added `docs/afterhours_transfer_agreement_signal_checkpoint_20260318.md` as the checkpoint note for this upstream-signal agreement benchmark.


### Transfer responsiveness-factor benchmark
- Added `scripts/run_afterhours_transfer_responsiveness_factor_benchmark.py` to test a cleaner follow-up to the agreement-signal result:
  - can the current compact upstream `Q&A` / observability proxy block be distilled into **one learnable responsiveness factor** instead of another wider handcrafted family?
- The benchmark keeps the shell fixed:
  - disagreement still falls back to `pre_call_market_only`
  - only agreement events are refined
- Reused the same temporal router outputs plus the compact side-input assets, with complete join coverage:
  - `257 / 257` temporal rows have all side inputs
- Benchmarked four distilled factor families with two one-component builders (`PCA1`, `PLS1`) and two compact route variants (`factor_only`, `geometry_plus_factor`):
  - `responsiveness_core`
  - `responsiveness_plus_observability`
  - `directness_coverage_core`
  - `observability_directness_core`
- Wrote the outputs under `results/afterhours_transfer_responsiveness_factor_benchmark_role_aware_audio_lsa4_real/`.
- Main result:
  - best distilled route is **`responsiveness_plus_observability__pls1__geometry_plus_factor ≈ 0.998639314`**
  - this is essentially tied with the earlier `geometry_plus_hybrid ≈ 0.998638845` and `geometry_only ≈ 0.998638784`, but still below **`agreement_pre_only_abstention ≈ 0.998640168`**
  - the best `factor_only` route is **`directness_coverage_core__pca1__factor_only ≈ 0.998559`**, which is cleaner and numerically stronger than the earlier raw quality-only families, but still well below the geometry-led routes
- Interpretation:
  - the current upstream proxy pool does contain a coherent low-dimensional **responsiveness / directness / coverage / evasion** direction
  - so if we revisit the existing upstream block, a single learnable factor is now the cleanest representation
  - but that distilled factor still does **not** beat hard abstention, so the next real gain likely still needs a genuinely stronger transferable signal source rather than more recombinations of the same proxies
- Added `docs/afterhours_transfer_responsiveness_factor_checkpoint_20260318.md` as the checkpoint note for this distilled-latent transfer benchmark.


### Transfer QA-content factor benchmark
- Added `scripts/run_afterhours_transfer_qa_content_factor_benchmark.py` to test a genuinely different upstream source from the earlier responsiveness proxy block:
  - can the richer, mostly underused QA benchmark **content/accountability** features help transfer more than the old directness / evasion / coverage family?
- The benchmark keeps the shell fixed:
  - disagreement still falls back to `pre_call_market_only`
  - only agreement events are refined
- Reused the temporal router outputs and merged them with the full QA benchmark feature table plus `a4_strict_row_share` from the clean aligned panel; coverage is complete:
  - `257 / 257` temporal rows have both panel and full QA benchmark inputs
- Benchmarked five one-factor families with `PCA1` / `PLS1` and two compact route variants (`factor_only`, `geometry_plus_factor`):
  - `specificity_fidelity_core`
  - `commitment_attribution_core`
  - `content_accountability_core`
  - `specificity_directness_hybrid`
  - `specificity_plus_observability`
- Wrote the outputs under `results/afterhours_transfer_qa_content_factor_benchmark_role_aware_audio_lsa4_real/`.
- Main result:
  - best standalone route is **`content_accountability_core__pca1__factor_only ≈ 0.998592197`**, which is the strongest `factor_only` compact route seen so far in the repo and numerically above both `pre_call_market_only ≈ 0.998482062` and the previous best responsiveness `factor_only ≈ 0.998558665`
  - but the best geometry-coupled route is only **`content_accountability_core__pca1__geometry_plus_factor ≈ 0.998638784`**, which lands exactly at `geometry_only` and still below the previous best responsiveness `geometry_plus_factor ≈ 0.998639314` and hard abstention `≈ 0.998640168`
- Interpretation:
  - the richer QA benchmark table does contain a second low-dimensional semantic axis that is qualitatively different from the earlier responsiveness latent
  - but right now it behaves more like a **compressed complementary expert signal** than a better trust / abstention signal
  - so this checkpoint strengthens the view that current transfer progress needs either a new expert-integration idea or a genuinely stronger upstream signal source, not another small gating tweak
- Added `docs/afterhours_transfer_qa_content_factor_checkpoint_20260318.md` as the checkpoint note for this richer QA-content factor benchmark.


### Transfer factor-expert integration benchmark
- Added `scripts/run_afterhours_transfer_factor_expert_integration.py` to test the cleanest follow-up to the QA-content-factor result:
  - if the distilled responsiveness and content-accountability latents are real, can they work as **minimal direct complementary experts** rather than only as compact explanatory factors?
- Kept the design intentionally narrow:
  - reuse the clean aligned `after_hours` matched subset
  - build only three direct experts:
    - `pre_call_market_only`
    - `pre_call_market + A4 + responsiveness_factor`, gated by `a4_strict_row_share`
    - `pre_call_market + A4 + content_factor`, gated by `a4_strict_row_share`
  - then do per-ticker validation selection among those compact experts
- Reused the same matched assets and hard-abstention reference:
  - `results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv`
  - `results/features_real/event_text_audio_features.csv`
  - `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`
  - `results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv`
  - `results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/...`
- Coverage is complete on the matched subset:
  - `172 / 172` rows have all side inputs
- Wrote the outputs under `results/afterhours_transfer_factor_expert_integration_role_aware_audio_real/`.
- Main result:
  - `pre_call_market_only ≈ 0.998482062`
  - responsiveness factor expert `≈ 0.997590632`
  - content-accountability factor expert `≈ 0.997559927`
  - validation-selected compact factor expert `≈ 0.997580267`
  - hard abstention remains best at `≈ 0.998640168`
- The failure mode is informative:
  - validation selection chooses a factor expert for every held-out ticker (`5` responsiveness, `4` content, `0` pre-only)
  - but that selected route only beats `pre_call_market_only` on a minority of test tickers, so the compact factor experts are clearly overfitting validation rather than transferring robustly to the later window
- Interpretation:
  - this is a useful negative result, because it rules out the cleanest possible “minimal complementary-expert” upgrade
  - the distilled latents remain scientifically useful as compact explanatory axes
  - but they should **not** yet be promoted into the main transfer shell as direct experts
  - so the next gain still needs a stronger upstream transferable signal source, not another direct factor-expert integration trick
- Added `docs/afterhours_transfer_factor_expert_integration_checkpoint_20260318.md` as the checkpoint note for this minimal complementary-expert benchmark.


### Transfer role-text signal benchmark
- Added `scripts/run_afterhours_transfer_role_text_signal_benchmark.py` to test a genuinely new upstream source after the factor-expert result:
  - can raw role-specific `Q&A` text provide a stronger compact transfer signal than the current handcrafted proxy families?
- Reused the temporal transfer-confirmation shell and kept the protocol fixed:
  - train on `val2020_test_post2020` agreement rows
  - tune on `val2021_test_post2021`
  - refit on `2020 + 2021`
  - test on full `val2022_test_post2022`
  - disagreement still falls back to `pre_call_market_only`
- Attached `question_text` and `answer_text` from `results/features_real/event_text_audio_features.csv`; coverage is complete:
  - `257 / 257` temporal rows have both role-text fields
- Built one shared `TF-IDF + LSA(4)` basis over question and answer text, then benchmarked:
  - `question_role_lsa`
  - `answer_role_lsa`
  - `qa_role_gap_lsa`
  - `geometry_plus_answer_role`
  - `geometry_plus_role_gap`
  - `geometry_plus_dual_role`
  - with `geometry_only` as the current compact agreement-side reference
- Wrote the outputs under `results/afterhours_transfer_role_text_signal_benchmark_lsa4_real/`.
- Main result:
  - best standalone route is **`question_role_lsa ≈ 0.998621932`**
  - this is above `pre_call_market_only ≈ 0.998482062` and above the previous best standalone content-accountability factor `≈ 0.998592197`
  - but the best geometry-coupled route remains **`geometry_only ≈ 0.998638784`**, and hard abstention still remains best at `≈ 0.998640168`
  - answer-only and role-gap routes are weaker:
    - `answer_role_lsa ≈ 0.998553958`
    - `qa_role_gap_lsa ≈ 0.998474838`
- Interpretation:
  - this is the first strong evidence that **analyst-question semantics** are a cleaner transferable compact signal than the answer block or answer-question gap
  - but role-text still behaves more like a standalone complementary route than a better controller for the current transfer shell
  - so the transfer picture is now sharper: hard abstention remains the best method, while low-rank question-role text is the strongest new upstream compact signal
- Added `docs/afterhours_transfer_role_text_signal_checkpoint_20260320.md` as the checkpoint note for this role-text benchmark.


### Transfer role-semantic expert benchmark
- Added `scripts/run_afterhours_transfer_role_semantic_expert_benchmark.py` to test the cleanest direct follow-up to the role-text signal result:
  - if `question_role_lsa` is the strongest standalone compact transfer signal, can it replace pooled `qna_lsa` as the semantic core of the matched transfer expert?
- Reused the clean aligned matched `after_hours` unseen-ticker shell with complete side-input coverage:
  - `172 / 172` rows have features, old audio, aligned audio, and QA features
- Benchmarked direct gated experts:
  - `pre_call_market_only`
  - `pre_call_market + A4 + qna_lsa`
  - `pre_call_market + A4 + question_role_lsa`
  - `pre_call_market + A4 + answer_role_lsa`
  - `pre_call_market + A4 + qna_lsa + aligned_audio_svd`
  - `pre_call_market + A4 + question_role_lsa + aligned_audio_svd`
- Added a deliberately conservative selection layer over only three candidates:
  - `pre_call_market_only`
  - `qna_lsa + aligned_audio`
  - `question_role_lsa + aligned_audio`
- Wrote the outputs under `results/afterhours_transfer_role_semantic_expert_benchmark_role_aware_audio_real/`.
- Main result:
  - `pre_call_market_only ≈ 0.998482062`
  - `A4 + qna_lsa ≈ 0.997706362`
  - `A4 + question_role_lsa ≈ 0.997562102`
  - `A4 + answer_role_lsa ≈ 0.997589376`
  - `A4 + qna_lsa + aligned_audio ≈ 0.997785330`
  - `A4 + question_role_lsa + aligned_audio ≈ 0.997581610`
  - conservative validation-selected semantic-core route `≈ 0.997569753`
  - hard abstention remains clearly best at `≈ 0.998640168`
- Interpretation:
  - the previous role-text result does **not** carry over as a direct semantic-core replacement inside the current transfer expert shell
  - pooled `qna_lsa` remains the better direct semantic expert, while question-role text remains a stronger **standalone complementary signal**
  - even the more disciplined two-core validation selector still overfits and degrades late-window held-out performance, including a clear loss vs hard abstention (`p(MSE) ≈ 0.00175`)
- Added `docs/afterhours_transfer_role_semantic_expert_checkpoint_20260320.md` as the checkpoint note for this role-semantic replacement benchmark.


### Fixed-split role-semantic mainline benchmark
- Added `scripts/run_afterhours_role_semantic_mainline_benchmark.py` to test the clean mainline follow-up to the recent transfer role-text result:
  - if `question_role_lsa` is the strongest standalone compact transfer signal, does it also strengthen the fixed-split clean `after_hours` headline?
- Reused the strict fixed-split semantic setup:
  - clean `after_hours`
  - `shock_minus_pre`
  - train `<=2021`, val `2022`, test `>2022`
  - current pooled semantic reference remains `qna_lsa=64`
- Coverage is complete on the fixed-split subset:
  - train `89`, val `23`, test `60`
  - `172 / 172` rows have pooled `Q&A`, question, and answer text
- Benchmarked three compact role-semantic uses:
  1. direct replacement of pooled `qna_lsa` with `question_role_lsa` / `answer_role_lsa`
  2. raw stacking of role text on top of pooled `qna_lsa`
  3. orthogonalized stacking via a learnable residualization step from pooled `qna_lsa -> role_text`
- Wrote the outputs under `results/afterhours_role_semantic_mainline_benchmark_clean_real/`.
- Main result:
  - pooled `qna_lsa` still dominates the fixed-split semantic headline:
    - `pre_call_market + A4 + qna_lsa ≈ 0.9271`
    - `pre_call_market + controls + A4 + qna_lsa ≈ 0.9347`
  - direct role-semantic replacements are clearly worse:
    - `pre + A4 + question_role ≈ 0.8997`
    - `pre + A4 + answer_role ≈ 0.9142`
    - `pre + controls + A4 + question_role ≈ 0.9082`
    - `pre + controls + A4 + answer_role ≈ 0.9099`
  - raw question-role stacking also hurts the mainline:
    - `pre + A4 + qna + question ≈ 0.9220` vs `0.9271` (`p(MSE) ≈ 0.00325`)
    - `pre + controls + A4 + qna + question ≈ 0.9294` vs `0.9347` (`p(MSE) ≈ 0.00425`)
  - answer-role stacking is closer to neutral but still not an upgrade:
    - `pre + A4 + qna + answer ≈ 0.9267`
    - `pre + controls + A4 + qna + answer ≈ 0.9337`
  - orthogonalizing role text against pooled `qna_lsa` does **not** rescue it:
    - `pre + A4 + qna + question_resid ≈ 0.9049`
    - `pre + controls + A4 + qna + question_resid ≈ 0.8877`
    - `pre + A4 + qna + answer_resid ≈ 0.8952`
    - `pre + controls + A4 + qna + answer_resid ≈ 0.9034`
- The residualization metadata is useful:
  - question-role is much less reconstructable from pooled `Q&A` than answer-role (`val recon MSE ≈ 0.2404` vs `0.0378`)
  - so analyst-question text really does define a distinct semantic axis
  - but that distinct component still does **not** improve the fixed-split headline
- Interpretation:
  - this is a strong negative result that sharpens the story rather than weakening it
  - pooled `qna_lsa` remains the best compact semantic core for clean fixed-split `after_hours`
  - analyst-question semantics remain scientifically interesting, but they look more like a complementary transfer-side signal than a mainline additive feature
- Added `docs/afterhours_role_semantic_mainline_checkpoint_20260320.md` as the checkpoint note for this fixed-split role-semantic benchmark.


### Transfer question-role slice diagnostics
- Added `scripts/run_afterhours_transfer_question_role_slice_diagnostics.py` to answer the next clean research question after the transfer and fixed-split role-semantic negatives:
  - if analyst-question text is still the strongest standalone compact transfer signal, **where** is it actually helping?
- Reused the exact role-text transfer setup:
  - train `val2020_test_post2020`
  - validate `val2021_test_post2021`
  - test `val2022_test_post2022`
  - rebuild the same shared analyst-question / answer `TF-IDF + LSA(4)` basis
- Instead of adding a new model shell, the script diagnoses held-out latest-window behavior by:
  1. assigning each event to its dominant signed question-role component,
  2. comparing standalone `question_role_lsa` against `pre_call_market_only`, `geometry_only`, and hard abstention,
  3. measuring concentration, slice-level gains, ticker pockets, and top event-level wins / failures
- Wrote the outputs under `results/afterhours_transfer_question_role_slice_diagnostics_lsa4_real/`.
- Main result:
  - the pooled standalone gain over `pre_call_market_only` is real but very sparse:
    - question-role wins on only `10 / 60` held-out latest-window events
    - top `1` positive event contributes `≈ 46.7%` of the total positive gain mass
    - top `3` positive events contribute `≈ 89.6%`
    - top `5` positive events contribute `≈ 98.1%`
  - once the stronger transfer shell is available, question-role almost never wins:
    - mean MSE gain vs `geometry_only` is negative (`≈ -1.89e-10`)
    - mean MSE gain vs hard abstention is negative (`≈ -2.05e-10`)
    - win share falls to only `≈ 8.3%` vs `geometry_only` and `≈ 5.0%` vs hard abstention
- The positive pockets are still semantically meaningful:
  - `NVDA_2024_Q1` loads on a `data center / computing / nvidia` slice
  - `AAPL_2023_Q1` loads on an `iphone / china / data center` slice
  - `AMZN_2023_Q1` and `AMZN_2024_Q2` load on a `high performance / execution / compute / optimization` style slice
  - ticker-level average gains over `pre_call_market_only` are strongest for `NVDA`, `CSCO`, `AAPL`, and `AMZN`
- But the slice diagnostics also show why this should not become a new rule:
  - even the apparently “good” semantic slices still have low win shares
  - the same `high performance / execution / compute` pocket contains both strong positives and clear negatives (`AMZN_2024_Q1`, `AMZN_2024_Q3`, `NKE_2023_Q1`)
  - so the role-text signal looks more like a sparse event-level analyst-attention detector than a stable slice-wide controller
- Interpretation:
  - this is a useful research narrowing result, not just another diagnostic table
  - question-role semantics are still real and interpretable
  - but they should now be treated as a compact **diagnostic semantic axis**, not as a direct controller, mainline additive block, or new routing rule
- Added `docs/afterhours_transfer_question_role_slice_checkpoint_20260321.md` as the checkpoint note for this question-role slice diagnostic.


### Transfer question-role gate diagnostics
- Added `scripts/run_afterhours_transfer_question_role_gate_diagnostics.py` to answer the next mechanistic question after the question-role slice result:
  - if analyst-question semantics are now best understood as a sparse diagnostic signal, what are they actually doing relative to the current hard-abstention shell?
- Rather than fit another model, the script directly decomposes the latest held-out window into three decision states:
  1. `disagreement_auto_pre`
  2. `agreement_keep_agreed`
  3. `agreement_veto_to_pre`
- It joins the existing question-role slice diagnostics with the role-text transfer predictions and writes outputs under `results/afterhours_transfer_question_role_gate_diagnostics_lsa4_real/`.
- Main result:
  - question-role differs from hard abstention on only a small subset:
    - total events `60`
    - disagreement auto-pre `16`
    - agreement keep-agreed `29`
    - agreement veto-to-pre `15`
  - the useful standalone gain over `pre_call_market_only` comes almost entirely from the `agreement_keep_agreed` state:
    - mean gain vs `pre_call_market_only` `≈ 3.25e-09`
    - win share vs `pre_call_market_only` `≈ 34.5%`
    - gain vs hard abstention exactly `0`
  - the only place question-role makes a distinct decision is the `agreement_veto_to_pre` subset, and that decision is currently net negative:
    - mean gain vs hard abstention `≈ -8.20e-10`
    - win share vs hard abstention only `≈ 20%`
- The veto policy is not completely random:
  - vetoed agreement events have somewhat larger hard-abstention error on average (`≈ 2.40e-08` vs `≈ 1.75e-08` for kept agreement events)
  - so role-text is trying to abstain on somewhat riskier agreement cases
- But the veto logic is still too noisy:
  - strongest veto improvements include `AAPL_2023_Q4`, `NVDA_2023_Q1`, `MSFT_2025_Q1`
  - strongest veto failure is `NKE_2025_Q2`
  - the most heavily vetoed slice is the generic management-phrasing bucket, and its mean gain vs hard abstention is clearly negative
  - more interpretable topic pockets such as `data center / computing / nvidia` and `high performance / execution / compute` are less aggressively vetoed and are often simply kept identical to hard abstention
- Interpretation:
  - this checkpoint shows that question-role is not adding a genuinely better controller on top of the shell
  - its only distinctive behavior is a small agreement-veto policy, and that policy is still net harmful
  - this reinforces the cleaner research position:
    - analyst-question semantics are useful as a compact diagnostic lens on analyst-attention structure
    - but they should not be promoted into a new transfer rule or routing upgrade yet
- Added `docs/afterhours_transfer_question_role_gate_checkpoint_20260321.md` as the checkpoint note for this gate-diagnostic result.


### Transfer hard-abstention miss diagnostics
- Added `scripts/run_afterhours_transfer_hard_abstention_miss_diagnostics.py` to step back from the role-text family and profile the remaining failures of the current strongest transfer shell directly:
  - if hard abstention is still the best compact method, what do its held-out latest-window miss cases actually look like?
- Reused the latest-window event states from the question-role gate diagnostics and joined them with the existing clean `after_hours` panel, event text / QA interaction features, and QA benchmark features.
- Defined the hard miss subset as the top quartile of latest-window hard-abstention squared error and wrote the outputs under `results/afterhours_transfer_hard_abstention_miss_diagnostics_lsa4_real/`.
- Main result:
  - the remaining shell misses are sharply concentrated:
    - total events `60`
    - top miss events `15`
    - top-miss mean hard-abstention squared error `≈ 5.83e-08` vs `≈ 9.47e-10` for the rest
    - top-miss mean absolute error `≈ 1.90e-04` vs `≈ 2.64e-05` for the rest
  - the strongest standardized shifts are highly coherent:
    - denser `Q&A` (`qa_pair_count` effect `≈ +0.746`)
    - slightly weaker observability (`a4_strict_row_share` effect `≈ -0.602`, `a4_strict_high_conf_share` effect `≈ -0.574`)
    - lower directness / coverage (`qa_bench_direct_early_score_mean` effect `≈ -0.580`, `qa_bench_coverage_mean` effect `≈ -0.572`, `qa_bench_direct_answer_share` effect `≈ -0.558`)
    - more evasive / lower-overlap answering (`qa_bench_evasion_score_mean` effect `≈ +0.570`, `qa_pair_low_overlap_share` effect `≈ +0.539`, `qa_pair_answer_forward_rate_mean` effect `≈ +0.539`)
    - somewhat larger revenue surprise (`revenue_surprise_pct` effect `≈ +0.386`)
- Importantly, the hard-miss pocket is shell-wide rather than just role-text-specific:
  - `agreement_keep_agreed`: `5` top misses
  - `agreement_veto_to_pre`: `6` top misses
  - `disagreement_auto_pre`: `4` top misses
  - so the next signal search should not be framed as “repair the question-role veto”
- Interpretation:
  - this is a strong research-narrowing result
  - the current shell failures are most naturally described as dense, difficult `Q&A` under slightly weaker observability, with more evasive or low-overlap answers and surprise acting as a secondary modifier
  - that points the next compact signal search toward answerability / evasion / observability rather than more role-text routing
- Added `docs/afterhours_transfer_hard_abstention_miss_checkpoint_20260321.md` as the checkpoint note for this shell-miss diagnostic.


### Transfer answerability-factor benchmark
- Added `scripts/run_afterhours_transfer_answerability_factor_benchmark.py` to test the cleanest possible method move after the hard-abstention miss diagnostics:
  - if the remaining shell failures really look like dense, lower-directness, more evasive `Q&A`, can that profile be compressed into a single learnable factor that improves agreement-side abstention?
- Reused the temporal transfer setup and kept the shell simple:
  - train `val2020_test_post2020`
  - validate `val2021_test_post2021`
  - test `val2022_test_post2022`
  - disagreement rows still fallback to `pre_call_market_only`
  - only agreement rows are learnably refined
- Benchmarked three compact factor families:
  1. `answerability_core`
  2. `answerability_plus_observability`
  3. `answerability_observability_surprise`
  with `PCA1` / `PLS1` factor builders and `factor_only` / `geometry_plus_factor` route variants.
- Wrote the outputs under `results/afterhours_transfer_answerability_factor_benchmark_lsa4_real/`.
- Main result:
  - the best validation route is `answerability_core + PCA1 + geometry_plus_factor` with `alpha=10`, `val R^2 ≈ 0.9988408`
  - the held-out latest-window result is still just below the current shell:
    - `answerability_factor_route ≈ 0.99863849`
    - `hard_abstention ≈ 0.99864017`
    - `geometry_only ≈ 0.99863878`
    - `geometry_plus_hybrid ≈ 0.99863884`
  - paired tests confirm that this is competitive but not a new win:
    - vs hard abstention `p(MSE) ≈ 0.3865`
    - vs geometry only `p(MSE) ≈ 0.6335`
- The factor itself is scientifically useful and very interpretable:
  - strongest loadings are negative on `qa_bench_evasion_score_mean` / `qa_bench_high_evasion_share`
  - positive on `qa_bench_direct_early_score_mean`, `qa_bench_direct_answer_share`, and `qa_bench_coverage_mean`
  - negative on `qa_pair_low_overlap_share`
  - so the best latent is effectively a compact directness / coverage / low-evasion answerability axis
- But the route still does not isolate the latest high-risk agreement cases well enough:
  - it vetoes `15 / 44` agreement rows on the latest window
  - only `3 / 15` of those vetoes overlap the hard-abstention top-miss quartile
  - so the new factor is real, but its current use as a routing refinement remains under-targeted
- Interpretation:
  - the miss profile is genuinely learnable
  - but the current proxy pool seems to have been mostly exhausted by this one-factor answerability axis
  - adding extra observability or surprise inputs does not materially improve the result
  - hard abstention therefore remains the strongest compact transfer-side method in the repo
- Added `docs/afterhours_transfer_answerability_factor_checkpoint_20260321.md` as the checkpoint note for this compact answerability-factor benchmark.


### Transfer pair-tail factor benchmark
- Added `scripts/build_qa_pair_tail_features.py` to open a cleaner upstream signal source after the answerability-factor result:
  - if the shell-miss profile is not just an event-average phenomenon, can we summarize the **local worst-case `Q&A` pairs** directly?
- Built new pair-tail features from raw `A1` transcripts for the clean `after_hours` panel under `results/qa_pair_tail_features_real/`:
  - event coverage `172 / 172`
  - average pair count `≈ 7.41`
  - average max evasion `≈ 0.454`
  - average top-2 severity `≈ 2.743`
- The new features summarize local worst-case and dispersion structure such as:
  - tail evasion (`max`, `top2 mean`, high-evasion share)
  - bottom-tail directness / coverage
  - local severity combining question complexity, evasion, low coverage, and weak direct-early answering
  - first-two-pair summaries and compact nonresponse / mismatch shares
- Added `scripts/run_afterhours_transfer_pair_tail_factor_benchmark.py` to test whether these pair-tail signals improve agreement-side transfer refinement under the same temporal protocol:
  - train `val2020_test_post2020`
  - validate `val2021_test_post2021`
  - test `val2022_test_post2022`
  - disagreement rows still fallback to `pre_call_market_only`
- Benchmarked three compact families:
  1. `pair_tail_core`
  2. `pair_tail_dispersion`
  3. `pair_tail_with_observability`
  with `PCA1` / `PLS1` factors and `factor_only` / `geometry_plus_factor` routes.
- Wrote the outputs under `results/afterhours_transfer_pair_tail_factor_benchmark_lsa4_real/`.
- Main result:
  - the best validation route is `pair_tail_core + PCA1 + geometry_plus_factor` with `alpha=10`, `val R^2 ≈ 0.99884079`
  - held-out latest-window scores remain tightly clustered below the current shell ceiling:
    - `pair_tail_factor_route ≈ 0.99863855`
    - `hard_abstention ≈ 0.99864017`
    - `geometry_only ≈ 0.99863878`
    - previous `answerability_factor_route ≈ 0.99863849`
- Interpretation:
  - this is a useful refinement result, not a dead end
  - the learnable signal is now clearly local: worst-pair evasion, bottom-tail directness / coverage, and tail severity define a clean latent
  - but that local representation still does not identify the high-risk agreement failures well enough to beat hard abstention
  - only `3 / 15` vetoed agreement rows overlap the earlier hard-abstention top-miss quartile
  - so local pair tails matter, but the current handcrafted answerability proxy family still sits below the hard-abstention mainline
- Added `docs/afterhours_transfer_pair_tail_factor_checkpoint_20260321.md` as the checkpoint note for this pair-tail benchmark.


### Transfer pair-tail text benchmark
- Added `scripts/build_qa_pair_tail_text_views.py` to extend the pair-tail line from handcrafted answerability summaries into **local semantic text views** built directly from the top-severity `Q&A` pairs in raw `A1` transcripts.
- Built new hardest-pair text artifacts under `results/qa_pair_tail_text_views_real/`:
  - full event coverage `172 / 172`
  - average pair count `≈ 7.41`
  - average top-1 severity `≈ 3.082`
  - average top-2 severity mean `≈ 2.743`
- Added `scripts/run_afterhours_transfer_pair_tail_text_benchmark.py` to test whether the semantic content of the hardest local question / answer pair can improve agreement-side refinement under the same temporal protocol:
  - train `val2020_test_post2020`
  - validate `val2021_test_post2021`
  - test `val2022_test_post2022`
  - disagreement rows still fallback to `pre_call_market_only`
- Benchmarked four compact local text families:
  1. `tail_question_top1_lsa`
  2. `tail_answer_top1_lsa`
  3. `tail_qa_top1_lsa`
  4. `tail_qa_top2_lsa`
  with standalone and `geometry_plus_*` variants.
- Main result:
  - the best held-out latest-window route is **`tail_question_top1_lsa`** with test `R^2 ≈ 0.99865468`
  - this is above:
    - `hard_abstention ≈ 0.99864017`
    - `geometry_only ≈ 0.99863878`
    - `pair_tail_factor_route ≈ 0.99863855`
  - paired tests are small but directionally meaningful:
    - vs hard abstention `p(MSE) ≈ 0.04425`
    - vs geometry only `p(MSE) ≈ 0.0210`
    - vs pair-tail factor route `p(MSE) ≈ 0.0175`
- Mechanically, the gain is highly specific:
  - the route is identical to hard abstention on all `16` disagreement rows and on `24` agreement keep-agreed rows
  - the entire difference comes from a `20`-row agreement-veto subset
  - on that subset it wins `9 / 20` rows with net MSE gain `≈ 9.78e-09`
- Interpretation:
  - this is the first local-signal refinement in the current transfer line that actually edges past hard abstention on the held-out latest window
  - the gain is still small and should be treated as exploratory rather than promoted immediately to the mainline
  - but it is scientifically important because it suggests the next genuinely new upstream signal is **not** a broader answerability proxy family; it is the local semantic content of the hardest analyst question itself
- Added `docs/afterhours_transfer_pair_tail_text_checkpoint_20260322.md` as the checkpoint note for this result.


### Transfer pair-tail question-encoding benchmark
- Added `scripts/run_afterhours_transfer_pair_tail_question_encoding_benchmark.py` as a direct confirmation follow-up to the new pair-tail text result:
  - if the new held-out win really comes from the hardest analyst question, does it survive a compact encoding benchmark?
- Kept the same transfer protocol fixed:
  - train `val2020_test_post2020`
  - validate `val2021_test_post2021`
  - test `val2022_test_post2022`
  - disagreement rows still fallback to `pre_call_market_only`
- Benchmarked four compact encodings of `tail_top1_question_text`:
  1. `question_lsa4_bi` (current baseline)
  2. `question_stop_lsa2_bi`
  3. `question_stop_lsa4_bi`
  4. `question_stop_lsa8_bi`
  each with standalone and `geometry_plus_*` variants.
- Main result:
  - the original no-stopword bi-gram `LSA(4)` route remains the best:
    - `question_lsa4_bi ≈ 0.99865468`
    - `hard_abstention ≈ 0.99864017`
    - `geometry_only ≈ 0.99863878`
  - cleaner stopword-removed encodings are consistently weaker:
    - `question_stop_lsa8_bi ≈ 0.99859079`
    - `question_stop_lsa4_bi ≈ 0.99855614`
    - `question_stop_lsa2_bi ≈ 0.99851083`
- Interpretation:
  - this is a useful confirmation result rather than a null one
  - it shows the new pair-tail hardest-question win is not easily replaced by a nearby compact encoding variant
  - but it also sharpens the mechanism: the predictive content is probably not just topical semantics, because the more content-cleaned stopword-removed encodings are easier to read yet clearly worse
  - the signal seems to live more in **local analyst-question framing / interaction form** than in a simpler content-only embedding
  - geometry still does not help this line, so the right framing remains a narrow standalone semantic veto rather than a broader shell upgrade
- Added `docs/afterhours_transfer_pair_tail_question_encoding_checkpoint_20260322.md` as the checkpoint note for this encoding-confirmation result.


### Transfer pair-tail question slice confirmation
- Added `scripts/run_afterhours_transfer_pair_tail_question_slice_confirmation.py` to confirm where the new hardest-question local semantic route actually helps relative to hard abstention.
- Joined the held-out latest-window predictions with the clean `after_hours` panel, event-level feature table, and `Q&A` benchmark features for full `60 / 60` side-input coverage.
- Built a compact set of confirmation slices:
  1. `agreement_veto`
  2. `agreement_keep`
  3. `disagreement`
  4. `hard_miss_top_quartile`
  5. `high_qna_density`
  6. `low_directness_high_evasion`
  7. `weaker_observability`
  8. `dense_evasive_weak_obs`
- Main result:
  - the gain still comes entirely from the `agreement_veto` subset:
    - `n = 20`
    - net gain vs hard `≈ 9.78e-09`
    - win share `= 0.45`
    - `p(MSE) ≈ 0.047`
  - `agreement_keep` and `disagreement` remain exactly identical to hard abstention
- The broader slice picture is coherent rather than random:
  - `high_qna_density` carries about `80.6%` of the overall net gain
  - `low_directness_high_evasion` carries about `43.8%`
  - `weaker_observability` carries about `27.1%`
  - so the local-question route does line up with the same general difficult-`Q&A` direction identified by the earlier hard-abstention miss diagnostics
- But it is not just a generic hardest-case detector:
  - the `hard_miss_top_quartile` slice carries only about `35.1%` of the net gain
  - the most extreme combined slice `dense_evasive_weak_obs` is completely flat
- Interpretation:
  - this is a useful confirmation result
  - it shows the new hardest-question signal is structured and aligned with denser / lower-directness / more-evasive analyst-interaction pockets
  - but it remains a **narrow local veto signal**, not a broad detector of all hardest shell failures
  - this keeps the story disciplined: the next step should characterize the local framing pocket better, not add more routing complexity
- Added `docs/afterhours_transfer_pair_tail_question_slice_checkpoint_20260322.md` as the checkpoint note for this slice-confirmation result.


### Transfer pair-tail question-framing diagnostics
- Added `scripts/run_afterhours_transfer_pair_tail_question_framing_diagnostics.py` to characterize the local framing pocket behind the new hardest-question transfer signal without introducing a new method layer.
- Focused only on the current `agreement_veto` slice for the hardest-question route and split it into:
  - positive veto rows (`mse_gain_vs_hard > 0`)
  - negative veto rows (`mse_gain_vs_hard < 0`)
  - flat veto rows (`mse_gain_vs_hard = 0`)
- Main counts:
  - veto rows `20`
  - positive `9`
  - negative `4`
  - flat `7`
- Surface-form summary:
  - positive veto questions are shorter on average (`≈ 103` vs `≈ 134` tokens)
  - they have more hedging / framing language (`hedge_share ≈ 0.035` vs `≈ 0.016`)
  - slightly more first/second-person interaction cues
  - materially more numeric grounding (`numeric_token_share ≈ 0.019` vs `≈ 0.004`)
- Distinctive positive-side ngrams include:
  - `wondering`
  - `wondering if`
  - `if you could`
  - `i think`
  - `color`
  - `around`
  - `million`
- Distinctive negative-side ngrams include:
  - `disney`
  - `platform`
  - `sports`
  - `assets`
  - `content`
  but this side is partly topic / issuer contaminated, so it should be interpreted cautiously rather than as a clean general pattern
- Interpretation:
  - the new local-question signal is best described as a **clarificatory / model-building analyst-framing pocket** rather than a generic topic-semantic pocket
  - it seems to help most when analysts ask management to quantify, frame, or reconcile a difficult local operating picture
  - because the sample is still very small, this should remain a diagnostic insight rather than be promoted into a new rule or method claim
- Added `docs/afterhours_transfer_pair_tail_question_framing_checkpoint_20260322.md` as the checkpoint note for this framing-diagnostic result.


### Transfer pair-tail question-framing factor benchmark
- Added `scripts/run_afterhours_transfer_pair_tail_question_framing_factor_benchmark.py` to test the most natural compact-method follow-up to the new hardest-question line:
  - can the diagnosed local framing pocket be compressed into a small interpretable feature/factor route that reproduces the semantic lift?
- Built two compact framing families from `tail_top1_question_text`:
  1. `framing_core`
     - token count, numeric-token share, first/second-person shares, hedge share, follow-up marker count
  2. `framing_structure`
     - the above plus sentence-like count, question-mark count, and modal share
- Benchmarked each family as:
  - direct feature route
  - `PCA1` factor route
  - `geometry_plus_factor` route
- Main result:
  - no compact framing route reproduces the current hardest-question semantic win
  - held-out latest-window test scores are:
    - `geometry_only ≈ 0.99863878`
    - `geometry_plus_framing_core_factor_pca1 ≈ 0.99863878`
    - `geometry_plus_framing_structure_factor_pca1 ≈ 0.99863697`
    - `framing_structure_factor_pca1 ≈ 0.99862733`
    - `framing_structure_direct ≈ 0.99852090`
    - `framing_core_direct ≈ 0.99850684`
  - reference routes remain:
    - `hard_abstention ≈ 0.99864017`
    - `question_lsa4_bi ≈ 0.99865468`
- Interpretation:
  - this is a useful negative-result checkpoint
  - the local framing pocket is real and interpretable, but it is not adequately captured by a tiny handcrafted surface-form factor
  - direct framing features even look slightly attractive on validation but fail to transfer cleanly on held-out test, so turning the diagnostics directly into a new rule would be a mistake
  - the strongest hardest-question signal still contains information beyond these compact surface summaries
- Added `docs/afterhours_transfer_pair_tail_question_framing_factor_checkpoint_20260322.md` as the checkpoint note for this compact framing-factor benchmark.

### Transfer pair-tail question lexical-pattern benchmark
- Added `scripts/run_afterhours_transfer_pair_tail_question_lexical_pattern_benchmark.py` to test the next disciplined follow-up to the hardest-question line:
  - can a slightly richer but still compact lexical-pattern block recover part of the local hardest-question signal without reverting to the full semantic route?
- Built four compact lexical families from `tail_top1_question_text`:
  1. `clarify_modeling_lex`
  2. `quant_bridge_lex`
  3. `structural_probe_lex`
  4. `clarify_quant_lex`
- Benchmarked each family as:
  - direct lexical-count route
  - `PCA1` factor route
  - `geometry_plus_factor` route
- Main held-out latest-window result:
  - `question_lsa4_bi ≈ 0.99865468`
  - `clarify_modeling_lex_factor_pca1 ≈ 0.99864268`
  - `hard_abstention ≈ 0.99864017`
  - `geometry_only ≈ 0.99863878`
- Interpretation:
  - this is the first compact lexical-pattern route to edge above hard abstention while still staying meaningfully below the richer hardest-question semantic route
  - so there now appears to be a **small recoverable clarificatory / model-building core** inside the hardest-question signal
  - but the gain remains narrow and exploratory rather than a stable new method claim
- The best compact route is not a raw lexical-count route but a one-factor latent over the `clarify_modeling_lex` bundle.
  - strongest refit loadings are:
    - `wondering`
    - `wondering if`
    - `if you could`
    - `help us understand`
    - `help us think`
- Relative to hard abstention, the best compact route changes only `5 / 60` latest-window events, all on agreement rows.
  - `4` of those changes are positive
  - `1` is negative
  - `4 / 5` overlap with the richer `question_lsa4_bi` veto set
- So this benchmark further sharpens the story:
  - the hardest-question line is not reducible to tiny surface-form stats
  - but it does have a compact recoverable lexical core
  - and that core is specifically **clarificatory / model-building analyst framing**, not a broad topic or geometry-shell upgrade
- Added `docs/afterhours_transfer_pair_tail_question_lexical_pattern_checkpoint_20260322.md` as the checkpoint note for this new compact lexical-pattern result.

### Transfer pair-tail question lexical confirmation
- Added `scripts/run_afterhours_transfer_pair_tail_question_lexical_confirmation.py` to test the most important next question from the new lexical-pattern result:
  - is the compact `clarify_modeling_lex` factor actually capturing the most reliable core of the richer hardest-question semantic route, or is it just a smaller noisy alternative?
- The confirmation compares the compact lexical factor against:
  - hard abstention
  - the richer `question_lsa4_bi` route
  - overlap groups between their agreement-veto decisions
- Main overlap counts on the held-out latest window:
  - agreement rows `44`
  - disagreement rows `16`
  - lexical veto rows `7`
  - richer semantic veto rows `13`
  - shared veto rows `4`
  - semantic-only veto rows `9`
  - lexical-only veto rows `3`
- Main result:
  - the compact lexical factor is a **much more conservative** route than the richer semantic model
  - its clean value is concentrated in the `shared_veto` subset:
    - `n = 4`
    - lexical net gain vs hard `≈ 2.42e-09`
    - semantic net gain vs hard `≈ 2.42e-09`
    - both win shares `= 1.0`
- By contrast:
  - the richer semantic route keeps additional recall on `sem_only_veto` rows (`n = 9`, net gain `≈ 7.36e-09`)
  - but that extra subset is less pure and only wins on about `55.6%` of those rows
  - the lexical-only tail (`n = 3`) is weak and net negative (`≈ -7.31e-10`)
- Interpretation:
  - this is a useful confirmation result
  - the compact `clarify_modeling_lex` factor is **not** a replacement for the richer hardest-question semantic route
  - instead, it behaves like a **precision-oriented compact subset** of that broader mechanism
  - the compact factor recovers the cleanest shared-veto core, while the richer semantic route still carries additional upside beyond that compact core
- Added `docs/afterhours_transfer_pair_tail_question_lexical_confirmation_checkpoint_20260322.md` as the checkpoint note for this confirmation result.

### Transfer pair-tail question lexical margin benchmark
- Added `scripts/run_afterhours_transfer_pair_tail_question_lexical_margin_benchmark.py` to test the cleanest possible follow-up to the new compact lexical-confirmation result:
  - can a tiny learned confidence threshold keep the compact clarificatory core while filtering away its weak lexical-only tail?
- Fixed the route to the current best compact family:
  - `clarify_modeling_lex`
  - `PCA1` factor
- Only allowed **conservative** thresholds (`<= 0`) on the predicted gain signal, so the route could only become more selective, not more aggressive.
- Main result:
  - validation chooses `threshold = 0.0`
  - validation use-agreed share is `1.0`
  - validation veto share is `0.0`
- Held-out latest-window test therefore stays exactly the same as the base compact factor:
  - base factor `R^2 ≈ 0.99864268`
  - margin route `R^2 ≈ 0.99864268`
  - veto rows stay `7`
  - `p(MSE)` margin vs base factor `= 1.0`
- Interpretation:
  - this is a useful negative result
  - the compact clarificatory factor is real, but earlier windows do **not** provide learnable support for an extra confidence-margin filter
  - so the weak lexical-only tail is not something we can cleanly remove with a tiny threshold tweak
  - this keeps the story disciplined: the compact core is still a research-confirmed local mechanism, but not yet a stable standalone method block
- Added `docs/afterhours_transfer_pair_tail_question_lexical_margin_checkpoint_20260322.md` as the checkpoint note for this margin benchmark.

### Transfer pair-tail question shared-core diagnostics
- Added `scripts/run_afterhours_transfer_pair_tail_question_shared_core_diagnostics.py` to answer the next research question after the lexical-margin negative result:
  - what exactly is the compact clarificatory core,
  - how does it differ from the richer `question_lsa4_bi` veto tail,
  - and why does it seem to appear only in a narrow late-window pocket?
- The diagnostic rebuilds the latest held-out overlap groups:
  - `shared_veto`
  - `sem_only_veto`
  - `lex_only_veto`
  and then joins them with hardest-question text plus local pair-tail features.
- Main overlap counts:
  - `shared_veto = 4`
  - `sem_only_veto = 9`
  - `lex_only_veto = 3`
- Year breakdown:
  - `2023`: shared `2`, sem-only `6`, lex-only `1`
  - `2024`: shared `2`, sem-only `2`, lex-only `2`
  - `2025`: shared `0`, sem-only `1`, lex-only `0`
- So the compact clarificatory core is real, but it is **not** a generic late-window rule; it is a very small `2023–2024` subset inside the held-out latest split, while the richer semantic route still carries an extra `2025` tail.
- Group-level gain structure:
  - `shared_veto` is the cleanest subset:
    - lexical net MSE gain vs hard `≈ 2.42e-09`
    - semantic net MSE gain vs hard `≈ 2.42e-09`
    - both win shares vs hard `= 1.0`
  - `sem_only_veto` keeps more recall:
    - semantic net gain vs hard `≈ 7.36e-09`
    - semantic win share vs hard `≈ 0.556`
  - `lex_only_veto` remains weak and net negative:
    - lexical net gain vs hard `≈ -7.31e-10`
- Text / lexical diagnostics show the shared core is strongly aligned with:
  - `wondering`
  - `wondering if`
  - `if you could`
  - plus a bit more numeric grounding
- In contrast, the sem-only tail is broader, denser, and more mixed:
  - mean `qa_pair_count` is higher (`≈ 6.22` vs `≈ 4.5`)
  - it contains real upside, but also more topic contamination and broader strategic probes (for example Disney / ESPN / asset-separation style rows)
- Interpretation:
  - this is an important structural checkpoint
  - the compact `clarify_modeling_lex` factor now has a much clearer meaning:
    - it is the **shared clarificatory / model-building core** of the richer hardest-question semantic route
  - the richer `question_lsa4_bi` route still carries extra recall beyond that core
  - but that extra recall is less pure and less compressible
  - the compact core is therefore best treated as a precision-oriented explanatory subset, not yet a standalone stable method
- Added `docs/afterhours_transfer_pair_tail_question_shared_core_checkpoint_20260322.md` as the checkpoint note for this shared-core / temporal-emergence diagnostic.

### Transfer pair-tail question signature taxonomy
- Added `scripts/run_afterhours_transfer_pair_tail_question_signature_taxonomy.py` to answer the next natural follow-up after the shared-core diagnostic:
  - what is the broader taxonomy of the richer hardest-question semantic tail?
  - is the compact factor just “clarificatory language,” or something more specific?
- The taxonomy keeps the same narrow lexical-family palette:
  - `clarify`
  - `quant`
  - `structural`
  and labels each active veto row by its compact family signature.
- Main pattern:
  - **every single `shared_veto` row is `clarify+quant`**
    - `4 / 4`
  - by contrast, `sem_only_veto` is heterogeneous:
    - `clarify+structural = 3 / 9`
    - `clarify+quant = 2 / 9`
    - `quant = 1 / 9`
    - `clarify = 1 / 9`
    - `quant+structural = 1 / 9`
    - `structural = 1 / 9`
- This means the compact lexical factor is not just picking up generic clarificatory phrasing.
  - it is specifically recovering a **clarify + quantitative-bridge** pocket.
- The shared `clarify+quant` core is the cleanest subtype:
  - `n = 4`
  - lexical net MSE gain vs hard `≈ 2.42e-09`
  - semantic net MSE gain vs hard `≈ 2.42e-09`
  - both win shares vs hard `= 1.0`
- The sem-only tail breaks into two useful research branches:
  1. **operational / quantitative bridge tail**
     - `clarify+quant` (`n = 2`, semantic net gain `≈ 2.92e-09`, win share `= 1.0`)
     - `quant` (`n = 1`, semantic net gain `≈ 3.43e-09`, win share `= 1.0`)
  2. **structural / strategic probe tail**
     - `clarify+structural` (`n = 3`, semantic net gain `≈ 1.63e-09`, win share `≈ 0.333`)
     - `structural` / `quant+structural` singletons are negative
- Temporal implication:
  - the compact shared core appears only in `2023–2024`
  - the only remaining `2025` semantic-tail row is `clarify+quant`
  - so the compact factor is not randomly unstable; it is tuned to the clean `clarify+quant` core, while the last surviving tail in `2025` already lies beyond that tiny lexical signature
- Interpretation:
  - this is a strong research clarification
  - the compact factor is high-precision partly because it **avoids** the noisier structural / strategic branch
  - the richer semantic route carries both:
    - the clean compact `clarify+quant` core
    - and a broader, more mixed tail
  - this suggests the next useful direction is a slightly richer but still compact local representation that tries to extend recall into the operational / quantitative tail **without** opening the structural / strategic branch
- Added `docs/afterhours_transfer_pair_tail_question_signature_taxonomy_checkpoint_20260322.md` as the checkpoint note for this taxonomy result.

### Transfer pair-tail question non-structural encoding benchmark
- Added `scripts/run_afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark.py` to test the cleanest next hypothesis from the signature-taxonomy result:
  - if the useful local signal is really clarificatory / operational / quantitative rather than structural,
  - can we explicitly remove the structural-probe vocabulary and keep the strongest hardest-question route intact?
- The benchmark masks only the existing `structural_probe` phrase family from the hardest question text, then reruns compact question encodings:
  - `question_mask_struct_lsa4_bi`
  - `question_mask_struct_lsa8_bi`
  - `question_mask_struct_lsa4_uni`
  - plus geometry-coupled versions for confirmation
- Main result:
  - **`question_mask_struct_lsa4_bi ≈ 0.99865468`**
  - this is exactly the same held-out result as the current raw strongest route:
    - `question_lsa4_bi ≈ 0.99865468`
  - and still above:
    - `hard_abstention ≈ 0.99864017`
    - `geometry_only ≈ 0.99863878`
- Relative to the existing raw hardest-question route:
  - `p(MSE) = 1.0`
  - all pooled metric differences are exactly `0`
- Interpretation:
  - this is a very useful confirmation result even though it is not an improvement
  - the strongest local hardest-question signal survives **unchanged** after structural masking
  - so the best current local signal is already living in the **non-structural** part of the question
  - this strongly supports the taxonomy result that the retained upside is coming from clarificatory / operational / quantitative wording rather than structural / strategic probe language
- Larger masked variants are worse:
  - `question_mask_struct_lsa8_bi ≈ 0.99857873`
  - `question_mask_struct_lsa4_uni ≈ 0.99847473`
  so the current compact `LSA(4)` bi-gram scale still looks right.
- Geometry still does not help:
  - all geometry-plus masked variants remain below the standalone masked question route
- Added `docs/afterhours_transfer_pair_tail_question_nonstructural_encoding_checkpoint_20260322.md` as the checkpoint note for this non-structural encoding confirmation.

### Transfer pair-tail non-structural multiview benchmark
- Added `scripts/run_afterhours_transfer_pair_tail_nonstructural_multiview_benchmark.py` to test the next compact follow-up after the non-structural encoding confirmation:
  - if the useful local tail is already non-structural,
  - can we extend recall by adding one more compact local view from the hardest answer or top-1 `Q&A` pair text?
- All views stay tightly constrained:
  - structural masking remains on
  - `LSA(4)` bi-gram encodings remain on
  - no new shell or router is introduced
- Families compared:
  - `question_mask_struct_lsa4_bi`
  - `answer_mask_struct_lsa4_bi`
  - `qa_mask_struct_lsa4_bi`
  - `question_plus_answer_mask_struct_lsa4_bi`
  - `question_plus_qa_mask_struct_lsa4_bi`
- Main result:
  - the single hardest-question view still wins:
    - **`question_mask_struct_lsa4_bi ≈ 0.99865468`**
  - answer-only and pair-only are both weaker:
    - `answer_mask_struct_lsa4_bi ≈ 0.99855053`
    - `qa_mask_struct_lsa4_bi ≈ 0.99855719`
  - and the multiview combinations are also weaker:
    - `question_plus_qa_mask_struct_lsa4_bi ≈ 0.99852148`
    - `question_plus_answer_mask_struct_lsa4_bi ≈ 0.99850934`
- Interpretation:
  - this is a very useful narrowing result
  - the current retained local signal is not just non-structural; it is also strongly **question-centric**
  - bringing in the paired answer or top-1 `Q&A` text adds noise faster than it adds useful recall
  - so the next promising direction is **not** broader local multiview fusion
  - it is a better compact representation of the hardest question itself
- Added `docs/afterhours_transfer_pair_tail_nonstructural_multiview_checkpoint_20260323.md` as the checkpoint note for this non-structural multiview narrowing result.

### Transfer pair-tail question supervised encoding benchmark
- Added `scripts/run_afterhours_transfer_pair_tail_question_supervised_encoding_benchmark.py` to test the cleanest learnable follow-up after the non-structural multiview negative result:
  - if the local signal is really question-centric, can a compact supervised text subspace on the masked hardest question beat the current unsupervised `LSA(4)` route?
- The benchmark keeps the same object and same protocol:
  - structurally masked hardest-question text
  - TF-IDF bi-grams
  - compact PLS subspaces with `1`, `2`, and `4` components
  - same agreement-side gain-prediction shell
- Main result:
  - best supervised family is `question_mask_struct_pls4_bi`
  - but it only reaches `≈ 0.99858590`
  - this is below:
    - `question_mask_struct_lsa4_bi ≈ 0.99865468`
    - `question_lsa4_bi ≈ 0.99865468`
    - `hard_abstention ≈ 0.99864017`
- The weaker supervised variants are lower still:
  - `question_mask_struct_pls2_bi ≈ 0.99856058`
  - `question_mask_struct_pls1_bi ≈ 0.99855619`
- Interpretation:
  - this is a useful negative result
  - at the current matched-sample scale, compact supervised question subspaces look less transferable than the stable unsupervised masked-question encoding
  - the learned PLS factors also appear more idiosyncratic, with top phrases such as:
    - `medical and`
    - `about linearity`
    - `retail margin`
  - so this path seems to drift toward small train-window-specific phrase pockets rather than a robust new local question axis
- Added `docs/afterhours_transfer_pair_tail_question_supervised_encoding_checkpoint_20260323.md` as the checkpoint note for this supervised-encoding negative result.

### Project stage report (LaTeX)
- Added `docs/project_stage_report_20260323.tex` as a 3--5 page stage report in LaTeX.
- The report consolidates the current project storyline into one accessible document:
  - research background under noisy timing,
  - main hypotheses and problem framing,
  - current experimental design,
  - the strongest fixed-split `after_hours` results,
  - the transfer-side abstention story,
  - the newest hardest-question exploratory line,
  - and the current limitations.
- The tone is intentionally more natural and slightly explanatory than the checkpoint memos so that it can function as a handoff-ready or discussion-ready project snapshot.

### English LaTeX stage report
- Added `docs/project_stage_report_20260323_en.tex` as an English version of the stage report.
- The English version keeps the same factual backbone as the Chinese report:
  - noisy-timing problem framing,
  - current contribution hierarchy,
  - main fixed-split `after_hours` results,
  - transfer-side abstention story,
  - and the newest hardest-question exploratory line.
- The wording is intentionally more natural and discussion-friendly so it can be reused for advisor updates, external collaboration, or early paper-facing communication.

### Chinese Word stage report
- Added `docs/project_stage_report_20260323_cn.docx` as a Chinese Word version of the current stage report.
- To keep the document reproducible inside the repository, also added `scripts/build_project_stage_report_docx.py`, which writes the `.docx` directly from a compact structured content block rather than relying on an external Word installation.
- The Word version is intended for easier offline editing, advisor sharing, and quick annotation outside the LaTeX workflow.

### Stage-evaluation consolidation
- Added `docs/stage_evaluation_consolidation_20260323.md` as a phase-review memo.
- The memo does not introduce new experiments; instead it explicitly freezes the current storyline hierarchy into:
  - main fixed-split contribution,
  - secondary transfer-side abstention extension,
  - exploratory hardest-question local signal,
  - and a demoted set of paths that should no longer be treated as headline candidates.
- This is meant to support the next stage decision as a controlled evaluation rather than another broad search round.

### ACM-style comprehensive stage report
- Added `docs/acm_stage_report_20260323.tex` as a comprehensive 8--12 page English report in ACM article format, built directly from the current repository evidence rather than from a hand-written summary alone.
- Added `scripts/build_acm_stage_report_artifacts.py` to extract the current QC counts, split sizes, benchmark results, and key comparison metrics from the live result files and turn them into reproducible report assets.
- Generated `docs/acm_stage_report_20260323_assets/` with:
  - figure-ready PDFs for data coverage, fixed-split storyline, transfer storyline, and the hardest-question branch;
  - LaTeX tables for data coverage, mainline results, transfer routes, and demoted paths;
  - a compact `metrics_summary.json` snapshot used by the report builder.
- Vendored the minimal official ACM article class files under `third_party/acmart/` so the report can compile inside the repository without depending on a separate system-level template install.
- Compiled the report successfully to `docs/acm_stage_report_20260323.pdf`; the current build is 9 pages and stays within the requested 8--12 page range.
- The report consolidates the full current storyline:
  - data assets and integrity constraints,
  - target redesign around `shock_minus_pre`,
  - fixed-split `after_hours` mainline,
  - conditional and negative extension results,
  - transfer-side abstention as the cleanest method interpretation,
  - and the newest hardest-question exploratory signal branch.
- Added `docs/acm_stage_report_20260323_assets.zip` as a lightweight downloadable archive of the generated ACM-report figures/tables/summary assets so the asset package can be fetched more easily from GitHub.

## 2026-04-06

### Professor feedback consolidation
- Read and translated the professor's first-report feedback into a repository-level action memo.
- Added `docs/professor_feedback_action_memo_20260406.md` to separate:
  - what the feedback really means,
  - what later March work already fixed internally,
  - what still remains unresolved for a conference paper,
  - and what kind of new method is actually justified by the current evidence.
- The main conclusion is now explicit:
  - the project already has a credible empirical pilot,
  - but it still needs a paper-grade task definition, clearer benchmark protocol, and one real method contribution.

### Paper restructuring plan
- Added `docs/paper_restructure_outline_20260406.md` as a conference-paper rewrite plan.
- The outline freezes the required next-paper logic:
  - decide explicitly between the current off-hours shock formulation and a cleaner during-ECC target family;
  - formalize the target windows, sample period, split rule, prior, and feature groups;
  - keep one main benchmark ladder;
  - and demote transfer / hardest-question material out of the headline body unless it directly supports the main method story.

### New-method direction update
- Re-read the current result stack before proposing the next method direction:
  - `shock_minus_pre` remains materially stronger than the raw target;
  - broad pooled off-hours evidence is still mostly market-dominated;
  - the clean `after_hours` `A4 + compact Q&A` line remains the safest fixed-split ECC increment;
  - abstention remains the cleanest transfer-side interpretation;
  - and the current prior-gated prototype still does not beat the ridge residual core on the strongest bundles.
- Based on that evidence, the recommended next model direction is no longer:
  - more feature sprawl,
  - more router families,
  - or heavier sequence expansion.
- The recommended next model direction is now:
  - a compact prior-aware, observability-gated, `Q&A`-centered residual model,
  - with optional multi-horizon heads if the professor-preferred during-ECC target family survives comparison.

### POG-QA prototype implementation
- Added `scripts/run_pog_qa_residual.py` as the first runnable prototype of the new paper-facing method:
  - `POG-QA = Prior-Aware Observability-Gated Q&A Residual`
- The method is intentionally narrower than the older gate experiments:
  - it keeps the same-ticker prior and the structured residual core always active,
  - then learns whether a dialog correction should be trusted,
  - and whether that correction should lean more on `Q&A` semantics or on weak-label answer-quality structure.
- The current script now supports:
  - the existing `shock_minus_pre` target,
  - plus extended target variants such as `within_minus_pre`, `log_within_over_pre`, and `post_minus_within`,
  - so the same method shell can be reused for the professor-requested target-family comparison.
- The prototype writes not only prediction summaries but also gate diagnostics:
  - join/filter coverage,
  - route and trust gate statistics,
  - per-event gate outputs,
  - and expert-level top coefficients.
- Added `docs/pog_qa_method_note_20260406.md` to formalize:
  - why this architecture is the right next method given the current evidence,
  - how it differs from the earlier failed global residual gate,
  - and how it responds to the professor's request for a clearer and more innovative method contribution.

### POG-QA smoke validation
- Ran `python -m py_compile scripts/run_pog_qa_residual.py` successfully.
- Because the local shared raw panel is still absent, validated executability with a synthetic end-to-end smoke panel instead of a real rerun.
- Verified that `POG-QA` completes end-to-end on the synthetic panel for:
  - `shock_minus_pre`
  - `within_minus_pre`
- The smoke runs confirmed that the script now writes the expected artifact set:
  - `pog_qa_residual_summary.json`
  - `pog_qa_residual_predictions.csv`
  - `pog_qa_residual_gate_diagnostics.csv`
- This does **not** count as a real empirical result update; it only verifies that the new method pipeline, target extension path, and diagnostics output are runnable before the next real-data pass.

### I-POG-QA method upgrade
- Upgraded the earlier `POG-QA` prototype into the current paper-facing `I-POG-QA` line:
  - `I-POG-QA = Incremental Prior-Aware Observability-Gated Q&A Residual`
- The upgrade does not widen the feature pile; it makes the method more paper-like in two specific ways:
  - a direction-aligned monotone trust gate
  - an incrementality regularizer on the applied dialog effect
- The monotone trust gate now uses aligned reliability features so the trust branch cannot assign the wrong sign to core observability / answerability signals.
- The incrementality regularizer now penalizes covariance between the applied dialog effect and:
  - the standardized same-ticker prior reference
  - the standardized structured-base residual reference
- Added a light activation regularizer so the model defaults back toward `prior + base_residual` unless the `Q&A` correction is worth using.
- The gate-diagnostic output is now more explicit and paper-readable:
  - prior prediction
  - base residual component
  - mixed dialog correction
  - applied dialog effect
  - and per-feature trust-direction metadata

### I-POG-QA validation refresh
- Re-ran `python -m py_compile scripts/run_pog_qa_residual.py` after the `I-POG-QA` upgrade.
- Re-ran synthetic end-to-end smoke validation for:
  - `shock_minus_pre`
  - `within_minus_pre`
- The upgraded script completed end-to-end and now writes the new artifact set:
  - `i_pog_qa_residual_summary.json`
  - `i_pog_qa_residual_predictions.csv`
  - `i_pog_qa_residual_gate_diagnostics.csv`
- The smoke runs also confirm that the training summary now records:
  - trust-feature directions
  - incrementality / activation hyperparameters
  - and incrementality diagnostics on the applied dialog effect

### Recent related-work positioning refresh
- Added `docs/recent_related_work_positioning_20260406.md` to pin the method against the 2024-2026 frontier rather than against older multimodal ECC papers alone.
- The main positioning conclusion is now explicit:
  - recent work already occupies the broad `LLM + multimodal fusion` lane,
  - transcript identity leakage makes strong priors mandatory,
  - and the clean remaining novelty slot is a strong-prior, noisy-observability, selectively trusted `Q&A` increment model.
- This is the repository-level justification for making `I-POG-QA` the absolute main method line instead of reopening audio-first, whole-transcript, or router-heavy branches.

### I-POG-QA benchmark-suite implementation
- Added `scripts/run_i_pog_qa_benchmark_suite.py` as the formal benchmark and ablation runner for the current main method line.
- The suite is designed to produce the method-facing main paper table rather than to reopen broad model search.
- It keeps the prior-aware baseline ladder:
  - `prior_only`
  - `residual_base_structured`
  - `residual_base_plus_semantic`
  - `residual_base_plus_quality`
  - `residual_base_plus_semantic_plus_quality`
- It then compares the retained `I-POG-QA` main model against the key method ablations:
  - no incrementality regularization
  - no activation regularization
  - free-sign trust gate
  - always-on trust gate
  - semantic-only route
  - quality-only route
- The suite writes one compact artifact family for paper use:
  - `i_pog_qa_benchmark_suite_summary.json`
  - `i_pog_qa_benchmark_suite_predictions.csv`
  - `i_pog_qa_benchmark_suite_gate_diagnostics.csv`

### I-POG-QA benchmark-suite note
- Added `docs/i_pog_qa_benchmark_suite_note_20260406.md` to document:
  - what each retained variant tests,
  - which questions the suite is meant to answer,
  - and the exact real-data run order once the processed panel becomes available locally.

### I-POG-QA benchmark-suite smoke validation
- Ran `python -m py_compile scripts/run_i_pog_qa_benchmark_suite.py` successfully.
- Ran the full suite end-to-end on the synthetic smoke panel for `shock_minus_pre`.
- This confirms that the repository now has a runnable main-table path for:
  - full `I-POG-QA`
  - retained baselines
  - and the core ablation family
- This still does **not** count as a new real empirical result, because the local machine still lacks the real processed panel needed for a real rerun.

### Google Drive data-restore clarification
- Confirmed that the current blocker is not missing processing logic inside the repo.
- The real issue is that this local workspace does not currently contain:
  - the original raw Google Drive DJ30 package
  - or the rebuilt intermediate artifacts such as `results/panel_real/event_modeling_panel.csv`
- Added `docs/data_restore_from_drive_20260406.md` to reconstruct the original data path clearly:
  - raw Drive folder placement
  - manifest step
  - QC step
  - intraday target build
  - modeling panel build
  - event text/audio feature build
  - `Q&A` benchmark feature build
- Added `scripts/restore_drive_pipeline.ps1` as a local PowerShell helper to rebuild the missing `qc_real / targets_real / panel_real / features_real / qa_benchmark_features_real` artifacts once the Drive folders are restored under `data/raw/dj30_shared`.
- Verified the PowerShell helper with a parser-level syntax check.
