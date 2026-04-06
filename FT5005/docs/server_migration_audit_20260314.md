# Server Migration Audit 2026-03-14

## Purpose

This note records the current ACM ICAIF project state after the cross-server migration, identifies the canonical project paths on the current machine, and logs newly discovered research and implementation risks that need to be addressed before the next modeling push.

## Canonical Paths

- Canonical storage root: `/media/volume/dataset/xma8/work/icaif_ecc_news_attention`
- Compatibility symlink: `/ocean/projects/cis250100p/xma8/icaif_ecc_news_attention`
- Historical local note only: `/home/exouser/ACM_ICAIF.txt`

Important clarification:
- `/home/exouser/ACM_ICAIF.txt` is not the authoritative project workspace.
- That file is a mixed historical note containing course instructions, older news-track brainstorming, and sensitive SSH material. Treat it as context only, not as the current source of truth.

## Current Server Inventory

Top-level repository areas found on the current server:
- `README.md`
- `docs/`
- `scripts/`
- `results/`
- `data/`
- `.git/`

Key documentation files currently present:
- `docs/research_plan.md`
- `docs/progress_log.md`
- `docs/implementation_plan.md`
- `docs/data_inventory.md`
- `docs/data_qc_protocol.md`
- `docs/novelty_integrity_storyline.md`
- `docs/sota_workflow_gap_analysis.md`
- `docs/literature_map.md`
- `docs/proxy_feasibility.md`
- `docs/data_request_email.md`
- `docs/followup_email_draft.md`
- `docs/followup_questions.md`

Key pipeline scripts currently present:
- `scripts/build_event_manifest.py`
- `scripts/run_initial_qc.py`
- `scripts/build_intraday_targets.py`
- `scripts/build_modeling_panel.py`
- `scripts/build_event_text_audio_features.py`
- `scripts/build_real_audio_features.py`
- `scripts/build_qa_benchmark_features.py`
- `scripts/build_role_aware_sequence_features.py`
- `scripts/build_sentence_aligned_sequence_features.py`
- `scripts/build_weak_section_sequence_features.py`
- `scripts/run_structured_baselines.py`
- `scripts/run_text_tfidf_baselines.py`
- `scripts/run_dense_multimodal_ablation_baselines.py`
- `scripts/run_identity_classical_baselines.py`
- `scripts/run_prior_augmented_tabular_baselines.py`
- `scripts/run_prior_residual_ridge_baselines.py`
- `scripts/run_target_variant_experiments.py`
- `scripts/run_regime_residual_experiments.py`
- `scripts/run_regime_subset_residual_experiments.py`
- `scripts/run_sequence_bin_baselines.py`

Key result folders currently present:
- `results/qc_real_clean/`
- `results/targets_real/`
- `results/panel_real/`
- `results/features_real/`
- `results/audio_real/`
- `results/text_baselines_real/`
- `results/identity_classical_baselines_real/`
- `results/prior_residual_baselines_real/`
- `results/target_variant_experiments_real/`
- `results/regime_residual_experiments_real/`
- `results/regime_subset_experiments_real/`
- `results/sequence_real/`
- `results/sequence_baselines_real/`

Observed storage snapshot:
- `data`: about `16G`
- `results`: about `285M`
- `docs`: about `124K`
- `scripts`: about `704K`

## Current Authoritative Research State

The current repo state confirms that the active project is no longer the earlier news-amplification draft. The authoritative research path is the multimodal DJ30 earnings-call pipeline built around:
- `A1` structured transcript text
- `A2` scheduled transcript HTML
- `A3` call audio
- `A4` noisy timed sentence segments
- `C1` and `C2` analyst tables
- `D` 5-minute stock data

Current observed package and panel status:
- `A1 = 2146`
- `A2 = 842`
- `A3 = 796`
- `A4 = 670`
- `D = 25` ticker files
- joined modeling panel rows = `553`

Current missing tickers in `D`:
- `CRM`
- `CVX`
- `PG`
- `TRV`
- `V`

## New Findings After The Migration Audit

### 1. Regime labeling bug in `scheduled_hour_et`

The panel builder previously converted `scheduled_datetime` into an integer hour only. That meant events such as `09:30` were stored as `9`, then classified as `pre_market` because the regime scripts use the threshold `hour < 9.5`.

This has now been fixed in `scripts/build_modeling_panel.py` so `scheduled_hour_et` preserves fractional hours such as:
- `08:30 -> 8.5`
- `09:30 -> 9.5`
- `17:00 -> 17.0`

Measured impact on the current `553`-event panel:
- old regime counts: `after_hours=204`, `pre_market=293`, `market_hours=56`
- corrected regime counts: `after_hours=204`, `pre_market=273`, `market_hours=76`
- changed events: `20`
- all currently identified changed events are `GS` calls scheduled at `09:30`

Measured impact on the off-hours subset used by the strongest current result:
- old off-hours split: `train=248`, `val=82`, `test=167`
- corrected off-hours split: `train=239`, `val=78`, `test=160`

Rerun status after the fix:
- the affected regime-based result files were rerun after rebuilding the panel
- corrected off-hours subset result:
  - `prior_only` test `R^2`: about `0.191`
  - `residual_dense` test `R^2`: about `0.835`
  - `residual_dense_plus_qna_lsa` test `R^2`: about `0.897`

Updated interpretation:
- the off-hours result remains genuinely strong after the fix
- but it is slightly weaker than the pre-fix headline number of about `0.901`

### 2. Strongest results do not yet isolate ECC incremental value cleanly

The strongest `shock_minus_pre` and off-hours experiments still use `STRUCTURED_FEATURES` that include contemporaneous market-side variables such as:
- `pre_60m_rv`
- `pre_60m_vw_rv`
- `pre_60m_volume_sum`
- `within_call_rv`
- `within_call_vw_rv`
- `within_call_volume_sum`

These variables are scientifically useful if the claim is "forecast post-call shock using pre-call state plus within-call market behavior plus ECC information."

They are not sufficient if the claim is "text/audio alone add strong signal."

Required next comparison ladder:
- prior only
- market-state only
- ECC-only text/timing/audio
- market-state plus ECC

That ladder has now been run on the corrected panel via `scripts/run_signal_decomposition_benchmarks.py`.

Full-sample `shock_minus_pre` results beyond the same-ticker prior:
- `market_only` test `R^2`: about `0.904`
- `market_plus_controls` test `R^2`: about `0.909`
- `ecc_text_timing_only` test `R^2`: about `-0.022`
- `ecc_text_timing_plus_audio` test `R^2`: about `-0.176`
- `market_controls_plus_ecc_text_timing` test `R^2`: about `0.891`
- `market_controls_plus_ecc_plus_audio` test `R^2`: about `0.898`

Corrected off-hours `shock_minus_pre` results show the same pattern:
- `market_only` test `R^2`: about `0.907`
- `market_plus_controls` test `R^2`: about `0.909`
- `ecc_text_timing_only` test `R^2`: about `-0.076`
- `ecc_text_timing_plus_audio` test `R^2`: about `-0.682`
- `market_controls_plus_ecc_text_timing` test `R^2`: about `0.898`
- `market_controls_plus_ecc_plus_audio` test `R^2`: about `0.907`

This is now the clearest current integrity finding in the project:
- the headline `shock_minus_pre` performance is currently driven primarily by market-side information,
- ECC text/timing features do not yet beat the market-only residual benchmark out of sample,
- and current audio features do not add a clean incremental gain.

So the current high `R^2` results should not be framed as transcript-only, audio-driven, or even clearly ECC-incremental wins.

Strict-clean sensitivity after excluding `html_integrity_flag=fail` shows the same basic pattern:
- clean full-sample split: `train=244`, `val=74`, `test=156`
- clean off-hours split: `train=212`, `val=62`, `test=137`
- clean full-sample `market_plus_controls` test `R^2`: about `0.911`
- clean full-sample `ecc_text_timing_only` test `R^2`: about `-0.074`
- clean off-hours `market_plus_controls` test `R^2`: about `0.911`
- clean off-hours `ecc_text_timing_only` test `R^2`: about `-0.054`

So the current market-dominant conclusion is not just a side effect of the known low-quality HTML rows.

### 3. Audio is still tested with a weak call-level proxy, not alignment-aware features

`scripts/build_real_audio_features.py` currently extracts coarse chunk summaries from a few short samples across the whole mp3 file. That is useful as a first sanity check, but it is not a fair test of timestamp-aware audio modeling.

What is still missing:
- `A4`-aligned utterance-level audio segmentation
- role-aware audio aggregation
- within-call audio features linked to the same timeline used for the market target

So the current negative audio result should be read as:
- "the coarse sampled audio proxy is weak"

not as:
- "audio has no value"

Important additional clarification after the decomposition benchmark:
- the strongest current `shock_minus_pre` results in the repo are not just "weakly helped" by audio
- under the current feature design, adding audio does not improve on the best market-only residual benchmark

### 4. QC filtering is still permissive

The current panel summary still contains `79` rows with `html_integrity_flag = fail`.

Current behavior:
- these rows remain inside the modeling sample
- the integrity flag is used as a control feature, not as an exclusion rule

This is acceptable for exploratory work, but it means we still need:
- a strict-clean sample rerun
- a fail-excluded sensitivity table

The strict-clean decomposition sensitivity is now complete and it does not overturn the current main conclusion.

What still remains useful:
- extending the same clean-sample check to any future headline ECC-specific model
- reporting the clean-sample comparison explicitly in the paper

### 5. Raw intraday targets still lack market adjustment

The repo still assumes no benchmark intraday ETF data is available. That keeps the project feasible, but it weakens the finance interpretation of the target because the current realized-volatility and shock outcomes are not yet benchmark-adjusted.

Minimum upgrade still needed:
- `SPY`, `DIA`, or sector ETF intraday controls

### 6. Sample size and coverage remain pilot-scale

The current usable sample is still:
- `25` tickers
- `553` linked events

This is a strong pilot and a good course-project base, but it is still small for stronger generalization claims. The paper will need either:
- careful pilot framing, or
- a later restricted-sample scale-up

### 7. Temporal validation is still too thin

Most saved results still rely on one main split:
- train `<= 2021`
- validation `= 2022`
- test `>= 2023`

That is useful, but not enough for a final paper-quality claim. We still need:
- rolling-origin validation
- multiple temporal folds
- uncertainty intervals or bootstrap confidence bands

### 8. Reproducibility gap after migration

The current server scan did not find a local repo-managed environment such as:
- `.venv`
- `venv`
- `pyproject.toml`
- `requirements.txt`
- `environment.yml`

The default `python3` on this server can run the lightweight panel rebuild, but it does not currently provide the modeling stack required by the baseline scripts.

Current workaround now in place:
- reconstructed local modeling environment at `/home/exouser/.venvs/icaif_ecc_news_attention`
- verified key packages:
  - `numpy 2.4.3`
  - `scipy 1.17.1`
  - `scikit-learn 1.8.0`
- used that environment to rerun the key identity, target-variant, and regime experiments from the corrected panel
- added `requirements-modeling-min.txt` as a repo-managed minimal package file for the current core rerun stack

So there is now a second migration task in addition to the research tasks:
- extend the new minimal package file into a fuller end-to-end environment spec before trusting long-run reproducibility for the whole pipeline

### 9. `Q&A` v2 features recover partial ECC signal, but not a robust incremental win

The refreshed `Q&A` benchmark layer in `scripts/build_qa_benchmark_features.py` now includes:
- stronger directness and early-answer features
- richer certainty, justification, temporal-framing, and attribution markers
- explicit restatement, drift, numeric-mismatch, and short-evasion proxies

These refreshed features were written to:
- `results/qa_benchmark_features_v2_real/qa_benchmark_features.csv`

The most important reruns now saved on the current server are:
- `results/signal_decomposition_qav2_real/`
- `results/target_variant_experiments_qav2_real/`
- `results/regime_subset_experiments_qav2_real/`
- `results/regime_residual_experiments_qav2_real/`

Updated evidence from those reruns:
- full-sample `ecc_text_timing_only` improves from about `-0.022` under `qav1` to about `0.095` under `qav2`
- off-hours `ecc_text_timing_only` improves from about `-0.076` to about `0.139` on the all-HTML slice
- but clean-sample `ecc_text_timing_only` remains weak at about `-0.005` on the full sample and about `-0.085` on off-hours only
- the strongest market baselines still dominate:
  - full-sample `market_only` remains about `0.904`
  - full-sample `market_plus_controls` remains about `0.909`
  - off-hours `market_plus_controls` remains about `0.909`
- adding `qav2` ECC features back on top of the market bundles still does not beat the best market-only benchmark

Updated interpretation:
- the server migration did not just uncover a timing bug and an environment gap
- it also clarified that our stronger `Q&A` semantics are meaningful but still sample-sensitive
- current evidence supports the more careful claim:
  - "heuristic `Q&A` upgrades recover some ECC-only signal,"
- but not yet:
  - "ECC features robustly add value beyond strong market controls"

This is now formalized in:
- `docs/qna_signal_checkpoint_20260314.md`
- `results/research_checkpoints_real/qna_signal_checkpoint_20260314.json`

## Immediate Next Actions

Recommended priority order:

1. Extend the new minimal package file into a fuller end-to-end environment spec.
2. Replace heuristic event-level `qav2` features with transferred pair-level `Q&A` quality or evasion scores.
3. Keep the benchmark ladder fixed for every new experiment: `prior_only`, `market_only`, `market_plus_controls`, `ECC_only`, `market_plus_ECC`.
4. Require the `exclude html_integrity_flag=fail` rerun for any future headline ECC-specific model.
5. Add market-adjusted or benchmark-controlled intraday targets if ETF bars can be obtained.
6. Upgrade evaluation from one split to rolling temporal validation with uncertainty intervals.

## Current Status Summary

The migration did not lose the project repository or the data. The main repo, scripts, outputs, and shared package are present and internally consistent on the current server.

The most important newly discovered issue was not missing files. It was that a key timing-derived regime field had been truncated to integer hours, which distorted the reported regime and off-hours experiments. That bug is now fixed in code, the panel has been rebuilt, and the key regime outputs have been rerun.

The most important updated scientific takeaway is now:
- the off-hours `shock_minus_pre` result is still very strong after correction,
- but the corrected rerun no longer supports the old simplification that a global regime-agnostic model is clearly safer than regime-specific fitting.
- and the newer `qav2` reruns now further clarify that richer heuristic `Q&A` features recover some ECC-only signal without yet delivering a robust incremental gain beyond the strongest market-side benchmarks.
