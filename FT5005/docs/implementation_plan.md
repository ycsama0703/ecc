# Implementation Plan

## Goal

Turn the current pilot into a publishable-quality off-hours ECC volatility-shock study.

The implementation should now be organised around one locked main setting:

- subset: `pre_market + after_hours`
- target: `shock_minus_pre`
- forecasting frame: same-ticker prior plus residual model

Everything else should be treated as:

- robustness
- ablation
- or next-stage extension

## Stage 1: Lock the Main Setting

This stage is already mostly complete, but it needs to be treated as the canonical setup in all later work.

Canonical definitions:

- subset:
  - keep `pre_market`
  - keep `after_hours`
  - exclude `market_hours` from the headline model

- target:
  - `shock_minus_pre = post_call_60m_rv - pre_60m_rv`

- main baseline:
  - `same_ticker_expanding_mean`

- main model family:
  - residual ridge

Deliverables:

- one canonical config file or documented command set
- one frozen main result table

## Stage 2: Finish the Integrity Package

The current project already has good baseline awareness, but the integrity layer still needs to be tightened.

Must-have checks:

1. year-wise stability
- report results by test year, not only pooled test

2. ticker concentration
- check whether a small number of firms dominate the gain
- report groupwise errors

3. stricter timing robustness
- small start-time jitter already exists
- rerun the fixed main setting on the off-hours shock target under timing perturbations

4. QC sensitivity
- compare strict versus broad `A4`
- compare with and without weaker transcript files

5. target-selection transparency
- document that raw target, ratio targets, and shock targets were all tested
- show that the final target was chosen because it best matches the event-specific forecasting objective

Deliverables:

- one robustness summary table
- one appendix-ready integrity section

## Stage 3: Finalise the Core Ablation Story

This stage is now central.

The strict off-hours ablation already shows:

- `structured_only` is currently the strongest holdout model
- `extra` hurts on top of that core
- `qna_lsa` helps relative to weaker overbuilt stacks
- `qa_benchmark` and `audio` are not robust test-side wins

That means the core ablation story should be:

1. `prior_only`
2. `structured_only`
3. `structured + extra`
4. `structured + extra + qna_lsa`
5. `structured + extra + qna_lsa + qa_benchmark`
6. `... + audio`

And every comparison should carry:

- test `R^2`
- RMSE
- MAE
- paired bootstrap intervals
- paired permutation p-values

Deliverables:

- main paper ablation table
- significance table
- one concise narrative of which additions help and which do not

## Stage 4: Improve the Method, Not the Feature Pile

The current main weakness is not feature count.

The current main weakness is that the strongest model is still simple.

The next method step should therefore be a **target-aware, prior-aware model improvement**, not blind multimodal complexity.

Best candidates:

1. prior-gated residual model
- let the model adapt how much to trust the prior versus event features

2. regime-aware residual model with shrinkage
- not fully separate models
- a shared global core with regime-specific adjustments

3. point-in-time `Q&A` residual model
- allow dense `Q&A` semantics to only correct the prediction when their signal is strong

Avoid for now:

- end-to-end LLM predictor
- full sequence transformer over noisy A4
- complex audio-first architectures

Deliverables:

- one new method that is genuinely about incremental-value design
- not just another wider feature union

## Stage 5: Scale Beyond DJ30

This is the biggest structural gap between the current version and a submission-ready paper.

Required next sample:

- larger restricted sample such as `SP500` or `SP1500`
- even if raw text/audio cannot leave AIDF, embeddings and derived features should

Scale-up principle:

- do not redesign the task again
- transfer the locked main setting:
  - off-hours only
  - shock target
  - prior-aware residual evaluation

Needed outputs:

- sample size table
- coverage table
- comparison of DJ30 pilot versus larger-sample replication

Without this stage, the paper remains a strong pilot rather than a solid submission.

## Stage 6: Multimodal Recovery Only If It Becomes Real

Right now, audio does not have a robust test-side contribution in the best fixed setting.

That means audio should be treated as:

- secondary
- conditional
- worth revisiting only after the main event-level paper is stable

If revisited, the right direction is:

- stronger acoustic features
- cleaner role-aware speaker segmentation
- explicit tests of whether delivery helps only in `Q&A`

But do not force audio into the headline if it does not earn its place.

## Blocking Challenges

These are the main reasons the current version is not yet submission-ready.

### 1. Sample-size challenge

- best current result is still on a relatively small off-hours subset
- current strong numbers are credible, but still vulnerable to external-validity criticism

### 2. Method-contribution challenge

- current strongest model is not methodologically novel enough for top venues
- need a sharper prior-aware forecasting contribution

### 3. Target-selection challenge

- strongest result comes after several rounds of target redesign
- needs transparent justification and robustness

### 4. Multimodal-claim challenge

- audio and heavier `Q&A` stacks do not yet justify a strong multimodal headline

### 5. Generalisation challenge

- no large-sample replication yet
- no hard evidence that the result extends beyond DJ30

## Publishability Checklist

### Must fix before a serious submission

- lock the main setting in writing and code
- add year-wise and ticker-wise robustness
- build one stronger prior-aware method beyond plain ridge residual
- replicate on a larger restricted sample
- produce a final ablation plus significance package
- justify target and subset selection explicitly

### Strongly recommended

- add benchmark-adjusted market target if data become available
- test a leave-one-firm-out or harder generalisation split
- strengthen the write-up around why off-hours is financially meaningful

### Nice to have

- stronger audio
- stronger role-aware sequence extension
- benchmark-transfer `Q&A` labels from larger external datasets

## Next 2-Stage Execution Plan

### Next stage: turn pilot into solid paper core

Immediate work:

1. freeze `off-hours + shock target` as canonical
2. finish robustness and significance tables
3. add year-wise and ticker-wise diagnostics
4. prototype one prior-aware method improvement
5. draft the main paper tables and figures

Definition of done:

- a complete DJ30 paper core with honest contribution and limitations

### Following stage: make it submission-capable

Next work:

1. scale to a larger restricted sample
2. rerun the canonical task on that sample
3. compare pilot and large-sample results
4. keep only the feature blocks that remain robust

Definition of done:

- a paper that no longer depends on DJ30-only evidence

## Immediate Tasks for the Team

Member 1:
- year-wise and ticker-wise robustness

Member 2:
- larger-sample feasibility and data-transfer path

Member 3:
- prior-aware method prototype

Member 4:
- main paper tables, significance, and figure preparation

## Decision Rule

If the larger-sample replication confirms that:

- off-hours shock remains easier than raw level,
- same-ticker prior remains a serious baseline,
- residual event features still add value,

then the final paper should be framed as:

**Off-hours earnings-call volatility-shock prediction with prior-aware event modeling under noisy timing.**

If that replication fails, then the honest fallback is:

- present the current project as a strong DJ30 pilot and methodological case study,
- not as a full top-tier submission.
