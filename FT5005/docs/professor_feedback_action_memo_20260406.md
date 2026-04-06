# Professor Feedback Action Memo 2026-04-06

## Purpose

This memo translates the professor's feedback on the first report into:

- a repo-level diagnosis;
- a paper-facing decision framework;
- and a concrete next-stage action plan.

The goal is not to restate the email. The goal is to identify what the feedback means for the current project state after the later March experiment rounds.

## One-paragraph verdict

The project is no longer a weak exploratory class project. It already has a real empirical core, a clear baseline ladder, and multiple negative results that improve integrity. But the professor's criticism is still correct: the repository currently behaves more like a strong research notebook than a conference paper. The main gaps are now paper definition, target clarity, benchmark clarity, and method contribution. In other words, the project does not mainly fail because of missing effort. It mainly fails because the paper-safe question, the formal experimental protocol, and the new method are not yet locked tightly enough.

## What the feedback is really saying

### 1. Writing quality is not the real bottleneck by itself

The professor starts with tone and clarity, but the deeper issue is structural. The current draft still reads too much like:

- a chronological research report;
- a repository summary;
- or a stage handoff note.

It does not yet read like a conference paper with:

- one precise task definition;
- one precise target family;
- one benchmark protocol;
- and one main method contribution.

### 2. The target variable is still under-defined at the paper level

This is the most important scientific point in the feedback.

The codebase already computes:

- `pre_60m_rv`;
- `within_call_rv`;
- `post_call_60m_rv`;
- and derived targets such as `shock_minus_pre`.

So the issue is not that the target was never defined in code. The issue is that it was not defined in finance-paper language clearly enough for an external reader.

The professor's two cases imply a real decision gate:

1. Case 1: off-hours pre/post formulation
2. Case 2: during-ECC and post-ECC formulation

He is explicitly signaling that Case 2 may be academically cleaner because it is more tightly tied to the call itself and may include less extra information leakage than a purely off-hours post-call design.

### 3. Sample period, holdout logic, and time split must become first-class paper objects

The repository now has temporal split logic and fixed split sizes, but these are still described too informally in the paper-facing material.

For a conference paper, the following must be explicit and tabulated:

- exact sample start date and end date;
- event counts after each filtering step;
- train / validation / test split rule;
- what counts as holdout;
- what counts as a harder transfer split.

### 4. The prior is conceptually central but still under-explained in prose

The repository already uses a clear `same-ticker expanding mean` prior-style construction. But the paper still treats the prior more like a background baseline than a formal object. The professor is asking for:

- a definition;
- a formula;
- and an explanation of why it is the right integrity hurdle.

### 5. Feature construction and forecasting method are still too implicit

The codebase has a real feature taxonomy:

- market features;
- controls;
- ECC structured features from `A1/A2/A4`;
- compact `Q&A` semantic features;
- weak-label `Q&A` benchmark features;
- audio and aligned-audio branches.

But the draft still does not present them as a clean methods section with:

- grouped inputs;
- inclusion logic;
- and benchmark-by-benchmark comparison.

### 6. Benchmarking is not missing in the repository, but it is missing in the paper logic

The repo already contains:

- identity-aware baselines;
- market and control baselines;
- residual ridge baselines;
- target-variant comparisons;
- ablation ladders;
- significance tests;
- and several method extensions.

So the true issue is not "no benchmarks exist." The issue is:

- they are not yet turned into one stable conference-paper benchmark ladder;
- they are not yet mapped explicitly onto the literature the professor cited;
- and they are not yet clearly separated into baselines, ablations, extensions, and exploratory side branches.

### 7. The professor is also telling us the paper needs one real method contribution

Current strongest models are still methodologically modest:

- prior-aware residual ridge;
- careful feature selection;
- observability-aware transfer abstention.

Those are good science and good integrity, but they are not yet a strong conference-level method story on their own. The professor is effectively saying:

- keep the empirical core;
- but add one paper-worthy model.

## What later repo work has already fixed

### A. The project now has a much cleaner mainline

The March consolidation work already narrowed the project toward:

- `shock_minus_pre`;
- strong priors;
- clean `after_hours`;
- `A4 + compact Q&A semantics`;
- and a narrower incremental ECC claim.

### B. The repository now has a credible benchmark and integrity stack

The repo already includes:

- temporal splits;
- same-ticker priors;
- feature-group ladders;
- ablation chains;
- paired bootstrap and permutation tests;
- robustness summaries;
- and several negative findings that have been explicitly documented.

### C. The project already knows what should be demoted

The later repo state already demotes:

- heavy sequence variants;
- generic multimodal stacking;
- audio as a headline claim;
- and router complexity for its own sake.

## What still remains unresolved even after the later work

### U1. The paper still has not fully resolved Case 1 versus Case 2

The code supports both a broader event-window framing and a within-call window. But the project has not yet made a final paper decision on whether the main target should be:

- off-hours post-call shock;
- within-call shock;
- or a two-horizon formulation that jointly studies both.

### U2. The exact paper protocol is still under-specified

We still need one canonical, paper-facing statement of:

- sample period;
- filtering pipeline;
- split logic;
- holdout definition;
- and transfer definition.

### U3. The method contribution is still insufficient

The prior-gated prototype already taught us something useful:

- naive global gating is not enough;
- on the strong bundles it does not beat the residual ridge core;
- and the learned gates are close to constant.

So the next method cannot be:

- "just add one more generic gate."

It must be more tightly aligned to:

- `Q&A` structure;
- noisy observability;
- and strong prior correction.

### U4. The current paper story still mixes three different layers

These must be kept separate:

1. stable fixed-split mainline;
2. secondary transfer-side abstention extension;
3. exploratory hardest-question signal discovery.

## What the current experiments say about the next method

### 1. Target redesign is real and should stay central

The target-variant experiments show a sharp contrast:

- raw `post_call_60m_rv` remains prior-dominated and weak;
- ratio targets are better;
- `shock_minus_pre` is dramatically stronger.

Representative repository numbers:

- `raw_post_call_60m_rv`: best test `R^2` remains below the prior baseline;
- `log_post_over_pre`: competitive but not dominant;
- `shock_minus_pre`: best test `R^2` rises to about `0.884`.

### 2. Broad off-hours evidence is useful, but the clean ECC increment is narrower

On the broader corrected off-hours signal-decomposition benchmark:

- `prior_only` is about `0.198`;
- `market_only` is about `0.905`;
- `market_plus_controls` is about `0.911`;
- `ecc_text_timing_only` is negative;
- `market_controls_plus_ecc_text_timing` is not better than the best market-side result.

This means broad pooled evidence is still mostly market-dominated.

### 3. The clean `after_hours` ladder remains the best fixed-split paper-safe evidence

Representative fixed-split numbers:

- `pre_call_market_only about 0.9174`
- `pre_call_market_plus_controls about 0.9194`
- `pre_call_market_plus_a4 about 0.9014`
- `pre_call_market_plus_a4_plus_qna_lsa about 0.9271`
- `pre_call_market_plus_controls_plus_a4_plus_qna_lsa about 0.9347`

This shows:

- `A4` alone is not enough;
- compact `Q&A` semantics are what make the `A4` branch credible;
- the real value is therefore not generic multimodality, but structured and reliability-aware ECC correction.

### 4. Transfer results say restraint matters more than complexity

The strongest pooled temporal transfer route is still the abstention-based fallback:

- `consensus_fallback_pre_only about 0.997919`

This should remain a secondary extension, not the new main method.

### 5. The prior-gated prototype gives a useful negative design lesson

Current summary:

- on `structured`, gated is materially worse than ridge;
- on `structured_plus_extra`, gated barely edges ridge and not on the strongest bundle;
- on `structured_plus_extra_plus_qna_lsa`, gated falls back below ridge;
- gate mean stays around `0.15`;
- gate variance is tiny on the richer bundles.

This means the current gate learned almost no event-specific adaptivity.

### 6. The unseen-ticker result says the signal is real but heterogeneous

The harder unseen-ticker stress result is still strong overall:

- `prior_only` is negative;
- `residual_structured_only` remains strongly positive;
- but median ticker-level performance is much lower than pooled overall performance.

The next model should improve robustness, not just pooled score.

## Recommended decision on the target family

We should no longer treat the current off-hours target as automatically final.

### Main target decision protocol

1. Define two paper-level target families formally.
2. Run one unified comparison table under the same split and benchmark ladder.
3. Promote only one family to the headline.

### Default recommendation

Given the professor's comments and the current repo state, the default recommendation is:

- keep the current off-hours `shock_minus_pre` line as the empirical anchor for now;
- but seriously evaluate a `during` or `during+post` target family as the cleaner paper-facing alternative;
- if the during-call line is competitive and financially cleaner, promote it;
- if not, keep off-hours as the headline and present during-call as a robustness or secondary horizon.

## Recommended new method direction

The most credible next method is a small, structured, prior-aware model that directly reflects the current evidence.

### Prior-Aware Observability-Gated Q&A Residual Network

Working shorthand:

- `POG-QA`

Core design:

1. Prior branch
- keep the current strong prior and pre-call market backbone explicit;
- do not bury it inside a black-box predictor.

2. ECC residual branch
- encode `Q&A` at the round or pair level rather than only as one pooled event document;
- keep the representation compact and regularized;
- use event-level `A4` observability and reliability statistics as structured side information.

3. Observability gate
- let the model decide how much the ECC residual should correct the prior;
- but gate this correction using a small set of interpretable signals:
  - `A4` coverage / reliability;
  - `Q&A` density;
  - directness / answerability proxies;
  - regime and timing context.

4. Optional low-rank audio side branch
- include aligned or role-aware audio only as a compact side channel;
- do not make audio the architectural center.

5. Multi-horizon option
- if Case 2 survives comparison, use a shared representation with two heads:
  - during-call shock;
  - post-call shock.

Why this is the right direction:

- it respects the strong prior;
- it uses `Q&A` structure, matching the cited IJCAI line;
- it treats `A4` as noisy observability, matching the actual data;
- it avoids sequence overreach on a small sample;
- it gives a real method contribution without pretending that a giant model is justified.

## What not to make the next method

Avoid:

- full end-to-end transformer over all sentence segments;
- audio-first architecture;
- another generic expert-routing shell;
- another nearly constant global gate;
- or a paper whose main novelty is just "we added one more modality."

## Required benchmark ladder for the next method round

Any new method paper round should compare against at least:

1. `prior_only`
2. `market_only`
3. `market_plus_controls`
4. current best residual ridge mainline
5. compact `A4 + Q&A` residual ridge
6. one literature-inspired `Q&A` attention baseline
7. one literature-inspired compact audio-text fusion baseline if audio is retained
8. proposed `POG-QA`

## Immediate action plan

### Phase 1: paper definition cleanup

1. Write a formal task-definition subsection.
2. Add an event filtering and split table.
3. Add a precise prior definition and formula.
4. Add a benchmark-ladder table with method families grouped clearly.

### Phase 2: target decision

1. Compare off-hours and during-call target families under the same benchmark stack.
2. Freeze the headline target.
3. Demote the non-headline target to robustness or secondary analysis.

### Phase 3: new method implementation

1. Build a compact pair-level `Q&A` representation branch.
2. Build an observability gate using `A4` and answerability features.
3. Keep the prior explicit and residualized.
4. Add multi-horizon heads only if the target decision supports them.

### Phase 4: paper-facing comparison

1. Compare with literature-inspired baselines.
2. Report split sizes, holdout rule, and significance.
3. Keep transfer abstention as a secondary extension.
4. Keep hardest-question as exploratory only.

## Final recommendation

The professor's feedback does not require a project reset. It requires a project compression:

- fewer headline claims;
- cleaner definitions;
- one better method;
- and a tighter mapping from repo evidence to paper contribution.
