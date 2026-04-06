# Solid Storyline and Contribution Memo 2026-03-16

## 1. Why This Memo Exists

The project now has enough results that the main risk is no longer "we do not have anything interesting."

The main risk is:

- mixing together results from different pipeline stages
- overclaiming multimodality or method novelty
- and failing to articulate the real research contribution in a clean, limitation-driven way

This memo is meant to lock the paper story to the strongest corrected-panel evidence we currently have.

## 2. The Real Problem We Are Solving

The project should not be framed as:

- "build the biggest multimodal model for all earnings calls"

The real problem is narrower and more defensible:

- how to measure incremental market reaction to an earnings call under noisy scheduled-time anchoring,
- while separating true event-specific ECC signal from:
  - firm identity,
  - pre-call market state,
  - and feature-sprawl overfitting.

That is a stronger scientific problem because it is driven by limitations in the data and in naive evaluation design.

## 3. The Key Limitations Driving The Current Design

### L1. Raw post-call volatility is confounded by firm-level scale

Evidence:
- same-ticker historical priors are very strong on the raw target
- raw `post_call_60m_rv` makes it too easy to confuse scale with event information

Design response:
- move to `shock_minus_pre = post_call_60m_rv - pre_60m_rv`

### L2. Regular-hours calls have a noisier information environment

Evidence:
- regime results are clearly stronger outside regular trading hours
- `market_hours` remains smaller and less stable

Design response:
- lock the main pilot to `pre_market + after_hours`

### L3. Scheduled time is noisy

Evidence:
- `A2` time is scheduled time, not actual start time
- migration audit found a regime-labeling bug when `scheduled_hour_et` was truncated to integers

Design response:
- fix the timing field
- rerun corrected-panel results
- build the paper around noisy timing as an explicit constraint rather than hiding it

### L4. ECC signal can be overstated if market-side controls are not separated

Evidence:
- signal decomposition shows `market_only` and `market_plus_controls` already reach around `0.904-0.911` test `R^2`
- ECC-only models do not dominate those baselines

Design response:
- enforce a benchmark ladder:
  - `prior_only`
  - `market_only`
  - `market_plus_controls`
  - `ECC_only`
  - `market_plus_ECC`

### L5. Fancy architecture can look like progress without adding real signal

Evidence:
- prior-gated residual, stacked experts, nonlinear tree experts, and gated HGBR do not beat the strongest simple ridge baseline

Design response:
- treat negative architecture results as part of the contribution
- position the work around research design and integrity, not just model sophistication

## 4. The Current Strongest Corrected-Panel Result

The most important corrected-panel result is now:

- sample: `pre_market + after_hours`
- target: `shock_minus_pre`
- evaluation: same-ticker prior plus residual prediction
- main model: `residual_structured_only`

### Corrected all-HTML split

- split: `train=239`, `val=78`, `test=160`
- `prior_only` test `R^2 ≈ 0.191`
- `residual_structured_only` test `R^2 ≈ 0.912`

Comparison models:
- `structured + extra ≈ 0.871`
- `structured + extra + qna_lsa ≈ 0.901`
- `structured + extra + qna_lsa + qa_benchmark ≈ 0.891`
- `... + audio ≈ 0.901`

Interpretation:
- the simplest finance-aware structured residual model is still the strongest corrected holdout result

### Corrected clean split

- split: `train=212`, `val=62`, `test=137`
- `prior_only` test `R^2 ≈ 0.198`
- `residual_structured_only` test `R^2 ≈ 0.913`

Comparison models:
- `structured + extra ≈ 0.862`
- `structured + extra + qna_lsa ≈ 0.907`
- `structured + extra + qna_lsa + qa_benchmark ≈ 0.903`
- `... + audio ≈ 0.867`

Interpretation:
- the corrected main result is not being manufactured by the known `html_integrity_flag=fail` rows

## 5. What The Corrected Ablation Actually Says

The corrected ablation gives a surprisingly clean scientific message:

1. `structured_only` strongly beats `prior_only`
2. adding the current `extra` block hurts
3. adding dense `Q&A LSA` recovers a significant part of that loss
4. but the best `Q&A`-augmented models still do not clearly beat `structured_only`
5. `qa_benchmark` and `audio` remain secondary, not headline, evidence

This means the project should not currently claim:

- "more modalities help"

It should claim something closer to:

- once the target and sample are defined correctly, simple finance-aware event structure carries most of the stable signal, and richer `Q&A` semantics provide partial but not dominant incremental evidence

## 6. Why The Result Is Now Much More Solid

### R1. It survives corrected-panel rerun

The strongest mainline result is no longer just a pre-fix artifact.

### R2. It survives clean-sample filtering

`exclude html_integrity_flag=fail` does not overturn the main result.

### R3. It survives multiple years

Corrected all-HTML `residual_structured_only` remains strong by year:
- `2023 ≈ 0.798`
- `2024 ≈ 0.930`
- `2025 ≈ 0.935`

Corrected clean:
- `2023 ≈ 0.807`
- `2024 ≈ 0.923`
- `2025 ≈ 0.957`

### R4. It survives both off-hours regimes

Corrected all-HTML:
- `after_hours ≈ 0.917`
- `pre_market ≈ 0.714`

Corrected clean:
- `after_hours ≈ 0.915`
- `pre_market ≈ 0.726`

### R5. It survives harder generalisation better than expected

Corrected unseen-ticker stress:
- `prior_only ≈ -0.056`
- `residual_structured_only ≈ 0.991`
- median ticker-level `R^2 ≈ 0.580`

That does not mean cross-firm generalisation is "solved."
It does mean the signal is not reducible to same-company memorisation.

### R6. It survives leave-one-ticker-out influence checks

Concentration is real:
- top-name SSE gain share remains around `0.78-0.79`

But minimum leave-one-ticker-out `R^2` gain versus `prior_only` is still positive:
- corrected all-HTML: about `0.446`
- corrected clean: about `0.464`

This is exactly the kind of honest-but-strong robustness story we want.

## 7. What The Negative Results Contribute

Negative results are not side notes here.
They sharpen the paper's research contribution.

### N1. Market decomposition

The decomposition result tells us:
- current high `R^2` is still mainly market-driven
- so any ECC claim must be incremental and carefully bounded

### N2. Q&A v2 is useful but not a breakthrough

`qav2` recovers positive ECC-only signal on all-HTML slices, but does not create a robust headline gain over the best market baselines.

### N3. Hybrid architecture is not the missing ingredient

Regime-gated blends, positive stacks, and nonlinear gated stacks do not beat the strongest simple market ridge baseline.

### N4. Prior-gated residual does not rescue the method story

Corrected off-hours:
- `structured` ridge `≈ 0.912`
- `structured` gated `≈ 0.806`

So the paper should not force a weak method novelty story just because a stronger method label sounds more exciting.

## 8. The Most Credible Novel Contribution Right Now

If we write the contribution list honestly today, the novelty is not:

- a new SOTA architecture
- or a robust multimodal fusion win

The novelty is closer to this:

### C1. Problem formulation contribution

Show that off-hours `volatility shock` is a much better research object than raw post-call volatility level for ECC event modeling.

### C2. Evaluation contribution

Show that prior-aware and market-aware benchmark ladders fundamentally change the interpretation of ECC prediction performance.

### C3. Integrity contribution

Show that a migration/timing audit materially changes the regime story, and that corrected-panel reruns still support a strong off-hours structured signal.

### C4. Empirical finding

Show that simple finance-aware structured ECC features already explain off-hours post-call volatility shock very strongly on the corrected panel, while richer `Q&A` semantics help in a secondary, partially incremental way.

### C5. Negative-results contribution

Show that neither heavier multimodal unions nor more complex gating/stacking architectures automatically improve on the corrected structured core.

That is a clean, defensible, limitation-driven contribution package.

## 9. Safe Main Claim

The safest and strongest current claim is:

> Under noisy scheduled-time anchoring, off-hours earnings-call volatility shock can be predicted far better than same-ticker historical priors using simple finance-aware structured event features; richer `Q&A` semantics provide partial additional signal, but current multimodal and architecture-heavy extensions do not yet deliver robust incremental gains.

This is much stronger than a vague "multimodal finance" paper and much safer than overclaiming method novelty.

## 10. What We Should Not Claim

Do not claim:

- audio is a robust driver of the result
- current `Q&A` heuristics beat strong market baselines
- we have already proved a new SOTA architecture
- the strongest result is fully de-risked beyond DJ30
- ticker concentration is negligible

## 11. Recommended Paper Structure

### Introduction

Lead with:
- noisy scheduled time
- target confounding under raw volatility
- identity leakage risk
- need for prior-aware and market-aware evaluation

### Research question

Can off-hours earnings-call information explain incremental post-call volatility shock beyond same-ticker priors under noisy timing?

### Methods

Center the paper on:
- target redesign
- corrected-panel audit
- prior-aware residual evaluation
- benchmark ladder

### Results

Lead with corrected off-hours structured result, then:
- clean sensitivity
- year/regime robustness
- concentration and leave-one-ticker-out
- unseen-ticker stress
- `Q&A` and architecture as secondary layers

### Discussion

Emphasize:
- why structured features dominate
- why `Q&A` remains promising but secondary
- why negative architecture results matter

## 12. Immediate Next Step For Maximum Scientific Payoff

If the goal is to strengthen the paper rather than just add more experiments, the next highest-value step is:

1. keep the corrected off-hours structured result as the locked empirical core
2. upgrade `Q&A` supervision from heuristics to transferred pair-level labels
3. rerun the same benchmark ladder
4. only if that succeeds, revisit heavier sequence or multimodal methods

In other words:
- the project now deserves a more solid research story,
- and that story is driven by better problem design and cleaner evidence,
- not by forcing another round of flashy modeling.
