# Research Plan

## Working Title

Off-Hours Earnings-Call Volatility Shock Prediction Under Noisy Timing

## Current Mainline

The project should no longer be described as a generic multimodal ECC volatility paper.

The strongest current research object is:

- sample: `pre_market + after_hours` earnings conference calls
- target: `shock_minus_pre = post_call_60m_rv - pre_60m_rv`
- evaluation frame: same-ticker prior plus residual prediction
- feature frame: finance-aware event-level ECC signals with `Q&A` semantics as incremental evidence

This is a materially better framing than the earlier raw-volatility version because:

- raw post-call volatility level is heavily dominated by firm-specific scale
- same-ticker prior is a very strong baseline on the raw target
- off-hours calls provide a cleaner information-release environment
- the shock target better isolates the market response that can plausibly be attributed to the call

## Updated Main Claim

For off-hours earnings calls, event-level earnings-call features explain incremental post-call volatility shock beyond same-ticker historical priors, even under noisy scheduled-time anchoring.

The right emphasis is:

- incremental volatility shock rather than raw volatility level
- off-hours events rather than all calls pooled together
- prior-aware residual prediction rather than naive direct regression
- disciplined ablation rather than indiscriminate multimodal complexity

## What Changed

Earlier versions of the project assumed the hardest and most publishable problem was:

- timestamp-aware multimodal sequence modeling of all calls
- with raw post-call volatility as the main target

That is no longer the best current reading of the evidence.

The experiments now show:

- raw target performance is bottlenecked by same-ticker identity and volatility scale
- the project becomes much stronger when the target is redesigned around volatility shock
- off-hours calls are much cleaner than regular-hours calls
- heavy feature accumulation does not automatically help once the target and subset are chosen well

So the current strongest version is a more disciplined finance paper:

- choose the right target
- choose the right event subset
- beat the right prior
- show which ECC signals truly add incremental value

## Current Empirical Picture

On the raw `post_call_60m_rv` target:

- `ticker_expanding_mean` is the strongest hard baseline on test
- richer ECC features help on validation but do not consistently beat the prior

On redesigned targets:

- `log_post_over_pre` becomes clearly learnable
- `shock_minus_pre` becomes the strongest target by far

On `shock_minus_pre` with the off-hours subset:

- `prior_only` test `R^2` is about `0.196`
- `residual_structured_only` test `R^2` is about `0.916`
- adding `extra`, `qa_benchmark`, and `audio` does not improve over the simplest strong residual core
- `Q&A LSA` helps relative to a weaker overbuilt model, but still does not clearly beat the structured residual core

This means the main result is now:

- not that multimodal fusion is universally strong
- but that the right target and event subset unlock a strong, prior-aware ECC forecasting signal

## Primary Research Questions

RQ1. Can ECC event features predict off-hours post-call volatility shock beyond same-ticker priors?

RQ2. Which feature families are truly necessary once the target is defined as volatility shock instead of raw level?

RQ3. Are `Q&A` semantics useful as incremental signals, or is most of the value already carried by structured finance-aware event features?

RQ4. How sensitive are the results to noisy scheduled-time anchoring and regime choice?

RQ5. Does the signal generalise beyond DJ30 once the pipeline is transferred to a larger restricted sample?

## Primary Sample

Current pilot sample:

- DJ30 firms
- calls linked across `A1`, `A2`, `A4`, `C1`, `C2`, and `D`
- current event panel size: `553`

Primary analysis subset:

- keep only `pre_market` and `after_hours`
- drop `market_hours` from the main result because it is smaller and noisier

Current strongest split for the main result:

- train `248`
- validation `82`
- test `167`

This subset is not just a convenience filter. It is now part of the empirical object:

- after-hours and pre-market earnings calls are where the call itself is most plausibly the main information event

## Data Tables and Their Roles

### A1

Use for:

- structured transcript components
- question/answer boundaries
- speaker and role parsing
- `Q&A`-based feature construction

### A2

Use for:

- scheduled call timestamp
- transcript header metadata
- transcript completeness and integrity checks

Constraint:

- scheduled time is not actual start time

### A3

Use for:

- audio-derived features
- optional delivery-based checks

Current role in the paper:

- supplementary modality
- not current headline source of gain

### A4

Use for:

- sentence-level timing inside the call
- timing quality and coverage features
- strict versus broad QC

Constraint:

- noisy supervision rather than gold-standard alignment

### C1 and C2

Use for:

- earnings surprise
- analyst dispersion and coverage
- baseline event controls

### D

Use for:

- pre-call, within-call, and post-call 5-minute market activity
- target construction for raw and shock-style volatility outcomes

Important note:

- the presence of extended-hours bars makes the off-hours design feasible

## Target Construction

### Primary target

`shock_minus_pre = post_call_60m_rv - pre_60m_rv`

Why this is now the primary target:

- removes a large amount of firm-level volatility level
- better matches the idea of incremental market reaction to the call
- strongly reduces the advantage of same-ticker scale alone
- produces a much more convincing out-of-sample signal

### Secondary targets

- `log_post_over_pre`
- `log_post_over_within`
- raw `post_call_60m_rv` only as a robustness or contrast target

The raw target should no longer be the headline result.

## Main Modeling Strategy

### Core forecasting frame

Use a prior-aware residual design:

`prediction = same_ticker_expanding_prior + residual_model(ECC_features)`

This is the right frame because:

- it directly tests whether ECC features add event-specific signal
- it avoids hiding identity effects inside a larger regressor
- it is more honest than comparing only to weak baselines

### Current strongest model class

- residual ridge on event-level structured ECC features

Current interpretation:

- simple, regularized event-level models are stronger and more stable than the current heavier multimodal stack

### Feature ladder

1. `structured`
- pre-call and within-call market controls
- call duration
- scheduled hour
- analyst and surprise controls
- transcript and timing quality statistics

2. `extra`
- finance-aware event features
- uncertainty
- negativity
- guidance rates
- evasiveness proxies
- `Q&A` interaction summaries

3. `qna_lsa`
- dense semantic representation of `Q&A` text

4. `qa_benchmark`
- weak-label benchmark-style `Q&A` metrics

5. `audio`
- audio summaries and embeddings

Current ranking from the fixed main setting:

- strongest holdout result: `structured_only`
- useful but not dominant: `Q&A LSA`
- currently not robust: `qa_benchmark`, `audio`

## Evaluation Design

### Temporal split

- Train: `<= 2021`
- Validation: `2022`
- Test: `>= 2023`

### Main metrics

- `R^2`
- RMSE
- MAE

### Required comparisons

1. `prior_only`
2. `residual_structured_only`
3. `residual_structured_plus_extra`
4. `residual_structured_plus_extra_plus_qna_lsa`
5. `residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark`
6. `... + audio`

### Significance and robustness

The fixed main setting now requires:

- paired bootstrap confidence intervals
- paired sign-permutation tests on test errors
- yearly or subperiod stability
- ticker concentration checks
- strict timing robustness checks

## What Is Actually Novel Now

The strongest remaining novelty is not a giant multimodal architecture.

The current defensible contribution package is:

1. identifying the correct financial target:
- off-hours volatility shock instead of raw post-call level

2. identity-aware evaluation:
- explicit comparison against same-ticker priors

3. finance-aware ECC event modeling:
- showing which call-derived signals matter after target redesign

4. negative-but-important ablation evidence:
- more modalities and more feature blocks do not automatically improve the best holdout result

This is narrower but more publishable than the earlier generic multimodal story.

## Main Weaknesses Still Open

Even after the new mainline, the project is not yet top-tier ready.

Current weaknesses:

- sample is still only DJ30-scale
- the best result comes after target and subset redesign, so target-selection skepticism must be addressed
- the current strongest method is still simple
- multimodal contribution is not yet robust
- external validity beyond DJ30 is still unproven
- the paper still needs stronger year-wise and ticker-wise robustness

## Minimum Strong Paper Claim

If current findings continue to hold, the strongest near-term paper claim is:

For off-hours earnings calls, a prior-aware event-level ECC model predicts post-call volatility shock far better than same-ticker priors, and careful target design matters more than piling on additional modalities.

That is a real and defensible finance result.

It is not yet enough for a top-tier ACL or ICAIF paper by itself, but it is a strong core around which a publishable version can still be built if:

- the sample is scaled up
- robustness is expanded
- and a sharper methodological contribution is added on top of the current residual framework
