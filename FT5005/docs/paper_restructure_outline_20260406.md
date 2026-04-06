# Conference Paper Restructure Outline 2026-04-06

## Goal

Restructure the current project from a stage-report style write-up into a conference-paper style manuscript that is:

- clearer on the task;
- clearer on the protocol;
- stronger on benchmark comparison;
- and centered on one paper-worthy method contribution.

## One-sentence paper principle

The paper should no longer read as "we tried many multimodal ECC ideas and here are the latest good results." It should read as:

**under noisy call timing, can structured earnings-call signals improve high-frequency volatility-shock prediction beyond strong pre-call priors, and can a new prior-aware `Q&A`-centered method do this more effectively than existing benchmark families?**

## Recommended storyline hierarchy

### Layer A: mainline

- one main target family;
- one main fixed-split benchmark;
- one main method;
- one main incremental ECC claim.

### Layer B: secondary extension

- harder transfer or abstention analysis;
- included to show robustness and method discipline;
- not used as the main paper headline.

### Layer C: exploratory appendix branch

- hardest-question local signal;
- useful and interesting;
- but not allowed to replace the main paper narrative.

## Paper question and target framing

### Candidate A: off-hours post-call shock

Possible formulation:

- `pre_60m_rv`
- `within_call_rv`
- `post_call_60m_rv`
- headline target:
  - `post_call_60m_rv - pre_60m_rv`

Main issue to explain:

- whether overnight close-to-open movement is treated as part of the ECC-driven shock or outside the target window.

### Candidate B: during-call and post-call relative shock

Possible formulations:

- `within_call_rv - pre_60m_rv`
- `post_call_60m_rv - pre_60m_rv`

This is closer to the professor's suggested research object because it is more tightly connected to the call itself.

### Recommended decision rule

1. Run the same benchmark ladder on both target families.
2. Prefer the target that is:
   - financially cleaner;
   - easier to define precisely;
   - and still empirically competitive.
3. Keep only one as the headline target.
4. Put the other in robustness or secondary analysis.

## Recommended title direction

Avoid:

- generic "multimodal earnings call prediction";
- or broad "text-audio finance" titles.

Prefer titles in this shape:

1. **Prior-Aware Earnings-Call Volatility Shock Prediction Under Noisy Timing**
2. **Learning Incremental Earnings-Call Signals Beyond Strong Market Priors**
3. **Prior-Aware `Q&A`-Centered Volatility Shock Forecasting from Noisy Earnings Calls**
4. **I-POG-QA: Incremental Prior-Aware Observability-Gated `Q&A` Residual Learning for Earnings-Call Volatility Shock**

## Recommended section structure

## 1. Introduction

### What this section must do

- define the problem clearly in the first page;
- explain why naive ECC prediction is not enough;
- explain why strong priors make the task hard;
- and state the exact contribution package.

### Suggested logic

1. Earnings calls contain information, but timing is noisy.
2. High-frequency market reaction is financially meaningful but hard to measure cleanly.
3. Same-firm or pre-call priors are strong and can dominate weak ECC claims.
4. We study whether ECC features add incremental value beyond those priors.
5. We propose a prior-aware `Q&A`-centered method under noisy observability.

### Introduction contributions block

Suggested final contribution structure:

1. A clear event-level volatility-shock task definition under noisy ECC timing.
2. A benchmark protocol with strong prior-aware baselines and temporal holdout.
3. A new prior-aware observability-gated `Q&A` residual method.
4. Evidence that the clean incremental signal is narrow, structured, and not a generic multimodal effect.

## 2. Related Work

This section should be grouped, not listed chronologically.

### 2.1 Multimodal ECC market prediction

Need to position against:

- Qin and Yang (2019);
- MAEC / aligned ECC benchmark line;
- VolTAGE and similar text-audio fusion work.

Main message:

- prior work shows multimodal ECC prediction is real;
- but most work either aggregates too coarsely or does not treat noisy timing and strong priors as central.

### 2.2 `Q&A`-aware financial dialogue modeling

Need to position against:

- Financial Risk Prediction with Multi-Round Q&A Attention Network;
- recent `Q&A` quality or evasion works;
- answerability / non-response literature.

Main message:

- `Q&A` structure matters more than flat transcript pooling;
- but it has not yet been integrated cleanly into our high-frequency prior-aware volatility setting.

### 2.3 Identity-aware and prior-aware evaluation

Need to cite the line showing transcript models can be identity-dominated.

Main message:

- strong priors are not optional baselines;
- they are part of the contribution standard.

## 3. Data and Task Definition

### 3.1 Data assets

Describe:

- `A1` transcript JSON;
- `A2` scheduled time HTML;
- `A3` audio;
- `A4` noisy sentence timing;
- `C1/C2` analyst controls;
- `D` 5-minute market bars.

### 3.2 Event construction and filtering

Must include one table with:

- raw counts by source;
- matched event counts;
- missing ticker counts in `D`;
- final modeling panel counts;
- clean `after_hours` subset counts if retained.

### 3.3 Timing assumptions

Must say explicitly:

- `A2` gives scheduled time, not guaranteed actual start time;
- `A4` is noisy observability / alignment supervision, not gold timestamp truth.

### 3.4 Target definition

This subsection needs formulas, not only prose.

Must define:

- `pre_60m_rv`;
- `within_call_rv`;
- `post_call_60m_rv`;
- headline target;
- secondary targets.

Also state clearly:

- whether overnight jump is included or excluded;
- whether the call window uses scheduled start plus `A4`-estimated duration;
- and why the chosen target is the most event-specific one.

### 3.5 Sample period and split

Must include:

- exact sample start date and end date from the final event panel;
- train / validation / test rule;
- fixed-split sizes;
- transfer split definition;
- and a statement that all splits are temporal.

## 4. Baselines and Benchmark Protocol

### 4.1 Prior and market baselines

Need a formula for:

- `prior_ticker_expanding_mean`

And a clear distinction between:

- prior-only;
- market-only;
- market-plus-controls.

### 4.2 ECC baselines

Need grouped benchmark families:

1. ECC structured only
2. ECC structured plus compact `Q&A` semantics
3. ECC plus weak-label `Q&A` features
4. ECC plus audio or aligned-audio

### 4.3 Literature-inspired baselines

Need at least:

1. Multi-round `Q&A` attention style baseline
2. Compact text-audio co-attention or late-fusion baseline

Important note:

- these can be adapted approximations using current data and sample size;
- they do not need to be identical replications if the dataset differs.

### 4.4 Evaluation protocol

Must state:

- primary metric;
- secondary metrics;
- significance testing;
- robustness slices;
- and what counts as the main benchmark versus transfer extension.

## 5. Proposed Method

### Recommended method

#### Incremental Prior-Aware Observability-Gated `Q&A` Residual Network

Abbreviation:

- `I-POG-QA`

### Core idea

Predict the target as:

- a strong prior backbone;
- plus a selectively trusted ECC residual.

Conceptually:

`prediction = prior + base_residual + trust_gate * (route_gate * semantic_correction + (1-route_gate) * accountability_correction)`

### 5.1 Prior branch

Inputs:

- same-ticker prior;
- pre-call market features;
- optional basic controls.

Role:

- provide the strong baseline forecast;
- define the hurdle the ECC branch must beat.

### 5.2 `Q&A` residual branch

Inputs:

- compact round-level or pair-level `Q&A` text representation;
- `A4` timing / observability features;
- optional low-rank aligned audio factors.

Design principle:

- keep the branch compact;
- model `Q&A` structure directly;
- avoid full heavy sequence modeling on the current sample.

### 5.3 Observability gate

Inputs:

- `A4` reliability / coverage;
- answerability or directness proxies;
- regime context;
- optional uncertainty about the prior.

Role:

- determine when the ECC residual should meaningfully adjust the prior.
- keep the trust branch direction-consistent with observability / answerability evidence.

### 5.4 Incrementality regularization

Role:

- constrain the applied dialog effect to remain incremental beyond the strong prior and structured base shell.

### 5.5 Multi-horizon head option

If Case 2 survives target selection:

- use shared ECC representation;
- predict both `during` and `post` shock with separate heads;
- allow the paper to argue that the method captures both immediate and short-horizon reaction.

### Why this method fits the current data

1. It preserves the strong prior benchmark.
2. It uses `Q&A` structure, matching the professor's cited literature.
3. It uses `A4` as noisy trust information, matching the actual data-generating constraints.
4. It stays compact enough for the current sample size.
5. It is more method-like than plain residual ridge without becoming a large-model overreach.

## 6. Main Results Section

### Table 1: data and sample summary

Must show:

- raw source coverage;
- usable panel counts;
- final headline sample counts;
- split sizes.

### Table 2: target comparison

Compare:

- raw level;
- log ratio;
- shock target;
- during-call candidate if retained.

This table justifies the final headline `Y`.

### Table 3: benchmark ladder

Compare:

1. prior-only
2. market-only
3. market-plus-controls
4. current best residual ridge mainline
5. literature-inspired baselines
6. proposed `I-POG-QA`

This is the main paper table.

### Table 4: ablation of the proposed method

Possible ablations:

- no gate;
- no incrementality regularization;
- non-monotone trust gate;
- no `A4`;
- no pair-level `Q&A`;
- no audio side branch;
- single-horizon versus multi-horizon.

### Figure 1: target-window illustration

This should visually answer the professor's target question.

### Figure 2: model diagram

Show:

- prior branch;
- residual branch;
- observability gate;
- optional multi-horizon heads.

### Figure 3: main benchmark ladder

Simple performance chart for fast reader comprehension.

## 7. Robustness and Extensions

### Robustness that must stay in the main paper

- year-wise stability;
- ticker concentration or influence;
- timing jitter sensitivity;
- strict versus broad QC;
- headline target versus alternative target family.

### Secondary extension material

- transfer abstention;
- harder unseen-ticker split;
- audio-side diagnostics.

### Appendix or demoted material

- hardest-question local branch;
- router-family search history;
- large set of negative feature-sprawl attempts.

## 8. Discussion

This section should openly state:

- the strongest signal is narrow rather than broad;
- the paper is about incremental ECC value beyond priors;
- noisy timing is a first-class constraint;
- audio is conditional rather than universally helpful;
- and broader external validity still depends on later scale-up.

## 9. Conclusion

The conclusion should not claim:

- generic multimodal dominance;
- universal transfer gains;
- or solved ECC understanding.

It should claim:

- a cleaner task definition;
- an integrity-aware benchmark protocol;
- one new prior-aware `Q&A`-centered method;
- and evidence that structured ECC signals can add incremental value beyond strong priors in the right setting.

## Immediate writing tasks

### Task 1

Rewrite the current abstract around:

- task definition;
- strong prior baseline;
- noisy timing;
- and one main method.

### Task 2

Rewrite the experimental setup section into:

- data;
- event construction;
- target definition;
- sample and split;
- benchmark ladder.

### Task 3

Rewrite the result section so that:

- target comparison comes before model comparison;
- benchmark ladder comes before extensions;
- and the transfer / hardest-question material is clearly separated.

### Task 4

Insert a formal method section for `I-POG-QA`.

### Task 5

Add a related-work section explicitly built around the professor's suggested references.

## Final restructuring rule

If a result does not help answer one of the following questions, it should not stay in the main paper body:

1. What exactly is the task?
2. Why is this target financially meaningful?
3. What is the right integrity benchmark?
4. What does the new method do beyond the current residual ridge line?
5. Does the new method beat recognizable baselines in the headline setting?

Everything else should be shortened, moved, or demoted.
