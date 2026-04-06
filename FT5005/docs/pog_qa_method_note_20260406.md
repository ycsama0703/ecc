# I-POG-QA Method Note (2026-04-06)

## Purpose

This note upgrades the earlier `POG-QA` prototype into the current paper-facing method:

- `I-POG-QA = Incremental Prior-Aware Observability-Gated Q&A Residual`

The key change is conceptual, not cosmetic:

- the model should not merely mix `Q&A` corrections;
- it should only trust those corrections when observability and answerability support them;
- and it should be regularized to learn an **increment beyond** the strong same-ticker prior and structured base shell.

## Why this is the right next method

The repository evidence is already narrow enough to rule out several directions.

What the result stack currently says:

- `shock_minus_pre` remains the cleanest current target anchor.
- On the strict fixed-split clean `after_hours` setting, the safest ECC increment remains:
  - `pre_call_market + controls + A4 + compact Q&A semantics`
- The earlier global prior-gated residual idea underperformed on the strongest bundles because it used one shallow gate to scale the entire residual branch.
- Transfer-side work suggests that richer router complexity is not the main bottleneck.
- `Q&A` signal is real, but it is conditional:
  - some events are better explained by semantics,
  - some by answer-quality / evasion / accountability structure,
  - and many should still default back to the stable structured shell.

That implies a disciplined method objective:

- keep the strong prior;
- keep the stable structured residual core always active;
- let the model decide when a dialog correction is trustworthy;
- let it decide whether the correction should be more semantic or more accountability-driven;
- and keep that correction explicitly incremental rather than identity-like.

## Method definition

`I-POG-QA` predicts:

`prediction = prior + base_residual + trust_gate * (route_gate * semantic_correction + (1 - route_gate) * quality_correction)`

Components:

1. `prior`
- same-ticker expanding mean
- identical backbone to the current prior-aware evaluation protocol

2. `base_residual`
- residual ridge on:
  - pre-call market features
  - control variables
  - `A4` observability / timing-quality features
- always active

3. `semantic_correction`
- compact `Q&A` text semantic expert
- currently built from `qna_text` TF-IDF + low-rank `LSA`

4. `quality_correction`
- compact weak-label `Q&A` answer-quality / accountability expert
- built from directness, coverage, evasion, specificity-gap, delay, and numeric-mismatch style features

5. `trust_gate`
- decides whether the dialog correction branch should activate
- uses observability and answerability features
- is now **direction-aligned and monotone**

6. `route_gate`
- decides whether the applied dialog correction should lean more on semantic content or on answer-quality structure

## What changed versus the first POG-QA prototype

### 1. Monotone trust gate

The trust gate is no longer a free-sign linear gate.

It now uses:

- direction-aligned reliability features;
- nonnegative trust weights after parameterization;
- so higher observability / answerability evidence cannot perversely reduce trust.

In practice:

- higher `a4_strict_row_share`, `a4_median_match_score`, `qa_bench_coverage_mean`, and `qa_bench_direct_answer_share` should increase trust;
- higher `qa_bench_evasion_score_mean`, `qa_pair_low_overlap_share`, `qa_bench_short_evasive_share`, and `qa_bench_numeric_mismatch_share` should decrease trust.

This makes the method more defendable as a reliability-aware model rather than a generic learned gate.

### 2. Incrementality constraint

The model now regularizes the **applied dialog effect** so it does not simply reconstruct the prior or the structured base residual.

Current implementation:

- penalize covariance between the applied dialog effect and:
  - standardized prior reference
  - standardized base-residual reference

This does not guarantee perfect orthogonality, but it does make the modeling objective consistent with the actual paper question:

- does `Q&A` provide an incremental event-specific correction beyond strong priors?

### 3. Default-to-base activation regularization

The model also lightly penalizes unnecessary dialog-effect magnitude.

Purpose:

- default back to `prior + base_residual` unless there is enough evidence that the `Q&A` correction is worth activating.

This matches the main empirical reading of the repository:

- `Q&A` is not a universal always-on signal;
- it is selective.

## Why this is more novel than a wider benchmark ladder

The innovation is not "more features."

The innovation is the structure:

- it is **prior-aware** rather than raw end-to-end;
- it is **hierarchical**, because the stable structured shell is separated from the higher-variance dialog shell;
- it is **observability-gated**, because timing / coverage reliability controls whether dialog should matter;
- it is **dual-expert on `Q&A`**, because semantics and accountability are treated as different expert families;
- it is **incrementality-regularized**, because the dialog effect is explicitly pushed to be something other than a disguised prior/base proxy.

This is a better paper contribution than:

- another larger feature union;
- another transfer router family;
- another generic global gate;
- or a heavier whole-transcript architecture unsupported by the current sample.

## Why it addresses the professor's feedback

This direction directly helps on the professor's main concerns:

1. clearer method contribution
- `I-POG-QA` is an explicit architecture, not a broad modeling pile

2. clearer feature construction
- prior, structured base, semantic expert, quality expert, trust gate, and route gate are all separable blocks

3. clearer benchmark story
- it is naturally compared against:
  - `prior_only`
  - structured residual ridge
  - structured + semantic ridge
  - structured + quality ridge
  - structured + semantic + quality ridge
  - `I-POG-QA`

4. cleaner target discussion
- the implementation supports:
  - `shock_minus_pre`
  - `within_minus_pre`
  - `log_within_over_pre`
  - `post_minus_within`
- so the repo can compare the current off-hours headline target with the professor-preferred during-ECC family using the same method shell

## Current implementation

Runnable script:

- `scripts/run_pog_qa_residual.py`

Current implementation flow:

1. fit the same-ticker prior
2. fit the structured residual base
3. fit semantic and quality experts on top of the base prediction
4. train the gated dialog layer with:
   - monotone trust parameterization
   - incrementality regularization
   - activation regularization

Current output artifacts:

- `i_pog_qa_residual_summary.json`
- `i_pog_qa_residual_predictions.csv`
- `i_pog_qa_residual_gate_diagnostics.csv`

The gate-diagnostic table is important because it makes the method paper-readable:

- prior prediction
- base residual component
- semantic correction
- quality correction
- route gate
- trust gate
- mixed dialog correction
- applied dialog effect
- final prediction

## What success should look like

`I-POG-QA` does not need to dominate every setting.

It needs to show one clean thing:

- on the locked main setting, it should improve over the best ungated `Q&A` extension by learning **when** and **which kind** of dialog correction to trust, while preserving the prior-aware structured shell.

Even if it fails to improve, the result is still informative:

- it would show that the conference paper should keep a simpler ridge mainline and present `I-POG-QA` as a disciplined negative method attempt rather than as the headline model.

## Immediate next run plan

1. Run `I-POG-QA` on the locked clean `after_hours + shock_minus_pre` setting.
2. Compare against the fixed benchmark ladder under the same split.
3. Run the same shell on `within_minus_pre` and `log_within_over_pre`.
4. Decide whether the conference headline should stay on the off-hours shock target or move toward the during-ECC family.
