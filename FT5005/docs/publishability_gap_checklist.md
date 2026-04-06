# Publishability Gap Checklist

## Bottom Line

The current project is no longer a weak class project.

It is now a strong research pilot with:

- a credible financial target redesign
- hard identity-aware baselines
- meaningful out-of-sample gains in the right setting
- real ablation and significance testing

But it is still **not submission-ready** for a strong ICAIF or ACL-style venue.

The main reason is not poor execution.

The main reason is that the current strongest version is still:

- small-sample
- pilot-scale
- methodologically conservative
- and only partially generalised

## What Is Already Strong

### 1. The project now has a real empirical core

The move to:

- off-hours calls
- shock target
- prior-aware residual evaluation

is a serious research improvement, not cosmetic tuning.

### 2. The evaluation is much more honest than before

The project now includes:

- same-ticker priors
- OLS and HAR-style baselines
- strict ablations
- paired bootstrap intervals
- paired permutation tests

This is well above ordinary course-project quality.

### 3. The current best result is financially meaningful

The strongest current setting says:

- the call helps explain incremental volatility shock,
- especially outside regular market hours,
- beyond a same-ticker historical baseline.

That is a real finding.

## Main Weaknesses Right Now

## W1. Sample size and external validity

Current strongest result is still based on:

- DJ30 only
- off-hours subset
- test size around `167` events

Why this matters:

- reviewers can dismiss the result as DJ30-specific
- large-cap firms have cleaner coverage, cleaner transcripts, and different market microstructure
- the paper currently lacks broad-market evidence

Current risk level:

- high

What is needed:

- larger restricted sample replication
- at minimum `SP500`-style expansion

## W2. Method contribution is still too modest

Current strongest model is basically:

- same-ticker prior
- plus residual ridge

Even though the task design is now much better, the method itself is still not novel enough for a strong venue.

Why this matters:

- ICAIF may tolerate a strong empirical paper more than ACL would
- but ACL/finance-NLP style reviewers will still ask what the method contribution really is

Current risk level:

- high

What is needed:

- one sharper prior-aware model contribution
- something like a prior-gated or regime-aware residual architecture

Current update:

- a first prior-gated residual prototype has now been tested
- it improves validation in some bundles
- but it does not beat the simpler ridge residual core on holdout test

Interpretation:

- the method gap is now more concrete, not less
- we now know one plausible prior-aware extension is still insufficient

## W3. Target and subset were selected through exploration

The strongest result appears only after:

- trying multiple targets
- trying multiple regimes
- narrowing to the off-hours shock setup

This is scientifically reasonable during discovery, but dangerous in final presentation.

Why this matters:

- reviewers may view it as ex post target tuning
- even if the choice is actually well motivated

Current risk level:

- high

What is needed:

- explicit justification from financial microstructure
- transparent reporting of alternative targets
- clear statement of why the final target is the most event-specific and economically meaningful

## W4. Multimodal headline is not yet earned

Current evidence says:

- audio is not a robust holdout winner
- `qa_benchmark` is not a robust holdout winner
- the strongest model is not the heaviest multimodal stack

Why this matters:

- "multimodal" is currently more of a data asset than a proven contribution
- overselling that point would weaken the paper

Current risk level:

- medium to high

What is needed:

- either recover a real, stable multimodal gain
- or narrow the paper headline away from multimodal novelty

## W5. Sequence and alignment story is still incomplete

Current sequence route:

- has signal
- but remains unstable
- and is not the best current main result

Why this matters:

- this was originally a major part of the conceptual novelty
- without it, the paper becomes more event-level and less temporally granular

Current risk level:

- medium

Current update:

- year-wise stability is now partly addressed
- ticker concentration analysis is now partly addressed
- leave-one-ticker-out influence analysis is now partly addressed
- one harder unseen-ticker stress test is now partly addressed

What remains:

- paper-quality reporting and interpretation of the harder split
- ticker-blocked or larger-sample harder generalisation beyond DJ30
- larger-sample replication
- paper-ready robustness tables

What is needed:

- either improve the sequence story later
- or explicitly demote it to extension material

## W6. Robustness still needs one more level

Current robustness is good, but not yet enough for a final submission.

Still needed:

- year-wise stability
- ticker concentration analysis
- harder generalisation splits
- larger-sample replication

Current risk level:

- medium

## W7. Current best model may be too parsimonious for a high-impact claim

This is a subtle problem.

The fact that the best holdout model is simple is a strength for integrity.
It is also a challenge for publishability.

Why this matters:

- simple can be good
- but simple without a broader empirical or theoretical package can look under-ambitious

Current risk level:

- medium

What is needed:

- keep the parsimony
- but wrap it in stronger design, stronger robustness, and larger-sample evidence

## Challenges That Need Active Management

### C1. Do not drift back into feature sprawl

The latest ablation already showed:

- more features can hurt

So the project should not respond to every weakness by adding more modalities.

### C2. Do not let the paper overclaim multimodality

If audio remains weak, say so.
That is still publishable if the core event-design insight is strong enough.

### C3. Do not let the paper under-explain the target redesign

The entire new mainline depends on convincing the reader that:

- off-hours shock is the right target
- not just the best empirical accident

### C4. Do not let the current strong result remain DJ30-only

This is the biggest structural blocker.

### C5. Do not ignore concentration just because the result stays positive

The latest robustness pass shows two facts at once:

- the model still beats `prior_only` after removing any single ticker
- but most of the aggregate gain is concentrated in a very small number of names

Both need to be reported together.

### C6. Do not oversell the unseen-ticker stress test

The new harder split is encouraging, but:

- it uses a different baseline because same-ticker history is unavailable
- it still shows heterogeneous ticker-level performance

So it should support the external-validity story, not replace the main temporal benchmark.

## Checklist

## Must-Have Before Serious Submission

- lock the final main task definition in writing
- add year-wise and ticker-wise robustness
- add a larger-sample replication
- add one stronger prior-aware method
- keep full ablation and significance package
- justify target and subset selection explicitly
- add at least one harder generalisation split beyond the current temporal split

## Strongly Recommended

- benchmark-adjusted target if data become available
- harder split such as leave-firm-out or ticker-blocked testing
- more explicit comparison between raw level target and shock target in the paper
- better integration of the financial story around off-hours information release
- explicit reporting of gain concentration across tickers

## Optional but Useful

- stronger audio feature stack
- improved `Q&A` weak labels
- better sequence extension

## What Is Missing For ACL-Like Quality

To be clear, the current version is not ACL-level yet because:

- dataset scale is too limited
- method novelty is still too weak
- multimodal contribution is not robust enough
- the strongest result still looks like a strong pilot finding rather than a finished research package

To close that gap, the project would need:

- larger-sample validation
- a sharper modeling contribution
- a cleaner connection to recent dialogue or financial NLP methodology

## What Is Missing For ICAIF-Like Quality

The gap is somewhat smaller here, but still real.

Current version is not ICAIF-ready because:

- generalisability is unproven
- the strongest evidence is still on a small pilot subset
- the final empirical story is not yet locked with enough robustness

To close that gap, the project needs:

- large-sample replication
- stronger robustness
- clean tables and figures
- a more explicit contribution claim around target design and prior-aware forecasting

## Next-Stage Research Plan

## Phase 1. Freeze and strengthen the current core

Goal:

- turn the current best setting into a paper-quality core

Tasks:

1. freeze:
- off-hours
- shock target
- prior-aware residual

2. add:
- year-wise stability
- ticker concentration
- robustness appendix tables

3. finalise:
- main ablation table
- significance table
- contribution narrative

Definition of success:

- a fully coherent DJ30 paper core with no loose ends

## Phase 2. Add one real method contribution

Goal:

- move beyond plain residual ridge

Best candidate directions:

1. prior-gated residual model
2. regime-aware shrinkage model
3. sparse event-feature selection with explicit prior correction

Definition of success:

- one method that is still simple and defensible
- but clearly more original than current ridge

## Phase 3. Scale the sample

Goal:

- remove the biggest external-validity objection

Tasks:

1. get larger restricted sample
2. keep the same target and evaluation frame
3. rerun the canonical pipeline

Definition of success:

- the result is no longer DJ30-only

## Phase 4. Decide how to frame the paper

If the larger-sample result holds:

- write it as a submission-target paper

If it weakens materially:

- write it as a strong pilot / workshop / class-publication-track study
- keep the method and integrity lessons as the main contribution

## Recommendation

Right now, the highest-ROI path is:

1. stop expanding feature families blindly
2. consolidate the current off-hours shock result
3. build one stronger prior-aware model
4. scale the sample

That is the shortest path from "strong pilot" to "actually publishable paper."
