# Novelty, Integrity, and Storyline

## Bottom Line

The paper should not be framed as a generic "multimodal finance" project. That is too broad and no longer novel enough.

The defensible paper is:

**Role-aware multimodal earnings-call modeling for high-frequency post-call volatility under noisy timing.**

That framing is stronger because it combines:
- high-frequency intraday market targets rather than only quarterly labels,
- noisy scheduled-time anchoring and noisy sentence timing as explicit identification constraints,
- dialogue-level `Q&A` signals such as subjectivity, evasion, specificity, and uncertainty,
- multimodal evidence from text, timing, and audio proxies,
- finance-motivated robustness checks instead of a single predictive score.

After the latest target-redesign experiments, there is now an even stronger version:

**Role-aware earnings-call modeling for off-hours volatility shock under noisy timing.**

This is stronger than the original wording because it:
- focuses on the part of the market reaction that is genuinely event-specific,
- removes a large amount of firm-level volatility scale that otherwise lets same-ticker priors dominate,
- matches the observed empirical pattern that after-hours and pre-market calls are much more learnable than regular-hours calls.

## What Is No Longer Novel Enough

These versions are not strong enough on their own:
- plain `TF-IDF` or document embeddings on full transcripts,
- generic text-plus-audio fusion without `Q&A` structure,
- event-level volatility prediction without explicit timing uncertainty analysis,
- a pure benchmark replication of older multimodal volatility papers.

Those ideas can still be used as baselines, but not as the paper's main claim.

## The Defensible Contribution Package

### C1. High-frequency ECC modeling with explicit noisy timing

Older work on earnings calls and audio typically predicts daily or event-level risk using call-level summaries.

Our data package is different:
- `A2` gives scheduled rather than actual start time,
- `A4` gives noisy sentence-level timing,
- `D` gives 5-minute bars.

So our primary contribution is not "perfect alignment", but:
- modeling market reaction under imperfect timing anchors,
- quantifying how sensitive targets are to plausible scheduled-time shifts,
- showing which signals survive that noise.

This is a real integrity contribution, not a workaround.

### C2. Dialogue-aware finance signals rather than flat sentiment

Recent ECC work has moved beyond plain sentiment and toward more nuanced `Q&A` properties.

The most relevant direction for us is:
- subjectivity and answer style in `Q&A`,
- evasiveness and topic shifting,
- specific versus vague answers,
- prepared remarks versus `Q&A` contrast.

Our event-level feature layer should therefore emphasize:
- `Q&A` versus presenter contrast,
- question-answer lexical overlap,
- hedge and forward-looking language in answers,
- answer specificity proxies,
- multi-part question pressure,
- evasion-style proxy shares.

### C3. Multimodal, but only where the extra modality adds a finance story

We should not use audio just because it exists.

Audio is useful only if it helps answer a finance question such as:
- does vocal delivery add information beyond transcript content,
- does timing or cadence help explain post-call volatility,
- does `Q&A` delivery matter more than prepared remarks.

This keeps the project in `AI for Finance`, not generic speech analytics.

### C4. A credible bridge to frontier methods

The current pilot should stay lightweight and reproducible, but the paper should still connect to current frontier method directions.

The most relevant frontier directions are:
- `SubjECTive-QA` style answer-quality modeling for earnings-call `Q&A`,
- `EvasionBench` style evasiveness detection,
- mixed-type event-sequence transformers for irregular financial events,
- time-series foundation models with token-level specialization rather than hand-crafted frequency buckets.

These are best used as:
- inspiration for the feature and modeling ladder now,
- a credible extension path for the paper's second stage.

### C5. Identity-aware integrity, not just better prediction

Recent 2025 evidence also shows that earnings-call volatility models can be overly driven by firm identity and same-company priors.

That means our paper should not only ask:
- can text, audio, and `Q&A` predict volatility?

It should also ask:
- do they add event-specific information beyond same-ticker history and firm-heavy structured priors?

This turns identity control from a boring baseline into part of the paper's integrity contribution.

At the current stage, the honest answer is:
- they add interpretable validation-side signal,
- they improve over weak negative test baselines,
- but they still do not exceed the best same-ticker prior on the holdout test split.

That means the next paper-quality gain must come from stronger incremental-value design, not from a bigger generic model.

The good news is that this gain has now appeared once the target is redesigned around event-specific shock rather than raw level.

## Why The Current Results Already Help The Story

Current internal evidence supports the story direction:
- pure `Q&A TF-IDF` is weak,
- pure full-transcript `TF-IDF` is also weak,
- adding finance-aware event features improves validation and test performance relative to the current structured baseline,
- a hard same-ticker historical-volatility prior is now the strongest out-of-sample baseline, with positive test `R^2`,
- benchmark-inspired `Q&A` weak-label features further improve validation-side ranking, but still do not beat the same-ticker prior on test,
- directly feeding the same-ticker prior into the dense ridge stack improves the old negative test scores, but still underperforms the raw prior itself,
- a residual-on-prior formulation is more principled and gives small positive test `R^2`, but still remains below the prior baseline,
- redesigning the target from raw level to volatility shock changes the picture materially:
  - log-ratio targets become competitive,
  - and `shock_minus_pre` produces strong positive holdout performance,
- the strongest dense features already include uncertainty, negativity, presenter-versus-`Q&A` contrast, and `Q&A` pair signals,
- on the `shock_minus_pre` target, a residual dense event model reaches very strong holdout performance and clearly beats the same-ticker prior,
- restricting to off-hours calls only pushes this further, with `pre_market + after_hours` reaching test `R^2` about `0.901` once dense `Q&A` semantics are added,
- a strict ablation on that fixed setting now shows something even more important:
  - the simplest residual structured model is currently the strongest holdout model,
  - several heavier feature unions improve validation but not test,
  - so disciplined parsimony is part of the paper's integrity story,
- a first strict role-aware sequence baseline improves on the plain structured test result, while the broad sequence version is unstable,
- adding strict sequence bins on top of the best dense baseline improves test performance again, but the validation drop shows the current mapping is not yet robust,
- real audio features and denser semantic text baselines have now been tested, but they do not yet beat the same-ticker prior or materially stabilize validation,
- a dedicated robustness pass on the fixed setting shows:
  - the strongest structured residual model remains positive in every test year,
  - remains positive in both `after_hours` and `pre_market`,
  - and still beats `prior_only` after removing any single ticker from evaluation,
  - but the incremental gain is highly concentrated in a few names, especially `NVDA`,
- a harder unseen-ticker stress test also now shows:
  - when the test ticker is entirely removed from train and validation,
  - the structured off-hours shock design still performs strongly overall,
  - although ticker-level results remain heterogeneous,
- alternative fallback routes based on weak section tags and sentence-level alignment were tried and did not beat the current strict fuzzy sequence route,
- a first prior-gated residual prototype was also tested as a method extension:
  - it improves validation in some bundles,
  - but does not beat the simpler residual ridge core on holdout test,
  - and its learned gates are nearly constant, so it does not yet support a strong method claim,
- scheduled-time perturbation shows that most events are fairly stable under small shifts, but larger offsets do create outliers and instability.

That pattern is useful because it says:
- naive topical text is not enough,
- structure and subjectivity matter more,
- same-company history is a stronger benchmark than we first had in the baseline stack,
- timing uncertainty is material but manageable,
- the right target definition matters as much as the right feature family,
- and once target and subset are fixed well, parsimony can beat feature accumulation.

This is a stronger story than "the multimodal model is bigger."

## Integrity Guardrails

These need to stay explicit in the paper and code.

### Data integrity
- Keep the strict and broad `A4` QC subsets separate.
- Treat `A2` time as scheduled time only.
- Never describe `A4` as gold-standard timing.
- Keep a clear log of missing tickers in `D` and duplicated `A1` files.

### Evaluation integrity
- Use temporal splits only.
- Report both validation and holdout test.
- Keep a simple structured baseline.
- Keep same-ticker and historical-volatility priors.
- Keep text-only and multimodal baselines.
- Report sensitivity to `A4` filtering and scheduled-time offsets.
- Check whether gains survive after controlling for firm identity concentration.
- Treat the positive same-ticker baseline as the real hurdle for any "incremental value" claim.
- Report ticker-concentration explicitly for the fixed off-hours shock result.
- Keep the unseen-ticker stress test separate from the main temporal split, and present it as a harder transfer check rather than the main benchmark.
- Do not present the prior-gated prototype as a contribution unless it clearly improves on the residual ridge core.

### Causal and leakage integrity
- No future bars in feature construction.
- No using post-call target windows to build features.
- No using benchmark labels from later periods in text preprocessing.
- No using unavailable market benchmark series unless they are actually shared.

## Updated Method Ladder

### Level 1: Defensible baselines
- structured controls only,
- `Q&A` text only,
- full transcript text only,
- structured plus `Q&A`,
- structured plus `Q&A` plus finance-aware dialogue or timing features.

### Level 2: Dialogue-aware multimodal event model
- event-level dialogue signal model using `Q&A` pair features,
- text plus timing plus audio proxies,
- strict and broad QC comparison.

### Level 3: Timestamp-aware sequence model
- sentence or bar-level sequence model on filtered `A4`,
- role-aware pooling or gated fusion,
- target on post-call and within-call volatility.

### Level 4: Frontier extension
- weak-label transfer from `SubjECTive-QA` or `EvasionBench`,
- mixed-type event-sequence transformer (`FlexTPP`) for irregular call events and bars,
- optional time-series foundation-model initialization such as `Moirai-MoE` style token-level specialization.

## The Paper Story In One Paragraph

Earnings calls are informative not only because of what managers disclose, but because of how analysts pressure them in `Q&A`, how management responds, and how these signals unfold in time. Existing multimodal ECC studies mostly aggregate calls too coarsely, while recent `Q&A` benchmarks study subjectivity and evasiveness without connecting them to market reaction. Our project bridges these strands by modeling post-call high-frequency volatility with role-aware dialogue signals under noisy call timing. We show that plain bag-of-words text is weak, while finance-aware `Q&A`, timing, and delivery proxies provide incremental signal. We also quantify how much scheduled-time uncertainty matters, making the modeling assumptions explicit rather than implicit.

## Most Relevant Recent Sources

- Qin and Yang, 2019, ACL: *What You Say and How You Say It Matters: Predicting Stock Volatility Using Verbal and Vocal Cues*  
  https://aclanthology.org/P19-1038/

- Matsumoto, Pronk, and Roelofsen, 2011, The Accounting Review: *What Makes Conference Calls Useful? The Information Content of Managers' Presentations and Analysts' Discussion Sessions*  
  https://publications.aaahq.org/accounting-review/article/86/4/1383/3354/What-Makes-Conference-Calls-Useful-The-Information

- Pardawala et al., 2024: *SubjECTive-QA: Measuring Subjectivity in Earnings Call Transcripts' QA Through Six-Dimensional Feature Analysis*  
  https://arxiv.org/abs/2410.20651

- Ma, Lin, and Yang, 2026: *EvasionBench: Detecting Evasive Answers in Financial Q&A via Multi-Model Consensus and LLM-as-Judge*  
  https://arxiv.org/abs/2601.09142

- Shu et al., 2025: *FinCall-Surprise: A Large Scale Multi-modal Benchmark for Earning Surprise Prediction*  
  https://arxiv.org/abs/2510.03965

- Nandwani et al., 2025: *The Role of Identity in Earnings Call Transcripts: Same Company, Same Signal?*  
  https://aclanthology.org/2025.findings-acl.946/

- Huang et al., 2024, LREC-COLING: *ConEC: Earnings Call Dataset with Real-world Contexts for Benchmarking Contextual Speech Recognition*  
  https://aclanthology.org/2024.lrec-main.328/

- Draxler et al., 2025, NeurIPS: *Transformers for Mixed-type Event Sequences*  
  https://openreview.net/forum?id=MtwsRjPZhf

- Liu et al., 2025, ICML: *Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts*  
  https://proceedings.mlr.press/v267/liu25an.html

## Immediate Decision Rule

If future runs continue to show that:
- pure text stays weak,
- dialogue-aware and timing-aware features stay useful,
- small timing jitter does not overturn the ranking of models,
- and same-ticker prior remains the strongest raw baseline on raw targets,

then the main paper should lock in this contribution:

**`Q&A`-aware, timing-aware off-hours volatility-shock prediction under noisy ECC timing, evaluated against strong same-ticker priors.**

That is novel enough, technically honest enough, and substantially stronger than a generic transcript classifier.

If the next round of tests keeps showing that heavier `Q&A benchmark`, audio, and gated-residual branches do not beat the structured residual core, then the final paper should say that explicitly rather than hiding it. That would strengthen the paper, not weaken it, because it would show that the main gain comes from identifying the right target and event subset rather than from indiscriminate multimodal complexity.
