# SOTA Workflow and Gap Analysis

## Goal

This memo extends the current mainline into a full research workflow and maps the most relevant recent methods, modules, and open projects onto that workflow.

The point is not to chase every new model. The point is to decide:
- which recent methods are actually relevant to our ECC volatility problem,
- which ones are useful now versus later,
- which ones strengthen novelty and integrity,
- and which ones are likely to waste time or dilute the paper.

## Current Mainline

The current strongest and safest story is still:

**event-level finance-aware modeling of post-call high-frequency volatility, with strict role-aware sequence modeling kept as an extension rather than the main claim.**

This is where we currently stand:
- event-level finance-aware features are the strongest stable baseline,
- strict sequence features contain signal but are not yet validation-stable,
- broad sequence features are too noisy,
- alternative weak section and sentence-level sequence routes did not outperform the current strict fuzzy route.

So the current question is not "what is the fanciest model we can import?"

The current question is:

**what recent methods can most credibly improve the existing mainline without breaking the story or the timeline?**

## Full Workflow Extrapolation

The project can be viewed as a six-stage pipeline.

### Stage 1: Data and event integrity

Inputs:
- `A1` structured ECC transcript JSON
- `A2` HTML transcript and scheduled time
- `A3` audio
- `A4` noisy timed sentence alignment
- `C1/C2` surprise and analyst controls
- `D` 5-minute bars

Current status:
- this layer is already usable,
- but the integrity bottlenecks remain scheduled-time noise, `A4` noise, and missing `D` tickers.

Main upgrades that matter here:
- better actual-start approximation or explicit timing uncertainty handling,
- reproducible event-level QC versions,
- optional market benchmark bars for residualized targets.

### Stage 2: Target construction

Current main target:
- post-call 60-minute realized volatility proxy

Near-term improvements:
- residualized or market-adjusted post-call volatility once a benchmark index or ETF series is available,
- signed reaction or jump indicator as a secondary target,
- post-call volume shock as an additional market-attention style outcome,
- event-time sensitivity analysis by scheduled-hour and after-hours status.

### Stage 3: Representation learning

This is the layer where most realistic gains still remain.

There are four sub-problems:
- finance-aware text representation,
- `Q&A` semantics,
- audio representation,
- event or sequence representation.

### Stage 4: Predictive modeling

Current best practical model:
- dense event-level finance-aware ridge baseline

Next likely modeling gains:
- event-level nonlinear tabular models,
- multimodal late fusion,
- strict sequence fusion only after sequence labeling quality improves.

### Stage 5: Evaluation and robustness

This is a major part of paper quality, not a side section.

Necessary checks:
- rolling or expanding temporal validation,
- strict versus broad QC comparison,
- scheduled-time jitter robustness,
- ablation of text, audio, and `Q&A` features,
- calibration of gains by subperiod and firm.

### Stage 6: Scale-up

If the DJ30 pilot is stable, the next real scale-up path is:
- compute restricted embeddings onsite,
- export non-restricted derived features,
- retrain on larger `SP500/SP1500`-style samples if allowed.

That scale-up should happen only after the event-level mainline is locked.

## SOTA Landscape By Layer

## 1. ECC and benchmark datasets

### Most relevant recent resources

`FinCall-Surprise` is the strongest recent open benchmark directly adjacent to our problem.
It provides `2,688` corporate calls with text, audio, and slides for earnings-surprise prediction from `2019` to `2021`.
It matters because it is currently the cleanest open multimodal ECC benchmark for pretraining ideas, external sanity checks, and dataset positioning.

`ConEC` is not a volatility benchmark, but it is highly relevant because it frames earnings calls as a contextual ASR problem with real-world supplementary context.
It matters for us because our current sequence bottleneck is partly an alignment problem.

`SubjECTive-QA` and `EvasionBench` are not market-prediction datasets, but they are likely the most relevant recent `Q&A` signal resources.
They matter because our strongest event-level results already point toward uncertainty, negativity, and `Q&A` interaction signals rather than raw topic words.

### Implication for us

The paper should explicitly position itself between two lines:
- multimodal ECC market prediction,
- earnings-call `Q&A` quality or evasion modeling.

That positioning is stronger than pretending we are only doing generic multimodal finance.

### Closest adjacent papers we should actually learn from

There are several recent papers that are closer to our mainline than generic finance NLP work.

`What You Say and How You Say It Matters` remains the canonical verbal-plus-vocal volatility paper.
It is older, but still the cleanest baseline for why audio belongs in this topic at all.

`AMA-LSTM` and `SH-Mix` matter because they represent the stronger recent benchmark-engineering line for multimodal ECC prediction.
They show that hierarchical multimodal fusion, augmentation, and robustness-oriented training are still active design patterns in this literature.

`DialogueGAT` matters because it treats earnings calls as dialogue structure rather than flat text.
Conceptually, it supports our role-aware and turn-aware direction even though our current best implementation is still event-level rather than graph-based.

`What Is the Impact of Managerial Non-Responses in Earnings Calls?` is directly aligned with our `Q&A` story.
It matters because it treats non-response as an economically meaningful signal rather than a nuisance.

`The Sound of Risk` is another close adjacent work.
It reinforces the argument that delivery and vocal cues are still underused in financial risk prediction.

Most importantly, `Same Company, Same Signal` is the strongest recent integrity warning for this whole line of work.
It shows that transcript-based volatility models can be dominated by ticker identity and same-firm priors rather than by event-specific information.
That means identity-confounding controls are no longer optional in a serious ECC paper.

## 2. Finance-aware text and `Q&A` modeling

### What is currently SOTA-adjacent for our use case

The strongest recent direction is not a generic financial LLM as the final predictor.
The strongest direction is **task-specific weak supervision or representation learning around `Q&A` semantics**.

Most useful ideas from recent work:
- six-dimensional subjectivity decomposition from `SubjECTive-QA`,
- evasion taxonomies and classifiers from `EvasionBench`,
- role and turn sensitivity rather than document-level pooling.

### What this means for our pipeline

The best near-term text upgrade is:
- not "replace ridge with an LLM",
- but "use recent `Q&A` benchmarks to build stronger financial dialogue features."

High-value additions we do not yet have:
- question specificity versus answer specificity,
- answer directness,
- forward-looking versus backward-looking answer balance,
- answer coverage ratio,
- explicit evasion probability or weak labels.

### Additional integrity implication

Recent evidence also says we need to separate:
- event-specific dialogue information,
- from firm identity and historical volatility priors.

So stronger `Q&A` modeling is only publishable if we also show it beats or complements:
- same-firm historical volatility baselines,
- simple prior-based rules,
- and firm-heavy structured controls.

### Useful open projects

`FinGPT` is relevant as an open financial LLM project, but mainly as a feature-generation or weak-label tool.
It is not the right main predictive backbone for our current sample size and current problem design.

Best use of `FinGPT` in our project:
- prompt-based weak labeling,
- explanation generation,
- earnings-call `Q&A` taxonomies,
- financial phrase normalization.

## 3. Audio and speech stack

### What is most relevant now

Our largest current technical gap is audio.
Right now the project only uses lightweight audio proxies.
That is a clear gap relative to what the data allows.

The most useful open audio stack is:
- `Whisper` for stronger transcript or segment verification,
- `pyannote` for diarization or speaker-change support,
- `openSMILE` for interpretable acoustic features,
- `WavLM` or related self-supervised speech encoders for dense embeddings.

### Why this matters

This is where the project can still gain novelty without distorting the story.

If we add:
- reproducible acoustic features,
- speaker-aware segmentation,
- better segment quality control,

then the paper can make a stronger multimodal claim while still staying in a finance frame.

### What to avoid

Do not turn the project into a generic ASR or speech paper.
Audio should only be added where it supports a finance question:
- does delivery add incremental information,
- does `Q&A` delivery matter more than prepared remarks,
- does audio help under timing noise.

## 4. Time-series foundation models

### Relevant current projects

Open TSFM ecosystem now includes:
- `Chronos`
- `TimesFM`
- `MOMENT`
- `Moirai` and `Moirai-MoE`

These are real, active open projects, and several have official repositories or model releases.

### But are they the right main model here?

Usually no, not as an end-to-end replacement for our current pipeline.

Reason:
- these models are strongest on dense numeric forecasting tasks,
- our problem is an event-conditioned, mixed-modality market-reaction problem,
- and our key bottleneck is not raw numeric forecasting capacity but event representation and alignment quality.

### Best use of TSFMs in our project

They are most useful as:
- a stronger numeric-only baseline for the market side,
- a residual volatility forecaster to subtract generic market dynamics,
- or a downstream module once we have richer exogenous event features.

In other words:
- `Chronos` or `TimesFM` can help with the bar series,
- but they should not become the whole paper.

## 5. Mixed-type event sequence models

### Most conceptually relevant recent method

`Transformers for Mixed-type Event Sequences` is the closest recent conceptual match to the "ECC timed segments plus market reactions" problem.

It matters because our data is not a plain time series.
It is a mixed-type event stream:
- utterances,
- speaker roles,
- segment durations,
- market bars,
- possibly jumps or post-call windows.

### Why it is not the immediate next step

This class of model is attractive only if the upstream event representation is already reliable.

Right now:
- strict sequence features show some signal,
- but sequence labeling is not yet stable enough.

So this should be treated as a future architecture target, not the next immediate implementation.

## 6. Multimodal fusion models

### What is relevant

Recent multimodal model directions such as sparse mixture architectures or unified multimodal transformers are useful conceptually.

But for our current dataset and timeline, the main lesson is simpler:
- use modular fusion,
- keep text, audio, and timing separately inspectable,
- only move to more complex fusion after each modality proves incremental value.

### Practical implication

The next model step should be:
- event-level nonlinear fusion,
- then strict-sequence fusion,
- then a lightweight recurrent or temporal model.

Not:
- jump straight into a huge multimodal foundation model.

## Open Projects and Modules Worth Considering

## High priority now

### `openSMILE`

Why:
- interpretable acoustic features,
- easy to justify in a finance paper,
- low conceptual risk.

Use:
- eGeMAPS,
- energy,
- speaking-rate,
- pause and voicing proxies.

### `Whisper`

Why:
- open and well-supported,
- useful for segment verification and optional re-alignment,
- useful if we later need stronger transcript confidence around noisy `A4`.

Use:
- selective transcript verification,
- mismatch auditing,
- optional segment confidence features.

### `pyannote`

Why:
- speaker diarization and speaker-change support,
- useful if we want cleaner management-only or analyst-only audio signals.

Use:
- management versus analyst voice separation,
- better Q&A transition detection.

### `WavLM`

Why:
- strong general self-supervised speech representation,
- much better dense audio representation than file-size proxies.

Use:
- segment embeddings,
- pooled answer-level or call-level audio representations,
- audio branch in event-level fusion.

## Medium priority

### `FinGPT`

Why:
- useful as a financial weak-label or extraction engine.

Use:
- prompt-based subjectivity or evasion labeling,
- explanation and feature auditing.

Do not use as:
- the main forecasting model.

### `Chronos` / `TimesFM` / `MOMENT` / `Moirai`

Why:
- useful numeric baselines and exogenous residual models.

Use:
- residual volatility prediction,
- stronger market-side numeric-only baselines,
- possibly covariate-aware numeric branches.

Do not use as:
- the main multimodal event model without a strong event representation layer.

## Lower priority now, higher priority later

### `FlexTPP`-style mixed-type event transformers

Why:
- conceptually strong fit.

Use later:
- once segment-level labels and event representation are stronger.

### `FinCall-Surprise` transfer

Why:
- open multimodal ECC benchmark.

Use later:
- external pretraining or transfer experiments,
- or external validation if task transfer is reasonable.

## Recommended Improvement Roadmap

## Highest-ROI changes

### 1. Add identity-aware baselines and integrity checks

This is the highest-urgency integrity gap.

Recommended first version:
- same-ticker historical post-earnings volatility baseline,
- global historical post-earnings volatility baseline,
- firm fixed-effect or ticker-embedding sanity checks,
- subgroup reporting by ticker frequency and event count.

Expected payoff:
- much stronger credibility,
- direct protection against the main criticism in the latest ACL 2025 ECC volatility paper,
- cleaner interpretation of whether text, audio, and `Q&A` add event-specific information.

### 2. Add real audio features

This is the biggest remaining gap.

Recommended first version:
- `openSMILE` eGeMAPS,
- duration and pause features,
- pooled `WavLM` embeddings if environment permits.

Expected payoff:
- stronger multimodal claim,
- better novelty,
- better connection to the classic verbal-vocal literature.

### 3. Add `Q&A` weak labels from recent benchmarks

Recommended labels:
- subjectivity dimensions from `SubjECTive-QA`,
- evasion or directness from `EvasionBench`,
- answer coverage or specificity score.

Expected payoff:
- stronger finance-specific text features,
- better story than raw lexical features,
- clearer novelty.

### 4. Add a stronger market-side baseline

Best next step:
- residual or market-adjusted post-call volatility if a benchmark series is available,
- otherwise a stronger numeric bar forecast baseline using an open TSFM.

Expected payoff:
- more defensible "incremental value of ECC information" claim.

### 5. Tighten temporal evaluation

Needed:
- rolling-origin validation,
- confidence intervals or bootstrap for key metrics,
- subgroup analysis by call timing and firm,
- identity-confounding checks,
- same-ticker prior baselines,
- and explicit value-added beyond historical volatility.

Expected payoff:
- much stronger integrity section.

## Worth doing after that

### 6. Add reproducible audio-text late fusion

Once audio features exist, do:
- event-level text branch,
- event-level audio branch,
- simple learned or gated fusion.

### 7. Revisit strict sequence only after better labels

The sequence path is not dead.
But it should wait for:
- better role labels,
- better management-versus-analyst tagging,
- or stronger segment embeddings.

## What should not be prioritized now

- end-to-end huge multimodal foundation models,
- broad sequence modeling on noisy labels,
- sentence-level full alignment as the main path,
- generic LLM prompting as the main predictor,
- TSFMs as a total replacement for the ECC representation layer.

## Recommended Mainline From Here

For the next serious phase of the project, the safest and strongest path is:

1. Keep the current event-level finance-aware baseline as the mainline.
2. Add identity-aware historical-volatility and same-ticker baselines.
3. Add real audio features.
4. Add `Q&A` weak labels using recent benchmark taxonomies.
5. Add stronger numeric-only baselines or residualized targets.
6. Keep strict sequence modeling as an extension once labels improve.

That route best balances:
- novelty,
- integrity,
- feasibility,
- and fit with the actual data we have.

## Sources

- FinCall-Surprise: https://arxiv.org/abs/2510.03965
- SubjECTive-QA: https://arxiv.org/abs/2410.20651
- EvasionBench: https://arxiv.org/abs/2601.09142
- ConEC: https://aclanthology.org/2024.lrec-main.328/
- Same Company, Same Signal: https://aclanthology.org/2025.findings-acl.946/
- What Is the Impact of Managerial Non-Responses in Earnings Calls?: https://arxiv.org/abs/2505.18419
- The Sound of Risk: https://arxiv.org/abs/2508.18653
- AMA-LSTM: https://aclanthology.org/2024.naacl-industry.32/
- SH-Mix: https://aclanthology.org/2024.lrec-main.1244/
- DialogueGAT: https://aclanthology.org/2022.findings-emnlp.117/
- What You Say and How You Say It Matters: https://aclanthology.org/P19-1038/
- Transformers for Mixed-type Event Sequences: https://openreview.net/forum?id=MtwsRjPZhf
- Chronos: https://github.com/amazon-science/chronos-forecasting
- MOMENT: https://github.com/moment-timeseries-foundation-model/moment
- Moirai / Uni2TS: https://github.com/SalesforceAIResearch/uni2ts
- TimesFM: https://research.google/blog/a-decoder-only-foundation-model-for-time-series-forecasting/
- FinGPT: https://github.com/AI4Finance-Foundation/FinGPT
- Whisper: https://github.com/openai/whisper
- pyannote speaker diarization: https://huggingface.co/pyannote/speaker-diarization
- openSMILE docs: https://audeering.github.io/opensmile/
- openSMILE repo: https://github.com/audeering/opensmile
- WavLM: https://arxiv.org/abs/2110.13900
