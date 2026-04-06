# SOTA and Novelty Flexible Roadmap 2026-03-14

## Why We Need A Flexible Roadmap Now

The current post-migration reruns have clarified an important fact:
- our strongest current `shock_minus_pre` performance is primarily driven by market-side features,
- ECC text and timing do not yet show stable incremental gains beyond the strongest market-only residual benchmark,
- current audio features do not rescue that gap.

The newer `qav2` checkpoint sharpens that reading rather than overturning it:
- richer heuristic `Q&A` features do recover positive ECC-only signal on the all-HTML sample,
- but those gains weaken sharply on the clean sample,
- and they still do not beat the strongest market-only or market-plus-controls benchmarks.

That does not kill the project.

It means the project now needs a disciplined branching strategy:
- keep the current path only if we can recover genuine ECC incremental value,
- otherwise pivot quickly to a stronger dataset or method direction without losing novelty.

## Non-Negotiable Integrity Gates

Any future main-path paper idea should satisfy all of these:

1. Beat or clearly complement strong market-only and same-ticker priors.
2. Survive clean-sample sensitivity checks.
3. Avoid generic "multimodal finance" framing.
4. Make the contribution explicit:
   - new benchmark,
   - new data linkage,
   - or new incremental-signal finding under strong controls.

## Best Current Path: Q&A Quality and Evasion Under Strong Market Controls

Most promising near-term idea:
- predict off-hours post-call volatility shock,
- but frame the real scientific question as whether `Q&A` answer quality, subjectivity, or evasiveness adds value beyond market-only and same-ticker priors.

Why this is still promising:
- `SubjECTive-QA` introduces six-dimensional subjectivity labels for earnings-call `Q&A` and is publicly available under CC BY 4.0.
  Source: https://arxiv.org/abs/2410.20651
- `EvasionBench` introduces a large-scale benchmark for managerial evasion in earnings-call `Q&A`.
  Source: https://arxiv.org/abs/2601.09142
- `Same Company, Same Signal` shows why identity and prior volatility must be treated as first-class baselines rather than afterthoughts.
  Source: https://aclanthology.org/2025.findings-acl.946/

What would make this path novel:
- not just building another transcript model,
- but showing whether benchmark-transferred `Q&A` quality or evasion signals have incremental value for high-frequency post-call shock after controlling for:
  - same-ticker priors,
  - pre-call market state,
  - within-call market behavior.

Concrete next upgrades on this path:
- replace current heuristic `qa_benchmark` features with transferred predictions from `SubjECTive-QA` and `EvasionBench`
- model question-answer pairs directly instead of only event-level aggregates
- test whether gains concentrate in:
  - off-hours calls,
  - high-surprise events,
  - low-coverage firms,
  - or unusually adversarial `Q&A`

Why this now matters even more:
- the current heuristic `qav2` refresh already shows that better `Q&A` semantics can revive ECC-only signal,
- so the next credible jump is not a larger generic model,
- it is cleaner supervision and pair-level `Q&A` reasoning under the same benchmark ladder.

## Best Pivot If ECC Incremental Value Still Fails: Public Reproducible Event Benchmark

If the current DJ30 path still cannot beat the market-only residual benchmark after better `Q&A` modeling, the strongest pivot is:
- build a public, reproducible benchmark around earnings-event information flow,
- rather than keeping the paper centered on proprietary transcript signal alone.

Most usable public-source building blocks:

### SEC EDGAR

Use:
- `8-K` Item `2.02` earnings-release events
- submissions history
- XBRL company facts

Official source:
- SEC EDGAR APIs and bulk archives
- https://www.sec.gov/search-filings/edgar-application-programming-interfaces

Why it matters:
- gives scalable, official, timestamped earnings-event and fundamentals infrastructure
- supports a reproducible public benchmark around:
  - press release timing,
  - earnings surprise context,
  - filing-linked metadata

### Public or low-friction intraday bars

Use:
- 5-minute OHLCV around the event window

Official provider documentation example:
- Alpha Vantage intraday API
- https://www.alphavantage.co/documentation/

Why it matters:
- enables a public prototype if professor-side benchmark bars or larger proprietary intraday data do not arrive

### Public earnings-call corpora

`ConEC`
- public-domain earnings calls with contextual materials such as slides, news release, and participant lists
- Source: https://aclanthology.org/2024.lrec-main.328/

`FinCall-Surprise`
- large-scale open-source multimodal earnings-call benchmark with transcripts, full audio, and slides
- Source: https://arxiv.org/abs/2510.03965

Why this pivot can still be novel:
- a reproducible public benchmark that isolates:
  - press release information,
  - `Q&A` incremental information,
  - and post-call market shock,
- would be more credible and reusable than a small opaque proprietary setup

## Audio Path: Only Worth It If We Go Alignment-Aware

The current coarse audio proxy path is not enough.

If we keep audio as a serious pillar, the upgraded version should be:
- utterance-aligned,
- `Q&A`-aware,
- and evaluated against market-only baselines.

Why this is still a legitimate SOTA-adjacent path:
- `The Sound of Risk` shows that richer acoustic dynamics in earnings calls can matter for volatility-related outcomes.
  Source: https://arxiv.org/abs/2508.18653
- `ConEC` provides a public benchmark for contextual ASR on earnings calls.
  Source: https://aclanthology.org/2024.lrec-main.328/
- `FinCall-Surprise` provides open-source multimodal call data with full audio.
  Source: https://arxiv.org/abs/2510.03965

What would make the audio path novel for us:
- not generic call-level text-audio fusion,
- but measuring acoustic state shifts from prepared remarks to spontaneous `Q&A` under noisy timing, while explicitly testing whether they add incremental value beyond market-only signal.

## Method Path: Sequence Models Only After Signal Isolation

If we upgrade the method stack, the next method jump should happen only after the signal-isolation problem is solved.

Two useful architectural inspirations:
- `Transformers for Mixed-type Event Sequences`
  - Source: https://openreview.net/forum?id=d554f5e4685637fd9f1025d5e1a7c498008755d6
- `Moirai-MoE`
  - Source: https://proceedings.mlr.press/v267/liu25an.html

Why these matter:
- our problem is naturally a mixed event sequence:
  - transcript turns,
  - speaker roles,
  - timing marks,
  - and market bars.
- but method novelty only becomes credible after we show there is non-trivial ECC-specific signal to model

So sequence modeling is an amplifier path, not a rescue path.

The latest hybrid-architecture checkpoint reinforces that rule:
- nonlinear market experts underperform the simpler market ridge baseline,
- validation-time stacking mostly collapses back onto the same full ridge expert,
- and regime-gated blending gets close but still does not beat the strongest market-only baseline.

So heavier architecture is not currently the missing ingredient.

## Benchmark and Data Ideas Worth Borrowing, Not Necessarily Following Directly

`MFinMeeting`
- multilingual, multi-sector, multi-task financial meeting understanding benchmark
- Source: https://aclanthology.org/2025.findings-acl.14/
- useful as inspiration for meeting-level task design and evaluation breadth

This is probably not our main path, but it is a good reminder that "financial meeting understanding" is itself becoming a benchmarked area, not just a one-off downstream task.

## What We Should Avoid

Avoid these paper stories unless the evidence changes materially:

- "multimodal ECC prediction" without strong market-only baselines
- "audio helps" without alignment-aware features and incremental tests
- "ECC semantics drive volatility" if the benchmark ladder still shows market features dominate
- a larger or fancier model before the data and evaluation story are clean

## Recommended Decision Rule

Use the following branching rule from now on:

### Stay on current DJ30 path if:
- transferred `Q&A` quality or evasion signals beat the market-only residual benchmark on at least one defensible slice,
- especially off-hours or high-surprise subsets,
- and the gain survives clean-sample checks

### Pivot to public-data benchmark path if:
- market-only remains dominant even after stronger `Q&A` transfer features,
- or professor-side restricted data access becomes the bottleneck

### Invest in audio only if:
- we can build `A4`-aligned utterance-level audio features
- and those features show incremental gains over the market-only benchmark

## Current Recommendation

As of now, the best next research move is:
- keep the current DJ30 path alive,
- but redefine the true target as proving incremental `Q&A` quality or evasion signal beyond market-only priors,
- while preparing a public-data benchmark pivot in parallel using SEC and open earnings-call resources.

That gives us the best chance to stay both:
- scientifically honest,
- and genuinely novel if the current main path stalls.

Operationally, the current checkpoint means:
- do not sell `qav2` as a finished answer,
- do treat it as evidence that the `Q&A` route is still alive,
- and only escalate to heavier multimodal or sequence methods after transferred pair-level `Q&A` signals have had one fair attempt against the benchmark ladder.
