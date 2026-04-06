# Data QC Protocol

## Purpose

This protocol defines the first-pass quality control required before any modeling on the DJ30 multimodal ECC package.

The latest reply from Kelvin and Yu Ta confirms three issues:
- `A2` uses scheduled rather than actual time
- HTML completeness must be inferred from file size or text length
- `A4` may contain incorrect timestamps and should be filtered using `match_score` or manual comparison

So QC is not optional. It is part of the method.

## QC Outputs

Every event should end with:
- HTML integrity flag
- A4 row-level quality flag
- A4 event-level quality summary
- strict or broad subset membership

## Step 1: HTML Integrity Check For A2

Goal:
- detect failed or partial scrapes

Signals to compute per file:
- raw file size in bytes
- visible text length after stripping HTML tags
- number of visible paragraphs
- number of participant lines in the header

Recommended detection logic:
- compare file size and visible text length within the same firm across quarters
- flag files with strong downward outliers
- also flag files with abnormally low paragraph count relative to nearby calls

Practical rule:
- do not hard-delete flagged files immediately
- assign `html_integrity_flag = fail`, `warn`, or `pass`

Why:
- some downstream work can still rely on `A1` even if `A2` is incomplete

## Step 2: Structural Sanity Check For A4

Goal:
- remove obviously broken timed segments

Hard failures:
- missing `start_sec`
- missing `end_sec`
- `end_sec <= start_sec`
- negative times
- non-monotonic segment order within the same event

Rows with any hard failure should be dropped.

## Step 3: Text-Match Quality Check For A4

Goal:
- keep only rows where the timed segment likely matches the official text

Available fields:
- `official_text`
- `asr_matched_text`
- `match_score`
- `block_avg_score`
- `overall_TFIDF`

Required computed signals:
- normalized text similarity between `official_text` and `asr_matched_text`
- token overlap ratio

Recommended starting subsets:

Strict subset:
- no hard failure
- non-empty `official_text`
- `match_score` present and high
- `block_avg_score` present and high
- normalized text similarity high

Broad subset:
- no hard failure
- at least acceptable `match_score` or acceptable normalized text similarity

Important note:
- the exact threshold should be calibrated after plotting the empirical distribution on the first linked sample
- do not pretend the current thresholds are ground truth

Suggested first audit heuristic:
- inspect the first 50 linked events
- then choose one strict threshold set and one broad threshold set
- keep both subsets in the experiments

## Step 4: Event-Level A4 Quality Summary

Goal:
- decide whether an event is usable as a whole

Compute per event:
- total A4 rows
- number and share of rows that pass the strict filter
- number and share of rows that pass the broad filter
- median and minimum `match_score`
- total timed coverage in seconds

Recommended event inclusion logic:

Strict event subset:
- enough timed rows survive strict filtering
- timed coverage remains plausible for a real call

Broad event subset:
- enough timed rows survive broad filtering
- event still retains meaningful sequence information

Events failing both should be excluded from the main models.

## Step 5: A1-To-A4 Mapping QC

Goal:
- propagate role and component-type labels from `A1` onto timed sentence rows in `A4`

Checks:
- preserve transcript order
- avoid many-to-many explosions
- inspect a sample of mapped Q&A sections manually

This mapping should also receive a confidence flag:
- `high`
- `medium`
- `low`

Low-confidence mappings should not drive the main Q&A analyses.

## Step 6: Event-Time Anchoring QC

Goal:
- connect call-relative timing to market-clock timing without overclaiming precision

Current assumption:
- `A2` start time is scheduled, not actual

Protocol:
- anchor the event to scheduled time for the first pass
- keep an event-level `start_time_uncertain = true` flag
- run robustness checks with small start-time jitters later

This should be stated explicitly in the paper.

## Step 7: Final Modeling Subsets

Define at least three subsets:

1. Core strict subset
- clean HTML or acceptable fallback
- strong A4 quality
- complete `A3`, `A4`, `C1`, and `D`

2. Broad research subset
- relaxed A4 quality
- still complete enough for sequence modeling

3. Text-only fallback subset
- if audio exists but aligned timing quality is too weak, retain for bag-of-utterances or section-level baselines only

## Required QC Deliverables

Before any main model is reported, create:
- one QC summary table
- one QC funnel chart
- one histogram of `match_score`
- one manual audit sheet for a small sample of events

## Non-Negotiable Rule

Main headline results should be reported on the strict subset first.

Broad-subset results are useful, but only as supplementary evidence that the findings are not driven by over-filtering.
