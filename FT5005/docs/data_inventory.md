# Data Inventory

## Purpose

This document summarises the actual DJ30 package shared by Kelvin and Yu Ta and how each table should be used in the current multimodal ECC project.

## Shared Package Overview

The Google Drive folder contains the following groups:
- `A1.ECC_Text_Json_DJ30`
- `A2.ECC_Text_html_DJ30`
- `A3.ECC_Audio_DJ30`
- `A4.ECC_Timestamp_DJ30`
- `B.Compustat`
- `C.Analyst`
- `D.Stock_5min_DJ30`
- `[Readme]`

The DJ30 ticker list in the readme includes 30 firms such as `AAPL`, `JPM`, `MSFT`, `NVDA`, and `WMT`.

## Table-By-Table Notes

### A1.ECC_Text_Json_DJ30

Source:
- Capital IQ

Format:
- JSON

Coverage:
- 2005 to 2025

Observed sample schema:
- top-level keys include `transcriptid`, `keydevid`, `companyid`, `companyname`, `headline`, `transcriptcreationdate`, `mostimportantdate`, and `components`
- each item in `components` includes:
  - `componentid`
  - `componenttypename`
  - `text`
  - `componentorder`
  - `personname`
  - `companyofperson`

Why it matters:
- best source for structured utterance type
- clean `Question`, `Answer`, and `Operator` markers
- likely the best base for text-side segmentation

Main caveat:
- fiscal-quarter alignment is imperfect if done by title only

### A2.ECC_Text_html_DJ30

Source:
- Seeking Alpha

Format:
- HTML

Coverage:
- 2018 Q3 to 2026 Q1, with uneven partials at the edges

Observed sample schema:
- header contains firm, quarter, call date, and scheduled start time
- also contains participant lists and the readable transcript body

Why it matters:
- human-readable transcript
- scheduled start time in the header
- participant lists useful for speaker-role checks

Main caveat:
- around 10 percent of files are incomplete
- scheduled time is not always actual time
- latest reply confirms that the visible time should be treated as scheduled time, not actual time
- integrity must be checked by file size or HTML text length because no ready-made integrity flag is provided

### A3.ECC_Audio_DJ30

Source:
- Seeking Alpha

Format:
- MP3

Coverage:
- 2018 Q4 onward with strong coverage from 2019 to 2024

Why it matters:
- required for audio embeddings
- required for utterance-level acoustic analysis
- combined with `A4`, enables exact segmentation inside the audio file

Current constraint:
- only a small external set of sentence-level audio embeddings and acoustic features exists
- the main research pipeline should generate its own features directly from `A3`

### A4.ECC_Timestamp_DJ30

Source:
- alignment of `A1` and `A2` using the provider's `Align1`

Format:
- CSV

Coverage:
- 2018 Q4 to 2025 Q3, partial at edges

Observed sample schema:
- `sentence_id`
- `official_text`
- `asr_matched_text`
- `start_sec`
- `end_sec`
- `match_score`
- `block_avg_score`
- `overall_TFIDF`

Why it matters:
- this is the key bridge between text and audio
- it gives sentence-level timing inside the call
- it enables audio cutting, duration features, and timestamp-aware sequence models

Main caveat:
- timestamps are relative to the audio file, not guaranteed exact wall-clock market time
- the unit is closer to sentence-level alignment than to clean vendor utterance blocks
- `A1` component rows should not be assumed to line up one-to-one with `A4`
- the provider has explicitly warned that `A4` may contain quality inconsistencies
- `match_score` and manual comparison should be used to filter incorrect timestamps

### B1.Compustat Execucomp Annual Compensation

Use:
- optional executive metadata and titles
- not required for the first pilot

### B2.Compustat Fundamentals Quarterly

Use:
- optional accounting controls
- may help if the paper adds firm-level controls beyond event-only signals

### B3.Compustat Index Constituents

Use:
- cross-table identifiers such as `gvkey`, `tic`, and `companyid`
- essential linking bridge

### C1.Surprise_DJ30

Source:
- S&P Capital IQ Pro

Coverage:
- 2005 to 2025

Observed sample schema:
- actual and estimate fields for metrics such as EBITDA, EPS, and revenue
- `Earnings Announce Date`
- identifiers including ticker and entity fields

Why it matters:
- event timestamp anchor
- earnings surprise control
- call-level baseline predictors

### C2.AnalystForecast_DJ30

Source:
- S&P Capital IQ Pro

Coverage:
- 2005 to 2025

Observed sample schema:
- mean, median, high, low, standard deviation, and number of estimates
- metrics include normalised EPS, net income, and revenue

Why it matters:
- analyst coverage
- forecast dispersion
- another event-level control table

### D.Stock_5min_DJ30

Format:
- per-ticker CSV files

Coverage:
- 2000 Jan to 2025 May

Observed sample schema:
- `DateTime`
- `Open`
- `High`
- `Low`
- `Close`
- `Volume`
- `Ticker`

Why it matters:
- primary target construction
- within-call and post-call volatility
- volume response around the call

Main caveat:
- no bid, ask, spread, or adjustment fields listed
- no benchmark ETF file has been shared so far
- the shared folder currently contains only 25 ticker files rather than the full 30-name DJ30 list
- missing tickers in `D` are `CRM`, `CVX`, `PG`, `TRV`, and `V`
- observed sample days confirm that the bars include extended-hours timestamps, not only regular trading hours

## Practical Data Strategy

### Primary pilot inputs

For the first end-to-end model, only use:
- `A1`
- `A2`
- `A3`
- `A4`
- `C1`
- `C2`
- `D`
- `B3`

This keeps the pipeline focused and avoids unnecessary accounting baggage.

### Event linking logic

Recommended key path:
- `A1.companyid` or ticker
- `B3` for ID harmonisation
- `C1` for earnings announcement date and quarter
- filename plus quarter metadata for `A2`, `A3`, and `A4`
- ticker and calendar time for `D`
- fuzzy text overlap between `A1.components.text` and `A4.official_text` when role labels must be propagated to timed sentence segments

### Suggested event definition

One event should contain:
- firm and fiscal quarter
- scheduled call start time from `A2`
- earnings announcement timestamp from `C1`
- transcript utterances from `A1`
- audio file from `A3`
- utterance timing from `A4`
- 5-minute bars from `D`

## Immediate Data Gaps To Clarify

1. Is there a benchmark intraday series such as `SPY` or `DIA` for market-adjusted returns?
2. Is there a missing-file list for `A3` and `A4`?
3. For larger samples, can raw text and audio be used onsite while only embeddings are taken home?

## Current Working Assumptions After Latest Reply

1. Proceed without benchmark intraday ETF data unless it is later shared.
2. Treat `A2` time as scheduled time and model the resulting uncertainty explicitly.
3. Generate audio features in-house from `A3`.
4. Treat `A4` as noisy alignment supervision and filter aggressively before training.
5. Treat `D.Stock_5min_DJ30` as extended-hours intraday data with incomplete ticker coverage.
