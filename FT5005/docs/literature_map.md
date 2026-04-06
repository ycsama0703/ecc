# Literature Map

## Current Positioning

The project is now positioned at the intersection of:
- multimodal financial NLP
- earnings conference call analysis
- high-frequency market reaction modeling
- weak or noisy sequence alignment

The earlier business-news intermediary framing is no longer the primary literature anchor.

## Core References From The Professor's Draft

### Multimodal ECC prediction

1. Li, Yang, Smyth, and Dong (2020).
   Title: MAEC: A Multimodal Aligned Earnings Conference Call Dataset for Financial Risk Prediction.
   Why it matters: establishes a benchmark dataset and validates multimodal ECC as a real research area.

2. Qin and Yang (2019).
   Title: What You Say and How You Say It Matters: Predicting Stock Volatility Using Verbal and Vocal Cues.
   Why it matters: direct evidence that text and vocal delivery jointly predict volatility.

3. Ye, Yang, Li, and Zong (2020).
   Title: Financial Risk Prediction with Multi-Round Q&A Attention Network.
   Why it matters: motivates giving special attention to the Q&A rather than treating the whole call uniformly.

4. Sawhney, Agarwal, Wadhwa, and Shah (2020).
   Title: VolTAGE: Volatility Forecasting via Text-Audio Fusion with Graph Convolution Networks for Earnings Calls.
   Why it matters: shows that text-audio fusion is already a credible finance-ML path, but mostly at the call level rather than via explicit within-call timing.

### Vocal delivery in finance

5. Mayew and Venkatachalam (2012).
   Title: The Power of Voice: Managerial Affective States and Future Firm Performance.
   Why it matters: classic motivation for why audio is not merely decorative information.

6. Baik, Choi, and Kim (2025).
   Title: Vocal Delivery Quality in Earnings Conference Calls.
   Why it matters: very recent accounting evidence that vocal delivery contains incremental information.

### Unaligned or weakly aligned multimodal modeling

7. Tsai et al. (2019).
   Title: Multimodal Transformer for Unaligned Multimodal Language Sequences.
   Why it matters: foundational model family for asynchronous multimodal sequence fusion.

8. Chang et al. (2019).
   Title: D3TW: Discriminative Differentiable Dynamic Time Warping for Weakly Supervised Action Alignment and Segmentation.
   Why it matters: the closest structural precedent for latent alignment when utterance clock times are not observed.

9. Cuturi (2013).
   Title: Sinkhorn Distances: Lightspeed Computation of Optimal Transport.
   Why it matters: algorithmic basis for soft optimal-transport alignment.

10. Cuturi and Blondel (2017).
    Title: Soft-DTW: A Differentiable Loss Function for Time-Series.
    Why it matters: core method for stretch-goal latent timestamp learning.

### Finance and volatility targets

11. Kogan et al. (2009).
    Title: Predicting Risk from Financial Reports with Regression.
    Why it matters: text can predict volatility, which legitimises the target even before multimodal extensions.

12. Easley, Kiefer, O'Hara, and Paperman (1996).
    Title: Liquidity, Information, and Infrequently Traded Stocks.
    Why it matters: supports using trading volume as part of the information-arrival proxy.

## What Is Novel In Our Current Plan

The novelty is no longer "news selects important ECC content." The novelty is now:

1. using noisy timed sentence segments from a real DJ30 multimodal ECC package;
2. predicting high-frequency within-call and post-call volatility, not only one post-event scalar;
3. testing whether audio adds value once the utterance order and timing are respected;
4. using the timestamp-aligned pilot as the bridge to a future weak-alignment setting for larger restricted samples.

## Updated Literature Gap

Existing ECC papers generally fall into three buckets:
- call-level text models
- call-level text-plus-audio models
- Q&A-aware but not fully high-frequency sequence-aware models

Our strongest gap statement is:
- there is still limited evidence on whether timed-sentence multimodal ECC sequences improve high-frequency reaction modeling relative to bag-of-utterances or coarse-section baselines, especially when the timing labels themselves are noisy.

## What We Should Not Overclaim

- We should not claim minute-perfect causal attribution from specific utterances to specific price moves.
- We should not claim a fully unsupervised alignment contribution if we rely on `A4` timestamps for the pilot.
- We should not claim microstructure-level spread effects because the current `D` table does not include quotes.

## Recommended Paper Structure

1. Motivation: multimodal ECCs and high-frequency reactions
2. Data: DJ30 text, audio, timestamps, analyst data, 5-minute bars
3. Baselines: controls, bag-of-utterances, section-level
4. Main model: timestamp-aware multimodal sequence model on filtered noisy timed segments
5. Extension: weak alignment when timestamps are hidden
6. Economic interpretation: Q&A and post-call volatility
