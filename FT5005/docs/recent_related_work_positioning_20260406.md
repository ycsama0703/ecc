# Recent Related-Work Positioning for I-POG-QA (2026-04-06)

## Purpose

This note narrows the recent-literature comparison set to the 2024-2026 window and asks one practical question:

- where is the cleanest remaining innovation slot for our paper, given the current repository evidence?

Short answer:

- not another broad `LLM summary + multimodal fusion` system
- not a whole-transcript graph or heavy sequence headline
- not an audio-first paper
- yes to a strong-prior, noisy-timing, selectively trusted `Q&A`-increment model

That is why the main method should be `I-POG-QA`.

## Closest recent papers and what they imply

### 1. Identity and prior baselines are now mandatory

- [Same Company, Same Signal: The Role of Identity in Earnings Call Transcripts](https://aclanthology.org/2025.findings-acl.946/) (Yu et al., Findings ACL 2025)

What it contributes:

- shows transcript models can be dominated by ticker identity rather than event-specific financial signal
- introduces strong historical-volatility baselines such as `PEV` and `STPEV`
- argues that prior volatility information can outperform transcript-based models

What this means for us:

- prior-aware evaluation is not optional
- our method must be written as an incremental correction beyond a strong prior, not as raw transcript prediction

### 2. The broad LLM-fusion lane is already occupied

- [ECC Analyzer: Extract Trading Signal from Earnings Conference Calls using Large Language Model for Stock Performance Prediction](https://arxiv.org/abs/2404.18470) (Cao et al., 2024; later presented in the ICAIF 2024 ecosystem)
- [RiskLabs: Predicting Financial Risk Using Large Language Model Based on Multi-Sources Data](https://arxiv.org/abs/2404.07452) (Cao et al., 2024)

What they contribute:

- use LLM-driven extraction and multimodal fusion
- combine ECC text, audio, and surrounding market information
- position themselves as richer unstructured-data financial-risk systems

What this means for us:

- if we simply build another `LLM + text + audio + market` fusion framework, the novelty will be weak
- our contribution needs to be narrower and more defensible:
  - strong prior backbone
  - noisy observability handling
  - selective `Q&A` correction rather than full-transcript fusion

### 3. `Q&A` behavior itself is a real signal family

- [Capturing Analysts' Questioning Strategies in Earnings Calls via a Question Cornering Score (QCS)](https://aclanthology.org/2024.finnlp-2.10/) (D'Agostino et al., FinNLP 2024)

What it contributes:

- formalizes analysts' questioning strategy as a measurable object
- supports the idea that how analysts press and how management responds are not reducible to pooled transcript sentiment

What this means for us:

- separating `what is discussed` from `how it is answered` is methodologically justified
- our `semantic expert` versus `quality/accountability expert` split is not arbitrary

### 4. Real-world earnings-call context is noisy

- [ConEC: Earnings Call Dataset with Real-world Contexts for Benchmarking Contextual Speech Recognition](https://aclanthology.org/2024.lrec-main.328/) (Huang et al., LREC-COLING 2024)

What it contributes:

- shows that real earnings-call contextual signals are noisier than synthetic or idealized settings
- treats earnings calls as a hard real-world speech/context benchmark

What this means for us:

- using `A4` as noisy observability rather than as clean ground-truth timing is well aligned with the recent evidence
- the trust mechanism in our model should be framed as reliability-aware, not as perfect alignment recovery

### 5. Whole-transcript conversational graph modeling is no longer an open lane

- [Learning from Earnings Calls: Graph-Based Conversational Modeling for Financial Prediction](https://pubsonline.informs.org/doi/10.1287/isre.2023.0519) (Yang et al., Information Systems Research, published online January 16, 2026; accepted November 2, 2025)

What it contributes:

- models earnings-call transcripts as conversational graphs
- uses graph neural networks to capture topics, relations, and cross-references for risk prediction

What this means for us:

- a new paper centered on whole-transcript graph architecture would now face a direct novelty collision
- our cleaner lane is not to out-architect graph models on transcript length
- our cleaner lane is to model a narrow, event-specific, prior-aware `Q&A` increment

### 6. Audio-first volatility work is becoming more crowded

- [The Sound of Risk: A Multimodal Physics-Informed Acoustic Model for Forecasting Market Volatility and Enhancing Market Interpretability](https://arxiv.org/abs/2508.18653) (Chen et al., arXiv 2025)

What it contributes:

- pushes an audio-heavy multimodal volatility story with a strong interpretability angle

What this means for us:

- audio can remain an extension or robustness branch
- it should not be the headline method contribution for the current paper

## Older but still relevant roots

These are not part of the 2024-2026 novelty window, but they remain important foundations:

- Qin and Yang (2019): audio-textual co-attention for earnings calls
- Financial Risk Prediction with Multi-Round Q&A Attention Network (IJCAI 2020)

These works still justify keeping the `Q&A` unit central, but they do not define the current novelty frontier by themselves.

## The main gap that still looks open

The most defensible remaining gap is this:

- existing recent work either
  - learns broad multimodal fusion,
  - models whole transcripts,
  - or studies `Q&A` / audio / context as stand-alone signal families;
- but the current literature still leaves room for a method that explicitly asks:
  - under strong same-ticker priors,
  - under noisy call timing and observability,
  - when does a compact `Q&A` correction deserve to change the forecast at all?

This is the exact slot that `I-POG-QA` occupies.

## Why I-POG-QA is differentiated

`I-POG-QA` is different from the recent neighboring work in four ways.

### 1. It is explicitly prior-aware

The method begins from the same-ticker expanding prior and structured base shell.

That directly responds to Yu et al. (2025), where historical volatility can dominate transcript models.

### 2. It uses `A4` as trust information, not as clean timestamp truth

The model does not assume perfect alignment.

Instead:

- `A4` quality and answerability features inform whether the dialog correction should be trusted

### 3. It treats `Q&A` semantics and `Q&A` accountability as different experts

This is narrower and more disciplined than flat transcript pooling.

It also aligns with the `QCS`-style reading that questioning and answering behavior matter as a separate signal family.

### 4. It regularizes the dialog effect to stay incremental

The method penalizes covariance between the applied dialog effect and:

- the standardized prior reference
- the standardized base-residual reference

This is the clearest paper-level answer to the current identity-overfitting critique.

## Recommended contribution statement

If the paper keeps the current mainline, the contribution package should read approximately like this:

1. We define a prior-aware high-frequency ECC volatility-shock task under noisy timing and evaluate it with strong temporal holdout.
2. We show that the most credible ECC increment is narrow:
   - clean `after_hours`
   - strong market/base shell
   - `A4`
   - compact `Q&A`
3. We propose `I-POG-QA`, a prior-aware observability-gated `Q&A` residual model with:
   - monotone trust gating
   - semantic versus accountability experts
   - incrementality regularization
4. We show that the paper's method question is not "how to fuse everything," but "when a trusted `Q&A` increment should modify a strong prior forecast."

## Benchmark implications

To make the method story solid, the next experiment table should include:

1. strong prior baseline
2. structured prior-aware base
3. structured + semantic ridge
4. structured + quality ridge
5. structured + semantic + quality ridge
6. `I-POG-QA`

Recommended secondary comparisons:

- a literature-inspired `ECC Analyzer`-style lightweight baseline
- target-family comparison between `shock_minus_pre` and `within_minus_pre`

## Bottom line

The recent literature does not support another broad feature-expansion story.

It does support a much sharper paper:

- a strong-prior
- noisy-observability
- selectively trusted
- `Q&A`-centered incremental model

That is the rationale for making `I-POG-QA` the absolute main method line.
