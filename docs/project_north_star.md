# Project North Star

## 1. Core Research Question

This project answers one question:

> When should earnings conference call (ECC) information be trusted to correct a strong market prior under noisy timing and imperfect observability?

---

## 2. Core Idea (One Sentence)

We model ECC signals as noisy residual corrections and use proxy-based latent reliability to determine how much they should be trusted.

---

## 3. What This Paper IS

- A **supervised regression framework**
- A **statistical decision model under uncertainty**
- A **reliability-aware residual correction method**
- A method combining:
  - market prior
  - ECC residual signal
  - proxy-based reliability
  - minimax shrinkage + abstention

---

## 4. What This Paper is NOT

- NOT an alignment algorithm paper
- NOT a pure NLP feature extraction paper
- NOT a multimodal architecture paper
- NOT a trading strategy paper
- NOT a causal inference paper

---

## 5. Main Experimental Setup (DO NOT CHANGE)

- Sample: clean after-hours
- Target: shock_minus_pre
- Split: time-based

---

## 6. Core Model Structure

y = μ + r  
z ≈ r  
ŷ = μ̂ + α * z  

Where:
- μ = market prior
- z = ECC residual
- α = reliability-controlled shrinkage

---

## 7. Core Innovation

NOT:
- better features
- better encoding

BUT:

- ECC signals are noisy
- reliability is latent
- proxies are noisy
- decision must be robust

---

## 8. What Must Be Proven in Experiments

1. ECC adds incremental value beyond market baseline
2. Using ECC naively is unstable
3. Reliability-aware correction improves performance
4. Minimax + abstention improves robustness

---

## 9. Hard Constraints

- DO NOT redefine target
- DO NOT change main sample
- DO NOT mix benchmark and ablation
- DO NOT introduce new tasks without justification

---

## 10. Output Goal

A clean paper showing:

> ECC should be used conditionally, not universally