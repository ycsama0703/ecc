# Benchmark Specification

## Goal

Compare our method against EXISTING methods.

NOT internal variants.

---

## Benchmark Categories

### 1. Market Baselines

- Market Only
- Market + Controls

Purpose:
Strong real-world baseline.

---

### 2. Text Baselines

- TF-IDF + Elastic Net
- FinBERT pooled

Purpose:
Standard NLP baselines.

---

### 3. ECC Structural Models

- MR-QA

Purpose:
Structured ECC modeling.

---

### 4. Multimodal Models

- Text + Audio Fusion (MDRM-style)

Purpose:
Standard multimodal approach.

---

## Model Input Rules

| Model | Market | ECC | Proxy |
|------|--------|-----|-------|
| Market | ✓ | ✗ | ✗ |
| FinBERT | ✓ | ✓ | ✗ |
| MR-QA | ✓ | ✓ | ✗ |
| Fusion | ✓ | ✓ | ✗ |
| Ours | ✓ | ✓ | ✓ |

---

## Important Constraints

- All models use SAME train/test split
- No leakage allowed
- Same evaluation metrics
- No hyperparameter unfair advantage

---

## Main Table Must Include

- MSE
- MAE
- R²
- Spearman

Optional:
- Risk-coverage (for ours)

---

## DO NOT

- Include ablation models here
- Mix variants into benchmark table