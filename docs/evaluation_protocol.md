# Evaluation Protocol

## 1. Metrics

### Primary Metrics

- MSE
- MAE
- R² (Out-of-sample)
- Spearman Correlation

---

### Selective Prediction Metrics (if used)

- Coverage
- Accepted-set MSE
- Risk-Coverage Curve
- AURC

---

## 2. Calculation Rules

### R²

R² = 1 - MSE_model / MSE_baseline

Baseline = mean prediction or market baseline

---

### Spearman

Rank correlation between:
- predicted y
- true y

---

## 3. Aggregation

- Compute metrics on TEST set only
- Report average over seeds if applicable

---

## 4. Visualization

Must include:

- Main result table
- Risk-Coverage curve (if abstention used)

---

## 5. Statistical Stability

Recommended:

- bootstrap confidence intervals
- permutation test for proxy importance

---

## 6. DO NOT

- report train metrics as results
- mix validation results into final table