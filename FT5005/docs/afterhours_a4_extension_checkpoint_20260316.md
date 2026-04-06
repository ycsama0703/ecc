# After-Hours A4 Extension Checkpoint 2026-03-16

## 1. Purpose

The previous feature-group checkpoint showed that the most credible incremental ECC value now lives in `clean after_hours`, especially through the coarse `A4` observability or alignment block.

The next question was:

- once `A4` is already helping, what kind of extension is actually useful on top of it?

This checkpoint tests two different directions:

- compact `Q&A` semantics via `qna_lsa`
- high-dimensional strict-bin sequence features derived from `A4`

## 2. Experimental Design

Locked setting:

- regime: `after_hours`
- target: `shock_minus_pre`
- formulation: same-ticker prior plus residual ridge
- slices:
  - all-HTML
  - clean `exclude html_integrity_flag=fail`

Compared models:

- `market_plus_controls`
- `market_controls_plus_a4`
- `market_controls_plus_a4_plus_qna_lsa`
- `market_controls_plus_a4_plus_role_sequence`
- `market_controls_plus_a4_plus_weak_sequence`
- `market_controls_plus_a4_plus_qna_lsa_plus_role_sequence`
- `market_controls_plus_a4_plus_qna_lsa_plus_weak_sequence`

New script:

- `scripts/run_afterhours_a4_extensions.py`

Result files:

- `results/afterhours_a4_extensions_real/afterhours_a4_extensions_summary.json`
- `results/afterhours_a4_extensions_clean_real/afterhours_a4_extensions_summary.json`

## 3. Main Findings

### 3.1 `Q&A` semantics are the right extension on top of `A4`

All-HTML `after_hours`:

- `market_plus_controls`: `R^2 ≈ 0.893`
- `market_controls_plus_a4`: `R^2 ≈ 0.892`
- `market_controls_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.927`

Clean `after_hours`:

- `market_plus_controls`: `R^2 ≈ 0.875`
- `market_controls_plus_a4`: `R^2 ≈ 0.909`
- `market_controls_plus_a4_plus_qna_lsa`: `R^2 ≈ 0.935`

Significance on the clean slice:

- `A4 -> A4 + qna_lsa`
  - permutation `mse p ≈ 0.037`
  - permutation `mae p ≈ 0.058`

Interpretation:

- on the refined `after_hours` task, compact `Q&A` semantics are a genuinely promising extension,
- especially once the model already includes the useful coarse `A4` observability block.

### 3.2 High-dimensional sequence bins are not the right next step

All-HTML `after_hours`:

- `A4 + role_sequence`: `R^2 ≈ 0.850`
- `A4 + weak_sequence`: `R^2 ≈ 0.856`

Clean `after_hours`:

- `A4 + role_sequence`: `R^2 ≈ 0.841`
- `A4 + weak_sequence`: `R^2 ≈ 0.866`

These are materially worse than `A4` alone and much worse than `A4 + qna_lsa`.

Clean significance:

- `A4 -> A4 + role_sequence`
  - permutation `mse p ≈ 0.0088`
  - permutation `mae p ≈ 0.009`
- `A4 -> A4 + weak_sequence`
  - permutation `mse p ≈ 0.0043`
  - permutation `mae p ≈ 0.001`

Interpretation:

- more detailed `A4` sequence bins are not adding useful generalisable signal here,
- they are currently acting like a high-dimensional overfitting path.

### 3.3 Sequence still hurts even after adding `Q&A`

All-HTML `after_hours`:

- `A4 + qna_lsa`: `R^2 ≈ 0.927`
- `A4 + qna_lsa + role_sequence`: `R^2 ≈ 0.861`
- `A4 + qna_lsa + weak_sequence`: `R^2 ≈ 0.917`

Clean `after_hours`:

- `A4 + qna_lsa`: `R^2 ≈ 0.935`
- `A4 + qna_lsa + role_sequence`: `R^2 ≈ 0.839`
- `A4 + qna_lsa + weak_sequence`: `R^2 ≈ 0.886`

Clean significance:

- `A4 + qna_lsa -> + role_sequence`
  - permutation `mse p ≈ 0.00025`
  - permutation `mae p ≈ 0.0`
- `A4 + qna_lsa -> + weak_sequence`
  - permutation `mse p ≈ 0.006`
  - permutation `mae p ≈ 0.0`

So the message is not just "sequence does not help by itself."
It is stronger:

- sequence currently damages the best `after_hours` `A4 + semantics` model.

## 4. What This Means Scientifically

This checkpoint makes the contribution path much clearer.

The useful extension is:

- coarse `A4` observability or alignment structure
- plus compact `Q&A` semantic summarisation

The unhelpful extension is:

- large fixed-bin sequence expansions, even though they sound more sophisticated.

That is exactly the kind of limitation-driven negative result that helps the paper:

- noisy scheduled timing does matter,
- coarse alignment quality matters,
- compact semantics matter,
- but current high-dimensional sequence modeling is too brittle for this sample size and supervision quality.

## 5. Best Next Step

The next technical step should now focus on:

- improving the `A4 + qna_lsa` line on `clean after_hours`,
- not on building a larger sequence architecture over the current bins.

If we keep pushing the current sample, the highest-value direction is:

- a compact `after_hours` model centered on `market + controls + A4 + Q&A semantics`,
- with any further extension judged against that benchmark rather than against the weaker pooled off-hours models.
