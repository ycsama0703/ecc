# After-Hours Transfer Role-Text Signal Checkpoint 2026-03-20

## Purpose

After the factor-expert integration result, the next research question was:

## is there a genuinely stronger upstream transferable signal in the **raw role-specific Q/A text**, rather than in another recombination of handcrafted proxy features?

This is a cleaner test than more router work because it changes the source of information:

- not more routing geometry,
- not more quality-rule proxies,
- not another direct factor-expert trick,
- but a compact low-rank view of what analysts ask and what management answers.

That is a more defensible next step if the goal is:

- less rule-based structure,
- more learnable signal,
- compactness,
- and better transferability.

## Design

New script:

- `scripts/run_afterhours_transfer_role_text_signal_benchmark.py`

Inputs:

- temporal router outputs:
  - `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`
- event-level text features:
  - `results/features_real/event_text_audio_features.csv`

Coverage is complete:

- `257 / 257` temporal rows have both `question_text` and `answer_text`

The transfer protocol stays fixed:

- train on `val2020_test_post2020` agreement rows
- tune on `val2021_test_post2021` agreement rows
- refit on `2020 + 2021` agreement rows
- test on full `val2022_test_post2022`
- disagreement still falls back to `pre_call_market_only`

### New upstream representation

For each split, the script fits one **shared TF-IDF + LSA basis** on the union of:

- `question_text`
- `answer_text`

Then it builds three compact role-aware text views:

1. `question_role_lsa`
2. `answer_role_lsa`
3. `qa_role_gap_lsa = answer_role_lsa - question_role_lsa`

with `lsa_components = 4`.

### Families benchmarked

1. `geometry_only`
2. `question_role_lsa`
3. `answer_role_lsa`
4. `qa_role_gap_lsa`
5. `geometry_plus_answer_role`
6. `geometry_plus_role_gap`
7. `geometry_plus_dual_role`

Outputs:

- `results/afterhours_transfer_role_text_signal_benchmark_lsa4_real/afterhours_transfer_role_text_signal_benchmark_summary.json`
- `results/afterhours_transfer_role_text_signal_benchmark_lsa4_real/afterhours_transfer_role_text_signal_benchmark_overview.csv`
- `results/afterhours_transfer_role_text_signal_benchmark_lsa4_real/afterhours_transfer_role_text_signal_benchmark_tuning.csv`
- `results/afterhours_transfer_role_text_signal_benchmark_lsa4_real/afterhours_transfer_role_text_signal_benchmark_test_predictions.csv`
- `results/afterhours_transfer_role_text_signal_benchmark_lsa4_real/afterhours_transfer_role_text_signal_benchmark_component_terms.csv`

## Main Findings

### 1. Analyst-question text is now the strongest standalone compact transfer route in the repo

Best factor-only route:

- **`question_role_lsa ≈ 0.998621932`**

Comparison points:

- `pre_call_market_only ≈ 0.998482062`
- previous best standalone content-accountability factor `≈ 0.998592197`
- `geometry_only ≈ 0.998638784`
- `hard abstention ≈ 0.998640168`

So this is the strongest standalone compact route seen so far in the transfer branch.

That matters because it is not coming from a handcrafted benchmark table. It comes directly from the low-rank analyst-question text.

But it still does **not** beat the current best shell:

- vs `pre_call_market_only`: `p(MSE) ≈ 0.233`
- vs hard abstention: `p(MSE) ≈ 0.943`

So the right reading is:

## question-role text is a strong new compact signal, but still not a replacement for the hard-abstention transfer shell

### 2. Question text is more transferable than answer text or question-answer gap

Standalone role-text routes:

- `question_role_lsa ≈ 0.998621932`
- `answer_role_lsa ≈ 0.998553958`
- `qa_role_gap_lsa ≈ 0.998474838`

This is one of the most informative parts of the checkpoint.

The strongest raw role-text signal is not:

- the answer content alone,
- and not the answer-minus-question gap,

but the **analyst-question semantic block** itself.

That suggests a more precise research hypothesis:

## cross-ticker transfer is more sensitive to what analysts choose to ask than to the full answer semantics taken in isolation

### 3. Geometry-plus-role-text families do not improve the current shell

Best geometry-coupled family:

- `geometry_only ≈ 0.998638784`

Role-text augmentations:

- `geometry_plus_dual_role ≈ 0.998638776`
- `geometry_plus_answer_role ≈ 0.998638309`
- `geometry_plus_role_gap ≈ 0.998637874`

All of them are effectively tied or slightly worse than plain `geometry_only`, and all remain below hard abstention.

So the new role-text source behaves like the recent content factor:

## it is useful as a standalone complementary compact signal, but not yet as a better trust / gating signal

### 4. The new role-text signal is interpretable at the topic level

The shared role-text basis is not just capturing generic verbosity.

The more interpretable components in the refit basis include topic axes such as:

- `AI / cloud / software / data center / NVIDIA`
- `Disney / parks / content / Hulu / ESPN`
- `patients / phase / study / obesity`

So the new signal is not only compact, but also substantively legible:

## it is picking up cross-company topical structure in analyst questions and management answers

The key empirical point is that the **question-side** projection transfers better than the answer-side projection.

## Updated Interpretation

This checkpoint meaningfully strengthens the current research picture.

### What we now know

1. Raw role-specific Q/A text contains a stronger transferable compact signal than the earlier handcrafted factor families alone.
2. The strongest standalone signal comes from **analyst questions**, not answers.
3. But the new role-text signal still does not improve the current geometry-led / hard-abstention shell.
4. So it should currently be treated as a **new compact complementary route**, not as a replacement transfer controller.

### What that means

The transfer hierarchy is now cleaner:

- **hard abstention** remains the strongest compact transfer-side method,
- **geometry-only** remains the best agreement-side controller,
- **question-role low-rank text** is now the strongest standalone compact complementary signal,
- while answer-only and question-answer-gap text views are weaker.

So the next meaningful gain should probably come from:

- integrating the question-role semantic block more carefully than the earlier factor-expert selection did,
- or finding another upstream signal that complements question-role semantics without destabilizing the current shell.
