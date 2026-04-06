# I-POG-QA Benchmark Suite Note (2026-04-06)

## Purpose

This note defines the benchmark suite that should be used for the paper's main method table once the processed real panel is available locally.

The suite runner is:

- `scripts/run_i_pog_qa_benchmark_suite.py`

Its role is not to search broadly.

Its role is to produce one clean comparison set for the final paper:

- strong prior-aware baselines
- the retained ungated `Q&A` baselines
- the full `I-POG-QA`
- and the minimum set of ablations needed to explain what the method is actually doing

## What the suite includes

### Prior-aware baselines

The suite keeps these benchmark anchors:

1. `prior_only`
2. `residual_base_structured`
3. `residual_base_plus_semantic`
4. `residual_base_plus_quality`
5. `residual_base_plus_semantic_plus_quality`

These are the comparison points the new method must beat or at least meaningfully clarify.

### Main method

The retained main method is:

- `i_pog_qa_full`

This is the full `I-POG-QA`:

- monotone trust gate
- learned semantic-versus-quality route gate
- incrementality regularization
- activation regularization

### Key ablations

The suite currently includes:

1. `i_pog_qa_no_incrementality`
- removes the incrementality regularizer
- tests whether the explicit "beyond-prior/base" constraint matters

2. `i_pog_qa_no_activation_reg`
- removes the default-to-base activation penalty
- tests whether the model benefits from being conservative about using `Q&A` corrections

3. `i_pog_qa_free_trust`
- replaces the monotone trust gate with a free-sign trust gate
- tests whether the reliability gate should remain direction-constrained

4. `i_pog_qa_no_trust_gate`
- sets the trust gate to always on
- tests whether event-level reliability gating matters at all

5. `i_pog_qa_semantic_only`
- fixes the route to the semantic expert only
- tests whether the quality/accountability expert is actually needed

6. `i_pog_qa_quality_only`
- fixes the route to the quality expert only
- tests whether the semantic expert is actually needed

## What this suite is designed to answer

The suite is meant to answer four method questions cleanly:

1. Does the full `I-POG-QA` improve over the best ungated `Q&A` extension?
2. Does monotone reliability gating help relative to free or no trust gating?
3. Does incrementality regularization help keep the method aligned with the paper's question?
4. Is the useful dialog correction primarily semantic, primarily accountability-driven, or genuinely mixed?

## Output artifacts

The suite writes:

- `i_pog_qa_benchmark_suite_summary.json`
- `i_pog_qa_benchmark_suite_predictions.csv`
- `i_pog_qa_benchmark_suite_gate_diagnostics.csv`

These outputs are intended to support:

- the main results table
- the method ablation table
- and event-level gate diagnostics for paper interpretation

## Immediate real-data run order

Once the processed real panel is locally available, the recommended order is:

1. Run the suite on `clean after_hours + shock_minus_pre`
2. Freeze the main table and read whether `I-POG-QA` actually beats the best ungated baseline
3. Re-run the suite on `within_minus_pre`
4. Decide which target family becomes the conference headline and which becomes robustness

## Current status

The suite has already passed:

- `py_compile`
- synthetic end-to-end smoke execution

So the remaining blocker is not method plumbing.

The remaining blocker is access to the real processed panel on this machine.
