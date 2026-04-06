# After-Hours Transfer Question-Role Gate Diagnostics Checkpoint 2026-03-21

## Purpose

The previous slice-diagnostics checkpoint already showed that analyst-question semantics are real but sparse:

- standalone `question_role_lsa` beats `pre_call_market_only` only on a small minority of held-out events,
- and once `geometry_only` or hard abstention is available, it almost never becomes the best prediction.

That raises the next clean question:

## what is the question-role family actually doing relative to hard abstention?

This is an important mechanistic question because the role-text family is not a free-form predictor in the current transfer shell.
It is effectively making a compact decision about whether to:

- keep the agreed transfer correction,
- or veto it and fall back to `pre_call_market_only`.

So the real issue is:

## is question-role text learning a useful agreement filter, or just a noisy veto rule?

## Design

New script:

- `scripts/run_afterhours_transfer_question_role_gate_diagnostics.py`

Inputs:

- question-role slice diagnostics:
  - `results/afterhours_transfer_question_role_slice_diagnostics_lsa4_real/`
- role-text transfer benchmark predictions:
  - `results/afterhours_transfer_role_text_signal_benchmark_lsa4_real/`

The script works directly on the held-out latest-window test rows and separates them into three states:

1. `disagreement_auto_pre`
   - router disagreement already forces the family back to `pre_call_market_only`
2. `agreement_keep_agreed`
   - question-role keeps the agreed transfer correction
3. `agreement_veto_to_pre`
   - question-role overrides the agreement and abstains back to `pre_call_market_only`

Outputs:

- `results/afterhours_transfer_question_role_gate_diagnostics_lsa4_real/afterhours_transfer_question_role_gate_diagnostics_summary.json`
- `results/afterhours_transfer_question_role_gate_diagnostics_lsa4_real/afterhours_transfer_question_role_gate_diagnostics_states.csv`
- `results/afterhours_transfer_question_role_gate_diagnostics_lsa4_real/afterhours_transfer_question_role_gate_diagnostics_components.csv`
- `results/afterhours_transfer_question_role_gate_diagnostics_lsa4_real/afterhours_transfer_question_role_gate_diagnostics_tickers.csv`
- `results/afterhours_transfer_question_role_gate_diagnostics_lsa4_real/afterhours_transfer_question_role_gate_diagnostics_events.csv`

## Main Findings

### 1. The question-role family differs from hard abstention only on a small veto subset

Held-out latest-window counts:

- total events: `60`
- disagreement auto-pre: `16`
- agreement keep-agreed: `29`
- agreement veto-to-pre: `15`

This means:

- on `16` disagreement events, question-role does nothing beyond the existing shell
- on `29` agreement events, question-role is **identical** to hard abstention
- only on `15` events does it make a distinct decision by vetoing the agreed expert

So the whole remaining difference between question-role and hard abstention lives inside a very small subset.

### 2. The “keep agreed” state is where the standalone gain over `pre_call_market_only` comes from

For `agreement_keep_agreed` events:

- mean gain vs `pre_call_market_only` is **positive** (`≈ 3.25e-09`)
- win share vs `pre_call_market_only` is **`≈ 34.5%`**
- gain vs hard abstention is exactly `0`

This is the cleanest reading of the role-text family:

## when it keeps the agreed expert, it is basically riding the same useful agreement events already captured by the existing shell

### 3. The veto subset is where question-role tries to be smarter than hard abstention, but mostly fails

For `agreement_veto_to_pre` events:

- mean gain vs hard abstention is **negative** (`≈ -8.20e-10`)
- win share vs hard abstention is only **`≈ 20%`**
- gain vs `pre_call_market_only` is exactly `0`

So the distinctive action of the question-role family is:

- not to create a better transfer prediction directly,
- but to **veto** some agreement events.

And that veto policy is currently not good enough.

## the question-role family behaves like a noisy agreement filter rather than a stronger controller

### 4. The veto policy is not completely random, but it is still too unstable

The veto subset does look slightly more difficult on average:

- mean hard-abstention squared error on `agreement_keep_agreed` events:
  - `≈ 1.75e-08`
- mean hard-abstention squared error on `agreement_veto_to_pre` events:
  - `≈ 2.40e-08`

So the question-role gate is not firing on totally easy cases.
It is trying to abstain on somewhat riskier agreement events.

But the outcome is still weak because the veto decisions are unreliable.

Largest veto improvements vs hard abstention:

- `AAPL_2023_Q4`
- `NVDA_2023_Q1`
- `MSFT_2025_Q1`

Largest veto failure:

- `NKE_2025_Q2`

So even inside the veto subset, the policy is highly heterogeneous.

### 5. The strongest veto concentration sits in generic / weakly structured slices, not in the cleanest topic pockets

Examples from the component-level summaries:

- the most heavily vetoed slice is the generic management-phrasing bucket
  - veto share within agreement `≈ 0.8`
  - mean gain vs hard is clearly negative
- more interpretable topic pockets such as:
  - `data center / computing / nvidia`
  - `high performance / execution / compute`
  - are less aggressively vetoed
  - and are often just kept identical to hard abstention

Ticker-level summaries tell the same story:

- `AMGN` agreement events are vetoed almost mechanically, but with zero gain
- `NKE` vetoes are actively harmful on average
- `AMZN`, `DIS`, and `IBM` are mostly kept, so question-role adds little beyond the shell there

This is important because it suggests the current role-text family is not surfacing a robust new semantic correction regime.

## it is mostly deciding when **not** to trust agreement, and that decision is still too noisy to beat hard abstention

## Updated Interpretation

This checkpoint makes the role-text line much cleaner.

### What we now know

1. The role-text family only differs from hard abstention on a small agreement-veto subset.
2. Its useful behavior mostly comes from the same agreement events already captured by the shell.
3. The extra veto policy is the only new decision it makes, and that policy is currently net negative.
4. The veto logic is not random — it does target somewhat riskier agreement events — but it is still too unstable to become a new transfer rule.

### What that means

The right way to keep question-role semantics in the project is now much narrower:

- **not** as a direct mainline feature,
- **not** as a replacement semantic expert,
- **not** as a new routing rule,
- but as a compact diagnostic lens on where analyst-attention structure differs from the safer hard-abstention shell.

That is still a useful research finding, because it tells us exactly what role-text is and is not doing in the current project.
