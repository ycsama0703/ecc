# After-Hours Transfer Question-Role Slice Diagnostics Checkpoint 2026-03-21

## Purpose

After the last two role-text checkpoints, the picture was already fairly clear:

- `question_role_lsa` is the strongest **standalone** compact transfer signal,
- but it does **not** improve the geometry-led / hard-abstention shell,
- and it does **not** improve the fixed-split clean `after_hours` mainline either.

So the next research question is no longer:

- how do we force question-role text into the main model?

It is instead:

## where is analyst-question semantics actually helping, and what kind of signal is it carrying?

That is a better research move because it keeps the method compact and interpretable, and it tells us whether question-role text should be treated as:

- a broad transferable controller,
- a mainline additive feature,
- or a much narrower diagnostic semantic axis.

## Design

New script:

- `scripts/run_afterhours_transfer_question_role_slice_diagnostics.py`

Inputs:

- temporal transfer shell:
  - `results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real/`
- role-text benchmark outputs:
  - `results/afterhours_transfer_role_text_signal_benchmark_lsa4_real/`
- event-level text features:
  - `results/features_real/event_text_audio_features.csv`

Protocol:

- reuse the exact role-text transfer setup:
  - train split: `val2020_test_post2020`
  - validation split: `val2021_test_post2021`
  - test split: `val2022_test_post2022`
- rebuild the shared analyst-question / answer `TF-IDF + LSA(4)` basis
- assign each held-out latest-window event to its **dominant signed question-role component**
- compare standalone `question_role_lsa` against:
  - `pre_call_market_only`
  - `geometry_only`
  - hard abstention (`agreement_pre_only_abstention`)

Outputs:

- `results/afterhours_transfer_question_role_slice_diagnostics_lsa4_real/afterhours_transfer_question_role_slice_diagnostics_summary.json`
- `results/afterhours_transfer_question_role_slice_diagnostics_lsa4_real/afterhours_transfer_question_role_slice_diagnostics_slices.csv`
- `results/afterhours_transfer_question_role_slice_diagnostics_lsa4_real/afterhours_transfer_question_role_slice_diagnostics_tickers.csv`
- `results/afterhours_transfer_question_role_slice_diagnostics_lsa4_real/afterhours_transfer_question_role_slice_diagnostics_events.csv`

## Main Findings

### 1. The standalone question-role gain over `pre_call_market_only` is real, but extremely sparse

Held-out latest-window overall:

- `question_role_lsa` still sits above `pre_call_market_only` in pooled test `R²`
- but the event-level gain pattern is highly concentrated:
  - question-role beats `pre_call_market_only` on only **`10 / 60`** events
  - top `1` positive event contributes **`≈ 46.7%`** of all positive gain mass
  - top `3` positive events contribute **`≈ 89.6%`**
  - top `5` positive events contribute **`≈ 98.1%`**

So the pooled improvement is not coming from broad, stable lift across the held-out latest window.

## it comes from a very small number of concentrated event pockets

### 2. Once geometry or hard abstention is available, question-role semantics almost never win

Overall held-out latest-window comparisons:

- mean MSE gain of question-role vs `pre_call_market_only`: **positive** (`≈ 1.57e-09`)
- mean MSE gain vs `geometry_only`: **negative** (`≈ -1.89e-10`)
- mean MSE gain vs hard abstention: **negative** (`≈ -2.05e-10`)

Win shares:

- vs `pre_call_market_only`: **`≈ 16.7%`**
- vs `geometry_only`: **`≈ 8.3%`**
- vs hard abstention: **`≈ 5.0%`**

So the right interpretation is:

- question-role text can help as a standalone compact signal,
- but it is rarely the best prediction once the stronger transfer shell is already in place.

### 3. The positive pockets are interpretable, but still not stable enough to become a new rule

The biggest positive events are semantically meaningful:

- `NVDA_2024_Q1`
  - dominant slice: **data center / computing / nvidia**
- `AAPL_2023_Q1`
  - dominant slice: **iphone / china / data center**
- `AMZN_2023_Q1`, `AMZN_2024_Q2`
  - dominant slice: **high performance / execution / compute / optimization**
- `CSCO_2023_Q2`
  - strong analyst follow-up / next-fiscal-year / supply-visibility style question pocket

Ticker-level averages also point in the same direction:

- strongest mean gains vs `pre_call_market_only` show up for:
  - `NVDA`
  - `CSCO`
  - `AAPL`
  - `AMZN`

That is useful because it says analyst-question semantics are not random noise.

But the key limitation is just as important:

### 4. Even within the apparent “good” semantic pockets, the signal is still highly heterogeneous

Examples:

- the `data center / computing / nvidia` slice has positive mean gain,
  - but its win share vs `pre_call_market_only` is still only **`0.1`**
- the `high performance / execution / compute` slice contains both:
  - positive `AMZN_2023_Q1`, `AMZN_2024_Q2`
  - negative `AMZN_2024_Q1`, `AMZN_2024_Q3`, `NKE_2023_Q1`
- the latest strong negative outliers include:
  - `CSCO_2024_Q2`
  - `IBM_2024_Q1`

So even where the semantics look coherent, the gain is **not** broad enough to justify a new deterministic routing rule.

## question-role semantics are carrying a real signal, but it behaves like a sparse event-level analyst-attention detector, not a stable slice-wide controller

## Updated Interpretation

This checkpoint narrows the role-text story in a very useful way.

### What we now know

1. `question_role_lsa` is still the strongest standalone compact transfer signal.
2. Its pooled gain over `pre_call_market_only` is driven by a very small number of events, not broad event-level lift.
3. Once `geometry_only` or hard abstention is available, question-role almost never becomes the better prediction.
4. The positive pockets are semantically interpretable, but they are still too sparse and heterogeneous to justify a new routing rule.

### What that means

The cleanest reading now is:

- analyst-question semantics should **not** be promoted into the main transfer shell as a direct controller,
- and they should **not** be promoted into the fixed-split headline as an additive feature,
- but they **should** be retained as a compact diagnostic semantic axis for understanding where analyst attention exposes special event pockets.

That is a useful research outcome because it preserves the strongest lesson from the role-text line without forcing us into a more complicated or less transferable model.
