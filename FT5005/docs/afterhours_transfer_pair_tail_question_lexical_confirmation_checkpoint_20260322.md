# After-Hours Transfer Pair-Tail Question Lexical Confirmation Checkpoint 2026-03-22

## Purpose

The newest compact lexical-pattern benchmark gave us the first encouraging small positive result on the hardest-question line:

- a one-factor `clarify_modeling_lex` latent slightly edged above hard abstention,
- but remained below the richer `question_lsa4_bi` route.

That makes the next research question very natural:

## is this compact lexical factor actually capturing the most reliable core of the richer hardest-question semantic route,
## or is it just a smaller and noisier alternative?

This checkpoint answers that directly by comparing the compact lexical factor against:

- the current hard-abstention shell,
- the richer hardest-question semantic route,
- and the overlap structure of their agreement-veto behavior.

## Design

New script:

- `scripts/run_afterhours_transfer_pair_tail_question_lexical_confirmation.py`

Inputs:

- lexical-pattern benchmark predictions
- panel-side observability fields
- compact event feature bundle
- benchmark-style `Q&A` quality fields

Outputs:

- `results/afterhours_transfer_pair_tail_question_lexical_confirmation_real/afterhours_transfer_pair_tail_question_lexical_confirmation_summary.json`
- `results/afterhours_transfer_pair_tail_question_lexical_confirmation_real/afterhours_transfer_pair_tail_question_lexical_confirmation_groups.csv`
- `results/afterhours_transfer_pair_tail_question_lexical_confirmation_real/afterhours_transfer_pair_tail_question_lexical_confirmation_years.csv`
- `results/afterhours_transfer_pair_tail_question_lexical_confirmation_real/afterhours_transfer_pair_tail_question_lexical_confirmation_events.csv`

The main comparison partitions the held-out latest window into five overlap groups:

1. `shared_veto`
   - both the compact lexical factor and the richer semantic route veto the agreed prediction
2. `sem_only_veto`
   - only the richer `question_lsa4_bi` route vetoes
3. `lex_only_veto`
   - only the compact lexical factor vetoes
4. `both_keep_agreed`
   - both routes keep the agreed prediction
5. `disagreement`
   - both inherit the default hard-abstention fallback behavior

## Main Findings

### 1. The compact lexical factor is much more conservative than the richer semantic route

Held-out latest-window overlap counts:

- agreement rows: `44`
- disagreement rows: `16`
- compact lexical veto rows: `7`
- richer semantic veto rows: `13`
- shared veto rows: `4`
- semantic-only veto rows: `9`
- lexical-only veto rows: `3`

So the compact lexical factor is clearly not trying to replicate the full semantic route.
Instead it is acting like a **high-precision, low-recall subset** of that broader behavior.

### 2. The shared veto subset is the clean positive core

On the `shared_veto` rows:

- `n = 4`
- compact lexical net gain vs hard abstention `≈ 2.42e-09`
- richer semantic net gain vs hard abstention `≈ 2.42e-09`
- compact lexical win share `= 1.0`
- richer semantic win share `= 1.0`

So the rows where the compact factor agrees with the richer semantic route are exactly the rows where the signal looks cleanest.

## this shared-veto subset is the precision core of the local hardest-question signal

### 3. The richer semantic route gets more recall, but also carries the mixed tail

On the `sem_only_veto` rows:

- `n = 9`
- compact lexical net gain vs hard abstention `= 0`
- richer semantic net gain vs hard abstention `≈ 7.36e-09`
- richer semantic win share `≈ 0.556`
- richer semantic paired `p(MSE) ≈ 0.2045`

This is an important pattern:

- the richer semantic route captures additional upside that the compact factor misses,
- but it also carries a noticeably less pure subset than the shared veto core.

So the compact lexical factor is not “better” than the richer semantic route.
It is **more selective**.

### 4. The lexical-only tail is exactly where the compact factor is least trustworthy

On the `lex_only_veto` rows:

- `n = 3`
- compact lexical net gain vs hard abstention `≈ -7.31e-10`
- compact lexical win share `= 0`
- richer semantic net gain vs hard abstention `= 0`

The main negative example is:

- `IBM_2023_Q4`

So the one place where the compact lexical factor departs from the richer semantic route on its own is precisely where it underperforms.

## this is strong evidence that the compact factor is useful only as a partial core,
## not as a standalone replacement for the richer semantic route

### 5. The compact lexical factor concentrates on slightly “cleaner” difficult interaction rows

Group averages show:

#### shared veto
- `qa_pair_count ≈ 4.5`
- directness `≈ 0.814`
- evasion `≈ 0.220`
- `A4` row-share `≈ 0.717`

#### semantic-only veto
- `qa_pair_count ≈ 6.22`
- directness `≈ 0.781`
- evasion `≈ 0.204`
- `A4` row-share `≈ 0.724`

This is subtle but useful:

- the compact factor is not just chasing the densest or noisiest cases,
- it seems to isolate a narrower subset where the clarificatory/model-building framing is already strong enough to be usable in a compact form.

That matches the earlier reading:

## the compact factor is capturing the most recoverable part of the local framing pocket,
## not the entire difficult-interaction region

### 6. The compact factor is mainly a `2023–2024` phenomenon so far

Year-wise net gain vs hard abstention:

- `2023`: `≈ 3.88e-10`
- `2024`: `≈ 1.31e-09`
- `2025`: `0.0`

By contrast, the richer semantic route still shows a small positive tail in `2025`.

So another honest reading is:

- the compact clarificatory core is real,
- but it is currently even narrower than the richer semantic route in time.

## Updated Interpretation

This is a strong confirmation checkpoint in a very research-useful way.

### What we now know

1. The compact lexical factor is **not** a noisy duplicate of the richer semantic route.
2. It behaves like a **precision-oriented subset** of the richer hardest-question veto mechanism.
3. Its strongest contribution comes exactly from the shared-veto rows, where both routes agree and all observed gains are positive.
4. The richer semantic route still captures additional upside beyond that compact core.
5. The lexical-only departures are the weak point and should not be promoted into a new method claim.

### What that means

The current best reading is:

## the hardest-question signal has a compact clarificatory core,
## and that compact core is the cleanest part of the broader semantic route,
## but it is still only a subset of the full transferable signal

This is a good place to be scientifically because it sharpens the hierarchy:

- `hard_abstention` remains the safest transfer-side mainline,
- `question_lsa4_bi` remains the stronger richer exploratory route,
- `clarify_modeling_lex_factor_pca1` is now the best compact confirmation of the same local mechanism.

So the next step should not be to keep expanding lexical families.
Instead, it should test whether this compact clarificatory factor remains stable on the specific slices where the richer semantic route is strongest, and whether the lexical-only tail can be filtered away without giving up the shared positive core.
