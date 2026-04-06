# Q&A Signal Checkpoint 2026-03-14

This note compares the original `qa_benchmark` stack (`qav1`) with the richer heuristic `Q&A` feature refresh (`qav2`) and locks in the current interpretation before the next modeling branch.

## Coverage

- `qa_benchmark_features_v2` covers `553` events, `552` with at least one detected pair, and about `9.63` pairs per event.

## Decomposition checkpoint

| Slice | Market-only | Market+controls | ECC-only qav1 | ECC-only qav2 | Delta | ECC+audio qav2 | Best market+ECC qav2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full sample, all HTML rows | 0.904 | 0.909 | -0.022 | 0.095 | +0.117 | 0.032 | 0.884 |
| Full sample, exclude html_integrity_flag=fail | 0.901 | 0.911 | -0.074 | -0.005 | +0.069 | 0.187 | 0.878 |
| Off-hours only, all HTML rows | 0.907 | 0.909 | -0.076 | 0.139 | +0.215 | 0.180 | 0.892 |
| Off-hours only, exclude html_integrity_flag=fail | 0.905 | 0.911 | -0.054 | -0.085 | -0.032 | -0.013 | 0.905 |

Key reading from the table:

- Full-sample ECC-only improves from `-0.022` to `0.095`, but still stays far below `market_only=0.904` and `market_plus_controls=0.909`.
- Off-hours ECC-only also improves on the all-HTML slice from `-0.076` to `0.139`, showing that richer Q&A semantics recover some event-specific signal.
- Clean-sample robustness is still weak: full-sample ECC-only falls to `-0.005` and off-hours ECC-only falls to `-0.085`.
- Even when `qav2` helps ECC-only models, adding ECC back on top of market controls still underperforms the best market baseline. Example: full-sample `market_plus_controls=0.909` versus `market_controls_plus_ecc=0.884`.

## Headline model comparison

| Checkpoint | qav1 | qav2 | Delta |
| --- | ---: | ---: | ---: |
| Full-sample shock best mixed model | 0.890 | 0.884 | -0.006 |
| Off-hours shock best mixed model | 0.897 | 0.891 | -0.005 |
| Global regime residual model | 0.842 | 0.882 | +0.040 |
| Regime-specific residual model | 0.858 | 0.799 | -0.059 |

## Current decision

- `qav2` is useful because it recovers positive ECC-only signal on the all-HTML sample and improves the global residual regime model.
- `qav2` is not enough for a main claim of incremental ECC value because the gains do not survive the clean-sample slices and do not beat the strongest market-only benchmarks.
- The current repo should therefore describe `qav2` as a partial signal-recovery step, not a paper-level breakthrough.

## Next gate before any bigger model push

1. Replace heuristic event-level `qav2` features with transferred pair-level labels or scores inspired by `SubjECTive-QA` and `EvasionBench`.
2. Keep the benchmark ladder fixed: `prior_only`, `market_only`, `market_plus_controls`, `ECC_only`, and `market_plus_ECC`.
3. Require any new ECC claim to survive the `exclude html_integrity_flag=fail` rerun before it becomes part of the paper narrative.
4. If stronger transferred `Q&A` signals still fail to beat the market-only residual benchmark, pivot to the public reproducible benchmark path rather than scaling up the same heuristic feature family.

