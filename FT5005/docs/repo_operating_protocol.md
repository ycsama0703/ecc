# Repository Operating Protocol

Purpose:
- keep all future ICAIF work scoped to this repository only
- make progress recoverable through frequent git checkpoints
- prevent oversized intermediate artifacts from destabilizing the repo

Canonical project scope:
- repository root: `/media/volume/dataset/xma8/work/icaif_ecc_news_attention`
- all active code, notes, experiment summaries, and compact reproducibility artifacts should stay inside this folder
- do not mix operating notes or progress tracking with unrelated projects

What should be tracked in git:
- code under `scripts/`
- paper-facing or handoff-facing docs under `docs/`
- `README.md`
- compact result artifacts that support a checkpoint, such as:
  - summary JSON files
  - prediction CSV files
  - small event-level feature tables needed to rerun a benchmark

What should stay local and be ignored:
- raw data under `data/`
- bulky caches such as decoded wav caches
- sentence-level aligned-audio tables
- temporary panel subsets and smoke intermediates
- any artifact that is too large or too low-level to be a stable repo checkpoint

Checkpoint cadence:
1. finish one coherent experiment block
2. update `docs/progress_log.md`
3. update `README.md` if the main project reading changed
4. commit code + docs + compact results
5. push to `origin/main`

Preferred checkpoint granularity:
- at least one push per meaningful experiment round
- do not leave major findings only in an uncommitted working tree

Current near-term continuation rule:
- prioritize the locked ICAIF mainline
- prefer `after_hours` / `shock_minus_pre` / prior-aware evaluation work over unrelated exploratory branches
- treat larger sequence expansions and bulky intermediate dumps as secondary unless they change the main conclusion
