# Project State and Paper Alignment 2026-03-18

## Why This Note Exists

The repo now contains enough checkpoints that the main risk is no longer lack of work.

The main risk is losing the hierarchy between:

- the strongest overall benchmark result,
- the strongest incremental ECC claim,
- the transfer-side method extension,
- and the parts that should remain extension material rather than headline claims.

This note is meant to give the project a single paper-facing alignment snapshot before the next research round continues.

---

## 1. Current Repo State In One Paragraph

As of `2026-03-18`, the project is no longer in an early exploration phase. The repo has a stable corrected off-hours benchmark, a narrower and more credible clean `after_hours` ECC-increment line, a clarified negative result on heavy sequence and generic multimodal stacking, and a more coherent transfer-side extension centered on **reliability-aware abstention** rather than brute-force routing complexity. The project is now in a **paper-story consolidation plus targeted research extension** stage, not a raw discovery stage.

---

## 2. What Is Currently Locked

### 2.1 Problem framing

The safe framing is now:

- **Off-hours earnings-call volatility-shock prediction under noisy timing**
- evaluation against strong same-ticker / pre-call market priors
- with `A4` treated as noisy observability / alignment supervision rather than gold timestamps

This is a better framing than:

- generic multimodal ECC prediction
- generic sequence modeling
- or “more modalities always help”

### 2.2 Main target and sample

The paper-safe core remains:

- target: `shock_minus_pre`
- strongest regime: off-hours, especially clean `after_hours`
- strong prior-aware evaluation rather than raw pooled prediction alone

### 2.3 Fixed-split headline

The current strongest incremental ECC line is:

- **clean `after_hours` + `A4 + compact Q&A semantics`**
- with the fixed temporal split semantic bottleneck still best at **`lsa=64`**

This is the safest current headline contribution because it is:

- narrower,
- more interpretable,
- and better aligned with the data limitations.

---

## 3. What The Best Results Actually Mean

### 3.1 Strongest overall benchmark result

The repo still contains a very strong corrected pooled off-hours benchmark result.

Safe reading:

- this is the strongest overall predictive result in the current repo,
- but it remains substantially **market-dominated**.

So it is valuable as:

- a benchmark anchor,
- proof that the corrected target / evaluation stack is real,
- and evidence that the repo has a strong empirical core.

But it is **not** by itself the cleanest ECC-increment claim.

### 3.2 Strongest incremental ECC claim

The cleanest current incremental contribution is narrower:

- in clean `after_hours`,
- `A4` observability / alignment signal improves over the market-control baseline,
- and compact `Q&A` semantics improves that line further in the fixed split.

So the most defensible main claim is not:

- “multimodal modeling wins everywhere”

It is closer to:

- under noisy timing, carefully filtered `after_hours` calls contain usable incremental ECC signal,
- and the most credible part of that increment currently comes from `A4` plus compact `Q&A` semantics.

---

## 4. What Is No Longer A Good Main Story

The repo has now accumulated enough evidence to demote several earlier directions.

### 4.1 Heavy sequence modeling

Sequence-heavy / strict sequence routes are not the best current result and often hurt.

Safe status:

- **extension at best**
- not headline material right now

### 4.2 Generic multimodal stacking

The current evidence does not support a broad claim that adding all modalities helps.

Safe status:

- **not headline-safe**
- use only as a carefully bounded extension statement

### 4.3 Audio as the primary contribution

Aligned audio became much more concrete over the last two days, but the repo now says something narrower:

- raw aligned audio is too noisy,
- compressed aligned audio can help on some branches,
- but audio does **not** robustly improve the strongest fixed-split semantic line.

Safe status:

- **supporting extension, not the main claim**

### 4.4 Bigger gate/router complexity as a claim in itself

The latest transfer-side search strongly suggests that the gain does not come from making the router increasingly elaborate.

Safe status:

- more complexity is not the story,
- **reliability-aware trust / abstention** is the story.

---

## 5. Current Best Transfer-Side Reading

The transfer-side story has evolved a lot and is now much cleaner than earlier in the week.

### 5.1 What transfer does **not** currently support

The repo does **not** currently support a strong claim that semantic or audio transfer reliably beats `pre_call_market_only` across ticker-held-out and temporal slices.

That stronger claim would overstate the evidence.

### 5.2 What transfer **does** currently support

The most defensible current method interpretation is:

## **reliability-aware abstention**

Meaning:

- when the retained hybrid router views agree, the transfer-side correction is more credible,
- when they disagree, the safest fallback is often the strongest market baseline.

### 5.3 Current strongest temporal transfer route

The current best pooled temporal transfer route is:

- **agreement-triggered abstention to `pre_call_market_only`**
- pooled `R² ≈ 0.997919`

Important caveat:

- it still does **not** significantly beat raw `pre_call_market_only`
- so it is a **coherent extension**, not a headline victory claim.

### 5.4 Why the newest abstention diagnostic matters

The newest checkpoint strengthens the transfer story in a more scientific way:

- the pooled **agreement** subset is where the transfer-side lift lives,
- the pooled **disagreement** subset is where `pre_call_market_only` is still safest,
- and this gives us a more defensible interpretation than “routing helps.”

Safe wording:

- **agreement reveals where transfer signal is credible**
- **disagreement reveals where abstention is safer**

That is currently the cleanest method idea in the repo.

---

## 6. What The Project Phase Is Right Now

The project is best described as:

## **paper-story consolidation + targeted research extension**

Not:

- early ideation
- raw baseline setup
- or blind feature expansion

Why:

- the repo now has many checkpointed results,
- most important negative results have been made explicit,
- the headline and extension layers are distinguishable,
- and the remaining work is about strengthening the right claims rather than inventing an entirely new direction.

---

## 7. What Is Strong Enough For A Paper Core

### Core paper-safe claims

1. **The target / evaluation redesign is real and important**
   - `shock_minus_pre`
   - off-hours focus
   - prior-aware evaluation

2. **The strongest credible ECC increment is narrower than a generic multimodal claim**
   - clean `after_hours`
   - `A4`
   - compact `Q&A` semantics

3. **Data limitations matter and shape the method**
   - noisy scheduled time
   - imperfect `A4`
   - audio noise
   - transfer fragility

4. **The best transfer-side idea is reliability-aware abstention**
   - not aggressive routing everywhere
   - not more expert families
   - not more sequence complexity

### Supporting but non-headline material

- aligned / role-aware audio improvements on selected branches
- transfer routing family benchmarks
- gate family searches
- negative sequence results
- complementary-expert negative results

These are useful, but they should support the main story rather than compete with it.

---

## 8. Main Remaining Weaknesses

### 8.1 External validity remains limited

The strongest results are still DJ30 / pilot-scale.

This remains one of the biggest publishability risks.

### 8.2 The method contribution is better, but still not fully “closed”

`reliability-aware abstention` is much cleaner than earlier routing stories, but it is still an extension result with temporal sensitivity and modest effect size.

### 8.3 The transfer story is honest but not fully uniform

The latest diagnostics still show temporal sensitivity:

- not every agreement slice is best under the transfer correction,
- not every disagreement slice is best under `pre_call_market_only`.

So the interpretation is good,
not universal.

### 8.4 Paper-ready reporting is not finished

The repo has the science more than it has the final presentation.

Still needed:

- a compact paper-facing scorecard
- a clean separation of headline vs extension tables
- figure-ready summary for the fixed-split and transfer stories

---

## 9. Recommended Research Priorities From Here

If the goal is to **keep going deeper**, the next steps should be selective.

### Priority A — tighten the current transfer claim, not expand blindly

Best immediate research question:

- **why does the latest-window disagreement slice give a small `qa_benchmark_svd` edge?**

Why this matters:

- it is the clearest current exception to the abstention story,
- and understanding that exception will either strengthen the method claim or reveal where the claim must be narrowed.

### Priority B — build paper-facing evidence, not just more checkpoints

Needed next:

- one concise scorecard covering:
  - strongest overall benchmark
  - strongest fixed-split ECC increment
  - transfer-side abstention extension
  - main caveats / non-claims

This is high leverage because it will make subsequent research more disciplined.

### Priority C — only then decide whether to add one more deeper research branch

If we continue beyond that, the next serious branch should be one of:

1. **temporal-sensitivity diagnosis of the abstention story**, or
2. **broader-sample replication / restricted larger-sample extension**

The repo currently gives less support for returning to:

- heavier sequence architectures
- richer direct complementary experts
- or broader handcrafted feature piling

---

## 10. Bottom Line

The project is in a much healthier position than it was a few days ago.

The current stable hierarchy is:

### strongest overall benchmark
- corrected pooled off-hours benchmark

### strongest paper-safe ECC increment
- **clean `after_hours` + `A4 + compact Q&A semantics`**

### strongest current transfer-side method extension
- **reliability-aware abstention**
  - agreement-supported correction when both router views align
  - abstain to `pre_call_market_only` when reliability evidence is mixed

### safest current research stance
- keep the fixed-split headline unchanged,
- keep transfer as a bounded extension,
- and continue deeper research by tightening the exceptions and robustness of the abstention story rather than expanding model complexity again.
