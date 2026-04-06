# Stage Evaluation Consolidation 2026-03-23

## Why This Note Exists

The repo now has enough checkpoints that the main risk is no longer lack of activity.

The main risk is the opposite: letting too many partially positive directions sit at the same level, and then losing the difference between:

- what is already paper-safe,
- what is a good secondary method contribution,
- what is currently exploratory but still worth keeping,
- and what should now be actively demoted.

This note is meant to support a **stage evaluation** rather than another open-ended research round.

---

## 1. One-paragraph verdict

As of `2026-03-23`, the project has a **real and defensible empirical core**. The strongest fixed-split story is now stable: under noisy timing, clean `after_hours` calls contain usable incremental ECC signal, and the most credible part of that signal currently comes from **`A4` observability/alignment plus compact `Q&A` semantics** on top of strong market priors. The transfer-side story is also substantially cleaner than before: the strongest current method reading is **reliability-aware abstention**, not more routing complexity. In parallel, the new hardest-question line has become a meaningful exploratory finding: the strongest local signal seems to live in a **question-centric, non-structural local representation**, though this branch is still too narrow to be treated as a main method claim. Overall, this is no longer an unfocused exploration repo. It is now a repo with **one stable mainline, one coherent secondary extension, and one promising exploratory side signal**.

---

## 2. What is currently locked

### 2.1 Problem framing

The safe framing is now:

- **off-hours earnings-call volatility-shock prediction under noisy timing**;
- target: `shock_minus_pre` rather than raw post-call volatility level;
- evaluation against strong pre-call / market priors;
- `A4` treated as noisy observability or alignment supervision rather than as gold timestamps.

This framing is clearly stronger than:

- generic multimodal ECC prediction,
- heavy sequence-first modeling,
- or the vague claim that ``more modalities help.''

### 2.2 Main fixed-split paper-safe line

The current strongest and safest incremental ECC claim is:

- **clean `after_hours` + `A4 + compact Q&A semantics`**
- with the current best fixed-split semantic bottleneck still at **`lsa=64`**.

The key fixed-split reference points remain:

- `pre_call_market_only`: `R² ≈ 0.9174`
- `pre_call_market_plus_controls`: `R² ≈ 0.9194`
- `pre_call_market_plus_a4_plus_qna_lsa`: `R² ≈ 0.9271`
- `pre_call_market_plus_controls_plus_a4_plus_qna_lsa`: `R² ≈ 0.9347`

This line is narrow, interpretable, and aligned with the actual data quality constraints.

### 2.3 Strongest broad benchmark anchor

The repo still contains a very strong corrected pooled off-hours benchmark:

- `prior_only`: `R² ≈ 0.1978`
- `residual_structured_only`: `R² ≈ 0.9127`

This is important as a benchmark anchor and as evidence that the corrected target is real. But it remains more **market-dominated** than the narrower clean `after_hours` ECC-increment line.

---

## 3. What should count as the current contribution hierarchy

### Layer A — main contribution

**Credible ECC increment under noisy timing is narrow and structured.**

The current paper-safe main claim is not that generic multimodal ECC modeling wins, but that under noisy timing, a carefully filtered `after_hours` setting yields usable incremental ECC signal, and the cleanest current evidence points to **`A4` plus compact `Q&A` semantics**.

### Layer B — secondary method contribution

**Transfer requires reliability-aware abstention.**

The strongest transfer-side interpretation is now methodologically clean:

- agreement among retained views supports selective correction,
- disagreement is safest under fallback to the market baseline.

This is a stronger and cleaner story than earlier gate-search or router-complexity narratives.

### Layer C — exploratory but meaningful signal finding

**The hardest analyst question may carry a sharper local signal than the full exchange.**

The strongest local exploratory route remains:

- `question_lsa4_bi ≈ 0.99865468`
- and the masked non-structural version is identical:
  - `question_mask_struct_lsa4_bi ≈ 0.99865468`

That is above:

- `hard_abstention ≈ 0.99864017`
- `geometry_only ≈ 0.99863878`

But the effect is still small, and this branch remains exploratory.

---

## 4. What is now demoted and should stop claiming headline status

The repo now has enough evidence to actively demote several paths.

### 4.1 Heavy sequence modeling

Status:

- **not headline material**
- at best extension / appendix material

Reason:

- sequence-heavy routes repeatedly underperform the cleaner fixed-split semantic line.

### 4.2 Generic multimodal stacking

Status:

- **not a safe main story**

Reason:

- the current evidence does not support a broad claim that simply adding more modalities robustly helps.

### 4.3 Audio as the primary contribution

Status:

- **supporting extension only**

Reason:

- aligned audio became more concrete, but it does not robustly improve the strongest fixed-split semantic line.

### 4.4 More gate / router complexity as a contribution in itself

Status:

- **demoted**

Reason:

- the strongest transfer-side reading is now about trust, abstention, and reliability; not about increasingly elaborate routing shells.

### 4.5 Hardest-question branch as a replacement for the mainline

Status:

- **do not promote yet**

Reason:

- this is currently a narrow local signal, not a replacement for the main fixed-split storyline.

---

## 5. Stage evaluation scorecard

| Dimension | Current status | Reading |
|---|---|---|
| Problem framing | Strong | `shock_minus_pre` + noisy timing + prior-aware evaluation is now coherent and well justified |
| Fixed-split empirical core | Strong | clean `after_hours` + `A4 + Q&A` is stable and interpretable |
| Transfer-side method story | Medium to strong | abstention story is coherent, but gains are still modest |
| Exploratory signal discovery | Medium | hardest-question signal is real and interesting, but still narrow |
| Novelty clarity | Medium | stronger than before, but still needs disciplined framing |
| External validity | Weak to medium | still pilot-scale, still a live risk |
| Storyline discipline | Stronger than before | the repo is much more hierarchical now |

### Bottom-line stage evaluation

If we had to evaluate the project **today**, the fairest reading would be:

> **The project is viable and has a credible mainline.**
> It is no longer in an uncontrolled exploration phase.
> But it is also **not yet at a final paper-freeze stage**.

A reasonable stage label would be:

## **consolidated mainline + selective high-value extension phase**

That is a healthier place than “broad exploration,” but it is not yet “final polished contribution stack.”

---

## 6. What the current repo most likely supports in a phase review

### What we can say confidently

- the project has a stable core framing;
- the main fixed-split result is not a fluke and not merely generic market memory;
- the transfer story has improved from complexity search to a cleaner abstention-based interpretation;
- the repo has accumulated enough negative evidence to justify why several earlier routes are no longer center stage.

### What we should not overstate

- we should not claim that transfer semantics robustly beat the strongest market baseline everywhere;
- we should not claim that audio is a mainline contribution;
- we should not claim that the hardest-question line is already a mature standalone method;
- we should not pretend the pilot-scale setting has already solved external-validity concerns.

---

## 7. Recommendation for the next stage

The phase-evaluation implication is clear:

### 7.1 Freeze the hierarchy

From this point onward, the default hierarchy should be:

1. **mainline:** clean `after_hours` + `A4 + compact Q&A semantics`
2. **secondary method extension:** reliability-aware abstention
3. **exploratory research branch:** hardest-question non-structural local signal

### 7.2 Stop broad exploration

The repo should stop spending major effort on:

- more sequence variants,
- broader multimodal piling,
- more router families for their own sake,
- or replacing the mainline with the hardest-question branch.

### 7.3 Evaluate the next step with a gate

If the project enters another research round, it should do so under a simple gate:

- either pursue a **high-value confirmation / scale-up** of the mainline,
- or pursue one **tightly controlled exploratory branch** that has a realistic chance of becoming a stronger contribution.

At the moment, the most sensible exploratory branch remains the hardest-question line, but only if it stays compact and does not start inflating the outer shell again.

---

## 8. Final take

The most important change is not that the repo found a single dramatic new score. The most important change is that it now knows the difference between:

- what is stable,
- what is promising,
- and what should be left behind.

That is exactly what a good stage evaluation should produce.
