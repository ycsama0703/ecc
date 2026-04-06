# Experiment Matrix

## Purpose

Define ALL experiments to run.

Avoid uncontrolled expansion.

---

## Main Experiments

| ID | Name | Type |
|----|------|------|
| E1 | Market Only | Benchmark |
| E2 | Market + Controls | Benchmark |
| E3 | FinBERT | Benchmark |
| E4 | MR-QA | Benchmark |
| E5 | Fusion | Benchmark |
| E6 | Ours (Minimax) | Main |
| E7 | Ours + Abstention | Main |

---

## Ablation

| ID | Name |
|----|------|
| A1 | No Gate |
| A2 | Observed Gate |
| A3 | Minimax Gate |
| A4 | Minimax + Abstention |
| A5 | Proxy Permutation |

---

## Mechanism Experiments

| ID | Name |
|----|------|
| M1 | Reliability vs Error |
| M2 | Proxy Noise Injection |
| M3 | Audio vs Reliability Bucket |

---

## Robustness

| ID | Name |
|----|------|
| R1 | Time Window Variation |
| R2 | Sample Variation |
| R3 | Temporal Split Variation |

---

## Rules

- DO NOT add new experiments without approval
- Every experiment must map to a paper section