#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dj30_qc_utils import write_csv, write_json

MODEL = "tail_question_top1_lsa"
HARD = "agreement_pre_only_abstention"
TOKEN_RE = re.compile(r"[A-Za-z0-9']+")

FIRST_PERSON = {"i", "we", "me", "us", "our", "ours", "my", "mine"}
SECOND_PERSON = {"you", "your", "yours", "you're", "youve", "you'd", "youll"}
MODALS = {"can", "could", "would", "should", "may", "might", "will"}
HEDGES = {"maybe", "just", "think", "guess", "kind", "sort", "wondering"}
FOLLOWUP_MARKERS = [
    "follow up",
    "to clarify",
    "can you talk",
    "give us color",
    "give any color",
    "how do i think",
    "pick up",
    "continue that",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the local framing pocket behind the hardest-question transfer signal."
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path(
            "results/afterhours_transfer_pair_tail_text_benchmark_lsa4_real/"
            "afterhours_transfer_pair_tail_text_benchmark_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--text-views-csv",
        type=Path,
        default=Path("results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_pair_tail_question_framing_diagnostics_real"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def ngrams(tokens: list[str], n: int) -> list[str]:
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def text_stats(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    lower = text.lower()
    numeric = sum(any(ch.isdigit() for ch in tok) for tok in tokens)
    return {
        "token_count": float(len(tokens)),
        "sentence_like_count": float(sum(ch in '.?!' for ch in text) or 0),
        "question_mark_count": float(text.count('?')),
        "numeric_token_share": float(numeric / len(tokens)) if tokens else 0.0,
        "first_person_share": float(sum(tok in FIRST_PERSON for tok in tokens) / len(tokens)) if tokens else 0.0,
        "second_person_share": float(sum(tok in SECOND_PERSON for tok in tokens) / len(tokens)) if tokens else 0.0,
        "modal_share": float(sum(tok in MODALS for tok in tokens) / len(tokens)) if tokens else 0.0,
        "hedge_share": float(sum(tok in HEDGES for tok in tokens) / len(tokens)) if tokens else 0.0,
        "followup_marker_count": float(sum(lower.count(marker) for marker in FOLLOWUP_MARKERS)),
    }


def mean_stats(rows, field: str) -> dict[str, float]:
    vals = defaultdict(list)
    for row in rows:
        stats = text_stats(row[field])
        for k, v in stats.items():
            vals[k].append(v)
    return {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}


def distinctive_ngrams(texts_a: list[str], texts_b: list[str], *, top_k: int) -> list[dict[str, float | str]]:
    ca = Counter()
    cb = Counter()
    total_a = 0
    total_b = 0
    for text in texts_a:
        toks = tokenize(text)
        grams = toks + ngrams(toks, 2) + ngrams(toks, 3)
        ca.update(grams)
        total_a += len(grams)
    for text in texts_b:
        toks = tokenize(text)
        grams = toks + ngrams(toks, 2) + ngrams(toks, 3)
        cb.update(grams)
        total_b += len(grams)
    vocab = set(ca) | set(cb)
    rows = []
    for gram in vocab:
        if len(gram) <= 1:
            continue
        a = ca[gram]
        b = cb[gram]
        if a + b < 2:
            continue
        score = math.log((a + 0.5) / (total_a + 0.5 * len(vocab))) - math.log((b + 0.5) / (total_b + 0.5 * len(vocab)))
        rows.append({
            "ngram": gram,
            "count_a": a,
            "count_b": b,
            "log_odds": float(score),
        })
    return sorted(rows, key=lambda r: r["log_odds"], reverse=True)[:top_k]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = list(csv.DictReader(args.predictions_csv.resolve().open()))
    text_lookup = {r["event_key"]: r for r in csv.DictReader(args.text_views_csv.resolve().open())}

    rows = []
    for row in pred_rows:
        if int(float(row.get("agreement", 0))) != 1:
            continue
        if int(float(row.get(f"{MODEL}__use_agreed", 0))) != 0:
            continue
        key = row["event_key"]
        text = text_lookup[key]["tail_top1_question_text"]
        y = float(row["target"])
        hard = float(row[HARD])
        pred = float(row[MODEL])
        gain = (y - hard) ** 2 - (y - pred) ** 2
        label = "positive" if gain > 0 else "negative" if gain < 0 else "flat"
        rows.append({
            "event_key": key,
            "ticker": row.get("ticker", ""),
            "year": row.get("year", ""),
            "mse_gain_vs_hard": gain,
            "label": label,
            "tail_top1_question_text": text,
            **text_stats(text),
        })

    positive = [r for r in rows if r["label"] == "positive"]
    negative = [r for r in rows if r["label"] == "negative"]
    flat = [r for r in rows if r["label"] == "flat"]

    summary = {
        "counts": {
            "agreement_veto_rows": len(rows),
            "positive_veto_rows": len(positive),
            "negative_veto_rows": len(negative),
            "flat_veto_rows": len(flat),
        },
        "mean_stats": {
            "all_veto": mean_stats(rows, "tail_top1_question_text"),
            "positive_veto": mean_stats(positive, "tail_top1_question_text"),
            "negative_veto": mean_stats(negative, "tail_top1_question_text"),
            "flat_veto": mean_stats(flat, "tail_top1_question_text"),
        },
        "positive_vs_negative_distinctive_ngrams": distinctive_ngrams(
            [r["tail_top1_question_text"] for r in positive],
            [r["tail_top1_question_text"] for r in negative],
            top_k=args.top_k,
        ),
        "negative_vs_positive_distinctive_ngrams": distinctive_ngrams(
            [r["tail_top1_question_text"] for r in negative],
            [r["tail_top1_question_text"] for r in positive],
            top_k=args.top_k,
        ),
        "top_positive_events": [
            {k: r[k] for k in ["event_key", "ticker", "year", "mse_gain_vs_hard", "label", "tail_top1_question_text"]}
            for r in sorted(positive, key=lambda r: r["mse_gain_vs_hard"], reverse=True)[:8]
        ],
        "top_negative_events": [
            {k: r[k] for k in ["event_key", "ticker", "year", "mse_gain_vs_hard", "label", "tail_top1_question_text"]}
            for r in sorted(negative, key=lambda r: r["mse_gain_vs_hard"])[:8]
        ],
    }

    write_json(output_dir / "afterhours_transfer_pair_tail_question_framing_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_framing_rows.csv", rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_framing_positive_ngrams.csv", summary["positive_vs_negative_distinctive_ngrams"])
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_framing_negative_ngrams.csv", summary["negative_vs_positive_distinctive_ngrams"])


if __name__ == "__main__":
    main()
