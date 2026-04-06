#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dj30_qc_utils import write_csv, write_json
from run_afterhours_transfer_expert_selection import summarize_significance
from run_afterhours_transfer_pair_tail_question_framing_diagnostics import (
    distinctive_ngrams,
    text_stats,
)

LEX = "clarify_modeling_lex_factor_pca1"
SEM = "question_lsa4_bi"
HARD = "agreement_pre_only_abstention"
TEXT_COL = "tail_top1_question_text"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose the shared compact-vs-semantic hardest-question veto core and its late-window emergence."
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path(
            "results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/"
            "afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_test_predictions.csv"
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
        default=Path("results/afterhours_transfer_pair_tail_question_shared_core_diagnostics_real"),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_mean(rows: list[dict], key: str) -> float:
    vals = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.mean(vals)) if len(vals) else 0.0


def mean_text_stats(rows: list[dict]) -> dict[str, float]:
    if not rows:
        return {}
    agg = defaultdict(list)
    for row in rows:
        for key, val in text_stats(row[TEXT_COL]).items():
            agg[key].append(val)
    return {key: float(np.mean(vals)) for key, vals in agg.items()}


def summarize_group(rows: list[dict], args: argparse.Namespace) -> dict[str, float | int | str]:
    if not rows:
        return {}
    y = np.asarray([float(r["target"]) for r in rows], dtype=float)
    hard = np.asarray([float(r[HARD]) for r in rows], dtype=float)
    lex = np.asarray([float(r[LEX]) for r in rows], dtype=float)
    sem = np.asarray([float(r[SEM]) for r in rows], dtype=float)
    out = {
        "group": rows[0]["overlap_group"],
        "n": len(rows),
        "year_span": ",".join(sorted({str(r["year"]) for r in rows})),
        "lex_net_mse_gain_vs_hard": float(np.sum((y - hard) ** 2 - (y - lex) ** 2)),
        "sem_net_mse_gain_vs_hard": float(np.sum((y - hard) ** 2 - (y - sem) ** 2)),
        "lex_mean_mse_gain_vs_hard": float(np.mean((y - hard) ** 2 - (y - lex) ** 2)),
        "sem_mean_mse_gain_vs_hard": float(np.mean((y - hard) ** 2 - (y - sem) ** 2)),
        "lex_win_share_vs_hard": float(np.mean(((y - hard) ** 2 - (y - lex) ** 2) > 0.0)),
        "sem_win_share_vs_hard": float(np.mean(((y - hard) ** 2 - (y - sem) ** 2) > 0.0)),
        "mean_qa_pair_count": safe_mean(rows, "qa_pair_count"),
        "mean_direct_early": safe_mean(rows, "tail_top1_direct_early"),
        "mean_evasion": safe_mean(rows, "tail_top1_evasion"),
        "mean_coverage": safe_mean(rows, "tail_top1_coverage"),
        "mean_severity": safe_mean(rows, "tail_top1_severity"),
    }
    out["lex_p_mse_vs_hard"] = summarize_significance(
        y, hard, lex, args.bootstrap_iters, args.perm_iters, args.seed
    )["mse_gain_pvalue"]
    out["sem_p_mse_vs_hard"] = summarize_significance(
        y, hard, sem, args.bootstrap_iters, args.perm_iters, args.seed
    )["mse_gain_pvalue"]
    for key, val in mean_text_stats(rows).items():
        out[f"text__{key}"] = val
    return out


def activation_summary(rows: list[dict], feature_names: list[str], group: str) -> list[dict[str, float | str]]:
    out = []
    for feat in feature_names:
        vals = np.asarray([float(r[feat]) for r in rows], dtype=float)
        out.append(
            {
                "group": group,
                "feature": feat,
                "mean_value": float(np.mean(vals)) if len(vals) else 0.0,
                "nonzero_share": float(np.mean(vals > 0.0)) if len(vals) else 0.0,
                "max_value": float(np.max(vals)) if len(vals) else 0.0,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = list(csv.DictReader(args.predictions_csv.resolve().open()))
    text_lookup = {row["event_key"]: row for row in csv.DictReader(args.text_views_csv.resolve().open())}

    clarify_features = sorted(
        key for key in pred_rows[0].keys() if key.startswith("clarify_modeling__") and not key.endswith("__use_agreed")
    )

    rows = []
    for row in pred_rows:
        key = row["event_key"]
        text_row = text_lookup.get(key, {})
        question_text = text_row.get(TEXT_COL, "")
        agreement = int(float(row.get("agreement", 0)))
        lex_use_agreed = int(float(row.get(f"{LEX}__use_agreed", 1)))
        lex_veto = int(agreement == 1 and lex_use_agreed == 0)
        sem_veto = int(agreement == 1 and abs(float(row[SEM]) - float(row[HARD])) > 1e-15)
        if lex_veto and sem_veto:
            overlap_group = "shared_veto"
        elif sem_veto:
            overlap_group = "sem_only_veto"
        elif lex_veto:
            overlap_group = "lex_only_veto"
        else:
            overlap_group = "other"

        enriched = {
            "event_key": key,
            "ticker": row.get("ticker", ""),
            "year": str(row.get("year", "")),
            "target": float(row["target"]),
            HARD: float(row[HARD]),
            LEX: float(row[LEX]),
            SEM: float(row[SEM]),
            "agreement": agreement,
            "lex_veto": lex_veto,
            "sem_veto": sem_veto,
            "overlap_group": overlap_group,
            "qa_pair_count": float(text_row.get("qa_pair_count", 0.0) or 0.0),
            "tail_top1_direct_early": float(text_row.get("tail_top1_direct_early", 0.0) or 0.0),
            "tail_top1_evasion": float(text_row.get("tail_top1_evasion", 0.0) or 0.0),
            "tail_top1_coverage": float(text_row.get("tail_top1_coverage", 0.0) or 0.0),
            "tail_top1_severity": float(text_row.get("tail_top1_severity", 0.0) or 0.0),
            TEXT_COL: question_text,
        }
        for feat in clarify_features:
            enriched[feat] = float(row.get(feat, 0.0) or 0.0)
        if overlap_group != "other":
            rows.append(enriched)

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["overlap_group"]].append(row)

    year_counts = []
    for year in sorted({row["year"] for row in rows}):
        counter = Counter(row["overlap_group"] for row in rows if row["year"] == year)
        year_counts.append(
            {
                "year": year,
                "shared_veto": counter.get("shared_veto", 0),
                "sem_only_veto": counter.get("sem_only_veto", 0),
                "lex_only_veto": counter.get("lex_only_veto", 0),
            }
        )

    group_summaries = [
        summarize_group(grouped[name], args)
        for name in ["shared_veto", "sem_only_veto", "lex_only_veto"]
        if grouped[name]
    ]

    activation_rows = []
    for name in ["shared_veto", "sem_only_veto", "lex_only_veto"]:
        activation_rows.extend(activation_summary(grouped[name], clarify_features, name))

    shared_texts = [row[TEXT_COL] for row in grouped["shared_veto"]]
    sem_only_texts = [row[TEXT_COL] for row in grouped["sem_only_veto"]]
    lex_only_texts = [row[TEXT_COL] for row in grouped["lex_only_veto"]]

    summary = {
        "config": {
            "predictions_csv": str(args.predictions_csv.resolve()),
            "text_views_csv": str(args.text_views_csv.resolve()),
            "top_k": args.top_k,
        },
        "overlap_counts": {
            "rows_with_any_veto": len(rows),
            "shared_veto_rows": len(grouped["shared_veto"]),
            "sem_only_veto_rows": len(grouped["sem_only_veto"]),
            "lex_only_veto_rows": len(grouped["lex_only_veto"]),
        },
        "year_group_counts": year_counts,
        "group_summaries": group_summaries,
        "shared_vs_sem_only_distinctive_ngrams": distinctive_ngrams(
            shared_texts, sem_only_texts, top_k=args.top_k
        ),
        "sem_only_vs_shared_distinctive_ngrams": distinctive_ngrams(
            sem_only_texts, shared_texts, top_k=args.top_k
        ),
        "shared_vs_lex_only_distinctive_ngrams": distinctive_ngrams(
            shared_texts, lex_only_texts, top_k=min(args.top_k, 10)
        ),
        "group_mean_text_stats": {
            name: mean_text_stats(grouped[name]) for name in ["shared_veto", "sem_only_veto", "lex_only_veto"] if grouped[name]
        },
        "top_shared_veto_events": [
            {
                "event_key": row["event_key"],
                "ticker": row["ticker"],
                "year": row["year"],
                "lex_gain_vs_hard": (row["target"] - row[HARD]) ** 2 - (row["target"] - row[LEX]) ** 2,
                "sem_gain_vs_hard": (row["target"] - row[HARD]) ** 2 - (row["target"] - row[SEM]) ** 2,
                TEXT_COL: row[TEXT_COL],
            }
            for row in sorted(
                grouped["shared_veto"],
                key=lambda r: (r["target"] - r[HARD]) ** 2 - (r["target"] - r[LEX]) ** 2,
                reverse=True,
            )
        ],
        "top_sem_only_veto_events": [
            {
                "event_key": row["event_key"],
                "ticker": row["ticker"],
                "year": row["year"],
                "sem_gain_vs_hard": (row["target"] - row[HARD]) ** 2 - (row["target"] - row[SEM]) ** 2,
                TEXT_COL: row[TEXT_COL],
            }
            for row in sorted(
                grouped["sem_only_veto"],
                key=lambda r: (r["target"] - r[HARD]) ** 2 - (r["target"] - r[SEM]) ** 2,
                reverse=True,
            )
        ],
        "lexical_only_tail_events": [
            {
                "event_key": row["event_key"],
                "ticker": row["ticker"],
                "year": row["year"],
                "lex_gain_vs_hard": (row["target"] - row[HARD]) ** 2 - (row["target"] - row[LEX]) ** 2,
                TEXT_COL: row[TEXT_COL],
            }
            for row in sorted(
                grouped["lex_only_veto"],
                key=lambda r: (r["target"] - r[HARD]) ** 2 - (r["target"] - r[LEX]) ** 2,
                reverse=True,
            )
        ],
    }

    write_json(output_dir / "afterhours_transfer_pair_tail_question_shared_core_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_shared_core_rows.csv", rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_shared_core_year_counts.csv", year_counts)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_shared_core_activation_summary.csv", activation_rows)
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_question_shared_core_shared_vs_sem_only_ngrams.csv",
        summary["shared_vs_sem_only_distinctive_ngrams"],
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_question_shared_core_sem_only_vs_shared_ngrams.csv",
        summary["sem_only_vs_shared_distinctive_ngrams"],
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_question_shared_core_shared_vs_lex_only_ngrams.csv",
        summary["shared_vs_lex_only_distinctive_ngrams"],
    )


if __name__ == "__main__":
    main()
