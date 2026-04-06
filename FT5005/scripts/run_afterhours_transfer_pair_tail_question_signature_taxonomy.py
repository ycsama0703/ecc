#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dj30_qc_utils import write_csv, write_json
from run_afterhours_transfer_pair_tail_question_framing_diagnostics import text_stats

LEX = "clarify_modeling_lex_factor_pca1"
SEM = "question_lsa4_bi"
HARD = "agreement_pre_only_abstention"
TEXT_COL = "tail_top1_question_text"
FAMILY_PREFIXES = {
    "clarify": "clarify_modeling__",
    "quant": "quant_bridge__",
    "structural": "structural_probe__",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Characterize the hardest-question shared core and semantic tail by compact lexical-family signatures."
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
        default=Path("results/afterhours_transfer_pair_tail_question_signature_taxonomy_real"),
    )
    return parser.parse_args()


def signature(scores: dict[str, float]) -> str:
    active = [name for name, val in scores.items() if val > 0.0]
    return "+".join(active) if active else "none"


def safe_mean(rows: list[dict], key: str) -> float:
    vals = np.asarray([float(row[key]) for row in rows], dtype=float)
    return float(np.mean(vals)) if len(vals) else 0.0


def summarize_cell(rows: list[dict], overlap_group: str, sig: str) -> dict[str, float | int | str]:
    y = np.asarray([float(r["target"]) for r in rows], dtype=float)
    hard = np.asarray([float(r[HARD]) for r in rows], dtype=float)
    lex = np.asarray([float(r[LEX]) for r in rows], dtype=float)
    sem = np.asarray([float(r[SEM]) for r in rows], dtype=float)
    return {
        "overlap_group": overlap_group,
        "signature": sig,
        "n": len(rows),
        "years": ",".join(sorted({str(r["year"]) for r in rows})),
        "lex_net_mse_gain_vs_hard": float(np.sum((y - hard) ** 2 - (y - lex) ** 2)),
        "sem_net_mse_gain_vs_hard": float(np.sum((y - hard) ** 2 - (y - sem) ** 2)),
        "lex_win_share_vs_hard": float(np.mean(((y - hard) ** 2 - (y - lex) ** 2) > 0.0)),
        "sem_win_share_vs_hard": float(np.mean(((y - hard) ** 2 - (y - sem) ** 2) > 0.0)),
        "mean_qa_pair_count": safe_mean(rows, "qa_pair_count"),
        "mean_direct_early": safe_mean(rows, "tail_top1_direct_early"),
        "mean_evasion": safe_mean(rows, "tail_top1_evasion"),
        "mean_coverage": safe_mean(rows, "tail_top1_coverage"),
        "mean_severity": safe_mean(rows, "tail_top1_severity"),
        "mean_numeric_token_share": safe_mean(rows, "numeric_token_share"),
        "mean_hedge_share": safe_mean(rows, "hedge_share"),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_rows = list(csv.DictReader(args.predictions_csv.resolve().open()))
    text_lookup = {row["event_key"]: row for row in csv.DictReader(args.text_views_csv.resolve().open())}
    family_cols = {
        name: [key for key in pred_rows[0].keys() if key.startswith(prefix)]
        for name, prefix in FAMILY_PREFIXES.items()
    }

    rows = []
    for row in pred_rows:
        key = row["event_key"]
        text_row = text_lookup.get(key, {})
        question_text = text_row.get(TEXT_COL, "")
        agreement = int(float(row.get("agreement", 0)))
        lex_veto = int(agreement == 1 and int(float(row.get(f"{LEX}__use_agreed", 1))) == 0)
        sem_veto = int(agreement == 1 and abs(float(row[SEM]) - float(row[HARD])) > 1e-15)
        if lex_veto and sem_veto:
            overlap_group = "shared_veto"
        elif sem_veto:
            overlap_group = "sem_only_veto"
        elif lex_veto:
            overlap_group = "lex_only_veto"
        else:
            continue

        scores = {
            name: float(sum(float(row[col]) for col in cols))
            for name, cols in family_cols.items()
        }
        stats = text_stats(question_text)
        rows.append(
            {
                "event_key": key,
                "ticker": row.get("ticker", ""),
                "year": str(row.get("year", "")),
                "overlap_group": overlap_group,
                "family_signature": signature(scores),
                "target": float(row["target"]),
                HARD: float(row[HARD]),
                LEX: float(row[LEX]),
                SEM: float(row[SEM]),
                "qa_pair_count": float(text_row.get("qa_pair_count", 0.0) or 0.0),
                "tail_top1_direct_early": float(text_row.get("tail_top1_direct_early", 0.0) or 0.0),
                "tail_top1_evasion": float(text_row.get("tail_top1_evasion", 0.0) or 0.0),
                "tail_top1_coverage": float(text_row.get("tail_top1_coverage", 0.0) or 0.0),
                "tail_top1_severity": float(text_row.get("tail_top1_severity", 0.0) or 0.0),
                TEXT_COL: question_text,
                **scores,
                **stats,
            }
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["overlap_group"], row["family_signature"])].append(row)

    cell_rows = []
    for key in sorted(grouped):
        cell_rows.append(summarize_cell(grouped[key], key[0], key[1]))

    counts_by_group = []
    for overlap_group in ["shared_veto", "sem_only_veto", "lex_only_veto"]:
        counter = Counter(row["family_signature"] for row in rows if row["overlap_group"] == overlap_group)
        total = sum(counter.values())
        for sig, count in sorted(counter.items()):
            counts_by_group.append(
                {
                    "overlap_group": overlap_group,
                    "family_signature": sig,
                    "count": count,
                    "share_within_group": float(count / total) if total else 0.0,
                }
            )

    year_signature_rows = []
    for year in sorted({row["year"] for row in rows}):
        counter = Counter(
            (row["overlap_group"], row["family_signature"]) for row in rows if row["year"] == year
        )
        for (overlap_group, sig), count in sorted(counter.items()):
            year_signature_rows.append(
                {
                    "year": year,
                    "overlap_group": overlap_group,
                    "family_signature": sig,
                    "count": count,
                }
            )

    examples = defaultdict(list)
    for row in rows:
        gain_lex = (row["target"] - row[HARD]) ** 2 - (row["target"] - row[LEX]) ** 2
        gain_sem = (row["target"] - row[HARD]) ** 2 - (row["target"] - row[SEM]) ** 2
        examples[(row["overlap_group"], row["family_signature"])].append(
            {
                "event_key": row["event_key"],
                "ticker": row["ticker"],
                "year": row["year"],
                "lex_gain_vs_hard": gain_lex,
                "sem_gain_vs_hard": gain_sem,
                "qa_pair_count": row["qa_pair_count"],
                "tail_top1_direct_early": row["tail_top1_direct_early"],
                "tail_top1_evasion": row["tail_top1_evasion"],
                TEXT_COL: row[TEXT_COL],
            }
        )

    summary = {
        "config": {
            "predictions_csv": str(args.predictions_csv.resolve()),
            "text_views_csv": str(args.text_views_csv.resolve()),
        },
        "main_pattern": {
            "shared_veto_signature_counts": {
                row["family_signature"]: row["count"]
                for row in counts_by_group
                if row["overlap_group"] == "shared_veto"
            },
            "sem_only_veto_signature_counts": {
                row["family_signature"]: row["count"]
                for row in counts_by_group
                if row["overlap_group"] == "sem_only_veto"
            },
            "lex_only_veto_signature_counts": {
                row["family_signature"]: row["count"]
                for row in counts_by_group
                if row["overlap_group"] == "lex_only_veto"
            },
        },
        "cell_summaries": cell_rows,
        "counts_by_group": counts_by_group,
        "year_signature_rows": year_signature_rows,
        "key_examples": {
            f"{group}__{sig}": rows_[:5]
            for (group, sig), rows_ in examples.items()
        },
    }

    write_json(output_dir / "afterhours_transfer_pair_tail_question_signature_taxonomy_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_signature_taxonomy_rows.csv", rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_signature_taxonomy_cell_summary.csv", cell_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_signature_taxonomy_counts.csv", counts_by_group)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_signature_taxonomy_year_counts.csv", year_signature_rows)


if __name__ == "__main__":
    main()
