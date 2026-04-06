#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json
from run_afterhours_transfer_expert_selection import summarize_significance
from run_structured_baselines import metrics

LEX = "clarify_modeling_lex_factor_pca1"
SEM = "question_lsa4_bi"
HARD = "agreement_pre_only_abstention"
PRE = "residual_pre_call_market_only"

SIDE_PANEL_FEATURES = [
    "a4_strict_row_share",
    "a4_strict_high_conf_share",
    "revenue_surprise_pct",
]
SIDE_FEATURES = [
    "qa_pair_count",
    "qna_word_count",
    "answer_to_question_word_ratio",
]
SIDE_QA = [
    "qa_bench_direct_early_score_mean",
    "qa_bench_evasion_score_mean",
    "qa_bench_coverage_mean",
    "qa_bench_direct_answer_share",
    "qa_bench_nonresponse_share",
]
PROFILE_KEYS = [
    "qa_pair_count",
    "qa_bench_direct_early_score_mean",
    "qa_bench_evasion_score_mean",
    "qa_bench_coverage_mean",
    "a4_strict_row_share",
    "a4_strict_high_conf_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm how the compact clarificatory lexical factor relates to the richer hardest-question semantic route."
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
        "--panel-csv",
        type=Path,
        default=Path("results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv"),
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("results/features_real/event_text_audio_features.csv"),
    )
    parser.add_argument(
        "--qa-csv",
        type=Path,
        default=Path("results/qa_benchmark_features_v2_real/qa_benchmark_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_pair_tail_question_lexical_confirmation_real"),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_lookup(path: Path) -> dict[str, dict[str, str]]:
    return {row["event_key"]: row for row in load_csv_rows(path.resolve()) if row.get("event_key")}


def add_side_inputs(rows, panel_lookup, features_lookup, qa_lookup):
    coverage = {"rows": len(rows), "with_panel": 0, "with_features": 0, "with_qa": 0, "with_all": 0}
    for row in rows:
        key = row["event_key"]
        p = panel_lookup.get(key)
        f = features_lookup.get(key)
        q = qa_lookup.get(key)
        if p:
            coverage["with_panel"] += 1
            for name in SIDE_PANEL_FEATURES:
                row[name] = safe_float(p.get(name, ""))
        if f:
            coverage["with_features"] += 1
            for name in SIDE_FEATURES:
                row[name] = safe_float(f.get(name, ""))
        if q:
            coverage["with_qa"] += 1
            for name in SIDE_QA:
                row[name] = safe_float(q.get(name, ""))
        if p and f and q:
            coverage["with_all"] += 1
    return coverage


def arr(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=float)


def safe_mean(rows, key):
    vals = np.asarray(
        [
            float(v) if v not in (None, "") else np.nan
            for v in (r.get(key, np.nan) for r in rows)
        ],
        dtype=float,
    )
    if not len(vals):
        return 0.0
    if np.all(np.isnan(vals)):
        return float("nan")
    return float(np.nanmean(vals))


def add_diagnostics(rows):
    for row in rows:
        y = float(row["target"])
        pre = float(row[PRE])
        hard = float(row[HARD])
        lex = float(row[LEX])
        sem = float(row[SEM])
        row["hard_se"] = (y - hard) ** 2
        row["lex_se"] = (y - lex) ** 2
        row["sem_se"] = (y - sem) ** 2
        row["pre_se"] = (y - pre) ** 2
        row["lex_gain_vs_hard"] = row["hard_se"] - row["lex_se"]
        row["sem_gain_vs_hard"] = row["hard_se"] - row["sem_se"]
        row["lex_gain_vs_pre"] = row["pre_se"] - row["lex_se"]
        row["sem_gain_vs_pre"] = row["pre_se"] - row["sem_se"]
        row["agreement"] = int(float(row.get("agreement", 0)))
        row["lex_use_agreed"] = int(float(row.get(f"{LEX}__use_agreed", 1)))
        row["lex_veto"] = int(row["agreement"] == 1 and row["lex_use_agreed"] == 0)
        row["sem_veto"] = int(row["agreement"] == 1 and abs(sem - hard) > 1e-15)
        if row["lex_veto"] and row["sem_veto"]:
            row["overlap_group"] = "shared_veto"
        elif row["sem_veto"]:
            row["overlap_group"] = "sem_only_veto"
        elif row["lex_veto"]:
            row["overlap_group"] = "lex_only_veto"
        elif row["agreement"] == 1:
            row["overlap_group"] = "both_keep_agreed"
        else:
            row["overlap_group"] = "disagreement"


def summarize_route(name, rows, pred_key, hard_key, args):
    y = arr(rows, "target")
    pred = arr(rows, pred_key)
    hard = arr(rows, hard_key)
    out = {
        "name": name,
        "n": len(rows),
        "r2": metrics(y, pred)["r2"],
        "rmse": metrics(y, pred)["rmse"],
        "mae": metrics(y, pred)["mae"],
        "net_mse_gain_vs_hard": float(np.sum((y - hard) ** 2 - (y - pred) ** 2)),
        "mean_mse_gain_vs_hard": float(np.mean((y - hard) ** 2 - (y - pred) ** 2)),
        "win_share_vs_hard": float(np.mean(((y - hard) ** 2 - (y - pred) ** 2) > 0.0)),
    }
    out["p_mse_vs_hard"] = summarize_significance(
        y, hard, pred, args.bootstrap_iters, args.perm_iters, args.seed
    )["mse_gain_pvalue"]
    return out


def summarize_group(name, rows, args):
    if not rows:
        return None
    y = arr(rows, "target")
    hard = arr(rows, HARD)
    lex = arr(rows, LEX)
    sem = arr(rows, SEM)
    out = {
        "group": name,
        "n": len(rows),
        "lex_net_mse_gain_vs_hard": float(np.sum(arr(rows, "lex_gain_vs_hard"))),
        "sem_net_mse_gain_vs_hard": float(np.sum(arr(rows, "sem_gain_vs_hard"))),
        "lex_mean_mse_gain_vs_hard": float(np.mean(arr(rows, "lex_gain_vs_hard"))),
        "sem_mean_mse_gain_vs_hard": float(np.mean(arr(rows, "sem_gain_vs_hard"))),
        "lex_win_share_vs_hard": float(np.mean(arr(rows, "lex_gain_vs_hard") > 0.0)),
        "sem_win_share_vs_hard": float(np.mean(arr(rows, "sem_gain_vs_hard") > 0.0)),
        "lex_use_agreed_share": float(np.mean(arr(rows, "lex_use_agreed"))),
        "sem_veto_share": float(np.mean(arr(rows, "sem_veto"))),
    }
    for key in PROFILE_KEYS:
        out[f"mean__{key}"] = safe_mean(rows, key)
    out["lex_p_mse_vs_hard"] = summarize_significance(
        y, hard, lex, args.bootstrap_iters, args.perm_iters, args.seed
    )["mse_gain_pvalue"]
    out["sem_p_mse_vs_hard"] = summarize_significance(
        y, hard, sem, args.bootstrap_iters, args.perm_iters, args.seed
    )["mse_gain_pvalue"]
    return out


def top_events(rows, gain_key, limit=8, reverse=True):
    ordered = sorted(rows, key=lambda r: float(r[gain_key]), reverse=reverse)
    keep = []
    for r in ordered[:limit]:
        keep.append(
            {
                "event_key": r["event_key"],
                "ticker": r.get("ticker", ""),
                "year": int(float(r["year"])) if r.get("year") not in (None, "") else None,
                "overlap_group": r["overlap_group"],
                "lex_veto": int(r["lex_veto"]),
                "sem_veto": int(r["sem_veto"]),
                "lex_gain_vs_hard": float(r["lex_gain_vs_hard"]),
                "sem_gain_vs_hard": float(r["sem_gain_vs_hard"]),
            }
        )
    return keep


def main() -> None:
    args = parse_args()
    csv.field_size_limit(sys.maxsize)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.predictions_csv.resolve().open()))
    panel_lookup = load_lookup(args.panel_csv)
    features_lookup = load_lookup(args.features_csv)
    qa_lookup = load_lookup(args.qa_csv)
    coverage = add_side_inputs(rows, panel_lookup, features_lookup, qa_lookup)
    add_diagnostics(rows)

    lex_veto_rows = [r for r in rows if int(r["lex_veto"]) == 1]
    sem_veto_rows = [r for r in rows if int(r["sem_veto"]) == 1]
    shared_veto_rows = [r for r in rows if r["overlap_group"] == "shared_veto"]
    sem_only_rows = [r for r in rows if r["overlap_group"] == "sem_only_veto"]
    lex_only_rows = [r for r in rows if r["overlap_group"] == "lex_only_veto"]

    overlap = {
        "test_rows": len(rows),
        "agreement_rows": int(sum(int(r["agreement"]) for r in rows)),
        "disagreement_rows": int(sum(1 - int(r["agreement"]) for r in rows)),
        "lex_veto_rows": len(lex_veto_rows),
        "sem_veto_rows": len(sem_veto_rows),
        "shared_veto_rows": len(shared_veto_rows),
        "sem_only_veto_rows": len(sem_only_rows),
        "lex_only_veto_rows": len(lex_only_rows),
        "lex_precision_to_sem_veto": float(len(shared_veto_rows) / len(lex_veto_rows)) if lex_veto_rows else 0.0,
        "lex_recall_of_sem_veto": float(len(shared_veto_rows) / len(sem_veto_rows)) if sem_veto_rows else 0.0,
    }

    group_order = [
        "shared_veto",
        "sem_only_veto",
        "lex_only_veto",
        "both_keep_agreed",
        "disagreement",
    ]
    group_rows = []
    for name in group_order:
        subset = [r for r in rows if r["overlap_group"] == name]
        summary = summarize_group(name, subset, args)
        if summary is not None:
            group_rows.append(summary)

    year_rows = []
    for year in sorted({r["year"] for r in rows}):
        subset = [r for r in rows if r["year"] == year]
        year_rows.append(
            {
                "year": year,
                "n": len(subset),
                "lex_net_mse_gain_vs_hard": float(np.sum(arr(subset, "lex_gain_vs_hard"))),
                "sem_net_mse_gain_vs_hard": float(np.sum(arr(subset, "sem_gain_vs_hard"))),
                "lex_veto_rows": int(sum(int(r["lex_veto"]) for r in subset)),
                "sem_veto_rows": int(sum(int(r["sem_veto"]) for r in subset)),
                "shared_veto_rows": int(sum(r["overlap_group"] == "shared_veto" for r in subset)),
            }
        )

    route_summary = {
        "hard": summarize_route("hard", rows, HARD, HARD, args),
        "lex": summarize_route("lex", rows, LEX, HARD, args),
        "sem": summarize_route("sem", rows, SEM, HARD, args),
        "pre": summarize_route("pre", rows, PRE, HARD, args),
    }

    event_rows = []
    for r in rows:
        if r["overlap_group"] in {"shared_veto", "sem_only_veto", "lex_only_veto"}:
            event_rows.append(
                {
                    "event_key": r["event_key"],
                    "ticker": r.get("ticker", ""),
                    "year": r.get("year", ""),
                    "overlap_group": r["overlap_group"],
                    "agreement": int(r["agreement"]),
                    "lex_veto": int(r["lex_veto"]),
                    "sem_veto": int(r["sem_veto"]),
                    "lex_gain_vs_hard": float(r["lex_gain_vs_hard"]),
                    "sem_gain_vs_hard": float(r["sem_gain_vs_hard"]),
                    "qa_pair_count": safe_float(r.get("qa_pair_count", "")),
                    "qa_bench_direct_early_score_mean": safe_float(r.get("qa_bench_direct_early_score_mean", "")),
                    "qa_bench_evasion_score_mean": safe_float(r.get("qa_bench_evasion_score_mean", "")),
                    "a4_strict_row_share": safe_float(r.get("a4_strict_row_share", "")),
                }
            )

    summary = {
        "config": {
            "predictions_csv": str(args.predictions_csv.resolve()),
            "panel_csv": str(args.panel_csv.resolve()),
            "features_csv": str(args.features_csv.resolve()),
            "qa_csv": str(args.qa_csv.resolve()),
        },
        "coverage": coverage,
        "route_summary": route_summary,
        "overlap": overlap,
        "group_overview": group_rows,
        "year_overview": year_rows,
        "top_positive_lex_events": top_events(rows, "lex_gain_vs_hard", limit=10, reverse=True),
        "top_negative_lex_events": top_events(rows, "lex_gain_vs_hard", limit=10, reverse=False),
        "top_positive_sem_events": top_events(rows, "sem_gain_vs_hard", limit=10, reverse=True),
        "top_negative_sem_events": top_events(rows, "sem_gain_vs_hard", limit=10, reverse=False),
    }

    write_json(output_dir / "afterhours_transfer_pair_tail_question_lexical_confirmation_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_confirmation_groups.csv", group_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_confirmation_years.csv", year_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_confirmation_events.csv", event_rows)


if __name__ == "__main__":
    main()
