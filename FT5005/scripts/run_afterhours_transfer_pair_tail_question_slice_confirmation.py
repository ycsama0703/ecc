#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json
from run_afterhours_transfer_expert_selection import summarize_significance
from run_structured_baselines import metrics

MODEL = "tail_question_top1_lsa"
PRE = "residual_pre_call_market_only"
HARD = "agreement_pre_only_abstention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm where the hardest-question local semantic route helps relative to hard abstention."
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
        default=Path("results/afterhours_transfer_pair_tail_question_slice_confirmation_real"),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_lookup(path: Path) -> dict[str, dict[str, str]]:
    return {row["event_key"]: row for row in load_csv_rows(path.resolve()) if row.get("event_key")}


def add_side_inputs(rows, panel_lookup, features_lookup, qa_lookup):
    coverage = {
        "rows": len(rows),
        "with_panel": 0,
        "with_features": 0,
        "with_qa": 0,
        "with_all": 0,
    }
    for row in rows:
        key = row["event_key"]
        p = panel_lookup.get(key)
        f = features_lookup.get(key)
        q = qa_lookup.get(key)
        if p:
            coverage["with_panel"] += 1
            for name in ["a4_strict_row_share", "a4_strict_high_conf_share", "revenue_surprise_pct"]:
                row[name] = safe_float(p.get(name, ""))
        if f:
            coverage["with_features"] += 1
            for name in ["qa_pair_count", "qna_word_count", "answer_to_question_word_ratio"]:
                row[name] = safe_float(f.get(name, ""))
        if q:
            coverage["with_qa"] += 1
            for name in [
                "qa_bench_direct_early_score_mean",
                "qa_bench_evasion_score_mean",
                "qa_bench_coverage_mean",
                "qa_bench_direct_answer_share",
                "qa_bench_nonresponse_share",
            ]:
                row[name] = safe_float(q.get(name, ""))
        if p and f and q:
            coverage["with_all"] += 1
    return coverage


def arr(rows, key):
    return np.asarray([float(r[key]) for r in rows], dtype=float)


def add_diagnostics(rows):
    for row in rows:
        y = float(row["target"])
        pre = float(row[PRE])
        hard = float(row[HARD])
        pred = float(row[MODEL])
        row["hard_se"] = (y - hard) ** 2
        row["pred_se"] = (y - pred) ** 2
        row["pre_se"] = (y - pre) ** 2
        row["mse_gain_vs_hard"] = row["hard_se"] - row["pred_se"]
        row["mse_gain_vs_pre"] = row["pre_se"] - row["pred_se"]
        row["use_agreed"] = int(float(row.get(f"{MODEL}__use_agreed", 0)))
        row["agreement"] = int(float(row.get("agreement", 0)))


def percentile_threshold(rows, key, q):
    values = np.asarray([float(r[key]) for r in rows], dtype=float)
    return float(np.quantile(values, q))


def build_slice_masks(rows):
    direct_med = percentile_threshold(rows, "qa_bench_direct_early_score_mean", 0.5)
    evasion_med = percentile_threshold(rows, "qa_bench_evasion_score_mean", 0.5)
    density_med = percentile_threshold(rows, "qa_pair_count", 0.5)
    obs_med = percentile_threshold(rows, "a4_strict_row_share", 0.5)

    ordered = sorted(rows, key=lambda r: float(r["hard_se"]), reverse=True)
    top_quartile_keys = {r["event_key"] for r in ordered[: max(1, len(rows) // 4)]}

    masks = {
        "overall": lambda r: True,
        "agreement_veto": lambda r: int(r["agreement"]) == 1 and int(r["use_agreed"]) == 0,
        "agreement_keep": lambda r: int(r["agreement"]) == 1 and int(r["use_agreed"]) == 1,
        "disagreement": lambda r: int(r["agreement"]) == 0,
        "hard_miss_top_quartile": lambda r: r["event_key"] in top_quartile_keys,
        "high_qna_density": lambda r: float(r["qa_pair_count"]) >= density_med,
        "low_directness_high_evasion": lambda r: float(r["qa_bench_direct_early_score_mean"]) <= direct_med and float(r["qa_bench_evasion_score_mean"]) >= evasion_med,
        "weaker_observability": lambda r: float(r["a4_strict_row_share"]) <= obs_med,
        "dense_evasive_weak_obs": lambda r: float(r["qa_pair_count"]) >= density_med and float(r["qa_bench_direct_early_score_mean"]) <= direct_med and float(r["qa_bench_evasion_score_mean"]) >= evasion_med and float(r["a4_strict_row_share"]) <= obs_med,
    }
    thresholds = {
        "qa_pair_count_median": density_med,
        "qa_bench_direct_early_score_mean_median": direct_med,
        "qa_bench_evasion_score_mean_median": evasion_med,
        "a4_strict_row_share_median": obs_med,
        "hard_miss_top_quartile_size": len(top_quartile_keys),
    }
    return masks, thresholds


def summarize_slice(name, rows, all_rows, args):
    y = arr(rows, "target")
    hard = arr(rows, HARD)
    pred = arr(rows, MODEL)
    pre = arr(rows, PRE)
    out = {
        "slice": name,
        "n": len(rows),
        "pre_r2": metrics(y, pre)["r2"],
        "hard_r2": metrics(y, hard)["r2"],
        "route_r2": metrics(y, pred)["r2"],
        "route_minus_hard_r2": metrics(y, pred)["r2"] - metrics(y, hard)["r2"],
        "mean_mse_gain_vs_hard": float(np.mean(arr(rows, "mse_gain_vs_hard"))),
        "mean_mse_gain_vs_pre": float(np.mean(arr(rows, "mse_gain_vs_pre"))),
        "win_share_vs_hard": float(np.mean(arr(rows, "mse_gain_vs_hard") > 0.0)),
        "win_share_vs_pre": float(np.mean(arr(rows, "mse_gain_vs_pre") > 0.0)),
        "use_agreed_share": float(np.mean(arr(rows, "use_agreed"))),
        "agreement_share": float(np.mean(arr(rows, "agreement"))),
        "net_mse_gain_vs_hard": float(np.sum(arr(rows, "mse_gain_vs_hard"))),
        "net_mse_gain_vs_pre": float(np.sum(arr(rows, "mse_gain_vs_pre"))),
        "p_mse_vs_hard": summarize_significance(y, hard, pred, args.bootstrap_iters, args.perm_iters, args.seed)["mse_gain_pvalue"],
    }
    overall_net = float(np.sum(arr(all_rows, "mse_gain_vs_hard")))
    out["share_of_overall_net_gain_vs_hard"] = float(out["net_mse_gain_vs_hard"] / overall_net) if overall_net != 0 else 0.0
    return out


def top_events(rows, limit=8):
    return [
        {
            "event_key": r["event_key"],
            "ticker": r.get("ticker", ""),
            "year": int(float(r.get("year", 0))) if r.get("year") not in (None, "") else None,
            "mse_gain_vs_hard": float(r["mse_gain_vs_hard"]),
            "use_agreed": int(r["use_agreed"]),
            "agreement": int(r["agreement"]),
        }
        for r in rows[:limit]
    ]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.predictions_csv.resolve().open()))
    panel_lookup = load_lookup(args.panel_csv)
    features_lookup = load_lookup(args.features_csv)
    qa_lookup = load_lookup(args.qa_csv)
    coverage = add_side_inputs(rows, panel_lookup, features_lookup, qa_lookup)
    add_diagnostics(rows)

    masks, thresholds = build_slice_masks(rows)
    slice_rows = []
    detailed = {}
    for name, mask in masks.items():
        subset = [r for r in rows if mask(r)]
        if not subset:
            continue
        summary = summarize_slice(name, subset, rows, args)
        slice_rows.append(summary)
        ordered = sorted(subset, key=lambda r: float(r["mse_gain_vs_hard"]), reverse=True)
        detailed[name] = {
            "summary": summary,
            "top_positive_events": top_events(ordered, limit=8),
            "top_negative_events": top_events(sorted(subset, key=lambda r: float(r["mse_gain_vs_hard"])), limit=8),
        }

    summary = {
        "config": {
            "predictions_csv": str(args.predictions_csv.resolve()),
            "panel_csv": str(args.panel_csv.resolve()),
            "features_csv": str(args.features_csv.resolve()),
            "qa_csv": str(args.qa_csv.resolve()),
        },
        "coverage": coverage,
        "thresholds": thresholds,
        "slice_overview": slice_rows,
        "slice_details": detailed,
    }

    write_json(output_dir / "afterhours_transfer_pair_tail_question_slice_confirmation_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_slice_confirmation_overview.csv", slice_rows)


if __name__ == "__main__":
    main()
