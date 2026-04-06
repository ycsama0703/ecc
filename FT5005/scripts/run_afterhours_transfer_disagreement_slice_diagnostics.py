#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import site
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

from dj30_qc_utils import write_csv, write_json
from run_structured_baselines import metrics


MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_QA_EXPERT = "residual_pre_call_market_plus_a4_plus_qa_benchmark_svd_observability_gate"
MODEL_RETAINED = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate"
MODEL_SELECTED = "validation_selected_transfer_expert"
MODEL_PAIR_TREE = "conservative_tree_override_on_selected_expert"
MODEL_PLUS_TEXT_LOGISTIC = "conservative_logistic_override_on_selected_expert"

SPLITS = [
    "val2020_test_post2020",
    "val2021_test_post2021",
    "val2022_test_post2022",
]

PATTERN_LABELS = {
    (0, 1): "tree_sem_logistic_qa",
    (1, 0): "tree_qa_logistic_sem",
}

MODEL_ORDER = [
    MODEL_PRE_ONLY,
    MODEL_QA_EXPERT,
    MODEL_RETAINED,
    MODEL_SELECTED,
    MODEL_PAIR_TREE,
    MODEL_PLUS_TEXT_LOGISTIC,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose why the latest temporal disagreement slice slightly favors the QA expert and whether that reflects broad transfer drift or concentrated event pockets."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_disagreement_slice_diagnostics_role_aware_audio_lsa4_real"),
    )
    parser.add_argument("--latest-split", default="val2022_test_post2022")
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["event_key"]: row for row in csv.DictReader(handle)}


def disagreement_records(split_name: str, pair_rows: dict[str, dict[str, str]], logistic_rows: dict[str, dict[str, str]]) -> list[dict]:
    records = []
    for key in sorted(set(pair_rows) & set(logistic_rows)):
        pair_row = pair_rows[key]
        logistic_row = logistic_rows[key]
        tree_choose_qa = int(float(pair_row["tree_choose_qa"]))
        logistic_choose_qa = int(float(logistic_row["logistic_choose_qa"]))
        if tree_choose_qa == logistic_choose_qa:
            continue
        target = float(pair_row["target"])
        preds = {
            MODEL_PRE_ONLY: float(pair_row[MODEL_PRE_ONLY]),
            MODEL_QA_EXPERT: float(pair_row[MODEL_QA_EXPERT]),
            MODEL_RETAINED: float(pair_row[MODEL_RETAINED]),
            MODEL_SELECTED: float(pair_row[MODEL_SELECTED]),
            MODEL_PAIR_TREE: float(pair_row[MODEL_PAIR_TREE]),
            MODEL_PLUS_TEXT_LOGISTIC: float(logistic_row[MODEL_PLUS_TEXT_LOGISTIC]),
        }
        err2 = {f"{model_name}__err2": float((target - pred) ** 2) for model_name, pred in preds.items()}
        pattern = PATTERN_LABELS[(tree_choose_qa, logistic_choose_qa)]
        best_model = min(preds, key=lambda model_name: err2[f"{model_name}__err2"])
        records.append(
            {
                "split": split_name,
                "event_key": key,
                "ticker": pair_row["ticker"],
                "year": int(float(pair_row["year"])),
                "chosen_model": pair_row["chosen_model"],
                "tree_choose_qa": tree_choose_qa,
                "logistic_choose_qa": logistic_choose_qa,
                "pattern": pattern,
                "best_model": best_model,
                "target": target,
                **preds,
                **err2,
                "qa_vs_pre_mse_gain": err2[f"{MODEL_PRE_ONLY}__err2"] - err2[f"{MODEL_QA_EXPERT}__err2"],
                "sem_vs_pre_mse_gain": err2[f"{MODEL_PRE_ONLY}__err2"] - err2[f"{MODEL_RETAINED}__err2"],
                "qa_better_than_pre": int(err2[f"{MODEL_QA_EXPERT}__err2"] < err2[f"{MODEL_PRE_ONLY}__err2"]),
                "pre_better_than_qa": int(err2[f"{MODEL_PRE_ONLY}__err2"] < err2[f"{MODEL_QA_EXPERT}__err2"]),
                "qa_pre_tie": int(abs(err2[f"{MODEL_QA_EXPERT}__err2"] - err2[f"{MODEL_PRE_ONLY}__err2"]) < 1e-18),
            }
        )
    return records


def metric_block(rows: list[dict]) -> dict[str, dict[str, float]]:
    y_true = np.asarray([float(row["target"]) for row in rows], dtype=float)
    return {
        model_name: metrics(y_true, np.asarray([float(row[model_name]) for row in rows], dtype=float))
        for model_name in MODEL_ORDER
    }


def summarise_pattern_rows(rows: list[dict], split_name: str) -> list[dict]:
    summary_rows = []
    for pattern in ["tree_sem_logistic_qa", "tree_qa_logistic_sem"]:
        subset = [row for row in rows if row["pattern"] == pattern]
        if not subset:
            continue
        overall = metric_block(subset)
        row = {
            "split": split_name,
            "pattern": pattern,
            "test_size": len(subset),
            "qa_better_than_pre_count": int(sum(row["qa_better_than_pre"] for row in subset)),
            "pre_better_than_qa_count": int(sum(row["pre_better_than_qa"] for row in subset)),
            "qa_pre_tie_count": int(sum(row["qa_pre_tie"] for row in subset)),
            "qa_vs_pre_mse_gain_sum": float(sum(float(row["qa_vs_pre_mse_gain"]) for row in subset)),
            "best_model_count_pre": int(sum(row["best_model"] == MODEL_PRE_ONLY for row in subset)),
            "best_model_count_qa": int(sum(row["best_model"] == MODEL_QA_EXPERT for row in subset)),
            "best_model_count_sem": int(sum(row["best_model"] == MODEL_RETAINED for row in subset)),
        }
        for model_name, metric_dict in overall.items():
            row[f"{model_name}_r2"] = metric_dict["r2"]
            row[f"{model_name}_rmse"] = metric_dict["rmse"]
        summary_rows.append(row)
    return summary_rows


def latest_ticker_rows(rows: list[dict]) -> list[dict]:
    bucket = defaultdict(lambda: {"test_size": 0, "qa_vs_pre_mse_gain_sum": 0.0, "qa_better": 0, "pre_better": 0, "ties": 0})
    for row in rows:
        item = bucket[row["ticker"]]
        item["test_size"] += 1
        item["qa_vs_pre_mse_gain_sum"] += float(row["qa_vs_pre_mse_gain"])
        item["qa_better"] += int(row["qa_better_than_pre"])
        item["pre_better"] += int(row["pre_better_than_qa"])
        item["ties"] += int(row["qa_pre_tie"])
    out = []
    for ticker, values in bucket.items():
        out.append(
            {
                "ticker": ticker,
                "test_size": values["test_size"],
                "qa_vs_pre_mse_gain_sum": float(values["qa_vs_pre_mse_gain_sum"]),
                "qa_better_than_pre_count": values["qa_better"],
                "pre_better_than_qa_count": values["pre_better"],
                "qa_pre_tie_count": values["ties"],
            }
        )
    return sorted(out, key=lambda row: row["qa_vs_pre_mse_gain_sum"], reverse=True)


def latest_event_tables(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    positives = sorted(rows, key=lambda row: float(row["qa_vs_pre_mse_gain"]), reverse=True)
    negatives = sorted(rows, key=lambda row: float(row["qa_vs_pre_mse_gain"]))
    return positives, negatives


def main() -> None:
    args = parse_args()
    temporal_root = args.temporal_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_summary_path = temporal_root / "afterhours_transfer_router_temporal_confirmation_summary.json"
    temporal_summary = json.load(temporal_summary_path.open()) if temporal_summary_path.exists() else {}

    pooled_rows = []
    pattern_overview_rows = []
    split_summary = {}

    for split_name in SPLITS:
        pair_path = temporal_root / f"{split_name}__hybrid_pair_bench_tree" / "afterhours_transfer_conservative_router_predictions.csv"
        logistic_path = temporal_root / f"{split_name}__hybrid_plus_text_logistic" / "afterhours_transfer_conservative_router_predictions.csv"
        pair_rows = load_rows(pair_path)
        logistic_rows = load_rows(logistic_path)
        rows = disagreement_records(split_name, pair_rows, logistic_rows)
        pooled_rows.extend(rows)
        overall = metric_block(rows)
        pattern_rows = summarise_pattern_rows(rows, split_name)
        pattern_overview_rows.extend(pattern_rows)
        split_summary[split_name] = {
            "test_size": len(rows),
            "pattern_counts": dict(Counter(row["pattern"] for row in rows)),
            "chosen_model_counts": dict(Counter(row["chosen_model"] for row in rows)),
            "best_model_counts": dict(Counter(row["best_model"] for row in rows)),
            "qa_vs_pre": {
                "qa_better_count": int(sum(row["qa_better_than_pre"] for row in rows)),
                "pre_better_count": int(sum(row["pre_better_than_qa"] for row in rows)),
                "tie_count": int(sum(row["qa_pre_tie"] for row in rows)),
                "qa_vs_pre_mse_gain_sum": float(sum(float(row["qa_vs_pre_mse_gain"]) for row in rows)),
            },
            "overall": overall,
            "pattern_details": {row["pattern"]: row for row in pattern_rows},
        }

    pooled_overall = metric_block(pooled_rows)
    pooled_pattern_rows = summarise_pattern_rows(pooled_rows, "pooled")
    pattern_overview_rows.extend(pooled_pattern_rows)

    latest_rows = [row for row in pooled_rows if row["split"] == args.latest_split]
    latest_tickers = latest_ticker_rows(latest_rows)
    latest_positive_events, latest_negative_events = latest_event_tables(latest_rows)
    positive_total = float(sum(max(float(row["qa_vs_pre_mse_gain"]), 0.0) for row in latest_rows))
    negative_total = float(sum(max(-float(row["qa_vs_pre_mse_gain"]), 0.0) for row in latest_rows))
    net_total = float(sum(float(row["qa_vs_pre_mse_gain"]) for row in latest_rows))

    summary = {
        "source_temporal_root": str(temporal_root),
        "source_temporal_config": temporal_summary.get("config", {}),
        "patterns": {
            "tree_sem_logistic_qa": "tree prefers the retained semantic expert while logistic prefers the QA expert",
            "tree_qa_logistic_sem": "tree prefers the QA expert while logistic prefers the retained semantic expert",
        },
        "splits": split_summary,
        "pooled": {
            "test_size": len(pooled_rows),
            "pattern_counts": dict(Counter(row["pattern"] for row in pooled_rows)),
            "best_model_counts": dict(Counter(row["best_model"] for row in pooled_rows)),
            "qa_vs_pre": {
                "qa_better_count": int(sum(row["qa_better_than_pre"] for row in pooled_rows)),
                "pre_better_count": int(sum(row["pre_better_than_qa"] for row in pooled_rows)),
                "tie_count": int(sum(row["qa_pre_tie"] for row in pooled_rows)),
                "qa_vs_pre_mse_gain_sum": float(sum(float(row["qa_vs_pre_mse_gain"]) for row in pooled_rows)),
            },
            "overall": pooled_overall,
            "pattern_details": {row["pattern"]: row for row in pooled_pattern_rows},
        },
        "latest_split_focus": {
            "split": args.latest_split,
            "test_size": len(latest_rows),
            "pattern_counts": dict(Counter(row["pattern"] for row in latest_rows)),
            "best_model_counts": dict(Counter(row["best_model"] for row in latest_rows)),
            "qa_vs_pre": {
                "qa_better_count": int(sum(row["qa_better_than_pre"] for row in latest_rows)),
                "pre_better_count": int(sum(row["pre_better_than_qa"] for row in latest_rows)),
                "tie_count": int(sum(row["qa_pre_tie"] for row in latest_rows)),
                "qa_positive_gain_total": positive_total,
                "qa_negative_gain_total": negative_total,
                "qa_vs_pre_net_gain": net_total,
                "top_positive_event_share_of_positive_total": float(sum(max(float(row["qa_vs_pre_mse_gain"]), 0.0) for row in latest_positive_events[:1])) / positive_total if positive_total > 0 else 0.0,
                "top3_positive_event_share_of_positive_total": float(sum(max(float(row["qa_vs_pre_mse_gain"]), 0.0) for row in latest_positive_events[:3])) / positive_total if positive_total > 0 else 0.0,
            },
            "ticker_gain_rows": latest_tickers,
            "top_positive_events": latest_positive_events[:5],
            "top_negative_events": latest_negative_events[:5],
        },
    }

    write_csv(output_dir / "afterhours_transfer_disagreement_slice_pattern_overview.csv", pattern_overview_rows)
    write_csv(output_dir / "afterhours_transfer_disagreement_slice_event_rows.csv", pooled_rows)
    write_csv(output_dir / "afterhours_transfer_disagreement_slice_latest_ticker_gain.csv", latest_tickers)
    write_json(output_dir / "afterhours_transfer_disagreement_slice_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
