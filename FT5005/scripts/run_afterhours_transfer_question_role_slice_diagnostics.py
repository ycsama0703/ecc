#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dj30_qc_utils import write_csv, write_json
from run_afterhours_transfer_agreement_signal_benchmark import build_temporal_rows
from run_afterhours_transfer_role_text_signal_benchmark import (
    attach_role_text,
    build_shared_role_lsa_bundle,
)
from run_structured_baselines import metrics

STOPWORDS = {
    "the",
    "and",
    "to",
    "of",
    "that",
    "we",
    "you",
    "in",
    "on",
    "is",
    "it",
    "our",
    "so",
    "re",
    "ve",
    "about",
    "how",
    "your",
    "just",
    "can",
    "could",
    "or",
    "kind",
    "kind of",
    "maybe",
    "now",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose where standalone analyst-question role semantics help or fail inside "
            "the after-hours transfer benchmark."
        )
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--role-benchmark-root",
        type=Path,
        default=Path("results/afterhours_transfer_role_text_signal_benchmark_lsa4_real"),
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("results/features_real/event_text_audio_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_question_role_slice_diagnostics_lsa4_real"),
    )
    parser.add_argument("--train-split", default="val2020_test_post2020")
    parser.add_argument("--val-split", default="val2021_test_post2021")
    parser.add_argument("--test-split", default="val2022_test_post2022")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=4)
    parser.add_argument("--term-limit", type=int, default=8)
    parser.add_argument("--top-event-count", type=int, default=10)
    return parser.parse_args()


def load_prediction_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["row_key"]: row for row in csv.DictReader(handle)}


def truncate_text(text: str, limit: int = 220) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def filtered_terms(component_rows: list[dict[str, object]], direction: str, limit: int) -> list[str]:
    raw_terms = [str(row["term"]) for row in component_rows if row["direction"] == direction]
    kept = []
    for term in raw_terms:
        lowered = term.lower().strip()
        if lowered in STOPWORDS:
            continue
        if len(lowered) <= 2:
            continue
        kept.append(term)
        if len(kept) >= limit:
            break
    if kept:
        return kept
    return raw_terms[:limit]


def component_label_lookup(component_term_rows: list[dict[str, object]], term_limit: int) -> dict[tuple[int, int], str]:
    labels: dict[tuple[int, int], str] = {}
    component_ids = sorted({int(row["component"]) for row in component_term_rows})
    for component_id in component_ids:
        rows = [row for row in component_term_rows if int(row["component"]) == component_id]
        positive = filtered_terms(rows, "positive", term_limit)
        negative = filtered_terms(rows, "negative", term_limit)
        labels[(component_id, 1)] = " / ".join(positive)
        labels[(component_id, -1)] = " / ".join(negative)
    return labels


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def concentration_stats(gains: list[float]) -> dict[str, float]:
    positive = sorted((gain for gain in gains if gain > 0), reverse=True)
    total_positive_mass = float(sum(positive))
    payload = {
        "total_positive_mass": total_positive_mass,
        "positive_event_count": int(len(positive)),
    }
    for top_k in [1, 3, 5, 10]:
        top_mass = float(sum(positive[:top_k]))
        payload[f"top_{top_k}_positive_mass_share"] = (
            float(top_mass / total_positive_mass) if total_positive_mass > 0 else 0.0
        )
    return payload


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_rows = build_temporal_rows(args.temporal_root.resolve())
    coverage = attach_role_text(temporal_rows, args.features_csv.resolve())

    train_rows = [row for row in temporal_rows if row["split"] == args.train_split]
    val_rows = [row for row in temporal_rows if row["split"] == args.val_split]
    test_rows = [row for row in temporal_rows if row["split"] == args.test_split]
    role_bundle = build_shared_role_lsa_bundle(
        train_rows,
        val_rows,
        test_rows,
        max_features=args.max_features,
        min_df=args.min_df,
        lsa_components=args.lsa_components,
        term_limit=args.term_limit,
    )
    component_labels = component_label_lookup(role_bundle["component_term_rows"], args.term_limit)

    benchmark_summary = json.loads(
        (args.role_benchmark_root / "afterhours_transfer_role_text_signal_benchmark_summary.json").read_text()
    )
    prediction_rows = load_prediction_rows(
        args.role_benchmark_root / "afterhours_transfer_role_text_signal_benchmark_test_predictions.csv"
    )

    event_rows = []
    slice_groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    ticker_groups: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row, question_scores in zip(test_rows, role_bundle["question_test"]):
        prediction_row = prediction_rows.get(str(row["row_key"]))
        if prediction_row is None:
            raise SystemExit(f"missing role-text prediction row for {row['row_key']}")

        dominant_idx = int(np.argmax(np.abs(question_scores)))
        dominant_score = float(question_scores[dominant_idx])
        dominant_sign = 1 if dominant_score >= 0 else -1
        dominant_component = dominant_idx + 1
        slice_key = (dominant_component, dominant_sign)

        target = float(prediction_row["target"])
        pred_question = float(prediction_row["question_role_lsa"])
        pred_pre = float(prediction_row["residual_pre_call_market_only"])
        pred_geometry = float(prediction_row["geometry_only"])
        pred_hard = float(prediction_row["agreement_pre_only_abstention"])

        se_question = float((target - pred_question) ** 2)
        se_pre = float((target - pred_pre) ** 2)
        se_geometry = float((target - pred_geometry) ** 2)
        se_hard = float((target - pred_hard) ** 2)

        event_payload = {
            "row_key": row["row_key"],
            "event_key": row["event_key"],
            "ticker": row["ticker"],
            "year": int(row["year"]),
            "target": target,
            "question_role_pred": pred_question,
            "pre_call_market_pred": pred_pre,
            "geometry_only_pred": pred_geometry,
            "hard_abstention_pred": pred_hard,
            "question_role_se": se_question,
            "pre_call_market_se": se_pre,
            "geometry_only_se": se_geometry,
            "hard_abstention_se": se_hard,
            "question_role_mse_gain_vs_pre": se_pre - se_question,
            "question_role_mse_gain_vs_geometry": se_geometry - se_question,
            "question_role_mse_gain_vs_hard": se_hard - se_question,
            "question_role_better_than_pre": int(se_question < se_pre),
            "question_role_better_than_geometry": int(se_question < se_geometry),
            "question_role_better_than_hard": int(se_question < se_hard),
            "dominant_component": dominant_component,
            "dominant_sign": dominant_sign,
            "dominant_component_label": component_labels[slice_key],
            "dominant_score": dominant_score,
            "dominant_abs_score": abs(dominant_score),
            "question_text_preview": truncate_text(str(row.get("question_text", "") or "")),
        }
        event_rows.append(event_payload)
        slice_groups[slice_key].append(event_payload)
        ticker_groups[str(row["ticker"])].append(event_payload)

    test_y = np.asarray([float(row["target"]) for row in event_rows], dtype=float)
    question_pred = np.asarray([float(row["question_role_pred"]) for row in event_rows], dtype=float)
    pre_pred = np.asarray([float(row["pre_call_market_pred"]) for row in event_rows], dtype=float)
    geometry_pred = np.asarray([float(row["geometry_only_pred"]) for row in event_rows], dtype=float)
    hard_pred = np.asarray([float(row["hard_abstention_pred"]) for row in event_rows], dtype=float)

    overall = {
        "split": args.test_split,
        "event_count": len(event_rows),
        "question_role_metrics": metrics(test_y, question_pred),
        "pre_call_market_metrics": metrics(test_y, pre_pred),
        "geometry_only_metrics": metrics(test_y, geometry_pred),
        "hard_abstention_metrics": metrics(test_y, hard_pred),
        "mean_mse_gain_vs_pre": float(np.mean((test_y - pre_pred) ** 2 - (test_y - question_pred) ** 2)),
        "mean_mse_gain_vs_geometry": float(np.mean((test_y - geometry_pred) ** 2 - (test_y - question_pred) ** 2)),
        "mean_mse_gain_vs_hard": float(np.mean((test_y - hard_pred) ** 2 - (test_y - question_pred) ** 2)),
        "win_share_vs_pre": float(np.mean((test_y - question_pred) ** 2 < (test_y - pre_pred) ** 2)),
        "win_share_vs_geometry": float(np.mean((test_y - question_pred) ** 2 < (test_y - geometry_pred) ** 2)),
        "win_share_vs_hard": float(np.mean((test_y - question_pred) ** 2 < (test_y - hard_pred) ** 2)),
        "positive_gain_concentration_vs_pre": concentration_stats(
            [float(row["question_role_mse_gain_vs_pre"]) for row in event_rows]
        ),
        "positive_gain_concentration_vs_geometry": concentration_stats(
            [float(row["question_role_mse_gain_vs_geometry"]) for row in event_rows]
        ),
        "positive_gain_concentration_vs_hard": concentration_stats(
            [float(row["question_role_mse_gain_vs_hard"]) for row in event_rows]
        ),
    }

    slice_rows = []
    for (component_id, sign), rows_ in sorted(slice_groups.items(), key=lambda item: (-len(item[1]), item[0][0], item[0][1])):
        y = np.asarray([float(row["target"]) for row in rows_], dtype=float)
        pred_q = np.asarray([float(row["question_role_pred"]) for row in rows_], dtype=float)
        pred_pre = np.asarray([float(row["pre_call_market_pred"]) for row in rows_], dtype=float)
        pred_geo = np.asarray([float(row["geometry_only_pred"]) for row in rows_], dtype=float)
        pred_hard = np.asarray([float(row["hard_abstention_pred"]) for row in rows_], dtype=float)
        gains_pre = [float(row["question_role_mse_gain_vs_pre"]) for row in rows_]
        gains_geo = [float(row["question_role_mse_gain_vs_geometry"]) for row in rows_]
        gains_hard = [float(row["question_role_mse_gain_vs_hard"]) for row in rows_]
        slice_rows.append(
            {
                "dominant_component": component_id,
                "dominant_sign": sign,
                "dominant_component_label": component_labels[(component_id, sign)],
                "event_count": len(rows_),
                "mean_abs_score": float(np.mean([float(row["dominant_abs_score"]) for row in rows_])),
                "question_role_rmse": rmse(y, pred_q),
                "pre_call_market_rmse": rmse(y, pred_pre),
                "geometry_only_rmse": rmse(y, pred_geo),
                "hard_abstention_rmse": rmse(y, pred_hard),
                "mean_mse_gain_vs_pre": float(np.mean(gains_pre)),
                "mean_mse_gain_vs_geometry": float(np.mean(gains_geo)),
                "mean_mse_gain_vs_hard": float(np.mean(gains_hard)),
                "win_share_vs_pre": float(np.mean([gain > 0 for gain in gains_pre])),
                "win_share_vs_geometry": float(np.mean([gain > 0 for gain in gains_geo])),
                "win_share_vs_hard": float(np.mean([gain > 0 for gain in gains_hard])),
                "positive_gain_mass_share_vs_pre": (
                    float(sum(gain for gain in gains_pre if gain > 0) / overall["positive_gain_concentration_vs_pre"]["total_positive_mass"])
                    if overall["positive_gain_concentration_vs_pre"]["total_positive_mass"] > 0
                    else 0.0
                ),
            }
        )

    ticker_rows = []
    for ticker, rows_ in sorted(
        ticker_groups.items(),
        key=lambda item: np.mean([float(row["question_role_mse_gain_vs_pre"]) for row in item[1]]),
        reverse=True,
    ):
        gains_pre = [float(row["question_role_mse_gain_vs_pre"]) for row in rows_]
        gains_geo = [float(row["question_role_mse_gain_vs_geometry"]) for row in rows_]
        gains_hard = [float(row["question_role_mse_gain_vs_hard"]) for row in rows_]
        ticker_rows.append(
            {
                "ticker": ticker,
                "event_count": len(rows_),
                "mean_mse_gain_vs_pre": float(np.mean(gains_pre)),
                "mean_mse_gain_vs_geometry": float(np.mean(gains_geo)),
                "mean_mse_gain_vs_hard": float(np.mean(gains_hard)),
                "win_share_vs_pre": float(np.mean([gain > 0 for gain in gains_pre])),
                "win_share_vs_geometry": float(np.mean([gain > 0 for gain in gains_geo])),
                "win_share_vs_hard": float(np.mean([gain > 0 for gain in gains_hard])),
            }
        )

    event_rows_sorted = sorted(event_rows, key=lambda row: float(row["question_role_mse_gain_vs_pre"]), reverse=True)
    summary = {
        "config": {
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "lsa_components": int(role_bundle["n_components"]),
            "role_explained_variance_ratio_sum": float(role_bundle["explained_variance_ratio_sum"]),
        },
        "coverage": coverage,
        "benchmark_reference": {
            "question_role_family": benchmark_summary["best_factor_only_family_summary"]
            if benchmark_summary.get("best_factor_only_family") == "question_role_lsa"
            else next(
                family for family in benchmark_summary["families"] if family["family"] == "question_role_lsa"
            ),
            "geometry_only_family": next(
                family for family in benchmark_summary["families"] if family["family"] == "geometry_only"
            ),
            "best_family": benchmark_summary["best_family_summary"],
        },
        "overall": overall,
        "signed_component_slices": slice_rows,
        "ticker_summaries": ticker_rows,
        "top_positive_events_vs_pre": event_rows_sorted[: args.top_event_count],
        "top_negative_events_vs_pre": list(reversed(event_rows_sorted[-args.top_event_count:])),
    }

    write_json(output_dir / "afterhours_transfer_question_role_slice_diagnostics_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_question_role_slice_diagnostics_slices.csv", slice_rows)
    write_csv(output_dir / "afterhours_transfer_question_role_slice_diagnostics_tickers.csv", ticker_rows)
    write_csv(output_dir / "afterhours_transfer_question_role_slice_diagnostics_events.csv", event_rows_sorted)


if __name__ == "__main__":
    main()
