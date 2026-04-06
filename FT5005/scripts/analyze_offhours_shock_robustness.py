#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dj30_qc_utils import load_csv_rows, write_csv, write_json
from run_structured_baselines import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze year-wise, ticker-wise, and concentration robustness for off-hours shock predictions."
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument(
        "--models",
        default="prior_only,residual_structured_only,residual_structured_plus_extra_plus_qna_lsa",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/offhours_shock_robustness_real"),
    )
    return parser.parse_args()


def regime_label(scheduled_hour_et: str) -> str:
    hour = float(scheduled_hour_et)
    if hour < 9.5:
        return "pre_market"
    if hour < 16.0:
        return "market_hours"
    return "after_hours"


def load_prediction_rows(predictions_csv: Path, target_models: list[str]) -> dict[str, list[dict]]:
    grouped = defaultdict(list)
    for row in load_csv_rows(predictions_csv.resolve()):
        model_split = row.get("model_split", "")
        if not model_split.endswith("_test"):
            continue
        model_name = model_split[: -len("_test")]
        if model_name not in target_models:
            continue
        row["_model_name"] = model_name
        row["_y_true"] = float(row["y_true"])
        row["_y_pred"] = float(row["y_pred"])
        grouped[model_name].append(row)
    return grouped


def enrich_with_panel_info(rows_by_model: dict[str, list[dict]], panel_csv: Path) -> None:
    event_lookup = {}
    for row in load_csv_rows(panel_csv.resolve()):
        event_lookup[row["event_key"]] = row
    for rows in rows_by_model.values():
        for row in rows:
            meta = event_lookup.get(row["event_key"], {})
            row["_regime"] = regime_label(meta.get("scheduled_hour_et", "0"))
            row["_quarter"] = meta.get("quarter", "")


def subgroup_metrics(rows: list[dict], key: str) -> dict[str, dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    out = {}
    for group_name, group_rows in grouped.items():
        y_true = np.asarray([row["_y_true"] for row in group_rows], dtype=float)
        y_pred = np.asarray([row["_y_pred"] for row in group_rows], dtype=float)
        out[group_name] = {
            "n": len(group_rows),
            **metrics(y_true, y_pred),
        }
    return out


def concentration_analysis(
    candidate_rows: list[dict],
    prior_rows: list[dict],
) -> dict[str, object]:
    prior_by_event = {row["event_key"]: row for row in prior_rows}
    by_ticker = defaultdict(lambda: {"sse_gain": 0.0, "events": 0})
    total_gain = 0.0
    for row in candidate_rows:
        prior_row = prior_by_event[row["event_key"]]
        y = row["_y_true"]
        prior_err = (y - prior_row["_y_pred"]) ** 2
        cand_err = (y - row["_y_pred"]) ** 2
        gain = prior_err - cand_err
        total_gain += gain
        bucket = by_ticker[row["ticker"]]
        bucket["sse_gain"] += gain
        bucket["events"] += 1

    ranking = sorted(
        (
            {
                "ticker": ticker,
                "events": payload["events"],
                "sse_gain": float(payload["sse_gain"]),
            }
            for ticker, payload in by_ticker.items()
        ),
        key=lambda item: item["sse_gain"],
        reverse=True,
    )

    def share(top_k: int) -> float:
        if total_gain == 0:
            return 0.0
        return float(sum(item["sse_gain"] for item in ranking[:top_k]) / total_gain)

    return {
        "total_sse_gain_vs_prior": float(total_gain),
        "top_1_share": share(1),
        "top_3_share": share(3),
        "top_5_share": share(5),
        "ticker_ranking": ranking,
    }


def leave_one_ticker_out(
    candidate_rows: list[dict],
    prior_rows: list[dict],
) -> list[dict]:
    prior_by_event = {row["event_key"]: row for row in prior_rows}
    all_tickers = sorted({row["ticker"] for row in candidate_rows})
    out = []
    for ticker in all_tickers:
        kept = [row for row in candidate_rows if row["ticker"] != ticker]
        kept_prior = [prior_by_event[row["event_key"]] for row in kept]
        y_true = np.asarray([row["_y_true"] for row in kept], dtype=float)
        cand_pred = np.asarray([row["_y_pred"] for row in kept], dtype=float)
        prior_pred = np.asarray([row["_y_pred"] for row in kept_prior], dtype=float)
        cand_metrics = metrics(y_true, cand_pred)
        prior_metrics = metrics(y_true, prior_pred)
        out.append(
            {
                "held_out_ticker": ticker,
                "n_remaining": len(kept),
                "candidate_r2": cand_metrics["r2"],
                "prior_r2": prior_metrics["r2"],
                "r2_gain": cand_metrics["r2"] - prior_metrics["r2"],
                "candidate_rmse": cand_metrics["rmse"],
                "prior_rmse": prior_metrics["rmse"],
                "rmse_gain": prior_metrics["rmse"] - cand_metrics["rmse"],
            }
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_models = [item.strip() for item in args.models.split(",") if item.strip()]

    rows_by_model = load_prediction_rows(args.predictions_csv, target_models)
    enrich_with_panel_info(rows_by_model, args.panel_csv)

    summary = {
        "models": {},
    }
    influence_rows = []

    prior_rows = rows_by_model.get("prior_only")
    if prior_rows is None:
        raise SystemExit("prior_only predictions are required for concentration and influence analysis.")

    for model_name, rows in rows_by_model.items():
        y_true = np.asarray([row["_y_true"] for row in rows], dtype=float)
        y_pred = np.asarray([row["_y_pred"] for row in rows], dtype=float)
        model_summary = {
            "overall": {
                "n": len(rows),
                **metrics(y_true, y_pred),
            },
            "by_year": subgroup_metrics(rows, "year"),
            "by_ticker": subgroup_metrics(rows, "ticker"),
            "by_regime": subgroup_metrics(rows, "_regime"),
        }
        if model_name != "prior_only":
            model_summary["concentration_vs_prior"] = concentration_analysis(rows, prior_rows)
            influence = leave_one_ticker_out(rows, prior_rows)
            model_summary["leave_one_ticker_out_summary"] = {
                "min_r2_gain": float(min(item["r2_gain"] for item in influence)),
                "median_r2_gain": float(np.median([item["r2_gain"] for item in influence])),
                "max_r2_gain": float(max(item["r2_gain"] for item in influence)),
            }
            for item in influence:
                item["model"] = model_name
                influence_rows.append(item)
        summary["models"][model_name] = model_summary

    write_json(output_dir / "offhours_shock_robustness_summary.json", summary)
    write_csv(output_dir / "offhours_shock_leave_one_ticker_out.csv", influence_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
