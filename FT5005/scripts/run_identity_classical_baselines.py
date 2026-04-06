#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import site
import sys
from collections import defaultdict
from pathlib import Path

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json


STRUCTURED_FEATURES = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "pre_60m_volume_sum",
    "within_call_rv",
    "within_call_vw_rv",
    "within_call_volume_sum",
    "call_duration_min",
    "scheduled_hour_et",
    "revenue_surprise_pct",
    "ebitda_surprise_pct",
    "eps_gaap_surprise_pct",
    "analyst_eps_norm_num_est",
    "analyst_eps_norm_std",
    "analyst_revenue_num_est",
    "analyst_revenue_std",
    "analyst_net_income_num_est",
    "analyst_net_income_std",
    "a1_component_count",
    "a1_question_count",
    "a1_answer_count",
    "a1_qna_component_share",
    "a1_total_text_words",
    "a1_unique_speaker_count",
    "a2_paragraph_count",
    "a2_visible_word_count",
    "a2_size_ratio_vs_group",
    "a2_text_ratio_vs_group",
    "a4_kept_rows_for_duration",
    "a4_median_match_score",
    "a4_strict_row_share",
    "a4_broad_row_share",
    "a4_hard_fail_rows",
]

HAR_EVENT_FEATURES = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "within_call_rv",
    "within_call_vw_rv",
    "within_call_volume_sum",
    "call_duration_min",
    "scheduled_hour_et",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run identity-aware and classical volatility baselines."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/identity_classical_baselines_real"),
    )
    parser.add_argument("--target-col", default="post_call_60m_rv")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--ar-min-history", type=int, default=6)
    parser.add_argument("--ar-history-window", type=int, default=12)
    return parser.parse_args()


def quarter_num(text: str) -> int:
    value = (text or "").strip().upper()
    if value.startswith("Q") and value[1:].isdigit():
        return int(value[1:])
    return 0


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    baseline = float(np.mean(y_true))
    denom = float(np.sum((y_true - baseline) ** 2))
    r2 = 0.0 if denom == 0 else float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
    mse = float(np.mean((y_true - y_pred) ** 2))
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2}


def load_rows(path: Path, target_col: str) -> list[dict]:
    rows = []
    for row in load_csv_rows(path.resolve()):
        target_value = safe_float(row.get(target_col))
        year_value = safe_float(row.get("year"))
        if target_value is None or year_value is None:
            continue
        row["_target"] = target_value
        row["_year"] = int(year_value)
        row["_quarter_num"] = quarter_num(row.get("quarter", ""))
        rows.append(row)
    rows.sort(key=lambda item: (item["_year"], item["_quarter_num"], item.get("ticker", ""), item.get("event_key", "")))
    return rows


def split_rows(rows: list[dict], train_end_year: int, val_year: int) -> dict[str, list[dict]]:
    split = {"train": [], "val": [], "test": []}
    for row in rows:
        year_value = row["_year"]
        if year_value <= train_end_year:
            split["train"].append(row)
        elif year_value == val_year:
            split["val"].append(row)
        else:
            split["test"].append(row)
    return split


def build_dense_matrix(rows: list[dict], features: list[str], medians: dict[str, float] | None = None):
    if medians is None:
        medians = {}
        for feature in features:
            values = [safe_float(row.get(feature)) for row in rows]
            clean = [value for value in values if value is not None and math.isfinite(value)]
            medians[feature] = float(np.median(clean)) if clean else 0.0

    matrix = []
    for row in rows:
        vector = []
        for feature in features:
            value = safe_float(row.get(feature))
            if value is None or not math.isfinite(value):
                value = medians[feature]
            vector.append(value)
        matrix.append(vector)
    return np.asarray(matrix, dtype=float), medians


def standardize(train_x: np.ndarray, other_xs: list[np.ndarray]):
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1.0
    outputs = [(train_x - means) / stds]
    for matrix in other_xs:
        outputs.append((matrix - means) / stds)
    return outputs


def fit_ols(train_x: np.ndarray, train_y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    return np.linalg.lstsq(design, train_y, rcond=None)[0]


def predict_linear(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(matrix)), matrix])
    return design @ weights


def fit_identity_only(train_rows: list[dict]) -> dict[str, float]:
    by_ticker = defaultdict(list)
    for row in train_rows:
        by_ticker[row.get("ticker", "")].append(row["_target"])
    return {ticker: float(np.mean(values)) for ticker, values in by_ticker.items() if ticker}


def ar1_forecast(history: list[float], window: int) -> float:
    fit_history = history[-window:]
    if len(fit_history) < 2:
        return float(np.mean(fit_history))
    x = np.asarray(fit_history[:-1], dtype=float)
    y = np.asarray(fit_history[1:], dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    beta = 0.0 if denom == 0 else float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    alpha = y_mean - beta * x_mean
    return alpha + beta * fit_history[-1]


def sequential_history_predictions(
    rows: list[dict],
    train_end_year: int,
    val_year: int,
    train_rows: list[dict],
    ar_min_history: int,
    ar_history_window: int,
) -> tuple[dict[str, dict], list[dict]]:
    train_global_mean = float(np.mean([row["_target"] for row in train_rows]))
    train_ticker_means = fit_identity_only(train_rows)

    chronological = sorted(rows, key=lambda item: (item["_year"], item["_quarter_num"], item.get("ticker", ""), item.get("event_key", "")))
    history_by_ticker: dict[str, list[float]] = defaultdict(list)
    global_history: list[float] = []

    predictions_by_model = {
        "global_mean_prior": [],
        "ticker_last_prior": [],
        "ticker_expanding_mean": [],
        "ticker_train_mean": [],
        "ticker_only_identity": [],
        "ar1_recursive": [],
    }
    records = []

    for row in chronological:
        ticker = row.get("ticker", "")
        split_name = "train"
        if row["_year"] == val_year:
            split_name = "val"
        elif row["_year"] > val_year:
            split_name = "test"

        ticker_history = history_by_ticker.get(ticker, [])
        global_prior = float(np.mean(global_history)) if global_history else train_global_mean
        ticker_last = ticker_history[-1] if ticker_history else train_ticker_means.get(ticker, train_global_mean)
        ticker_mean = float(np.mean(ticker_history)) if ticker_history else train_ticker_means.get(ticker, train_global_mean)
        ticker_train_mean = train_ticker_means.get(ticker, train_global_mean)

        ar1_pred = ticker_mean
        if len(ticker_history) >= ar_min_history:
            ar1_pred = ar1_forecast(ticker_history, window=ar_history_window)

        if split_name in {"val", "test"}:
            row_record = {
                "event_key": row["event_key"],
                "ticker": ticker,
                "year": row["year"],
                "quarter": row.get("quarter", ""),
                "split": split_name,
                "y_true": row["_target"],
                "global_mean_prior": global_prior,
                "ticker_last_prior": ticker_last,
                "ticker_expanding_mean": ticker_mean,
                "ticker_train_mean": ticker_train_mean,
                "ticker_only_identity": ticker_train_mean,
                "ar1_recursive": ar1_pred,
            }
            records.append(row_record)
            for model_name in predictions_by_model:
                predictions_by_model[model_name].append(row_record)

        global_history.append(row["_target"])
        history_by_ticker[ticker].append(row["_target"])

    summary = {}
    for model_name, model_rows in predictions_by_model.items():
        val_rows = [row for row in model_rows if row["split"] == "val"]
        test_rows = [row for row in model_rows if row["split"] == "test"]
        summary[model_name] = {
            "val": metrics(
                np.asarray([row["y_true"] for row in val_rows], dtype=float),
                np.asarray([row[model_name] for row in val_rows], dtype=float),
            ),
            "test": metrics(
                np.asarray([row["y_true"] for row in test_rows], dtype=float),
                np.asarray([row[model_name] for row in test_rows], dtype=float),
            ),
        }
    return summary, records


def linear_model_summary(
    name: str,
    train_rows: list[dict],
    val_rows: list[dict],
    test_rows: list[dict],
    features: list[str],
) -> tuple[dict, list[dict]]:
    train_x, medians = build_dense_matrix(train_rows, features)
    val_x, _ = build_dense_matrix(val_rows, features, medians)
    test_x, _ = build_dense_matrix(test_rows, features, medians)
    train_z, val_z, test_z = standardize(train_x, [val_x, test_x])

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)

    weights = fit_ols(train_z, train_y)
    pred_val = predict_linear(val_z, weights)
    pred_test = predict_linear(test_z, weights)

    summary = {
        "feature_count": int(train_z.shape[1]),
        "features": features,
        "val": metrics(val_y, pred_val),
        "test": metrics(test_y, pred_test),
    }
    rows = []
    for split_name, split_rows_, preds in [
        ("val", val_rows, pred_val),
        ("test", test_rows, pred_test),
    ]:
        for row, pred in zip(split_rows_, preds):
            rows.append(
                {
                    "event_key": row["event_key"],
                    "ticker": row["ticker"],
                    "year": row["year"],
                    "quarter": row.get("quarter", ""),
                    "split": split_name,
                    "model": name,
                    "y_true": row["_target"],
                    "y_pred": float(pred),
                }
            )
    return summary, rows


def with_lag_feature(train_rows: list[dict], val_rows: list[dict], test_rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    chronological = sorted(
        train_rows + val_rows + test_rows,
        key=lambda item: (item["_year"], item["_quarter_num"], item.get("ticker", ""), item.get("event_key", "")),
    )
    history_by_ticker: dict[str, list[float]] = defaultdict(list)
    result = {"train": [], "val": [], "test": []}
    for row in chronological:
        ticker = row.get("ticker", "")
        history = history_by_ticker.get(ticker, [])
        new_row = dict(row)
        new_row["ticker_prev_target"] = history[-1] if history else None
        new_row["ticker_prev_mean_target"] = float(np.mean(history)) if history else None
        split_name = "train"
        if row in val_rows:
            split_name = "val"
        elif row in test_rows:
            split_name = "test"
        result[split_name].append(new_row)
        history_by_ticker[ticker].append(row["_target"])
    return result["train"], result["val"], result["test"]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.panel_csv, args.target_col)
    split = split_rows(rows, args.train_end_year, args.val_year)
    train_rows = split["train"]
    val_rows = split["val"]
    test_rows = split["test"]

    history_summary, history_records = sequential_history_predictions(
        rows,
        train_end_year=args.train_end_year,
        val_year=args.val_year,
        train_rows=train_rows,
        ar_min_history=args.ar_min_history,
        ar_history_window=args.ar_history_window,
    )

    structured_summary, structured_pred_rows = linear_model_summary(
        "ols_structured",
        train_rows,
        val_rows,
        test_rows,
        STRUCTURED_FEATURES,
    )

    train_lag, val_lag, test_lag = with_lag_feature(train_rows, val_rows, test_rows)
    har_summary, har_pred_rows = linear_model_summary(
        "har_event_ols",
        train_lag,
        val_lag,
        test_lag,
        HAR_EVENT_FEATURES + ["ticker_prev_target", "ticker_prev_mean_target"],
    )

    summary = {
        "target_col": args.target_col,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "models": {
            **history_summary,
            "ols_structured": structured_summary,
            "har_event_ols": har_summary,
        },
    }

    prediction_rows = []
    for row in history_records:
        for model_name in [
            "global_mean_prior",
            "ticker_last_prior",
            "ticker_expanding_mean",
            "ticker_train_mean",
            "ticker_only_identity",
            "ar1_recursive",
        ]:
            prediction_rows.append(
                {
                    "event_key": row["event_key"],
                    "ticker": row["ticker"],
                    "year": row["year"],
                    "quarter": row["quarter"],
                    "split": row["split"],
                    "model": model_name,
                    "y_true": row["y_true"],
                    "y_pred": row[model_name],
                }
            )
    prediction_rows.extend(structured_pred_rows)
    prediction_rows.extend(har_pred_rows)

    write_json(output_dir / "identity_classical_baseline_summary.json", summary)
    write_csv(output_dir / "identity_classical_baseline_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
