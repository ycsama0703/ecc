#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json


TARGET_SCALE = 1_000_000.0


NUMERIC_FEATURES = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple structured baselines on the event-level modeling panel."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/baselines_real"))
    parser.add_argument("--target-col", default="post_call_60m_rv")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--ridge-lambdas", default="0.1,1,10,100,1000")
    return parser.parse_args()


def load_panel(path: Path, target_col: str) -> list[dict]:
    rows = []
    for row in load_csv_rows(path):
        target_value = safe_float(row.get(target_col))
        year_value = safe_float(row.get("year"))
        if target_value is None or year_value is None:
            continue
        row["_target"] = target_value
        row["_year"] = int(year_value)
        rows.append(row)
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


def numeric_matrix(rows: list[dict], medians: dict[str, float] | None = None):
    if medians is None:
        medians = {}
        for feature in NUMERIC_FEATURES:
            values = [safe_float(row.get(feature)) for row in rows]
            clean = [value for value in values if value is not None and math.isfinite(value)]
            medians[feature] = float(np.median(clean)) if clean else 0.0

    html_levels = ["pass", "warn", "fail"]
    matrix = []
    for row in rows:
        vector = []
        for feature in NUMERIC_FEATURES:
            value = safe_float(row.get(feature))
            if value is None or not math.isfinite(value):
                value = medians[feature]
            vector.append(value)
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        for level in html_levels:
            vector.append(1.0 if html_flag == level else 0.0)
        matrix.append(vector)
    return np.asarray(matrix, dtype=float), medians


def standardize(train_x: np.ndarray, other_xs: list[np.ndarray]):
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1.0
    train_z = (train_x - means) / stds
    outputs = [train_z]
    for matrix in other_xs:
        outputs.append((matrix - means) / stds)
    return outputs, means, stds


def transform_target(values: np.ndarray) -> np.ndarray:
    return np.log1p(values * TARGET_SCALE)


def inverse_target(values: np.ndarray) -> np.ndarray:
    return np.expm1(values) / TARGET_SCALE


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, ridge_lambda: float) -> np.ndarray:
    x_design = np.column_stack([np.ones(len(train_x)), train_x])
    eye = np.eye(x_design.shape[1])
    eye[0, 0] = 0.0
    lhs = x_design.T @ x_design + ridge_lambda * eye
    rhs = x_design.T @ train_y
    return np.linalg.solve(lhs, rhs)


def predict_ridge(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(matrix)), matrix])
    return design @ weights


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    baseline = float(np.mean(y_true))
    denom = float(np.sum((y_true - baseline) ** 2))
    r2 = 0.0 if denom == 0 else float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_panel(args.panel_csv.resolve(), args.target_col)
    split = split_rows(rows, train_end_year=args.train_end_year, val_year=args.val_year)

    train_rows = split["train"]
    val_rows = split["val"]
    test_rows = split["test"]

    train_x, medians = numeric_matrix(train_rows)
    val_x, _ = numeric_matrix(val_rows, medians)
    test_x, _ = numeric_matrix(test_rows, medians)

    (train_z, val_z, test_z), means, stds = standardize(train_x, [val_x, test_x])

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)

    train_y_t = transform_target(train_y)

    mean_pred_val = np.full_like(val_y, np.mean(train_y), dtype=float)
    mean_pred_test = np.full_like(test_y, np.mean(train_y), dtype=float)

    ridge_lambdas = [float(item) for item in args.ridge_lambdas.split(",") if item.strip()]
    best_lambda = None
    best_val_rmse = None
    best_weights = None

    for ridge_lambda in ridge_lambdas:
        weights = fit_ridge(train_z, train_y_t, ridge_lambda=ridge_lambda)
        pred_val = inverse_target(predict_ridge(val_z, weights))
        val_rmse = metrics(val_y, pred_val)["rmse"]
        if best_val_rmse is None or val_rmse < best_val_rmse:
            best_lambda = ridge_lambda
            best_val_rmse = val_rmse
            best_weights = weights

    ridge_pred_val = inverse_target(predict_ridge(val_z, best_weights))
    ridge_pred_test = inverse_target(predict_ridge(test_z, best_weights))

    summary = {
        "target_col": args.target_col,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "mean_baseline": {
            "val": metrics(val_y, mean_pred_val),
            "test": metrics(test_y, mean_pred_test),
        },
        "ridge": {
            "best_lambda": best_lambda,
            "val": metrics(val_y, ridge_pred_val),
            "test": metrics(test_y, ridge_pred_test),
        },
        "feature_count": int(train_z.shape[1]),
        "features": NUMERIC_FEATURES + ["html_integrity_flag=pass", "html_integrity_flag=warn", "html_integrity_flag=fail"],
    }

    prediction_rows = []
    for split_name, split_rows_, y_true_arr, preds in [
        ("val_mean", val_rows, val_y, mean_pred_val),
        ("test_mean", test_rows, test_y, mean_pred_test),
        ("val_ridge", val_rows, val_y, ridge_pred_val),
        ("test_ridge", test_rows, test_y, ridge_pred_test),
    ]:
        for row, truth, pred in zip(split_rows_, y_true_arr, preds):
            prediction_rows.append(
                {
                    "model_split": split_name,
                    "event_key": row["event_key"],
                    "year": row["year"],
                    "ticker": row["ticker"],
                    "y_true": truth,
                    "y_pred": pred,
                }
            )

    write_json(output_dir / "structured_baseline_summary.json", summary)
    write_csv(output_dir / "structured_baseline_predictions.csv", prediction_rows)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

