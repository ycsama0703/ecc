#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json
from run_structured_baselines import NUMERIC_FEATURES, TARGET_SCALE, inverse_target, metrics, transform_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dense ridge baselines with strict and broad role-aware sequence features."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--sequence-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/sequence_baselines_real")
    )
    parser.add_argument("--target-col", default="post_call_60m_rv")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--ridge-lambdas", default="0.1,1,10,100,1000,10000")
    return parser.parse_args()


def load_joined_rows(panel_csv: Path, sequence_csv: Path, target_col: str) -> list[dict]:
    seq_lookup = {}
    for row in load_csv_rows(sequence_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            seq_lookup[event_key] = row

    rows = []
    for row in load_csv_rows(panel_csv.resolve()):
        event_key = row.get("event_key", "")
        target_value = safe_float(row.get(target_col))
        year_value = safe_float(row.get("year"))
        seq_row = seq_lookup.get(event_key)
        if not event_key or target_value is None or year_value is None or seq_row is None:
            continue
        merged = dict(row)
        merged.update(seq_row)
        merged["_target"] = target_value
        merged["_year"] = int(year_value)
        rows.append(merged)
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


def infer_sequence_features(rows: list[dict], prefix: str) -> list[str]:
    keys = set()
    for row in rows:
        for key in row:
            if key.startswith(prefix):
                keys.add(key)
    return sorted(keys)


def matrix_from_features(rows: list[dict], features: list[str], medians: dict[str, float] | None = None):
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
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        for level in ("pass", "warn", "fail"):
            vector.append(1.0 if html_flag == level else 0.0)
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


def run_model(train_rows, val_rows, test_rows, feature_names, ridge_lambdas):
    train_x, medians = matrix_from_features(train_rows, feature_names)
    val_x, _ = matrix_from_features(val_rows, feature_names, medians)
    test_x, _ = matrix_from_features(test_rows, feature_names, medians)
    train_z, val_z, test_z = standardize(train_x, [val_x, test_x])

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_y_t = transform_target(train_y)

    best_lambda = None
    best_weights = None
    best_val_rmse = None
    for ridge_lambda in ridge_lambdas:
        weights = fit_ridge(train_z, train_y_t, ridge_lambda)
        val_pred = inverse_target(predict_ridge(val_z, weights))
        val_rmse = metrics(val_y, val_pred)["rmse"]
        if best_val_rmse is None or val_rmse < best_val_rmse:
            best_lambda = ridge_lambda
            best_weights = weights
            best_val_rmse = val_rmse

    val_pred = inverse_target(predict_ridge(val_z, best_weights))
    test_pred = inverse_target(predict_ridge(test_z, best_weights))
    return {
        "best_lambda": best_lambda,
        "feature_count": int(train_z.shape[1]),
        "features": feature_names + [
            "html_integrity_flag=pass",
            "html_integrity_flag=warn",
            "html_integrity_flag=fail",
        ],
        "val": metrics(val_y, val_pred),
        "test": metrics(test_y, test_pred),
        "val_predictions": val_pred,
        "test_predictions": test_pred,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_joined_rows(args.panel_csv, args.sequence_csv, args.target_col)
    split = split_rows(rows, args.train_end_year, args.val_year)
    train_rows = split["train"]
    val_rows = split["val"]
    test_rows = split["test"]

    strict_features = infer_sequence_features(rows, "strict_")
    broad_features = infer_sequence_features(rows, "broad_")
    structured_features = NUMERIC_FEATURES

    ridge_lambdas = [float(item) for item in args.ridge_lambdas.split(",") if item.strip()]
    models = {
        "structured_only": structured_features,
        "strict_sequence_only": strict_features,
        "broad_sequence_only": broad_features,
        "structured_plus_strict_sequence": structured_features + strict_features,
        "structured_plus_broad_sequence": structured_features + broad_features,
    }

    summary = {
        "target_col": args.target_col,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "models": {},
    }
    prediction_rows = []

    for model_name, feature_names in models.items():
        result = run_model(train_rows, val_rows, test_rows, feature_names, ridge_lambdas)
        summary["models"][model_name] = {
            "best_lambda": result["best_lambda"],
            "feature_count": result["feature_count"],
            "val": result["val"],
            "test": result["test"],
        }
        for split_name, split_rows_, preds in [
            (f"{model_name}_val", val_rows, result["val_predictions"]),
            (f"{model_name}_test", test_rows, result["test_predictions"]),
        ]:
            for row, pred in zip(split_rows_, preds):
                prediction_rows.append(
                    {
                        "model_split": split_name,
                        "event_key": row["event_key"],
                        "year": row["year"],
                        "ticker": row["ticker"],
                        "y_true": row["_target"],
                        "y_pred": pred,
                    }
                )

    write_json(output_dir / "sequence_baseline_summary.json", summary)
    write_csv(output_dir / "sequence_baseline_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
