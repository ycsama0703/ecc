#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
from sklearn.linear_model import Ridge

from dj30_qc_utils import write_csv, write_json
from run_dense_multimodal_ablation_baselines import (
    build_dense_matrix,
    build_text_lsa_bundle,
    infer_prefixed_feature_names,
    load_joined_rows,
    split_rows,
    standardize,
    top_coefficients,
)
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_structured_baselines import metrics
from run_text_tfidf_baselines import EXTRA_DENSE_FEATURES, STRUCTURED_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run residual ridge baselines on top of same-ticker expanding prior."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/prior_residual_baselines_real"),
    )
    parser.add_argument("--target-col", default="post_call_60m_rv")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=64)
    return parser.parse_args()


def build_dense_bundle(train_rows, val_rows, test_rows, feature_names):
    train_x, medians, extra_names = build_dense_matrix(train_rows, feature_names)
    val_x, _, _ = build_dense_matrix(val_rows, feature_names, medians)
    test_x, _, _ = build_dense_matrix(test_rows, feature_names, medians)
    train_z, val_z, test_z = standardize(train_x, [val_x, test_x])
    return {
        "train": train_z,
        "val": val_z,
        "test": test_z,
        "feature_names": feature_names + extra_names,
    }


def fit_residual_ridge(
    train_x: np.ndarray,
    train_prior: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_prior: np.ndarray,
    val_y: np.ndarray,
    alphas: list[float],
):
    train_residual = train_y - train_prior
    best_alpha = None
    best_model = None
    best_val_pred = None
    best_val_rmse = None
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(train_x, train_residual)
        val_pred = val_prior + model.predict(val_x)
        val_rmse = metrics(val_y, val_pred)["rmse"]
        if best_val_rmse is None or val_rmse < best_val_rmse:
            best_alpha = alpha
            best_model = model
            best_val_pred = val_pred
            best_val_rmse = val_rmse
    return best_alpha, best_model, best_val_pred


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.audio_csv,
        args.target_col,
        args.qa_csv,
    )
    rows = attach_ticker_prior(rows, args.train_end_year)
    split = split_rows(rows, args.train_end_year, args.val_year)
    train_rows = split["train"]
    val_rows = split["val"]
    test_rows = split["test"]

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    dense_bundles = {
        "structured": build_dense_bundle(train_rows, val_rows, test_rows, STRUCTURED_FEATURES),
        "extra": build_dense_bundle(train_rows, val_rows, test_rows, EXTRA_DENSE_FEATURES),
        "qa_benchmark": build_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            infer_prefixed_feature_names(rows, "qa_bench_"),
        ),
    }
    text_bundles = {
        "qna_lsa": build_text_lsa_bundle(
            train_rows,
            val_rows,
            test_rows,
            text_col="qna_text",
            max_features=args.max_features,
            min_df=args.min_df,
            lsa_components=args.lsa_components,
        )
    }

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
    val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
    test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)

    model_specs = {
        "prior_only": [],
        "residual_structured_plus_extra": ["structured", "extra"],
        "residual_structured_plus_extra_plus_qa_benchmark": ["structured", "extra", "qa_benchmark"],
        "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark": [
            "structured",
            "extra",
            "qna_lsa",
            "qa_benchmark",
        ],
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

    for model_name, bundle_names in model_specs.items():
        if not bundle_names:
            pred_val = val_prior
            pred_test = test_prior
            summary["models"][model_name] = {
                "family": "prior_passthrough",
                "feature_count": 1,
                "val": metrics(val_y, pred_val),
                "test": metrics(test_y, pred_test),
            }
        else:
            train_parts = []
            val_parts = []
            test_parts = []
            feature_names = []
            for bundle_name in bundle_names:
                bundle = dense_bundles.get(bundle_name) or text_bundles.get(bundle_name)
                train_parts.append(bundle["train"])
                val_parts.append(bundle["val"])
                test_parts.append(bundle["test"])
                feature_names.extend(bundle["feature_names"])
            train_x = np.hstack(train_parts)
            val_x = np.hstack(val_parts)
            test_x = np.hstack(test_parts)

            best_alpha, best_model, pred_val = fit_residual_ridge(
                train_x, train_prior, train_y, val_x, val_prior, val_y, alphas
            )
            pred_test = test_prior + best_model.predict(test_x)

            summary["models"][model_name] = {
                "family": "residual_ridge",
                "feature_count": int(train_x.shape[1]),
                "best_alpha": best_alpha,
                "val": metrics(val_y, pred_val),
                "test": metrics(test_y, pred_test),
                "top_coefficients": top_coefficients(best_model, feature_names),
            }

        for split_name, split_rows_, preds in [
            (f"{model_name}_val", val_rows, pred_val),
            (f"{model_name}_test", test_rows, pred_test),
        ]:
            for row, pred in zip(split_rows_, preds):
                prediction_rows.append(
                    {
                        "model_split": split_name,
                        "event_key": row["event_key"],
                        "ticker": row["ticker"],
                        "year": row["year"],
                        "y_true": row["_target"],
                        "y_pred": float(pred),
                    }
                )

    write_json(output_dir / "prior_residual_baseline_summary.json", summary)
    write_csv(output_dir / "prior_residual_baseline_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
