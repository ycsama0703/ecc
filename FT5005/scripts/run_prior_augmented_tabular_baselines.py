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
from sklearn.ensemble import HistGradientBoostingRegressor

from run_dense_multimodal_ablation_baselines import (
    build_dense_matrix,
    build_text_lsa_bundle,
    fit_and_select_alpha,
    infer_audio_feature_names,
    infer_prefixed_feature_names,
    load_joined_rows,
    split_rows,
    standardize,
    top_coefficients,
)
from run_structured_baselines import inverse_target, metrics, transform_target
from run_text_tfidf_baselines import EXTRA_DENSE_FEATURES, STRUCTURED_FEATURES
from dj30_qc_utils import safe_float, write_csv, write_json


QUARTER_NUM = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run prior-augmented ridge and nonlinear tabular baselines."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/prior_augmented_baselines_real"),
    )
    parser.add_argument("--target-col", default="post_call_60m_rv")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=64)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--skip-hgbr", action="store_true")
    return parser.parse_args()


def attach_ticker_prior(
    rows: list[dict[str, str]],
    train_end_year: int,
) -> list[dict[str, str]]:
    chronological = []
    for row in rows:
        item = dict(row)
        item["_quarter_num"] = QUARTER_NUM.get((row.get("quarter") or "").strip().upper(), 0)
        chronological.append(item)

    train_rows = [row for row in chronological if row["_year"] <= train_end_year]
    train_global_mean = float(np.mean([row["_target"] for row in train_rows]))

    train_ticker_means = {}
    ticker_groups = {}
    for row in train_rows:
        ticker_groups.setdefault(row.get("ticker", ""), []).append(row["_target"])
    for ticker, values in ticker_groups.items():
        train_ticker_means[ticker] = float(np.mean(values)) if values else train_global_mean

    history_by_ticker: dict[str, list[float]] = {}
    chronological.sort(
        key=lambda item: (
            item["_year"],
            item["_quarter_num"],
            item.get("ticker", ""),
            item.get("event_key", ""),
        )
    )

    enriched = []
    for row in chronological:
        ticker = row.get("ticker", "")
        ticker_history = history_by_ticker.get(ticker, [])
        prior = (
            float(np.mean(ticker_history))
            if ticker_history
            else train_ticker_means.get(ticker, train_global_mean)
        )
        row["prior_ticker_expanding_mean"] = str(prior)
        history_by_ticker.setdefault(ticker, []).append(row["_target"])
        enriched.append(row)
    return enriched


def build_dense_bundle(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
):
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


def fit_and_select_hgbr(
    train_x: np.ndarray,
    train_y_t: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    random_state: int,
):
    grid = [
        {"learning_rate": 0.05, "max_depth": 3, "max_leaf_nodes": 15, "min_samples_leaf": 10, "l2_regularization": 0.0},
        {"learning_rate": 0.03, "max_depth": 4, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 0.1},
    ]
    best_params = None
    best_model = None
    best_val_pred = None
    best_val_rmse = None

    for params in grid:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=150,
            early_stopping=False,
            random_state=random_state,
            **params,
        )
        model.fit(train_x, train_y_t)
        val_pred = inverse_target(model.predict(val_x))
        val_rmse = metrics(val_y, val_pred)["rmse"]
        if best_val_rmse is None or val_rmse < best_val_rmse:
            best_params = params
            best_model = model
            best_val_pred = val_pred
            best_val_rmse = val_rmse
    return best_params, best_model, best_val_pred


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading rows", flush=True)
    rows = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.audio_csv,
        args.target_col,
        args.qa_csv,
    )
    print(f"joined_rows={len(rows)}", flush=True)
    rows = attach_ticker_prior(rows, args.train_end_year)
    split = split_rows(rows, args.train_end_year, args.val_year)
    train_rows = split["train"]
    val_rows = split["val"]
    test_rows = split["test"]
    print(
        f"split train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}",
        flush=True,
    )

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    print("building dense bundles", flush=True)
    dense_bundles = {
        "prior": build_dense_bundle(train_rows, val_rows, test_rows, ["prior_ticker_expanding_mean"]),
        "structured": build_dense_bundle(train_rows, val_rows, test_rows, STRUCTURED_FEATURES),
        "extra": build_dense_bundle(train_rows, val_rows, test_rows, EXTRA_DENSE_FEATURES),
        "qa_benchmark": build_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            infer_prefixed_feature_names(rows, "qa_bench_"),
        ),
        "audio": build_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            infer_audio_feature_names(rows),
        ),
    }

    print("building text bundle", flush=True)
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
    train_y_t = transform_target(train_y)

    model_specs = {
        "ridge_prior_only": ("ridge", ["prior"]),
        "ridge_prior_plus_structured": ("ridge", ["prior", "structured"]),
        "ridge_prior_plus_structured_plus_extra": ("ridge", ["prior", "structured", "extra"]),
        "ridge_prior_plus_structured_plus_extra_plus_qa_benchmark": (
            "ridge",
            ["prior", "structured", "extra", "qa_benchmark"],
        ),
        "ridge_prior_plus_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark": (
            "ridge",
            ["prior", "structured", "extra", "qna_lsa", "qa_benchmark"],
        ),
        "hgbr_prior_only": ("hgbr", ["prior"]),
        "hgbr_prior_plus_structured_plus_extra": ("hgbr", ["prior", "structured", "extra"]),
        "hgbr_prior_plus_structured_plus_extra_plus_qa_benchmark": (
            "hgbr",
            ["prior", "structured", "extra", "qa_benchmark"],
        ),
    }
    if args.skip_hgbr:
        model_specs = {
            name: spec for name, spec in model_specs.items() if spec[0] != "hgbr"
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

    for model_name, (model_family, bundle_names) in model_specs.items():
        print(f"fitting {model_name}", flush=True)
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

        if model_family == "ridge":
            best_params, best_model, pred_val = fit_and_select_alpha(
                train_x, train_y_t, val_x, val_y, alphas
            )
            pred_test = inverse_target(best_model.predict(test_x))
            extra_info = {
                "best_alpha": best_params,
                "top_coefficients": top_coefficients(best_model, feature_names),
            }
        else:
            best_params, best_model, pred_val = fit_and_select_hgbr(
                train_x, train_y_t, val_x, val_y, args.random_state
            )
            pred_test = inverse_target(best_model.predict(test_x))
            extra_info = {
                "best_params": best_params,
            }

        summary["models"][model_name] = {
            "family": model_family,
            "feature_count": int(train_x.shape[1]),
            "val": metrics(val_y, pred_val),
            "test": metrics(test_y, pred_test),
            **extra_info,
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

    write_json(output_dir / "prior_augmented_baseline_summary.json", summary)
    write_csv(output_dir / "prior_augmented_baseline_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
