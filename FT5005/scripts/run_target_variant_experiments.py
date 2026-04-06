#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import site
import sys
from pathlib import Path

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
from sklearn.linear_model import Ridge

from dj30_qc_utils import safe_float, write_json
from run_dense_multimodal_ablation_baselines import (
    build_dense_matrix,
    build_text_lsa_bundle,
    infer_prefixed_feature_names,
    load_joined_rows,
    split_rows,
    standardize,
)
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_structured_baselines import metrics
from run_text_tfidf_baselines import EXTRA_DENSE_FEATURES, STRUCTURED_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw and normalized volatility targets under prior-aware residual models."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/target_variant_experiments_real"),
    )
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-8)
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


def fit_residual_ridge(train_x, train_prior, train_y, val_x, val_prior, val_y, alphas):
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


def derived_targets(row: dict[str, str], eps: float) -> dict[str, float | None]:
    post = safe_float(row.get("post_call_60m_rv"))
    pre = safe_float(row.get("pre_60m_rv"))
    within = safe_float(row.get("within_call_rv"))
    if post is None or pre is None or within is None:
        return {
            "raw_post_call_60m_rv": None,
            "log_post_over_pre": None,
            "log_post_over_within": None,
            "shock_minus_pre": None,
        }
    return {
        "raw_post_call_60m_rv": post,
        "log_post_over_pre": float(math.log(post + eps) - math.log(pre + eps)),
        "log_post_over_within": float(math.log(post + eps) - math.log(within + eps)),
        "shock_minus_pre": float(post - pre),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base_rows = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.audio_csv,
        "post_call_60m_rv",
        args.qa_csv,
    )
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    summary = {
        "split_config": {
            "train_end_year": args.train_end_year,
            "val_year": args.val_year,
        },
        "variants": {},
    }

    for variant_name in [
        "raw_post_call_60m_rv",
        "log_post_over_pre",
        "log_post_over_within",
        "shock_minus_pre",
    ]:
        rows = []
        for row in base_rows:
            target_map = derived_targets(row, args.eps)
            target_value = target_map[variant_name]
            if target_value is None or not math.isfinite(target_value):
                continue
            item = dict(row)
            item["_target"] = float(target_value)
            rows.append(item)

        rows = attach_ticker_prior(rows, args.train_end_year)
        split = split_rows(rows, args.train_end_year, args.val_year)
        train_rows = split["train"]
        val_rows = split["val"]
        test_rows = split["test"]

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
        text_bundle = build_text_lsa_bundle(
            train_rows,
            val_rows,
            test_rows,
            text_col="qna_text",
            max_features=args.max_features,
            min_df=args.min_df,
            lsa_components=args.lsa_components,
        )

        train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
        val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
        test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
        train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
        val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
        test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)

        variant_summary = {
            "split_sizes": {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
            },
            "models": {
                "prior_only": {
                    "val": metrics(val_y, val_prior),
                    "test": metrics(test_y, test_prior),
                }
            },
        }

        model_specs = {
            "residual_structured_plus_extra": ["structured", "extra"],
            "residual_structured_plus_extra_plus_qa_benchmark": ["structured", "extra", "qa_benchmark"],
            "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark": [
                "structured",
                "extra",
                "qa_benchmark",
                "qna_lsa",
            ],
        }

        for model_name, bundle_names in model_specs.items():
            train_parts = []
            val_parts = []
            test_parts = []
            for bundle_name in bundle_names:
                bundle = text_bundle if bundle_name == "qna_lsa" else dense_bundles[bundle_name]
                train_parts.append(bundle["train"])
                val_parts.append(bundle["val"])
                test_parts.append(bundle["test"])
            train_x = np.hstack(train_parts)
            val_x = np.hstack(val_parts)
            test_x = np.hstack(test_parts)

            best_alpha, best_model, pred_val = fit_residual_ridge(
                train_x, train_prior, train_y, val_x, val_prior, val_y, alphas
            )
            pred_test = test_prior + best_model.predict(test_x)
            variant_summary["models"][model_name] = {
                "best_alpha": best_alpha,
                "val": metrics(val_y, pred_val),
                "test": metrics(test_y, pred_test),
            }

        summary["variants"][variant_name] = variant_summary

    write_json(output_dir / "target_variant_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
