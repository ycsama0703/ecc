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

from dj30_qc_utils import write_csv, write_json
from run_dense_multimodal_ablation_baselines import (
    build_text_lsa_bundle,
    infer_audio_feature_names,
    infer_prefixed_feature_names,
    load_joined_rows,
)
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets
from run_text_tfidf_baselines import EXTRA_DENSE_FEATURES, STRUCTURED_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run strict off-hours shock-target ablations and paired significance tests."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/offhours_shock_ablation_real"),
    )
    parser.add_argument("--include-regimes", default="pre_market,after_hours")
    parser.add_argument("--exclude-html-flags", default="")
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def regime_label(row: dict[str, str]) -> str:
    hour = float(row.get("scheduled_hour_et", 0.0))
    if hour < 9.5:
        return "pre_market"
    if hour < 16.0:
        return "market_hours"
    return "after_hours"


def paired_bootstrap_deltas(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    iters: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    rmse_diffs = []
    r2_diffs = []
    mae_diffs = []
    for _ in range(iters):
        idx = rng.integers(0, n, size=n)
        m_a = metrics(y_true[idx], pred_a[idx])
        m_b = metrics(y_true[idx], pred_b[idx])
        rmse_diffs.append(m_a["rmse"] - m_b["rmse"])
        mae_diffs.append(m_a["mae"] - m_b["mae"])
        r2_diffs.append(m_b["r2"] - m_a["r2"])
    return {
        "rmse_diff_mean": float(np.mean(rmse_diffs)),
        "rmse_diff_ci_low": float(np.quantile(rmse_diffs, 0.025)),
        "rmse_diff_ci_high": float(np.quantile(rmse_diffs, 0.975)),
        "mae_diff_mean": float(np.mean(mae_diffs)),
        "mae_diff_ci_low": float(np.quantile(mae_diffs, 0.025)),
        "mae_diff_ci_high": float(np.quantile(mae_diffs, 0.975)),
        "r2_gain_mean": float(np.mean(r2_diffs)),
        "r2_gain_ci_low": float(np.quantile(r2_diffs, 0.025)),
        "r2_gain_ci_high": float(np.quantile(r2_diffs, 0.975)),
    }


def paired_sign_permutation_pvalue(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    iters: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    sq_diff = (y_true - pred_a) ** 2 - (y_true - pred_b) ** 2
    abs_diff = np.abs(y_true - pred_a) - np.abs(y_true - pred_b)
    obs_sq = float(np.mean(sq_diff))
    obs_abs = float(np.mean(abs_diff))
    sq_stats = []
    abs_stats = []
    signs = np.array([-1.0, 1.0])
    for _ in range(iters):
        flip = rng.choice(signs, size=len(y_true), replace=True)
        sq_stats.append(float(np.mean(sq_diff * flip)))
        abs_stats.append(float(np.mean(abs_diff * flip)))
    sq_stats = np.asarray(sq_stats, dtype=float)
    abs_stats = np.asarray(abs_stats, dtype=float)
    return {
        "mse_gain_mean": obs_sq,
        "mse_gain_pvalue": float(np.mean(np.abs(sq_stats) >= abs(obs_sq))),
        "mae_gain_mean": obs_abs,
        "mae_gain_pvalue": float(np.mean(np.abs(abs_stats) >= abs(obs_abs))),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    base_rows = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.audio_csv,
        "post_call_60m_rv",
        args.qa_csv,
    )

    rows = []
    for row in base_rows:
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        if html_flag in exclude_html_flags:
            continue
        if regime_label(row) not in include_regimes:
            continue
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        rows.append(item)

    rows = attach_ticker_prior(rows, args.train_end_year)
    train_rows = [row for row in rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in rows if row["_year"] == args.val_year]
    test_rows = [row for row in rows if row["_year"] > args.val_year]

    bundles = {
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
        "qna_lsa": build_text_lsa_bundle(
            train_rows,
            val_rows,
            test_rows,
            text_col="qna_text",
            max_features=8000,
            min_df=2,
            lsa_components=64,
        ),
    }

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
    val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
    test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)

    model_specs = {
        "prior_only": [],
        "residual_structured_only": ["structured"],
        "residual_structured_plus_extra": ["structured", "extra"],
        "residual_structured_plus_extra_plus_qa_benchmark": ["structured", "extra", "qa_benchmark"],
        "residual_structured_plus_extra_plus_qna_lsa": ["structured", "extra", "qna_lsa"],
        "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark": [
            "structured",
            "extra",
            "qna_lsa",
            "qa_benchmark",
        ],
        "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark_plus_audio": [
            "structured",
            "extra",
            "qna_lsa",
            "qa_benchmark",
            "audio",
        ],
    }

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted(include_regimes),
        "exclude_html_flags": sorted(exclude_html_flags),
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "models": {},
        "significance": {},
    }
    prediction_rows = []
    test_predictions = {}

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
            train_x = np.hstack([bundles[name]["train"] for name in bundle_names])
            val_x = np.hstack([bundles[name]["val"] for name in bundle_names])
            test_x = np.hstack([bundles[name]["test"] for name in bundle_names])
            best_alpha, best_model, pred_val = fit_residual_ridge(
                train_x,
                train_prior,
                train_y,
                val_x,
                val_prior,
                val_y,
                alphas,
            )
            pred_test = test_prior + best_model.predict(test_x)
            summary["models"][model_name] = {
                "family": "residual_ridge",
                "feature_count": int(train_x.shape[1]),
                "best_alpha": best_alpha,
                "val": metrics(val_y, pred_val),
                "test": metrics(test_y, pred_test),
            }

        test_predictions[model_name] = np.asarray(pred_test, dtype=float)
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

    comparison_chain = [
        ("prior_only", "residual_structured_only"),
        ("residual_structured_only", "residual_structured_plus_extra"),
        ("residual_structured_only", "residual_structured_plus_extra_plus_qna_lsa"),
        (
            "residual_structured_only",
            "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark",
        ),
        ("residual_structured_plus_extra", "residual_structured_plus_extra_plus_qa_benchmark"),
        ("residual_structured_plus_extra", "residual_structured_plus_extra_plus_qna_lsa"),
        (
            "residual_structured_plus_extra_plus_qna_lsa",
            "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark",
        ),
        (
            "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark",
            "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark_plus_audio",
        ),
        ("prior_only", "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark"),
    ]

    for baseline_name, candidate_name in comparison_chain:
        pred_a = test_predictions[baseline_name]
        pred_b = test_predictions[candidate_name]
        summary["significance"][f"{baseline_name}__vs__{candidate_name}"] = {
            "bootstrap": paired_bootstrap_deltas(
                test_y, pred_a, pred_b, args.bootstrap_iters, args.seed
            ),
            "permutation": paired_sign_permutation_pvalue(
                test_y, pred_a, pred_b, args.perm_iters, args.seed + 1
            ),
        }

    write_json(output_dir / "offhours_shock_ablation_summary.json", summary)
    write_csv(output_dir / "offhours_shock_ablation_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
