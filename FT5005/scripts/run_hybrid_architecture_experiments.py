#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import site
import sys
from pathlib import Path

# Keep these small experiments reproducible instead of spawning hundreds of BLAS threads.
for env_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(env_name, "1")

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from dj30_qc_utils import write_json
from run_dense_multimodal_ablation_baselines import (
    build_text_lsa_bundle,
    infer_prefixed_feature_names,
    load_joined_rows,
)
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_signal_decomposition_benchmarks import (
    CONTROL_FEATURES,
    ECC_EXTRA_NON_AUDIO_FEATURES,
    ECC_STRUCTURED_FEATURES,
    MARKET_FEATURES,
    regime_label,
)
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


GATE_FEATURES = [
    "scheduled_hour_et",
    "pre_60m_rv",
    "within_call_rv",
    "call_duration_min",
    "revenue_surprise_pct",
    "a4_median_match_score",
    "a4_strict_row_share",
    "qa_bench_pair_count",
    "qa_bench_evasion_score_mean",
    "qa_bench_high_evasion_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Try hybrid residual architectures on the shock target using market and ECC experts."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/hybrid_architecture_qav2_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="")
    parser.add_argument("--exclude-html-flags", default="")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def bundle_name_from_regimes(include_regimes: set[str]) -> str:
    if not include_regimes:
        return "all_regimes"
    return "-".join(sorted(include_regimes))


def build_regime_metrics(
    rows: list[dict[str, str]],
    predictions: np.ndarray,
) -> dict[str, dict[str, float]]:
    by_regime: dict[str, list[tuple[float, float]]] = {}
    for row, pred in zip(rows, predictions):
        regime = row["_regime"]
        by_regime.setdefault(regime, []).append((row["_target"], float(pred)))
    output = {}
    for regime, pairs in sorted(by_regime.items()):
        truth = np.asarray([item[0] for item in pairs], dtype=float)
        preds = np.asarray([item[1] for item in pairs], dtype=float)
        output[regime] = metrics(truth, preds)
    return output


def fit_residual_hgbr(
    train_x: np.ndarray,
    train_prior: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_prior: np.ndarray,
    val_y: np.ndarray,
    random_state: int,
):
    grid = [
        {
            "learning_rate": 0.05,
            "max_depth": 3,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 10,
            "l2_regularization": 0.0,
        },
        {
            "learning_rate": 0.03,
            "max_depth": 4,
            "max_leaf_nodes": 15,
            "min_samples_leaf": 20,
            "l2_regularization": 0.1,
        },
    ]
    train_residual = train_y - train_prior
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
        model.fit(train_x, train_residual)
        val_pred = val_prior + model.predict(val_x)
        val_rmse = metrics(val_y, val_pred)["rmse"]
        if best_val_rmse is None or val_rmse < best_val_rmse:
            best_params = params
            best_model = model
            best_val_pred = val_pred
            best_val_rmse = val_rmse
    return best_params, best_model, best_val_pred


def select_global_blend_weight(
    val_y: np.ndarray,
    market_pred: np.ndarray,
    full_pred: np.ndarray,
) -> tuple[float, np.ndarray]:
    best_weight = 0.0
    best_pred = market_pred
    best_rmse = metrics(val_y, market_pred)["rmse"]
    for weight in np.linspace(0.0, 1.0, 21):
        blended = (1.0 - weight) * market_pred + weight * full_pred
        rmse = metrics(val_y, blended)["rmse"]
        if rmse < best_rmse:
            best_weight = float(weight)
            best_pred = blended
            best_rmse = rmse
    return best_weight, best_pred


def select_regime_blend_weights(
    val_rows: list[dict[str, str]],
    val_y: np.ndarray,
    market_pred_val: np.ndarray,
    full_pred_val: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    global_weight, _ = select_global_blend_weight(val_y, market_pred_val, full_pred_val)
    weights: dict[str, float] = {}
    predictions = np.asarray(market_pred_val, dtype=float).copy()
    regimes = sorted({row["_regime"] for row in val_rows})
    for regime in regimes:
        idx = np.asarray([row["_regime"] == regime for row in val_rows], dtype=bool)
        if idx.sum() < 8:
            weights[regime] = global_weight
            predictions[idx] = (1.0 - global_weight) * market_pred_val[idx] + global_weight * full_pred_val[idx]
            continue
        best_weight = global_weight
        best_pred = (1.0 - best_weight) * market_pred_val[idx] + best_weight * full_pred_val[idx]
        best_rmse = metrics(val_y[idx], best_pred)["rmse"]
        for weight in np.linspace(0.0, 1.0, 21):
            blended = (1.0 - weight) * market_pred_val[idx] + weight * full_pred_val[idx]
            rmse = metrics(val_y[idx], blended)["rmse"]
            if rmse < best_rmse:
                best_weight = float(weight)
                best_pred = blended
                best_rmse = rmse
        weights[regime] = best_weight
        predictions[idx] = best_pred
    return weights, predictions


def apply_regime_blend(
    rows: list[dict[str, str]],
    market_pred: np.ndarray,
    full_pred: np.ndarray,
    weights: dict[str, float],
) -> np.ndarray:
    output = np.asarray(market_pred, dtype=float).copy()
    for idx, row in enumerate(rows):
        weight = weights.get(row["_regime"], 0.0)
        output[idx] = (1.0 - weight) * market_pred[idx] + weight * full_pred[idx]
    return output


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}

    rows = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.audio_csv,
        "post_call_60m_rv",
        args.qa_csv,
    )

    filtered_rows = []
    for row in rows:
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        if html_flag in exclude_html_flags:
            continue
        regime = regime_label(row)
        if include_regimes and regime not in include_regimes:
            continue
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        item["_regime"] = regime
        filtered_rows.append(item)

    filtered_rows = attach_ticker_prior(filtered_rows, args.train_end_year)
    train_rows = [row for row in filtered_rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in filtered_rows if row["_year"] == args.val_year]
    test_rows = [row for row in filtered_rows if row["_year"] > args.val_year]

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    qa_feature_names = infer_prefixed_feature_names(filtered_rows, "qa_bench_")

    dense_bundles = {
        "market": build_dense_bundle(train_rows, val_rows, test_rows, MARKET_FEATURES),
        "controls": build_dense_bundle(train_rows, val_rows, test_rows, CONTROL_FEATURES),
        "ecc_structured": build_dense_bundle(train_rows, val_rows, test_rows, ECC_STRUCTURED_FEATURES),
        "ecc_extra": build_dense_bundle(train_rows, val_rows, test_rows, ECC_EXTRA_NON_AUDIO_FEATURES),
        "qa_benchmark": build_dense_bundle(train_rows, val_rows, test_rows, qa_feature_names),
        "gate": build_dense_bundle(train_rows, val_rows, test_rows, GATE_FEATURES),
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
    }

    base_predictions: dict[str, dict[str, np.ndarray]] = {}
    model_specs = {
        "market_controls_ridge": {
            "family": "residual_ridge",
            "bundles": ["market", "controls"],
        },
        "market_controls_hgbr": {
            "family": "residual_hgbr",
            "bundles": ["market", "controls"],
        },
        "ecc_text_ridge": {
            "family": "residual_ridge",
            "bundles": ["ecc_structured", "ecc_extra", "qa_benchmark", "qna_lsa"],
        },
        "full_ridge": {
            "family": "residual_ridge",
            "bundles": ["market", "controls", "ecc_structured", "ecc_extra", "qa_benchmark", "qna_lsa"],
        },
    }

    for model_name, spec in model_specs.items():
        train_parts = []
        val_parts = []
        test_parts = []
        feature_count = 0
        for bundle_name in spec["bundles"]:
            bundle = text_bundle if bundle_name == "qna_lsa" else dense_bundles[bundle_name]
            train_parts.append(bundle["train"])
            val_parts.append(bundle["val"])
            test_parts.append(bundle["test"])
            feature_count += int(bundle["train"].shape[1])

        train_x = np.hstack(train_parts)
        val_x = np.hstack(val_parts)
        test_x = np.hstack(test_parts)

        if spec["family"] == "residual_ridge":
            best_param, best_model, pred_val = fit_residual_ridge(
                train_x, train_prior, train_y, val_x, val_prior, val_y, alphas
            )
            pred_test = test_prior + best_model.predict(test_x)
            extra = {"best_alpha": best_param}
        else:
            best_param, best_model, pred_val = fit_residual_hgbr(
                train_x, train_prior, train_y, val_x, val_prior, val_y, args.random_state
            )
            pred_test = test_prior + best_model.predict(test_x)
            extra = {"best_params": best_param}

        base_predictions[model_name] = {
            "val": pred_val,
            "test": pred_test,
            "val_residual": pred_val - val_prior,
            "test_residual": pred_test - test_prior,
        }
        summary["models"][model_name] = {
            "family": spec["family"],
            "feature_count": feature_count,
            "val": metrics(val_y, pred_val),
            "test": metrics(test_y, pred_test),
            "val_by_regime": build_regime_metrics(val_rows, pred_val),
            "test_by_regime": build_regime_metrics(test_rows, pred_test),
            **extra,
        }

    summary["models"]["prior_only"] = {
        "family": "prior_passthrough",
        "feature_count": 1,
        "val": metrics(val_y, val_prior),
        "test": metrics(test_y, test_prior),
        "val_by_regime": build_regime_metrics(val_rows, val_prior),
        "test_by_regime": build_regime_metrics(test_rows, test_prior),
    }

    global_weight, global_blend_val = select_global_blend_weight(
        val_y,
        base_predictions["market_controls_ridge"]["val"],
        base_predictions["full_ridge"]["val"],
    )
    global_blend_test = (
        (1.0 - global_weight) * base_predictions["market_controls_ridge"]["test"]
        + global_weight * base_predictions["full_ridge"]["test"]
    )
    summary["models"]["global_market_full_blend"] = {
        "family": "validation_selected_blend",
        "blend_weight_full": global_weight,
        "base_models": ["market_controls_ridge", "full_ridge"],
        "val": metrics(val_y, global_blend_val),
        "test": metrics(test_y, global_blend_test),
        "val_by_regime": build_regime_metrics(val_rows, global_blend_val),
        "test_by_regime": build_regime_metrics(test_rows, global_blend_test),
    }

    regime_weights, regime_blend_val = select_regime_blend_weights(
        val_rows,
        val_y,
        base_predictions["market_controls_ridge"]["val"],
        base_predictions["full_ridge"]["val"],
    )
    regime_blend_test = apply_regime_blend(
        test_rows,
        base_predictions["market_controls_ridge"]["test"],
        base_predictions["full_ridge"]["test"],
        regime_weights,
    )
    summary["models"]["regime_gated_market_full_blend"] = {
        "family": "regime_gated_blend",
        "base_models": ["market_controls_ridge", "full_ridge"],
        "regime_weights": regime_weights,
        "val": metrics(val_y, regime_blend_val),
        "test": metrics(test_y, regime_blend_test),
        "val_by_regime": build_regime_metrics(val_rows, regime_blend_val),
        "test_by_regime": build_regime_metrics(test_rows, regime_blend_test),
    }

    stack_train_x = np.column_stack(
        [
            base_predictions["market_controls_ridge"]["val_residual"],
            base_predictions["market_controls_hgbr"]["val_residual"],
            base_predictions["ecc_text_ridge"]["val_residual"],
            base_predictions["full_ridge"]["val_residual"],
        ]
    )
    stack_test_x = np.column_stack(
        [
            base_predictions["market_controls_ridge"]["test_residual"],
            base_predictions["market_controls_hgbr"]["test_residual"],
            base_predictions["ecc_text_ridge"]["test_residual"],
            base_predictions["full_ridge"]["test_residual"],
        ]
    )
    stack_feature_names = [
        "market_controls_ridge_residual",
        "market_controls_hgbr_residual",
        "ecc_text_ridge_residual",
        "full_ridge_residual",
    ]
    stack_model = LinearRegression(positive=True)
    stack_model.fit(stack_train_x, val_y - val_prior)
    stack_pred_val = val_prior + stack_model.predict(stack_train_x)
    stack_pred_test = test_prior + stack_model.predict(stack_test_x)
    summary["models"]["positive_stack_base_experts"] = {
        "family": "positive_linear_stack_on_validation",
        "base_models": stack_feature_names,
        "meta_intercept": float(stack_model.intercept_),
        "meta_coefficients": {
            name: float(value) for name, value in zip(stack_feature_names, stack_model.coef_.ravel())
        },
        "val": metrics(val_y, stack_pred_val),
        "test": metrics(test_y, stack_pred_test),
        "val_by_regime": build_regime_metrics(val_rows, stack_pred_val),
        "test_by_regime": build_regime_metrics(test_rows, stack_pred_test),
    }

    gated_train_x = np.hstack([stack_train_x, dense_bundles["gate"]["val"]])
    gated_test_x = np.hstack([stack_test_x, dense_bundles["gate"]["test"]])
    gated_params, gated_model, gated_pred_val = fit_residual_hgbr(
        gated_train_x,
        val_prior,
        val_y,
        gated_train_x,
        val_prior,
        val_y,
        args.random_state,
    )
    gated_pred_test = test_prior + gated_model.predict(gated_test_x)
    summary["models"]["gated_stack_hgbr"] = {
        "family": "validation_trained_gated_stack_hgbr",
        "base_models": stack_feature_names,
        "gate_feature_count": int(dense_bundles["gate"]["val"].shape[1]),
        "best_params": gated_params,
        "val": metrics(val_y, gated_pred_val),
        "test": metrics(test_y, gated_pred_test),
        "val_by_regime": build_regime_metrics(val_rows, gated_pred_val),
        "test_by_regime": build_regime_metrics(test_rows, gated_pred_test),
    }

    regime_suffix = bundle_name_from_regimes(include_regimes)
    html_suffix = "all_html" if not exclude_html_flags else "exclude-" + "-".join(sorted(exclude_html_flags))
    output_path = output_dir / f"hybrid_architecture_{args.target_variant}_{regime_suffix}_{html_suffix}.json"
    write_json(output_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
