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

from dj30_qc_utils import write_json
from run_dense_multimodal_ablation_baselines import (
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
        description="Run regime-specific residual ridge experiments for a target variant."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/regime_residual_experiments_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min-train-per-regime", type=int, default=25)
    return parser.parse_args()


def regime_label(row: dict[str, str]) -> str:
    hour = float(row.get("scheduled_hour_et", 0.0))
    if hour < 9.5:
        return "pre_market"
    if hour < 16.0:
        return "market_hours"
    return "after_hours"


def evaluate_by_regime(rows, predictions):
    overall = metrics(
        np.asarray([row["_target"] for row in rows], dtype=float),
        np.asarray(predictions, dtype=float),
    )
    by_regime = {}
    grouped = {}
    for row, pred in zip(rows, predictions):
        grouped.setdefault(row["_regime"], {"y": [], "p": []})
        grouped[row["_regime"]]["y"].append(row["_target"])
        grouped[row["_regime"]]["p"].append(float(pred))
    for regime, payload in grouped.items():
        by_regime[regime] = metrics(np.asarray(payload["y"], dtype=float), np.asarray(payload["p"], dtype=float))
    return overall, by_regime


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
    rows = []
    for row in base_rows:
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        item["_regime"] = regime_label(row)
        rows.append(item)

    rows = attach_ticker_prior(rows, args.train_end_year)
    train_rows = [row for row in rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in rows if row["_year"] == args.val_year]
    test_rows = [row for row in rows if row["_year"] > args.val_year]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    features = STRUCTURED_FEATURES + EXTRA_DENSE_FEATURES + infer_prefixed_feature_names(rows, "qa_bench_")
    global_bundle = build_dense_bundle(train_rows, val_rows, test_rows, features)

    def arrays(split_rows_, bundle_name):
        if bundle_name == "global":
            return global_bundle
        raise ValueError(bundle_name)

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
    val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
    test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)

    best_alpha, global_model, global_val_pred = fit_residual_ridge(
        global_bundle["train"], train_prior, train_y, global_bundle["val"], val_prior, val_y, alphas
    )
    global_test_pred = test_prior + global_model.predict(global_bundle["test"])
    global_val_overall, global_val_by_regime = evaluate_by_regime(val_rows, global_val_pred)
    global_test_overall, global_test_by_regime = evaluate_by_regime(test_rows, global_test_pred)

    regime_predictions_val = []
    regime_predictions_test = []
    regime_info = {}
    regime_specific_val_by_event = {}
    regime_specific_test_by_event = {}

    for regime in ["pre_market", "market_hours", "after_hours"]:
        regime_train = [row for row in train_rows if row["_regime"] == regime]
        regime_val = [row for row in val_rows if row["_regime"] == regime]
        regime_test = [row for row in test_rows if row["_regime"] == regime]

        if len(regime_train) < args.min_train_per_regime:
            for row in regime_val:
                regime_predictions_val.append((row["event_key"], float(row["prior_ticker_expanding_mean"])))
            for row in regime_test:
                regime_predictions_test.append((row["event_key"], float(row["prior_ticker_expanding_mean"])))
            regime_info[regime] = {
                "mode": "prior_fallback",
                "train_size": len(regime_train),
                "val_size": len(regime_val),
                "test_size": len(regime_test),
            }
            continue

        bundle = build_dense_bundle(regime_train, regime_val, regime_test, features)
        r_train_y = np.asarray([row["_target"] for row in regime_train], dtype=float)
        r_val_y = np.asarray([row["_target"] for row in regime_val], dtype=float)
        r_train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in regime_train], dtype=float)
        r_val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in regime_val], dtype=float)
        r_test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in regime_test], dtype=float)

        r_alpha, r_model, r_val_pred = fit_residual_ridge(
            bundle["train"], r_train_prior, r_train_y, bundle["val"], r_val_prior, r_val_y, alphas
        )
        r_test_pred = r_test_prior + r_model.predict(bundle["test"])

        for row, pred in zip(regime_val, r_val_pred):
            regime_predictions_val.append((row["event_key"], float(pred)))
            regime_specific_val_by_event[row["event_key"]] = float(pred)
        for row, pred in zip(regime_test, r_test_pred):
            regime_predictions_test.append((row["event_key"], float(pred)))
            regime_specific_test_by_event[row["event_key"]] = float(pred)

        regime_info[regime] = {
            "mode": "fit_regime_specific",
            "best_alpha": r_alpha,
            "train_size": len(regime_train),
            "val_size": len(regime_val),
            "test_size": len(regime_test),
        }

    val_pred_map = dict(regime_predictions_val)
    test_pred_map = dict(regime_predictions_test)
    regime_val_pred = [val_pred_map[row["event_key"]] for row in val_rows]
    regime_test_pred = [test_pred_map[row["event_key"]] for row in test_rows]

    global_val_map = {row["event_key"]: float(pred) for row, pred in zip(val_rows, global_val_pred)}
    global_test_map = {row["event_key"]: float(pred) for row, pred in zip(test_rows, global_test_pred)}
    chosen_modes = {}
    for regime in ["pre_market", "market_hours", "after_hours"]:
        regime_specific_val_metrics = evaluate_by_regime(
            [row for row in val_rows if row["_regime"] == regime],
            [val_pred_map[row["event_key"]] for row in val_rows if row["_regime"] == regime],
        )[0]
        global_regime_val_metrics = global_val_by_regime.get(regime)
        if global_regime_val_metrics is None or regime_specific_val_metrics["rmse"] >= global_regime_val_metrics["rmse"]:
            chosen_modes[regime] = "global"
        else:
            chosen_modes[regime] = "regime_specific"

    hybrid_val_pred = []
    for row in val_rows:
        if chosen_modes[row["_regime"]] == "regime_specific":
            hybrid_val_pred.append(regime_specific_val_by_event[row["event_key"]])
        else:
            hybrid_val_pred.append(global_val_map[row["event_key"]])
    hybrid_test_pred = []
    for row in test_rows:
        if chosen_modes[row["_regime"]] == "regime_specific":
            hybrid_test_pred.append(regime_specific_test_by_event[row["event_key"]])
        else:
            hybrid_test_pred.append(global_test_map[row["event_key"]])

    summary = {
        "target_variant": args.target_variant,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "global_residual_model": {
            "best_alpha": best_alpha,
            "val": global_val_overall,
            "test": global_test_overall,
            "val_by_regime": global_val_by_regime,
            "test_by_regime": global_test_by_regime,
        },
        "regime_specific_residual_model": {
            "regime_info": regime_info,
            "val": evaluate_by_regime(val_rows, regime_val_pred)[0],
            "test": evaluate_by_regime(test_rows, regime_test_pred)[0],
            "val_by_regime": evaluate_by_regime(val_rows, regime_val_pred)[1],
            "test_by_regime": evaluate_by_regime(test_rows, regime_test_pred)[1],
        },
        "hybrid_val_selected_model": {
            "chosen_modes": chosen_modes,
            "val": evaluate_by_regime(val_rows, hybrid_val_pred)[0],
            "test": evaluate_by_regime(test_rows, hybrid_test_pred)[0],
            "val_by_regime": evaluate_by_regime(val_rows, hybrid_val_pred)[1],
            "test_by_regime": evaluate_by_regime(test_rows, hybrid_test_pred)[1],
        },
        "prior_only": {
            "val": evaluate_by_regime(val_rows, [float(row["prior_ticker_expanding_mean"]) for row in val_rows])[0],
            "test": evaluate_by_regime(test_rows, [float(row["prior_ticker_expanding_mean"]) for row in test_rows])[0],
            "val_by_regime": evaluate_by_regime(val_rows, [float(row["prior_ticker_expanding_mean"]) for row in val_rows])[1],
            "test_by_regime": evaluate_by_regime(test_rows, [float(row["prior_ticker_expanding_mean"]) for row in test_rows])[1],
        },
    }

    write_json(output_dir / f"regime_residual_summary_{args.target_variant}.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
