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
    build_text_lsa_bundle,
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
        description="Run residual-on-prior experiments on a selected call-timing subset."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/regime_subset_experiments_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="pre_market,after_hours")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def regime_label(row: dict[str, str]) -> str:
    hour = float(row.get("scheduled_hour_et", 0.0))
    if hour < 9.5:
        return "pre_market"
    if hour < 16.0:
        return "market_hours"
    return "after_hours"


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}

    base_rows = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.audio_csv,
        "post_call_60m_rv",
        args.qa_csv,
    )

    rows = []
    for row in base_rows:
        regime = regime_label(row)
        if regime not in include_regimes:
            continue
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        item["_regime"] = regime
        rows.append(item)

    rows = attach_ticker_prior(rows, args.train_end_year)
    train_rows = [row for row in rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in rows if row["_year"] == args.val_year]
    test_rows = [row for row in rows if row["_year"] > args.val_year]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    dense_features = STRUCTURED_FEATURES + EXTRA_DENSE_FEATURES + infer_prefixed_feature_names(rows, "qa_bench_")
    dense_bundle = build_dense_bundle(train_rows, val_rows, test_rows, dense_features)
    lsa_bundle = build_text_lsa_bundle(
        train_rows,
        val_rows,
        test_rows,
        text_col="qna_text",
        max_features=8000,
        min_df=2,
        lsa_components=64,
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
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "prior_only": {
            "val": metrics(val_y, val_prior),
            "test": metrics(test_y, test_prior),
        },
    }

    model_specs = {
        "residual_dense": [dense_bundle],
        "residual_dense_plus_qna_lsa": [dense_bundle, lsa_bundle],
    }
    for model_name, bundles in model_specs.items():
        train_x = np.hstack([bundle["train"] for bundle in bundles])
        val_x = np.hstack([bundle["val"] for bundle in bundles])
        test_x = np.hstack([bundle["test"] for bundle in bundles])
        best_alpha, best_model, pred_val = fit_residual_ridge(
            train_x, train_prior, train_y, val_x, val_prior, val_y, alphas
        )
        pred_test = test_prior + best_model.predict(test_x)
        summary[model_name] = {
            "best_alpha": best_alpha,
            "val": metrics(val_y, pred_val),
            "test": metrics(test_y, pred_test),
        }

    write_json(
        output_dir / f"regime_subset_summary_{args.target_variant}_{'-'.join(sorted(include_regimes))}.json",
        summary,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
