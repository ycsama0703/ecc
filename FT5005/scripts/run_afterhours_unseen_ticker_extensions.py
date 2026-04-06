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
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle, load_joined_rows
from run_offhours_shock_ablations import regime_label
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_signal_decomposition_benchmarks import CONTROL_FEATURES
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


MARKET_FEATURES = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "pre_60m_volume_sum",
    "within_call_rv",
    "within_call_vw_rv",
    "within_call_volume_sum",
]

A4_STRUCTURED_FEATURES = [
    "a4_kept_rows_for_duration",
    "a4_median_match_score",
    "a4_strict_row_share",
    "a4_broad_row_share",
    "a4_hard_fail_rows",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unseen-ticker stress tests for the after-hours A4 plus Q&A extension line."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_unseen_ticker_extensions_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min-test-events", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=64)
    return parser.parse_args()


def constant_prior(length: int, value: float) -> np.ndarray:
    return np.full(length, float(value), dtype=float)


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
        reg = regime_label(row)
        if reg not in include_regimes:
            continue
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        item["_regime"] = reg
        rows.append(item)

    candidate_tickers = sorted(
        {row["ticker"] for row in rows if row["_year"] > args.val_year and row.get("ticker")}
    )

    prediction_rows = []
    skipped = {}

    for ticker in candidate_tickers:
        train_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] <= args.train_end_year]
        val_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] == args.val_year]
        test_rows = [row for row in rows if row["ticker"] == ticker and row["_year"] > args.val_year]

        if len(test_rows) < args.min_test_events or not train_rows or not val_rows:
            skipped[ticker] = {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}
            continue

        global_prior = float(np.mean([row["_target"] for row in train_rows]))
        train_prior = constant_prior(len(train_rows), global_prior)
        val_prior = constant_prior(len(val_rows), global_prior)
        test_prior = constant_prior(len(test_rows), global_prior)
        train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
        val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)

        bundles = {
            "market": build_dense_bundle(train_rows, val_rows, test_rows, MARKET_FEATURES),
            "controls": build_dense_bundle(train_rows, val_rows, test_rows, CONTROL_FEATURES),
            "a4": build_dense_bundle(train_rows, val_rows, test_rows, A4_STRUCTURED_FEATURES),
            "qna_lsa": build_text_lsa_bundle(
                train_rows,
                val_rows,
                test_rows,
                text_col="qna_text",
                max_features=args.max_features,
                min_df=args.min_df,
                lsa_components=args.lsa_components,
            ),
        }

        model_specs = {
            "residual_market_plus_controls": ["market", "controls"],
            "residual_market_controls_plus_a4": ["market", "controls", "a4"],
            "residual_market_controls_plus_a4_plus_qna_lsa": ["market", "controls", "a4", "qna_lsa"],
        }

        preds_by_model = {"prior_only": test_prior}
        for model_name, bundle_names in model_specs.items():
            train_x = np.hstack([bundles[name]["train"] for name in bundle_names])
            val_x = np.hstack([bundles[name]["val"] for name in bundle_names])
            test_x = np.hstack([bundles[name]["test"] for name in bundle_names])
            _, best_model, _ = fit_residual_ridge(
                train_x,
                train_prior,
                train_y,
                val_x,
                val_prior,
                val_y,
                alphas,
            )
            preds_by_model[model_name] = test_prior + best_model.predict(test_x)

        for idx, row in enumerate(test_rows):
            out_row = {
                "event_key": row["event_key"],
                "ticker": ticker,
                "year": row["_year"],
                "regime": row["_regime"],
                "target": row["_target"],
            }
            for model_name, preds in preds_by_model.items():
                out_row[model_name] = float(preds[idx])
            prediction_rows.append(out_row)

    if not prediction_rows:
        raise SystemExit("no eligible held-out tickers for unseen-ticker extension stress test")

    ordered_prediction_rows = sorted(prediction_rows, key=lambda item: (item["ticker"], item["year"], item["event_key"]))
    write_csv(output_dir / "afterhours_unseen_ticker_predictions.csv", ordered_prediction_rows)

    model_names = [
        "prior_only",
        "residual_market_plus_controls",
        "residual_market_controls_plus_a4",
        "residual_market_controls_plus_a4_plus_qna_lsa",
    ]
    y_true = np.asarray([row["target"] for row in ordered_prediction_rows], dtype=float)
    ticker_summary = {}
    for ticker in sorted({row["ticker"] for row in ordered_prediction_rows}):
        ticker_rows = [row for row in ordered_prediction_rows if row["ticker"] == ticker]
        ticker_summary[ticker] = {
            model_name: metrics(
                np.asarray([row["target"] for row in ticker_rows], dtype=float),
                np.asarray([row[model_name] for row in ticker_rows], dtype=float),
            )
            | {"n": len(ticker_rows)}
            for model_name in model_names
        }

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted(include_regimes),
        "exclude_html_flags": sorted(exclude_html_flags),
        "candidate_tickers": len(candidate_tickers),
        "evaluated_tickers": len(ticker_summary),
        "skipped_tickers": skipped,
        "overall_test_size": len(ordered_prediction_rows),
        "overall": {
            model_name: metrics(
                y_true,
                np.asarray([row[model_name] for row in ordered_prediction_rows], dtype=float),
            )
            for model_name in model_names
        },
        "median_ticker_r2": {
            model_name: float(np.median([payload[model_name]["r2"] for payload in ticker_summary.values()]))
            for model_name in model_names
        },
        "by_ticker": ticker_summary,
    }
    write_json(output_dir / "afterhours_unseen_ticker_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
