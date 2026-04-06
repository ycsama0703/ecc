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

from dj30_qc_utils import safe_float, write_csv, write_json
from run_afterhours_audio_upgrade_benchmark import (
    A4_STRUCTURED_FEATURES,
    PRE_CALL_MARKET_FEATURES,
    build_compressed_dense_bundle,
    load_joined_rows,
)
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle
from run_offhours_shock_ablations import (
    paired_bootstrap_deltas,
    paired_sign_permutation_pvalue,
    regime_label,
)
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_SEMANTIC = "residual_pre_call_market_plus_a4_plus_qna_lsa"
MODEL_SEMANTIC_GATED = "residual_pre_call_market_plus_a4_plus_qna_lsa_observability_gate"
MODEL_SEMANTIC_AUDIO = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd"
MODEL_SEMANTIC_AUDIO_GATED = (
    "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run observability-gated unseen-ticker after-hours audio transfer benchmarks."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_observability_gated_audio_unseen_ticker_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="fail")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min-test-events", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=4)
    parser.add_argument("--aligned-prefix", default="aligned_audio__")
    parser.add_argument("--aligned-compressed-components", type=int, default=8)
    parser.add_argument("--gate-feature", default="a4_strict_row_share")
    parser.add_argument("--gate-quantiles", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def constant_prior(length: int, value: float) -> np.ndarray:
    return np.full(length, float(value), dtype=float)


def parse_quantiles(raw: str) -> list[float]:
    quantiles = [float(item) for item in raw.split(",") if item.strip()]
    return sorted({min(max(value, 0.0), 1.0) for value in quantiles})


def gate_scores(rows: list[dict[str, str]], gate_feature: str) -> np.ndarray:
    return np.asarray([safe_float(row.get(gate_feature)) or 0.0 for row in rows], dtype=float)


def choose_gate_threshold(
    rows: list[dict[str, str]],
    base_pred: np.ndarray,
    branch_pred: np.ndarray,
    gate_feature: str,
    quantiles: list[float],
) -> tuple[float, dict[str, float]]:
    y_true = np.asarray([row["_target"] for row in rows], dtype=float)
    scores = gate_scores(rows, gate_feature)
    candidate_thresholds = sorted({float(np.quantile(scores, q)) for q in quantiles})

    best_payload = None
    for threshold in candidate_thresholds:
        pred = np.where(scores >= threshold, branch_pred, base_pred)
        metric_payload = metrics(y_true, pred)
        payload = {
            "threshold": float(threshold),
            "activation_rate": float(np.mean(scores >= threshold)),
            "rmse": float(metric_payload["rmse"]),
            "r2": float(metric_payload["r2"]),
        }
        if best_payload is None or payload["rmse"] < best_payload["rmse"]:
            best_payload = payload

    assert best_payload is not None
    return best_payload["threshold"], best_payload


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    quantiles = parse_quantiles(args.gate_quantiles)
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    base_rows, coverage = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.old_audio_csv,
        args.aligned_audio_csv,
        args.aligned_prefix,
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
    threshold_rows = []
    skipped = {}

    overall_y = []
    overall_preds = {
        MODEL_PRE_ONLY: [],
        MODEL_SEMANTIC: [],
        MODEL_SEMANTIC_GATED: [],
        MODEL_SEMANTIC_AUDIO: [],
        MODEL_SEMANTIC_AUDIO_GATED: [],
    }
    ticker_summary = {}

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
        test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)

        bundles = {
            "pre_call_market": build_dense_bundle(train_rows, val_rows, test_rows, PRE_CALL_MARKET_FEATURES),
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
        aligned_feature_names = [
            key
            for key in train_rows[0].keys()
            if key.startswith(args.aligned_prefix) and train_rows[0].get(key, "") != ""
        ]
        bundles["aligned_audio_svd"] = build_compressed_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            aligned_feature_names,
            args.aligned_compressed_components,
            prefix="aligned_audio_svd",
        )

        model_specs = {
            MODEL_PRE_ONLY: ["pre_call_market"],
            MODEL_SEMANTIC: ["pre_call_market", "a4", "qna_lsa"],
            MODEL_SEMANTIC_AUDIO: ["pre_call_market", "a4", "qna_lsa", "aligned_audio_svd"],
        }
        preds = {}
        val_preds = {}
        for model_name, bundle_names in model_specs.items():
            train_x = np.hstack([bundles[name]["train"] for name in bundle_names])
            val_x = np.hstack([bundles[name]["val"] for name in bundle_names])
            test_x = np.hstack([bundles[name]["test"] for name in bundle_names])
            _, best_model, val_pred = fit_residual_ridge(
                train_x,
                train_prior,
                train_y,
                val_x,
                val_prior,
                val_y,
                alphas,
            )
            test_pred = test_prior + best_model.predict(test_x)
            preds[model_name] = np.asarray(test_pred, dtype=float)
            val_preds[model_name] = np.asarray(val_pred, dtype=float)

        threshold_sem, threshold_sem_meta = choose_gate_threshold(
            val_rows,
            val_preds[MODEL_PRE_ONLY],
            val_preds[MODEL_SEMANTIC],
            args.gate_feature,
            quantiles,
        )
        threshold_audio, threshold_audio_meta = choose_gate_threshold(
            val_rows,
            val_preds[MODEL_PRE_ONLY],
            val_preds[MODEL_SEMANTIC_AUDIO],
            args.gate_feature,
            quantiles,
        )

        test_scores = gate_scores(test_rows, args.gate_feature)
        preds[MODEL_SEMANTIC_GATED] = np.where(test_scores >= threshold_sem, preds[MODEL_SEMANTIC], preds[MODEL_PRE_ONLY])
        preds[MODEL_SEMANTIC_AUDIO_GATED] = np.where(
            test_scores >= threshold_audio,
            preds[MODEL_SEMANTIC_AUDIO],
            preds[MODEL_PRE_ONLY],
        )

        threshold_rows.append(
            {
                "ticker": ticker,
                "gate_feature": args.gate_feature,
                "threshold_semantic": float(threshold_sem),
                "val_activation_rate_semantic": float(threshold_sem_meta["activation_rate"]),
                "threshold_semantic_audio": float(threshold_audio),
                "val_activation_rate_semantic_audio": float(threshold_audio_meta["activation_rate"]),
                "test_activation_rate_semantic": float(np.mean(test_scores >= threshold_sem)),
                "test_activation_rate_semantic_audio": float(np.mean(test_scores >= threshold_audio)),
                "test_events": len(test_rows),
            }
        )

        ticker_summary[ticker] = {
            MODEL_PRE_ONLY: metrics(test_y, preds[MODEL_PRE_ONLY]) | {"n": len(test_rows)},
            MODEL_SEMANTIC: metrics(test_y, preds[MODEL_SEMANTIC]) | {"n": len(test_rows)},
            MODEL_SEMANTIC_GATED: metrics(test_y, preds[MODEL_SEMANTIC_GATED]) | {"n": len(test_rows)},
            MODEL_SEMANTIC_AUDIO: metrics(test_y, preds[MODEL_SEMANTIC_AUDIO]) | {"n": len(test_rows)},
            MODEL_SEMANTIC_AUDIO_GATED: metrics(test_y, preds[MODEL_SEMANTIC_AUDIO_GATED]) | {"n": len(test_rows)},
            "threshold_semantic": float(threshold_sem),
            "val_activation_rate_semantic": float(threshold_sem_meta["activation_rate"]),
            "threshold_semantic_audio": float(threshold_audio),
            "val_activation_rate_semantic_audio": float(threshold_audio_meta["activation_rate"]),
            "test_activation_rate_semantic": float(np.mean(test_scores >= threshold_sem)),
            "test_activation_rate_semantic_audio": float(np.mean(test_scores >= threshold_audio)),
        }

        overall_y.extend(test_y.tolist())
        for model_name in overall_preds:
            overall_preds[model_name].extend(preds[model_name].tolist())

        for idx, row in enumerate(test_rows):
            prediction_rows.append(
                {
                    "event_key": row["event_key"],
                    "ticker": ticker,
                    "year": row["_year"],
                    "regime": row["_regime"],
                    "target": row["_target"],
                    "gate_feature": args.gate_feature,
                    "gate_value": float(test_scores[idx]),
                    "threshold_semantic": float(threshold_sem),
                    "gate_active_semantic": int(test_scores[idx] >= threshold_sem),
                    "threshold_semantic_audio": float(threshold_audio),
                    "gate_active_semantic_audio": int(test_scores[idx] >= threshold_audio),
                    MODEL_PRE_ONLY: float(preds[MODEL_PRE_ONLY][idx]),
                    MODEL_SEMANTIC: float(preds[MODEL_SEMANTIC][idx]),
                    MODEL_SEMANTIC_GATED: float(preds[MODEL_SEMANTIC_GATED][idx]),
                    MODEL_SEMANTIC_AUDIO: float(preds[MODEL_SEMANTIC_AUDIO][idx]),
                    MODEL_SEMANTIC_AUDIO_GATED: float(preds[MODEL_SEMANTIC_AUDIO_GATED][idx]),
                }
            )

    if not prediction_rows:
        raise SystemExit("no eligible held-out tickers for observability-gated audio unseen-ticker benchmark")

    overall_y_np = np.asarray(overall_y, dtype=float)
    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted(include_regimes),
        "exclude_html_flags": sorted(exclude_html_flags),
        "coverage": coverage,
        "config": {
            "lsa_components": args.lsa_components,
            "aligned_prefix": args.aligned_prefix,
            "aligned_compressed_components": args.aligned_compressed_components,
            "gate_feature": args.gate_feature,
            "gate_quantiles": quantiles,
        },
        "candidate_tickers": len(candidate_tickers),
        "evaluated_tickers": len(ticker_summary),
        "skipped_tickers": skipped,
        "overall_test_size": len(prediction_rows),
        "overall": {
            model_name: metrics(overall_y_np, np.asarray(preds, dtype=float))
            for model_name, preds in overall_preds.items()
        },
        "median_ticker_r2": {
            model_name: float(np.median([payload[model_name]["r2"] for payload in ticker_summary.values()]))
            for model_name in overall_preds
        },
        "mean_test_activation_rate": {
            MODEL_SEMANTIC_GATED: float(
                np.mean([payload["test_activation_rate_semantic"] for payload in ticker_summary.values()])
            ),
            MODEL_SEMANTIC_AUDIO_GATED: float(
                np.mean([payload["test_activation_rate_semantic_audio"] for payload in ticker_summary.values()])
            ),
        },
        "significance": {
            f"{MODEL_SEMANTIC_AUDIO}__vs__{MODEL_SEMANTIC_AUDIO_GATED}": {
                **paired_bootstrap_deltas(
                    overall_y_np,
                    np.asarray(overall_preds[MODEL_SEMANTIC_AUDIO], dtype=float),
                    np.asarray(overall_preds[MODEL_SEMANTIC_AUDIO_GATED], dtype=float),
                    args.bootstrap_iters,
                    args.seed,
                ),
                **paired_sign_permutation_pvalue(
                    overall_y_np,
                    np.asarray(overall_preds[MODEL_SEMANTIC_AUDIO], dtype=float),
                    np.asarray(overall_preds[MODEL_SEMANTIC_AUDIO_GATED], dtype=float),
                    args.perm_iters,
                    args.seed,
                ),
            },
            f"{MODEL_PRE_ONLY}__vs__{MODEL_SEMANTIC_AUDIO_GATED}": {
                **paired_bootstrap_deltas(
                    overall_y_np,
                    np.asarray(overall_preds[MODEL_PRE_ONLY], dtype=float),
                    np.asarray(overall_preds[MODEL_SEMANTIC_AUDIO_GATED], dtype=float),
                    args.bootstrap_iters,
                    args.seed,
                ),
                **paired_sign_permutation_pvalue(
                    overall_y_np,
                    np.asarray(overall_preds[MODEL_PRE_ONLY], dtype=float),
                    np.asarray(overall_preds[MODEL_SEMANTIC_AUDIO_GATED], dtype=float),
                    args.perm_iters,
                    args.seed,
                ),
            },
            f"{MODEL_SEMANTIC_GATED}__vs__{MODEL_SEMANTIC_AUDIO_GATED}": {
                **paired_bootstrap_deltas(
                    overall_y_np,
                    np.asarray(overall_preds[MODEL_SEMANTIC_GATED], dtype=float),
                    np.asarray(overall_preds[MODEL_SEMANTIC_AUDIO_GATED], dtype=float),
                    args.bootstrap_iters,
                    args.seed,
                ),
                **paired_sign_permutation_pvalue(
                    overall_y_np,
                    np.asarray(overall_preds[MODEL_SEMANTIC_GATED], dtype=float),
                    np.asarray(overall_preds[MODEL_SEMANTIC_AUDIO_GATED], dtype=float),
                    args.perm_iters,
                    args.seed,
                ),
            },
        },
        "by_ticker": ticker_summary,
    }

    ordered_rows = sorted(prediction_rows, key=lambda item: (item["ticker"], item["year"], item["event_key"]))
    write_csv(output_dir / "afterhours_observability_gated_audio_unseen_ticker_predictions.csv", ordered_rows)
    write_csv(output_dir / "afterhours_observability_gated_audio_unseen_ticker_thresholds.csv", threshold_rows)
    write_json(output_dir / "afterhours_observability_gated_audio_unseen_ticker_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
