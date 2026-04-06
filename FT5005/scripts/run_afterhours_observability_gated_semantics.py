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
from run_afterhours_precall_semantic_ladder import (
    A4_STRUCTURED_FEATURES,
    PRE_CALL_MARKET_FEATURES,
)
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle, load_joined_rows
from run_offhours_shock_ablations import (
    paired_bootstrap_deltas,
    paired_sign_permutation_pvalue,
    regime_label,
)
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_SEMANTIC = "residual_pre_call_market_plus_a4_plus_qna_lsa"
MODEL_GATED = "residual_pre_call_market_plus_a4_plus_qna_lsa_observability_gate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run observability-gated after-hours semantic checkpoints on fixed-split and unseen-ticker settings."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_observability_gated_semantics_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="fail")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--fixed-lsa-components", type=int, default=64)
    parser.add_argument("--unseen-lsa-components", type=int, default=4)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--min-test-events", type=int, default=3)
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
    semantic_pred: np.ndarray,
    gate_feature: str,
    quantiles: list[float],
) -> tuple[float, dict[str, float]]:
    y_true = np.asarray([row["_target"] for row in rows], dtype=float)
    scores = gate_scores(rows, gate_feature)
    candidate_thresholds = sorted({float(np.quantile(scores, q)) for q in quantiles})

    best_payload = None
    for threshold in candidate_thresholds:
        pred = np.where(scores >= threshold, semantic_pred, base_pred)
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


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
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

    return attach_ticker_prior(rows, args.train_end_year)


def fit_pre_and_semantic_predictions(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    *,
    lsa_components: int,
    alphas: list[float],
    max_features: int,
    min_df: int,
    prior_mode: str,
) -> dict[str, np.ndarray]:
    bundles = {
        "pre_call_market": build_dense_bundle(train_rows, val_rows, test_rows, PRE_CALL_MARKET_FEATURES),
        "a4": build_dense_bundle(train_rows, val_rows, test_rows, A4_STRUCTURED_FEATURES),
        "qna_lsa": build_text_lsa_bundle(
            train_rows,
            val_rows,
            test_rows,
            text_col="qna_text",
            max_features=max_features,
            min_df=min_df,
            lsa_components=lsa_components,
        ),
    }

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)

    if prior_mode == "ticker":
        train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
        val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
        test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)
    elif prior_mode == "global":
        prior_value = float(np.mean(train_y))
        train_prior = constant_prior(len(train_rows), prior_value)
        val_prior = constant_prior(len(val_rows), prior_value)
        test_prior = constant_prior(len(test_rows), prior_value)
    else:
        raise ValueError(f"unsupported prior mode: {prior_mode}")

    model_specs = {
        MODEL_PRE_ONLY: ["pre_call_market"],
        MODEL_SEMANTIC: ["pre_call_market", "a4", "qna_lsa"],
    }

    out = {
        "train_prior": train_prior,
        "val_prior": val_prior,
        "test_prior": test_prior,
    }
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
        out[f"{model_name}_val"] = np.asarray(val_pred, dtype=float)
        out[f"{model_name}_test"] = np.asarray(test_pred, dtype=float)
    return out


def build_fixed_summary(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    alphas: list[float],
    quantiles: list[float],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    train_rows = [row for row in rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in rows if row["_year"] == args.val_year]
    test_rows = [row for row in rows if row["_year"] > args.val_year]

    pred_payload = fit_pre_and_semantic_predictions(
        train_rows,
        val_rows,
        test_rows,
        lsa_components=args.fixed_lsa_components,
        alphas=alphas,
        max_features=args.max_features,
        min_df=args.min_df,
        prior_mode="ticker",
    )

    threshold, threshold_meta = choose_gate_threshold(
        val_rows,
        pred_payload[f"{MODEL_PRE_ONLY}_val"],
        pred_payload[f"{MODEL_SEMANTIC}_val"],
        args.gate_feature,
        quantiles,
    )

    test_scores = gate_scores(test_rows, args.gate_feature)
    gated_pred = np.where(
        test_scores >= threshold,
        pred_payload[f"{MODEL_SEMANTIC}_test"],
        pred_payload[f"{MODEL_PRE_ONLY}_test"],
    )
    y_test = np.asarray([row["_target"] for row in test_rows], dtype=float)

    prediction_rows = []
    for idx, row in enumerate(test_rows):
        prediction_rows.append(
            {
                "split": "fixed_test",
                "event_key": row["event_key"],
                "ticker": row["ticker"],
                "year": row["_year"],
                "regime": row["_regime"],
                "target": row["_target"],
                "gate_feature": args.gate_feature,
                "gate_value": float(test_scores[idx]),
                "gate_threshold": float(threshold),
                "gate_active": int(test_scores[idx] >= threshold),
                MODEL_PRE_ONLY: float(pred_payload[f"{MODEL_PRE_ONLY}_test"][idx]),
                MODEL_SEMANTIC: float(pred_payload[f"{MODEL_SEMANTIC}_test"][idx]),
                MODEL_GATED: float(gated_pred[idx]),
            }
        )

    summary = {
        "lsa_components": args.fixed_lsa_components,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "gate_feature": args.gate_feature,
        "selected_threshold": float(threshold),
        "selected_threshold_metrics": threshold_meta,
        "test_activation_rate": float(np.mean(test_scores >= threshold)),
        "models": {
            MODEL_PRE_ONLY: {"test": metrics(y_test, pred_payload[f"{MODEL_PRE_ONLY}_test"])},
            MODEL_SEMANTIC: {"test": metrics(y_test, pred_payload[f"{MODEL_SEMANTIC}_test"])},
            MODEL_GATED: {"test": metrics(y_test, gated_pred)},
        },
        "significance": {
            f"{MODEL_SEMANTIC}__vs__{MODEL_GATED}": {
                **paired_bootstrap_deltas(
                    y_test,
                    pred_payload[f"{MODEL_SEMANTIC}_test"],
                    gated_pred,
                    args.bootstrap_iters,
                    args.seed,
                ),
                **paired_sign_permutation_pvalue(
                    y_test,
                    pred_payload[f"{MODEL_SEMANTIC}_test"],
                    gated_pred,
                    args.perm_iters,
                    args.seed,
                ),
            },
            f"{MODEL_PRE_ONLY}__vs__{MODEL_GATED}": {
                **paired_bootstrap_deltas(
                    y_test,
                    pred_payload[f"{MODEL_PRE_ONLY}_test"],
                    gated_pred,
                    args.bootstrap_iters,
                    args.seed,
                ),
                **paired_sign_permutation_pvalue(
                    y_test,
                    pred_payload[f"{MODEL_PRE_ONLY}_test"],
                    gated_pred,
                    args.perm_iters,
                    args.seed,
                ),
            },
        },
    }
    return summary, prediction_rows


def build_unseen_summary(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    alphas: list[float],
    quantiles: list[float],
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]]]:
    candidate_tickers = sorted({row["ticker"] for row in rows if row["_year"] > args.val_year and row.get("ticker")})
    skipped = {}
    prediction_rows = []
    threshold_rows = []

    y_true_all = []
    pre_only_all = []
    semantic_all = []
    gated_all = []
    ticker_summary = {}

    for ticker in candidate_tickers:
        train_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] <= args.train_end_year]
        val_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] == args.val_year]
        test_rows = [row for row in rows if row["ticker"] == ticker and row["_year"] > args.val_year]

        if len(test_rows) < args.min_test_events or not train_rows or not val_rows:
            skipped[ticker] = {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}
            continue

        pred_payload = fit_pre_and_semantic_predictions(
            train_rows,
            val_rows,
            test_rows,
            lsa_components=args.unseen_lsa_components,
            alphas=alphas,
            max_features=args.max_features,
            min_df=args.min_df,
            prior_mode="global",
        )

        threshold, threshold_meta = choose_gate_threshold(
            val_rows,
            pred_payload[f"{MODEL_PRE_ONLY}_val"],
            pred_payload[f"{MODEL_SEMANTIC}_val"],
            args.gate_feature,
            quantiles,
        )
        test_scores = gate_scores(test_rows, args.gate_feature)
        gated_pred = np.where(
            test_scores >= threshold,
            pred_payload[f"{MODEL_SEMANTIC}_test"],
            pred_payload[f"{MODEL_PRE_ONLY}_test"],
        )
        y_true = np.asarray([row["_target"] for row in test_rows], dtype=float)

        y_true_all.extend(y_true.tolist())
        pre_only_all.extend(pred_payload[f"{MODEL_PRE_ONLY}_test"].tolist())
        semantic_all.extend(pred_payload[f"{MODEL_SEMANTIC}_test"].tolist())
        gated_all.extend(gated_pred.tolist())

        ticker_summary[ticker] = {
            "threshold": float(threshold),
            "val_activation_rate": float(threshold_meta["activation_rate"]),
            "test_activation_rate": float(np.mean(test_scores >= threshold)),
            MODEL_PRE_ONLY: metrics(y_true, pred_payload[f"{MODEL_PRE_ONLY}_test"]) | {"n": len(test_rows)},
            MODEL_SEMANTIC: metrics(y_true, pred_payload[f"{MODEL_SEMANTIC}_test"]) | {"n": len(test_rows)},
            MODEL_GATED: metrics(y_true, gated_pred) | {"n": len(test_rows)},
        }
        threshold_rows.append(
            {
                "ticker": ticker,
                "threshold": float(threshold),
                "val_activation_rate": float(threshold_meta["activation_rate"]),
                "test_activation_rate": float(np.mean(test_scores >= threshold)),
                "test_events": len(test_rows),
            }
        )

        for idx, row in enumerate(test_rows):
            prediction_rows.append(
                {
                    "split": "unseen_ticker_test",
                    "event_key": row["event_key"],
                    "ticker": ticker,
                    "year": row["_year"],
                    "regime": row["_regime"],
                    "target": row["_target"],
                    "gate_feature": args.gate_feature,
                    "gate_value": float(test_scores[idx]),
                    "gate_threshold": float(threshold),
                    "gate_active": int(test_scores[idx] >= threshold),
                    MODEL_PRE_ONLY: float(pred_payload[f"{MODEL_PRE_ONLY}_test"][idx]),
                    MODEL_SEMANTIC: float(pred_payload[f"{MODEL_SEMANTIC}_test"][idx]),
                    MODEL_GATED: float(gated_pred[idx]),
                }
            )

    if not prediction_rows:
        raise SystemExit("no eligible held-out tickers for observability-gated unseen-ticker semantic test")

    y_true_all_np = np.asarray(y_true_all, dtype=float)
    pre_only_all_np = np.asarray(pre_only_all, dtype=float)
    semantic_all_np = np.asarray(semantic_all, dtype=float)
    gated_all_np = np.asarray(gated_all, dtype=float)

    summary = {
        "lsa_components": args.unseen_lsa_components,
        "gate_feature": args.gate_feature,
        "candidate_tickers": len(candidate_tickers),
        "evaluated_tickers": len(ticker_summary),
        "skipped_tickers": skipped,
        "overall_test_size": len(prediction_rows),
        "overall": {
            MODEL_PRE_ONLY: metrics(y_true_all_np, pre_only_all_np),
            MODEL_SEMANTIC: metrics(y_true_all_np, semantic_all_np),
            MODEL_GATED: metrics(y_true_all_np, gated_all_np),
        },
        "median_ticker_r2": {
            MODEL_PRE_ONLY: float(np.median([payload[MODEL_PRE_ONLY]["r2"] for payload in ticker_summary.values()])),
            MODEL_SEMANTIC: float(np.median([payload[MODEL_SEMANTIC]["r2"] for payload in ticker_summary.values()])),
            MODEL_GATED: float(np.median([payload[MODEL_GATED]["r2"] for payload in ticker_summary.values()])),
        },
        "mean_test_activation_rate": float(np.mean([payload["test_activation_rate"] for payload in ticker_summary.values()])),
        "significance": {
            f"{MODEL_SEMANTIC}__vs__{MODEL_GATED}": {
                **paired_bootstrap_deltas(
                    y_true_all_np,
                    semantic_all_np,
                    gated_all_np,
                    args.bootstrap_iters,
                    args.seed,
                ),
                **paired_sign_permutation_pvalue(
                    y_true_all_np,
                    semantic_all_np,
                    gated_all_np,
                    args.perm_iters,
                    args.seed,
                ),
            },
            f"{MODEL_PRE_ONLY}__vs__{MODEL_GATED}": {
                **paired_bootstrap_deltas(
                    y_true_all_np,
                    pre_only_all_np,
                    gated_all_np,
                    args.bootstrap_iters,
                    args.seed,
                ),
                **paired_sign_permutation_pvalue(
                    y_true_all_np,
                    pre_only_all_np,
                    gated_all_np,
                    args.perm_iters,
                    args.seed,
                ),
            },
        },
        "by_ticker": ticker_summary,
    }
    return summary, prediction_rows, threshold_rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    quantiles = parse_quantiles(args.gate_quantiles)
    rows = build_rows(args)

    fixed_summary, fixed_predictions = build_fixed_summary(rows, args, alphas, quantiles)
    unseen_summary, unseen_predictions, threshold_rows = build_unseen_summary(rows, args, alphas, quantiles)

    write_csv(output_dir / "afterhours_observability_gated_fixed_predictions.csv", fixed_predictions)
    write_csv(output_dir / "afterhours_observability_gated_unseen_predictions.csv", unseen_predictions)
    write_csv(output_dir / "afterhours_observability_gated_unseen_thresholds.csv", threshold_rows)

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted({item.strip() for item in args.include_regimes.split(",") if item.strip()}),
        "exclude_html_flags": sorted(
            {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
        ),
        "gate_feature": args.gate_feature,
        "fixed_split": fixed_summary,
        "unseen_ticker": unseen_summary,
    }
    write_json(output_dir / "afterhours_observability_gated_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
