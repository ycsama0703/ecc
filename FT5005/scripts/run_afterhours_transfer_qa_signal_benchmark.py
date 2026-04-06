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

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json
from run_afterhours_audio_upgrade_benchmark import (
    A4_STRUCTURED_FEATURES,
    PRE_CALL_MARKET_FEATURES,
    build_compressed_dense_bundle,
    load_prefixed_lookup,
)
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle, infer_prefixed_feature_names
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
MODEL_SEMANTIC_GATED = f"{MODEL_SEMANTIC}_observability_gate"
MODEL_QA_CORE = "residual_pre_call_market_plus_a4_plus_qa_quality_core"
MODEL_QA_CORE_GATED = f"{MODEL_QA_CORE}_observability_gate"
MODEL_QA_SVD = "residual_pre_call_market_plus_a4_plus_qa_benchmark_svd"
MODEL_QA_SVD_GATED = f"{MODEL_QA_SVD}_observability_gate"
MODEL_SEMANTIC_QA_CORE = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_qa_quality_core"
MODEL_SEMANTIC_QA_CORE_GATED = f"{MODEL_SEMANTIC_QA_CORE}_observability_gate"
MODEL_SEMANTIC_QA_SVD = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_qa_benchmark_svd"
MODEL_SEMANTIC_QA_SVD_GATED = f"{MODEL_SEMANTIC_QA_SVD}_observability_gate"
MODEL_SEMANTIC_AUDIO = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd"
MODEL_SEMANTIC_AUDIO_GATED = f"{MODEL_SEMANTIC_AUDIO}_observability_gate"
MODEL_SEMANTIC_QA_CORE_AUDIO = (
    "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_qa_quality_core_plus_aligned_audio_svd"
)
MODEL_SEMANTIC_QA_CORE_AUDIO_GATED = f"{MODEL_SEMANTIC_QA_CORE_AUDIO}_observability_gate"
MODEL_SEMANTIC_QA_SVD_AUDIO = (
    "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_qa_benchmark_svd_plus_aligned_audio_svd"
)
MODEL_SEMANTIC_QA_SVD_AUDIO_GATED = f"{MODEL_SEMANTIC_QA_SVD_AUDIO}_observability_gate"

QA_CORE_FEATURES = [
    "qa_bench_coverage_mean",
    "qa_bench_direct_answer_share",
    "qa_bench_evasion_score_mean",
    "qa_bench_pair_count",
    "qa_bench_specificity_gap_mean",
    "qa_bench_delay_share_mean",
    "qa_bench_short_evasive_share",
    "qa_bench_numeric_mismatch_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark transfer-side Q&A signal upgrades under the fixed simple after-hours observability gate."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_qa_signal_benchmark_real"),
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
    parser.add_argument("--qa-compressed-components", type=int, default=8)
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


def load_joined_rows_with_qa_and_aligned(
    panel_csv: Path,
    features_csv: Path,
    old_audio_csv: Path,
    aligned_audio_csv: Path,
    qa_csv: Path,
    aligned_prefix: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    feature_lookup = {}
    for row in load_csv_rows(features_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            feature_lookup[event_key] = row

    old_audio_lookup = {}
    for row in load_csv_rows(old_audio_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            old_audio_lookup[event_key] = row

    aligned_audio_lookup = load_prefixed_lookup(
        aligned_audio_csv.resolve(),
        prefix=aligned_prefix,
        meta_fields={"event_key", "ticker", "year", "quarter"},
    )

    qa_lookup = {}
    for row in load_csv_rows(qa_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            qa_lookup[event_key] = row

    coverage = {
        "panel_rows": 0,
        "with_features": 0,
        "with_old_audio": 0,
        "with_aligned_audio": 0,
        "with_qa": 0,
        "with_all_side_inputs": 0,
        "joined_rows": 0,
    }

    rows = []
    for row in load_csv_rows(panel_csv.resolve()):
        coverage["panel_rows"] += 1
        event_key = row.get("event_key", "")
        year_value = row.get("year", "")
        feature_row = feature_lookup.get(event_key)
        old_audio_row = old_audio_lookup.get(event_key)
        aligned_audio_row = aligned_audio_lookup.get(event_key)
        qa_row = qa_lookup.get(event_key)

        if feature_row is not None:
            coverage["with_features"] += 1
        if old_audio_row is not None:
            coverage["with_old_audio"] += 1
        if aligned_audio_row is not None:
            coverage["with_aligned_audio"] += 1
        if qa_row is not None:
            coverage["with_qa"] += 1
        if feature_row is not None and old_audio_row is not None and aligned_audio_row is not None and qa_row is not None:
            coverage["with_all_side_inputs"] += 1

        if (
            not event_key
            or not year_value
            or feature_row is None
            or old_audio_row is None
            or aligned_audio_row is None
            or qa_row is None
        ):
            continue

        merged = dict(row)
        merged.update(feature_row)
        merged.update(old_audio_row)
        merged.update(aligned_audio_row)
        merged.update(qa_row)
        merged["_year"] = int(float(year_value))
        rows.append(merged)

    coverage["joined_rows"] = len(rows)
    return rows, coverage


def summarize_significance(
    y_true: np.ndarray,
    pred_a: list[float],
    pred_b: list[float],
    bootstrap_iters: int,
    perm_iters: int,
    seed: int,
) -> dict[str, float]:
    pred_a_np = np.asarray(pred_a, dtype=float)
    pred_b_np = np.asarray(pred_b, dtype=float)
    return {
        **paired_bootstrap_deltas(y_true, pred_a_np, pred_b_np, bootstrap_iters, seed),
        **paired_sign_permutation_pvalue(y_true, pred_a_np, pred_b_np, perm_iters, seed),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    quantiles = parse_quantiles(args.gate_quantiles)
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    base_rows, coverage = load_joined_rows_with_qa_and_aligned(
        args.panel_csv,
        args.features_csv,
        args.old_audio_csv,
        args.aligned_audio_csv,
        args.qa_csv,
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

    if not rows:
        raise SystemExit("no rows available after joining and filtering")

    qa_all_feature_names = infer_prefixed_feature_names(rows, "qa_bench_")
    qa_core_feature_names = [name for name in QA_CORE_FEATURES if name in qa_all_feature_names]
    aligned_feature_names = [
        key for key in rows[0].keys() if key.startswith(args.aligned_prefix) and rows[0].get(key, "") != ""
    ]

    if not qa_core_feature_names:
        raise SystemExit("no qa_quality_core features found in joined rows")
    if not qa_all_feature_names:
        raise SystemExit("no qa_benchmark features found in joined rows")
    if not aligned_feature_names:
        raise SystemExit("no aligned audio features found in joined rows")

    candidate_tickers = sorted(
        {row["ticker"] for row in rows if row["_year"] > args.val_year and row.get("ticker")}
    )

    branch_specs = {
        MODEL_SEMANTIC: ["pre_call_market", "a4", "qna_lsa"],
        MODEL_QA_CORE: ["pre_call_market", "a4", "qa_quality_core"],
        MODEL_QA_SVD: ["pre_call_market", "a4", "qa_benchmark_svd"],
        MODEL_SEMANTIC_QA_CORE: ["pre_call_market", "a4", "qna_lsa", "qa_quality_core"],
        MODEL_SEMANTIC_QA_SVD: ["pre_call_market", "a4", "qna_lsa", "qa_benchmark_svd"],
        MODEL_SEMANTIC_AUDIO: ["pre_call_market", "a4", "qna_lsa", "aligned_audio_svd"],
        MODEL_SEMANTIC_QA_CORE_AUDIO: [
            "pre_call_market",
            "a4",
            "qna_lsa",
            "qa_quality_core",
            "aligned_audio_svd",
        ],
        MODEL_SEMANTIC_QA_SVD_AUDIO: [
            "pre_call_market",
            "a4",
            "qna_lsa",
            "qa_benchmark_svd",
            "aligned_audio_svd",
        ],
    }
    gated_model_names = {
        MODEL_SEMANTIC: MODEL_SEMANTIC_GATED,
        MODEL_QA_CORE: MODEL_QA_CORE_GATED,
        MODEL_QA_SVD: MODEL_QA_SVD_GATED,
        MODEL_SEMANTIC_QA_CORE: MODEL_SEMANTIC_QA_CORE_GATED,
        MODEL_SEMANTIC_QA_SVD: MODEL_SEMANTIC_QA_SVD_GATED,
        MODEL_SEMANTIC_AUDIO: MODEL_SEMANTIC_AUDIO_GATED,
        MODEL_SEMANTIC_QA_CORE_AUDIO: MODEL_SEMANTIC_QA_CORE_AUDIO_GATED,
        MODEL_SEMANTIC_QA_SVD_AUDIO: MODEL_SEMANTIC_QA_SVD_AUDIO_GATED,
    }

    prediction_rows = []
    threshold_rows = []
    skipped = {}

    overall_y = []
    overall_preds = {MODEL_PRE_ONLY: []}
    for model_name in branch_specs:
        overall_preds[model_name] = []
        overall_preds[gated_model_names[model_name]] = []

    ticker_summary = {}
    bundle_meta = None

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
            "qa_quality_core": build_dense_bundle(train_rows, val_rows, test_rows, qa_core_feature_names),
            "qa_benchmark_svd": build_compressed_dense_bundle(
                train_rows,
                val_rows,
                test_rows,
                qa_all_feature_names,
                args.qa_compressed_components,
                prefix="qa_benchmark_svd",
            ),
            "aligned_audio_svd": build_compressed_dense_bundle(
                train_rows,
                val_rows,
                test_rows,
                aligned_feature_names,
                args.aligned_compressed_components,
                prefix="aligned_audio_svd",
            ),
        }

        if bundle_meta is None:
            bundle_meta = {
                "qna_lsa_components": int(bundles["qna_lsa"]["train"].shape[1]),
                "qa_benchmark_svd": {
                    "input_feature_count": len(qa_all_feature_names),
                    "n_components": bundles["qa_benchmark_svd"]["n_components"],
                    "explained_variance_ratio_sum": bundles["qa_benchmark_svd"]["explained_variance_ratio_sum"],
                },
                "aligned_audio_svd": {
                    "input_feature_count": len(aligned_feature_names),
                    "n_components": bundles["aligned_audio_svd"]["n_components"],
                    "explained_variance_ratio_sum": bundles["aligned_audio_svd"]["explained_variance_ratio_sum"],
                },
            }

        preds = {}
        val_preds = {}
        train_x = bundles["pre_call_market"]["train"]
        val_x = bundles["pre_call_market"]["val"]
        test_x = bundles["pre_call_market"]["test"]
        _, best_model, val_pred = fit_residual_ridge(
            train_x,
            train_prior,
            train_y,
            val_x,
            val_prior,
            val_y,
            alphas,
        )
        preds[MODEL_PRE_ONLY] = np.asarray(test_prior + best_model.predict(test_x), dtype=float)
        val_preds[MODEL_PRE_ONLY] = np.asarray(val_pred, dtype=float)

        for model_name, bundle_names in branch_specs.items():
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
            preds[model_name] = np.asarray(test_prior + best_model.predict(test_x), dtype=float)
            val_preds[model_name] = np.asarray(val_pred, dtype=float)

        test_scores = gate_scores(test_rows, args.gate_feature)
        ticker_summary[ticker] = {MODEL_PRE_ONLY: metrics(test_y, preds[MODEL_PRE_ONLY]) | {"n": len(test_rows)}}

        for model_name, gated_name in gated_model_names.items():
            threshold, threshold_meta = choose_gate_threshold(
                val_rows,
                val_preds[MODEL_PRE_ONLY],
                val_preds[model_name],
                args.gate_feature,
                quantiles,
            )
            preds[gated_name] = np.where(test_scores >= threshold, preds[model_name], preds[MODEL_PRE_ONLY])
            threshold_rows.append(
                {
                    "ticker": ticker,
                    "model_name": model_name,
                    "gated_model_name": gated_name,
                    "gate_feature": args.gate_feature,
                    "threshold": float(threshold),
                    "val_activation_rate": float(threshold_meta["activation_rate"]),
                    "test_activation_rate": float(np.mean(test_scores >= threshold)),
                    "test_events": len(test_rows),
                }
            )
            ticker_summary[ticker][model_name] = metrics(test_y, preds[model_name]) | {"n": len(test_rows)}
            ticker_summary[ticker][gated_name] = metrics(test_y, preds[gated_name]) | {"n": len(test_rows)}
            ticker_summary[ticker][f"threshold__{gated_name}"] = float(threshold)
            ticker_summary[ticker][f"val_activation_rate__{gated_name}"] = float(threshold_meta["activation_rate"])
            ticker_summary[ticker][f"test_activation_rate__{gated_name}"] = float(np.mean(test_scores >= threshold))

        overall_y.extend(test_y.tolist())
        for model_name in overall_preds:
            overall_preds[model_name].extend(preds[model_name].tolist())

        threshold_lookup = {row["gated_model_name"]: row for row in threshold_rows if row["ticker"] == ticker}
        for idx, row in enumerate(test_rows):
            out_row = {
                "event_key": row["event_key"],
                "ticker": ticker,
                "year": row["_year"],
                "regime": row["_regime"],
                "target": row["_target"],
                "gate_feature": args.gate_feature,
                "gate_value": float(test_scores[idx]),
                MODEL_PRE_ONLY: float(preds[MODEL_PRE_ONLY][idx]),
            }
            for model_name, gated_name in gated_model_names.items():
                threshold_payload = threshold_lookup[gated_name]
                out_row[model_name] = float(preds[model_name][idx])
                out_row[gated_name] = float(preds[gated_name][idx])
                out_row[f"threshold__{gated_name}"] = float(threshold_payload["threshold"])
                out_row[f"gate_active__{gated_name}"] = int(test_scores[idx] >= threshold_payload["threshold"])
            prediction_rows.append(out_row)

    if not prediction_rows:
        raise SystemExit("no eligible held-out tickers for transfer QA signal benchmark")

    overall_y_np = np.asarray(overall_y, dtype=float)
    reference_model = MODEL_SEMANTIC_AUDIO_GATED
    significance = {
        f"{MODEL_PRE_ONLY}__vs__{reference_model}": summarize_significance(
            overall_y_np,
            overall_preds[MODEL_PRE_ONLY],
            overall_preds[reference_model],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        )
    }
    for gated_name in gated_model_names.values():
        if gated_name == reference_model:
            continue
        significance[f"{reference_model}__vs__{gated_name}"] = summarize_significance(
            overall_y_np,
            overall_preds[reference_model],
            overall_preds[gated_name],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        )

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted(include_regimes),
        "exclude_html_flags": sorted(exclude_html_flags),
        "coverage": coverage,
        "config": {
            "lsa_components": args.lsa_components,
            "qa_compressed_components": args.qa_compressed_components,
            "aligned_prefix": args.aligned_prefix,
            "aligned_compressed_components": args.aligned_compressed_components,
            "gate_feature": args.gate_feature,
            "gate_quantiles": quantiles,
        },
        "feature_groups": {
            "pre_call_market": PRE_CALL_MARKET_FEATURES,
            "a4": A4_STRUCTURED_FEATURES,
            "qa_quality_core": qa_core_feature_names,
            "qa_benchmark_all_count": len(qa_all_feature_names),
            "qna_lsa_components": bundle_meta["qna_lsa_components"] if bundle_meta else None,
            "qa_benchmark_svd": bundle_meta["qa_benchmark_svd"] if bundle_meta else None,
            "aligned_audio_svd": bundle_meta["aligned_audio_svd"] if bundle_meta else None,
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
            gated_name: float(np.mean([payload[f"test_activation_rate__{gated_name}"] for payload in ticker_summary.values()]))
            for gated_name in gated_model_names.values()
        },
        "significance": significance,
        "by_ticker": ticker_summary,
    }

    ordered_rows = sorted(prediction_rows, key=lambda item: (item["ticker"], item["year"], item["event_key"]))
    ordered_threshold_rows = sorted(threshold_rows, key=lambda item: (item["ticker"], item["gated_model_name"]))
    write_csv(output_dir / "afterhours_transfer_qa_signal_predictions.csv", ordered_rows)
    write_csv(output_dir / "afterhours_transfer_qa_signal_thresholds.csv", ordered_threshold_rows)
    write_json(output_dir / "afterhours_transfer_qa_signal_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
