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
from run_afterhours_audio_upgrade_benchmark import (
    A4_STRUCTURED_FEATURES,
    PRE_CALL_MARKET_FEATURES,
    build_compressed_dense_bundle,
)
from run_afterhours_transfer_expert_selection import (
    MODEL_PRE_ONLY,
    MODEL_SEM_AUDIO_EXPERT,
    constant_prior,
    parse_quantiles,
    summarize_significance,
)
from run_afterhours_transfer_qa_signal_benchmark import (
    choose_gate_threshold,
    load_joined_rows_with_qa_and_aligned,
)
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle, infer_prefixed_feature_names
from run_offhours_shock_ablations import regime_label
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


PAIR_CORE_FEATURES = [
    "qa_pair_count",
    "qa_pair_low_overlap_share",
    "qa_pair_question_words_mean",
    "qa_pair_answer_words_mean",
    "qa_pair_answer_digit_rate_mean",
    "qa_pair_answer_hedge_rate_mean",
    "qa_pair_answer_assertive_rate_mean",
    "qa_pair_answer_forward_rate_mean",
    "qa_multi_part_question_share",
    "qa_evasive_proxy_share",
]

TEXT_LITE_FEATURES = [
    "answer_to_question_word_ratio",
    "qna_word_count",
    "question_mark_per_1k_words",
]

MODEL_QA_SVD_EXPERT = "residual_pre_call_market_plus_a4_plus_qa_benchmark_svd_observability_gate"
MODEL_QA_PAIR_EXPERT = "residual_pre_call_market_plus_a4_plus_qa_pair_core_observability_gate"
MODEL_HYBRID_PAIR_BENCH_EXPERT = "residual_pre_call_market_plus_a4_plus_hybrid_pair_bench_observability_gate"
MODEL_HYBRID_PLUS_TEXT_EXPERT = "residual_pre_call_market_plus_a4_plus_hybrid_plus_text_observability_gate"
MODEL_HYBRID_PAIR_BENCH_AUDIO_EXPERT = (
    "residual_pre_call_market_plus_a4_plus_hybrid_pair_bench_plus_aligned_audio_svd_observability_gate"
)
MODEL_HYBRID_PLUS_TEXT_AUDIO_EXPERT = (
    "residual_pre_call_market_plus_a4_plus_hybrid_plus_text_plus_aligned_audio_svd_observability_gate"
)

BRANCH_SPECS = {
    MODEL_QA_SVD_EXPERT: ["pre_call_market", "a4", "qa_benchmark_svd"],
    MODEL_QA_PAIR_EXPERT: ["pre_call_market", "a4", "qa_pair_core"],
    MODEL_HYBRID_PAIR_BENCH_EXPERT: ["pre_call_market", "a4", "qa_pair_core", "qa_benchmark_svd"],
    MODEL_HYBRID_PLUS_TEXT_EXPERT: [
        "pre_call_market",
        "a4",
        "qa_pair_core",
        "qa_benchmark_svd",
        "text_lite",
    ],
    MODEL_HYBRID_PAIR_BENCH_AUDIO_EXPERT: [
        "pre_call_market",
        "a4",
        "qa_pair_core",
        "qa_benchmark_svd",
        "aligned_audio_svd",
    ],
    MODEL_HYBRID_PLUS_TEXT_AUDIO_EXPERT: [
        "pre_call_market",
        "a4",
        "qa_pair_core",
        "qa_benchmark_svd",
        "text_lite",
        "aligned_audio_svd",
    ],
}

SELECTED_MODEL_NAMES = {
    MODEL_QA_SVD_EXPERT: "validation_selected_transfer_expert__qa_benchmark_svd",
    MODEL_QA_PAIR_EXPERT: "validation_selected_transfer_expert__qa_pair_core",
    MODEL_HYBRID_PAIR_BENCH_EXPERT: "validation_selected_transfer_expert__hybrid_pair_bench",
    MODEL_HYBRID_PLUS_TEXT_EXPERT: "validation_selected_transfer_expert__hybrid_plus_text",
    MODEL_HYBRID_PAIR_BENCH_AUDIO_EXPERT: "validation_selected_transfer_expert__hybrid_pair_bench_audio",
    MODEL_HYBRID_PLUS_TEXT_AUDIO_EXPERT: "validation_selected_transfer_expert__hybrid_plus_text_audio",
}

BRANCH_FAMILY_LABELS = {
    MODEL_QA_SVD_EXPERT: "qa_benchmark_svd",
    MODEL_QA_PAIR_EXPERT: "qa_pair_core",
    MODEL_HYBRID_PAIR_BENCH_EXPERT: "hybrid_pair_bench",
    MODEL_HYBRID_PLUS_TEXT_EXPERT: "hybrid_plus_text",
    MODEL_HYBRID_PAIR_BENCH_AUDIO_EXPERT: "hybrid_pair_bench_audio",
    MODEL_HYBRID_PLUS_TEXT_AUDIO_EXPERT: "hybrid_plus_text_audio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark stronger complementary transfer experts against the retained semantic+audio expert."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_complementary_expert_benchmark_real"),
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


def median_ticker_r2(ticker_summary: dict[str, dict[str, object]], model_name: str) -> float:
    values = []
    for payload in ticker_summary.values():
        per_model = payload.get("model_r2", {})
        value = per_model.get(model_name)
        if value is not None and math.isfinite(value):
            values.append(float(value))
    return float(np.median(values)) if values else float("nan")


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
    aligned_feature_names = [
        key for key in rows[0].keys() if key.startswith(args.aligned_prefix) and rows[0].get(key, "") != ""
    ]
    pair_feature_names = [name for name in PAIR_CORE_FEATURES if name in rows[0]]
    text_lite_feature_names = [name for name in TEXT_LITE_FEATURES if name in rows[0]]

    if not qa_all_feature_names:
        raise SystemExit("no qa_benchmark features found in joined rows")
    if not aligned_feature_names:
        raise SystemExit("no aligned audio features found in joined rows")
    if not pair_feature_names:
        raise SystemExit("no pair_core features found in joined rows")
    if not text_lite_feature_names:
        raise SystemExit("no text_lite features found in joined rows")

    candidate_tickers = sorted({row["ticker"] for row in rows if row["_year"] > args.val_year and row.get("ticker")})

    prediction_rows = []
    selection_rows = []
    skipped = {}
    overall_y = []
    overall_preds = {
        MODEL_PRE_ONLY: [],
        MODEL_SEM_AUDIO_EXPERT: [],
        MODEL_QA_SVD_EXPERT: [],
        MODEL_QA_PAIR_EXPERT: [],
        MODEL_HYBRID_PAIR_BENCH_EXPERT: [],
        MODEL_HYBRID_PLUS_TEXT_EXPERT: [],
        **{name: [] for name in SELECTED_MODEL_NAMES.values()},
    }
    ticker_summary = {}
    selection_counts = {
        selected_model_name: {MODEL_PRE_ONLY: 0, MODEL_SEM_AUDIO_EXPERT: 0, branch_model: 0}
        for branch_model, selected_model_name in SELECTED_MODEL_NAMES.items()
    }
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
            "qa_pair_core": build_dense_bundle(train_rows, val_rows, test_rows, pair_feature_names),
            "text_lite": build_dense_bundle(train_rows, val_rows, test_rows, text_lite_feature_names),
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
                "qa_pair_core_features": pair_feature_names,
                "text_lite_features": text_lite_feature_names,
            }

        model_specs = {
            MODEL_PRE_ONLY: ["pre_call_market"],
            "sem_audio_branch": ["pre_call_market", "a4", "qna_lsa", "aligned_audio_svd"],
            **{branch_model: bundle_names for branch_model, bundle_names in BRANCH_SPECS.items()},
        }
        val_preds = {}
        test_preds = {}
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
            val_preds[model_name] = np.asarray(val_pred, dtype=float)
            test_preds[model_name] = np.asarray(test_prior + best_model.predict(test_x), dtype=float)

        sem_audio_threshold, sem_audio_meta = choose_gate_threshold(
            val_rows,
            val_preds[MODEL_PRE_ONLY],
            val_preds["sem_audio_branch"],
            args.gate_feature,
            quantiles,
        )
        val_scores = np.asarray([float(row.get(args.gate_feature) or 0.0) for row in val_rows], dtype=float)
        test_scores = np.asarray([float(row.get(args.gate_feature) or 0.0) for row in test_rows], dtype=float)
        val_preds[MODEL_SEM_AUDIO_EXPERT] = np.where(
            val_scores >= sem_audio_threshold,
            val_preds["sem_audio_branch"],
            val_preds[MODEL_PRE_ONLY],
        )
        test_preds[MODEL_SEM_AUDIO_EXPERT] = np.where(
            test_scores >= sem_audio_threshold,
            test_preds["sem_audio_branch"],
            test_preds[MODEL_PRE_ONLY],
        )

        gate_meta = {MODEL_SEM_AUDIO_EXPERT: sem_audio_meta}
        for branch_model in BRANCH_SPECS:
            threshold, branch_meta = choose_gate_threshold(
                val_rows,
                val_preds[MODEL_PRE_ONLY],
                val_preds[branch_model],
                args.gate_feature,
                quantiles,
            )
            val_preds[branch_model] = np.where(
                val_scores >= threshold,
                val_preds[branch_model],
                val_preds[MODEL_PRE_ONLY],
            )
            test_preds[branch_model] = np.where(
                test_scores >= threshold,
                test_preds[branch_model],
                test_preds[MODEL_PRE_ONLY],
            )
            gate_meta[branch_model] = branch_meta

        family_choices = {}
        for branch_model, selected_model_name in SELECTED_MODEL_NAMES.items():
            candidates = {
                MODEL_PRE_ONLY: val_preds[MODEL_PRE_ONLY],
                MODEL_SEM_AUDIO_EXPERT: val_preds[MODEL_SEM_AUDIO_EXPERT],
                branch_model: val_preds[branch_model],
            }
            chosen_model = min(candidates, key=lambda name: metrics(val_y, candidates[name])["rmse"])
            selection_counts[selected_model_name][chosen_model] += 1
            family_choices[selected_model_name] = chosen_model
            test_preds[selected_model_name] = np.asarray(test_preds[chosen_model], dtype=float)
            selection_rows.append(
                {
                    "ticker": ticker,
                    "selected_model_name": selected_model_name,
                    "candidate_branch": branch_model,
                    "chosen_model": chosen_model,
                    "test_events": len(test_rows),
                    "threshold_candidate": gate_meta[branch_model]["threshold"],
                    "threshold_sem_audio": gate_meta[MODEL_SEM_AUDIO_EXPERT]["threshold"],
                    "val_r2_pre_only": metrics(val_y, val_preds[MODEL_PRE_ONLY])["r2"],
                    "val_r2_sem_audio": metrics(val_y, val_preds[MODEL_SEM_AUDIO_EXPERT])["r2"],
                    "val_r2_candidate": metrics(val_y, val_preds[branch_model])["r2"],
                }
            )

        model_r2 = {}
        model_rmse = {}
        for model_name in overall_preds:
            metric_payload = metrics(test_y, test_preds[model_name])
            model_r2[model_name] = float(metric_payload["r2"])
            model_rmse[model_name] = float(metric_payload["rmse"])

        ticker_summary[ticker] = {
            "test_events": len(test_rows),
            "model_r2": model_r2,
            "model_rmse": model_rmse,
            "selected_choices": family_choices,
            "gate_meta": gate_meta,
        }

        for idx, row in enumerate(test_rows):
            record = {
                "event_key": row["event_key"],
                "ticker": ticker,
                "year": row["_year"],
                "target": float(test_y[idx]),
            }
            for model_name in overall_preds:
                record[model_name] = float(test_preds[model_name][idx])
                overall_preds[model_name].append(float(test_preds[model_name][idx]))
            overall_y.append(float(test_y[idx]))
            prediction_rows.append(record)

    if not overall_y:
        raise SystemExit("no held-out ticker predictions available")

    y_true = np.asarray(overall_y, dtype=float)
    overall = {model_name: metrics(y_true, np.asarray(preds, dtype=float)) for model_name, preds in overall_preds.items()}
    qa_selected_model = SELECTED_MODEL_NAMES[MODEL_QA_SVD_EXPERT]
    significance = {
        f"{qa_selected_model}__vs__{MODEL_SEM_AUDIO_EXPERT}": summarize_significance(
            y_true,
            overall_preds[MODEL_SEM_AUDIO_EXPERT],
            overall_preds[qa_selected_model],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
    }
    for branch_model, selected_model_name in SELECTED_MODEL_NAMES.items():
        if selected_model_name != qa_selected_model:
            significance[f"{selected_model_name}__vs__{qa_selected_model}"] = summarize_significance(
                y_true,
                overall_preds[qa_selected_model],
                overall_preds[selected_model_name],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            )
        significance[f"{selected_model_name}__vs__{MODEL_SEM_AUDIO_EXPERT}"] = summarize_significance(
            y_true,
            overall_preds[MODEL_SEM_AUDIO_EXPERT],
            overall_preds[selected_model_name],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        )
        significance[f"{selected_model_name}__vs__{MODEL_PRE_ONLY}"] = summarize_significance(
            y_true,
            overall_preds[MODEL_PRE_ONLY],
            overall_preds[selected_model_name],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        )

    median_ticker = {
        model_name: median_ticker_r2(ticker_summary, model_name)
        for model_name in overall_preds
    }

    summary = {
        "config": {
            "target_variant": args.target_variant,
            "include_regimes": sorted(include_regimes),
            "exclude_html_flags": sorted(exclude_html_flags),
            "train_end_year": args.train_end_year,
            "val_year": args.val_year,
            "lsa_components": args.lsa_components,
            "qa_compressed_components": args.qa_compressed_components,
            "aligned_compressed_components": args.aligned_compressed_components,
            "gate_feature": args.gate_feature,
            "gate_quantiles": quantiles,
        },
        "coverage": coverage,
        "feature_meta": bundle_meta,
        "branch_families": BRANCH_FAMILY_LABELS,
        "overall": overall,
        "median_ticker_r2": median_ticker,
        "selection_counts": selection_counts,
        "skipped_tickers": skipped,
        "significance": significance,
        "ticker_summary": ticker_summary,
    }

    write_csv(output_dir / "afterhours_transfer_complementary_expert_predictions.csv", prediction_rows)
    write_csv(output_dir / "afterhours_transfer_complementary_expert_selection.csv", selection_rows)
    write_json(output_dir / "afterhours_transfer_complementary_expert_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
