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
)
from run_afterhours_transfer_event_router import (
    DEFAULT_ROUTER_FEATURES,
    MODEL_EVENT_LOGISTIC,
    MODEL_EVENT_TREE,
    build_router_matrix,
    fit_event_router_logistic,
    fit_event_router_tree,
    router_feature_names,
)
from run_afterhours_transfer_expert_selection import (
    MODEL_PRE_ONLY,
    MODEL_QA_EXPERT,
    MODEL_SEM_AUDIO_EXPERT,
    MODEL_VALIDATION_SELECTED,
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


MODEL_LOGISTIC_CONSERVATIVE = "conservative_logistic_override_on_selected_expert"
MODEL_TREE_CONSERVATIVE = "conservative_tree_override_on_selected_expert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark conservative event-level overrides on top of the selected transfer expert."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_conservative_router_real"),
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
    parser.add_argument("--router-features", default=DEFAULT_ROUTER_FEATURES)
    parser.add_argument("--router-c", type=float, default=1.0)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--tree-min-leaf-frac", type=float, default=0.08)
    parser.add_argument("--override-thresholds", default="0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_feature_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_thresholds(raw: str) -> list[float]:
    thresholds = [float(item.strip()) for item in raw.split(",") if item.strip()]
    cleaned = sorted({min(max(value, 0.5), 0.999) for value in thresholds})
    return cleaned


def conservative_override_predictions(
    default_model: str,
    qa_probs: np.ndarray,
    qa_preds: np.ndarray,
    sem_preds: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    qa_probs = np.asarray(qa_probs, dtype=float)
    qa_preds = np.asarray(qa_preds, dtype=float)
    sem_preds = np.asarray(sem_preds, dtype=float)
    if default_model == MODEL_QA_EXPERT:
        choose_qa = qa_probs > (1.0 - threshold)
    else:
        choose_qa = qa_probs >= threshold
    preds = np.where(choose_qa, qa_preds, sem_preds)
    return np.asarray(preds, dtype=float), choose_qa.astype(int)


def choose_override_threshold(
    y_true: np.ndarray,
    default_model: str,
    qa_probs: np.ndarray,
    qa_preds: np.ndarray,
    sem_preds: np.ndarray,
    thresholds: list[float],
) -> tuple[float, dict[str, float]]:
    best_threshold = thresholds[0]
    best_payload = None
    for threshold in thresholds:
        preds, choose_qa = conservative_override_predictions(
            default_model,
            qa_probs,
            qa_preds,
            sem_preds,
            threshold,
        )
        score = metrics(y_true, preds)
        payload = {
            "threshold": float(threshold),
            "rmse": float(score["rmse"]),
            "mae": float(score["mae"]),
            "r2": float(score["r2"]),
            "qa_share": float(np.mean(choose_qa)),
        }
        if best_payload is None or payload["rmse"] < best_payload["rmse"]:
            best_payload = payload
            best_threshold = float(threshold)
    return best_threshold, best_payload or {
        "threshold": float(best_threshold),
        "rmse": float("inf"),
        "mae": float("inf"),
        "r2": float("-inf"),
        "qa_share": 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    quantiles = parse_quantiles(args.gate_quantiles)
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    router_features = parse_feature_list(args.router_features)
    full_router_feature_names = router_feature_names(router_features)
    override_thresholds = parse_thresholds(args.override_thresholds)

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
    if not qa_all_feature_names:
        raise SystemExit("no qa_benchmark features found in joined rows")
    if not aligned_feature_names:
        raise SystemExit("no aligned audio features found in joined rows")

    candidate_tickers = sorted({row["ticker"] for row in rows if row["_year"] > args.val_year and row.get("ticker")})

    prediction_rows = []
    detail_rows = []
    skipped = {}
    overall_y = []
    overall_preds = {
        MODEL_PRE_ONLY: [],
        MODEL_QA_EXPERT: [],
        MODEL_SEM_AUDIO_EXPERT: [],
        MODEL_VALIDATION_SELECTED: [],
        MODEL_LOGISTIC_CONSERVATIVE: [],
        MODEL_TREE_CONSERVATIVE: [],
    }
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

        model_specs = {
            MODEL_PRE_ONLY: ["pre_call_market"],
            "qa_svd_branch": ["pre_call_market", "a4", "qa_benchmark_svd"],
            "sem_audio_branch": ["pre_call_market", "a4", "qna_lsa", "aligned_audio_svd"],
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

        qa_threshold, qa_meta = choose_gate_threshold(
            val_rows,
            val_preds[MODEL_PRE_ONLY],
            val_preds["qa_svd_branch"],
            args.gate_feature,
            quantiles,
        )
        sem_audio_threshold, sem_audio_meta = choose_gate_threshold(
            val_rows,
            val_preds[MODEL_PRE_ONLY],
            val_preds["sem_audio_branch"],
            args.gate_feature,
            quantiles,
        )

        val_scores = np.asarray([safe_float(row.get(args.gate_feature)) or 0.0 for row in val_rows], dtype=float)
        test_scores = np.asarray([safe_float(row.get(args.gate_feature)) or 0.0 for row in test_rows], dtype=float)
        val_preds[MODEL_QA_EXPERT] = np.where(
            val_scores >= qa_threshold,
            val_preds["qa_svd_branch"],
            val_preds[MODEL_PRE_ONLY],
        )
        test_preds[MODEL_QA_EXPERT] = np.where(
            test_scores >= qa_threshold,
            test_preds["qa_svd_branch"],
            test_preds[MODEL_PRE_ONLY],
        )
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

        candidate_experts = {
            MODEL_PRE_ONLY: val_preds[MODEL_PRE_ONLY],
            MODEL_QA_EXPERT: val_preds[MODEL_QA_EXPERT],
            MODEL_SEM_AUDIO_EXPERT: val_preds[MODEL_SEM_AUDIO_EXPERT],
        }
        chosen_model = min(candidate_experts, key=lambda name: metrics(val_y, candidate_experts[name])["rmse"])
        val_preds[MODEL_VALIDATION_SELECTED] = np.asarray(val_preds[chosen_model], dtype=float)
        test_preds[MODEL_VALIDATION_SELECTED] = np.asarray(test_preds[chosen_model], dtype=float)

        val_router_x = build_router_matrix(
            val_rows,
            router_features,
            val_preds[MODEL_QA_EXPERT],
            val_preds[MODEL_SEM_AUDIO_EXPERT],
            args.gate_feature,
            qa_threshold,
            sem_audio_threshold,
        )
        test_router_x = build_router_matrix(
            test_rows,
            router_features,
            test_preds[MODEL_QA_EXPERT],
            test_preds[MODEL_SEM_AUDIO_EXPERT],
            args.gate_feature,
            qa_threshold,
            sem_audio_threshold,
        )
        val_labels = (
            np.abs(val_y - val_preds[MODEL_QA_EXPERT]) < np.abs(val_y - val_preds[MODEL_SEM_AUDIO_EXPERT])
        ).astype(int)

        logistic_probs_val, logistic_meta = fit_event_router_logistic(
            val_router_x,
            val_labels,
            val_router_x,
            args.seed,
            args.router_c,
            full_router_feature_names,
        )
        logistic_probs_test, _ = fit_event_router_logistic(
            val_router_x,
            val_labels,
            test_router_x,
            args.seed,
            args.router_c,
            full_router_feature_names,
        )
        tree_probs_val, tree_meta = fit_event_router_tree(
            val_router_x,
            val_labels,
            val_router_x,
            args.seed,
            args.tree_depth,
            args.tree_min_leaf_frac,
            full_router_feature_names,
        )
        tree_probs_test, _ = fit_event_router_tree(
            val_router_x,
            val_labels,
            test_router_x,
            args.seed,
            args.tree_depth,
            args.tree_min_leaf_frac,
            full_router_feature_names,
        )

        logistic_threshold, logistic_threshold_meta = choose_override_threshold(
            val_y,
            chosen_model,
            logistic_probs_val,
            val_preds[MODEL_QA_EXPERT],
            val_preds[MODEL_SEM_AUDIO_EXPERT],
            override_thresholds,
        )
        tree_threshold, tree_threshold_meta = choose_override_threshold(
            val_y,
            chosen_model,
            tree_probs_val,
            val_preds[MODEL_QA_EXPERT],
            val_preds[MODEL_SEM_AUDIO_EXPERT],
            override_thresholds,
        )
        test_preds[MODEL_LOGISTIC_CONSERVATIVE], logistic_choose_qa = conservative_override_predictions(
            chosen_model,
            logistic_probs_test,
            test_preds[MODEL_QA_EXPERT],
            test_preds[MODEL_SEM_AUDIO_EXPERT],
            logistic_threshold,
        )
        test_preds[MODEL_TREE_CONSERVATIVE], tree_choose_qa = conservative_override_predictions(
            chosen_model,
            tree_probs_test,
            test_preds[MODEL_QA_EXPERT],
            test_preds[MODEL_SEM_AUDIO_EXPERT],
            tree_threshold,
        )

        detail_rows.append(
            {
                "ticker": ticker,
                "test_events": len(test_rows),
                "chosen_model": chosen_model,
                "qa_threshold": float(qa_threshold),
                "sem_audio_threshold": float(sem_audio_threshold),
                "qa_val_activation_rate": float(qa_meta["activation_rate"]),
                "sem_audio_val_activation_rate": float(sem_audio_meta["activation_rate"]),
                "logistic_override_threshold": float(logistic_threshold),
                "logistic_val_qa_share": float(logistic_threshold_meta["qa_share"]),
                "logistic_test_qa_share": float(np.mean(logistic_choose_qa)),
                "tree_override_threshold": float(tree_threshold),
                "tree_val_qa_share": float(tree_threshold_meta["qa_share"]),
                "tree_test_qa_share": float(np.mean(tree_choose_qa)),
                "router_label_qa_share": float(np.mean(val_labels)),
                "event_logistic_mode": logistic_meta["mode"],
                "event_tree_mode": tree_meta["mode"],
                "event_tree_depth": tree_meta.get("depth", 0),
                "event_tree_leaves": tree_meta.get("leaves", 0),
                "event_tree_used_features": "|".join(tree_meta.get("used_features", [])),
                "event_logistic_top_features": "|".join(
                    f"{name}:{weight:.4g}" for name, weight in logistic_meta.get("top_features", [])
                ),
            }
        )

        ticker_summary[ticker] = {
            model_name: metrics(test_y, test_preds[model_name]) | {"n": len(test_rows)}
            for model_name in overall_preds
        }
        ticker_summary[ticker]["chosen_model"] = chosen_model
        ticker_summary[ticker]["logistic_override_threshold"] = float(logistic_threshold)
        ticker_summary[ticker]["tree_override_threshold"] = float(tree_threshold)
        ticker_summary[ticker]["event_logistic_top_features"] = logistic_meta.get("top_features", [])
        ticker_summary[ticker]["event_tree_used_features"] = tree_meta.get("used_features", [])

        overall_y.extend(test_y.tolist())
        for model_name in overall_preds:
            overall_preds[model_name].extend(test_preds[model_name].tolist())

        for idx, row in enumerate(test_rows):
            prediction_rows.append(
                {
                    "event_key": row["event_key"],
                    "ticker": ticker,
                    "year": row["_year"],
                    "regime": row["_regime"],
                    "target": row["_target"],
                    "chosen_model": chosen_model,
                    MODEL_PRE_ONLY: float(test_preds[MODEL_PRE_ONLY][idx]),
                    MODEL_QA_EXPERT: float(test_preds[MODEL_QA_EXPERT][idx]),
                    MODEL_SEM_AUDIO_EXPERT: float(test_preds[MODEL_SEM_AUDIO_EXPERT][idx]),
                    MODEL_VALIDATION_SELECTED: float(test_preds[MODEL_VALIDATION_SELECTED][idx]),
                    MODEL_LOGISTIC_CONSERVATIVE: float(test_preds[MODEL_LOGISTIC_CONSERVATIVE][idx]),
                    MODEL_TREE_CONSERVATIVE: float(test_preds[MODEL_TREE_CONSERVATIVE][idx]),
                    "logistic_choose_qa": int(logistic_choose_qa[idx]),
                    "tree_choose_qa": int(tree_choose_qa[idx]),
                }
            )

    if not prediction_rows:
        raise SystemExit("no eligible held-out tickers for conservative router benchmark")

    overall_y_np = np.asarray(overall_y, dtype=float)
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
            "router_features": router_features,
            "router_c": args.router_c,
            "tree_depth": args.tree_depth,
            "tree_min_leaf_frac": args.tree_min_leaf_frac,
            "override_thresholds": override_thresholds,
        },
        "feature_groups": {
            "pre_call_market": PRE_CALL_MARKET_FEATURES,
            "a4": A4_STRUCTURED_FEATURES,
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
        "significance": {
            f"{MODEL_VALIDATION_SELECTED}__vs__{MODEL_LOGISTIC_CONSERVATIVE}": summarize_significance(
                overall_y_np,
                overall_preds[MODEL_VALIDATION_SELECTED],
                overall_preds[MODEL_LOGISTIC_CONSERVATIVE],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_VALIDATION_SELECTED}__vs__{MODEL_TREE_CONSERVATIVE}": summarize_significance(
                overall_y_np,
                overall_preds[MODEL_VALIDATION_SELECTED],
                overall_preds[MODEL_TREE_CONSERVATIVE],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_SEM_AUDIO_EXPERT}__vs__{MODEL_LOGISTIC_CONSERVATIVE}": summarize_significance(
                overall_y_np,
                overall_preds[MODEL_SEM_AUDIO_EXPERT],
                overall_preds[MODEL_LOGISTIC_CONSERVATIVE],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_SEM_AUDIO_EXPERT}__vs__{MODEL_TREE_CONSERVATIVE}": summarize_significance(
                overall_y_np,
                overall_preds[MODEL_SEM_AUDIO_EXPERT],
                overall_preds[MODEL_TREE_CONSERVATIVE],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
        },
        "by_ticker": ticker_summary,
    }

    ordered_rows = sorted(prediction_rows, key=lambda item: (item["ticker"], item["year"], item["event_key"]))
    ordered_detail_rows = sorted(detail_rows, key=lambda item: item["ticker"])
    write_csv(output_dir / "afterhours_transfer_conservative_router_predictions.csv", ordered_rows)
    write_csv(output_dir / "afterhours_transfer_conservative_router_details.csv", ordered_detail_rows)
    write_json(output_dir / "afterhours_transfer_conservative_router_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
