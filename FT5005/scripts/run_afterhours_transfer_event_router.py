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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from dj30_qc_utils import safe_float, write_csv, write_json
from run_afterhours_audio_upgrade_benchmark import (
    A4_STRUCTURED_FEATURES,
    PRE_CALL_MARKET_FEATURES,
    build_compressed_dense_bundle,
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


MODEL_EVENT_LOGISTIC = "event_logistic_router_on_gated_transfer_experts"
MODEL_EVENT_TREE = "event_tree_router_on_gated_transfer_experts"

DEFAULT_ROUTER_FEATURES = ",".join(
    [
        "a4_strict_row_share",
        "a4_strict_high_conf_share",
        "a4_strict_segment_count",
        "qa_pair_count",
        "qa_pair_low_overlap_share",
        "qa_bench_direct_answer_share",
        "qa_bench_evasion_score_mean",
        "qa_bench_coverage_mean",
        "aligned_audio__aligned_audio_sentence_count",
        "qna_word_count",
        "answer_to_question_word_ratio",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark event-level routers over the retained gated after-hours transfer experts."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_event_router_real"),
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
    parser.add_argument("--router-c", type=float, default=0.5)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--tree-min-leaf-frac", type=float, default=0.08)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_feature_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def build_router_matrix(
    rows: list[dict[str, object]],
    feature_names: list[str],
    qa_preds: np.ndarray,
    sem_audio_preds: np.ndarray,
    gate_feature: str,
    qa_threshold: float,
    sem_audio_threshold: float,
) -> np.ndarray:
    matrix = []
    for idx, row in enumerate(rows):
        score = safe_float(row.get(gate_feature)) or 0.0
        values = [safe_float(row.get(name)) or 0.0 for name in feature_names]
        qa_pred = float(qa_preds[idx])
        sem_pred = float(sem_audio_preds[idx])
        values.extend(
            [
                1.0 if score >= qa_threshold else 0.0,
                1.0 if score >= sem_audio_threshold else 0.0,
                qa_pred,
                sem_pred,
                sem_pred - qa_pred,
                abs(sem_pred - qa_pred),
            ]
        )
        matrix.append(values)
    return np.asarray(matrix, dtype=float)


def router_feature_names(base_feature_names: list[str]) -> list[str]:
    return list(base_feature_names) + [
        "qa_gate_active",
        "sem_audio_gate_active",
        "qa_expert_pred",
        "sem_audio_expert_pred",
        "sem_minus_qa_pred",
        "abs_sem_minus_qa_pred",
    ]


def fit_event_router_logistic(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    c_value: float,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    counts = np.bincount(train_labels.astype(int), minlength=2)
    if np.count_nonzero(counts) < 2:
        prob = float(1.0 if counts[1] > 0 else 0.0)
        probs = np.full(test_x.shape[0], prob, dtype=float)
        return probs, {
            "mode": "constant",
            "qa_class_share": float(np.mean(train_labels)),
        }
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(train_x, train_labels)
    probs = model.predict_proba(test_x)[:, 1]
    coef = model.named_steps["clf"].coef_[0]
    ranked = sorted(
        zip(feature_names, coef.tolist()),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return np.asarray(probs, dtype=float), {
        "mode": "logistic",
        "qa_class_share": float(np.mean(train_labels)),
        "coef_l1": float(np.sum(np.abs(coef))),
        "coef_max_abs": float(np.max(np.abs(coef))) if coef.size else 0.0,
        "top_features": ranked[:5],
    }


def fit_event_router_tree(
    train_x: np.ndarray,
    train_labels: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    max_depth: int,
    min_leaf_frac: float,
    feature_names: list[str],
) -> tuple[np.ndarray, dict[str, float]]:
    counts = np.bincount(train_labels.astype(int), minlength=2)
    if np.count_nonzero(counts) < 2:
        prob = float(1.0 if counts[1] > 0 else 0.0)
        probs = np.full(test_x.shape[0], prob, dtype=float)
        return probs, {
            "mode": "constant",
            "qa_class_share": float(np.mean(train_labels)),
        }
    min_leaf = max(5, int(math.ceil(train_x.shape[0] * min_leaf_frac)))
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_leaf,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train_x, train_labels)
    probs = model.predict_proba(test_x)[:, 1]
    return np.asarray(probs, dtype=float), {
        "mode": "tree",
        "qa_class_share": float(np.mean(train_labels)),
        "depth": int(model.get_depth()),
        "leaves": int(model.get_n_leaves()),
        "used_features": sorted(
            {
                feature_names[index]
                for index in model.tree_.feature
                if index >= 0 and index < len(feature_names)
            }
        ),
    }


def routed_predictions(
    qa_probs: np.ndarray,
    qa_preds: np.ndarray,
    sem_audio_preds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    choose_qa = np.asarray(qa_probs >= 0.5, dtype=bool)
    preds = np.where(choose_qa, qa_preds, sem_audio_preds)
    return np.asarray(preds, dtype=float), choose_qa.astype(int)


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

    candidate_tickers = sorted(
        {row["ticker"] for row in rows if row["_year"] > args.val_year and row.get("ticker")}
    )

    prediction_rows = []
    detail_rows = []
    skipped = {}
    overall_y = []
    overall_preds = {
        MODEL_PRE_ONLY: [],
        MODEL_QA_EXPERT: [],
        MODEL_SEM_AUDIO_EXPERT: [],
        MODEL_VALIDATION_SELECTED: [],
        MODEL_EVENT_LOGISTIC: [],
        MODEL_EVENT_TREE: [],
    }
    ticker_summary = {}
    choice_counts = {
        MODEL_PRE_ONLY: 0,
        MODEL_QA_EXPERT: 0,
        MODEL_SEM_AUDIO_EXPERT: 0,
    }
    tree_feature_usage_counts = {}
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
        choice_counts[chosen_model] += 1
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
            np.abs(val_y - val_preds[MODEL_QA_EXPERT])
            < np.abs(val_y - val_preds[MODEL_SEM_AUDIO_EXPERT])
        ).astype(int)

        logistic_probs_test, logistic_meta = fit_event_router_logistic(
            val_router_x,
            val_labels,
            test_router_x,
            args.seed,
            args.router_c,
            full_router_feature_names,
        )
        tree_probs_test, tree_meta = fit_event_router_tree(
            val_router_x,
            val_labels,
            test_router_x,
            args.seed,
            args.tree_depth,
            args.tree_min_leaf_frac,
            full_router_feature_names,
        )
        for feature_name in tree_meta.get("used_features", []):
            tree_feature_usage_counts[feature_name] = tree_feature_usage_counts.get(feature_name, 0) + 1
        test_preds[MODEL_EVENT_LOGISTIC], logistic_choose_qa = routed_predictions(
            logistic_probs_test,
            test_preds[MODEL_QA_EXPERT],
            test_preds[MODEL_SEM_AUDIO_EXPERT],
        )
        test_preds[MODEL_EVENT_TREE], tree_choose_qa = routed_predictions(
            tree_probs_test,
            test_preds[MODEL_QA_EXPERT],
            test_preds[MODEL_SEM_AUDIO_EXPERT],
        )

        detail_rows.append(
            {
                "ticker": ticker,
                "test_events": len(test_rows),
                "chosen_model": chosen_model,
                "val_rmse_pre_only": float(metrics(val_y, val_preds[MODEL_PRE_ONLY])["rmse"]),
                "val_rmse_qa_expert": float(metrics(val_y, val_preds[MODEL_QA_EXPERT])["rmse"]),
                "val_rmse_sem_audio_expert": float(metrics(val_y, val_preds[MODEL_SEM_AUDIO_EXPERT])["rmse"]),
                "qa_threshold": float(qa_threshold),
                "qa_val_activation_rate": float(qa_meta["activation_rate"]),
                "qa_test_activation_rate": float(np.mean(test_scores >= qa_threshold)),
                "sem_audio_threshold": float(sem_audio_threshold),
                "sem_audio_val_activation_rate": float(sem_audio_meta["activation_rate"]),
                "sem_audio_test_activation_rate": float(np.mean(test_scores >= sem_audio_threshold)),
                "router_label_qa_share": float(np.mean(val_labels)),
                "event_logistic_test_qa_share": float(np.mean(logistic_choose_qa)),
                "event_tree_test_qa_share": float(np.mean(tree_choose_qa)),
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
        ticker_summary[ticker]["qa_threshold"] = float(qa_threshold)
        ticker_summary[ticker]["sem_audio_threshold"] = float(sem_audio_threshold)
        ticker_summary[ticker]["router_label_qa_share"] = float(np.mean(val_labels))
        ticker_summary[ticker]["event_logistic_test_qa_share"] = float(np.mean(logistic_choose_qa))
        ticker_summary[ticker]["event_tree_test_qa_share"] = float(np.mean(tree_choose_qa))
        ticker_summary[ticker]["event_tree_used_features"] = tree_meta.get("used_features", [])
        ticker_summary[ticker]["event_logistic_top_features"] = logistic_meta.get("top_features", [])

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
                    MODEL_PRE_ONLY: float(test_preds[MODEL_PRE_ONLY][idx]),
                    MODEL_QA_EXPERT: float(test_preds[MODEL_QA_EXPERT][idx]),
                    MODEL_SEM_AUDIO_EXPERT: float(test_preds[MODEL_SEM_AUDIO_EXPERT][idx]),
                    MODEL_VALIDATION_SELECTED: float(test_preds[MODEL_VALIDATION_SELECTED][idx]),
                    MODEL_EVENT_LOGISTIC: float(test_preds[MODEL_EVENT_LOGISTIC][idx]),
                    MODEL_EVENT_TREE: float(test_preds[MODEL_EVENT_TREE][idx]),
                    "event_logistic_choose_qa": int(logistic_choose_qa[idx]),
                    "event_tree_choose_qa": int(tree_choose_qa[idx]),
                }
            )

    if not prediction_rows:
        raise SystemExit("no eligible held-out tickers for transfer event router benchmark")

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
        "ticker_level_choice_counts": choice_counts,
        "event_tree_feature_usage_counts": dict(sorted(tree_feature_usage_counts.items())),
        "overall": {
            model_name: metrics(overall_y_np, np.asarray(preds, dtype=float))
            for model_name, preds in overall_preds.items()
        },
        "median_ticker_r2": {
            model_name: float(np.median([payload[model_name]["r2"] for payload in ticker_summary.values()]))
            for model_name in overall_preds
        },
        "significance": {
            f"{MODEL_SEM_AUDIO_EXPERT}__vs__{MODEL_VALIDATION_SELECTED}": summarize_significance(
                overall_y_np,
                overall_preds[MODEL_SEM_AUDIO_EXPERT],
                overall_preds[MODEL_VALIDATION_SELECTED],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_SEM_AUDIO_EXPERT}__vs__{MODEL_EVENT_LOGISTIC}": summarize_significance(
                overall_y_np,
                overall_preds[MODEL_SEM_AUDIO_EXPERT],
                overall_preds[MODEL_EVENT_LOGISTIC],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_SEM_AUDIO_EXPERT}__vs__{MODEL_EVENT_TREE}": summarize_significance(
                overall_y_np,
                overall_preds[MODEL_SEM_AUDIO_EXPERT],
                overall_preds[MODEL_EVENT_TREE],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_VALIDATION_SELECTED}__vs__{MODEL_EVENT_LOGISTIC}": summarize_significance(
                overall_y_np,
                overall_preds[MODEL_VALIDATION_SELECTED],
                overall_preds[MODEL_EVENT_LOGISTIC],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_VALIDATION_SELECTED}__vs__{MODEL_EVENT_TREE}": summarize_significance(
                overall_y_np,
                overall_preds[MODEL_VALIDATION_SELECTED],
                overall_preds[MODEL_EVENT_TREE],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
        },
        "by_ticker": ticker_summary,
    }

    ordered_rows = sorted(prediction_rows, key=lambda item: (item["ticker"], item["year"], item["event_key"]))
    ordered_detail_rows = sorted(detail_rows, key=lambda item: item["ticker"])
    write_csv(output_dir / "afterhours_transfer_event_router_predictions.csv", ordered_rows)
    write_csv(output_dir / "afterhours_transfer_event_router_details.csv", ordered_detail_rows)
    write_json(output_dir / "afterhours_transfer_event_router_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
