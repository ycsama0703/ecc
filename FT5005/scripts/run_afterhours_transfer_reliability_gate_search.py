#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import site
import sys
from dataclasses import dataclass
from pathlib import Path

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dj30_qc_utils import safe_float, write_csv, write_json
from run_afterhours_audio_upgrade_benchmark import (
    A4_STRUCTURED_FEATURES,
    PRE_CALL_MARKET_FEATURES,
    build_compressed_dense_bundle,
    load_joined_rows,
)
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle
from run_offhours_shock_ablations import regime_label
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_BRANCH = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd"

SIMPLE_GATE_FEATURE = "a4_strict_row_share"
SINGLE_FEATURE_CANDIDATES = [
    "a4_strict_row_share",
    "a4_broad_row_share",
    "mapped_row_share",
    "aligned_audio_sentence_count",
    "qa_align_score_mean",
    "qa_bench_coverage_mean",
    "qa_bench_direct_answer_share",
    "qa_bench_evasion_score_mean",
]
CONJUNCTIVE_FEATURE_CANDIDATES = [
    "a4_strict_row_share",
    "a4_broad_row_share",
    "mapped_row_share",
    "qa_align_score_mean",
    "qa_bench_coverage_mean",
]
LEARNED_GATE_FEATURES = [
    "a4_strict_row_share",
    "a4_broad_row_share",
    "mapped_row_share",
    "aligned_audio_sentence_count",
    "qa_align_score_mean",
    "qa_bench_coverage_mean",
    "qa_bench_direct_answer_share",
    "qa_bench_evasion_score_mean",
    "qa_bench_pair_count",
]


@dataclass
class Fold:
    ticker: str
    val_rows: list[dict[str, str]]
    test_rows: list[dict[str, str]]
    val_pre: np.ndarray
    val_branch: np.ndarray
    test_pre: np.ndarray
    test_branch: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search simple and flexible transfer-side reliability gates for the matched role-aware after-hours audio branch."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_reliability_gate_search_real"),
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
    parser.add_argument("--simple-gate-quantiles", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--search-quantiles", default="0.2,0.4,0.5,0.6,0.8")
    parser.add_argument("--logistic-c", type=float, default=0.1)
    return parser.parse_args()


def constant_prior(length: int, value: float) -> np.ndarray:
    return np.full(length, float(value), dtype=float)


def parse_quantiles(raw: str) -> list[float]:
    quantiles = [float(item) for item in raw.split(",") if item.strip()]
    return sorted({min(max(value, 0.0), 1.0) for value in quantiles})


def feature_vector(rows: list[dict[str, str]], names: list[str]) -> np.ndarray:
    matrix = []
    for row in rows:
        vector = []
        for name in names:
            value = safe_float(row.get(name))
            vector.append(0.0 if value is None or not math.isfinite(value) else float(value))
        matrix.append(vector)
    return np.asarray(matrix, dtype=float)


def one_feature(rows: list[dict[str, str]], name: str) -> np.ndarray:
    return feature_vector(rows, [name])[:, 0]


def choose_threshold(
    y_true: np.ndarray,
    base_pred: np.ndarray,
    branch_pred: np.ndarray,
    scores: np.ndarray,
    quantiles: list[float],
    directions: tuple[str, ...] = ("ge",),
) -> tuple[float, str, float, dict[str, float]]:
    candidate_thresholds = sorted({float(np.quantile(scores, q)) for q in quantiles})
    best = None
    for direction in directions:
        for threshold in candidate_thresholds:
            mask = scores >= threshold if direction == "ge" else scores <= threshold
            pred = np.where(mask, branch_pred, base_pred)
            metric_payload = metrics(y_true, pred)
            payload = {
                "direction": direction,
                "threshold": float(threshold),
                "activation_rate": float(np.mean(mask)),
                "rmse": float(metric_payload["rmse"]),
                "r2": float(metric_payload["r2"]),
            }
            if best is None or payload["rmse"] < best[3]["rmse"]:
                best = (threshold, direction, float(np.mean(mask)), payload)
    assert best is not None
    return best


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    base_rows, _ = load_joined_rows(
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
        rows.append(item)
    return rows


def fit_fold_predictions(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    args: argparse.Namespace,
    alphas: list[float],
) -> Fold:
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
        key for key in train_rows[0].keys() if key.startswith(args.aligned_prefix) and train_rows[0].get(key, "") != ""
    ]
    bundles["aligned_audio_svd"] = build_compressed_dense_bundle(
        train_rows,
        val_rows,
        test_rows,
        aligned_feature_names,
        args.aligned_compressed_components,
        prefix="aligned_audio_svd",
    )

    prior_value = float(np.mean([row["_target"] for row in train_rows]))
    train_prior = constant_prior(len(train_rows), prior_value)
    val_prior = constant_prior(len(val_rows), prior_value)
    test_prior = constant_prior(len(test_rows), prior_value)
    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)

    model_specs = {
        MODEL_PRE_ONLY: ["pre_call_market"],
        MODEL_BRANCH: ["pre_call_market", "a4", "qna_lsa", "aligned_audio_svd"],
    }
    preds = {}
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
        preds[f"{model_name}_val"] = np.asarray(val_pred, dtype=float)
        preds[f"{model_name}_test"] = np.asarray(test_prior + best_model.predict(test_x), dtype=float)

    return Fold(
        ticker=test_rows[0]["ticker"],
        val_rows=val_rows,
        test_rows=test_rows,
        val_pre=preds[f"{MODEL_PRE_ONLY}_val"],
        val_branch=preds[f"{MODEL_BRANCH}_val"],
        test_pre=preds[f"{MODEL_PRE_ONLY}_test"],
        test_branch=preds[f"{MODEL_BRANCH}_test"],
    )


def evaluate_overall(pred_map: dict[str, list[float]], y_true: list[float], by_ticker: dict[str, dict[str, list[float]]]):
    overall = {
        name: metrics(np.asarray(y_true, dtype=float), np.asarray(preds, dtype=float))
        for name, preds in pred_map.items()
    }
    median_ticker_r2 = {
        name: float(
            np.median(
                [
                    metrics(np.asarray(payload["y_true"], dtype=float), np.asarray(payload[name], dtype=float))["r2"]
                    for payload in by_ticker.values()
                ]
            )
        )
        for name in pred_map
    }
    return overall, median_ticker_r2


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    simple_quantiles = parse_quantiles(args.simple_gate_quantiles)
    search_quantiles = parse_quantiles(args.search_quantiles)
    rows = build_rows(args)

    candidate_tickers = sorted({row["ticker"] for row in rows if row["_year"] > args.val_year and row.get("ticker")})
    folds: list[Fold] = []
    skipped = {}
    for ticker in candidate_tickers:
        train_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] <= args.train_end_year]
        val_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] == args.val_year]
        test_rows = [row for row in rows if row["ticker"] == ticker and row["_year"] > args.val_year]
        if len(test_rows) < args.min_test_events or not train_rows or not val_rows:
            skipped[ticker] = {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}
            continue
        folds.append(fit_fold_predictions(train_rows, val_rows, test_rows, args, alphas))

    if not folds:
        raise SystemExit("no eligible held-out tickers for transfer reliability gate search")

    # Shared validation pools for flexible gates.
    pooled_val_y = np.concatenate([np.asarray([row["_target"] for row in fold.val_rows], dtype=float) for fold in folds])
    pooled_val_pre = np.concatenate([fold.val_pre for fold in folds])
    pooled_val_branch = np.concatenate([fold.val_branch for fold in folds])

    # Shared learned logistic gate.
    pooled_val_features = np.vstack([feature_vector(fold.val_rows, LEARNED_GATE_FEATURES) for fold in folds])
    pooled_improve = ((pooled_val_y - pooled_val_branch) ** 2 < (pooled_val_y - pooled_val_pre) ** 2).astype(int)
    scaler = StandardScaler()
    pooled_val_scaled = scaler.fit_transform(pooled_val_features)
    logistic = LogisticRegression(
        C=args.logistic_c,
        class_weight="balanced",
        max_iter=5000,
        random_state=42,
    )
    logistic.fit(pooled_val_scaled, pooled_improve)
    pooled_val_scores = logistic.predict_proba(pooled_val_scaled)[:, 1]
    logistic_threshold, _, _, logistic_meta = choose_threshold(
        pooled_val_y,
        pooled_val_pre,
        pooled_val_branch,
        pooled_val_scores,
        simple_quantiles,
        directions=("ge",),
    )

    # Shared conjunctive gate search.
    best_conjunctive = None
    for idx_a, feature_a in enumerate(CONJUNCTIVE_FEATURE_CANDIDATES):
        score_a = np.concatenate([one_feature(fold.val_rows, feature_a) for fold in folds])
        thresholds_a = sorted({float(np.quantile(score_a, q)) for q in search_quantiles})
        for feature_b in CONJUNCTIVE_FEATURE_CANDIDATES[idx_a + 1 :]:
            score_b = np.concatenate([one_feature(fold.val_rows, feature_b) for fold in folds])
            thresholds_b = sorted({float(np.quantile(score_b, q)) for q in search_quantiles})
            for direction_a in ("ge", "le"):
                for direction_b in ("ge", "le"):
                    for threshold_a in thresholds_a:
                        mask_a = score_a >= threshold_a if direction_a == "ge" else score_a <= threshold_a
                        for threshold_b in thresholds_b:
                            mask_b = score_b >= threshold_b if direction_b == "ge" else score_b <= threshold_b
                            mask = mask_a & mask_b
                            pred = np.where(mask, pooled_val_branch, pooled_val_pre)
                            metric_payload = metrics(pooled_val_y, pred)
                            payload = {
                                "feature_a": feature_a,
                                "direction_a": direction_a,
                                "threshold_a": float(threshold_a),
                                "feature_b": feature_b,
                                "direction_b": direction_b,
                                "threshold_b": float(threshold_b),
                                "activation_rate": float(np.mean(mask)),
                                "rmse": float(metric_payload["rmse"]),
                                "r2": float(metric_payload["r2"]),
                            }
                            if best_conjunctive is None or payload["rmse"] < best_conjunctive["rmse"]:
                                best_conjunctive = payload
    assert best_conjunctive is not None

    model_names = [
        MODEL_PRE_ONLY,
        MODEL_BRANCH,
        f"{MODEL_BRANCH}_simple_local_{SIMPLE_GATE_FEATURE}",
        f"{MODEL_BRANCH}_single_feature_search",
        f"{MODEL_BRANCH}_shared_logistic_gate",
        f"{MODEL_BRANCH}_shared_conjunctive_gate",
    ]
    prediction_rows = []
    pred_map = {name: [] for name in model_names}
    y_true_all = []
    by_ticker = {}
    gate_rows = []

    learned_coef = logistic.coef_[0].tolist()

    for fold in folds:
        y_test = np.asarray([row["_target"] for row in fold.test_rows], dtype=float)
        y_val = np.asarray([row["_target"] for row in fold.val_rows], dtype=float)

        # Simple local A4 strict gate.
        simple_scores_val = one_feature(fold.val_rows, SIMPLE_GATE_FEATURE)
        simple_threshold, simple_direction, _, simple_meta = choose_threshold(
            y_val,
            fold.val_pre,
            fold.val_branch,
            simple_scores_val,
            simple_quantiles,
            directions=("ge",),
        )
        simple_scores_test = one_feature(fold.test_rows, SIMPLE_GATE_FEATURE)
        simple_mask_test = simple_scores_test >= simple_threshold
        simple_pred_test = np.where(simple_mask_test, fold.test_branch, fold.test_pre)

        # Local single-feature search.
        best_single = None
        for feature_name in SINGLE_FEATURE_CANDIDATES:
            feature_scores_val = one_feature(fold.val_rows, feature_name)
            threshold, direction, _, meta = choose_threshold(
                y_val,
                fold.val_pre,
                fold.val_branch,
                feature_scores_val,
                simple_quantiles,
                directions=("ge", "le"),
            )
            if best_single is None or meta["rmse"] < best_single["meta"]["rmse"]:
                best_single = {
                    "feature": feature_name,
                    "direction": direction,
                    "threshold": float(threshold),
                    "meta": meta,
                }
        assert best_single is not None
        single_scores_test = one_feature(fold.test_rows, best_single["feature"])
        if best_single["direction"] == "ge":
            single_mask_test = single_scores_test >= best_single["threshold"]
        else:
            single_mask_test = single_scores_test <= best_single["threshold"]
        single_pred_test = np.where(single_mask_test, fold.test_branch, fold.test_pre)

        # Shared learned logistic gate.
        learned_scores_test = logistic.predict_proba(scaler.transform(feature_vector(fold.test_rows, LEARNED_GATE_FEATURES)))[:, 1]
        learned_mask_test = learned_scores_test >= logistic_threshold
        learned_pred_test = np.where(learned_mask_test, fold.test_branch, fold.test_pre)

        # Shared conjunctive gate.
        conjunctive_scores_a = one_feature(fold.test_rows, best_conjunctive["feature_a"])
        conjunctive_scores_b = one_feature(fold.test_rows, best_conjunctive["feature_b"])
        mask_a = (
            conjunctive_scores_a >= best_conjunctive["threshold_a"]
            if best_conjunctive["direction_a"] == "ge"
            else conjunctive_scores_a <= best_conjunctive["threshold_a"]
        )
        mask_b = (
            conjunctive_scores_b >= best_conjunctive["threshold_b"]
            if best_conjunctive["direction_b"] == "ge"
            else conjunctive_scores_b <= best_conjunctive["threshold_b"]
        )
        conjunctive_mask_test = mask_a & mask_b
        conjunctive_pred_test = np.where(conjunctive_mask_test, fold.test_branch, fold.test_pre)

        fold_preds = {
            MODEL_PRE_ONLY: fold.test_pre,
            MODEL_BRANCH: fold.test_branch,
            f"{MODEL_BRANCH}_simple_local_{SIMPLE_GATE_FEATURE}": simple_pred_test,
            f"{MODEL_BRANCH}_single_feature_search": single_pred_test,
            f"{MODEL_BRANCH}_shared_logistic_gate": learned_pred_test,
            f"{MODEL_BRANCH}_shared_conjunctive_gate": conjunctive_pred_test,
        }

        by_ticker[fold.ticker] = {"y_true": y_test.tolist()}
        for name, preds in fold_preds.items():
            by_ticker[fold.ticker][name] = np.asarray(preds, dtype=float).tolist()
            pred_map[name].extend(np.asarray(preds, dtype=float).tolist())
        y_true_all.extend(y_test.tolist())

        gate_rows.append(
            {
                "ticker": fold.ticker,
                "simple_gate_feature": SIMPLE_GATE_FEATURE,
                "simple_gate_threshold": float(simple_threshold),
                "simple_gate_activation_rate_test": float(np.mean(simple_mask_test)),
                "single_feature_search_feature": best_single["feature"],
                "single_feature_search_direction": best_single["direction"],
                "single_feature_search_threshold": float(best_single["threshold"]),
                "single_feature_search_activation_rate_test": float(np.mean(single_mask_test)),
                "shared_logistic_threshold": float(logistic_threshold),
                "shared_logistic_activation_rate_test": float(np.mean(learned_mask_test)),
                "shared_conjunctive_feature_a": best_conjunctive["feature_a"],
                "shared_conjunctive_direction_a": best_conjunctive["direction_a"],
                "shared_conjunctive_threshold_a": float(best_conjunctive["threshold_a"]),
                "shared_conjunctive_feature_b": best_conjunctive["feature_b"],
                "shared_conjunctive_direction_b": best_conjunctive["direction_b"],
                "shared_conjunctive_threshold_b": float(best_conjunctive["threshold_b"]),
                "shared_conjunctive_activation_rate_test": float(np.mean(conjunctive_mask_test)),
            }
        )

        for idx, row in enumerate(fold.test_rows):
            prediction_rows.append(
                {
                    "event_key": row["event_key"],
                    "ticker": fold.ticker,
                    "year": row["_year"],
                    "target": row["_target"],
                    MODEL_PRE_ONLY: float(fold.test_pre[idx]),
                    MODEL_BRANCH: float(fold.test_branch[idx]),
                    f"{MODEL_BRANCH}_simple_local_{SIMPLE_GATE_FEATURE}": float(simple_pred_test[idx]),
                    f"{MODEL_BRANCH}_single_feature_search": float(single_pred_test[idx]),
                    f"{MODEL_BRANCH}_shared_logistic_gate": float(learned_pred_test[idx]),
                    f"{MODEL_BRANCH}_shared_conjunctive_gate": float(conjunctive_pred_test[idx]),
                }
            )

    overall, median_ticker_r2 = evaluate_overall(pred_map, y_true_all, by_ticker)
    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted({item.strip() for item in args.include_regimes.split(",") if item.strip()}),
        "exclude_html_flags": sorted(
            {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
        ),
        "candidate_tickers": len(candidate_tickers),
        "evaluated_tickers": len(folds),
        "skipped_tickers": skipped,
        "overall_test_size": len(y_true_all),
        "models": overall,
        "median_ticker_r2": median_ticker_r2,
        "gate_search": {
            "simple_local_feature": SIMPLE_GATE_FEATURE,
            "single_feature_candidates": SINGLE_FEATURE_CANDIDATES,
            "learned_gate_features": LEARNED_GATE_FEATURES,
            "learned_logistic_c": args.logistic_c,
            "learned_logistic_threshold": float(logistic_threshold),
            "learned_logistic_val_metrics": logistic_meta,
            "learned_logistic_coefficients": {
                name: float(value) for name, value in zip(LEARNED_GATE_FEATURES, learned_coef)
            },
            "shared_conjunctive_best": best_conjunctive,
        },
        "by_ticker": {
            ticker: {
                name: metrics(np.asarray(payload["y_true"], dtype=float), np.asarray(payload[name], dtype=float))
                | {"n": len(payload["y_true"])}
                for name in model_names
            }
            for ticker, payload in by_ticker.items()
        },
    }

    write_csv(output_dir / "afterhours_transfer_reliability_gate_search_predictions.csv", prediction_rows)
    write_csv(output_dir / "afterhours_transfer_reliability_gate_search_gate_details.csv", gate_rows)
    write_json(output_dir / "afterhours_transfer_reliability_gate_search_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
