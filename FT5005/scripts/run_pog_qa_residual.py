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
from run_afterhours_precall_semantic_ladder import A4_STRUCTURED_FEATURES, PRE_CALL_MARKET_FEATURES
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle
from run_offhours_shock_ablations import (
    paired_bootstrap_deltas,
    paired_sign_permutation_pvalue,
    regime_label,
)
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_signal_decomposition_benchmarks import CONTROL_FEATURES
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


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

TRUST_GATE_FEATURES = [
    "a4_kept_rows_for_duration",
    "a4_median_match_score",
    "a4_strict_row_share",
    "a4_broad_row_share",
    "qa_bench_coverage_mean",
    "qa_bench_direct_answer_share",
    "qa_bench_evasion_score_mean",
    "qa_bench_pair_count",
    "qa_pair_count",
    "qa_pair_low_overlap_share",
    "qa_bench_short_evasive_share",
    "qa_bench_numeric_mismatch_share",
]

ROUTE_GATE_FEATURES = [
    "qa_bench_coverage_mean",
    "qa_bench_direct_answer_share",
    "qa_bench_evasion_score_mean",
    "qa_bench_specificity_gap_mean",
    "qa_bench_delay_share_mean",
    "qa_bench_question_complexity_mean",
    "qa_bench_numeric_mismatch_share",
    "qa_bench_short_evasive_share",
    "qa_pair_count",
    "qa_pair_question_words_mean",
]

TRUST_GATE_DIRECTION_BY_FEATURE = {
    "a4_kept_rows_for_duration": 1.0,
    "a4_median_match_score": 1.0,
    "a4_strict_row_share": 1.0,
    "a4_broad_row_share": 1.0,
    "qa_bench_coverage_mean": 1.0,
    "qa_bench_direct_answer_share": 1.0,
    "qa_bench_pair_count": 1.0,
    "qa_pair_count": 1.0,
    "qa_bench_evasion_score_mean": -1.0,
    "qa_pair_low_overlap_share": -1.0,
    "qa_bench_short_evasive_share": -1.0,
    "qa_bench_numeric_mismatch_share": -1.0,
    "html_integrity_flag=pass": 1.0,
    "html_integrity_flag=warn": -1.0,
    "html_integrity_flag=fail": -1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Incremental Prior-Aware Observability-Gated Q&A Residual "
            "(I-POG-QA) model. The model keeps a stable structured residual core, "
            "gates only the Q&A correction branch, and regularizes that branch to "
            "stay incrementally informative beyond the strong prior/base shell."
        )
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/i_pog_qa_residual_real"),
    )
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="fail")
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=64)
    parser.add_argument("--learning-rates", default="0.01,0.03,0.05")
    parser.add_argument("--l2-values", default="0.0,1e-4,1e-3,1e-2")
    parser.add_argument("--incrementality-lambdas", default="0.0,0.01,0.03,0.1")
    parser.add_argument("--activation-lambdas", default="0.0,1e-4,3e-4,1e-3")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    return parser.parse_args()


def derived_targets_extended(row: dict[str, str], eps: float) -> dict[str, float | None]:
    target_map = derived_targets(row, eps)
    pre = safe_float(row.get("pre_60m_rv"))
    within = safe_float(row.get("within_call_rv"))
    post = safe_float(row.get("post_call_60m_rv"))
    target_map["within_minus_pre"] = None
    target_map["log_within_over_pre"] = None
    if within is not None and pre is not None:
        target_map["within_minus_pre"] = float(within - pre)
        target_map["log_within_over_pre"] = float(math.log(within + eps) - math.log(pre + eps))
    if post is not None and within is not None:
        target_map["post_minus_within"] = float(post - within)
    else:
        target_map["post_minus_within"] = None
    return target_map


def load_joined_rows(
    panel_csv: Path,
    features_csv: Path,
    qa_csv: Path,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    feature_lookup = {}
    for row in load_csv_rows(features_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            feature_lookup[event_key] = row

    qa_lookup = {}
    for row in load_csv_rows(qa_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            qa_lookup[event_key] = row

    coverage = {
        "panel_rows": 0,
        "with_features": 0,
        "with_qa": 0,
        "with_all_side_inputs": 0,
        "joined_rows": 0,
    }
    rows = []
    for row in load_csv_rows(panel_csv.resolve()):
        coverage["panel_rows"] += 1
        event_key = row.get("event_key", "")
        year_value = safe_float(row.get("year"))
        feature_row = feature_lookup.get(event_key)
        qa_row = qa_lookup.get(event_key)
        if feature_row is not None:
            coverage["with_features"] += 1
        if qa_row is not None:
            coverage["with_qa"] += 1
        if feature_row is not None and qa_row is not None:
            coverage["with_all_side_inputs"] += 1
        if not event_key or year_value is None or feature_row is None or qa_row is None:
            continue
        merged = dict(row)
        merged.update(feature_row)
        merged.update(qa_row)
        merged["_year"] = int(year_value)
        rows.append(merged)
    coverage["joined_rows"] = len(rows)
    return rows, coverage


def build_rows(args: argparse.Namespace) -> tuple[list[dict[str, str]], dict[str, object]]:
    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    base_rows, coverage = load_joined_rows(args.panel_csv, args.features_csv, args.qa_csv)
    rows = []
    filtered_counts = {
        "kept_rows": 0,
        "dropped_html_flag": 0,
        "dropped_regime": 0,
        "dropped_target": 0,
    }
    for row in base_rows:
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        if html_flag in exclude_html_flags:
            filtered_counts["dropped_html_flag"] += 1
            continue
        if include_regimes and regime_label(row) not in include_regimes:
            filtered_counts["dropped_regime"] += 1
            continue
        target_value = derived_targets_extended(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            filtered_counts["dropped_target"] += 1
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        rows.append(item)
    filtered_counts["kept_rows"] = len(rows)
    return attach_ticker_prior(rows, args.train_end_year), {
        "join_coverage": coverage,
        "filtering": filtered_counts,
    }


def top_coefficients(feature_names: list[str], coefficients: np.ndarray, limit: int = 8) -> list[dict[str, float | str]]:
    coeffs = np.asarray(coefficients, dtype=float)
    if coeffs.size == 0:
        return []
    order = np.argsort(np.abs(coeffs))[::-1][:limit]
    return [
        {
            "feature": feature_names[int(idx)],
            "coefficient": float(coeffs[int(idx)]),
            "abs_coefficient": float(abs(coeffs[int(idx)])),
        }
        for idx in order
    ]


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def trust_feature_direction(feature_name: str) -> float:
    return float(TRUST_GATE_DIRECTION_BY_FEATURE.get(feature_name, 1.0))


def aligned_trust_feature_label(feature_name: str) -> str:
    direction = trust_feature_direction(feature_name)
    prefix = "trust_up::" if direction >= 0.0 else "trust_down::"
    return f"{prefix}{feature_name}"


def build_monotone_trust_bundle(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
) -> dict[str, object]:
    bundle = build_dense_bundle(train_rows, val_rows, test_rows, feature_names)
    directions = np.asarray([trust_feature_direction(name) for name in bundle["feature_names"]], dtype=float)
    return {
        **bundle,
        "train": bundle["train"] * directions,
        "val": bundle["val"] * directions,
        "test": bundle["test"] * directions,
        "aligned_feature_names": [aligned_trust_feature_label(name) for name in bundle["feature_names"]],
        "direction_signs": [float(item) for item in directions],
    }


def init_gate_params(route_dim: int, trust_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray | float]:
    route_scale = 0.01
    return {
        "route_w": rng.normal(0.0, route_scale, size=route_dim),
        "route_b": 0.0,
        "trust_raw_w": rng.normal(-2.0, 0.1, size=trust_dim),
        "trust_b": -1.5,
    }


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def softplus(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return np.log1p(np.exp(-np.abs(values))) + np.maximum(values, 0.0)


def trust_weight_vector(params: dict[str, np.ndarray | float]) -> np.ndarray:
    return softplus(np.asarray(params["trust_raw_w"], dtype=float))


def normalized_centered(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    centered = arr - float(np.mean(arr))
    scale = float(np.std(centered))
    if scale == 0.0:
        return np.zeros_like(centered)
    return centered / scale


def covariance_against_reference(values: np.ndarray, reference: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    return float(np.mean(centered * reference))


def safe_correlation(values: np.ndarray, reference: np.ndarray) -> float:
    centered = np.asarray(values, dtype=float) - float(np.mean(values))
    denom = float(np.linalg.norm(centered) * np.linalg.norm(reference))
    if denom == 0.0:
        return 0.0
    return float(np.dot(centered, reference) / denom)


def incrementality_diagnostics(
    prior: np.ndarray,
    base_pred: np.ndarray,
    applied_dialog_effect: np.ndarray,
) -> dict[str, float]:
    prior_ref = normalized_centered(prior)
    base_residual_ref = normalized_centered(base_pred - prior)
    return {
        "applied_dialog_effect_mean": float(np.mean(applied_dialog_effect)),
        "applied_dialog_effect_abs_mean": float(np.mean(np.abs(applied_dialog_effect))),
        "cov_with_prior_ref": covariance_against_reference(applied_dialog_effect, prior_ref),
        "cov_with_base_residual_ref": covariance_against_reference(applied_dialog_effect, base_residual_ref),
        "corr_with_prior_ref": safe_correlation(applied_dialog_effect, prior_ref),
        "corr_with_base_residual_ref": safe_correlation(applied_dialog_effect, base_residual_ref),
    }


def predict_gate_model(
    route_x: np.ndarray,
    trust_x: np.ndarray,
    base_pred: np.ndarray,
    sem_corr: np.ndarray,
    qual_corr: np.ndarray,
    params: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    route_logits = route_x @ params["route_w"] + params["route_b"]
    trust_logits = trust_x @ trust_weight_vector(params) + params["trust_b"]
    route_gate = sigmoid(route_logits)
    trust_gate = sigmoid(trust_logits)
    mixed_dialog_corr = route_gate * sem_corr + (1.0 - route_gate) * qual_corr
    applied_dialog_effect = trust_gate * mixed_dialog_corr
    pred = base_pred + applied_dialog_effect
    return pred, route_gate, trust_gate, mixed_dialog_corr, applied_dialog_effect


def train_gate_model(
    train_route_x: np.ndarray,
    train_trust_x: np.ndarray,
    train_prior: np.ndarray,
    train_base_pred: np.ndarray,
    train_sem_corr: np.ndarray,
    train_qual_corr: np.ndarray,
    train_y: np.ndarray,
    val_route_x: np.ndarray,
    val_trust_x: np.ndarray,
    val_base_pred: np.ndarray,
    val_sem_corr: np.ndarray,
    val_qual_corr: np.ndarray,
    val_y: np.ndarray,
    learning_rates: list[float],
    l2_values: list[float],
    incrementality_lambdas: list[float],
    activation_lambdas: list[float],
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    rng_master = np.random.default_rng(seed)
    best = None
    prior_ref = normalized_centered(train_prior)
    base_residual_ref = normalized_centered(train_base_pred - train_prior)
    reference_vectors = [prior_ref, base_residual_ref]
    n = max(len(train_y), 1)

    for learning_rate in learning_rates:
        for l2_value in l2_values:
            for incrementality_lambda in incrementality_lambdas:
                for activation_lambda in activation_lambdas:
                    rng = np.random.default_rng(int(rng_master.integers(1_000_000_000)))
                    params = init_gate_params(train_route_x.shape[1], train_trust_x.shape[1], rng)
                    adam_state = {
                        "m_route_w": np.zeros_like(params["route_w"]),
                        "v_route_w": np.zeros_like(params["route_w"]),
                        "m_route_b": 0.0,
                        "v_route_b": 0.0,
                        "m_trust_raw_w": np.zeros_like(params["trust_raw_w"]),
                        "v_trust_raw_w": np.zeros_like(params["trust_raw_w"]),
                        "m_trust_b": 0.0,
                        "v_trust_b": 0.0,
                    }
                    beta1 = 0.9
                    beta2 = 0.999
                    eps_opt = 1e-8
                    best_local_val = None
                    best_local_params = None
                    stale = 0

                    for step in range(1, epochs + 1):
                        pred, route_gate, trust_gate, mixed_dialog_corr, applied_dialog_effect = predict_gate_model(
                            train_route_x,
                            train_trust_x,
                            train_base_pred,
                            train_sem_corr,
                            train_qual_corr,
                            params,
                        )
                        err = pred - train_y
                        dloss_deffect = 2.0 * err / n
                        if activation_lambda:
                            dloss_deffect = dloss_deffect + (2.0 * activation_lambda * applied_dialog_effect / n)
                        if incrementality_lambda:
                            for reference in reference_vectors:
                                covariance = covariance_against_reference(applied_dialog_effect, reference)
                                dloss_deffect = dloss_deffect + (
                                    2.0 * incrementality_lambda * covariance * reference / n
                                )

                        dloss_dmixed = dloss_deffect * trust_gate
                        dloss_dtrust = dloss_deffect * mixed_dialog_corr
                        dloss_droute = dloss_dmixed * (train_sem_corr - train_qual_corr)
                        dloss_dtrust_logit = dloss_dtrust * trust_gate * (1.0 - trust_gate)
                        dloss_droute_logit = dloss_droute * route_gate * (1.0 - route_gate)

                        trust_w = trust_weight_vector(params)
                        grads = {
                            "route_w": train_route_x.T @ dloss_droute_logit + 2.0 * l2_value * params["route_w"],
                            "route_b": float(np.sum(dloss_droute_logit)),
                            "trust_raw_w": (
                                (train_trust_x.T @ dloss_dtrust_logit + 2.0 * l2_value * trust_w)
                                * sigmoid(np.asarray(params["trust_raw_w"], dtype=float))
                            ),
                            "trust_b": float(np.sum(dloss_dtrust_logit)),
                        }

                        for key, grad in grads.items():
                            m_key = f"m_{key}"
                            v_key = f"v_{key}"
                            adam_state[m_key] = beta1 * adam_state[m_key] + (1.0 - beta1) * grad
                            adam_state[v_key] = beta2 * adam_state[v_key] + (1.0 - beta2) * (grad * grad)
                            m_hat = adam_state[m_key] / (1.0 - beta1 ** step)
                            v_hat = adam_state[v_key] / (1.0 - beta2 ** step)
                            params[key] = params[key] - learning_rate * m_hat / (np.sqrt(v_hat) + eps_opt)

                        val_pred, _, _, _, _ = predict_gate_model(
                            val_route_x,
                            val_trust_x,
                            val_base_pred,
                            val_sem_corr,
                            val_qual_corr,
                            params,
                        )
                        val_rmse = metrics(val_y, val_pred)["rmse"]
                        if best_local_val is None or val_rmse < best_local_val:
                            best_local_val = val_rmse
                            best_local_params = {
                                "route_w": params["route_w"].copy(),
                                "route_b": float(params["route_b"]),
                                "trust_raw_w": params["trust_raw_w"].copy(),
                                "trust_b": float(params["trust_b"]),
                            }
                            stale = 0
                        else:
                            stale += 1
                        if stale >= patience:
                            break

                    final_val_pred, final_route_gate, final_trust_gate, _, final_val_effect = predict_gate_model(
                        val_route_x,
                        val_trust_x,
                        val_base_pred,
                        val_sem_corr,
                        val_qual_corr,
                        best_local_params,
                    )
                    candidate = {
                        "learning_rate": learning_rate,
                        "l2": l2_value,
                        "incrementality_lambda": incrementality_lambda,
                        "activation_lambda": activation_lambda,
                        "params": best_local_params,
                        "val_pred": final_val_pred,
                        "val_rmse": metrics(val_y, final_val_pred)["rmse"],
                        "val_route_gate_mean": float(np.mean(final_route_gate)),
                        "val_trust_gate_mean": float(np.mean(final_trust_gate)),
                        "val_applied_dialog_effect_abs_mean": float(np.mean(np.abs(final_val_effect))),
                    }
                    if best is None or candidate["val_rmse"] < best["val_rmse"]:
                        best = candidate

    assert best is not None
    return best, np.asarray(best["val_pred"], dtype=float)


def extend_prediction_rows(
    prediction_rows: list[dict[str, object]],
    model_name: str,
    split_name: str,
    rows: list[dict[str, str]],
    predictions: np.ndarray,
) -> None:
    for row, pred in zip(rows, predictions):
        prediction_rows.append(
            {
                "model_name": model_name,
                "split": split_name,
                "event_key": row.get("event_key", ""),
                "ticker": row.get("ticker", ""),
                "year": row.get("year", ""),
                "quarter": row.get("quarter", ""),
                "target": row["_target"],
                "prediction": float(pred),
            }
        )


def extend_gate_rows(
    gate_rows: list[dict[str, object]],
    split_name: str,
    rows: list[dict[str, str]],
    prior_pred: np.ndarray,
    base_pred: np.ndarray,
    sem_corr: np.ndarray,
    qual_corr: np.ndarray,
    route_gate: np.ndarray,
    trust_gate: np.ndarray,
    mixed_dialog_corr: np.ndarray,
    applied_dialog_effect: np.ndarray,
    predictions: np.ndarray,
) -> None:
    for idx, row in enumerate(rows):
        gate_rows.append(
            {
                "split": split_name,
                "event_key": row.get("event_key", ""),
                "ticker": row.get("ticker", ""),
                "year": row.get("year", ""),
                "quarter": row.get("quarter", ""),
                "target": row["_target"],
                "prior_prediction": float(prior_pred[idx]),
                "base_residual_component": float(base_pred[idx] - prior_pred[idx]),
                "base_prediction": float(base_pred[idx]),
                "semantic_correction": float(sem_corr[idx]),
                "quality_correction": float(qual_corr[idx]),
                "route_gate": float(route_gate[idx]),
                "trust_gate": float(trust_gate[idx]),
                "mixed_dialog_correction": float(mixed_dialog_corr[idx]),
                "applied_dialog_effect": float(applied_dialog_effect[idx]),
                "increment_over_base": float(predictions[idx] - base_pred[idx]),
                "prediction": float(predictions[idx]),
            }
        )


def summarize_significance(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    bootstrap_iters: int,
    perm_iters: int,
    seed: int,
) -> dict[str, float]:
    return {
        **paired_bootstrap_deltas(y_true, pred_a, pred_b, bootstrap_iters, seed),
        **paired_sign_permutation_pvalue(y_true, pred_a, pred_b, perm_iters, seed),
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, coverage_payload = build_rows(args)
    train_rows = [row for row in rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in rows if row["_year"] == args.val_year]
    test_rows = [row for row in rows if row["_year"] > args.val_year]

    if not train_rows or not val_rows or not test_rows:
        raise SystemExit(
            f"insufficient rows after filtering: train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
        )

    alphas = parse_float_list(args.alphas)
    learning_rates = parse_float_list(args.learning_rates)
    l2_values = parse_float_list(args.l2_values)
    incrementality_lambdas = parse_float_list(args.incrementality_lambdas)
    activation_lambdas = parse_float_list(args.activation_lambdas)

    base_feature_names = PRE_CALL_MARKET_FEATURES + CONTROL_FEATURES + A4_STRUCTURED_FEATURES
    bundles = {
        "base": build_dense_bundle(train_rows, val_rows, test_rows, base_feature_names),
        "quality": build_dense_bundle(train_rows, val_rows, test_rows, QA_CORE_FEATURES),
        "trust": build_monotone_trust_bundle(train_rows, val_rows, test_rows, TRUST_GATE_FEATURES),
        "route": build_dense_bundle(train_rows, val_rows, test_rows, ROUTE_GATE_FEATURES),
        "semantic": build_text_lsa_bundle(
            train_rows,
            val_rows,
            test_rows,
            text_col="qna_text",
            max_features=args.max_features,
            min_df=args.min_df,
            lsa_components=args.lsa_components,
        ),
    }
    bundles["base_plus_semantic"] = {
        "train": np.hstack([bundles["base"]["train"], bundles["semantic"]["train"]]),
        "val": np.hstack([bundles["base"]["val"], bundles["semantic"]["val"]]),
        "test": np.hstack([bundles["base"]["test"], bundles["semantic"]["test"]]),
        "feature_names": bundles["base"]["feature_names"] + bundles["semantic"]["feature_names"],
    }
    bundles["base_plus_quality"] = {
        "train": np.hstack([bundles["base"]["train"], bundles["quality"]["train"]]),
        "val": np.hstack([bundles["base"]["val"], bundles["quality"]["val"]]),
        "test": np.hstack([bundles["base"]["test"], bundles["quality"]["test"]]),
        "feature_names": bundles["base"]["feature_names"] + bundles["quality"]["feature_names"],
    }
    bundles["base_plus_semantic_plus_quality"] = {
        "train": np.hstack([bundles["base"]["train"], bundles["semantic"]["train"], bundles["quality"]["train"]]),
        "val": np.hstack([bundles["base"]["val"], bundles["semantic"]["val"], bundles["quality"]["val"]]),
        "test": np.hstack([bundles["base"]["test"], bundles["semantic"]["test"], bundles["quality"]["test"]]),
        "feature_names": (
            bundles["base"]["feature_names"]
            + bundles["semantic"]["feature_names"]
            + bundles["quality"]["feature_names"]
        ),
    }

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
    val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
    test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)

    summary = {
        "method_name": "I-POG-QA",
        "method_aliases": ["POG-QA"],
        "method_long_name": "Incremental Prior-Aware Observability-Gated Q&A Residual",
        "formulation": {
            "prediction": "prior + base_residual + trust_gate * (route_gate * semantic_correction + (1-route_gate) * quality_correction)",
            "base_branch": "prior-aware residual ridge on pre-call market + controls + A4 observability features",
            "semantic_branch": "Q&A semantic residual expert trained on top of the structured base prediction",
            "quality_branch": "Q&A answer-quality/accountability residual expert trained on top of the structured base prediction",
            "gate_design": "trust gate decides when dialog corrections are reliable; route gate decides whether semantic or answer-quality correction should dominate",
            "incrementality_constraint": "penalize covariance between the applied dialog effect and standardized prior/base-residual references",
            "monotone_trust_gate": "trust weights are constrained to be nonnegative after direction-aligning reliability features",
            "activation_regularizer": "penalize unnecessary dialog effect magnitude so the model defaults back to prior + structured base when evidence is weak",
        },
        "innovation_claims": [
            "Keeps the same strong same-ticker prior and structured residual core instead of replacing them.",
            "Gates only the high-variance Q&A correction branch, which directly addresses the failure mode of the earlier global residual gate.",
            "Separates Q&A information into semantic and answer-quality experts, then learns a reliability-aware mixture instead of collapsing them into one wider stack.",
            "Constrains the trust gate to be direction-consistent with observability and answerability signals instead of letting reliability weights flip arbitrarily.",
            "Regularizes the applied dialog effect to stay incrementally informative beyond the strong prior and structured base shell.",
            "Supports both the current off-hours shock target and the professor-motivated during-call target family via extended target variants.",
        ],
        "target_variant": args.target_variant,
        "supported_target_variants": [
            "raw_post_call_60m_rv",
            "log_post_over_pre",
            "log_post_over_within",
            "shock_minus_pre",
            "within_minus_pre",
            "log_within_over_pre",
            "post_minus_within",
        ],
        "include_regimes": sorted({item.strip() for item in args.include_regimes.split(",") if item.strip()}),
        "exclude_html_flags": sorted(
            {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
        ),
        "coverage": coverage_payload,
        "split_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "training_grid": {
            "alphas": alphas,
            "learning_rates": learning_rates,
            "l2_values": l2_values,
            "incrementality_lambdas": incrementality_lambdas,
            "activation_lambdas": activation_lambdas,
            "epochs": args.epochs,
            "patience": args.patience,
        },
        "feature_groups": {
            "base": base_feature_names,
            "semantic_text": bundles["semantic"]["feature_names"],
            "quality_core": QA_CORE_FEATURES,
            "trust_gate_raw": bundles["trust"]["feature_names"],
            "trust_gate_aligned": bundles["trust"]["aligned_feature_names"],
            "route_gate": bundles["route"]["feature_names"],
        },
        "trust_gate_feature_directions": [
            {
                "feature": feature_name,
                "direction": "up" if direction >= 0.0 else "down",
                "aligned_feature": aligned_name,
            }
            for feature_name, direction, aligned_name in zip(
                bundles["trust"]["feature_names"],
                bundles["trust"]["direction_signs"],
                bundles["trust"]["aligned_feature_names"],
            )
        ],
        "models": {},
        "significance": {},
    }
    prediction_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    test_predictions: dict[str, np.ndarray] = {}

    # Prior-only baseline.
    summary["models"]["prior_only"] = {
        "family": "prior_passthrough",
        "val": metrics(val_y, val_prior),
        "test": metrics(test_y, test_prior),
    }
    test_predictions["prior_only"] = np.asarray(test_prior, dtype=float)
    extend_prediction_rows(prediction_rows, "prior_only", "val", val_rows, val_prior)
    extend_prediction_rows(prediction_rows, "prior_only", "test", test_rows, test_prior)

    # Ridge baselines on top of the same prior.
    baseline_specs = {
        "residual_base_structured": "base",
        "residual_base_plus_semantic": "base_plus_semantic",
        "residual_base_plus_quality": "base_plus_quality",
        "residual_base_plus_semantic_plus_quality": "base_plus_semantic_plus_quality",
    }
    for model_name, bundle_name in baseline_specs.items():
        bundle = bundles[bundle_name]
        best_alpha, best_model, pred_val = fit_residual_ridge(
            bundle["train"],
            train_prior,
            train_y,
            bundle["val"],
            val_prior,
            val_y,
            alphas,
        )
        pred_test = test_prior + best_model.predict(bundle["test"])
        summary["models"][model_name] = {
            "family": "residual_ridge",
            "feature_group": bundle_name,
            "feature_count": int(bundle["train"].shape[1]),
            "best_alpha": best_alpha,
            "val": metrics(val_y, pred_val),
            "test": metrics(test_y, pred_test),
        }
        test_predictions[model_name] = np.asarray(pred_test, dtype=float)
        extend_prediction_rows(prediction_rows, model_name, "val", val_rows, pred_val)
        extend_prediction_rows(prediction_rows, model_name, "test", test_rows, pred_test)

    # Stage 1: stable structured base.
    base_bundle = bundles["base"]
    base_alpha, base_model, base_val_pred = fit_residual_ridge(
        base_bundle["train"],
        train_prior,
        train_y,
        base_bundle["val"],
        val_prior,
        val_y,
        alphas,
    )
    base_train_pred = train_prior + base_model.predict(base_bundle["train"])
    base_test_pred = test_prior + base_model.predict(base_bundle["test"])

    # Stage 2: semantic and quality corrections on top of the structured base.
    semantic_bundle = bundles["semantic"]
    semantic_alpha, semantic_model, semantic_val_pred = fit_residual_ridge(
        semantic_bundle["train"],
        base_train_pred,
        train_y,
        semantic_bundle["val"],
        base_val_pred,
        val_y,
        alphas,
    )
    semantic_train_corr = semantic_model.predict(semantic_bundle["train"])
    semantic_val_corr = semantic_model.predict(semantic_bundle["val"])
    semantic_test_corr = semantic_model.predict(semantic_bundle["test"])

    quality_bundle = bundles["quality"]
    quality_alpha, quality_model, quality_val_pred = fit_residual_ridge(
        quality_bundle["train"],
        base_train_pred,
        train_y,
        quality_bundle["val"],
        base_val_pred,
        val_y,
        alphas,
    )
    quality_train_corr = quality_model.predict(quality_bundle["train"])
    quality_val_corr = quality_model.predict(quality_bundle["val"])
    quality_test_corr = quality_model.predict(quality_bundle["test"])

    # Stage 3: gate only the higher-variance Q&A correction branch.
    gate_best, gate_val_pred = train_gate_model(
        bundles["route"]["train"],
        bundles["trust"]["train"],
        train_prior,
        base_train_pred,
        semantic_train_corr,
        quality_train_corr,
        train_y,
        bundles["route"]["val"],
        bundles["trust"]["val"],
        base_val_pred,
        semantic_val_corr,
        quality_val_corr,
        val_y,
        learning_rates,
        l2_values,
        incrementality_lambdas,
        activation_lambdas,
        args.epochs,
        args.patience,
        args.seed,
    )
    gate_test_pred, route_gate_test, trust_gate_test, mixed_dialog_corr_test, applied_dialog_effect_test = predict_gate_model(
        bundles["route"]["test"],
        bundles["trust"]["test"],
        base_test_pred,
        semantic_test_corr,
        quality_test_corr,
        gate_best["params"],
    )
    _, route_gate_val, trust_gate_val, mixed_dialog_corr_val, applied_dialog_effect_val = predict_gate_model(
        bundles["route"]["val"],
        bundles["trust"]["val"],
        base_val_pred,
        semantic_val_corr,
        quality_val_corr,
        gate_best["params"],
    )
    gate_model_key = "i_pog_qa_residual"

    summary["models"][gate_model_key] = {
        "family": "hierarchical_residual_mixture",
        "structured_base": {
            "best_alpha": base_alpha,
            "feature_count": int(base_bundle["train"].shape[1]),
            "top_coefficients": top_coefficients(base_bundle["feature_names"], np.asarray(base_model.coef_, dtype=float)),
        },
        "semantic_expert": {
            "best_alpha": semantic_alpha,
            "feature_count": int(semantic_bundle["train"].shape[1]),
            "top_coefficients": top_coefficients(
                semantic_bundle["feature_names"],
                np.asarray(semantic_model.coef_, dtype=float),
            ),
        },
        "quality_expert": {
            "best_alpha": quality_alpha,
            "feature_count": int(quality_bundle["train"].shape[1]),
            "top_coefficients": top_coefficients(
                quality_bundle["feature_names"],
                np.asarray(quality_model.coef_, dtype=float),
            ),
        },
        "gate": {
            "learning_rate": gate_best["learning_rate"],
            "l2": gate_best["l2"],
            "incrementality_lambda": gate_best["incrementality_lambda"],
            "activation_lambda": gate_best["activation_lambda"],
            "trust_constraint": "direction-aligned monotone gate via nonnegative trust weights",
            "val_route_gate_mean": gate_best["val_route_gate_mean"],
            "val_trust_gate_mean": gate_best["val_trust_gate_mean"],
            "val_applied_dialog_effect_abs_mean": gate_best["val_applied_dialog_effect_abs_mean"],
            "test_route_gate_mean": float(np.mean(route_gate_test)),
            "test_route_gate_std": float(np.std(route_gate_test)),
            "test_trust_gate_mean": float(np.mean(trust_gate_test)),
            "test_trust_gate_std": float(np.std(trust_gate_test)),
            "test_mixed_dialog_correction_abs_mean": float(np.mean(np.abs(mixed_dialog_corr_test))),
            "test_applied_dialog_effect_abs_mean": float(np.mean(np.abs(applied_dialog_effect_test))),
            "top_route_features": top_coefficients(
                bundles["route"]["feature_names"],
                np.asarray(gate_best["params"]["route_w"], dtype=float),
            ),
            "top_trust_features": top_coefficients(
                bundles["trust"]["aligned_feature_names"],
                trust_weight_vector(gate_best["params"]),
            ),
            "val_incrementality": incrementality_diagnostics(val_prior, base_val_pred, applied_dialog_effect_val),
            "test_incrementality": incrementality_diagnostics(test_prior, base_test_pred, applied_dialog_effect_test),
        },
        "val": metrics(val_y, gate_val_pred),
        "test": metrics(test_y, gate_test_pred),
    }
    test_predictions[gate_model_key] = np.asarray(gate_test_pred, dtype=float)
    extend_prediction_rows(prediction_rows, gate_model_key, "val", val_rows, gate_val_pred)
    extend_prediction_rows(prediction_rows, gate_model_key, "test", test_rows, gate_test_pred)
    extend_gate_rows(
        gate_rows,
        "val",
        val_rows,
        val_prior,
        base_val_pred,
        semantic_val_corr,
        quality_val_corr,
        route_gate_val,
        trust_gate_val,
        mixed_dialog_corr_val,
        applied_dialog_effect_val,
        gate_val_pred,
    )
    extend_gate_rows(
        gate_rows,
        "test",
        test_rows,
        test_prior,
        base_test_pred,
        semantic_test_corr,
        quality_test_corr,
        route_gate_test,
        trust_gate_test,
        mixed_dialog_corr_test,
        applied_dialog_effect_test,
        gate_test_pred,
    )

    comparisons = [
        ("residual_base_structured", "residual_base_plus_semantic"),
        ("residual_base_plus_semantic", "residual_base_plus_semantic_plus_quality"),
        ("residual_base_plus_semantic_plus_quality", gate_model_key),
        ("residual_base_plus_quality", gate_model_key),
    ]
    for model_a, model_b in comparisons:
        summary["significance"][f"{model_a}__vs__{model_b}"] = summarize_significance(
            test_y,
            test_predictions[model_a],
            test_predictions[model_b],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        )

    write_json(output_dir / "i_pog_qa_residual_summary.json", summary)
    write_csv(output_dir / "i_pog_qa_residual_predictions.csv", prediction_rows)
    write_csv(output_dir / "i_pog_qa_residual_gate_diagnostics.csv", gate_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
