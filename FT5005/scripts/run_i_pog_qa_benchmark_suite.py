#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import site
import sys
from pathlib import Path

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np

from dj30_qc_utils import write_csv, write_json
from run_afterhours_precall_semantic_ladder import A4_STRUCTURED_FEATURES, PRE_CALL_MARKET_FEATURES
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle
from run_pog_qa_residual import (
    QA_CORE_FEATURES,
    ROUTE_GATE_FEATURES,
    TRUST_GATE_FEATURES,
    build_monotone_trust_bundle,
    build_rows,
    incrementality_diagnostics,
    parse_float_list,
    safe_correlation,
    sigmoid,
    softplus,
    summarize_significance,
    top_coefficients,
)
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_signal_decomposition_benchmarks import CONTROL_FEATURES
from run_structured_baselines import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the I-POG-QA benchmark suite with key ablations. "
            "This compares the full model against simplified trust, route, "
            "and incrementality variants under the same temporal split."
        )
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/i_pog_qa_benchmark_suite_real"),
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


def init_variant_params(
    route_dim: int,
    trust_dim: int,
    route_mode: str,
    trust_mode: str,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    params: dict[str, np.ndarray | float] = {}
    if route_mode == "learned":
        params["route_w"] = rng.normal(0.0, 0.01, size=route_dim)
        params["route_b"] = 0.0
    if trust_mode == "monotone":
        params["trust_raw_w"] = rng.normal(-2.0, 0.1, size=trust_dim)
        params["trust_b"] = -1.5
    elif trust_mode == "free":
        params["trust_w"] = rng.normal(0.0, 0.01, size=trust_dim)
        params["trust_b"] = -1.0
    return params


def clone_params(params: dict[str, np.ndarray | float]) -> dict[str, np.ndarray | float]:
    copied: dict[str, np.ndarray | float] = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = float(value)
    return copied


def normalize_vector(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    centered = arr - float(np.mean(arr))
    scale = float(np.std(centered))
    if scale == 0.0:
        return np.zeros_like(centered)
    return centered / scale


def variant_trust_weights(params: dict[str, np.ndarray | float], trust_mode: str) -> np.ndarray:
    if trust_mode == "monotone":
        return softplus(np.asarray(params["trust_raw_w"], dtype=float))
    if trust_mode == "free":
        return np.asarray(params["trust_w"], dtype=float)
    return np.asarray([], dtype=float)


def fixed_route_gate(route_mode: str, n: int) -> np.ndarray:
    if route_mode == "fixed_semantic":
        return np.ones(n, dtype=float)
    if route_mode == "fixed_quality":
        return np.zeros(n, dtype=float)
    if route_mode == "uniform":
        return np.full(n, 0.5, dtype=float)
    raise ValueError(f"unsupported fixed route mode: {route_mode}")


def fixed_trust_gate(trust_mode: str, n: int) -> np.ndarray:
    if trust_mode == "always_on":
        return np.ones(n, dtype=float)
    raise ValueError(f"unsupported fixed trust mode: {trust_mode}")


def predict_variant_gate_model(
    route_x: np.ndarray,
    trust_x: np.ndarray,
    base_pred: np.ndarray,
    sem_corr: np.ndarray,
    qual_corr: np.ndarray,
    params: dict[str, np.ndarray | float],
    route_mode: str,
    trust_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if route_mode == "learned":
        route_logits = route_x @ np.asarray(params["route_w"], dtype=float) + float(params["route_b"])
        route_gate = sigmoid(route_logits)
    else:
        route_gate = fixed_route_gate(route_mode, len(base_pred))

    if trust_mode == "monotone":
        trust_logits = trust_x @ variant_trust_weights(params, trust_mode) + float(params["trust_b"])
        trust_gate = sigmoid(trust_logits)
    elif trust_mode == "free":
        trust_logits = trust_x @ variant_trust_weights(params, trust_mode) + float(params["trust_b"])
        trust_gate = sigmoid(trust_logits)
    else:
        trust_gate = fixed_trust_gate(trust_mode, len(base_pred))

    mixed_dialog_corr = route_gate * sem_corr + (1.0 - route_gate) * qual_corr
    applied_dialog_effect = trust_gate * mixed_dialog_corr
    pred = base_pred + applied_dialog_effect
    return pred, route_gate, trust_gate, mixed_dialog_corr, applied_dialog_effect


def train_variant_gate_model(
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
    route_mode: str,
    trust_mode: str,
) -> tuple[dict[str, object], np.ndarray]:
    rng_master = np.random.default_rng(seed)
    best = None
    prior_ref = normalize_vector(train_prior)
    base_residual_ref = normalize_vector(train_base_pred - train_prior)
    reference_vectors = [prior_ref, base_residual_ref]
    n = max(len(train_y), 1)
    learn_route = route_mode == "learned"
    learn_trust = trust_mode in {"monotone", "free"}

    candidate_grid = [
        {
            "learning_rate": learning_rate,
            "l2": l2_value,
            "incrementality_lambda": incrementality_lambda,
            "activation_lambda": activation_lambda,
        }
        for learning_rate in learning_rates
        for l2_value in l2_values
        for incrementality_lambda in incrementality_lambdas
        for activation_lambda in activation_lambdas
    ]
    if not learn_route and not learn_trust:
        candidate_grid = [
            {
                "learning_rate": 0.0,
                "l2": 0.0,
                "incrementality_lambda": 0.0,
                "activation_lambda": 0.0,
            }
        ]

    for candidate_spec in candidate_grid:
        learning_rate = float(candidate_spec["learning_rate"])
        l2_value = float(candidate_spec["l2"])
        incrementality_lambda = float(candidate_spec["incrementality_lambda"])
        activation_lambda = float(candidate_spec["activation_lambda"])
        rng = np.random.default_rng(int(rng_master.integers(1_000_000_000)))
        params = init_variant_params(train_route_x.shape[1], train_trust_x.shape[1], route_mode, trust_mode, rng)

        if learn_route or learn_trust:
            adam_state: dict[str, np.ndarray | float] = {}
            if learn_route:
                adam_state["m_route_w"] = np.zeros_like(params["route_w"])
                adam_state["v_route_w"] = np.zeros_like(params["route_w"])
                adam_state["m_route_b"] = 0.0
                adam_state["v_route_b"] = 0.0
            if learn_trust:
                trust_key = "trust_raw_w" if trust_mode == "monotone" else "trust_w"
                adam_state[f"m_{trust_key}"] = np.zeros_like(params[trust_key])
                adam_state[f"v_{trust_key}"] = np.zeros_like(params[trust_key])
                adam_state["m_trust_b"] = 0.0
                adam_state["v_trust_b"] = 0.0

            beta1 = 0.9
            beta2 = 0.999
            eps_opt = 1e-8
            best_local_val = None
            best_local_params = None
            stale = 0

            for step in range(1, epochs + 1):
                pred, route_gate, trust_gate, mixed_dialog_corr, applied_dialog_effect = predict_variant_gate_model(
                    train_route_x,
                    train_trust_x,
                    train_base_pred,
                    train_sem_corr,
                    train_qual_corr,
                    params,
                    route_mode,
                    trust_mode,
                )
                err = pred - train_y
                dloss_deffect = 2.0 * err / n
                if activation_lambda:
                    dloss_deffect = dloss_deffect + (2.0 * activation_lambda * applied_dialog_effect / n)
                if incrementality_lambda:
                    centered_effect = applied_dialog_effect - float(np.mean(applied_dialog_effect))
                    for reference in reference_vectors:
                        covariance = float(np.mean(centered_effect * reference))
                        dloss_deffect = dloss_deffect + (2.0 * incrementality_lambda * covariance * reference / n)

                grads: dict[str, np.ndarray | float] = {}
                if learn_route:
                    dloss_dmixed = dloss_deffect * trust_gate
                    dloss_droute = dloss_dmixed * (train_sem_corr - train_qual_corr)
                    dloss_droute_logit = dloss_droute * route_gate * (1.0 - route_gate)
                    grads["route_w"] = train_route_x.T @ dloss_droute_logit + 2.0 * l2_value * np.asarray(
                        params["route_w"], dtype=float
                    )
                    grads["route_b"] = float(np.sum(dloss_droute_logit))

                if learn_trust:
                    dloss_dtrust = dloss_deffect * mixed_dialog_corr
                    dloss_dtrust_logit = dloss_dtrust * trust_gate * (1.0 - trust_gate)
                    if trust_mode == "monotone":
                        trust_w = variant_trust_weights(params, trust_mode)
                        grads["trust_raw_w"] = (
                            (train_trust_x.T @ dloss_dtrust_logit + 2.0 * l2_value * trust_w)
                            * sigmoid(np.asarray(params["trust_raw_w"], dtype=float))
                        )
                    else:
                        grads["trust_w"] = train_trust_x.T @ dloss_dtrust_logit + 2.0 * l2_value * np.asarray(
                            params["trust_w"], dtype=float
                        )
                    grads["trust_b"] = float(np.sum(dloss_dtrust_logit))

                for key, grad in grads.items():
                    m_key = f"m_{key}"
                    v_key = f"v_{key}"
                    adam_state[m_key] = 0.9 * adam_state[m_key] + 0.1 * grad
                    adam_state[v_key] = 0.999 * adam_state[v_key] + 0.001 * (grad * grad)
                    m_hat = adam_state[m_key] / (1.0 - beta1 ** step)
                    v_hat = adam_state[v_key] / (1.0 - beta2 ** step)
                    params[key] = params[key] - learning_rate * m_hat / (np.sqrt(v_hat) + eps_opt)

                val_pred, _, _, _, _ = predict_variant_gate_model(
                    val_route_x,
                    val_trust_x,
                    val_base_pred,
                    val_sem_corr,
                    val_qual_corr,
                    params,
                    route_mode,
                    trust_mode,
                )
                val_rmse = metrics(val_y, val_pred)["rmse"]
                if best_local_val is None or val_rmse < best_local_val:
                    best_local_val = val_rmse
                    best_local_params = clone_params(params)
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    break

            assert best_local_params is not None
            final_params = best_local_params
        else:
            final_params = params

        final_val_pred, final_route_gate, final_trust_gate, _, final_val_effect = predict_variant_gate_model(
            val_route_x,
            val_trust_x,
            val_base_pred,
            val_sem_corr,
            val_qual_corr,
            final_params,
            route_mode,
            trust_mode,
        )
        candidate = {
            "learning_rate": learning_rate,
            "l2": l2_value,
            "incrementality_lambda": incrementality_lambda,
            "activation_lambda": activation_lambda,
            "route_mode": route_mode,
            "trust_mode": trust_mode,
            "params": clone_params(final_params),
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
    rows_out: list[dict[str, object]],
    model_name: str,
    split_name: str,
    rows: list[dict[str, str]],
    predictions: np.ndarray,
) -> None:
    for row, pred in zip(rows, predictions):
        rows_out.append(
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
    rows_out: list[dict[str, object]],
    model_name: str,
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
        rows_out.append(
            {
                "model_name": model_name,
                "split": split_name,
                "event_key": row.get("event_key", ""),
                "ticker": row.get("ticker", ""),
                "year": row.get("year", ""),
                "quarter": row.get("quarter", ""),
                "target": row["_target"],
                "prior_prediction": float(prior_pred[idx]),
                "base_prediction": float(base_pred[idx]),
                "semantic_correction": float(sem_corr[idx]),
                "quality_correction": float(qual_corr[idx]),
                "route_gate": float(route_gate[idx]),
                "trust_gate": float(trust_gate[idx]),
                "mixed_dialog_correction": float(mixed_dialog_corr[idx]),
                "applied_dialog_effect": float(applied_dialog_effect[idx]),
                "prediction": float(predictions[idx]),
            }
        )


def gate_feature_names(trust_bundle: dict[str, object], trust_mode: str) -> list[str]:
    if trust_mode == "monotone":
        return list(trust_bundle["aligned_feature_names"])
    return list(trust_bundle["feature_names"])


def gate_constraint_label(trust_mode: str) -> str:
    if trust_mode == "monotone":
        return "direction-aligned monotone trust gate"
    if trust_mode == "free":
        return "free-sign learned trust gate"
    return "always-on trust passthrough"


def build_bundle_stack(base_bundle: dict[str, object], *extra_bundles: dict[str, object]) -> dict[str, object]:
    train_parts = [base_bundle["train"]]
    val_parts = [base_bundle["val"]]
    test_parts = [base_bundle["test"]]
    feature_names = list(base_bundle["feature_names"])
    for bundle in extra_bundles:
        train_parts.append(bundle["train"])
        val_parts.append(bundle["val"])
        test_parts.append(bundle["test"])
        feature_names.extend(bundle["feature_names"])
    return {
        "train": np.hstack(train_parts),
        "val": np.hstack(val_parts),
        "test": np.hstack(test_parts),
        "feature_names": feature_names,
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
    monotone_trust_bundle = build_monotone_trust_bundle(train_rows, val_rows, test_rows, TRUST_GATE_FEATURES)
    raw_trust_bundle = build_dense_bundle(train_rows, val_rows, test_rows, TRUST_GATE_FEATURES)
    route_bundle = build_dense_bundle(train_rows, val_rows, test_rows, ROUTE_GATE_FEATURES)
    base_bundle = build_dense_bundle(train_rows, val_rows, test_rows, base_feature_names)
    quality_bundle = build_dense_bundle(train_rows, val_rows, test_rows, QA_CORE_FEATURES)
    semantic_bundle = build_text_lsa_bundle(
        train_rows,
        val_rows,
        test_rows,
        text_col="qna_text",
        max_features=args.max_features,
        min_df=args.min_df,
        lsa_components=args.lsa_components,
    )
    base_plus_semantic_bundle = build_bundle_stack(base_bundle, semantic_bundle)
    base_plus_quality_bundle = build_bundle_stack(base_bundle, quality_bundle)
    base_plus_all_bundle = build_bundle_stack(base_bundle, semantic_bundle, quality_bundle)

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
    val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
    test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)

    prediction_rows: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    test_predictions: dict[str, np.ndarray] = {}

    summary = {
        "suite_name": "I-POG-QA benchmark suite",
        "target_variant": args.target_variant,
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
        "main_variant": "i_pog_qa_full",
        "baselines": {},
        "variants": {},
        "significance": {},
    }

    summary["baselines"]["prior_only"] = {
        "family": "prior_passthrough",
        "val": metrics(val_y, val_prior),
        "test": metrics(test_y, test_prior),
    }
    test_predictions["prior_only"] = np.asarray(test_prior, dtype=float)
    extend_prediction_rows(prediction_rows, "prior_only", "val", val_rows, val_prior)
    extend_prediction_rows(prediction_rows, "prior_only", "test", test_rows, test_prior)

    baseline_specs = {
        "residual_base_structured": base_bundle,
        "residual_base_plus_semantic": base_plus_semantic_bundle,
        "residual_base_plus_quality": base_plus_quality_bundle,
        "residual_base_plus_semantic_plus_quality": base_plus_all_bundle,
    }
    for model_name, bundle in baseline_specs.items():
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
        summary["baselines"][model_name] = {
            "family": "residual_ridge",
            "feature_count": int(bundle["train"].shape[1]),
            "best_alpha": best_alpha,
            "val": metrics(val_y, pred_val),
            "test": metrics(test_y, pred_test),
        }
        test_predictions[model_name] = np.asarray(pred_test, dtype=float)
        extend_prediction_rows(prediction_rows, model_name, "val", val_rows, pred_val)
        extend_prediction_rows(prediction_rows, model_name, "test", test_rows, pred_test)

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

    semantic_alpha, semantic_model, _ = fit_residual_ridge(
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

    quality_alpha, quality_model, _ = fit_residual_ridge(
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

    variant_specs = [
        {
            "name": "i_pog_qa_full",
            "route_mode": "learned",
            "trust_mode": "monotone",
            "incrementality_lambdas": incrementality_lambdas,
            "activation_lambdas": activation_lambdas,
            "description": "full I-POG-QA with monotone trust gate, learned route gate, incrementality regularization, and activation regularization",
        },
        {
            "name": "i_pog_qa_no_incrementality",
            "route_mode": "learned",
            "trust_mode": "monotone",
            "incrementality_lambdas": [0.0],
            "activation_lambdas": activation_lambdas,
            "description": "remove incrementality regularization while keeping the monotone trust gate",
        },
        {
            "name": "i_pog_qa_no_activation_reg",
            "route_mode": "learned",
            "trust_mode": "monotone",
            "incrementality_lambdas": incrementality_lambdas,
            "activation_lambdas": [0.0],
            "description": "remove the default-to-base activation regularizer",
        },
        {
            "name": "i_pog_qa_free_trust",
            "route_mode": "learned",
            "trust_mode": "free",
            "incrementality_lambdas": incrementality_lambdas,
            "activation_lambdas": activation_lambdas,
            "description": "replace the monotone trust gate with a free-sign learned trust gate",
        },
        {
            "name": "i_pog_qa_no_trust_gate",
            "route_mode": "learned",
            "trust_mode": "always_on",
            "incrementality_lambdas": [0.0],
            "activation_lambdas": [0.0],
            "description": "always-on trust branch; the dialog effect is never reliability-gated",
        },
        {
            "name": "i_pog_qa_semantic_only",
            "route_mode": "fixed_semantic",
            "trust_mode": "monotone",
            "incrementality_lambdas": incrementality_lambdas,
            "activation_lambdas": activation_lambdas,
            "description": "force the dialog branch to rely entirely on semantic correction",
        },
        {
            "name": "i_pog_qa_quality_only",
            "route_mode": "fixed_quality",
            "trust_mode": "monotone",
            "incrementality_lambdas": incrementality_lambdas,
            "activation_lambdas": activation_lambdas,
            "description": "force the dialog branch to rely entirely on quality correction",
        },
    ]

    for offset, variant_spec in enumerate(variant_specs):
        trust_bundle = monotone_trust_bundle if variant_spec["trust_mode"] == "monotone" else raw_trust_bundle
        best_variant, variant_val_pred = train_variant_gate_model(
            route_bundle["train"],
            np.asarray(trust_bundle["train"], dtype=float),
            train_prior,
            base_train_pred,
            semantic_train_corr,
            quality_train_corr,
            train_y,
            route_bundle["val"],
            np.asarray(trust_bundle["val"], dtype=float),
            base_val_pred,
            semantic_val_corr,
            quality_val_corr,
            val_y,
            learning_rates,
            l2_values,
            variant_spec["incrementality_lambdas"],
            variant_spec["activation_lambdas"],
            args.epochs,
            args.patience,
            args.seed + offset,
            variant_spec["route_mode"],
            variant_spec["trust_mode"],
        )
        variant_test_pred, route_gate_test, trust_gate_test, mixed_test, effect_test = predict_variant_gate_model(
            route_bundle["test"],
            np.asarray(trust_bundle["test"], dtype=float),
            base_test_pred,
            semantic_test_corr,
            quality_test_corr,
            best_variant["params"],
            variant_spec["route_mode"],
            variant_spec["trust_mode"],
        )
        _, route_gate_val, trust_gate_val, mixed_val, effect_val = predict_variant_gate_model(
            route_bundle["val"],
            np.asarray(trust_bundle["val"], dtype=float),
            base_val_pred,
            semantic_val_corr,
            quality_val_corr,
            best_variant["params"],
            variant_spec["route_mode"],
            variant_spec["trust_mode"],
        )

        trust_weights = variant_trust_weights(best_variant["params"], variant_spec["trust_mode"])
        trust_features = gate_feature_names(trust_bundle, variant_spec["trust_mode"])
        route_features = list(route_bundle["feature_names"]) if variant_spec["route_mode"] == "learned" else []
        summary["variants"][variant_spec["name"]] = {
            "description": variant_spec["description"],
            "route_mode": variant_spec["route_mode"],
            "trust_mode": variant_spec["trust_mode"],
            "structured_base": {
                "best_alpha": base_alpha,
                "feature_count": int(base_bundle["train"].shape[1]),
                "top_coefficients": top_coefficients(base_bundle["feature_names"], np.asarray(base_model.coef_, dtype=float)),
            },
            "semantic_expert": {
                "best_alpha": semantic_alpha,
                "feature_count": int(semantic_bundle["train"].shape[1]),
                "top_coefficients": top_coefficients(semantic_bundle["feature_names"], np.asarray(semantic_model.coef_, dtype=float)),
            },
            "quality_expert": {
                "best_alpha": quality_alpha,
                "feature_count": int(quality_bundle["train"].shape[1]),
                "top_coefficients": top_coefficients(quality_bundle["feature_names"], np.asarray(quality_model.coef_, dtype=float)),
            },
            "gate": {
                "learning_rate": best_variant["learning_rate"],
                "l2": best_variant["l2"],
                "incrementality_lambda": best_variant["incrementality_lambda"],
                "activation_lambda": best_variant["activation_lambda"],
                "trust_constraint": gate_constraint_label(variant_spec["trust_mode"]),
                "val_route_gate_mean": best_variant["val_route_gate_mean"],
                "val_trust_gate_mean": best_variant["val_trust_gate_mean"],
                "val_applied_dialog_effect_abs_mean": best_variant["val_applied_dialog_effect_abs_mean"],
                "test_route_gate_mean": float(np.mean(route_gate_test)),
                "test_route_gate_std": float(np.std(route_gate_test)),
                "test_trust_gate_mean": float(np.mean(trust_gate_test)),
                "test_trust_gate_std": float(np.std(trust_gate_test)),
                "test_mixed_dialog_correction_abs_mean": float(np.mean(np.abs(mixed_test))),
                "test_applied_dialog_effect_abs_mean": float(np.mean(np.abs(effect_test))),
                "top_route_features": top_coefficients(
                    route_features,
                    np.asarray(best_variant["params"].get("route_w", np.zeros(len(route_features))), dtype=float),
                )
                if route_features
                else [],
                "top_trust_features": top_coefficients(trust_features, trust_weights) if trust_weights.size else [],
                "val_incrementality": incrementality_diagnostics(val_prior, base_val_pred, effect_val),
                "test_incrementality": incrementality_diagnostics(test_prior, base_test_pred, effect_test),
                "effect_corr_with_semantic_corr_test": safe_correlation(effect_test, semantic_test_corr),
                "effect_corr_with_quality_corr_test": safe_correlation(effect_test, quality_test_corr),
            },
            "val": metrics(val_y, variant_val_pred),
            "test": metrics(test_y, variant_test_pred),
        }
        test_predictions[variant_spec["name"]] = np.asarray(variant_test_pred, dtype=float)
        extend_prediction_rows(prediction_rows, variant_spec["name"], "val", val_rows, variant_val_pred)
        extend_prediction_rows(prediction_rows, variant_spec["name"], "test", test_rows, variant_test_pred)
        extend_gate_rows(
            gate_rows,
            variant_spec["name"],
            "val",
            val_rows,
            val_prior,
            base_val_pred,
            semantic_val_corr,
            quality_val_corr,
            route_gate_val,
            trust_gate_val,
            mixed_val,
            effect_val,
            variant_val_pred,
        )
        extend_gate_rows(
            gate_rows,
            variant_spec["name"],
            "test",
            test_rows,
            test_prior,
            base_test_pred,
            semantic_test_corr,
            quality_test_corr,
            route_gate_test,
            trust_gate_test,
            mixed_test,
            effect_test,
            variant_test_pred,
        )

    significance_pairs = [
        ("residual_base_plus_semantic_plus_quality", "i_pog_qa_full"),
        ("i_pog_qa_full", "i_pog_qa_no_incrementality"),
        ("i_pog_qa_full", "i_pog_qa_no_activation_reg"),
        ("i_pog_qa_full", "i_pog_qa_free_trust"),
        ("i_pog_qa_full", "i_pog_qa_no_trust_gate"),
        ("i_pog_qa_full", "i_pog_qa_semantic_only"),
        ("i_pog_qa_full", "i_pog_qa_quality_only"),
    ]
    for model_a, model_b in significance_pairs:
        summary["significance"][f"{model_a}__vs__{model_b}"] = summarize_significance(
            test_y,
            test_predictions[model_a],
            test_predictions[model_b],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        )

    variant_ranking = [
        {
            "model_name": model_name,
            "test_r2": float(model_payload["test"]["r2"]),
            "test_rmse": float(model_payload["test"]["rmse"]),
        }
        for model_name, model_payload in summary["variants"].items()
    ]
    variant_ranking.sort(key=lambda item: item["test_r2"], reverse=True)
    summary["variant_ranking_by_test_r2"] = variant_ranking

    write_json(output_dir / "i_pog_qa_benchmark_suite_summary.json", summary)
    write_csv(output_dir / "i_pog_qa_benchmark_suite_predictions.csv", prediction_rows)
    write_csv(output_dir / "i_pog_qa_benchmark_suite_gate_diagnostics.csv", gate_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
