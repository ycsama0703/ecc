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
from run_dense_multimodal_ablation_baselines import (
    build_text_lsa_bundle,
    infer_prefixed_feature_names,
    load_joined_rows,
)
from run_offhours_shock_ablations import paired_bootstrap_deltas, paired_sign_permutation_pvalue
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets
from run_text_tfidf_baselines import EXTRA_DENSE_FEATURES, STRUCTURED_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train prior-gated residual models on the fixed off-hours shock setting."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/prior_gated_residual_real"),
    )
    parser.add_argument("--include-regimes", default="pre_market,after_hours")
    parser.add_argument("--exclude-html-flags", default="")
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--learning-rates", default="0.01,0.03")
    parser.add_argument("--l2-values", default="0.0,1e-4,1e-3,1e-2")
    parser.add_argument("--epochs", type=int, default=2500)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    return parser.parse_args()


def regime_label(row: dict[str, str]) -> str:
    hour = float(row.get("scheduled_hour_et", 0.0))
    if hour < 9.5:
        return "pre_market"
    if hour < 16.0:
        return "market_hours"
    return "after_hours"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def init_params(feature_dim: int, rng: np.random.Generator) -> dict[str, np.ndarray | float]:
    scale = 0.01
    return {
        "wr": rng.normal(0.0, scale, size=feature_dim),
        "br": 0.0,
        "wg": rng.normal(0.0, scale, size=feature_dim),
        "bg": -1.5,
    }


def predict_with_params(x: np.ndarray, prior: np.ndarray, params: dict[str, np.ndarray | float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    residual = x @ params["wr"] + params["br"]
    gate_logits = x @ params["wg"] + params["bg"]
    gate = sigmoid(gate_logits)
    pred = prior + gate * residual
    return pred, gate, residual


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))


def train_gated_model(
    train_x: np.ndarray,
    train_prior: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_prior: np.ndarray,
    val_y: np.ndarray,
    learning_rates: list[float],
    l2_values: list[float],
    epochs: int,
    patience: int,
    seed: int,
) -> tuple[dict, np.ndarray]:
    rng_master = np.random.default_rng(seed)
    best = None

    for lr in learning_rates:
        for l2 in l2_values:
            rng = np.random.default_rng(int(rng_master.integers(1_000_000_000)))
            params = init_params(train_x.shape[1], rng)
            adam_state = {
                "m_wr": np.zeros_like(params["wr"]),
                "v_wr": np.zeros_like(params["wr"]),
                "m_br": 0.0,
                "v_br": 0.0,
                "m_wg": np.zeros_like(params["wg"]),
                "v_wg": np.zeros_like(params["wg"]),
                "m_bg": 0.0,
                "v_bg": 0.0,
            }
            beta1 = 0.9
            beta2 = 0.999
            eps_opt = 1e-8
            best_local_val = None
            best_local_params = None
            stale = 0

            for step in range(1, epochs + 1):
                pred, gate, residual = predict_with_params(train_x, train_prior, params)
                err = pred - train_y
                n = max(len(train_y), 1)
                dloss_dpred = 2.0 * err / n
                dloss_dres = dloss_dpred * gate
                dloss_dgate = dloss_dpred * residual
                dloss_dlogit = dloss_dgate * gate * (1.0 - gate)

                grad_wr = train_x.T @ dloss_dres + 2.0 * l2 * params["wr"]
                grad_br = float(np.sum(dloss_dres))
                grad_wg = train_x.T @ dloss_dlogit + 2.0 * l2 * params["wg"]
                grad_bg = float(np.sum(dloss_dlogit))

                for key, grad, scalar in [
                    ("wr", grad_wr, False),
                    ("br", grad_br, True),
                    ("wg", grad_wg, False),
                    ("bg", grad_bg, True),
                ]:
                    m_key = f"m_{key}"
                    v_key = f"v_{key}"
                    adam_state[m_key] = beta1 * adam_state[m_key] + (1.0 - beta1) * grad
                    adam_state[v_key] = beta2 * adam_state[v_key] + (1.0 - beta2) * (grad * grad)
                    m_hat = adam_state[m_key] / (1.0 - beta1 ** step)
                    v_hat = adam_state[v_key] / (1.0 - beta2 ** step)
                    params[key] = params[key] - lr * m_hat / (np.sqrt(v_hat) + eps_opt)

                val_pred, _, _ = predict_with_params(val_x, val_prior, params)
                val_rmse = metrics(val_y, val_pred)["rmse"]
                if best_local_val is None or val_rmse < best_local_val:
                    best_local_val = val_rmse
                    best_local_params = {
                        "wr": params["wr"].copy(),
                        "br": float(params["br"]),
                        "wg": params["wg"].copy(),
                        "bg": float(params["bg"]),
                    }
                    stale = 0
                else:
                    stale += 1
                if stale >= patience:
                    break

            final_val_pred, _, _ = predict_with_params(val_x, val_prior, best_local_params)
            final_val_rmse = metrics(val_y, final_val_pred)["rmse"]
            candidate = {
                "learning_rate": lr,
                "l2": l2,
                "params": best_local_params,
                "val_rmse": final_val_rmse,
                "val_pred": final_val_pred,
            }
            if best is None or candidate["val_rmse"] < best["val_rmse"]:
                best = candidate

    return best, best["val_pred"]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    learning_rates = [float(item) for item in args.learning_rates.split(",") if item.strip()]
    l2_values = [float(item) for item in args.l2_values.split(",") if item.strip()]

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
        if regime_label(row) not in include_regimes:
            continue
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        rows.append(item)

    rows = attach_ticker_prior(rows, args.train_end_year)
    train_rows = [row for row in rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in rows if row["_year"] == args.val_year]
    test_rows = [row for row in rows if row["_year"] > args.val_year]

    bundles = {
        "structured": build_dense_bundle(train_rows, val_rows, test_rows, STRUCTURED_FEATURES),
        "structured_plus_extra": build_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            STRUCTURED_FEATURES + EXTRA_DENSE_FEATURES,
        ),
        "structured_plus_extra_plus_qna_lsa": None,
    }
    qna_lsa = build_text_lsa_bundle(
        train_rows,
        val_rows,
        test_rows,
        text_col="qna_text",
        max_features=8000,
        min_df=2,
        lsa_components=64,
    )
    dense_plus_extra = bundles["structured_plus_extra"]
    bundles["structured_plus_extra_plus_qna_lsa"] = {
        "train": np.hstack([dense_plus_extra["train"], qna_lsa["train"]]),
        "val": np.hstack([dense_plus_extra["val"], qna_lsa["val"]]),
        "test": np.hstack([dense_plus_extra["test"], qna_lsa["test"]]),
        "feature_names": dense_plus_extra["feature_names"] + qna_lsa["feature_names"],
    }

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
    val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
    test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted(include_regimes),
        "exclude_html_flags": sorted(exclude_html_flags),
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "models": {},
        "significance_vs_ridge": {},
    }
    prediction_rows = []

    for model_name, bundle in bundles.items():
        ridge_alpha, ridge_model, ridge_val_pred = fit_residual_ridge(
            bundle["train"], train_prior, train_y, bundle["val"], val_prior, val_y, alphas
        )
        ridge_test_pred = test_prior + ridge_model.predict(bundle["test"])

        gated_best, gated_val_pred = train_gated_model(
            bundle["train"],
            train_prior,
            train_y,
            bundle["val"],
            val_prior,
            val_y,
            learning_rates,
            l2_values,
            args.epochs,
            args.patience,
            args.seed,
        )
        gated_test_pred, gated_test_gate, gated_test_residual = predict_with_params(
            bundle["test"], test_prior, gated_best["params"]
        )

        summary["models"][model_name] = {
            "ridge": {
                "best_alpha": ridge_alpha,
                "val": metrics(val_y, ridge_val_pred),
                "test": metrics(test_y, ridge_test_pred),
            },
            "gated": {
                "learning_rate": gated_best["learning_rate"],
                "l2": gated_best["l2"],
                "val": metrics(val_y, gated_val_pred),
                "test": metrics(test_y, gated_test_pred),
                "test_gate_mean": float(np.mean(gated_test_gate)),
                "test_gate_std": float(np.std(gated_test_gate)),
                "test_abs_residual_mean": float(np.mean(np.abs(gated_test_residual))),
            },
        }

        summary["significance_vs_ridge"][model_name] = {
            "bootstrap": paired_bootstrap_deltas(
                test_y,
                ridge_test_pred,
                gated_test_pred,
                args.bootstrap_iters,
                args.seed,
            ),
            "permutation": paired_sign_permutation_pvalue(
                test_y,
                ridge_test_pred,
                gated_test_pred,
                args.perm_iters,
                args.seed + 1,
            ),
        }

        for split_name, split_rows_, preds in [
            (f"{model_name}_ridge_val", val_rows, ridge_val_pred),
            (f"{model_name}_ridge_test", test_rows, ridge_test_pred),
            (f"{model_name}_gated_val", val_rows, gated_val_pred),
            (f"{model_name}_gated_test", test_rows, gated_test_pred),
        ]:
            for row, pred in zip(split_rows_, preds):
                prediction_rows.append(
                    {
                        "model_split": split_name,
                        "event_key": row["event_key"],
                        "ticker": row["ticker"],
                        "year": row["year"],
                        "y_true": row["_target"],
                        "y_pred": float(pred),
                    }
                )

    write_json(output_dir / "prior_gated_residual_summary.json", summary)
    write_csv(output_dir / "prior_gated_residual_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
