#!/usr/bin/env python3

from __future__ import annotations

import argparse
import site
import sys
from pathlib import Path

import numpy as np

user_site = site.getusersitepackages()
user_site_removed = False
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)
    user_site_removed = True

try:
    from sklearn.linear_model import Ridge
except ModuleNotFoundError:
    if user_site_removed and isinstance(user_site, str) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    from sklearn.linear_model import Ridge

from dj30_qc_utils import write_csv, write_json
from run_afterhours_precall_semantic_ladder import (
    A4_STRUCTURED_FEATURES,
    PRE_CALL_MARKET_FEATURES,
    build_rows,
)
from run_afterhours_transfer_role_text_signal_benchmark import build_shared_role_lsa_bundle
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle
from run_offhours_shock_ablations import (
    paired_bootstrap_deltas,
    paired_sign_permutation_pvalue,
)
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_signal_decomposition_benchmarks import CONTROL_FEATURES
from run_structured_baselines import metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark role-specific analyst-question / answer semantics against the "
            "clean after-hours fixed-split mainline."
        )
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_role_semantic_mainline_benchmark_clean_real"),
    )
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="fail")
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--residual-alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--qna-max-features", type=int, default=8000)
    parser.add_argument("--qna-min-df", type=int, default=2)
    parser.add_argument("--qna-lsa-components", type=int, default=64)
    parser.add_argument("--role-max-features", type=int, default=4000)
    parser.add_argument("--role-min-df", type=int, default=2)
    parser.add_argument("--role-lsa-components", type=int, default=8)
    parser.add_argument("--term-limit", type=int, default=12)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def top_coefficients(model, feature_names: list[str], limit: int = 10) -> list[dict[str, float | str]]:
    coeffs = np.asarray(model.coef_, dtype=float)
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


def zscore_bundle(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1.0
    return (train_x - means) / stds, (val_x - means) / stds, (test_x - means) / stds


def fit_residualized_role_bundle(
    qna_bundle: dict[str, object],
    role_train: np.ndarray,
    role_val: np.ndarray,
    role_test: np.ndarray,
    feature_prefix: str,
    residual_alphas: list[float],
) -> tuple[dict[str, object], dict[str, float | int]]:
    best_alpha = None
    best_model = None
    best_val_mse = None
    for alpha in residual_alphas:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(qna_bundle["train"], role_train)
        val_pred = model.predict(qna_bundle["val"])
        val_mse = float(np.mean((role_val - val_pred) ** 2))
        if best_val_mse is None or val_mse < best_val_mse:
            best_alpha = alpha
            best_model = model
            best_val_mse = val_mse

    train_resid = role_train - best_model.predict(qna_bundle["train"])
    val_resid = role_val - best_model.predict(qna_bundle["val"])
    test_resid = role_test - best_model.predict(qna_bundle["test"])
    train_z, val_z, test_z = zscore_bundle(train_resid, val_resid, test_resid)
    feature_names = [f"{feature_prefix}_{idx+1}" for idx in range(train_z.shape[1])]
    return (
        {
            "train": train_z,
            "val": val_z,
            "test": test_z,
            "feature_names": feature_names,
        },
        {
            "best_alpha": float(best_alpha),
            "val_reconstruction_mse": float(best_val_mse),
            "feature_count": int(train_z.shape[1]),
        },
    )


def text_coverage(rows: list[dict[str, object]]) -> dict[str, int]:
    coverage = {
        "rows": len(rows),
        "with_qna_text": 0,
        "with_question_text": 0,
        "with_answer_text": 0,
        "with_both_role_texts": 0,
    }
    for row in rows:
        qna_text = str(row.get("qna_text", "") or "").strip()
        question_text = str(row.get("question_text", "") or "").strip()
        answer_text = str(row.get("answer_text", "") or "").strip()
        if qna_text:
            coverage["with_qna_text"] += 1
        if question_text:
            coverage["with_question_text"] += 1
        if answer_text:
            coverage["with_answer_text"] += 1
        if question_text and answer_text:
            coverage["with_both_role_texts"] += 1
    return coverage


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args)
    train_rows = [row for row in rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in rows if row["_year"] == args.val_year]
    test_rows = [row for row in rows if row["_year"] > args.val_year]

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    residual_alphas = [float(item) for item in args.residual_alphas.split(",") if item.strip()]

    bundles = {
        "pre_call_market": build_dense_bundle(train_rows, val_rows, test_rows, PRE_CALL_MARKET_FEATURES),
        "controls": build_dense_bundle(train_rows, val_rows, test_rows, CONTROL_FEATURES),
        "a4": build_dense_bundle(train_rows, val_rows, test_rows, A4_STRUCTURED_FEATURES),
        "qna_lsa": build_text_lsa_bundle(
            train_rows,
            val_rows,
            test_rows,
            text_col="qna_text",
            max_features=args.qna_max_features,
            min_df=args.qna_min_df,
            lsa_components=args.qna_lsa_components,
        ),
    }
    role_bundle = build_shared_role_lsa_bundle(
        train_rows,
        val_rows,
        test_rows,
        max_features=args.role_max_features,
        min_df=args.role_min_df,
        lsa_components=args.role_lsa_components,
        term_limit=args.term_limit,
    )
    bundles["question_role"] = {
        "train": role_bundle["question_train"],
        "val": role_bundle["question_val"],
        "test": role_bundle["question_test"],
        "feature_names": role_bundle["question_feature_names"],
    }
    bundles["answer_role"] = {
        "train": role_bundle["answer_train"],
        "val": role_bundle["answer_val"],
        "test": role_bundle["answer_test"],
        "feature_names": role_bundle["answer_feature_names"],
    }

    bundles["question_role_resid"], question_resid_meta = fit_residualized_role_bundle(
        bundles["qna_lsa"],
        role_bundle["question_train"],
        role_bundle["question_val"],
        role_bundle["question_test"],
        feature_prefix="question_role_resid",
        residual_alphas=residual_alphas,
    )
    bundles["answer_role_resid"], answer_resid_meta = fit_residualized_role_bundle(
        bundles["qna_lsa"],
        role_bundle["answer_train"],
        role_bundle["answer_val"],
        role_bundle["answer_test"],
        feature_prefix="answer_role_resid",
        residual_alphas=residual_alphas,
    )

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
    val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
    test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)

    model_specs = {
        "prior_only": [],
        "residual_pre_call_market_only": ["pre_call_market"],
        "residual_pre_call_market_plus_a4_plus_qna_lsa": ["pre_call_market", "a4", "qna_lsa"],
        "residual_pre_call_market_plus_a4_plus_question_role": ["pre_call_market", "a4", "question_role"],
        "residual_pre_call_market_plus_a4_plus_answer_role": ["pre_call_market", "a4", "answer_role"],
        "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_question_role": [
            "pre_call_market",
            "a4",
            "qna_lsa",
            "question_role",
        ],
        "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_answer_role": [
            "pre_call_market",
            "a4",
            "qna_lsa",
            "answer_role",
        ],
        "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_question_role_resid": [
            "pre_call_market",
            "a4",
            "qna_lsa",
            "question_role_resid",
        ],
        "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_answer_role_resid": [
            "pre_call_market",
            "a4",
            "qna_lsa",
            "answer_role_resid",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_question_role": [
            "pre_call_market",
            "controls",
            "a4",
            "question_role",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_answer_role": [
            "pre_call_market",
            "controls",
            "a4",
            "answer_role",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_question_role": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
            "question_role",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_answer_role": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
            "answer_role",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_question_role_resid": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
            "question_role_resid",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_answer_role_resid": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
            "answer_role_resid",
        ],
    }

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": [item.strip() for item in args.include_regimes.split(",") if item.strip()],
        "exclude_html_flags": [item.strip() for item in args.exclude_html_flags.split(",") if item.strip()],
        "split_sizes": {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)},
        "text_coverage": {
            "train": text_coverage(train_rows),
            "val": text_coverage(val_rows),
            "test": text_coverage(test_rows),
            "all_rows": text_coverage(rows),
        },
        "config": {
            "qna_lsa_components": args.qna_lsa_components,
            "role_lsa_components": int(role_bundle["n_components"]),
            "role_explained_variance_ratio_sum": float(role_bundle["explained_variance_ratio_sum"]),
        },
        "role_residualization": {
            "question_role_resid": question_resid_meta,
            "answer_role_resid": answer_resid_meta,
        },
        "models": {},
        "significance": {},
    }
    prediction_rows: list[dict[str, object]] = []
    overview_rows: list[dict[str, object]] = []
    test_predictions: dict[str, np.ndarray] = {}

    for model_name, bundle_names in model_specs.items():
        if not bundle_names:
            pred_val = val_prior
            pred_test = test_prior
            summary["models"][model_name] = {
                "family": "prior_passthrough",
                "feature_bundles": [],
                "feature_count": 1,
                "val": metrics(val_y, pred_val),
                "test": metrics(test_y, pred_test),
            }
        else:
            train_parts = []
            val_parts = []
            test_parts = []
            feature_names = []
            for bundle_name in bundle_names:
                bundle = bundles[bundle_name]
                train_parts.append(bundle["train"])
                val_parts.append(bundle["val"])
                test_parts.append(bundle["test"])
                feature_names.extend(bundle["feature_names"])
            train_x = np.hstack(train_parts)
            val_x = np.hstack(val_parts)
            test_x = np.hstack(test_parts)
            best_alpha, best_model, pred_val = fit_residual_ridge(
                train_x,
                train_prior,
                train_y,
                val_x,
                val_prior,
                val_y,
                alphas,
            )
            pred_test = test_prior + best_model.predict(test_x)
            summary["models"][model_name] = {
                "family": "residual_ridge",
                "feature_bundles": bundle_names,
                "feature_count": int(train_x.shape[1]),
                "best_alpha": float(best_alpha),
                "val": metrics(val_y, pred_val),
                "test": metrics(test_y, pred_test),
                "top_coefficients": top_coefficients(best_model, feature_names),
            }

        test_predictions[model_name] = np.asarray(pred_test, dtype=float)
        overview_rows.append(
            {
                "model": model_name,
                "feature_bundles": ",".join(bundle_names),
                "feature_count": summary["models"][model_name]["feature_count"],
                "val_r2": summary["models"][model_name]["val"]["r2"],
                "test_r2": summary["models"][model_name]["test"]["r2"],
                "test_rmse": summary["models"][model_name]["test"]["rmse"],
            }
        )
        for split_name, split_rows_, preds in [
            (f"{model_name}_val", val_rows, pred_val),
            (f"{model_name}_test", test_rows, pred_test),
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

    comparisons = [
        (
            "residual_pre_call_market_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_a4_plus_question_role",
        ),
        (
            "residual_pre_call_market_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_question_role",
        ),
        (
            "residual_pre_call_market_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_question_role_resid",
        ),
        (
            "residual_pre_call_market_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_a4_plus_answer_role",
        ),
        (
            "residual_pre_call_market_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_answer_role",
        ),
        (
            "residual_pre_call_market_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_answer_role_resid",
        ),
        (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_controls_plus_a4_plus_question_role",
        ),
        (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_question_role",
        ),
        (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_question_role_resid",
        ),
        (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_controls_plus_a4_plus_answer_role",
        ),
        (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_answer_role",
        ),
        (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_answer_role_resid",
        ),
    ]
    for model_a, model_b in comparisons:
        key = f"{model_a}__vs__{model_b}"
        summary["significance"][key] = {
            **paired_bootstrap_deltas(
                test_y,
                test_predictions[model_a],
                test_predictions[model_b],
                args.bootstrap_iters,
                args.seed,
            ),
            **paired_sign_permutation_pvalue(
                test_y,
                test_predictions[model_a],
                test_predictions[model_b],
                args.perm_iters,
                args.seed,
            ),
        }

    write_json(output_dir / "afterhours_role_semantic_mainline_benchmark_summary.json", summary)
    write_csv(output_dir / "afterhours_role_semantic_mainline_benchmark_overview.csv", overview_rows)
    write_csv(output_dir / "afterhours_role_semantic_mainline_benchmark_predictions.csv", prediction_rows)
    write_csv(
        output_dir / "afterhours_role_semantic_mainline_component_terms.csv",
        role_bundle["component_term_rows"],
    )


if __name__ == "__main__":
    main()
