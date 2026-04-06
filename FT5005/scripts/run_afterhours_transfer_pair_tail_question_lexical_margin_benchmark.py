#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from dj30_qc_utils import write_csv, write_json
from run_afterhours_transfer_agreement_signal_benchmark import (
    MODEL_AGREED,
    MODEL_HARD_ABSTENTION,
    MODEL_PRE_ONLY,
    agreement_gain_target,
    build_temporal_rows,
    fit_ridge,
    summarize_significance,
)
from run_afterhours_transfer_pair_tail_question_lexical_pattern_benchmark import (
    FAMILY_FEATURES,
    as_array,
    attach_lexical_features,
    build_factor,
    top_coefficients,
)
from run_structured_baselines import metrics

FAMILY = "clarify_modeling_lex"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a conservative margin threshold on the compact hardest-question lexical factor."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--text-views-csv",
        type=Path,
        default=Path("results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv"),
    )
    parser.add_argument(
        "--reference-predictions-csv",
        type=Path,
        default=Path(
            "results/afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_real/"
            "afterhours_transfer_pair_tail_question_lexical_pattern_benchmark_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_pair_tail_question_lexical_margin_benchmark_real"),
    )
    parser.add_argument("--train-split", default="val2020_test_post2020")
    parser.add_argument("--val-split", default="val2021_test_post2021")
    parser.add_argument("--test-split", default="val2022_test_post2022")
    parser.add_argument("--refit-train-splits", default="val2020_test_post2020,val2021_test_post2021")
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_reference_predictions(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    lookup = {row["event_key"]: row for row in csv.DictReader(path.open())}
    refs = {}
    for name, col in [
        (MODEL_PRE_ONLY, MODEL_PRE_ONLY),
        (MODEL_HARD_ABSTENTION, MODEL_HARD_ABSTENTION),
        ("question_lsa4_bi", "question_lsa4_bi"),
        ("clarify_modeling_lex_factor_pca1", "clarify_modeling_lex_factor_pca1"),
    ]:
        refs[name] = np.asarray([float(lookup[key][col]) for key in keys], dtype=float)
    return refs


def threshold_grid(signals: np.ndarray) -> list[float]:
    base = [0.0]
    quants = np.linspace(0.0, 0.5, 11)
    for q in quants:
        val = float(np.quantile(signals, q))
        if val <= 0.0:
            base.append(val)
    return sorted(set(base))


def route_from_signal(signal: np.ndarray, agreed: np.ndarray, pre: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    use_agreed = signal > threshold
    pred = np.where(use_agreed, agreed, pre)
    return pred, use_agreed.astype(int)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_temporal_rows(args.temporal_root.resolve())
    coverage, _ = attach_lexical_features(rows, args.text_views_csv.resolve())
    refit_train_splits = parse_list(args.refit_train_splits)
    alpha_grid = [float(x) for x in parse_list(args.alphas)]

    train_rows = [r for r in rows if r["split"] == args.train_split and int(r["agreement"]) == 1]
    val_rows = [r for r in rows if r["split"] == args.val_split and int(r["agreement"]) == 1]
    refit_rows = [r for r in rows if r["split"] in refit_train_splits and int(r["agreement"]) == 1]
    test_agreement_rows = [r for r in rows if r["split"] == args.test_split and int(r["agreement"]) == 1]
    test_full_rows = [r for r in rows if r["split"] == args.test_split]
    if not train_rows or not val_rows or not refit_rows or not test_agreement_rows or not test_full_rows:
        raise SystemExit("missing rows for lexical margin benchmark")

    features = FAMILY_FEATURES[FAMILY]
    tune_factor = build_factor(train_rows, val_rows, test_agreement_rows, features)
    refit_factor = build_factor(refit_rows, [], test_agreement_rows, features)

    train_x = tune_factor["train_factor"]
    val_x = tune_factor["val_factor"]
    refit_x = refit_factor["train_factor"]
    test_x = refit_factor["test_factor"]

    train_gain = agreement_gain_target(train_rows)
    val_target = as_array(val_rows, "target")
    val_agreed = as_array(val_rows, MODEL_AGREED)
    val_pre = as_array(val_rows, MODEL_PRE_ONLY)
    test_full_target = as_array(test_full_rows, "target")
    test_full_pre = as_array(test_full_rows, MODEL_PRE_ONLY)
    test_full_hard = as_array(test_full_rows, MODEL_HARD_ABSTENTION)
    test_agreed = as_array(test_agreement_rows, MODEL_AGREED)
    test_pre = as_array(test_agreement_rows, MODEL_PRE_ONLY)

    tuning_rows = []
    best_alpha = None
    best_threshold = None
    best_val_r2 = None
    best_val_payload = None
    for alpha in alpha_grid:
        model = fit_ridge(train_x, train_gain, alpha)
        val_signal = model.predict(val_x)
        for threshold in threshold_grid(val_signal):
            val_pred, val_use_agreed = route_from_signal(val_signal, val_agreed, val_pre, threshold)
            score = metrics(val_target, val_pred)
            row = {
                "alpha": float(alpha),
                "threshold": float(threshold),
                "val_r2": score["r2"],
                "val_rmse": score["rmse"],
                "val_mae": score["mae"],
                "val_use_agreed_share": float(np.mean(val_use_agreed)),
                "val_veto_share": float(np.mean(1 - val_use_agreed)),
            }
            tuning_rows.append(row)
            if best_val_r2 is None or score["r2"] > best_val_r2:
                best_val_r2 = score["r2"]
                best_alpha = float(alpha)
                best_threshold = float(threshold)
                best_val_payload = row

    refit_model = fit_ridge(refit_x, agreement_gain_target(refit_rows), best_alpha)
    test_signal = refit_model.predict(test_x)
    base_test_pred, base_use_agreed = route_from_signal(test_signal, test_agreed, test_pre, 0.0)
    margin_test_pred, margin_use_agreed = route_from_signal(test_signal, test_agreed, test_pre, best_threshold)

    base_lookup = {r["event_key"]: float(p) for r, p in zip(test_agreement_rows, base_test_pred)}
    margin_lookup = {r["event_key"]: float(p) for r, p in zip(test_agreement_rows, margin_test_pred)}
    signal_lookup = {r["event_key"]: float(s) for r, s in zip(test_agreement_rows, test_signal)}
    base_use_lookup = {r["event_key"]: int(u) for r, u in zip(test_agreement_rows, base_use_agreed)}
    margin_use_lookup = {r["event_key"]: int(u) for r, u in zip(test_agreement_rows, margin_use_agreed)}

    full_base_pred = np.asarray([base_lookup.get(r["event_key"], float(r[MODEL_PRE_ONLY])) for r in test_full_rows], dtype=float)
    full_margin_pred = np.asarray([margin_lookup.get(r["event_key"], float(r[MODEL_PRE_ONLY])) for r in test_full_rows], dtype=float)

    prediction_rows = [dict(r) for r in test_full_rows]
    for row in prediction_rows:
        key = row["event_key"]
        row["clarify_modeling_lex_factor_pca1_refit"] = base_lookup.get(key, float(row[MODEL_PRE_ONLY]))
        row["clarify_modeling_lex_factor_pca1_refit__predicted_gain_signal"] = signal_lookup.get(key, 0.0)
        row["clarify_modeling_lex_factor_pca1_refit__use_agreed"] = base_use_lookup.get(key, 0)
        row["clarify_modeling_margin_route"] = margin_lookup.get(key, float(row[MODEL_PRE_ONLY]))
        row["clarify_modeling_margin_route__predicted_gain_signal"] = signal_lookup.get(key, 0.0)
        row["clarify_modeling_margin_route__use_agreed"] = margin_use_lookup.get(key, 0)

    keys = [str(r["event_key"]) for r in test_full_rows]
    refs = load_reference_predictions(args.reference_predictions_csv.resolve(), keys)
    for idx, row in enumerate(prediction_rows):
        for name, arr_ in refs.items():
            row[name] = float(arr_[idx])

    base_metrics = metrics(test_full_target, full_base_pred)
    margin_metrics = metrics(test_full_target, full_margin_pred)
    summary = {
        "config": {
            "temporal_root": str(args.temporal_root.resolve()),
            "text_views_csv": str(args.text_views_csv.resolve()),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "refit_train_splits": refit_train_splits,
            "test_split": args.test_split,
            "alpha_grid": alpha_grid,
            "family": FAMILY,
        },
        "coverage": coverage,
        "feature_names": features,
        "factor_meta": {
            "explained_variance_ratio_tune": tune_factor["explained_variance_ratio"],
            "explained_variance_ratio_refit": refit_factor["explained_variance_ratio"],
            "refit_loadings": refit_factor["loadings"],
        },
        "selected": {
            "alpha": best_alpha,
            "threshold": best_threshold,
            "best_validation": best_val_payload,
            "coef_rows": top_coefficients(refit_model, [f"{FAMILY}_factor_pca1__score"]),
        },
        "reference": {
            "test_full_pre": metrics(test_full_target, test_full_pre),
            "test_full_hard": metrics(test_full_target, test_full_hard),
            "test_full_question_lsa4_bi": metrics(test_full_target, refs["question_lsa4_bi"]),
        },
        "routes": {
            "base_factor": {
                "metrics": base_metrics,
                "use_agreed_share": float(np.mean(base_use_agreed)),
                "veto_rows": int(np.sum(1 - base_use_agreed)),
                "significance_vs_hard": summarize_significance(test_full_target, test_full_hard, full_base_pred, args.bootstrap_iters, args.perm_iters, args.seed),
            },
            "margin_route": {
                "metrics": margin_metrics,
                "use_agreed_share": float(np.mean(margin_use_agreed)),
                "veto_rows": int(np.sum(1 - margin_use_agreed)),
                "significance_vs_hard": summarize_significance(test_full_target, test_full_hard, full_margin_pred, args.bootstrap_iters, args.perm_iters, args.seed),
                "significance_vs_base_factor": summarize_significance(test_full_target, full_base_pred, full_margin_pred, args.bootstrap_iters, args.perm_iters, args.seed),
                "significance_vs_question_lsa4_bi": summarize_significance(test_full_target, refs["question_lsa4_bi"], full_margin_pred, args.bootstrap_iters, args.perm_iters, args.seed),
            },
        },
    }

    overview_rows = [
        {
            "route": "base_factor",
            "selected_alpha": best_alpha,
            "threshold": 0.0,
            "test_r2": base_metrics["r2"],
            "test_rmse": base_metrics["rmse"],
            "test_mae": base_metrics["mae"],
            "use_agreed_share": float(np.mean(base_use_agreed)),
            "veto_rows": int(np.sum(1 - base_use_agreed)),
            "p_mse_vs_hard": summary["routes"]["base_factor"]["significance_vs_hard"]["mse_gain_pvalue"],
        },
        {
            "route": "margin_route",
            "selected_alpha": best_alpha,
            "threshold": best_threshold,
            "test_r2": margin_metrics["r2"],
            "test_rmse": margin_metrics["rmse"],
            "test_mae": margin_metrics["mae"],
            "use_agreed_share": float(np.mean(margin_use_agreed)),
            "veto_rows": int(np.sum(1 - margin_use_agreed)),
            "p_mse_vs_hard": summary["routes"]["margin_route"]["significance_vs_hard"]["mse_gain_pvalue"],
            "p_mse_vs_base_factor": summary["routes"]["margin_route"]["significance_vs_base_factor"]["mse_gain_pvalue"],
        },
    ]

    write_json(output_dir / "afterhours_transfer_pair_tail_question_lexical_margin_benchmark_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_margin_benchmark_tuning.csv", tuning_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_margin_benchmark_overview.csv", overview_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_question_lexical_margin_benchmark_test_predictions.csv", prediction_rows)


if __name__ == "__main__":
    main()
