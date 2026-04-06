#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
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
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    if user_site_removed and isinstance(user_site, str) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

from dj30_qc_utils import write_csv, write_json
from run_afterhours_transfer_expert_selection import summarize_significance
from run_structured_baselines import metrics


MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_QA_EXPERT = "residual_pre_call_market_plus_a4_plus_qa_benchmark_svd_observability_gate"
MODEL_RETAINED = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate"
MODEL_SELECTED = "validation_selected_transfer_expert"
MODEL_PAIR_TREE = "conservative_tree_override_on_selected_expert"
MODEL_PLUS_TEXT_LOGISTIC = "conservative_logistic_override_on_selected_expert"

MODEL_TRANSFER_CANDIDATE = "agreement_avg_transfer_candidate"
MODEL_HARD_ABSTENTION = "agreement_pre_only_abstention"
MODEL_GAIN_GATE = "compact_learnable_transfer_gain_gate"
MODEL_SOFT_TRUST = "compact_learnable_transfer_soft_trust"

FEATURE_NAMES = [
    "agreement",
    "candidate_minus_pre_pred",
    "qa_minus_pre_pred",
    "sem_minus_pre_pred",
    "selected_minus_pre_pred",
    "tree_minus_pre_pred",
    "logistic_minus_pre_pred",
    "abs_pair_minus_logistic_pred",
]

SPLITS = [
    "val2020_test_post2020",
    "val2021_test_post2021",
    "val2022_test_post2022",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compact learnable transfer trust calibrators against hard agreement-based abstention."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_learnable_trust_calibrator_role_aware_audio_lsa4_real"),
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


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["event_key"]: row for row in csv.DictReader(handle)}


def build_rows(temporal_root: Path) -> list[dict[str, float | int | str]]:
    rows = []
    for split_name in SPLITS:
        pair_path = temporal_root / f"{split_name}__hybrid_pair_bench_tree" / "afterhours_transfer_conservative_router_predictions.csv"
        logistic_path = temporal_root / f"{split_name}__hybrid_plus_text_logistic" / "afterhours_transfer_conservative_router_predictions.csv"
        pair_rows = load_rows(pair_path)
        logistic_rows = load_rows(logistic_path)
        keys = sorted(set(pair_rows) & set(logistic_rows))
        if not keys:
            raise SystemExit(f"no shared temporal rows for split {split_name}")

        for key in keys:
            pair_row = pair_rows[key]
            logistic_row = logistic_rows[key]
            pre = float(pair_row[MODEL_PRE_ONLY])
            qa = float(pair_row[MODEL_QA_EXPERT])
            retained = float(pair_row[MODEL_RETAINED])
            selected = float(pair_row[MODEL_SELECTED])
            pair_tree = float(pair_row[MODEL_PAIR_TREE])
            logistic = float(logistic_row[MODEL_PLUS_TEXT_LOGISTIC])
            target = float(pair_row["target"])
            tree_choose_qa = int(float(pair_row["tree_choose_qa"]))
            logistic_choose_qa = int(float(logistic_row["logistic_choose_qa"]))
            agreement = int(tree_choose_qa == logistic_choose_qa)
            agreed_pred = qa if tree_choose_qa == 1 else retained
            candidate = agreed_pred if agreement == 1 else 0.5 * (pair_tree + logistic)
            hard_abstention = agreed_pred if agreement == 1 else pre

            row = {
                "split": split_name,
                "event_key": key,
                "ticker": pair_row["ticker"],
                "year": int(float(pair_row["year"])),
                "target": target,
                MODEL_PRE_ONLY: pre,
                MODEL_QA_EXPERT: qa,
                MODEL_RETAINED: retained,
                MODEL_SELECTED: selected,
                MODEL_PAIR_TREE: pair_tree,
                MODEL_PLUS_TEXT_LOGISTIC: logistic,
                MODEL_TRANSFER_CANDIDATE: candidate,
                MODEL_HARD_ABSTENTION: hard_abstention,
                "agreement": agreement,
                "tree_choose_qa": tree_choose_qa,
                "logistic_choose_qa": logistic_choose_qa,
                "candidate_minus_pre_pred": candidate - pre,
                "qa_minus_pre_pred": qa - pre,
                "sem_minus_pre_pred": retained - pre,
                "selected_minus_pre_pred": selected - pre,
                "tree_minus_pre_pred": pair_tree - pre,
                "logistic_minus_pre_pred": logistic - pre,
                "pair_minus_logistic_pred": pair_tree - logistic,
                "abs_pair_minus_logistic_pred": abs(pair_tree - logistic),
            }
            rows.append(row)
    return rows


def as_array(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def feature_matrix(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=float)


def candidate_gain_target(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    target = as_array(rows, "target")
    pre = as_array(rows, MODEL_PRE_ONLY)
    candidate = as_array(rows, MODEL_TRANSFER_CANDIDATE)
    return (target - pre) ** 2 - (target - candidate) ** 2


def optimal_lambda_target(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    target = as_array(rows, "target")
    pre = as_array(rows, MODEL_PRE_ONLY)
    candidate = as_array(rows, MODEL_TRANSFER_CANDIDATE)
    delta = candidate - pre
    denom = delta ** 2
    out = np.zeros_like(delta)
    mask = denom > 0
    out[mask] = ((target[mask] - pre[mask]) * delta[mask]) / denom[mask]
    return np.clip(out, 0.0, 1.0)


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(train_x, train_y)
    return model


def calibrator_coefficients(model: Pipeline) -> list[dict[str, float | str]]:
    ridge = model.named_steps["ridge"]
    return [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in sorted(zip(FEATURE_NAMES, ridge.coef_.tolist()), key=lambda item: abs(item[1]), reverse=True)
    ]


def pick_best_alpha_for_gate(train_rows: list[dict[str, float | int | str]], val_rows: list[dict[str, float | int | str]], alpha_grid: list[float]) -> tuple[float, list[dict[str, float]]]:
    train_x = feature_matrix(train_rows)
    val_x = feature_matrix(val_rows)
    train_gain = candidate_gain_target(train_rows)
    val_target = as_array(val_rows, "target")
    val_pre = as_array(val_rows, MODEL_PRE_ONLY)
    val_candidate = as_array(val_rows, MODEL_TRANSFER_CANDIDATE)
    tuning_rows = []
    best_alpha = None
    best_r2 = None
    for alpha in alpha_grid:
        model = fit_ridge(train_x, train_gain, alpha)
        pred_gain = model.predict(val_x)
        pred = np.where(pred_gain > 0.0, val_candidate, val_pre)
        score = metrics(val_target, pred)
        tuning_rows.append(
            {
                "calibrator": MODEL_GAIN_GATE,
                "alpha": float(alpha),
                "val_r2": score["r2"],
                "val_rmse": score["rmse"],
                "val_mae": score["mae"],
                "val_candidate_share": float(np.mean(pred_gain > 0.0)),
                "val_pred_signal_min": float(np.min(pred_gain)),
                "val_pred_signal_max": float(np.max(pred_gain)),
            }
        )
        if best_r2 is None or score["r2"] > best_r2:
            best_r2 = score["r2"]
            best_alpha = float(alpha)
    return float(best_alpha), tuning_rows


def pick_best_alpha_for_soft(train_rows: list[dict[str, float | int | str]], val_rows: list[dict[str, float | int | str]], alpha_grid: list[float]) -> tuple[float, list[dict[str, float]]]:
    train_x = feature_matrix(train_rows)
    val_x = feature_matrix(val_rows)
    train_lambda = optimal_lambda_target(train_rows)
    val_target = as_array(val_rows, "target")
    val_pre = as_array(val_rows, MODEL_PRE_ONLY)
    val_candidate = as_array(val_rows, MODEL_TRANSFER_CANDIDATE)
    tuning_rows = []
    best_alpha = None
    best_r2 = None
    for alpha in alpha_grid:
        model = fit_ridge(train_x, train_lambda, alpha)
        pred_lambda = np.clip(model.predict(val_x), 0.0, 1.0)
        pred = val_pre + pred_lambda * (val_candidate - val_pre)
        score = metrics(val_target, pred)
        tuning_rows.append(
            {
                "calibrator": MODEL_SOFT_TRUST,
                "alpha": float(alpha),
                "val_r2": score["r2"],
                "val_rmse": score["rmse"],
                "val_mae": score["mae"],
                "val_lambda_mean": float(np.mean(pred_lambda)),
                "val_lambda_min": float(np.min(pred_lambda)),
                "val_lambda_max": float(np.max(pred_lambda)),
            }
        )
        if best_r2 is None or score["r2"] > best_r2:
            best_r2 = score["r2"]
            best_alpha = float(alpha)
    return float(best_alpha), tuning_rows


def main() -> None:
    args = parse_args()
    temporal_root = args.temporal_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(temporal_root)
    alpha_grid = [float(item) for item in parse_list(args.alphas)]
    refit_train_splits = parse_list(args.refit_train_splits)

    train_rows = [row for row in rows if row["split"] == args.train_split]
    val_rows = [row for row in rows if row["split"] == args.val_split]
    refit_rows = [row for row in rows if row["split"] in refit_train_splits]
    test_rows = [row for row in rows if row["split"] == args.test_split]
    if not train_rows or not val_rows or not refit_rows or not test_rows:
        raise SystemExit("missing train/val/refit/test rows for trust calibrator benchmark")

    gate_alpha, gate_tuning = pick_best_alpha_for_gate(train_rows, val_rows, alpha_grid)
    soft_alpha, soft_tuning = pick_best_alpha_for_soft(train_rows, val_rows, alpha_grid)

    refit_x = feature_matrix(refit_rows)
    test_x = feature_matrix(test_rows)
    refit_gain = candidate_gain_target(refit_rows)
    refit_lambda = optimal_lambda_target(refit_rows)

    gate_model = fit_ridge(refit_x, refit_gain, gate_alpha)
    soft_model = fit_ridge(refit_x, refit_lambda, soft_alpha)

    test_target = as_array(test_rows, "target")
    test_pre = as_array(test_rows, MODEL_PRE_ONLY)
    test_candidate = as_array(test_rows, MODEL_TRANSFER_CANDIDATE)
    test_hard = as_array(test_rows, MODEL_HARD_ABSTENTION)

    gate_signal = gate_model.predict(test_x)
    gate_pred = np.where(gate_signal > 0.0, test_candidate, test_pre)
    soft_lambda = np.clip(soft_model.predict(test_x), 0.0, 1.0)
    soft_pred = test_pre + soft_lambda * (test_candidate - test_pre)
    optimal_lambda = optimal_lambda_target(test_rows)

    test_prediction_rows = []
    for row, gate_value, gate_choice, soft_weight, best_weight, gate_out, soft_out in zip(
        test_rows,
        gate_signal,
        (gate_signal > 0.0).astype(int),
        soft_lambda,
        optimal_lambda,
        gate_pred,
        soft_pred,
    ):
        out = dict(row)
        out[MODEL_GAIN_GATE] = float(gate_out)
        out[MODEL_SOFT_TRUST] = float(soft_out)
        out["predicted_candidate_gain"] = float(gate_value)
        out["gate_use_candidate"] = int(gate_choice)
        out["predicted_lambda"] = float(soft_weight)
        out["optimal_lambda"] = float(best_weight)
        test_prediction_rows.append(out)

    tracked_models = [
        MODEL_PRE_ONLY,
        MODEL_TRANSFER_CANDIDATE,
        MODEL_HARD_ABSTENTION,
        MODEL_GAIN_GATE,
        MODEL_SOFT_TRUST,
    ]
    test_preds = {
        MODEL_PRE_ONLY: test_pre,
        MODEL_TRANSFER_CANDIDATE: test_candidate,
        MODEL_HARD_ABSTENTION: test_hard,
        MODEL_GAIN_GATE: gate_pred,
        MODEL_SOFT_TRUST: soft_pred,
    }

    val_target = as_array(val_rows, "target")
    val_reference = {
        MODEL_PRE_ONLY: metrics(val_target, as_array(val_rows, MODEL_PRE_ONLY)),
        MODEL_TRANSFER_CANDIDATE: metrics(val_target, as_array(val_rows, MODEL_TRANSFER_CANDIDATE)),
        MODEL_HARD_ABSTENTION: metrics(val_target, as_array(val_rows, MODEL_HARD_ABSTENTION)),
    }
    test_overall = {model_name: metrics(test_target, pred) for model_name, pred in test_preds.items()}
    test_significance = {
        f"{MODEL_GAIN_GATE}__vs__{MODEL_PRE_ONLY}": summarize_significance(
            test_target, test_preds[MODEL_PRE_ONLY], test_preds[MODEL_GAIN_GATE], args.bootstrap_iters, args.perm_iters, args.seed
        ),
        f"{MODEL_GAIN_GATE}__vs__{MODEL_HARD_ABSTENTION}": summarize_significance(
            test_target, test_preds[MODEL_HARD_ABSTENTION], test_preds[MODEL_GAIN_GATE], args.bootstrap_iters, args.perm_iters, args.seed
        ),
        f"{MODEL_GAIN_GATE}__vs__{MODEL_TRANSFER_CANDIDATE}": summarize_significance(
            test_target, test_preds[MODEL_TRANSFER_CANDIDATE], test_preds[MODEL_GAIN_GATE], args.bootstrap_iters, args.perm_iters, args.seed
        ),
        f"{MODEL_SOFT_TRUST}__vs__{MODEL_PRE_ONLY}": summarize_significance(
            test_target, test_preds[MODEL_PRE_ONLY], test_preds[MODEL_SOFT_TRUST], args.bootstrap_iters, args.perm_iters, args.seed
        ),
        f"{MODEL_SOFT_TRUST}__vs__{MODEL_HARD_ABSTENTION}": summarize_significance(
            test_target, test_preds[MODEL_HARD_ABSTENTION], test_preds[MODEL_SOFT_TRUST], args.bootstrap_iters, args.perm_iters, args.seed
        ),
        f"{MODEL_SOFT_TRUST}__vs__{MODEL_TRANSFER_CANDIDATE}": summarize_significance(
            test_target, test_preds[MODEL_TRANSFER_CANDIDATE], test_preds[MODEL_SOFT_TRUST], args.bootstrap_iters, args.perm_iters, args.seed
        ),
    }

    summary = {
        "config": {
            "temporal_root": str(temporal_root),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "refit_train_splits": refit_train_splits,
            "test_split": args.test_split,
            "feature_names": FEATURE_NAMES,
            "alpha_grid": alpha_grid,
        },
        "split_sizes": {
            args.train_split: len(train_rows),
            args.val_split: len(val_rows),
            args.test_split: len(test_rows),
        },
        "tuning": gate_tuning + soft_tuning,
        "selected_alpha": {
            MODEL_GAIN_GATE: gate_alpha,
            MODEL_SOFT_TRUST: soft_alpha,
        },
        "validation_reference": val_reference,
        "test_overall": test_overall,
        "test_significance": test_significance,
        "test_behavior": {
            MODEL_GAIN_GATE: {
                "candidate_share": float(np.mean(gate_signal > 0.0)),
                "pred_signal_min": float(np.min(gate_signal)),
                "pred_signal_max": float(np.max(gate_signal)),
                "coef_rows": calibrator_coefficients(gate_model),
            },
            MODEL_SOFT_TRUST: {
                "lambda_mean": float(np.mean(soft_lambda)),
                "lambda_min": float(np.min(soft_lambda)),
                "lambda_max": float(np.max(soft_lambda)),
                "optimal_lambda_mean": float(np.mean(optimal_lambda)),
                "coef_rows": calibrator_coefficients(soft_model),
            },
        },
    }

    write_csv(output_dir / "afterhours_transfer_learnable_trust_calibrator_tuning.csv", gate_tuning + soft_tuning)
    write_csv(output_dir / "afterhours_transfer_learnable_trust_calibrator_test_predictions.csv", test_prediction_rows)
    write_csv(output_dir / "afterhours_transfer_learnable_trust_calibrator_gate_coefficients.csv", summary["test_behavior"][MODEL_GAIN_GATE]["coef_rows"])
    write_csv(output_dir / "afterhours_transfer_learnable_trust_calibrator_soft_coefficients.csv", summary["test_behavior"][MODEL_SOFT_TRUST]["coef_rows"])
    write_json(output_dir / "afterhours_transfer_learnable_trust_calibrator_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
