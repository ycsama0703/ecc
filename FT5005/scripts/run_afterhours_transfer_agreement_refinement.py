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
MODEL_AGREED = "agreement_supported_pred"
MODEL_HARD_ABSTENTION = "agreement_pre_only_abstention"
MODEL_AGREEMENT_GAIN_GATE = "compact_learnable_agreement_gain_gate"
MODEL_AGREEMENT_SOFT_TRUST = "compact_learnable_agreement_soft_trust"

FEATURE_NAMES = [
    "agreed_choose_qa",
    "agreed_minus_pre_pred",
    "qa_minus_pre_pred",
    "sem_minus_pre_pred",
    "selected_minus_pre_pred",
    "pair_minus_pre_pred",
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
        description="Refine only the agreement side of the transfer abstention path with compact learnable calibrators."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_agreement_refinement_role_aware_audio_lsa4_real"),
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
                MODEL_AGREED: agreed_pred,
                MODEL_HARD_ABSTENTION: agreed_pred if agreement == 1 else pre,
                "agreement": agreement,
                "agreed_choose_qa": int(tree_choose_qa == 1),
                "tree_choose_qa": tree_choose_qa,
                "logistic_choose_qa": logistic_choose_qa,
                "agreed_minus_pre_pred": agreed_pred - pre,
                "qa_minus_pre_pred": qa - pre,
                "sem_minus_pre_pred": retained - pre,
                "selected_minus_pre_pred": selected - pre,
                "pair_minus_pre_pred": pair_tree - pre,
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


def agreement_gain_target(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    target = as_array(rows, "target")
    pre = as_array(rows, MODEL_PRE_ONLY)
    agreed = as_array(rows, MODEL_AGREED)
    return (target - pre) ** 2 - (target - agreed) ** 2


def optimal_lambda_target(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    target = as_array(rows, "target")
    pre = as_array(rows, MODEL_PRE_ONLY)
    agreed = as_array(rows, MODEL_AGREED)
    delta = agreed - pre
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


def main() -> None:
    args = parse_args()
    temporal_root = args.temporal_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(temporal_root)
    alpha_grid = [float(item) for item in parse_list(args.alphas)]
    refit_train_splits = parse_list(args.refit_train_splits)

    train_rows = [row for row in rows if row["split"] == args.train_split and int(row["agreement"]) == 1]
    val_rows = [row for row in rows if row["split"] == args.val_split and int(row["agreement"]) == 1]
    refit_rows = [row for row in rows if row["split"] in refit_train_splits and int(row["agreement"]) == 1]
    test_agreement_rows = [row for row in rows if row["split"] == args.test_split and int(row["agreement"]) == 1]
    test_full_rows = [row for row in rows if row["split"] == args.test_split]
    if not train_rows or not val_rows or not refit_rows or not test_agreement_rows or not test_full_rows:
        raise SystemExit("missing train/val/refit/test rows for agreement refinement benchmark")

    train_x = feature_matrix(train_rows)
    val_x = feature_matrix(val_rows)
    refit_x = feature_matrix(refit_rows)
    test_agreement_x = feature_matrix(test_agreement_rows)

    train_gain = agreement_gain_target(train_rows)
    refit_gain = agreement_gain_target(refit_rows)
    train_lambda = optimal_lambda_target(train_rows)
    refit_lambda = optimal_lambda_target(refit_rows)

    val_target = as_array(val_rows, "target")
    val_pre = as_array(val_rows, MODEL_PRE_ONLY)
    val_agreed = as_array(val_rows, MODEL_AGREED)

    gain_tuning = []
    soft_tuning = []
    best_gain_alpha = None
    best_gain_r2 = None
    best_soft_alpha = None
    best_soft_r2 = None

    for alpha in alpha_grid:
        gain_model = fit_ridge(train_x, train_gain, alpha)
        gain_signal = gain_model.predict(val_x)
        gain_pred = np.where(gain_signal > 0.0, val_agreed, val_pre)
        gain_score = metrics(val_target, gain_pred)
        gain_tuning.append(
            {
                "calibrator": MODEL_AGREEMENT_GAIN_GATE,
                "alpha": float(alpha),
                "val_r2": gain_score["r2"],
                "val_rmse": gain_score["rmse"],
                "val_mae": gain_score["mae"],
                "val_agreement_use_share": float(np.mean(gain_signal > 0.0)),
                "val_pred_signal_min": float(np.min(gain_signal)),
                "val_pred_signal_max": float(np.max(gain_signal)),
            }
        )
        if best_gain_r2 is None or gain_score["r2"] > best_gain_r2:
            best_gain_r2 = gain_score["r2"]
            best_gain_alpha = float(alpha)

        soft_model = fit_ridge(train_x, train_lambda, alpha)
        soft_lambda = np.clip(soft_model.predict(val_x), 0.0, 1.0)
        soft_pred = val_pre + soft_lambda * (val_agreed - val_pre)
        soft_score = metrics(val_target, soft_pred)
        soft_tuning.append(
            {
                "calibrator": MODEL_AGREEMENT_SOFT_TRUST,
                "alpha": float(alpha),
                "val_r2": soft_score["r2"],
                "val_rmse": soft_score["rmse"],
                "val_mae": soft_score["mae"],
                "val_lambda_mean": float(np.mean(soft_lambda)),
                "val_lambda_min": float(np.min(soft_lambda)),
                "val_lambda_max": float(np.max(soft_lambda)),
            }
        )
        if best_soft_r2 is None or soft_score["r2"] > best_soft_r2:
            best_soft_r2 = soft_score["r2"]
            best_soft_alpha = float(alpha)

    gain_model = fit_ridge(refit_x, refit_gain, best_gain_alpha)
    soft_model = fit_ridge(refit_x, refit_lambda, best_soft_alpha)

    gain_signal = gain_model.predict(test_agreement_x)
    agreement_gain_pred = np.where(gain_signal > 0.0, as_array(test_agreement_rows, MODEL_AGREED), as_array(test_agreement_rows, MODEL_PRE_ONLY))
    soft_lambda = np.clip(soft_model.predict(test_agreement_x), 0.0, 1.0)
    agreement_soft_pred = as_array(test_agreement_rows, MODEL_PRE_ONLY) + soft_lambda * (
        as_array(test_agreement_rows, MODEL_AGREED) - as_array(test_agreement_rows, MODEL_PRE_ONLY)
    )
    optimal_lambda = optimal_lambda_target(test_agreement_rows)

    agreement_prediction_lookup = {}
    for row, signal, gate_pred, lam, soft_pred, opt_lam in zip(
        test_agreement_rows,
        gain_signal,
        agreement_gain_pred,
        soft_lambda,
        agreement_soft_pred,
        optimal_lambda,
    ):
        agreement_prediction_lookup[row["event_key"]] = {
            MODEL_AGREEMENT_GAIN_GATE: float(gate_pred),
            MODEL_AGREEMENT_SOFT_TRUST: float(soft_pred),
            "predicted_gain_signal": float(signal),
            "gain_gate_use_agreed": int(signal > 0.0),
            "predicted_lambda": float(lam),
            "optimal_lambda": float(opt_lam),
        }

    test_prediction_rows = []
    full_preds = {
        MODEL_PRE_ONLY: [],
        MODEL_AGREED: [],
        MODEL_HARD_ABSTENTION: [],
        MODEL_AGREEMENT_GAIN_GATE: [],
        MODEL_AGREEMENT_SOFT_TRUST: [],
    }
    full_target = []

    for row in test_full_rows:
        record = dict(row)
        extra = agreement_prediction_lookup.get(
            row["event_key"],
            {
                MODEL_AGREEMENT_GAIN_GATE: float(row[MODEL_PRE_ONLY]),
                MODEL_AGREEMENT_SOFT_TRUST: float(row[MODEL_PRE_ONLY]),
                "predicted_gain_signal": 0.0,
                "gain_gate_use_agreed": 0,
                "predicted_lambda": 0.0,
                "optimal_lambda": 0.0,
            },
        )
        record.update(extra)
        test_prediction_rows.append(record)
        full_target.append(float(row["target"]))
        for model_name in full_preds:
            full_preds[model_name].append(float(record[model_name]))

    test_target = np.asarray(full_target, dtype=float)
    full_preds_np = {model_name: np.asarray(values, dtype=float) for model_name, values in full_preds.items()}

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
            "train_agreement_size": len(train_rows),
            "val_agreement_size": len(val_rows),
            "test_agreement_size": len(test_agreement_rows),
            "test_full_size": len(test_full_rows),
        },
        "tuning": gain_tuning + soft_tuning,
        "selected_alpha": {
            MODEL_AGREEMENT_GAIN_GATE: best_gain_alpha,
            MODEL_AGREEMENT_SOFT_TRUST: best_soft_alpha,
        },
        "validation_reference": {
            MODEL_PRE_ONLY: metrics(val_target, val_pre),
            MODEL_AGREED: metrics(val_target, val_agreed),
        },
        "test_agreement_subset": {
            MODEL_PRE_ONLY: metrics(as_array(test_agreement_rows, "target"), as_array(test_agreement_rows, MODEL_PRE_ONLY)),
            MODEL_AGREED: metrics(as_array(test_agreement_rows, "target"), as_array(test_agreement_rows, MODEL_AGREED)),
            MODEL_AGREEMENT_GAIN_GATE: metrics(as_array(test_agreement_rows, "target"), agreement_gain_pred),
            MODEL_AGREEMENT_SOFT_TRUST: metrics(as_array(test_agreement_rows, "target"), agreement_soft_pred),
        },
        "test_full_overall": {model_name: metrics(test_target, pred) for model_name, pred in full_preds_np.items()},
        "test_full_significance": {
            f"{MODEL_AGREEMENT_GAIN_GATE}__vs__{MODEL_HARD_ABSTENTION}": summarize_significance(
                test_target, full_preds_np[MODEL_HARD_ABSTENTION], full_preds_np[MODEL_AGREEMENT_GAIN_GATE], args.bootstrap_iters, args.perm_iters, args.seed
            ),
            f"{MODEL_AGREEMENT_GAIN_GATE}__vs__{MODEL_PRE_ONLY}": summarize_significance(
                test_target, full_preds_np[MODEL_PRE_ONLY], full_preds_np[MODEL_AGREEMENT_GAIN_GATE], args.bootstrap_iters, args.perm_iters, args.seed
            ),
            f"{MODEL_AGREEMENT_SOFT_TRUST}__vs__{MODEL_HARD_ABSTENTION}": summarize_significance(
                test_target, full_preds_np[MODEL_HARD_ABSTENTION], full_preds_np[MODEL_AGREEMENT_SOFT_TRUST], args.bootstrap_iters, args.perm_iters, args.seed
            ),
            f"{MODEL_AGREEMENT_SOFT_TRUST}__vs__{MODEL_PRE_ONLY}": summarize_significance(
                test_target, full_preds_np[MODEL_PRE_ONLY], full_preds_np[MODEL_AGREEMENT_SOFT_TRUST], args.bootstrap_iters, args.perm_iters, args.seed
            ),
        },
        "test_behavior": {
            MODEL_AGREEMENT_GAIN_GATE: {
                "agreement_use_share": float(np.mean(gain_signal > 0.0)),
                "pred_signal_min": float(np.min(gain_signal)),
                "pred_signal_max": float(np.max(gain_signal)),
                "coef_rows": calibrator_coefficients(gain_model),
            },
            MODEL_AGREEMENT_SOFT_TRUST: {
                "lambda_mean": float(np.mean(soft_lambda)),
                "lambda_min": float(np.min(soft_lambda)),
                "lambda_max": float(np.max(soft_lambda)),
                "optimal_lambda_mean": float(np.mean(optimal_lambda)),
                "coef_rows": calibrator_coefficients(soft_model),
            },
        },
    }

    write_csv(output_dir / "afterhours_transfer_agreement_refinement_tuning.csv", gain_tuning + soft_tuning)
    write_csv(output_dir / "afterhours_transfer_agreement_refinement_test_predictions.csv", test_prediction_rows)
    write_csv(output_dir / "afterhours_transfer_agreement_refinement_gain_coefficients.csv", summary["test_behavior"][MODEL_AGREEMENT_GAIN_GATE]["coef_rows"])
    write_csv(output_dir / "afterhours_transfer_agreement_refinement_soft_coefficients.csv", summary["test_behavior"][MODEL_AGREEMENT_SOFT_TRUST]["coef_rows"])
    write_json(output_dir / "afterhours_transfer_agreement_refinement_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
