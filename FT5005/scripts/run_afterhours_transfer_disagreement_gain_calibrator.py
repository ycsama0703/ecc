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
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from dj30_qc_utils import write_csv, write_json
from run_structured_baselines import metrics


MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_QA_EXPERT = "residual_pre_call_market_plus_a4_plus_qa_benchmark_svd_observability_gate"
MODEL_RETAINED = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate"
MODEL_GAIN_CALIBRATOR = "compact_learnable_disagreement_gain_calibrator"

FEATURE_NAMES = [
    "qa_minus_pre_pred",
    "sem_minus_pre_pred",
    "sem_minus_qa_pred",
    "abs_sem_minus_qa_pred",
    "tree_choose_qa",
    "logistic_choose_qa",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a compact learnable disagreement gain calibrator to test whether the latest QA pocket is actually transferable."
    )
    parser.add_argument(
        "--event-rows-csv",
        type=Path,
        default=Path(
            "results/afterhours_transfer_disagreement_slice_diagnostics_role_aware_audio_lsa4_real/afterhours_transfer_disagreement_slice_event_rows.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_disagreement_gain_calibrator_role_aware_audio_lsa4_real"),
    )
    parser.add_argument("--val-split", default="val2021_test_post2021")
    parser.add_argument("--test-split", default="val2022_test_post2022")
    parser.add_argument("--train-splits", default="val2020_test_post2020")
    parser.add_argument("--refit-train-splits", default="val2020_test_post2020,val2021_test_post2021")
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    return parser.parse_args()


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        qa_pred = float(row[MODEL_QA_EXPERT])
        pre_pred = float(row[MODEL_PRE_ONLY])
        sem_pred = float(row[MODEL_RETAINED])
        row["qa_minus_pre_pred"] = qa_pred - pre_pred
        row["sem_minus_pre_pred"] = sem_pred - pre_pred
        row["sem_minus_qa_pred"] = sem_pred - qa_pred
        row["abs_sem_minus_qa_pred"] = abs(sem_pred - qa_pred)
    return rows


def arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=float)
    gain = np.asarray([float(row["qa_vs_pre_mse_gain"]) for row in rows], dtype=float)
    target = np.asarray([float(row["target"]) for row in rows], dtype=float)
    pre = np.asarray([float(row[MODEL_PRE_ONLY]) for row in rows], dtype=float)
    qa = np.asarray([float(row[MODEL_QA_EXPERT]) for row in rows], dtype=float)
    return x, gain, target, pre, qa


def fit_gain_model(train_x: np.ndarray, train_gain: np.ndarray, alpha: float) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(train_x, train_gain)
    return model


def routed_predictions(pred_gain: np.ndarray, qa_pred: np.ndarray, pre_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    use_qa = np.asarray(pred_gain > 0.0, dtype=int)
    preds = np.where(use_qa == 1, qa_pred, pre_pred)
    return np.asarray(preds, dtype=float), use_qa


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(args.event_rows_csv.resolve())
    train_splits = parse_list(args.train_splits)
    refit_train_splits = parse_list(args.refit_train_splits)
    alpha_grid = [float(item) for item in parse_list(args.alphas)]

    train_rows = [row for row in rows if row["split"] in train_splits]
    val_rows = [row for row in rows if row["split"] == args.val_split]
    refit_rows = [row for row in rows if row["split"] in refit_train_splits]
    test_rows = [row for row in rows if row["split"] == args.test_split]

    if not train_rows or not val_rows or not test_rows:
        raise SystemExit("missing train/val/test rows for calibrator benchmark")

    train_x, train_gain, _, _, _ = arrays(train_rows)
    val_x, _, val_target, val_pre, val_qa = arrays(val_rows)
    refit_x, refit_gain, _, _, _ = arrays(refit_rows)
    test_x, _, test_target, test_pre, test_qa = arrays(test_rows)

    tuning_rows = []
    best_alpha = None
    best_r2 = None

    for alpha in alpha_grid:
        model = fit_gain_model(train_x, train_gain, alpha)
        val_pred_gain = model.predict(val_x)
        val_pred, val_use_qa = routed_predictions(val_pred_gain, val_qa, val_pre)
        score = metrics(val_target, val_pred)
        tuning_row = {
            "alpha": float(alpha),
            "val_r2": score["r2"],
            "val_rmse": score["rmse"],
            "val_mae": score["mae"],
            "val_qa_share": float(np.mean(val_use_qa)),
            "val_pred_gain_min": float(np.min(val_pred_gain)),
            "val_pred_gain_max": float(np.max(val_pred_gain)),
        }
        tuning_rows.append(tuning_row)
        if best_r2 is None or tuning_row["val_r2"] > best_r2:
            best_r2 = tuning_row["val_r2"]
            best_alpha = float(alpha)

    final_model = fit_gain_model(refit_x, refit_gain, best_alpha)
    test_pred_gain = final_model.predict(test_x)
    test_pred, test_use_qa = routed_predictions(test_pred_gain, test_qa, test_pre)

    ridge = final_model.named_steps["ridge"]
    coef_rows = [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in sorted(zip(FEATURE_NAMES, ridge.coef_.tolist()), key=lambda item: abs(item[1]), reverse=True)
    ]

    test_prediction_rows = []
    for row, pred_gain, use_qa, pred in zip(test_rows, test_pred_gain, test_use_qa, test_pred):
        out = dict(row)
        out[MODEL_GAIN_CALIBRATOR] = float(pred)
        out["predicted_gain"] = float(pred_gain)
        out["calibrator_use_qa"] = int(use_qa)
        test_prediction_rows.append(out)

    summary = {
        "config": {
            "event_rows_csv": str(args.event_rows_csv.resolve()),
            "train_splits": train_splits,
            "val_split": args.val_split,
            "refit_train_splits": refit_train_splits,
            "test_split": args.test_split,
            "feature_names": FEATURE_NAMES,
            "alpha_grid": alpha_grid,
        },
        "label_balance": {
            "train": {
                "size": len(train_rows),
                "positive_gain_count": int(sum(float(row["qa_vs_pre_mse_gain"]) > 0 for row in train_rows)),
                "negative_gain_count": int(sum(float(row["qa_vs_pre_mse_gain"]) < 0 for row in train_rows)),
                "tie_count": int(sum(float(row["qa_vs_pre_mse_gain"]) == 0 for row in train_rows)),
            },
            "val": {
                "size": len(val_rows),
                "positive_gain_count": int(sum(float(row["qa_vs_pre_mse_gain"]) > 0 for row in val_rows)),
                "negative_gain_count": int(sum(float(row["qa_vs_pre_mse_gain"]) < 0 for row in val_rows)),
                "tie_count": int(sum(float(row["qa_vs_pre_mse_gain"]) == 0 for row in val_rows)),
            },
            "test": {
                "size": len(test_rows),
                "positive_gain_count": int(sum(float(row["qa_vs_pre_mse_gain"]) > 0 for row in test_rows)),
                "negative_gain_count": int(sum(float(row["qa_vs_pre_mse_gain"]) < 0 for row in test_rows)),
                "tie_count": int(sum(float(row["qa_vs_pre_mse_gain"]) == 0 for row in test_rows)),
            },
        },
        "tuning": tuning_rows,
        "selected_alpha": float(best_alpha),
        "validation_reference": {
            MODEL_PRE_ONLY: metrics(val_target, val_pre),
            MODEL_QA_EXPERT: metrics(val_target, val_qa),
        },
        "test_overall": {
            MODEL_PRE_ONLY: metrics(test_target, test_pre),
            MODEL_QA_EXPERT: metrics(test_target, test_qa),
            MODEL_GAIN_CALIBRATOR: metrics(test_target, test_pred),
        },
        "test_calibrator": {
            "qa_share": float(np.mean(test_use_qa)),
            "predicted_gain_min": float(np.min(test_pred_gain)),
            "predicted_gain_max": float(np.max(test_pred_gain)),
            "coef_rows": coef_rows,
        },
    }

    write_csv(output_dir / "afterhours_transfer_disagreement_gain_calibrator_tuning.csv", tuning_rows)
    write_csv(output_dir / "afterhours_transfer_disagreement_gain_calibrator_test_predictions.csv", test_prediction_rows)
    write_csv(output_dir / "afterhours_transfer_disagreement_gain_calibrator_coefficients.csv", coef_rows)
    write_json(output_dir / "afterhours_transfer_disagreement_gain_calibrator_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
