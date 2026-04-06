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

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json
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

ALIGNED_AUDIO_META_FIELDS = {"event_key", "ticker", "year", "quarter"}

FAMILY_FEATURES = {
    "geometry_only": [
        "agreed_choose_qa",
        "agreed_minus_pre_pred",
        "qa_minus_pre_pred",
        "sem_minus_pre_pred",
        "selected_minus_pre_pred",
        "pair_minus_pre_pred",
        "logistic_minus_pre_pred",
        "abs_pair_minus_logistic_pred",
    ],
    "lite_quality": [
        "a4_strict_row_share",
        "a4_strict_high_conf_share",
        "qa_pair_count",
        "qa_bench_direct_answer_share",
        "qa_bench_evasion_score_mean",
        "qa_bench_coverage_mean",
        "aligned_audio__aligned_audio_sentence_count",
    ],
    "hybrid_quality": [
        "a4_strict_row_share",
        "a4_strict_high_conf_share",
        "qa_pair_count",
        "qa_pair_low_overlap_share",
        "qa_pair_answer_hedge_rate_mean",
        "qa_pair_answer_assertive_rate_mean",
        "qa_pair_answer_forward_rate_mean",
        "qa_multi_part_question_share",
        "qa_evasive_proxy_share",
        "qa_bench_direct_answer_share",
        "qa_bench_direct_early_score_mean",
        "qa_bench_evasion_score_mean",
        "qa_bench_high_evasion_share",
        "qa_bench_coverage_mean",
        "qa_bench_nonresponse_share",
        "aligned_audio__aligned_audio_sentence_count",
    ],
}
FAMILY_FEATURES["geometry_plus_hybrid"] = FAMILY_FEATURES["geometry_only"] + [
    feature for feature in FAMILY_FEATURES["hybrid_quality"] if feature not in FAMILY_FEATURES["geometry_only"]
]

SPLITS = [
    "val2020_test_post2020",
    "val2021_test_post2021",
    "val2022_test_post2022",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compact upstream signal families for agreement-side transfer refinement."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv"),
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("results/features_real/event_text_audio_features.csv"),
    )
    parser.add_argument(
        "--qa-csv",
        type=Path,
        default=Path("results/qa_benchmark_features_v2_real/qa_benchmark_features.csv"),
    )
    parser.add_argument(
        "--aligned-audio-csv",
        type=Path,
        default=Path("results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real"),
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


def load_prefixed_lookup(path: Path, prefix: str, meta_fields: set[str]) -> dict[str, dict[str, str]]:
    lookup = {}
    for row in load_csv_rows(path.resolve()):
        event_key = row.get("event_key", "")
        if not event_key:
            continue
        item = {}
        for key, value in row.items():
            if key in meta_fields:
                continue
            item[f"{prefix}{key}"] = value
        lookup[event_key] = item
    return lookup


def build_temporal_rows(temporal_root: Path) -> list[dict[str, float | int | str]]:
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

            rows.append({
                "row_key": f"{split_name}::{key}",
                "event_key": key,
                "split": split_name,
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
                "agreed_minus_pre_pred": agreed_pred - pre,
                "qa_minus_pre_pred": qa - pre,
                "sem_minus_pre_pred": retained - pre,
                "selected_minus_pre_pred": selected - pre,
                "pair_minus_pre_pred": pair_tree - pre,
                "logistic_minus_pre_pred": logistic - pre,
                "abs_pair_minus_logistic_pred": abs(pair_tree - logistic),
            })
    return rows


def attach_side_features(
    rows: list[dict[str, float | int | str]],
    panel_csv: Path,
    features_csv: Path,
    qa_csv: Path,
    aligned_audio_csv: Path,
) -> dict[str, int]:
    panel_lookup = {row["event_key"]: row for row in load_csv_rows(panel_csv.resolve()) if row.get("event_key")}
    feature_lookup = {row["event_key"]: row for row in load_csv_rows(features_csv.resolve()) if row.get("event_key")}
    qa_lookup = {row["event_key"]: row for row in load_csv_rows(qa_csv.resolve()) if row.get("event_key")}
    aligned_lookup = load_prefixed_lookup(aligned_audio_csv.resolve(), "aligned_audio__", ALIGNED_AUDIO_META_FIELDS)

    coverage = {
        "temporal_rows": len(rows),
        "with_panel": 0,
        "with_features": 0,
        "with_qa": 0,
        "with_aligned_audio": 0,
        "with_all_side_inputs": 0,
    }

    feature_cols = [
        "a4_strict_high_conf_share",
        "qa_pair_count",
        "qa_pair_low_overlap_share",
        "qa_pair_answer_hedge_rate_mean",
        "qa_pair_answer_assertive_rate_mean",
        "qa_pair_answer_forward_rate_mean",
        "qa_multi_part_question_share",
        "qa_evasive_proxy_share",
    ]
    qa_cols = [
        "qa_bench_direct_answer_share",
        "qa_bench_direct_early_score_mean",
        "qa_bench_evasion_score_mean",
        "qa_bench_high_evasion_share",
        "qa_bench_coverage_mean",
        "qa_bench_nonresponse_share",
    ]

    for row in rows:
        key = str(row["event_key"])
        panel_row = panel_lookup.get(key)
        feature_row = feature_lookup.get(key)
        qa_row = qa_lookup.get(key)
        aligned_row = aligned_lookup.get(key)

        if panel_row is not None:
            coverage["with_panel"] += 1
        if feature_row is not None:
            coverage["with_features"] += 1
        if qa_row is not None:
            coverage["with_qa"] += 1
        if aligned_row is not None:
            coverage["with_aligned_audio"] += 1
        if panel_row is not None and feature_row is not None and qa_row is not None and aligned_row is not None:
            coverage["with_all_side_inputs"] += 1

        row["a4_strict_row_share"] = safe_float(panel_row.get("a4_strict_row_share") if panel_row else None) or 0.0
        for col in feature_cols:
            row[col] = safe_float(feature_row.get(col) if feature_row else None) or 0.0
        for col in qa_cols:
            row[col] = safe_float(qa_row.get(col) if qa_row else None) or 0.0
        row["aligned_audio__aligned_audio_sentence_count"] = (
            safe_float(aligned_row.get("aligned_audio__aligned_audio_sentence_count") if aligned_row else None) or 0.0
        )

    return coverage


def as_array(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def feature_matrix(rows: list[dict[str, float | int | str]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(row.get(name, 0.0)) for name in feature_names] for row in rows], dtype=float)


def agreement_gain_target(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    target = as_array(rows, "target")
    pre = as_array(rows, MODEL_PRE_ONLY)
    agreed = as_array(rows, MODEL_AGREED)
    return (target - pre) ** 2 - (target - agreed) ** 2


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(train_x, train_y)
    return model


def top_coefficients(model: Pipeline, feature_names: list[str], limit: int = 8) -> list[dict[str, float | str]]:
    ridge = model.named_steps["ridge"]
    return [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in sorted(zip(feature_names, ridge.coef_.tolist()), key=lambda item: abs(item[1]), reverse=True)[:limit]
    ]


def main() -> None:
    args = parse_args()
    temporal_root = args.temporal_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_temporal_rows(temporal_root)
    coverage = attach_side_features(
        rows,
        args.panel_csv.resolve(),
        args.features_csv.resolve(),
        args.qa_csv.resolve(),
        args.aligned_audio_csv.resolve(),
    )
    alpha_grid = [float(item) for item in parse_list(args.alphas)]
    refit_train_splits = parse_list(args.refit_train_splits)

    train_rows = [row for row in rows if row["split"] == args.train_split and int(row["agreement"]) == 1]
    val_rows = [row for row in rows if row["split"] == args.val_split and int(row["agreement"]) == 1]
    refit_rows = [row for row in rows if row["split"] in refit_train_splits and int(row["agreement"]) == 1]
    test_agreement_rows = [row for row in rows if row["split"] == args.test_split and int(row["agreement"]) == 1]
    test_full_rows = [row for row in rows if row["split"] == args.test_split]
    if not train_rows or not val_rows or not refit_rows or not test_agreement_rows or not test_full_rows:
        raise SystemExit("missing train/val/refit/test rows for agreement signal benchmark")

    val_target = as_array(val_rows, "target")
    test_full_target = as_array(test_full_rows, "target")
    test_agreement_target = as_array(test_agreement_rows, "target")
    test_full_pre = as_array(test_full_rows, MODEL_PRE_ONLY)
    test_full_hard = as_array(test_full_rows, MODEL_HARD_ABSTENTION)
    test_full_agreed = as_array(test_full_rows, MODEL_AGREED)

    family_rows = []
    family_predictions = {
        MODEL_PRE_ONLY: test_full_pre,
        MODEL_AGREED: test_full_agreed,
        MODEL_HARD_ABSTENTION: test_full_hard,
    }
    prediction_rows = [dict(row) for row in test_full_rows]
    tuning_rows = []
    best_family_name = None
    best_family_r2 = None
    family_summaries = {}

    for family_name, feature_names in FAMILY_FEATURES.items():
        train_x = feature_matrix(train_rows, feature_names)
        val_x = feature_matrix(val_rows, feature_names)
        refit_x = feature_matrix(refit_rows, feature_names)
        test_agreement_x = feature_matrix(test_agreement_rows, feature_names)

        train_gain = agreement_gain_target(train_rows)
        best_alpha = None
        best_r2 = None
        best_payload = None

        for alpha in alpha_grid:
            model = fit_ridge(train_x, train_gain, alpha)
            val_signal = model.predict(val_x)
            val_pred = np.where(val_signal > 0.0, as_array(val_rows, MODEL_AGREED), as_array(val_rows, MODEL_PRE_ONLY))
            score = metrics(val_target, val_pred)
            payload = {
                "family": family_name,
                "alpha": float(alpha),
                "val_r2": score["r2"],
                "val_rmse": score["rmse"],
                "val_mae": score["mae"],
                "val_use_agreed_share": float(np.mean(val_signal > 0.0)),
                "val_signal_min": float(np.min(val_signal)),
                "val_signal_max": float(np.max(val_signal)),
            }
            tuning_rows.append(payload)
            if best_r2 is None or score["r2"] > best_r2:
                best_r2 = score["r2"]
                best_alpha = float(alpha)
                best_payload = payload

        assert best_payload is not None
        model = fit_ridge(refit_x, agreement_gain_target(refit_rows), best_alpha)
        test_signal = model.predict(test_agreement_x)
        test_agreement_pred = np.where(
            test_signal > 0.0,
            as_array(test_agreement_rows, MODEL_AGREED),
            as_array(test_agreement_rows, MODEL_PRE_ONLY),
        )
        pred_lookup = {row["event_key"]: float(pred) for row, pred in zip(test_agreement_rows, test_agreement_pred)}
        signal_lookup = {row["event_key"]: float(signal) for row, signal in zip(test_agreement_rows, test_signal)}
        use_lookup = {row["event_key"]: int(signal > 0.0) for row, signal in zip(test_agreement_rows, test_signal)}
        full_pred = np.asarray([pred_lookup.get(row["event_key"], float(row[MODEL_PRE_ONLY])) for row in test_full_rows], dtype=float)
        family_predictions[family_name] = full_pred

        for row in prediction_rows:
            event_key = row["event_key"]
            row[family_name] = pred_lookup.get(event_key, float(row[MODEL_PRE_ONLY]))
            row[f"{family_name}__predicted_gain_signal"] = signal_lookup.get(event_key, 0.0)
            row[f"{family_name}__use_agreed"] = use_lookup.get(event_key, 0)

        test_agreement_metrics = metrics(test_agreement_target, test_agreement_pred)
        test_full_metrics = metrics(test_full_target, full_pred)
        sig_vs_hard = summarize_significance(
            test_full_target, test_full_hard, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
        )
        sig_vs_pre = summarize_significance(
            test_full_target, test_full_pre, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
        )
        family_row = {
            "family": family_name,
            "feature_count": len(feature_names),
            "features": "|".join(feature_names),
            "selected_alpha": best_alpha,
            "val_r2": best_payload["val_r2"],
            "val_use_agreed_share": best_payload["val_use_agreed_share"],
            "test_agreement_r2": test_agreement_metrics["r2"],
            "test_agreement_rmse": test_agreement_metrics["rmse"],
            "test_full_r2": test_full_metrics["r2"],
            "test_full_rmse": test_full_metrics["rmse"],
            "test_full_mae": test_full_metrics["mae"],
            "test_full_p_mse_vs_hard": sig_vs_hard["mse_gain_pvalue"],
            "test_full_p_mse_vs_pre": sig_vs_pre["mse_gain_pvalue"],
            "test_use_agreed_share": float(np.mean(test_signal > 0.0)),
        }
        family_rows.append(family_row)
        family_summaries[family_name] = {
            "feature_names": feature_names,
            "selected_alpha": best_alpha,
            "best_validation": best_payload,
            "test_agreement_metrics": test_agreement_metrics,
            "test_full_metrics": test_full_metrics,
            "significance_vs_hard": sig_vs_hard,
            "significance_vs_pre": sig_vs_pre,
            "coef_rows": top_coefficients(model, feature_names),
        }
        if best_family_r2 is None or test_full_metrics["r2"] > best_family_r2:
            best_family_r2 = test_full_metrics["r2"]
            best_family_name = family_name

    best_vs_geometry = None
    if best_family_name and best_family_name != "geometry_only":
        best_vs_geometry = summarize_significance(
            test_full_target,
            family_predictions["geometry_only"],
            family_predictions[best_family_name],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        )

    summary = {
        "config": {
            "temporal_root": str(temporal_root),
            "panel_csv": str(args.panel_csv.resolve()),
            "features_csv": str(args.features_csv.resolve()),
            "qa_csv": str(args.qa_csv.resolve()),
            "aligned_audio_csv": str(args.aligned_audio_csv.resolve()),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "refit_train_splits": refit_train_splits,
            "test_split": args.test_split,
            "alpha_grid": alpha_grid,
        },
        "coverage": coverage,
        "split_sizes": {
            "train_agreement_size": len(train_rows),
            "val_agreement_size": len(val_rows),
            "test_agreement_size": len(test_agreement_rows),
            "test_full_size": len(test_full_rows),
        },
        "reference": {
            "validation_pre_r2": metrics(val_target, as_array(val_rows, MODEL_PRE_ONLY))["r2"],
            "validation_agreed_r2": metrics(val_target, as_array(val_rows, MODEL_AGREED))["r2"],
            "test_full_pre": metrics(test_full_target, test_full_pre),
            "test_full_agreed": metrics(test_full_target, test_full_agreed),
            "test_full_hard_abstention": metrics(test_full_target, test_full_hard),
        },
        "tuning": tuning_rows,
        "families": family_rows,
        "best_family": best_family_name,
        "best_family_summary": family_summaries.get(best_family_name),
        "best_family_vs_geometry_only": best_vs_geometry,
        "family_summaries": family_summaries,
    }

    write_csv(output_dir / "afterhours_transfer_agreement_signal_benchmark_overview.csv", family_rows)
    write_csv(output_dir / "afterhours_transfer_agreement_signal_benchmark_tuning.csv", tuning_rows)
    write_csv(output_dir / "afterhours_transfer_agreement_signal_benchmark_test_predictions.csv", prediction_rows)
    write_json(output_dir / "afterhours_transfer_agreement_signal_benchmark_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
