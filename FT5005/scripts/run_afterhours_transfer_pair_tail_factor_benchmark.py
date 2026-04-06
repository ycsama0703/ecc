#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
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
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    if user_site_removed and isinstance(user_site, str) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json
from run_afterhours_transfer_agreement_signal_benchmark import build_temporal_rows, parse_list
from run_afterhours_transfer_expert_selection import summarize_significance
from run_structured_baselines import metrics

MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_AGREED = "agreement_supported_pred"
MODEL_HARD_ABSTENTION = "agreement_pre_only_abstention"

GEOMETRY_FEATURES = [
    "agreed_choose_qa",
    "agreed_minus_pre_pred",
    "qa_minus_pre_pred",
    "sem_minus_pre_pred",
    "selected_minus_pre_pred",
    "pair_minus_pre_pred",
    "logistic_minus_pre_pred",
    "abs_pair_minus_logistic_pred",
]

FACTOR_FAMILIES = {
    "pair_tail_core": [
        "qa_tail_max_evasion",
        "qa_tail_top2_evasion_mean",
        "qa_tail_min_direct_early",
        "qa_tail_bottom2_direct_early_mean",
        "qa_tail_min_coverage",
        "qa_tail_bottom2_coverage_mean",
        "qa_tail_high_evasion_share",
        "qa_tail_low_overlap_share",
        "qa_tail_severity_max",
        "qa_tail_top2_severity_mean",
    ],
    "pair_tail_dispersion": [
        "qa_tail_max_evasion",
        "qa_tail_top2_evasion_mean",
        "qa_tail_min_direct_early",
        "qa_tail_bottom2_direct_early_mean",
        "qa_tail_min_coverage",
        "qa_tail_bottom2_coverage_mean",
        "qa_tail_high_evasion_share",
        "qa_tail_low_overlap_share",
        "qa_tail_severity_max",
        "qa_tail_top2_severity_mean",
        "qa_tail_evasion_std",
        "qa_tail_direct_early_std",
        "qa_tail_coverage_std",
        "qa_tail_severity_std",
        "qa_tail_nonresponse_share",
        "qa_tail_short_evasive_share",
        "qa_tail_numeric_mismatch_share",
    ],
    "pair_tail_with_observability": [
        "qa_tail_max_evasion",
        "qa_tail_top2_evasion_mean",
        "qa_tail_min_direct_early",
        "qa_tail_bottom2_direct_early_mean",
        "qa_tail_min_coverage",
        "qa_tail_bottom2_coverage_mean",
        "qa_tail_high_evasion_share",
        "qa_tail_low_overlap_share",
        "qa_tail_severity_max",
        "qa_tail_top2_severity_mean",
        "a4_strict_row_share",
        "a4_strict_high_conf_share",
        "a4_median_match_score",
    ],
}

FACTOR_METHODS = ("pca1", "pls1")
VARIANTS = ("factor_only", "geometry_plus_factor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pair-tail answerability factors as compact agreement-side transfer refinements."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--tail-features-csv",
        type=Path,
        default=Path("results/qa_pair_tail_features_real/qa_pair_tail_features.csv"),
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
        "--reference-predictions-csv",
        type=Path,
        default=Path(
            "results/afterhours_transfer_answerability_factor_benchmark_lsa4_real/"
            "afterhours_transfer_answerability_factor_benchmark_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_pair_tail_factor_benchmark_lsa4_real"),
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


def attach_features(
    rows: list[dict[str, float | int | str]],
    tail_features_csv: Path,
    panel_csv: Path,
    features_csv: Path,
) -> dict[str, int]:
    tail_lookup = {row["event_key"]: row for row in load_csv_rows(tail_features_csv.resolve()) if row.get("event_key")}
    panel_lookup = {row["event_key"]: row for row in load_csv_rows(panel_csv.resolve()) if row.get("event_key")}
    feature_lookup = {row["event_key"]: row for row in load_csv_rows(features_csv.resolve()) if row.get("event_key")}

    tail_cols = sorted({col for cols in FACTOR_FAMILIES.values() for col in cols if col.startswith("qa_tail_")})
    panel_cols = ["a4_strict_row_share", "a4_median_match_score"]
    feature_cols = ["a4_strict_high_conf_share"]

    coverage = {
        "temporal_rows": len(rows),
        "with_tail_features": 0,
        "with_panel": 0,
        "with_features": 0,
        "with_all_inputs": 0,
    }

    for row in rows:
        key = str(row["event_key"])
        tail_row = tail_lookup.get(key)
        panel_row = panel_lookup.get(key)
        feature_row = feature_lookup.get(key)
        if tail_row is not None:
            coverage["with_tail_features"] += 1
        if panel_row is not None:
            coverage["with_panel"] += 1
        if feature_row is not None:
            coverage["with_features"] += 1
        if tail_row is not None and panel_row is not None and feature_row is not None:
            coverage["with_all_inputs"] += 1

        for col in tail_cols:
            row[col] = safe_float(tail_row.get(col) if tail_row else None) or 0.0
        for col in panel_cols:
            row[col] = safe_float(panel_row.get(col) if panel_row else None) or 0.0
        for col in feature_cols:
            row[col] = safe_float(feature_row.get(col) if feature_row else None) or 0.0
    return coverage


def as_array(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def agreement_gain_target(rows: list[dict[str, float | int | str]]) -> np.ndarray:
    target = as_array(rows, "target")
    pre = as_array(rows, MODEL_PRE_ONLY)
    agreed = as_array(rows, MODEL_AGREED)
    return (target - pre) ** 2 - (target - agreed) ** 2


def feature_matrix(rows: list[dict[str, float | int | str]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=float)


def fit_ridge(train_x: np.ndarray, train_y: np.ndarray, alpha: float) -> Pipeline:
    model = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])
    model.fit(train_x, train_y)
    return model


def build_factor_scores(method: str, feature_names: list[str], fit_rows, apply_rows):
    train_y = agreement_gain_target(fit_rows)
    train_x = feature_matrix(fit_rows, feature_names)
    apply_x = feature_matrix(apply_rows, feature_names)

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    apply_x_scaled = scaler.transform(apply_x)

    if method == "pca1":
        model = PCA(n_components=1, random_state=42)
        model.fit(train_x_scaled)
        train_factor = model.transform(train_x_scaled)[:, 0]
        apply_factor = model.transform(apply_x_scaled)[:, 0]
        weights = model.components_[0]
        meta = {"component_explained_variance_ratio": float(model.explained_variance_ratio_[0])}
    elif method == "pls1":
        model = PLSRegression(n_components=1, scale=False)
        model.fit(train_x_scaled, train_y)
        train_factor = model.transform(train_x_scaled)[:, 0]
        apply_factor = model.transform(apply_x_scaled)[:, 0]
        weights = model.x_weights_[:, 0]
        meta = {"x_score_variance": float(np.var(train_factor))}
    else:
        raise ValueError(method)

    corr = 0.0
    if np.std(train_factor) > 0 and np.std(train_y) > 0:
        corr = float(np.corrcoef(train_factor, train_y)[0, 1])
    if corr < 0.0:
        train_factor = -train_factor
        apply_factor = -apply_factor
        weights = -weights
        corr = -corr
    meta.update(
        {
            "train_gain_correlation": corr,
            "train_factor_mean": float(np.mean(train_factor)),
            "train_factor_std": float(np.std(train_factor)),
        }
    )
    loading_rows = [
        {"feature": feature, "loading": float(weight)}
        for feature, weight in sorted(zip(feature_names, weights.tolist()), key=lambda item: abs(item[1]), reverse=True)
    ]
    return train_factor.reshape(-1, 1), apply_factor.reshape(-1, 1), loading_rows, meta


def design_matrix(variant: str, rows, factor_scores: np.ndarray) -> np.ndarray:
    if variant == "factor_only":
        return factor_scores
    if variant == "geometry_plus_factor":
        return np.hstack([feature_matrix(rows, GEOMETRY_FEATURES), factor_scores])
    raise ValueError(variant)


def coeff_rows(model: Pipeline, variant: str):
    names = ["pair_tail_factor"] if variant == "factor_only" else GEOMETRY_FEATURES + ["pair_tail_factor"]
    ridge = model.named_steps["ridge"]
    return [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in sorted(zip(names, ridge.coef_.tolist()), key=lambda item: abs(item[1]), reverse=True)
    ]


def build_full_predictions(full_rows, agreement_rows, predicted_gain_signal: np.ndarray):
    gain_lookup = {str(row["event_key"]): float(sig) for row, sig in zip(agreement_rows, predicted_gain_signal.tolist())}
    preds = []
    pred_rows = []
    for row in full_rows:
        pre = float(row[MODEL_PRE_ONLY])
        agreed = float(row[MODEL_AGREED])
        if int(row["agreement"]) == 1:
            signal = float(gain_lookup[str(row["event_key"])])
            use_agreed = int(signal > 0.0)
            pred = agreed if use_agreed else pre
        else:
            signal = 0.0
            use_agreed = 0
            pred = pre
        preds.append(pred)
        pred_rows.append(
            {
                "row_key": f"{row['split']}::{row['event_key']}",
                "event_key": row["event_key"],
                "split": row["split"],
                "ticker": row["ticker"],
                "year": int(row["year"]),
                "target": float(row["target"]),
                MODEL_PRE_ONLY: pre,
                MODEL_AGREED: agreed,
                MODEL_HARD_ABSTENTION: float(row[MODEL_HARD_ABSTENTION]),
                "agreement": int(row["agreement"]),
                "predicted_gain_signal": signal,
                "use_agreed": use_agreed,
                "prediction": pred,
            }
        )
    return np.asarray(preds, dtype=float), pred_rows


def load_reference_predictions(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    lookup = {row["event_key"]: row for row in csv.DictReader(path.open())}
    refs = {}
    for name, col in [
        (MODEL_PRE_ONLY, MODEL_PRE_ONLY),
        (MODEL_HARD_ABSTENTION, MODEL_HARD_ABSTENTION),
        ("geometry_only", "geometry_only"),
        ("answerability_factor_route", "prediction"),
    ]:
        refs[name] = np.asarray([float(lookup[key][col]) for key in keys], dtype=float)
    return refs


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_temporal_rows(args.temporal_root.resolve())
    coverage = attach_features(rows, args.tail_features_csv, args.panel_csv, args.features_csv)

    train_rows = [row for row in rows if row["split"] == args.train_split and int(row["agreement"]) == 1]
    val_rows_full = [row for row in rows if row["split"] == args.val_split]
    val_rows_agreement = [row for row in val_rows_full if int(row["agreement"]) == 1]
    refit_train_splits = parse_list(args.refit_train_splits)
    refit_rows = [row for row in rows if row["split"] in refit_train_splits and int(row["agreement"]) == 1]
    test_rows_full = [row for row in rows if row["split"] == args.test_split]
    test_rows_agreement = [row for row in test_rows_full if int(row["agreement"]) == 1]
    if not train_rows or not val_rows_agreement or not refit_rows or not test_rows_agreement:
        raise SystemExit("missing rows for pair-tail benchmark")

    alpha_grid = [float(item) for item in parse_list(args.alphas)]
    val_target = as_array(val_rows_full, "target")
    test_target = as_array(test_rows_full, "target")

    tuning_rows = []
    best_configs = []
    for family_name, family_features in FACTOR_FAMILIES.items():
        for method in FACTOR_METHODS:
            train_factor, val_factor, loading_rows, factor_meta = build_factor_scores(method, family_features, train_rows, val_rows_agreement)
            train_gain = agreement_gain_target(train_rows)
            for variant in VARIANTS:
                train_x = design_matrix(variant, train_rows, train_factor)
                val_x = design_matrix(variant, val_rows_agreement, val_factor)
                best_alpha = None
                best_r2 = None
                best_use = None
                for alpha in alpha_grid:
                    model = fit_ridge(train_x, train_gain, alpha)
                    val_signal = model.predict(val_x)
                    val_pred, _ = build_full_predictions(val_rows_full, val_rows_agreement, val_signal)
                    val_score = metrics(val_target, val_pred)
                    tuning_rows.append(
                        {
                            "family": family_name,
                            "method": method,
                            "variant": variant,
                            "alpha": float(alpha),
                            "val_r2": float(val_score["r2"]),
                            "val_rmse": float(val_score["rmse"]),
                            "val_mae": float(val_score["mae"]),
                            "val_agreement_use_agreed_share": float(np.mean(val_signal > 0.0)),
                            **factor_meta,
                        }
                    )
                    if best_r2 is None or float(val_score["r2"]) > best_r2:
                        best_r2 = float(val_score["r2"])
                        best_alpha = float(alpha)
                        best_use = float(np.mean(val_signal > 0.0))
                best_configs.append(
                    {
                        "family": family_name,
                        "method": method,
                        "variant": variant,
                        "best_alpha": best_alpha,
                        "val_r2": best_r2,
                        "val_agreement_use_agreed_share": best_use,
                        "loading_rows": loading_rows,
                        "factor_meta": factor_meta,
                    }
                )

    best_configs.sort(key=lambda row: row["val_r2"], reverse=True)
    chosen = best_configs[0]
    refit_factor, test_factor, loading_rows, factor_meta = build_factor_scores(
        chosen["method"], FACTOR_FAMILIES[chosen["family"]], refit_rows, test_rows_agreement
    )
    refit_x = design_matrix(chosen["variant"], refit_rows, refit_factor)
    test_x = design_matrix(chosen["variant"], test_rows_agreement, test_factor)
    final_model = fit_ridge(refit_x, agreement_gain_target(refit_rows), float(chosen["best_alpha"]))
    test_signal = final_model.predict(test_x)
    test_pred, test_pred_rows = build_full_predictions(test_rows_full, test_rows_agreement, test_signal)
    test_score = metrics(test_target, test_pred)

    keys = [str(row["event_key"]) for row in test_rows_full]
    refs = load_reference_predictions(args.reference_predictions_csv.resolve(), keys)
    significance = {
        name: summarize_significance(
            test_target,
            test_pred,
            ref_pred,
            bootstrap_iters=args.bootstrap_iters,
            perm_iters=args.perm_iters,
            seed=args.seed,
        )
        for name, ref_pred in refs.items()
    }

    ref_lookup = {key: {name: float(refs[name][idx]) for name in refs} for idx, key in enumerate(keys)}
    pred_rows = []
    for row in test_pred_rows:
        merged = dict(row)
        for name, value in ref_lookup[str(row["event_key"])].items():
            merged[name] = value
        merged["chosen_family"] = chosen["family"]
        merged["chosen_method"] = chosen["method"]
        merged["chosen_variant"] = chosen["variant"]
        pred_rows.append(merged)

    score_rows = [{"model": "pair_tail_factor_route", **test_score}]
    for name, pred in refs.items():
        score_rows.append({"model": name, **metrics(test_target, pred)})

    summary = {
        "coverage": coverage,
        "config": {
            "train_split": args.train_split,
            "val_split": args.val_split,
            "test_split": args.test_split,
            "refit_train_splits": refit_train_splits,
            "alphas": alpha_grid,
            "families": FACTOR_FAMILIES,
            "methods": list(FACTOR_METHODS),
            "variants": list(VARIANTS),
        },
        "counts": {
            "train_agreement_rows": len(train_rows),
            "val_rows_full": len(val_rows_full),
            "val_agreement_rows": len(val_rows_agreement),
            "refit_agreement_rows": len(refit_rows),
            "test_rows_full": len(test_rows_full),
            "test_agreement_rows": len(test_rows_agreement),
        },
        "best_validation_config": {
            "family": chosen["family"],
            "method": chosen["method"],
            "variant": chosen["variant"],
            "alpha": float(chosen["best_alpha"]),
            "val_r2": float(chosen["val_r2"]),
            "val_agreement_use_agreed_share": float(chosen["val_agreement_use_agreed_share"]),
        },
        "best_factor_meta": factor_meta,
        "best_factor_loadings": loading_rows,
        "best_route_coefficients": coeff_rows(final_model, chosen["variant"]),
        "test_scores": score_rows,
        "test_agreement_use_agreed_share": float(np.mean(test_signal > 0.0)),
        "test_significance": significance,
        "top_validation_configs": [
            {
                "family": row["family"],
                "method": row["method"],
                "variant": row["variant"],
                "alpha": float(row["best_alpha"]),
                "val_r2": float(row["val_r2"]),
                "val_agreement_use_agreed_share": float(row["val_agreement_use_agreed_share"]),
            }
            for row in best_configs[:8]
        ],
    }

    write_json(output_dir / "afterhours_transfer_pair_tail_factor_benchmark_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_pair_tail_factor_benchmark_tuning.csv", tuning_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_factor_benchmark_test_predictions.csv", pred_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_factor_benchmark_scores.csv", score_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_factor_benchmark_loadings.csv", loading_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_factor_benchmark_coefficients.csv", coeff_rows(final_model, chosen["variant"]))


if __name__ == "__main__":
    main()
