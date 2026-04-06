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
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    if user_site_removed and isinstance(user_site, str) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

from dj30_qc_utils import write_csv, write_json
from run_afterhours_transfer_agreement_signal_benchmark import (
    MODEL_AGREED,
    MODEL_HARD_ABSTENTION,
    MODEL_PRE_ONLY,
    agreement_gain_target,
    as_array,
    attach_side_features,
    build_temporal_rows,
    feature_matrix,
    fit_ridge,
    parse_list,
)
from run_afterhours_transfer_expert_selection import summarize_significance
from run_structured_baselines import metrics

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
    "responsiveness_core": [
        "qa_pair_answer_forward_rate_mean",
        "qa_evasive_proxy_share",
        "qa_bench_coverage_mean",
        "qa_bench_direct_early_score_mean",
        "qa_bench_evasion_score_mean",
        "qa_pair_count",
    ],
    "responsiveness_plus_observability": [
        "qa_pair_answer_forward_rate_mean",
        "qa_evasive_proxy_share",
        "qa_bench_coverage_mean",
        "qa_bench_direct_early_score_mean",
        "qa_bench_evasion_score_mean",
        "qa_pair_count",
        "a4_strict_row_share",
    ],
    "directness_coverage_core": [
        "qa_pair_answer_forward_rate_mean",
        "qa_bench_coverage_mean",
        "qa_bench_direct_early_score_mean",
        "qa_bench_evasion_score_mean",
        "qa_bench_direct_answer_share",
    ],
    "observability_directness_core": [
        "a4_strict_row_share",
        "a4_strict_high_conf_share",
        "qa_pair_count",
        "qa_bench_coverage_mean",
        "qa_bench_direct_early_score_mean",
        "qa_bench_evasion_score_mean",
    ],
}

FACTOR_METHODS = ("pca1", "pls1")
VARIANTS = ("factor_only", "geometry_plus_factor")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark distilled one-factor responsiveness latents for agreement-side transfer refinement."
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
        "--reference-predictions-csv",
        type=Path,
        default=Path(
            "results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/"
            "afterhours_transfer_agreement_signal_benchmark_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--reference-summary-json",
        type=Path,
        default=Path(
            "results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/"
            "afterhours_transfer_agreement_signal_benchmark_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_responsiveness_factor_benchmark_role_aware_audio_lsa4_real"),
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


def build_factor_scores(
    method: str,
    feature_names: list[str],
    fit_rows: list[dict[str, float | int | str]],
    apply_rows: list[dict[str, float | int | str]],
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | str]], dict[str, float]]:
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
        explained_variance = float(model.explained_variance_ratio_[0])
        factor_meta = {
            "component_explained_variance_ratio": explained_variance,
        }
    elif method == "pls1":
        model = PLSRegression(n_components=1, scale=False)
        model.fit(train_x_scaled, train_y)
        train_factor = model.transform(train_x_scaled)[:, 0]
        apply_factor = model.transform(apply_x_scaled)[:, 0]
        weights = model.x_weights_[:, 0]
        factor_meta = {
            "x_score_variance": float(np.var(train_factor)),
        }
    else:
        raise ValueError(f"unsupported factor method: {method}")

    corr = float(np.corrcoef(train_factor, train_y)[0, 1]) if np.std(train_factor) > 0 and np.std(train_y) > 0 else 0.0
    if corr < 0.0:
        train_factor = -train_factor
        apply_factor = -apply_factor
        weights = -weights
        corr = -corr

    loading_rows = [
        {"feature": feature, "loading": float(weight)}
        for feature, weight in sorted(
            zip(feature_names, weights.tolist()), key=lambda item: abs(item[1]), reverse=True
        )
    ]
    factor_meta["train_gain_correlation"] = corr
    factor_meta["train_factor_mean"] = float(np.mean(train_factor))
    factor_meta["train_factor_std"] = float(np.std(train_factor))
    return train_factor.reshape(-1, 1), apply_factor.reshape(-1, 1), loading_rows, factor_meta


def load_reference_predictions(path: Path) -> tuple[list[dict[str, str]], dict[str, np.ndarray]]:
    rows = list(csv.DictReader(path.open()))
    refs: dict[str, np.ndarray] = {}
    if not rows:
        return rows, refs
    wanted = [
        MODEL_PRE_ONLY,
        MODEL_AGREED,
        MODEL_HARD_ABSTENTION,
        "geometry_only",
        "geometry_plus_hybrid",
    ]
    for key in wanted:
        if key in rows[0]:
            refs[key] = np.asarray([float(row[key]) for row in rows], dtype=float)
    return rows, refs


def route_design_matrix(
    variant: str,
    rows: list[dict[str, float | int | str]],
    factor_scores: np.ndarray,
) -> np.ndarray:
    if variant == "factor_only":
        return factor_scores
    if variant == "geometry_plus_factor":
        return np.hstack([feature_matrix(rows, GEOMETRY_FEATURES), factor_scores])
    raise ValueError(f"unsupported variant: {variant}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_temporal_rows(args.temporal_root.resolve())
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
        raise SystemExit("missing train/val/refit/test rows for responsiveness-factor benchmark")

    _, ref_predictions = load_reference_predictions(args.reference_predictions_csv.resolve())
    reference_summary = None
    if args.reference_summary_json.resolve().exists():
        reference_summary = json.loads(args.reference_summary_json.resolve().read_text())

    val_target = as_array(val_rows, "target")
    test_agreement_target = as_array(test_agreement_rows, "target")
    test_full_target = as_array(test_full_rows, "target")
    test_full_pre = as_array(test_full_rows, MODEL_PRE_ONLY)
    test_full_agreed = as_array(test_full_rows, MODEL_AGREED)
    test_full_hard = as_array(test_full_rows, MODEL_HARD_ABSTENTION)

    overview_rows: list[dict[str, float | str | int]] = []
    tuning_rows: list[dict[str, float | str | int]] = []
    factor_loading_rows: list[dict[str, float | str | int]] = []
    prediction_rows = [dict(row) for row in test_full_rows]
    route_predictions = {
        MODEL_PRE_ONLY: test_full_pre,
        MODEL_AGREED: test_full_agreed,
        MODEL_HARD_ABSTENTION: test_full_hard,
    }
    route_summaries: dict[str, dict[str, object]] = {}

    best_route_key = None
    best_route_r2 = None
    best_factor_only_key = None
    best_factor_only_r2 = None
    best_geometry_plus_factor_key = None
    best_geometry_plus_factor_r2 = None

    for family_name, feature_names in FACTOR_FAMILIES.items():
        for method in FACTOR_METHODS:
            train_factor, val_factor, _, _ = build_factor_scores(
                method, feature_names, train_rows, val_rows
            )
            train_gain = agreement_gain_target(train_rows)

            best_variant_payloads: dict[str, dict[str, float | str | int]] = {}
            for variant in VARIANTS:
                train_x = route_design_matrix(variant, train_rows, train_factor)
                val_x = route_design_matrix(variant, val_rows, val_factor)
                best_payload = None
                for alpha in alpha_grid:
                    model = fit_ridge(train_x, train_gain, alpha)
                    val_signal = model.predict(val_x)
                    val_pred = np.where(val_signal > 0.0, as_array(val_rows, MODEL_AGREED), as_array(val_rows, MODEL_PRE_ONLY))
                    score = metrics(val_target, val_pred)
                    payload = {
                        "family": family_name,
                        "method": method,
                        "variant": variant,
                        "alpha": float(alpha),
                        "val_r2": score["r2"],
                        "val_rmse": score["rmse"],
                        "val_mae": score["mae"],
                        "val_use_agreed_share": float(np.mean(val_signal > 0.0)),
                        "val_signal_min": float(np.min(val_signal)),
                        "val_signal_max": float(np.max(val_signal)),
                    }
                    tuning_rows.append(payload)
                    if best_payload is None or float(payload["val_r2"]) > float(best_payload["val_r2"]):
                        best_payload = payload
                assert best_payload is not None
                best_variant_payloads[variant] = best_payload

            refit_factor, test_agreement_factor, refit_loadings, refit_factor_meta = build_factor_scores(
                method, feature_names, refit_rows, test_agreement_rows
            )
            for variant in VARIANTS:
                route_key = f"{family_name}__{method}__{variant}"
                best_alpha = float(best_variant_payloads[variant]["alpha"])
                refit_x = route_design_matrix(variant, refit_rows, refit_factor)
                test_agreement_x = route_design_matrix(variant, test_agreement_rows, test_agreement_factor)
                model = fit_ridge(refit_x, agreement_gain_target(refit_rows), best_alpha)
                test_signal = model.predict(test_agreement_x)
                test_agreement_pred = np.where(
                    test_signal > 0.0,
                    as_array(test_agreement_rows, MODEL_AGREED),
                    as_array(test_agreement_rows, MODEL_PRE_ONLY),
                )
                pred_lookup = {
                    row["event_key"]: float(pred) for row, pred in zip(test_agreement_rows, test_agreement_pred)
                }
                signal_lookup = {
                    row["event_key"]: float(signal) for row, signal in zip(test_agreement_rows, test_signal)
                }
                use_lookup = {
                    row["event_key"]: int(signal > 0.0) for row, signal in zip(test_agreement_rows, test_signal)
                }
                test_full_pred = np.asarray(
                    [pred_lookup.get(row["event_key"], float(row[MODEL_PRE_ONLY])) for row in test_full_rows], dtype=float
                )
                route_predictions[route_key] = test_full_pred
                for row in prediction_rows:
                    event_key = row["event_key"]
                    row[route_key] = pred_lookup.get(event_key, float(row[MODEL_PRE_ONLY]))
                    row[f"{route_key}__predicted_gain_signal"] = signal_lookup.get(event_key, 0.0)
                    row[f"{route_key}__use_agreed"] = use_lookup.get(event_key, 0)

                test_agreement_metrics = metrics(test_agreement_target, test_agreement_pred)
                test_full_metrics = metrics(test_full_target, test_full_pred)
                sig_vs_hard = summarize_significance(
                    test_full_target, test_full_hard, test_full_pred, args.bootstrap_iters, args.perm_iters, args.seed
                )
                sig_vs_pre = summarize_significance(
                    test_full_target, test_full_pre, test_full_pred, args.bootstrap_iters, args.perm_iters, args.seed
                )
                sig_vs_geometry = None
                if "geometry_only" in ref_predictions:
                    sig_vs_geometry = summarize_significance(
                        test_full_target,
                        ref_predictions["geometry_only"],
                        test_full_pred,
                        args.bootstrap_iters,
                        args.perm_iters,
                        args.seed,
                    )
                sig_vs_geometry_hybrid = None
                if "geometry_plus_hybrid" in ref_predictions:
                    sig_vs_geometry_hybrid = summarize_significance(
                        test_full_target,
                        ref_predictions["geometry_plus_hybrid"],
                        test_full_pred,
                        args.bootstrap_iters,
                        args.perm_iters,
                        args.seed,
                    )

                overview_row = {
                    "route": route_key,
                    "family": family_name,
                    "method": method,
                    "variant": variant,
                    "factor_feature_count": len(feature_names),
                    "selected_alpha": best_alpha,
                    "val_r2": float(best_variant_payloads[variant]["val_r2"]),
                    "val_use_agreed_share": float(best_variant_payloads[variant]["val_use_agreed_share"]),
                    "test_agreement_r2": test_agreement_metrics["r2"],
                    "test_agreement_rmse": test_agreement_metrics["rmse"],
                    "test_full_r2": test_full_metrics["r2"],
                    "test_full_rmse": test_full_metrics["rmse"],
                    "test_full_mae": test_full_metrics["mae"],
                    "test_full_use_agreed_share": float(np.mean(test_signal > 0.0)),
                    "test_full_p_mse_vs_hard": sig_vs_hard["mse_gain_pvalue"],
                    "test_full_p_mse_vs_pre": sig_vs_pre["mse_gain_pvalue"],
                    "test_full_p_mse_vs_geometry_only": None if sig_vs_geometry is None else sig_vs_geometry["mse_gain_pvalue"],
                    "test_full_p_mse_vs_geometry_plus_hybrid": None
                    if sig_vs_geometry_hybrid is None
                    else sig_vs_geometry_hybrid["mse_gain_pvalue"],
                    "train_factor_gain_corr": refit_factor_meta["train_gain_correlation"],
                    "train_factor_std": refit_factor_meta["train_factor_std"],
                }
                overview_rows.append(overview_row)
                route_summaries[route_key] = {
                    "family": family_name,
                    "method": method,
                    "variant": variant,
                    "feature_names": feature_names,
                    "selected_alpha": best_alpha,
                    "best_validation": best_variant_payloads[variant],
                    "factor_meta": refit_factor_meta,
                    "test_agreement_metrics": test_agreement_metrics,
                    "test_full_metrics": test_full_metrics,
                    "significance_vs_hard": sig_vs_hard,
                    "significance_vs_pre": sig_vs_pre,
                    "significance_vs_geometry_only": sig_vs_geometry,
                    "significance_vs_geometry_plus_hybrid": sig_vs_geometry_hybrid,
                    "top_factor_loadings": refit_loadings[:8],
                }
                for rank, loading_row in enumerate(refit_loadings, start=1):
                    factor_loading_rows.append(
                        {
                            "route": route_key,
                            "family": family_name,
                            "method": method,
                            "variant": variant,
                            "rank": rank,
                            **loading_row,
                        }
                    )

                route_r2 = float(test_full_metrics["r2"])
                if best_route_r2 is None or route_r2 > best_route_r2:
                    best_route_r2 = route_r2
                    best_route_key = route_key
                if variant == "factor_only" and (best_factor_only_r2 is None or route_r2 > best_factor_only_r2):
                    best_factor_only_r2 = route_r2
                    best_factor_only_key = route_key
                if variant == "geometry_plus_factor" and (
                    best_geometry_plus_factor_r2 is None or route_r2 > best_geometry_plus_factor_r2
                ):
                    best_geometry_plus_factor_r2 = route_r2
                    best_geometry_plus_factor_key = route_key

    best_vs_hard = None
    best_vs_geometry_only = None
    best_vs_geometry_plus_hybrid = None
    if best_route_key is not None:
        best_pred = route_predictions[best_route_key]
        best_vs_hard = summarize_significance(
            test_full_target, test_full_hard, best_pred, args.bootstrap_iters, args.perm_iters, args.seed
        )
        if "geometry_only" in ref_predictions:
            best_vs_geometry_only = summarize_significance(
                test_full_target,
                ref_predictions["geometry_only"],
                best_pred,
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            )
        if "geometry_plus_hybrid" in ref_predictions:
            best_vs_geometry_plus_hybrid = summarize_significance(
                test_full_target,
                ref_predictions["geometry_plus_hybrid"],
                best_pred,
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            )

    summary = {
        "config": {
            "temporal_root": str(args.temporal_root.resolve()),
            "panel_csv": str(args.panel_csv.resolve()),
            "features_csv": str(args.features_csv.resolve()),
            "qa_csv": str(args.qa_csv.resolve()),
            "aligned_audio_csv": str(args.aligned_audio_csv.resolve()),
            "reference_predictions_csv": str(args.reference_predictions_csv.resolve()),
            "reference_summary_json": str(args.reference_summary_json.resolve()),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "refit_train_splits": refit_train_splits,
            "test_split": args.test_split,
            "alpha_grid": alpha_grid,
            "factor_families": FACTOR_FAMILIES,
            "factor_methods": list(FACTOR_METHODS),
            "variants": list(VARIANTS),
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
            "reference_signal_summary": {
                "best_family": None if reference_summary is None else reference_summary.get("best_family"),
                "best_family_test_full_r2": None
                if reference_summary is None
                else reference_summary.get("best_family_summary", {}).get("test_full_metrics", {}).get("r2"),
            },
            "reference_prediction_metrics": {
                key: metrics(test_full_target, values) for key, values in ref_predictions.items()
            },
        },
        "overview": overview_rows,
        "tuning": tuning_rows,
        "factor_loadings": factor_loading_rows,
        "best_route": best_route_key,
        "best_route_summary": None if best_route_key is None else route_summaries.get(best_route_key),
        "best_factor_only_route": best_factor_only_key,
        "best_factor_only_summary": None if best_factor_only_key is None else route_summaries.get(best_factor_only_key),
        "best_geometry_plus_factor_route": best_geometry_plus_factor_key,
        "best_geometry_plus_factor_summary": None
        if best_geometry_plus_factor_key is None
        else route_summaries.get(best_geometry_plus_factor_key),
        "best_route_significance": {
            "vs_hard": best_vs_hard,
            "vs_geometry_only": best_vs_geometry_only,
            "vs_geometry_plus_hybrid": best_vs_geometry_plus_hybrid,
        },
        "route_summaries": route_summaries,
    }

    write_csv(output_dir / "afterhours_transfer_responsiveness_factor_benchmark_overview.csv", overview_rows)
    write_csv(output_dir / "afterhours_transfer_responsiveness_factor_benchmark_tuning.csv", tuning_rows)
    write_csv(output_dir / "afterhours_transfer_responsiveness_factor_benchmark_factor_loadings.csv", factor_loading_rows)
    write_csv(output_dir / "afterhours_transfer_responsiveness_factor_benchmark_test_predictions.csv", prediction_rows)
    write_json(output_dir / "afterhours_transfer_responsiveness_factor_benchmark_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
