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
from sklearn.decomposition import TruncatedSVD

from dj30_qc_utils import load_csv_rows, write_csv, write_json
from run_dense_multimodal_ablation_baselines import (
    build_dense_matrix,
    build_text_lsa_bundle,
    infer_aligned_audio_feature_names,
    infer_audio_feature_names,
    standardize,
)
from run_offhours_shock_ablations import (
    paired_bootstrap_deltas,
    paired_sign_permutation_pvalue,
    regime_label,
)
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_signal_decomposition_benchmarks import CONTROL_FEATURES
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


PRE_CALL_MARKET_FEATURES = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "pre_60m_volume_sum",
]

A4_STRUCTURED_FEATURES = [
    "a4_kept_rows_for_duration",
    "a4_median_match_score",
    "a4_strict_row_share",
    "a4_broad_row_share",
    "a4_hard_fail_rows",
]

ALIGNED_AUDIO_META_FIELDS = {"event_key", "ticker", "year", "quarter"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark old audio proxy versus A4-aligned acoustic aggregates on the after-hours line."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_audio_upgrade_benchmark_real"),
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
    parser.add_argument("--lsa-components", type=int, default=16)
    parser.add_argument("--aligned-prefix", default="aligned_audio__")
    parser.add_argument("--aligned-agg-suffixes", default="winsor_mean")
    parser.add_argument("--aligned-compressed-components", type=int, default=0)
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


def build_compressed_dense_bundle(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
    components: int,
    prefix: str,
):
    train_x, medians, extra_names = build_dense_matrix(train_rows, feature_names)
    val_x, _, _ = build_dense_matrix(val_rows, feature_names, medians)
    test_x, _, _ = build_dense_matrix(test_rows, feature_names, medians)
    train_z, val_z, test_z = standardize(train_x, [val_x, test_x])
    n_components = min(max(2, components), train_z.shape[0] - 1, train_z.shape[1])
    if n_components < 2:
        raise SystemExit(f"not enough train rows/features to compress bundle {prefix}")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_comp = svd.fit_transform(train_z)
    val_comp = svd.transform(val_z)
    test_comp = svd.transform(test_z)
    return {
        "train": train_comp,
        "val": val_comp,
        "test": test_comp,
        "feature_names": [f"{prefix}_{idx+1}" for idx in range(n_components)] + extra_names,
        "n_components": int(n_components),
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
    }


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


def load_joined_rows(
    panel_csv: Path,
    features_csv: Path,
    old_audio_csv: Path,
    aligned_audio_csv: Path,
    aligned_prefix: str,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    feature_lookup = {}
    for row in load_csv_rows(features_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            feature_lookup[event_key] = row

    old_audio_lookup = {}
    for row in load_csv_rows(old_audio_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            old_audio_lookup[event_key] = row

    aligned_audio_lookup = load_prefixed_lookup(
        aligned_audio_csv.resolve(),
        prefix=aligned_prefix,
        meta_fields=ALIGNED_AUDIO_META_FIELDS,
    )

    coverage = {
        "panel_rows": 0,
        "with_features": 0,
        "with_old_audio": 0,
        "with_aligned_audio": 0,
        "with_both_audio": 0,
        "joined_rows": 0,
    }

    rows = []
    for row in load_csv_rows(panel_csv.resolve()):
        coverage["panel_rows"] += 1
        event_key = row.get("event_key", "")
        year_value = row.get("year", "")
        feature_row = feature_lookup.get(event_key)
        old_audio_row = old_audio_lookup.get(event_key)
        aligned_audio_row = aligned_audio_lookup.get(event_key)

        if feature_row is not None:
            coverage["with_features"] += 1
        if old_audio_row is not None:
            coverage["with_old_audio"] += 1
        if aligned_audio_row is not None:
            coverage["with_aligned_audio"] += 1
        if old_audio_row is not None and aligned_audio_row is not None:
            coverage["with_both_audio"] += 1

        if not event_key or not year_value or feature_row is None or old_audio_row is None or aligned_audio_row is None:
            continue

        merged = dict(row)
        merged.update(feature_row)
        merged.update(old_audio_row)
        merged.update(aligned_audio_row)
        merged["_year"] = int(float(year_value))
        rows.append(merged)

    coverage["joined_rows"] = len(rows)
    return rows, coverage


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    aligned_agg_suffixes = tuple(item.strip() for item in args.aligned_agg_suffixes.split(",") if item.strip())
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    base_rows, coverage = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.old_audio_csv,
        args.aligned_audio_csv,
        args.aligned_prefix,
    )

    rows = []
    for row in base_rows:
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        if html_flag in exclude_html_flags:
            continue
        reg = regime_label(row)
        if reg not in include_regimes:
            continue
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        item["_regime"] = reg
        rows.append(item)

    rows = attach_ticker_prior(rows, args.train_end_year)
    train_rows = [row for row in rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in rows if row["_year"] == args.val_year]
    test_rows = [row for row in rows if row["_year"] > args.val_year]

    old_audio_feature_names = infer_audio_feature_names(rows)
    aligned_audio_feature_names = infer_aligned_audio_feature_names(
        rows,
        prefix=args.aligned_prefix,
        agg_suffixes=aligned_agg_suffixes,
    )

    bundles = {
        "pre_call_market": build_dense_bundle(train_rows, val_rows, test_rows, PRE_CALL_MARKET_FEATURES),
        "controls": build_dense_bundle(train_rows, val_rows, test_rows, CONTROL_FEATURES),
        "a4": build_dense_bundle(train_rows, val_rows, test_rows, A4_STRUCTURED_FEATURES),
        "old_audio": build_dense_bundle(train_rows, val_rows, test_rows, old_audio_feature_names),
        "aligned_audio": build_dense_bundle(train_rows, val_rows, test_rows, aligned_audio_feature_names),
        "qna_lsa": build_text_lsa_bundle(
            train_rows,
            val_rows,
            test_rows,
            text_col="qna_text",
            max_features=args.max_features,
            min_df=args.min_df,
            lsa_components=args.lsa_components,
        ),
    }
    if args.aligned_compressed_components > 0:
        bundles["aligned_audio_svd"] = build_compressed_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            aligned_audio_feature_names,
            args.aligned_compressed_components,
            prefix="aligned_audio_svd",
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
        "residual_pre_call_market_plus_controls": ["pre_call_market", "controls"],
        "residual_pre_call_market_plus_controls_plus_old_audio": [
            "pre_call_market",
            "controls",
            "old_audio",
        ],
        "residual_pre_call_market_plus_controls_plus_aligned_audio": [
            "pre_call_market",
            "controls",
            "aligned_audio",
        ],
        "residual_pre_call_market_plus_controls_plus_aligned_audio_svd": [
            "pre_call_market",
            "controls",
            "aligned_audio_svd",
        ],
        "residual_pre_call_market_plus_controls_plus_old_plus_aligned_audio": [
            "pre_call_market",
            "controls",
            "old_audio",
            "aligned_audio",
        ],
        "residual_pre_call_market_plus_controls_plus_a4": ["pre_call_market", "controls", "a4"],
        "residual_pre_call_market_plus_controls_plus_a4_plus_old_audio": [
            "pre_call_market",
            "controls",
            "a4",
            "old_audio",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_aligned_audio": [
            "pre_call_market",
            "controls",
            "a4",
            "aligned_audio",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_aligned_audio_svd": [
            "pre_call_market",
            "controls",
            "a4",
            "aligned_audio_svd",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_old_plus_aligned_audio": [
            "pre_call_market",
            "controls",
            "a4",
            "old_audio",
            "aligned_audio",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_old_audio": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
            "old_audio",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_aligned_audio": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
            "aligned_audio",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_aligned_audio_svd": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
            "aligned_audio_svd",
        ],
        "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_old_plus_aligned_audio": [
            "pre_call_market",
            "controls",
            "a4",
            "qna_lsa",
            "old_audio",
            "aligned_audio",
        ],
    }
    model_specs = {
        model_name: bundle_names
        for model_name, bundle_names in model_specs.items()
        if all(bundle_name in bundles for bundle_name in bundle_names)
    }

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted(include_regimes),
        "exclude_html_flags": sorted(exclude_html_flags),
        "coverage": coverage,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "config": {
            "alphas": alphas,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "lsa_components": args.lsa_components,
            "aligned_prefix": args.aligned_prefix,
            "aligned_agg_suffixes": list(aligned_agg_suffixes),
            "aligned_compressed_components": args.aligned_compressed_components,
        },
        "feature_groups": {
            "pre_call_market": PRE_CALL_MARKET_FEATURES,
            "controls": CONTROL_FEATURES,
            "a4": A4_STRUCTURED_FEATURES,
            "old_audio": old_audio_feature_names,
            "aligned_audio": aligned_audio_feature_names,
        },
        "models": {},
        "significance": {},
    }
    if "aligned_audio_svd" in bundles:
        summary["feature_groups"]["aligned_audio_svd"] = {
            "input_feature_count": len(aligned_audio_feature_names),
            "n_components": bundles["aligned_audio_svd"]["n_components"],
            "explained_variance_ratio_sum": bundles["aligned_audio_svd"]["explained_variance_ratio_sum"],
        }
    prediction_rows = []
    test_predictions = {}

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
                "best_alpha": best_alpha,
                "val": metrics(val_y, pred_val),
                "test": metrics(test_y, pred_test),
                "top_coefficients": top_coefficients(best_model, feature_names),
            }

        test_predictions[model_name] = pred_test
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
                        "regime": row["_regime"],
                        "html_integrity_flag": row.get("html_integrity_flag", ""),
                        "y_true": row["_target"],
                        "y_pred": float(pred),
                    }
                )

    comparisons = {
        "audio_upgrade_vs_old_audio_controls_only": (
            "residual_pre_call_market_plus_controls_plus_old_audio",
            "residual_pre_call_market_plus_controls_plus_aligned_audio",
        ),
        "audio_upgrade_vs_no_audio_controls_only": (
            "residual_pre_call_market_plus_controls",
            "residual_pre_call_market_plus_controls_plus_aligned_audio",
        ),
        "audio_upgrade_svd_vs_no_audio_controls_only": (
            "residual_pre_call_market_plus_controls",
            "residual_pre_call_market_plus_controls_plus_aligned_audio_svd",
        ),
        "merged_vs_aligned_controls_only": (
            "residual_pre_call_market_plus_controls_plus_aligned_audio",
            "residual_pre_call_market_plus_controls_plus_old_plus_aligned_audio",
        ),
        "audio_upgrade_vs_old_audio_no_semantics": (
            "residual_pre_call_market_plus_controls_plus_a4_plus_old_audio",
            "residual_pre_call_market_plus_controls_plus_a4_plus_aligned_audio",
        ),
        "audio_upgrade_vs_no_audio_no_semantics": (
            "residual_pre_call_market_plus_controls_plus_a4",
            "residual_pre_call_market_plus_controls_plus_a4_plus_aligned_audio",
        ),
        "audio_upgrade_svd_vs_no_audio_no_semantics": (
            "residual_pre_call_market_plus_controls_plus_a4",
            "residual_pre_call_market_plus_controls_plus_a4_plus_aligned_audio_svd",
        ),
        "merged_vs_aligned_no_semantics": (
            "residual_pre_call_market_plus_controls_plus_a4_plus_aligned_audio",
            "residual_pre_call_market_plus_controls_plus_a4_plus_old_plus_aligned_audio",
        ),
        "audio_upgrade_vs_old_audio_with_semantics": (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_old_audio",
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_aligned_audio",
        ),
        "audio_upgrade_vs_no_audio_with_semantics": (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_aligned_audio",
        ),
        "audio_upgrade_svd_vs_no_audio_with_semantics": (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa",
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_aligned_audio_svd",
        ),
        "merged_vs_aligned_with_semantics": (
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_aligned_audio",
            "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_plus_old_plus_aligned_audio",
        ),
    }
    comparisons = {
        label: pair
        for label, pair in comparisons.items()
        if pair[0] in test_predictions and pair[1] in test_predictions
    }
    for label, (baseline_name, challenger_name) in comparisons.items():
        baseline_pred = np.asarray(test_predictions[baseline_name], dtype=float)
        challenger_pred = np.asarray(test_predictions[challenger_name], dtype=float)
        summary["significance"][label] = {
            "baseline_model": baseline_name,
            "challenger_model": challenger_name,
            "baseline_test": summary["models"][baseline_name]["test"],
            "challenger_test": summary["models"][challenger_name]["test"],
            "bootstrap": paired_bootstrap_deltas(
                test_y,
                baseline_pred,
                challenger_pred,
                args.bootstrap_iters,
                args.seed,
            ),
            "permutation": paired_sign_permutation_pvalue(
                test_y,
                baseline_pred,
                challenger_pred,
                args.perm_iters,
                args.seed,
            ),
        }

    write_csv(output_dir / "afterhours_audio_upgrade_predictions.csv", prediction_rows)
    write_json(output_dir / "afterhours_audio_upgrade_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
