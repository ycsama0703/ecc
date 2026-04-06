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
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle, load_joined_rows
from run_offhours_shock_ablations import (
    paired_bootstrap_deltas,
    paired_sign_permutation_pvalue,
    regime_label,
)
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_signal_decomposition_benchmarks import (
    CONTROL_FEATURES,
    ECC_STRUCTURED_FEATURES,
    MARKET_FEATURES,
)
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


PRE_CALL_MARKET_FEATURES = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "pre_60m_volume_sum",
]

WITHIN_CALL_MARKET_FEATURES = [
    "within_call_rv",
    "within_call_vw_rv",
    "within_call_volume_sum",
]

A1_A2_STRUCTURED_FEATURES = [
    "a1_component_count",
    "a1_question_count",
    "a1_answer_count",
    "a1_qna_component_share",
    "a1_total_text_words",
    "a1_unique_speaker_count",
    "a2_paragraph_count",
    "a2_visible_word_count",
    "a2_size_ratio_vs_group",
    "a2_text_ratio_vs_group",
]

A4_STRUCTURED_FEATURES = [
    "a4_kept_rows_for_duration",
    "a4_median_match_score",
    "a4_strict_row_share",
    "a4_broad_row_share",
    "a4_hard_fail_rows",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run corrected off-hours feature-group ladders and paired significance tests."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/offhours_feature_group_ladder_real"),
    )
    parser.add_argument("--include-regimes", default="pre_market,after_hours")
    parser.add_argument("--exclude-html-flags", default="")
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=64)
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


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
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
    return attach_ticker_prior(rows, args.train_end_year)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(args)
    train_rows = [row for row in rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in rows if row["_year"] == args.val_year]
    test_rows = [row for row in rows if row["_year"] > args.val_year]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    bundles = {
        "pre_call_market": build_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            PRE_CALL_MARKET_FEATURES,
        ),
        "within_call_market": build_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            WITHIN_CALL_MARKET_FEATURES,
        ),
        "market": build_dense_bundle(train_rows, val_rows, test_rows, MARKET_FEATURES),
        "controls": build_dense_bundle(train_rows, val_rows, test_rows, CONTROL_FEATURES),
        "a1_a2_structure": build_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            A1_A2_STRUCTURED_FEATURES,
        ),
        "a4_structure": build_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            A4_STRUCTURED_FEATURES,
        ),
        "ecc_structure": build_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            ECC_STRUCTURED_FEATURES,
        ),
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

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in train_rows], dtype=float)
    val_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in val_rows], dtype=float)
    test_prior = np.asarray([float(row["prior_ticker_expanding_mean"]) for row in test_rows], dtype=float)

    model_specs = {
        "prior_only": [],
        "residual_pre_call_market_only": ["pre_call_market"],
        "residual_market_only": ["market"],
        "residual_market_plus_controls": ["market", "controls"],
        "residual_market_controls_plus_a1_a2": ["market", "controls", "a1_a2_structure"],
        "residual_market_controls_plus_a4": ["market", "controls", "a4_structure"],
        "residual_market_controls_plus_ecc_structure": ["market", "controls", "ecc_structure"],
        "residual_market_controls_plus_ecc_structure_plus_qna_lsa": [
            "market",
            "controls",
            "ecc_structure",
            "qna_lsa",
        ],
    }

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted({item.strip() for item in args.include_regimes.split(",") if item.strip()}),
        "exclude_html_flags": sorted(
            {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
        ),
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "feature_groups": {
            "pre_call_market": PRE_CALL_MARKET_FEATURES,
            "within_call_market": WITHIN_CALL_MARKET_FEATURES,
            "market": MARKET_FEATURES,
            "controls": CONTROL_FEATURES,
            "a1_a2_structure": A1_A2_STRUCTURED_FEATURES,
            "a4_structure": A4_STRUCTURED_FEATURES,
            "ecc_structure": ECC_STRUCTURED_FEATURES,
            "qna_lsa_components": int(bundles["qna_lsa"]["train"].shape[1]),
        },
        "models": {},
        "significance": {},
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

        test_predictions[model_name] = np.asarray(pred_test, dtype=float)
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
        ("prior_only", "residual_pre_call_market_only"),
        ("residual_pre_call_market_only", "residual_market_only"),
        ("residual_market_only", "residual_market_plus_controls"),
        ("residual_market_plus_controls", "residual_market_controls_plus_a1_a2"),
        ("residual_market_plus_controls", "residual_market_controls_plus_a4"),
        ("residual_market_plus_controls", "residual_market_controls_plus_ecc_structure"),
        (
            "residual_market_controls_plus_ecc_structure",
            "residual_market_controls_plus_ecc_structure_plus_qna_lsa",
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

    write_json(output_dir / "offhours_feature_group_ladder_summary.json", summary)
    write_csv(output_dir / "offhours_feature_group_ladder_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
