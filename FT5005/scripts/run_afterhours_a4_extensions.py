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

from dj30_qc_utils import load_csv_rows, write_csv, write_json
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle, load_joined_rows
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


A4_STRUCTURED_FEATURES = [
    "a4_kept_rows_for_duration",
    "a4_median_match_score",
    "a4_strict_row_share",
    "a4_broad_row_share",
    "a4_hard_fail_rows",
]

MARKET_FEATURES = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "pre_60m_volume_sum",
    "within_call_rv",
    "within_call_vw_rv",
    "within_call_volume_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test after-hours A4-centered extensions with sequence features and Q&A semantics."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument("--role-sequence-csv", type=Path, required=True)
    parser.add_argument("--weak-sequence-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_a4_extensions_real"),
    )
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="")
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--sequence-prefix", default="strict_")
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


def load_sequence_lookup(path: Path) -> dict[str, dict[str, str]]:
    lookup = {}
    for row in load_csv_rows(path.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            lookup[event_key] = row
    return lookup


def infer_prefixed_feature_names(rows: list[dict[str, str]], prefix: str) -> list[str]:
    names = set()
    for row in rows:
        for key in row:
            if key.startswith(prefix):
                names.add(key)
    return sorted(names)


def build_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    role_lookup = load_sequence_lookup(args.role_sequence_csv)
    weak_lookup = load_sequence_lookup(args.weak_sequence_csv)
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
        item.update({f"role_seq__{k}": v for k, v in role_lookup.get(row["event_key"], {}).items()})
        item.update({f"weak_seq__{k}": v for k, v in weak_lookup.get(row["event_key"], {}).items()})
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

    role_feature_names = infer_prefixed_feature_names(rows, f"role_seq__{args.sequence_prefix}")
    weak_feature_names = infer_prefixed_feature_names(rows, f"weak_seq__{args.sequence_prefix}")

    bundles = {
        "market": build_dense_bundle(train_rows, val_rows, test_rows, MARKET_FEATURES),
        "controls": build_dense_bundle(train_rows, val_rows, test_rows, CONTROL_FEATURES),
        "a4": build_dense_bundle(train_rows, val_rows, test_rows, A4_STRUCTURED_FEATURES),
        "role_sequence": build_dense_bundle(train_rows, val_rows, test_rows, role_feature_names),
        "weak_sequence": build_dense_bundle(train_rows, val_rows, test_rows, weak_feature_names),
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
        "residual_market_plus_controls": ["market", "controls"],
        "residual_market_controls_plus_a4": ["market", "controls", "a4"],
        "residual_market_controls_plus_a4_plus_qna_lsa": ["market", "controls", "a4", "qna_lsa"],
        "residual_market_controls_plus_a4_plus_role_sequence": ["market", "controls", "a4", "role_sequence"],
        "residual_market_controls_plus_a4_plus_weak_sequence": ["market", "controls", "a4", "weak_sequence"],
        "residual_market_controls_plus_a4_plus_qna_lsa_plus_role_sequence": [
            "market",
            "controls",
            "a4",
            "qna_lsa",
            "role_sequence",
        ],
        "residual_market_controls_plus_a4_plus_qna_lsa_plus_weak_sequence": [
            "market",
            "controls",
            "a4",
            "qna_lsa",
            "weak_sequence",
        ],
    }

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted({item.strip() for item in args.include_regimes.split(",") if item.strip()}),
        "exclude_html_flags": sorted(
            {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
        ),
        "sequence_prefix": args.sequence_prefix,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "feature_groups": {
            "market": MARKET_FEATURES,
            "controls": CONTROL_FEATURES,
            "a4": A4_STRUCTURED_FEATURES,
            "role_sequence_count": len(role_feature_names),
            "weak_sequence_count": len(weak_feature_names),
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
        ("residual_market_plus_controls", "residual_market_controls_plus_a4"),
        ("residual_market_controls_plus_a4", "residual_market_controls_plus_a4_plus_qna_lsa"),
        ("residual_market_controls_plus_a4", "residual_market_controls_plus_a4_plus_role_sequence"),
        ("residual_market_controls_plus_a4", "residual_market_controls_plus_a4_plus_weak_sequence"),
        (
            "residual_market_controls_plus_a4_plus_qna_lsa",
            "residual_market_controls_plus_a4_plus_qna_lsa_plus_role_sequence",
        ),
        (
            "residual_market_controls_plus_a4_plus_qna_lsa",
            "residual_market_controls_plus_a4_plus_qna_lsa_plus_weak_sequence",
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

    write_json(output_dir / "afterhours_a4_extensions_summary.json", summary)
    write_csv(output_dir / "afterhours_a4_extensions_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
