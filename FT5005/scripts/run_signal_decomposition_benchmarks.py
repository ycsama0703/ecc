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

from dj30_qc_utils import write_json
from run_dense_multimodal_ablation_baselines import (
    build_text_lsa_bundle,
    infer_audio_feature_names,
    infer_prefixed_feature_names,
    load_joined_rows,
)
from run_prior_augmented_tabular_baselines import attach_ticker_prior
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets


MARKET_FEATURES = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "pre_60m_volume_sum",
    "within_call_rv",
    "within_call_vw_rv",
    "within_call_volume_sum",
]

CONTROL_FEATURES = [
    "call_duration_min",
    "scheduled_hour_et",
    "revenue_surprise_pct",
    "ebitda_surprise_pct",
    "eps_gaap_surprise_pct",
    "analyst_eps_norm_num_est",
    "analyst_eps_norm_std",
    "analyst_revenue_num_est",
    "analyst_revenue_std",
    "analyst_net_income_num_est",
    "analyst_net_income_std",
]

ECC_STRUCTURED_FEATURES = [
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
    "a4_kept_rows_for_duration",
    "a4_median_match_score",
    "a4_strict_row_share",
    "a4_broad_row_share",
    "a4_hard_fail_rows",
]

ECC_EXTRA_NON_AUDIO_FEATURES = [
    "full_word_count",
    "qna_word_count",
    "question_word_count",
    "answer_word_count",
    "presenter_word_count",
    "operator_word_count",
    "full_sentence_count",
    "qna_sentence_count",
    "question_mark_count",
    "digit_char_share",
    "qna_word_share",
    "presenter_word_share",
    "operator_word_share",
    "answer_to_question_word_ratio",
    "avg_words_per_question",
    "avg_words_per_answer",
    "speaker_turn_concentration",
    "speaker_name_count_from_text",
    "question_mark_per_1k_words",
    "guidance_term_rate_full",
    "guidance_term_rate_qna",
    "guidance_term_rate_presenter",
    "guidance_term_rate_answer",
    "guidance_term_rate_question",
    "uncertainty_term_rate_full",
    "uncertainty_term_rate_qna",
    "uncertainty_term_rate_presenter",
    "uncertainty_term_rate_answer",
    "uncertainty_term_rate_question",
    "positive_term_rate_full",
    "positive_term_rate_qna",
    "positive_term_rate_presenter",
    "positive_term_rate_answer",
    "positive_term_rate_question",
    "negative_term_rate_full",
    "negative_term_rate_qna",
    "negative_term_rate_presenter",
    "negative_term_rate_answer",
    "negative_term_rate_question",
    "qa_pair_count",
    "qa_pair_overlap_mean",
    "qa_pair_overlap_median",
    "qa_pair_low_overlap_share",
    "qa_pair_question_words_mean",
    "qa_pair_answer_words_mean",
    "qa_pair_answer_digit_rate_mean",
    "qa_pair_answer_hedge_rate_mean",
    "qa_pair_answer_assertive_rate_mean",
    "qa_pair_answer_forward_rate_mean",
    "qa_multi_part_question_share",
    "qa_evasive_proxy_share",
    "qna_vs_presenter_uncertainty_gap",
    "qna_vs_presenter_guidance_gap",
    "answer_vs_question_uncertainty_gap",
    "answer_vs_question_negative_gap",
    "a4_strict_segment_count",
    "a4_strict_duration_sum_sec",
    "a4_strict_duration_mean_sec",
    "a4_strict_duration_median_sec",
    "a4_strict_duration_std_sec",
    "a4_strict_gap_mean_sec",
    "a4_strict_gap_max_sec",
    "a4_strict_span_sec",
    "a4_strict_high_conf_share",
    "a4_strict_overlap_warn_share",
    "a4_broad_segment_count",
    "a4_broad_duration_sum_sec",
    "a4_broad_duration_mean_sec",
    "a4_broad_duration_median_sec",
    "a4_broad_duration_std_sec",
    "a4_broad_gap_mean_sec",
    "a4_broad_gap_max_sec",
    "a4_broad_span_sec",
    "a4_broad_high_conf_share",
    "a4_broad_overlap_warn_share",
]

AUDIO_PROXY_FEATURES = [
    "has_audio_file",
    "a3_file_size_bytes",
    "a3_log_file_size",
    "a3_bytes_per_call_sec",
    "a3_bytes_per_strict_span_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose prior-aware signal into market, ECC text/timing, and audio bundles."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/signal_decomposition_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="")
    parser.add_argument("--exclude-html-flags", default="")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-8)
    return parser.parse_args()


def regime_label(row: dict[str, str]) -> str:
    hour = float(row.get("scheduled_hour_et", 0.0))
    if hour < 9.5:
        return "pre_market"
    if hour < 16.0:
        return "market_hours"
    return "after_hours"


def bundle_name_from_regimes(include_regimes: set[str]) -> str:
    if not include_regimes:
        return "all_regimes"
    return "-".join(sorted(include_regimes))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    rows = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.audio_csv,
        "post_call_60m_rv",
        args.qa_csv,
    )

    filtered_rows = []
    for row in rows:
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        if html_flag in exclude_html_flags:
            continue
        regime = regime_label(row)
        if include_regimes and regime not in include_regimes:
            continue
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        item["_regime"] = regime
        filtered_rows.append(item)

    filtered_rows = attach_ticker_prior(filtered_rows, args.train_end_year)
    train_rows = [row for row in filtered_rows if row["_year"] <= args.train_end_year]
    val_rows = [row for row in filtered_rows if row["_year"] == args.val_year]
    test_rows = [row for row in filtered_rows if row["_year"] > args.val_year]
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    qa_feature_names = infer_prefixed_feature_names(filtered_rows, "qa_bench_")
    real_audio_feature_names = infer_audio_feature_names(filtered_rows)

    bundles = {
        "market": build_dense_bundle(train_rows, val_rows, test_rows, MARKET_FEATURES),
        "controls": build_dense_bundle(train_rows, val_rows, test_rows, CONTROL_FEATURES),
        "ecc_structured": build_dense_bundle(train_rows, val_rows, test_rows, ECC_STRUCTURED_FEATURES),
        "ecc_extra_non_audio": build_dense_bundle(train_rows, val_rows, test_rows, ECC_EXTRA_NON_AUDIO_FEATURES),
        "audio_proxy": build_dense_bundle(train_rows, val_rows, test_rows, AUDIO_PROXY_FEATURES),
        "qa_benchmark": build_dense_bundle(train_rows, val_rows, test_rows, qa_feature_names),
        "real_audio": build_dense_bundle(train_rows, val_rows, test_rows, real_audio_feature_names),
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
        "market_only": ["market"],
        "market_plus_controls": ["market", "controls"],
        "qa_benchmark_only": ["qa_benchmark"],
        "qa_benchmark_plus_qna_lsa": ["qa_benchmark", "qna_lsa"],
        "ecc_text_timing_only": ["ecc_structured", "ecc_extra_non_audio", "qa_benchmark", "qna_lsa"],
        "ecc_text_timing_plus_audio": [
            "ecc_structured",
            "ecc_extra_non_audio",
            "qa_benchmark",
            "qna_lsa",
            "audio_proxy",
            "real_audio",
        ],
        "market_controls_plus_qa_benchmark": ["market", "controls", "qa_benchmark"],
        "market_controls_plus_qa_benchmark_plus_qna_lsa": ["market", "controls", "qa_benchmark", "qna_lsa"],
        "market_controls_plus_ecc_text_timing": [
            "market",
            "controls",
            "ecc_structured",
            "ecc_extra_non_audio",
            "qa_benchmark",
            "qna_lsa",
        ],
        "market_controls_plus_ecc_plus_audio": [
            "market",
            "controls",
            "ecc_structured",
            "ecc_extra_non_audio",
            "qa_benchmark",
            "qna_lsa",
            "audio_proxy",
            "real_audio",
        ],
    }

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted(include_regimes),
        "exclude_html_flags": sorted(exclude_html_flags),
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "feature_groups": {
            "market": MARKET_FEATURES,
            "controls": CONTROL_FEATURES,
            "ecc_structured": ECC_STRUCTURED_FEATURES,
            "ecc_extra_non_audio": ECC_EXTRA_NON_AUDIO_FEATURES,
            "audio_proxy": AUDIO_PROXY_FEATURES,
            "qa_benchmark": qa_feature_names,
            "real_audio": real_audio_feature_names,
            "qna_lsa_components": args.lsa_components,
        },
        "models": {},
    }

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
            continue

        train_parts = []
        val_parts = []
        test_parts = []
        feature_count = 0
        for bundle_name in bundle_names:
            bundle = bundles[bundle_name]
            train_parts.append(bundle["train"])
            val_parts.append(bundle["val"])
            test_parts.append(bundle["test"])
            feature_count += int(bundle["train"].shape[1])

        train_x = np.hstack(train_parts)
        val_x = np.hstack(val_parts)
        test_x = np.hstack(test_parts)

        best_alpha, best_model, pred_val = fit_residual_ridge(
            train_x, train_prior, train_y, val_x, val_prior, val_y, alphas
        )
        pred_test = test_prior + best_model.predict(test_x)
        summary["models"][model_name] = {
            "family": "residual_ridge",
            "feature_bundles": bundle_names,
            "feature_count": feature_count,
            "best_alpha": best_alpha,
            "val": metrics(val_y, pred_val),
            "test": metrics(test_y, pred_test),
    }

    regime_suffix = bundle_name_from_regimes(include_regimes)
    html_suffix = "all_html" if not exclude_html_flags else "exclude-" + "-".join(sorted(exclude_html_flags))
    output_path = output_dir / f"signal_decomposition_{args.target_variant}_{regime_suffix}_{html_suffix}.json"
    write_json(output_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
