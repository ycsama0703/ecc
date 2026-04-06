#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import site
import subprocess
import sys
from pathlib import Path

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

from dj30_qc_utils import write_csv, write_json


FAMILY_FEATURES = {
    "lite_baseline": [
        "a4_strict_row_share",
        "a4_strict_high_conf_share",
        "qa_pair_count",
        "qa_bench_direct_answer_share",
        "qa_bench_evasion_score_mean",
        "qa_bench_coverage_mean",
        "aligned_audio__aligned_audio_sentence_count",
    ],
    "pair_core": [
        "a4_strict_row_share",
        "a4_strict_high_conf_share",
        "qa_pair_count",
        "qa_pair_low_overlap_share",
        "qa_pair_question_words_mean",
        "qa_pair_answer_words_mean",
        "qa_pair_answer_digit_rate_mean",
        "qa_pair_answer_hedge_rate_mean",
        "qa_pair_answer_assertive_rate_mean",
        "qa_pair_answer_forward_rate_mean",
        "qa_multi_part_question_share",
        "qa_evasive_proxy_share",
        "aligned_audio__aligned_audio_sentence_count",
    ],
    "bench_directness": [
        "a4_strict_row_share",
        "a4_strict_high_conf_share",
        "qa_bench_direct_answer_share",
        "qa_bench_direct_early_score_mean",
        "qa_bench_evasion_score_mean",
        "qa_bench_high_evasion_share",
        "qa_bench_coverage_mean",
        "qa_bench_nonresponse_share",
        "qa_bench_short_evasive_share",
        "qa_bench_topic_drift_share",
        "aligned_audio__aligned_audio_sentence_count",
    ],
    "hybrid_pair_bench": [
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
    "hybrid_plus_text": [
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
        "answer_to_question_word_ratio",
        "qna_word_count",
        "question_mark_per_1k_words",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark conservative transfer-router signal families on the matched after-hours transfer slice."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_router_signal_family_benchmark_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="fail")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--lsa-components", type=int, default=4)
    parser.add_argument("--qa-compressed-components", type=int, default=8)
    parser.add_argument("--aligned-compressed-components", type=int, default=8)
    parser.add_argument("--router-c", type=float, default=1.0)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--tree-min-leaf-frac", type=float, default=0.08)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"scripts{os.pathsep}{existing}" if existing else "scripts"

    conservative_script = repo_root / "scripts" / "run_afterhours_transfer_conservative_router.py"
    python_bin = sys.executable

    rows = []
    family_summaries = {}

    for family_name, features in FAMILY_FEATURES.items():
        family_dir = output_dir / family_name
        cmd = [
            python_bin,
            str(conservative_script),
            "--panel-csv",
            str(args.panel_csv),
            "--features-csv",
            str(args.features_csv),
            "--old-audio-csv",
            str(args.old_audio_csv),
            "--aligned-audio-csv",
            str(args.aligned_audio_csv),
            "--qa-csv",
            str(args.qa_csv),
            "--output-dir",
            str(family_dir),
            "--target-variant",
            args.target_variant,
            "--include-regimes",
            args.include_regimes,
            "--exclude-html-flags",
            args.exclude_html_flags,
            "--train-end-year",
            str(args.train_end_year),
            "--val-year",
            str(args.val_year),
            "--lsa-components",
            str(args.lsa_components),
            "--qa-compressed-components",
            str(args.qa_compressed_components),
            "--aligned-compressed-components",
            str(args.aligned_compressed_components),
            "--router-c",
            str(args.router_c),
            "--tree-depth",
            str(args.tree_depth),
            "--tree-min-leaf-frac",
            str(args.tree_min_leaf_frac),
            "--router-features",
            ",".join(features),
        ]
        subprocess.run(cmd, cwd=repo_root, env=env, check=True, stdout=subprocess.DEVNULL)

        summary_path = family_dir / "afterhours_transfer_conservative_router_summary.json"
        family_summary = json.loads(summary_path.read_text())
        family_summaries[family_name] = family_summary
        rows.append(
            {
                "family": family_name,
                "feature_count": len(features),
                "features": "|".join(features),
                "tree_r2": family_summary["overall"]["conservative_tree_override_on_selected_expert"]["r2"],
                "tree_rmse": family_summary["overall"]["conservative_tree_override_on_selected_expert"]["rmse"],
                "tree_mae": family_summary["overall"]["conservative_tree_override_on_selected_expert"]["mae"],
                "tree_median_ticker_r2": family_summary["median_ticker_r2"]["conservative_tree_override_on_selected_expert"],
                "tree_p_mse_vs_selected": family_summary["significance"][
                    "validation_selected_transfer_expert__vs__conservative_tree_override_on_selected_expert"
                ]["mse_gain_pvalue"],
                "tree_p_mse_vs_retained": family_summary["significance"][
                    "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate__vs__conservative_tree_override_on_selected_expert"
                ]["mse_gain_pvalue"],
                "logistic_r2": family_summary["overall"]["conservative_logistic_override_on_selected_expert"]["r2"],
                "logistic_rmse": family_summary["overall"]["conservative_logistic_override_on_selected_expert"]["rmse"],
                "logistic_mae": family_summary["overall"]["conservative_logistic_override_on_selected_expert"]["mae"],
                "logistic_median_ticker_r2": family_summary["median_ticker_r2"][
                    "conservative_logistic_override_on_selected_expert"
                ],
                "logistic_p_mse_vs_selected": family_summary["significance"][
                    "validation_selected_transfer_expert__vs__conservative_logistic_override_on_selected_expert"
                ]["mse_gain_pvalue"],
                "logistic_p_mse_vs_retained": family_summary["significance"][
                    "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate__vs__conservative_logistic_override_on_selected_expert"
                ]["mse_gain_pvalue"],
            }
        )

    best_tree = max(rows, key=lambda row: row["tree_r2"])
    best_logistic = max(rows, key=lambda row: row["logistic_r2"])

    aggregate = {
        "config": {
            "target_variant": args.target_variant,
            "include_regimes": args.include_regimes,
            "exclude_html_flags": args.exclude_html_flags,
            "train_end_year": args.train_end_year,
            "val_year": args.val_year,
            "lsa_components": args.lsa_components,
            "qa_compressed_components": args.qa_compressed_components,
            "aligned_compressed_components": args.aligned_compressed_components,
            "router_c": args.router_c,
            "tree_depth": args.tree_depth,
            "tree_min_leaf_frac": args.tree_min_leaf_frac,
        },
        "families": rows,
        "best_tree_family": best_tree,
        "best_logistic_family": best_logistic,
        "baseline_reference_family": "lite_baseline",
    }

    write_csv(output_dir / "afterhours_transfer_router_signal_family_overview.csv", rows)
    write_json(output_dir / "afterhours_transfer_router_signal_family_summary.json", aggregate)
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
