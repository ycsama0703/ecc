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


HYBRID_PAIR_BENCH_FEATURES = [
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
]

HYBRID_PLUS_TEXT_FEATURES = HYBRID_PAIR_BENCH_FEATURES + [
    "answer_to_question_word_ratio",
    "qna_word_count",
    "question_mark_per_1k_words",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune the strongest current hybrid conservative transfer-router families."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_router_topfamily_tuning_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="fail")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--lsa-components", type=int, default=4)
    parser.add_argument("--qa-compressed-components", type=int, default=8)
    parser.add_argument("--aligned-compressed-components", type=int, default=8)
    parser.add_argument("--tree-depths", default="1,2,3,4")
    parser.add_argument("--tree-min-leaf-fracs", default="0.05,0.08,0.12,0.16")
    parser.add_argument("--logistic-c-grid", default="0.1,0.25,0.5,1,2,4,8")
    return parser.parse_args()


def parse_ints(raw: str) -> list[int]:
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


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

    tree_depths = parse_ints(args.tree_depths)
    tree_min_leaf_fracs = parse_floats(args.tree_min_leaf_fracs)
    logistic_c_grid = parse_floats(args.logistic_c_grid)

    rows = []

    for depth in tree_depths:
        for min_leaf_frac in tree_min_leaf_fracs:
            run_name = f"hybrid_pair_bench_tree_d{depth}_l{str(min_leaf_frac).replace('.', 'p')}"
            run_dir = output_dir / run_name
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
                str(run_dir),
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
                "1.0",
                "--tree-depth",
                str(depth),
                "--tree-min-leaf-frac",
                str(min_leaf_frac),
                "--router-features",
                ",".join(HYBRID_PAIR_BENCH_FEATURES),
            ]
            subprocess.run(cmd, cwd=repo_root, env=env, check=True, stdout=subprocess.DEVNULL)
            summary = json.loads((run_dir / "afterhours_transfer_conservative_router_summary.json").read_text())
            rows.append(
                {
                    "family": "hybrid_pair_bench_tree",
                    "config_name": run_name,
                    "tree_depth": depth,
                    "tree_min_leaf_frac": min_leaf_frac,
                    "router_c": 1.0,
                    "tree_r2": summary["overall"]["conservative_tree_override_on_selected_expert"]["r2"],
                    "tree_rmse": summary["overall"]["conservative_tree_override_on_selected_expert"]["rmse"],
                    "tree_mae": summary["overall"]["conservative_tree_override_on_selected_expert"]["mae"],
                    "tree_p_mse_vs_selected": summary["significance"][
                        "validation_selected_transfer_expert__vs__conservative_tree_override_on_selected_expert"
                    ]["mse_gain_pvalue"],
                    "tree_p_mse_vs_retained": summary["significance"][
                        "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate__vs__conservative_tree_override_on_selected_expert"
                    ]["mse_gain_pvalue"],
                }
            )

    for c_value in logistic_c_grid:
        run_name = f"hybrid_plus_text_logistic_c{str(c_value).replace('.', 'p')}"
        run_dir = output_dir / run_name
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
            str(run_dir),
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
            str(c_value),
            "--tree-depth",
            "2",
            "--router-features",
            ",".join(HYBRID_PLUS_TEXT_FEATURES),
        ]
        subprocess.run(cmd, cwd=repo_root, env=env, check=True, stdout=subprocess.DEVNULL)
        summary = json.loads((run_dir / "afterhours_transfer_conservative_router_summary.json").read_text())
        rows.append(
            {
                "family": "hybrid_plus_text_logistic",
                "config_name": run_name,
                "tree_depth": 2,
                "tree_min_leaf_frac": summary["config"]["tree_min_leaf_frac"],
                "router_c": c_value,
                "logistic_r2": summary["overall"]["conservative_logistic_override_on_selected_expert"]["r2"],
                "logistic_rmse": summary["overall"]["conservative_logistic_override_on_selected_expert"]["rmse"],
                "logistic_mae": summary["overall"]["conservative_logistic_override_on_selected_expert"]["mae"],
                "logistic_p_mse_vs_selected": summary["significance"][
                    "validation_selected_transfer_expert__vs__conservative_logistic_override_on_selected_expert"
                ]["mse_gain_pvalue"],
                "logistic_p_mse_vs_retained": summary["significance"][
                    "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate__vs__conservative_logistic_override_on_selected_expert"
                ]["mse_gain_pvalue"],
            }
        )

    best_tree = max(
        [row for row in rows if row["family"] == "hybrid_pair_bench_tree"],
        key=lambda row: row["tree_r2"],
    )
    best_logistic = max(
        [row for row in rows if row["family"] == "hybrid_plus_text_logistic"],
        key=lambda row: row["logistic_r2"],
    )

    summary = {
        "config": {
            "target_variant": args.target_variant,
            "include_regimes": args.include_regimes,
            "exclude_html_flags": args.exclude_html_flags,
            "train_end_year": args.train_end_year,
            "val_year": args.val_year,
            "lsa_components": args.lsa_components,
            "qa_compressed_components": args.qa_compressed_components,
            "aligned_compressed_components": args.aligned_compressed_components,
            "tree_depths": tree_depths,
            "tree_min_leaf_fracs": tree_min_leaf_fracs,
            "logistic_c_grid": logistic_c_grid,
        },
        "runs": rows,
        "best_tree": best_tree,
        "best_logistic": best_logistic,
    }

    write_csv(output_dir / "afterhours_transfer_router_topfamily_tuning_overview.csv", rows)
    write_json(output_dir / "afterhours_transfer_router_topfamily_tuning_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
