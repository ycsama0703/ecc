#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import site
import subprocess
import sys
from pathlib import Path

import numpy as np

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

from dj30_qc_utils import write_csv, write_json
from run_afterhours_transfer_expert_selection import summarize_significance
from run_structured_baselines import metrics


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

SPLITS = [
    {"name": "val2020_test_post2020", "train_end_year": 2019, "val_year": 2020},
    {"name": "val2021_test_post2021", "train_end_year": 2020, "val_year": 2021},
    {"name": "val2022_test_post2022", "train_end_year": 2021, "val_year": 2022},
]

MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_RETAINED = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate"
MODEL_SELECTED = "validation_selected_transfer_expert"
MODEL_PAIR_TREE = "conservative_tree_override_on_selected_expert"
MODEL_PLUS_TEXT_LOGISTIC = "conservative_logistic_override_on_selected_expert"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal confirmation benchmark for the strongest hybrid conservative transfer-router routes."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--old-audio-csv", type=Path, required=True)
    parser.add_argument("--aligned-audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="fail")
    parser.add_argument("--lsa-components", type=int, default=4)
    parser.add_argument("--qa-compressed-components", type=int, default=8)
    parser.add_argument("--aligned-compressed-components", type=int, default=8)
    parser.add_argument("--tree-depth", type=int, default=2)
    parser.add_argument("--tree-min-leaf-frac", type=float, default=0.08)
    parser.add_argument("--logistic-c", type=float, default=2.0)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[str, dict[str, float]]:
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = {}
        for row in reader:
            event_key = row["event_key"]
            rows[event_key] = row
    return rows


def ordered_arrays(
    rows: dict[str, dict[str, float]],
    model_names: list[str],
) -> tuple[list[str], np.ndarray, dict[str, np.ndarray]]:
    keys = sorted(rows)
    y_true = np.asarray([float(rows[key]["target"]) for key in keys], dtype=float)
    preds = {
        model_name: np.asarray([float(rows[key][model_name]) for key in keys], dtype=float)
        for model_name in model_names
    }
    return keys, y_true, preds


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

    split_rows = []
    split_summaries = {}

    for split in SPLITS:
        split_name = split["name"]

        pair_dir = output_dir / f"{split_name}__hybrid_pair_bench_tree"
        pair_cmd = [
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
            str(pair_dir),
            "--target-variant",
            args.target_variant,
            "--include-regimes",
            args.include_regimes,
            "--exclude-html-flags",
            args.exclude_html_flags,
            "--train-end-year",
            str(split["train_end_year"]),
            "--val-year",
            str(split["val_year"]),
            "--lsa-components",
            str(args.lsa_components),
            "--qa-compressed-components",
            str(args.qa_compressed_components),
            "--aligned-compressed-components",
            str(args.aligned_compressed_components),
            "--router-c",
            "1.0",
            "--tree-depth",
            str(args.tree_depth),
            "--tree-min-leaf-frac",
            str(args.tree_min_leaf_frac),
            "--router-features",
            ",".join(HYBRID_PAIR_BENCH_FEATURES),
        ]
        subprocess.run(pair_cmd, cwd=repo_root, env=env, check=True, stdout=subprocess.DEVNULL)

        plus_text_dir = output_dir / f"{split_name}__hybrid_plus_text_logistic"
        plus_text_cmd = [
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
            str(plus_text_dir),
            "--target-variant",
            args.target_variant,
            "--include-regimes",
            args.include_regimes,
            "--exclude-html-flags",
            args.exclude_html_flags,
            "--train-end-year",
            str(split["train_end_year"]),
            "--val-year",
            str(split["val_year"]),
            "--lsa-components",
            str(args.lsa_components),
            "--qa-compressed-components",
            str(args.qa_compressed_components),
            "--aligned-compressed-components",
            str(args.aligned_compressed_components),
            "--router-c",
            str(args.logistic_c),
            "--tree-depth",
            str(args.tree_depth),
            "--tree-min-leaf-frac",
            str(args.tree_min_leaf_frac),
            "--router-features",
            ",".join(HYBRID_PLUS_TEXT_FEATURES),
        ]
        subprocess.run(plus_text_cmd, cwd=repo_root, env=env, check=True, stdout=subprocess.DEVNULL)

        pair_summary = json.loads((pair_dir / "afterhours_transfer_conservative_router_summary.json").read_text())
        plus_text_summary = json.loads((plus_text_dir / "afterhours_transfer_conservative_router_summary.json").read_text())

        pair_predictions = load_predictions(pair_dir / "afterhours_transfer_conservative_router_predictions.csv")
        plus_text_predictions = load_predictions(plus_text_dir / "afterhours_transfer_conservative_router_predictions.csv")

        shared_keys = sorted(set(pair_predictions) & set(plus_text_predictions))
        if not shared_keys:
            raise SystemExit(f"no shared prediction rows for split {split_name}")

        merged = {}
        for key in shared_keys:
            pair_row = pair_predictions[key]
            plus_row = plus_text_predictions[key]
            merged[key] = {
                "target": pair_row["target"],
                MODEL_PRE_ONLY: pair_row[MODEL_PRE_ONLY],
                MODEL_RETAINED: pair_row[MODEL_RETAINED],
                MODEL_SELECTED: pair_row[MODEL_SELECTED],
                MODEL_PAIR_TREE: pair_row[MODEL_PAIR_TREE],
                MODEL_PLUS_TEXT_LOGISTIC: plus_row[MODEL_PLUS_TEXT_LOGISTIC],
            }

        _, y_true, preds = ordered_arrays(
            merged,
            [
                MODEL_PRE_ONLY,
                MODEL_RETAINED,
                MODEL_SELECTED,
                MODEL_PAIR_TREE,
                MODEL_PLUS_TEXT_LOGISTIC,
            ],
        )

        overall = {model_name: metrics(y_true, preds[model_name]) for model_name in preds}
        significance = {
            f"{MODEL_PAIR_TREE}__vs__{MODEL_RETAINED}": summarize_significance(
                y_true,
                preds[MODEL_RETAINED],
                preds[MODEL_PAIR_TREE],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_PAIR_TREE}__vs__{MODEL_SELECTED}": summarize_significance(
                y_true,
                preds[MODEL_SELECTED],
                preds[MODEL_PAIR_TREE],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_PAIR_TREE}__vs__{MODEL_PRE_ONLY}": summarize_significance(
                y_true,
                preds[MODEL_PRE_ONLY],
                preds[MODEL_PAIR_TREE],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_PLUS_TEXT_LOGISTIC}__vs__{MODEL_RETAINED}": summarize_significance(
                y_true,
                preds[MODEL_RETAINED],
                preds[MODEL_PLUS_TEXT_LOGISTIC],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_PLUS_TEXT_LOGISTIC}__vs__{MODEL_SELECTED}": summarize_significance(
                y_true,
                preds[MODEL_SELECTED],
                preds[MODEL_PLUS_TEXT_LOGISTIC],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_PLUS_TEXT_LOGISTIC}__vs__{MODEL_PRE_ONLY}": summarize_significance(
                y_true,
                preds[MODEL_PRE_ONLY],
                preds[MODEL_PLUS_TEXT_LOGISTIC],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_PLUS_TEXT_LOGISTIC}__vs__{MODEL_PAIR_TREE}": summarize_significance(
                y_true,
                preds[MODEL_PAIR_TREE],
                preds[MODEL_PLUS_TEXT_LOGISTIC],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
        }

        split_summary = {
            "split": split,
            "overall_test_size": len(shared_keys),
            "models": overall,
            "significance": significance,
            "pair_tree_source": {
                "router_features": HYBRID_PAIR_BENCH_FEATURES,
                "tree_depth": args.tree_depth,
                "tree_min_leaf_frac": args.tree_min_leaf_frac,
            },
            "plus_text_logistic_source": {
                "router_features": HYBRID_PLUS_TEXT_FEATURES,
                "router_c": args.logistic_c,
                "tree_depth": args.tree_depth,
                "tree_min_leaf_frac": args.tree_min_leaf_frac,
            },
        }
        split_summaries[split_name] = split_summary

        split_rows.append(
            {
                "split": split_name,
                "train_end_year": split["train_end_year"],
                "val_year": split["val_year"],
                "test_start_year": split["val_year"] + 1,
                "test_size": len(shared_keys),
                "pre_only_r2": overall[MODEL_PRE_ONLY]["r2"],
                "retained_r2": overall[MODEL_RETAINED]["r2"],
                "selected_r2": overall[MODEL_SELECTED]["r2"],
                "hybrid_pair_tree_r2": overall[MODEL_PAIR_TREE]["r2"],
                "hybrid_plus_text_logistic_r2": overall[MODEL_PLUS_TEXT_LOGISTIC]["r2"],
                "hybrid_pair_tree_p_vs_retained": significance[f"{MODEL_PAIR_TREE}__vs__{MODEL_RETAINED}"][
                    "mse_gain_pvalue"
                ],
                "hybrid_pair_tree_p_vs_selected": significance[f"{MODEL_PAIR_TREE}__vs__{MODEL_SELECTED}"][
                    "mse_gain_pvalue"
                ],
                "hybrid_pair_tree_p_vs_pre": significance[f"{MODEL_PAIR_TREE}__vs__{MODEL_PRE_ONLY}"][
                    "mse_gain_pvalue"
                ],
                "hybrid_plus_text_logistic_p_vs_retained": significance[
                    f"{MODEL_PLUS_TEXT_LOGISTIC}__vs__{MODEL_RETAINED}"
                ]["mse_gain_pvalue"],
                "hybrid_plus_text_logistic_p_vs_selected": significance[
                    f"{MODEL_PLUS_TEXT_LOGISTIC}__vs__{MODEL_SELECTED}"
                ]["mse_gain_pvalue"],
                "hybrid_plus_text_logistic_p_vs_pre": significance[
                    f"{MODEL_PLUS_TEXT_LOGISTIC}__vs__{MODEL_PRE_ONLY}"
                ]["mse_gain_pvalue"],
                "hybrid_plus_text_logistic_p_vs_pair_tree": significance[
                    f"{MODEL_PLUS_TEXT_LOGISTIC}__vs__{MODEL_PAIR_TREE}"
                ]["mse_gain_pvalue"],
            }
        )

    confirmation = {
        "config": {
            "target_variant": args.target_variant,
            "include_regimes": args.include_regimes,
            "exclude_html_flags": args.exclude_html_flags,
            "lsa_components": args.lsa_components,
            "qa_compressed_components": args.qa_compressed_components,
            "aligned_compressed_components": args.aligned_compressed_components,
            "pair_tree_depth": args.tree_depth,
            "pair_tree_min_leaf_frac": args.tree_min_leaf_frac,
            "plus_text_logistic_c": args.logistic_c,
        },
        "splits": split_rows,
        "details": split_summaries,
        "win_counts": {
            "hybrid_pair_tree_beats_selected": sum(
                1 for row in split_rows if row["hybrid_pair_tree_r2"] > row["selected_r2"]
            ),
            "hybrid_pair_tree_beats_retained": sum(
                1 for row in split_rows if row["hybrid_pair_tree_r2"] > row["retained_r2"]
            ),
            "hybrid_plus_text_logistic_beats_selected": sum(
                1 for row in split_rows if row["hybrid_plus_text_logistic_r2"] > row["selected_r2"]
            ),
            "hybrid_plus_text_logistic_beats_retained": sum(
                1 for row in split_rows if row["hybrid_plus_text_logistic_r2"] > row["retained_r2"]
            ),
            "hybrid_plus_text_logistic_beats_pair_tree": sum(
                1 for row in split_rows if row["hybrid_plus_text_logistic_r2"] > row["hybrid_pair_tree_r2"]
            ),
        },
    }

    write_csv(output_dir / "afterhours_transfer_router_temporal_confirmation_overview.csv", split_rows)
    write_json(output_dir / "afterhours_transfer_router_temporal_confirmation_summary.json", confirmation)
    print(json.dumps(confirmation, indent=2))


if __name__ == "__main__":
    main()
