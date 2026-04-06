#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from dj30_qc_utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep Q&A LSA bottleneck sizes for the clean after-hours pre-call semantic mainline."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_precall_bottleneck_sweep_clean_real"),
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
    parser.add_argument("--lsa-components-list", default="4,8,16,32,64")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_command(cmd: list[str], workdir: Path) -> None:
    completed = subprocess.run(cmd, cwd=workdir, check=True, capture_output=True, text=True)
    if completed.stdout.strip():
        print(completed.stdout.strip())


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[1]
    python_bin = sys.executable
    lsa_components = [int(item) for item in args.lsa_components_list.split(",") if item.strip()]

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": [item.strip() for item in args.include_regimes.split(",") if item.strip()],
        "exclude_html_flags": [item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()],
        "lsa_components": lsa_components,
        "runs": {},
    }

    common_args = [
        "--panel-csv",
        str(args.panel_csv),
        "--features-csv",
        str(args.features_csv),
        "--audio-csv",
        str(args.audio_csv),
        "--qa-csv",
        str(args.qa_csv),
        "--include-regimes",
        args.include_regimes,
        "--exclude-html-flags",
        args.exclude_html_flags,
        "--target-variant",
        args.target_variant,
        "--train-end-year",
        str(args.train_end_year),
        "--val-year",
        str(args.val_year),
        "--alphas",
        args.alphas,
        "--eps",
        str(args.eps),
        "--max-features",
        str(args.max_features),
        "--min-df",
        str(args.min_df),
    ]

    for lsa in lsa_components:
        ladder_dir = output_dir / f"ladder_lsa{lsa}"
        unseen_dir = output_dir / f"unseen_lsa{lsa}"
        run_command(
            [
                python_bin,
                "scripts/run_afterhours_precall_semantic_ladder.py",
                *common_args,
                "--output-dir",
                str(ladder_dir),
                "--lsa-components",
                str(lsa),
                "--bootstrap-iters",
                str(args.bootstrap_iters),
                "--perm-iters",
                str(args.perm_iters),
                "--seed",
                str(args.seed),
            ],
            repo_root,
        )
        run_command(
            [
                python_bin,
                "scripts/run_afterhours_precall_unseen_ticker.py",
                *common_args,
                "--output-dir",
                str(unseen_dir),
                "--lsa-components",
                str(lsa),
            ],
            repo_root,
        )

        ladder_summary = load_json(ladder_dir / "afterhours_precall_semantic_ladder_summary.json")
        unseen_summary = load_json(unseen_dir / "afterhours_precall_unseen_ticker_summary.json")
        summary["runs"][str(lsa)] = {
            "fixed_split": {
                "pre_call_market_only_r2": ladder_summary["models"]["residual_pre_call_market_only"]["test"]["r2"],
                "pre_call_market_plus_a4_r2": ladder_summary["models"]["residual_pre_call_market_plus_a4"]["test"]["r2"],
                "pre_call_market_plus_a4_plus_qna_lsa_r2": ladder_summary["models"][
                    "residual_pre_call_market_plus_a4_plus_qna_lsa"
                ]["test"]["r2"],
                "pre_call_market_plus_controls_plus_a4_plus_qna_lsa_r2": ladder_summary["models"][
                    "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa"
                ]["test"]["r2"],
            },
            "unseen_ticker": {
                "pre_call_market_only_r2": unseen_summary["overall"]["residual_pre_call_market_only"]["r2"],
                "pre_call_market_plus_a4_r2": unseen_summary["overall"]["residual_pre_call_market_plus_a4"]["r2"],
                "pre_call_market_plus_a4_plus_qna_lsa_r2": unseen_summary["overall"][
                    "residual_pre_call_market_plus_a4_plus_qna_lsa"
                ]["r2"],
                "pre_call_market_plus_controls_plus_a4_plus_qna_lsa_r2": unseen_summary["overall"][
                    "residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa"
                ]["r2"],
                "median_ticker_pre_call_market_plus_a4_plus_qna_lsa_r2": unseen_summary["median_ticker_r2"][
                    "residual_pre_call_market_plus_a4_plus_qna_lsa"
                ],
                "median_ticker_pre_call_market_plus_controls_plus_a4_plus_qna_lsa_r2": unseen_summary[
                    "median_ticker_r2"
                ]["residual_pre_call_market_plus_controls_plus_a4_plus_qna_lsa"],
            },
        }

    best_fixed = max(
        summary["runs"].items(),
        key=lambda item: item[1]["fixed_split"]["pre_call_market_plus_a4_plus_qna_lsa_r2"],
    )
    best_unseen = max(
        summary["runs"].items(),
        key=lambda item: item[1]["unseen_ticker"]["pre_call_market_plus_a4_plus_qna_lsa_r2"],
    )
    summary["best_by_setting"] = {
        "fixed_split_pre_call_market_plus_a4_plus_qna_lsa": {
            "lsa_components": int(best_fixed[0]),
            "r2": best_fixed[1]["fixed_split"]["pre_call_market_plus_a4_plus_qna_lsa_r2"],
        },
        "unseen_ticker_pre_call_market_plus_a4_plus_qna_lsa": {
            "lsa_components": int(best_unseen[0]),
            "r2": best_unseen[1]["unseen_ticker"]["pre_call_market_plus_a4_plus_qna_lsa_r2"],
        },
    }
    write_json(output_dir / "afterhours_precall_bottleneck_sweep_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
