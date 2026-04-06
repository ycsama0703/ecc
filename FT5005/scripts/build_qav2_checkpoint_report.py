#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build a Q&A v2 research checkpoint report from saved result summaries."
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument(
        "--output-md",
        type=Path,
        default=repo_root / "docs" / "qna_signal_checkpoint_20260314.md",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=repo_root / "results" / "research_checkpoints_real" / "qna_signal_checkpoint_20260314.json",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def r2(value: float) -> str:
    return f"{value:.3f}"


def delta(current: float, previous: float) -> str:
    diff = current - previous
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.3f}"


def model_r2(summary: dict, model_name: str) -> float:
    return float(summary["models"][model_name]["test"]["r2"])


def target_variant_r2(summary: dict, model_name: str) -> float:
    return float(summary["variants"]["shock_minus_pre"]["models"][model_name]["test"]["r2"])


def regime_subset_r2(summary: dict, model_name: str) -> float:
    return float(summary[model_name]["test"]["r2"])


def regime_residual_r2(summary: dict, model_name: str) -> float:
    return float(summary[model_name]["test"]["r2"])


def build_summary(repo_root: Path) -> dict:
    results_root = repo_root / "results"
    qav1_full = load_json(results_root / "signal_decomposition_real" / "signal_decomposition_shock_minus_pre_all_regimes.json")
    qav1_full_clean = load_json(results_root / "signal_decomposition_real" / "signal_decomposition_shock_minus_pre_all_regimes_exclude-fail.json")
    qav1_off = load_json(results_root / "signal_decomposition_real" / "signal_decomposition_shock_minus_pre_after_hours-pre_market.json")
    qav1_off_clean = load_json(results_root / "signal_decomposition_real" / "signal_decomposition_shock_minus_pre_after_hours-pre_market_exclude-fail.json")

    qav2_full = load_json(results_root / "signal_decomposition_qav2_real" / "signal_decomposition_shock_minus_pre_all_regimes_all_html.json")
    qav2_full_clean = load_json(results_root / "signal_decomposition_qav2_real" / "signal_decomposition_shock_minus_pre_all_regimes_exclude-fail.json")
    qav2_off = load_json(results_root / "signal_decomposition_qav2_real" / "signal_decomposition_shock_minus_pre_after_hours-pre_market_all_html.json")
    qav2_off_clean = load_json(results_root / "signal_decomposition_qav2_real" / "signal_decomposition_shock_minus_pre_after_hours-pre_market_exclude-fail.json")

    target_qav1 = load_json(results_root / "target_variant_experiments_real" / "target_variant_summary.json")
    target_qav2 = load_json(results_root / "target_variant_experiments_qav2_real" / "target_variant_summary.json")
    subset_qav1 = load_json(results_root / "regime_subset_experiments_real" / "regime_subset_summary_shock_minus_pre_after_hours-pre_market.json")
    subset_qav2 = load_json(results_root / "regime_subset_experiments_qav2_real" / "regime_subset_summary_shock_minus_pre_after_hours-pre_market.json")
    regime_qav1 = load_json(results_root / "regime_residual_experiments_real" / "regime_residual_summary_shock_minus_pre.json")
    regime_qav2 = load_json(results_root / "regime_residual_experiments_qav2_real" / "regime_residual_summary_shock_minus_pre.json")
    qa_v2_summary = load_json(results_root / "qa_benchmark_features_v2_real" / "qa_benchmark_features_summary.json")

    slices = {
        "full_all_html": {
            "label": "Full sample, all HTML rows",
            "qav1": qav1_full,
            "qav2": qav2_full,
        },
        "full_clean": {
            "label": "Full sample, exclude html_integrity_flag=fail",
            "qav1": qav1_full_clean,
            "qav2": qav2_full_clean,
        },
        "offhours_all_html": {
            "label": "Off-hours only, all HTML rows",
            "qav1": qav1_off,
            "qav2": qav2_off,
        },
        "offhours_clean": {
            "label": "Off-hours only, exclude html_integrity_flag=fail",
            "qav1": qav1_off_clean,
            "qav2": qav2_off_clean,
        },
    }

    decomposition = {}
    for key, item in slices.items():
        qav1 = item["qav1"]
        qav2 = item["qav2"]
        decomposition[key] = {
            "label": item["label"],
            "split_sizes": qav2["split_sizes"],
            "market_only_qav1": model_r2(qav1, "market_only"),
            "market_only_qav2": model_r2(qav2, "market_only"),
            "market_plus_controls_qav1": model_r2(qav1, "market_plus_controls"),
            "market_plus_controls_qav2": model_r2(qav2, "market_plus_controls"),
            "ecc_text_timing_only_qav1": model_r2(qav1, "ecc_text_timing_only"),
            "ecc_text_timing_only_qav2": model_r2(qav2, "ecc_text_timing_only"),
            "ecc_text_timing_plus_audio_qav1": model_r2(qav1, "ecc_text_timing_plus_audio"),
            "ecc_text_timing_plus_audio_qav2": model_r2(qav2, "ecc_text_timing_plus_audio"),
            "market_controls_plus_ecc_qav1": model_r2(qav1, "market_controls_plus_ecc_text_timing"),
            "market_controls_plus_ecc_qav2": model_r2(qav2, "market_controls_plus_ecc_text_timing"),
            "market_controls_plus_ecc_audio_qav1": model_r2(qav1, "market_controls_plus_ecc_plus_audio"),
            "market_controls_plus_ecc_audio_qav2": model_r2(qav2, "market_controls_plus_ecc_plus_audio"),
        }
        for optional_key in (
            "qa_benchmark_only",
            "qa_benchmark_plus_qna_lsa",
            "market_controls_plus_qa_benchmark",
            "market_controls_plus_qa_benchmark_plus_qna_lsa",
        ):
            if optional_key in qav2["models"]:
                decomposition[key][f"{optional_key}_qav2"] = model_r2(qav2, optional_key)

    headline = {
        "target_variant_shock_minus_pre": {
            "qav1_best_mixed": target_variant_r2(target_qav1, "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark"),
            "qav2_best_mixed": target_variant_r2(target_qav2, "residual_structured_plus_extra_plus_qna_lsa_plus_qa_benchmark"),
            "qav1_structured_plus_extra": target_variant_r2(target_qav1, "residual_structured_plus_extra"),
            "qav2_structured_plus_extra": target_variant_r2(target_qav2, "residual_structured_plus_extra"),
            "qav2_structured_plus_extra_plus_qa_benchmark": target_variant_r2(target_qav2, "residual_structured_plus_extra_plus_qa_benchmark"),
        },
        "offhours_subset_shock_minus_pre": {
            "qav1_dense_plus_qna_lsa": regime_subset_r2(subset_qav1, "residual_dense_plus_qna_lsa"),
            "qav2_dense_plus_qna_lsa": regime_subset_r2(subset_qav2, "residual_dense_plus_qna_lsa"),
            "qav1_dense": regime_subset_r2(subset_qav1, "residual_dense"),
            "qav2_dense": regime_subset_r2(subset_qav2, "residual_dense"),
        },
        "regime_residual_shock_minus_pre": {
            "qav1_global": regime_residual_r2(regime_qav1, "global_residual_model"),
            "qav2_global": regime_residual_r2(regime_qav2, "global_residual_model"),
            "qav1_regime_specific": regime_residual_r2(regime_qav1, "regime_specific_residual_model"),
            "qav2_regime_specific": regime_residual_r2(regime_qav2, "regime_specific_residual_model"),
            "qav1_hybrid": regime_residual_r2(regime_qav1, "hybrid_val_selected_model"),
            "qav2_hybrid": regime_residual_r2(regime_qav2, "hybrid_val_selected_model"),
        },
    }

    current_reading = {
        "qa_v2_feature_coverage": qa_v2_summary,
        "full_sample_ecc_only_improves_but_remains_below_market": True,
        "clean_sample_robustness_is_weak": True,
        "headline_mixed_models_do_not_improve": True,
        "market_only_baseline_still_dominant": True,
        "recommended_next_gate": "Move to transferred pair-level Q&A labels or public-data benchmark pivot only after preserving benchmark ladder discipline.",
    }

    return {
        "qa_feature_summary": qa_v2_summary,
        "decomposition": decomposition,
        "headline_models": headline,
        "current_reading": current_reading,
    }


def build_markdown(summary: dict) -> str:
    qa_summary = summary["qa_feature_summary"]
    decomposition = summary["decomposition"]
    headline = summary["headline_models"]

    lines = [
        "# Q&A Signal Checkpoint 2026-03-14",
        "",
        "This note compares the original `qa_benchmark` stack (`qav1`) with the richer heuristic `Q&A` feature refresh (`qav2`) and locks in the current interpretation before the next modeling branch.",
        "",
        "## Coverage",
        "",
        f"- `qa_benchmark_features_v2` covers `{qa_summary['num_events']}` events, `{qa_summary['num_events_with_pairs']}` with at least one detected pair, and about `{qa_summary['avg_pair_count']:.2f}` pairs per event.",
        "",
        "## Decomposition checkpoint",
        "",
        "| Slice | Market-only | Market+controls | ECC-only qav1 | ECC-only qav2 | Delta | ECC+audio qav2 | Best market+ECC qav2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for item in decomposition.values():
        lines.append(
            "| "
            + item["label"]
            + " | "
            + r2(item["market_only_qav2"])
            + " | "
            + r2(item["market_plus_controls_qav2"])
            + " | "
            + r2(item["ecc_text_timing_only_qav1"])
            + " | "
            + r2(item["ecc_text_timing_only_qav2"])
            + " | "
            + delta(item["ecc_text_timing_only_qav2"], item["ecc_text_timing_only_qav1"])
            + " | "
            + r2(item["ecc_text_timing_plus_audio_qav2"])
            + " | "
            + r2(item["market_controls_plus_ecc_qav2"])
            + " |"
        )

    lines.extend(
        [
            "",
            "Key reading from the table:",
            "",
            f"- Full-sample ECC-only improves from `{r2(decomposition['full_all_html']['ecc_text_timing_only_qav1'])}` to `{r2(decomposition['full_all_html']['ecc_text_timing_only_qav2'])}`, but still stays far below `market_only={r2(decomposition['full_all_html']['market_only_qav2'])}` and `market_plus_controls={r2(decomposition['full_all_html']['market_plus_controls_qav2'])}`.",
            f"- Off-hours ECC-only also improves on the all-HTML slice from `{r2(decomposition['offhours_all_html']['ecc_text_timing_only_qav1'])}` to `{r2(decomposition['offhours_all_html']['ecc_text_timing_only_qav2'])}`, showing that richer Q&A semantics recover some event-specific signal.",
            f"- Clean-sample robustness is still weak: full-sample ECC-only falls to `{r2(decomposition['full_clean']['ecc_text_timing_only_qav2'])}` and off-hours ECC-only falls to `{r2(decomposition['offhours_clean']['ecc_text_timing_only_qav2'])}`.",
            f"- Even when `qav2` helps ECC-only models, adding ECC back on top of market controls still underperforms the best market baseline. Example: full-sample `market_plus_controls={r2(decomposition['full_all_html']['market_plus_controls_qav2'])}` versus `market_controls_plus_ecc={r2(decomposition['full_all_html']['market_controls_plus_ecc_qav2'])}`.",
            "",
            "## Headline model comparison",
            "",
            "| Checkpoint | qav1 | qav2 | Delta |",
            "| --- | ---: | ---: | ---: |",
            "| Full-sample shock best mixed model | "
            + r2(headline["target_variant_shock_minus_pre"]["qav1_best_mixed"])
            + " | "
            + r2(headline["target_variant_shock_minus_pre"]["qav2_best_mixed"])
            + " | "
            + delta(
                headline["target_variant_shock_minus_pre"]["qav2_best_mixed"],
                headline["target_variant_shock_minus_pre"]["qav1_best_mixed"],
            )
            + " |",
            "| Off-hours shock best mixed model | "
            + r2(headline["offhours_subset_shock_minus_pre"]["qav1_dense_plus_qna_lsa"])
            + " | "
            + r2(headline["offhours_subset_shock_minus_pre"]["qav2_dense_plus_qna_lsa"])
            + " | "
            + delta(
                headline["offhours_subset_shock_minus_pre"]["qav2_dense_plus_qna_lsa"],
                headline["offhours_subset_shock_minus_pre"]["qav1_dense_plus_qna_lsa"],
            )
            + " |",
            "| Global regime residual model | "
            + r2(headline["regime_residual_shock_minus_pre"]["qav1_global"])
            + " | "
            + r2(headline["regime_residual_shock_minus_pre"]["qav2_global"])
            + " | "
            + delta(
                headline["regime_residual_shock_minus_pre"]["qav2_global"],
                headline["regime_residual_shock_minus_pre"]["qav1_global"],
            )
            + " |",
            "| Regime-specific residual model | "
            + r2(headline["regime_residual_shock_minus_pre"]["qav1_regime_specific"])
            + " | "
            + r2(headline["regime_residual_shock_minus_pre"]["qav2_regime_specific"])
            + " | "
            + delta(
                headline["regime_residual_shock_minus_pre"]["qav2_regime_specific"],
                headline["regime_residual_shock_minus_pre"]["qav1_regime_specific"],
            )
            + " |",
            "",
            "## Current decision",
            "",
            "- `qav2` is useful because it recovers positive ECC-only signal on the all-HTML sample and improves the global residual regime model.",
            "- `qav2` is not enough for a main claim of incremental ECC value because the gains do not survive the clean-sample slices and do not beat the strongest market-only benchmarks.",
            "- The current repo should therefore describe `qav2` as a partial signal-recovery step, not a paper-level breakthrough.",
            "",
            "## Next gate before any bigger model push",
            "",
            "1. Replace heuristic event-level `qav2` features with transferred pair-level labels or scores inspired by `SubjECTive-QA` and `EvasionBench`.",
            "2. Keep the benchmark ladder fixed: `prior_only`, `market_only`, `market_plus_controls`, `ECC_only`, and `market_plus_ECC`.",
            "3. Require any new ECC claim to survive the `exclude html_integrity_flag=fail` rerun before it becomes part of the paper narrative.",
            "4. If stronger transferred `Q&A` signals still fail to beat the market-only residual benchmark, pivot to the public reproducible benchmark path rather than scaling up the same heuristic feature family.",
            "",
        ]
    )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    summary = build_summary(repo_root)
    markdown = build_markdown(summary)

    args.output_md.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_md.resolve().write_text(markdown + "\n")

    args.output_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output_json.resolve().write_text(json.dumps(summary, indent=2) + "\n")

    print(markdown)


if __name__ == "__main__":
    main()
