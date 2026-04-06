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
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

from dj30_qc_utils import write_csv, write_json
from run_afterhours_transfer_expert_selection import summarize_significance
from run_structured_baselines import metrics


MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_QA_EXPERT = "residual_pre_call_market_plus_a4_plus_qa_benchmark_svd_observability_gate"
MODEL_RETAINED = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate"
MODEL_SELECTED = "validation_selected_transfer_expert"
MODEL_PAIR_TREE = "conservative_tree_override_on_selected_expert"
MODEL_PLUS_TEXT_LOGISTIC = "conservative_logistic_override_on_selected_expert"
MODEL_AGREED = "agreement_supported_pred"
MODEL_DISAGREEMENT_AVG = "disagreement_average"

SPLITS = [
    "val2020_test_post2020",
    "val2021_test_post2021",
    "val2022_test_post2022",
]

SUBSET_ORDER = [
    "agreement",
    "agreement_semantic",
    "agreement_qa",
    "disagreement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whether the transfer-router lift lives on agreement events while disagreement is best handled by abstaining to the market baseline."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_router_abstention_diagnostics_role_aware_audio_lsa4_real"),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["event_key"]: row for row in csv.DictReader(handle)}


def build_record(split_name: str, pair_row: dict[str, str], logistic_row: dict[str, str]) -> dict[str, float | int | str]:
    if pair_row[MODEL_SELECTED] != logistic_row[MODEL_SELECTED]:
        raise SystemExit("validation_selected_transfer_expert mismatch across pair/logistic temporal outputs")

    tree_choose_qa = int(float(pair_row["tree_choose_qa"]))
    logistic_choose_qa = int(float(logistic_row["logistic_choose_qa"]))
    agreement = int(tree_choose_qa == logistic_choose_qa)
    agreement_qa = int(agreement and tree_choose_qa == 1)
    agreement_semantic = int(agreement and tree_choose_qa == 0)
    disagreement = int(not agreement)

    return {
        "split": split_name,
        "event_key": pair_row["event_key"],
        "ticker": pair_row["ticker"],
        "year": int(float(pair_row["year"])),
        "target": float(pair_row["target"]),
        MODEL_PRE_ONLY: float(pair_row[MODEL_PRE_ONLY]),
        MODEL_QA_EXPERT: float(pair_row[MODEL_QA_EXPERT]),
        MODEL_RETAINED: float(pair_row[MODEL_RETAINED]),
        MODEL_SELECTED: float(pair_row[MODEL_SELECTED]),
        MODEL_PAIR_TREE: float(pair_row[MODEL_PAIR_TREE]),
        MODEL_PLUS_TEXT_LOGISTIC: float(logistic_row[MODEL_PLUS_TEXT_LOGISTIC]),
        MODEL_AGREED: float(pair_row[MODEL_QA_EXPERT]) if tree_choose_qa == 1 else float(pair_row[MODEL_RETAINED]),
        MODEL_DISAGREEMENT_AVG: 0.5 * (float(pair_row[MODEL_PAIR_TREE]) + float(logistic_row[MODEL_PLUS_TEXT_LOGISTIC])),
        "chosen_model": pair_row["chosen_model"],
        "tree_choose_qa": tree_choose_qa,
        "logistic_choose_qa": logistic_choose_qa,
        "agreement": agreement,
        "agreement_semantic": agreement_semantic,
        "agreement_qa": agreement_qa,
        "disagreement": disagreement,
        "selected_is_pre_only": int(pair_row["chosen_model"] == MODEL_PRE_ONLY),
        "selected_is_qa": int(pair_row["chosen_model"] == MODEL_QA_EXPERT),
        "selected_is_retained": int(pair_row["chosen_model"] == MODEL_RETAINED),
    }


def subset_rows(rows: list[dict[str, float | int | str]], subset_name: str) -> list[dict[str, float | int | str]]:
    return [row for row in rows if int(row[subset_name]) == 1]


def r2_rows_for_subset(rows: list[dict[str, float | int | str]], subset_name: str) -> dict[str, dict[str, float]]:
    y_true = np.asarray([float(row["target"]) for row in rows], dtype=float)
    model_names = [MODEL_PRE_ONLY, MODEL_QA_EXPERT, MODEL_RETAINED, MODEL_SELECTED, MODEL_PAIR_TREE, MODEL_PLUS_TEXT_LOGISTIC]
    if subset_name != "disagreement":
        model_names.append(MODEL_AGREED)
    if subset_name == "disagreement":
        model_names.append(MODEL_DISAGREEMENT_AVG)
    return {
        model_name: metrics(y_true, np.asarray([float(row[model_name]) for row in rows], dtype=float))
        for model_name in model_names
    }


def significance_for_subset(
    rows: list[dict[str, float | int | str]],
    subset_name: str,
    bootstrap_iters: int,
    perm_iters: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    y_true = np.asarray([float(row["target"]) for row in rows], dtype=float)

    def pred(model_name: str) -> np.ndarray:
        return np.asarray([float(row[model_name]) for row in rows], dtype=float)

    if subset_name == "agreement":
        return {
            f"{MODEL_AGREED}__vs__{MODEL_PRE_ONLY}": summarize_significance(
                y_true, pred(MODEL_PRE_ONLY), pred(MODEL_AGREED), bootstrap_iters, perm_iters, seed
            ),
            f"{MODEL_AGREED}__vs__{MODEL_RETAINED}": summarize_significance(
                y_true, pred(MODEL_RETAINED), pred(MODEL_AGREED), bootstrap_iters, perm_iters, seed
            ),
            f"{MODEL_AGREED}__vs__{MODEL_QA_EXPERT}": summarize_significance(
                y_true, pred(MODEL_QA_EXPERT), pred(MODEL_AGREED), bootstrap_iters, perm_iters, seed
            ),
            f"{MODEL_AGREED}__vs__{MODEL_SELECTED}": summarize_significance(
                y_true, pred(MODEL_SELECTED), pred(MODEL_AGREED), bootstrap_iters, perm_iters, seed
            ),
        }
    if subset_name == "disagreement":
        return {
            f"{MODEL_PRE_ONLY}__vs__{MODEL_RETAINED}": summarize_significance(
                y_true, pred(MODEL_RETAINED), pred(MODEL_PRE_ONLY), bootstrap_iters, perm_iters, seed
            ),
            f"{MODEL_PRE_ONLY}__vs__{MODEL_QA_EXPERT}": summarize_significance(
                y_true, pred(MODEL_QA_EXPERT), pred(MODEL_PRE_ONLY), bootstrap_iters, perm_iters, seed
            ),
            f"{MODEL_PRE_ONLY}__vs__{MODEL_SELECTED}": summarize_significance(
                y_true, pred(MODEL_SELECTED), pred(MODEL_PRE_ONLY), bootstrap_iters, perm_iters, seed
            ),
            f"{MODEL_PRE_ONLY}__vs__{MODEL_PAIR_TREE}": summarize_significance(
                y_true, pred(MODEL_PAIR_TREE), pred(MODEL_PRE_ONLY), bootstrap_iters, perm_iters, seed
            ),
            f"{MODEL_PRE_ONLY}__vs__{MODEL_PLUS_TEXT_LOGISTIC}": summarize_significance(
                y_true, pred(MODEL_PLUS_TEXT_LOGISTIC), pred(MODEL_PRE_ONLY), bootstrap_iters, perm_iters, seed
            ),
            f"{MODEL_PRE_ONLY}__vs__{MODEL_DISAGREEMENT_AVG}": summarize_significance(
                y_true, pred(MODEL_DISAGREEMENT_AVG), pred(MODEL_PRE_ONLY), bootstrap_iters, perm_iters, seed
            ),
        }
    return {}


def flatten_overview_row(split_name: str, subset_name: str, rows: list[dict[str, float | int | str]], overall: dict, significance: dict) -> dict:
    share = len(rows)
    row = {
        "split": split_name,
        "subset": subset_name,
        "test_size": len(rows),
        "agreement_rate": float(np.mean([int(r["agreement"]) for r in rows])) if rows else None,
        "qa_share": float(np.mean([int(r["tree_choose_qa"]) for r in rows])) if rows else None,
        "selected_pre_share": float(np.mean([int(r["selected_is_pre_only"]) for r in rows])) if rows else None,
        "selected_qa_share": float(np.mean([int(r["selected_is_qa"]) for r in rows])) if rows else None,
        "selected_retained_share": float(np.mean([int(r["selected_is_retained"]) for r in rows])) if rows else None,
    }
    for model_name, metric_dict in overall.items():
        row[f"{model_name}_r2"] = metric_dict["r2"]
        row[f"{model_name}_rmse"] = metric_dict["rmse"]
    for compare_key, sig in significance.items():
        row[f"{compare_key}_mse_gain_mean"] = sig["mse_gain_mean"]
        row[f"{compare_key}_mse_gain_pvalue"] = sig["mse_gain_pvalue"]
    return row


def main() -> None:
    args = parse_args()
    temporal_root = args.temporal_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_summary_path = temporal_root / "afterhours_transfer_router_temporal_confirmation_summary.json"
    temporal_summary = json.load(temporal_summary_path.open()) if temporal_summary_path.exists() else {}

    pooled_records: list[dict[str, float | int | str]] = []
    overview_rows: list[dict] = []
    split_details: dict[str, dict] = {}

    for split_name in SPLITS:
        pair_path = temporal_root / f"{split_name}__hybrid_pair_bench_tree" / "afterhours_transfer_conservative_router_predictions.csv"
        logistic_path = temporal_root / f"{split_name}__hybrid_plus_text_logistic" / "afterhours_transfer_conservative_router_predictions.csv"
        pair_rows = load_rows(pair_path)
        logistic_rows = load_rows(logistic_path)
        shared_keys = sorted(set(pair_rows) & set(logistic_rows))
        if not shared_keys:
            raise SystemExit(f"no shared temporal rows for split {split_name}")

        split_records = [build_record(split_name, pair_rows[key], logistic_rows[key]) for key in shared_keys]
        pooled_records.extend(split_records)

        subset_payload = {}
        for subset_name in SUBSET_ORDER:
            rows = subset_rows(split_records, subset_name)
            if not rows:
                continue
            overall = r2_rows_for_subset(rows, subset_name)
            significance = significance_for_subset(rows, subset_name, args.bootstrap_iters, args.perm_iters, args.seed)
            subset_payload[subset_name] = {
                "size": len(rows),
                "share": len(rows) / len(split_records),
                "overall": overall,
                "significance": significance,
            }
            overview_rows.append(flatten_overview_row(split_name, subset_name, rows, overall, significance))

        split_details[split_name] = {
            "test_size": len(split_records),
            "agreement_rate": float(np.mean([int(row["agreement"]) for row in split_records])),
            "agreement_qa_share": float(np.mean([int(row["agreement_qa"]) for row in split_records])),
            "agreement_semantic_share": float(np.mean([int(row["agreement_semantic"]) for row in split_records])),
            "disagreement_rate": float(np.mean([int(row["disagreement"]) for row in split_records])),
            "subsets": subset_payload,
        }

    pooled_details = {}
    for subset_name in SUBSET_ORDER:
        rows = subset_rows(pooled_records, subset_name)
        if not rows:
            continue
        overall = r2_rows_for_subset(rows, subset_name)
        significance = significance_for_subset(rows, subset_name, args.bootstrap_iters, args.perm_iters, args.seed)
        pooled_details[subset_name] = {
            "size": len(rows),
            "share": len(rows) / len(pooled_records),
            "overall": overall,
            "significance": significance,
        }
        overview_rows.append(flatten_overview_row("pooled", subset_name, rows, overall, significance))

    summary = {
        "source_temporal_root": str(temporal_root),
        "source_temporal_config": temporal_summary.get("config", {}),
        "subset_definitions": {
            "agreement": "tree and logistic choose the same transfer-side expert; evaluate the corresponding agreed retained/QA expert directly",
            "agreement_semantic": "agreement subset where both routers choose the retained semantic+audio expert",
            "agreement_qa": "agreement subset where both routers choose the QA benchmark expert",
            "disagreement": "tree and logistic disagree; compare whether abstaining to pre_call_market_only is safest",
        },
        "splits": split_details,
        "pooled": {
            "test_size": len(pooled_records),
            "agreement_rate": float(np.mean([int(row["agreement"]) for row in pooled_records])),
            "agreement_qa_share": float(np.mean([int(row["agreement_qa"]) for row in pooled_records])),
            "agreement_semantic_share": float(np.mean([int(row["agreement_semantic"]) for row in pooled_records])),
            "disagreement_rate": float(np.mean([int(row["disagreement"]) for row in pooled_records])),
            "subsets": pooled_details,
        },
        "pooled_takeaways": {
            "best_agreement_model_by_r2": max(
                [
                    {"model": model_name, "r2": pooled_details["agreement"]["overall"][model_name]["r2"]}
                    for model_name in pooled_details.get("agreement", {}).get("overall", {})
                ],
                key=lambda item: (item["r2"], item["model"] == MODEL_AGREED),
            ) if pooled_details.get("agreement") else None,
            "best_disagreement_model_by_r2": max(
                [
                    {"model": model_name, "r2": pooled_details["disagreement"]["overall"][model_name]["r2"]}
                    for model_name in pooled_details.get("disagreement", {}).get("overall", {})
                ],
                key=lambda item: item["r2"],
            ) if pooled_details.get("disagreement") else None,
        },
    }

    write_csv(output_dir / "afterhours_transfer_router_abstention_diagnostics_overview.csv", overview_rows)
    write_csv(output_dir / "afterhours_transfer_router_abstention_diagnostics_predictions.csv", pooled_records)
    write_json(output_dir / "afterhours_transfer_router_abstention_diagnostics_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
