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

FALLBACK_PRE = "consensus_fallback_pre_only"
FALLBACK_RETAINED = "consensus_fallback_semantic_backbone"
FALLBACK_SELECTED = "consensus_fallback_selected_expert"
FALLBACK_TREE = "consensus_fallback_pair_tree"
FALLBACK_LOGISTIC = "consensus_fallback_plus_text_logistic"
FALLBACK_AVG = "consensus_fallback_disagreement_average"

SPLITS = [
    "val2020_test_post2020",
    "val2021_test_post2021",
    "val2022_test_post2022",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark agreement-based transfer consensus fallback families using the temporal router confirmation outputs."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_router_consensus_fallback_benchmark_role_aware_audio_lsa4_real"),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["event_key"]: row for row in csv.DictReader(handle)}


def build_strategy_predictions(pair_row: dict[str, str], logistic_row: dict[str, str]) -> dict[str, float | int | str]:
    chosen_model = pair_row["chosen_model"]
    if chosen_model != logistic_row["chosen_model"]:
        raise SystemExit("chosen_model mismatch across pair/logistic temporal outputs")
    if pair_row[MODEL_SELECTED] != logistic_row[MODEL_SELECTED]:
        raise SystemExit("validation_selected_transfer_expert mismatch across pair/logistic temporal outputs")

    target = float(pair_row["target"])
    pre_only = float(pair_row[MODEL_PRE_ONLY])
    qa_expert = float(pair_row[MODEL_QA_EXPERT])
    retained = float(pair_row[MODEL_RETAINED])
    selected = float(pair_row[MODEL_SELECTED])
    pair_tree = float(pair_row[MODEL_PAIR_TREE])
    plus_text_logistic = float(logistic_row[MODEL_PLUS_TEXT_LOGISTIC])
    tree_choose_qa = int(float(pair_row["tree_choose_qa"]))
    logistic_choose_qa = int(float(logistic_row["logistic_choose_qa"]))
    selected_is_qa = int(chosen_model == MODEL_QA_EXPERT)
    agreement = int(tree_choose_qa == logistic_choose_qa)
    agreed_pred = qa_expert if tree_choose_qa == 1 else retained
    agreed_choose_qa = int(tree_choose_qa == 1)

    if agreement:
        qa_share_pre = qa_share_retained = qa_share_selected = qa_share_tree = qa_share_logistic = qa_share_avg = agreed_choose_qa
    else:
        qa_share_pre = 0
        qa_share_retained = 0
        qa_share_selected = selected_is_qa
        qa_share_tree = tree_choose_qa
        qa_share_logistic = logistic_choose_qa
        qa_share_avg = 0

    return {
        "target": target,
        MODEL_PRE_ONLY: pre_only,
        MODEL_RETAINED: retained,
        MODEL_SELECTED: selected,
        MODEL_PAIR_TREE: pair_tree,
        MODEL_PLUS_TEXT_LOGISTIC: plus_text_logistic,
        FALLBACK_PRE: agreed_pred if agreement else pre_only,
        FALLBACK_RETAINED: agreed_pred if agreement else retained,
        FALLBACK_SELECTED: agreed_pred if agreement else selected,
        FALLBACK_TREE: agreed_pred if agreement else pair_tree,
        FALLBACK_LOGISTIC: agreed_pred if agreement else plus_text_logistic,
        FALLBACK_AVG: agreed_pred if agreement else 0.5 * (pair_tree + plus_text_logistic),
        "agreement": agreement,
        "tree_choose_qa": tree_choose_qa,
        "logistic_choose_qa": logistic_choose_qa,
        "selected_is_qa": selected_is_qa,
        "qa_share_pre": qa_share_pre,
        "qa_share_retained": qa_share_retained,
        "qa_share_selected": qa_share_selected,
        "qa_share_tree": qa_share_tree,
        "qa_share_logistic": qa_share_logistic,
        "qa_share_avg": qa_share_avg,
        "chosen_model": chosen_model,
    }


def summarize_split(y_true: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {model_name: metrics(y_true, pred) for model_name, pred in preds.items()}


def main() -> None:
    args = parse_args()
    temporal_root = args.temporal_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_summary_path = temporal_root / "afterhours_transfer_router_temporal_confirmation_summary.json"
    temporal_summary = json.load(temporal_summary_path.open()) if temporal_summary_path.exists() else {}

    pooled_rows = []
    pooled_y = []
    tracked_models = [
        MODEL_PRE_ONLY,
        MODEL_RETAINED,
        MODEL_SELECTED,
        MODEL_PAIR_TREE,
        MODEL_PLUS_TEXT_LOGISTIC,
        FALLBACK_PRE,
        FALLBACK_RETAINED,
        FALLBACK_SELECTED,
        FALLBACK_TREE,
        FALLBACK_LOGISTIC,
        FALLBACK_AVG,
    ]
    pooled_preds = {model_name: [] for model_name in tracked_models}
    split_rows = []
    split_details = {}

    qa_share_key = {
        FALLBACK_PRE: "qa_share_pre",
        FALLBACK_RETAINED: "qa_share_retained",
        FALLBACK_SELECTED: "qa_share_selected",
        FALLBACK_TREE: "qa_share_tree",
        FALLBACK_LOGISTIC: "qa_share_logistic",
        FALLBACK_AVG: "qa_share_avg",
    }

    for split_name in SPLITS:
        pair_path = temporal_root / f"{split_name}__hybrid_pair_bench_tree" / "afterhours_transfer_conservative_router_predictions.csv"
        logistic_path = temporal_root / f"{split_name}__hybrid_plus_text_logistic" / "afterhours_transfer_conservative_router_predictions.csv"
        pair_rows = load_rows(pair_path)
        logistic_rows = load_rows(logistic_path)
        keys = sorted(set(pair_rows) & set(logistic_rows))
        if not keys:
            raise SystemExit(f"no shared temporal rows for split {split_name}")

        split_prediction_rows = []
        y_true = []
        split_preds = {model_name: [] for model_name in tracked_models}

        for key in keys:
            record = build_strategy_predictions(pair_rows[key], logistic_rows[key])
            y_true.append(record["target"])
            row = {
                "split": split_name,
                "event_key": key,
                "ticker": pair_rows[key]["ticker"],
                "year": int(float(pair_rows[key]["year"])),
                "chosen_model": record["chosen_model"],
                "agreement": record["agreement"],
                "tree_choose_qa": record["tree_choose_qa"],
                "logistic_choose_qa": record["logistic_choose_qa"],
                "selected_is_qa": record["selected_is_qa"],
                "qa_share_pre": record["qa_share_pre"],
                "qa_share_retained": record["qa_share_retained"],
                "qa_share_selected": record["qa_share_selected"],
                "qa_share_tree": record["qa_share_tree"],
                "qa_share_logistic": record["qa_share_logistic"],
                "qa_share_avg": record["qa_share_avg"],
                "target": record["target"],
            }
            for model_name in tracked_models:
                value = float(record[model_name])
                row[model_name] = value
                split_preds[model_name].append(value)
                pooled_preds[model_name].append(value)
            pooled_y.append(float(record["target"]))
            split_prediction_rows.append(row)
            pooled_rows.append(row)

        y_true_np = np.asarray(y_true, dtype=float)
        split_preds_np = {model_name: np.asarray(values, dtype=float) for model_name, values in split_preds.items()}
        overall = summarize_split(y_true_np, split_preds_np)

        split_significance = {
            f"{FALLBACK_PRE}__vs__{MODEL_PRE_ONLY}": summarize_significance(
                y_true_np, split_preds_np[MODEL_PRE_ONLY], split_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
            ),
            f"{FALLBACK_PRE}__vs__{FALLBACK_RETAINED}": summarize_significance(
                y_true_np, split_preds_np[FALLBACK_RETAINED], split_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
            ),
            f"{FALLBACK_PRE}__vs__{MODEL_PAIR_TREE}": summarize_significance(
                y_true_np, split_preds_np[MODEL_PAIR_TREE], split_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
            ),
            f"{FALLBACK_PRE}__vs__{MODEL_PLUS_TEXT_LOGISTIC}": summarize_significance(
                y_true_np, split_preds_np[MODEL_PLUS_TEXT_LOGISTIC], split_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
            ),
            f"{FALLBACK_PRE}__vs__{FALLBACK_AVG}": summarize_significance(
                y_true_np, split_preds_np[FALLBACK_AVG], split_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
            ),
        }

        split_row = {
            "split": split_name,
            "test_size": len(split_prediction_rows),
            "agreement_rate": float(np.mean([row["agreement"] for row in split_prediction_rows])),
            "pre_only_r2": overall[MODEL_PRE_ONLY]["r2"],
            "retained_r2": overall[MODEL_RETAINED]["r2"],
            "selected_r2": overall[MODEL_SELECTED]["r2"],
            "pair_tree_r2": overall[MODEL_PAIR_TREE]["r2"],
            "plus_text_logistic_r2": overall[MODEL_PLUS_TEXT_LOGISTIC]["r2"],
            "fallback_pre_r2": overall[FALLBACK_PRE]["r2"],
            "fallback_retained_r2": overall[FALLBACK_RETAINED]["r2"],
            "fallback_selected_r2": overall[FALLBACK_SELECTED]["r2"],
            "fallback_tree_r2": overall[FALLBACK_TREE]["r2"],
            "fallback_logistic_r2": overall[FALLBACK_LOGISTIC]["r2"],
            "fallback_avg_r2": overall[FALLBACK_AVG]["r2"],
            "fallback_pre_p_vs_pre_only": split_significance[f"{FALLBACK_PRE}__vs__{MODEL_PRE_ONLY}"]["mse_gain_pvalue"],
            "fallback_pre_p_vs_retained": split_significance[f"{FALLBACK_PRE}__vs__{FALLBACK_RETAINED}"]["mse_gain_pvalue"],
            "fallback_pre_p_vs_tree": split_significance[f"{FALLBACK_PRE}__vs__{MODEL_PAIR_TREE}"]["mse_gain_pvalue"],
            "fallback_pre_p_vs_logistic": split_significance[f"{FALLBACK_PRE}__vs__{MODEL_PLUS_TEXT_LOGISTIC}"]["mse_gain_pvalue"],
            "fallback_pre_p_vs_avg": split_significance[f"{FALLBACK_PRE}__vs__{FALLBACK_AVG}"]["mse_gain_pvalue"],
        }
        for model_name, key in qa_share_key.items():
            split_row[f"{model_name}_qa_share"] = float(np.mean([row[key] for row in split_prediction_rows]))

        split_rows.append(split_row)
        split_details[split_name] = {
            "overall": overall,
            "significance": split_significance,
        }

    pooled_y_np = np.asarray(pooled_y, dtype=float)
    pooled_preds_np = {model_name: np.asarray(values, dtype=float) for model_name, values in pooled_preds.items()}
    pooled_overall = summarize_split(pooled_y_np, pooled_preds_np)
    pooled_significance = {
        f"{FALLBACK_PRE}__vs__{MODEL_PRE_ONLY}": summarize_significance(
            pooled_y_np, pooled_preds_np[MODEL_PRE_ONLY], pooled_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
        ),
        f"{FALLBACK_PRE}__vs__{FALLBACK_RETAINED}": summarize_significance(
            pooled_y_np, pooled_preds_np[FALLBACK_RETAINED], pooled_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
        ),
        f"{FALLBACK_PRE}__vs__{MODEL_PAIR_TREE}": summarize_significance(
            pooled_y_np, pooled_preds_np[MODEL_PAIR_TREE], pooled_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
        ),
        f"{FALLBACK_PRE}__vs__{MODEL_PLUS_TEXT_LOGISTIC}": summarize_significance(
            pooled_y_np, pooled_preds_np[MODEL_PLUS_TEXT_LOGISTIC], pooled_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
        ),
        f"{FALLBACK_PRE}__vs__{FALLBACK_AVG}": summarize_significance(
            pooled_y_np, pooled_preds_np[FALLBACK_AVG], pooled_preds_np[FALLBACK_PRE], args.bootstrap_iters, args.perm_iters, args.seed
        ),
    }

    pooled_summary = {
        "overall": pooled_overall,
        "significance": pooled_significance,
        "agreement_rate": float(np.mean([row["agreement"] for row in pooled_rows])),
        **{
            f"{model_name}_qa_share": float(np.mean([row[key] for row in pooled_rows]))
            for model_name, key in qa_share_key.items()
        },
    }

    summary = {
        "source_temporal_root": str(temporal_root),
        "source_temporal_config": temporal_summary.get("config", {}),
        "splits": split_rows,
        "details": split_details,
        "pooled": pooled_summary,
        "pooled_ranking": sorted(
            [{"model": model_name, "r2": pooled_overall[model_name]["r2"]} for model_name in tracked_models],
            key=lambda item: item["r2"],
            reverse=True,
        ),
        "win_counts": {
            "fallback_pre_beats_pre_only": sum(1 for row in split_rows if row["fallback_pre_r2"] > row["pre_only_r2"]),
            "fallback_pre_beats_retained": sum(1 for row in split_rows if row["fallback_pre_r2"] > row["fallback_retained_r2"]),
            "fallback_pre_beats_tree": sum(1 for row in split_rows if row["fallback_pre_r2"] > row["pair_tree_r2"]),
            "fallback_pre_beats_logistic": sum(1 for row in split_rows if row["fallback_pre_r2"] > row["plus_text_logistic_r2"]),
        },
    }

    write_csv(output_dir / "afterhours_transfer_router_consensus_fallback_overview.csv", split_rows)
    write_csv(output_dir / "afterhours_transfer_router_consensus_fallback_predictions.csv", pooled_rows)
    write_json(output_dir / "afterhours_transfer_router_consensus_fallback_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
