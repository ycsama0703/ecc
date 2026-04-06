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
MODEL_CONSENSUS_SELECTED = "consensus_agreement_fallback_selected_expert"
MODEL_CONSENSUS_SEM = "consensus_agreement_semantic_backbone"

SPLITS = [
    "val2020_test_post2020",
    "val2021_test_post2021",
    "val2022_test_post2022",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build agreement-based consensus routes from the temporal transfer-router confirmation outputs."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_router_consensus_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open() as handle:
        return {row["event_key"]: row for row in csv.DictReader(handle)}


def consensus_predictions(
    pair_row: dict[str, str],
    logistic_row: dict[str, str],
) -> tuple[float, float, dict[str, float | int | str]]:
    target = float(pair_row["target"])
    pre_only = float(pair_row[MODEL_PRE_ONLY])
    qa_expert = float(pair_row[MODEL_QA_EXPERT])
    retained = float(pair_row[MODEL_RETAINED])
    selected = float(pair_row[MODEL_SELECTED])
    pair_tree = float(pair_row[MODEL_PAIR_TREE])
    logistic = float(logistic_row[MODEL_PLUS_TEXT_LOGISTIC])

    chosen_model = pair_row["chosen_model"]
    if chosen_model != logistic_row["chosen_model"]:
        raise SystemExit("chosen_model mismatch across pair/logistic temporal outputs")
    if pair_row[MODEL_SELECTED] != logistic_row[MODEL_SELECTED]:
        raise SystemExit("validation_selected_transfer_expert mismatch across pair/logistic temporal outputs")

    tree_choose_qa = int(float(pair_row["tree_choose_qa"]))
    logistic_choose_qa = int(float(logistic_row["logistic_choose_qa"]))
    selected_is_qa = int(chosen_model == MODEL_QA_EXPERT)

    agree = int(tree_choose_qa == logistic_choose_qa)
    if agree:
        agreed_pred = qa_expert if tree_choose_qa == 1 else retained
        consensus_selected = agreed_pred
        consensus_sem = agreed_pred
        consensus_selected_qa = int(tree_choose_qa == 1)
        consensus_sem_qa = int(tree_choose_qa == 1)
    else:
        consensus_selected = selected
        consensus_sem = retained
        consensus_selected_qa = selected_is_qa
        consensus_sem_qa = 0

    meta = {
        "target": target,
        MODEL_PRE_ONLY: pre_only,
        MODEL_QA_EXPERT: qa_expert,
        MODEL_RETAINED: retained,
        MODEL_SELECTED: selected,
        MODEL_PAIR_TREE: pair_tree,
        MODEL_PLUS_TEXT_LOGISTIC: logistic,
        MODEL_CONSENSUS_SELECTED: consensus_selected,
        MODEL_CONSENSUS_SEM: consensus_sem,
        "tree_choose_qa": tree_choose_qa,
        "logistic_choose_qa": logistic_choose_qa,
        "selected_is_qa": selected_is_qa,
        "consensus_selected_choose_qa": consensus_selected_qa,
        "consensus_sem_choose_qa": consensus_sem_qa,
        "agree": agree,
        "chosen_model": chosen_model,
    }
    return consensus_selected, consensus_sem, meta


def split_metrics(y_true: np.ndarray, preds: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {model_name: metrics(y_true, pred) for model_name, pred in preds.items()}


def main() -> None:
    args = parse_args()
    temporal_root = args.temporal_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal_summary_path = temporal_root / "afterhours_transfer_router_temporal_confirmation_summary.json"
    temporal_summary = json.load(temporal_summary_path.open()) if temporal_summary_path.exists() else {}

    per_split_rows = []
    split_details = {}
    pooled_prediction_rows = []
    pooled_y = []
    pooled_preds = {
        MODEL_PRE_ONLY: [],
        MODEL_RETAINED: [],
        MODEL_SELECTED: [],
        MODEL_PAIR_TREE: [],
        MODEL_PLUS_TEXT_LOGISTIC: [],
        MODEL_CONSENSUS_SELECTED: [],
        MODEL_CONSENSUS_SEM: [],
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
        preds = {model_name: [] for model_name in pooled_preds}

        for key in keys:
            pair_row = pair_rows[key]
            logistic_row = logistic_rows[key]
            _, _, meta = consensus_predictions(pair_row, logistic_row)
            y_true.append(meta["target"])
            for model_name in preds:
                preds[model_name].append(meta[model_name])

            split_record = {
                "split": split_name,
                "event_key": key,
                "ticker": pair_row["ticker"],
                "year": int(float(pair_row["year"])),
                "target": float(meta["target"]),
                "chosen_model": meta["chosen_model"],
                "agree": meta["agree"],
                "tree_choose_qa": meta["tree_choose_qa"],
                "logistic_choose_qa": meta["logistic_choose_qa"],
                "selected_is_qa": meta["selected_is_qa"],
                "consensus_selected_choose_qa": meta["consensus_selected_choose_qa"],
                "consensus_sem_choose_qa": meta["consensus_sem_choose_qa"],
            }
            for model_name in preds:
                split_record[model_name] = float(meta[model_name])
                pooled_preds[model_name].append(float(meta[model_name]))
            pooled_y.append(float(meta["target"]))
            split_prediction_rows.append(split_record)
            pooled_prediction_rows.append(split_record)

        y_true_np = np.asarray(y_true, dtype=float)
        preds_np = {model_name: np.asarray(values, dtype=float) for model_name, values in preds.items()}
        overall = split_metrics(y_true_np, preds_np)
        split_significance = {
            f"{MODEL_CONSENSUS_SELECTED}__vs__{MODEL_SELECTED}": summarize_significance(
                y_true_np,
                preds_np[MODEL_SELECTED],
                preds_np[MODEL_CONSENSUS_SELECTED],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_CONSENSUS_SELECTED}__vs__{MODEL_PAIR_TREE}": summarize_significance(
                y_true_np,
                preds_np[MODEL_PAIR_TREE],
                preds_np[MODEL_CONSENSUS_SELECTED],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_CONSENSUS_SELECTED}__vs__{MODEL_PLUS_TEXT_LOGISTIC}": summarize_significance(
                y_true_np,
                preds_np[MODEL_PLUS_TEXT_LOGISTIC],
                preds_np[MODEL_CONSENSUS_SELECTED],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_SELECTED}": summarize_significance(
                y_true_np,
                preds_np[MODEL_SELECTED],
                preds_np[MODEL_CONSENSUS_SEM],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_PAIR_TREE}": summarize_significance(
                y_true_np,
                preds_np[MODEL_PAIR_TREE],
                preds_np[MODEL_CONSENSUS_SEM],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_PLUS_TEXT_LOGISTIC}": summarize_significance(
                y_true_np,
                preds_np[MODEL_PLUS_TEXT_LOGISTIC],
                preds_np[MODEL_CONSENSUS_SEM],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_RETAINED}": summarize_significance(
                y_true_np,
                preds_np[MODEL_RETAINED],
                preds_np[MODEL_CONSENSUS_SEM],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
            f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_PRE_ONLY}": summarize_significance(
                y_true_np,
                preds_np[MODEL_PRE_ONLY],
                preds_np[MODEL_CONSENSUS_SEM],
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            ),
        }

        agreement_rate = float(np.mean([row["agree"] for row in split_prediction_rows]))
        selected_override_rate = float(
            np.mean(
                [
                    abs(row[MODEL_CONSENSUS_SELECTED] - row[MODEL_SELECTED]) > 1e-15
                    for row in split_prediction_rows
                ]
            )
        )
        sem_override_rate = float(
            np.mean(
                [
                    abs(row[MODEL_CONSENSUS_SEM] - row[MODEL_RETAINED]) > 1e-15
                    for row in split_prediction_rows
                ]
            )
        )
        split_row = {
            "split": split_name,
            "test_size": len(split_prediction_rows),
            "pre_only_r2": overall[MODEL_PRE_ONLY]["r2"],
            "retained_r2": overall[MODEL_RETAINED]["r2"],
            "selected_r2": overall[MODEL_SELECTED]["r2"],
            "pair_tree_r2": overall[MODEL_PAIR_TREE]["r2"],
            "plus_text_logistic_r2": overall[MODEL_PLUS_TEXT_LOGISTIC]["r2"],
            "consensus_selected_r2": overall[MODEL_CONSENSUS_SELECTED]["r2"],
            "consensus_sem_r2": overall[MODEL_CONSENSUS_SEM]["r2"],
            "agreement_rate": agreement_rate,
            "selected_override_rate": selected_override_rate,
            "sem_override_rate": sem_override_rate,
            "consensus_selected_qa_share": float(
                np.mean([row["consensus_selected_choose_qa"] for row in split_prediction_rows])
            ),
            "consensus_sem_qa_share": float(np.mean([row["consensus_sem_choose_qa"] for row in split_prediction_rows])),
            "consensus_sem_p_vs_selected": split_significance[
                f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_SELECTED}"
            ]["mse_gain_pvalue"],
            "consensus_sem_p_vs_tree": split_significance[
                f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_PAIR_TREE}"
            ]["mse_gain_pvalue"],
            "consensus_sem_p_vs_logistic": split_significance[
                f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_PLUS_TEXT_LOGISTIC}"
            ]["mse_gain_pvalue"],
            "consensus_sem_p_vs_retained": split_significance[
                f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_RETAINED}"
            ]["mse_gain_pvalue"],
            "consensus_sem_p_vs_pre": split_significance[
                f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_PRE_ONLY}"
            ]["mse_gain_pvalue"],
        }
        per_split_rows.append(split_row)
        split_details[split_name] = {
            "overall": overall,
            "significance": split_significance,
        }

    pooled_y_np = np.asarray(pooled_y, dtype=float)
    pooled_preds_np = {model_name: np.asarray(values, dtype=float) for model_name, values in pooled_preds.items()}
    pooled_overall = split_metrics(pooled_y_np, pooled_preds_np)
    pooled_significance = {
        f"{MODEL_CONSENSUS_SELECTED}__vs__{MODEL_SELECTED}": summarize_significance(
            pooled_y_np,
            pooled_preds_np[MODEL_SELECTED],
            pooled_preds_np[MODEL_CONSENSUS_SELECTED],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_SELECTED}": summarize_significance(
            pooled_y_np,
            pooled_preds_np[MODEL_SELECTED],
            pooled_preds_np[MODEL_CONSENSUS_SEM],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_PAIR_TREE}": summarize_significance(
            pooled_y_np,
            pooled_preds_np[MODEL_PAIR_TREE],
            pooled_preds_np[MODEL_CONSENSUS_SEM],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_PLUS_TEXT_LOGISTIC}": summarize_significance(
            pooled_y_np,
            pooled_preds_np[MODEL_PLUS_TEXT_LOGISTIC],
            pooled_preds_np[MODEL_CONSENSUS_SEM],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_RETAINED}": summarize_significance(
            pooled_y_np,
            pooled_preds_np[MODEL_RETAINED],
            pooled_preds_np[MODEL_CONSENSUS_SEM],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_CONSENSUS_SEM}__vs__{MODEL_PRE_ONLY}": summarize_significance(
            pooled_y_np,
            pooled_preds_np[MODEL_PRE_ONLY],
            pooled_preds_np[MODEL_CONSENSUS_SEM],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
    }

    pooled_summary = {
        "overall": pooled_overall,
        "significance": pooled_significance,
        "agreement_rate": float(np.mean([row["agree"] for row in pooled_prediction_rows])),
        "selected_override_rate": float(
            np.mean([abs(row[MODEL_CONSENSUS_SELECTED] - row[MODEL_SELECTED]) > 1e-15 for row in pooled_prediction_rows])
        ),
        "sem_override_rate": float(
            np.mean([abs(row[MODEL_CONSENSUS_SEM] - row[MODEL_RETAINED]) > 1e-15 for row in pooled_prediction_rows])
        ),
        "consensus_selected_qa_share": float(
            np.mean([row["consensus_selected_choose_qa"] for row in pooled_prediction_rows])
        ),
        "consensus_sem_qa_share": float(np.mean([row["consensus_sem_choose_qa"] for row in pooled_prediction_rows])),
    }

    summary = {
        "source_temporal_root": str(temporal_root),
        "source_temporal_config": temporal_summary.get("config", {}),
        "splits": per_split_rows,
        "details": split_details,
        "pooled": pooled_summary,
        "win_counts": {
            "consensus_sem_beats_selected": sum(
                1 for row in per_split_rows if row["consensus_sem_r2"] > row["selected_r2"]
            ),
            "consensus_sem_beats_pair_tree": sum(
                1 for row in per_split_rows if row["consensus_sem_r2"] > row["pair_tree_r2"]
            ),
            "consensus_sem_beats_logistic": sum(
                1 for row in per_split_rows if row["consensus_sem_r2"] > row["plus_text_logistic_r2"]
            ),
            "consensus_sem_beats_pre_only": sum(
                1 for row in per_split_rows if row["consensus_sem_r2"] > row["pre_only_r2"]
            ),
            "consensus_selected_beats_selected": sum(
                1 for row in per_split_rows if row["consensus_selected_r2"] > row["selected_r2"]
            ),
        },
    }

    write_csv(output_dir / "afterhours_transfer_router_consensus_overview.csv", per_split_rows)
    write_csv(output_dir / "afterhours_transfer_router_consensus_predictions.csv", pooled_prediction_rows)
    write_json(output_dir / "afterhours_transfer_router_consensus_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
