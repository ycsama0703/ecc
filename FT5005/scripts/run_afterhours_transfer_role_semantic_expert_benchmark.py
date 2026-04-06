#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import site
import sys
from collections import Counter
from pathlib import Path

import numpy as np

user_site = site.getusersitepackages()
user_site_removed = False
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)
    user_site_removed = True

try:
    from sklearn.decomposition import TruncatedSVD  # noqa: F401
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401
except ModuleNotFoundError:
    if user_site_removed and isinstance(user_site, str) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    from sklearn.decomposition import TruncatedSVD  # noqa: F401
    from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: F401

from run_afterhours_audio_upgrade_benchmark import (
    A4_STRUCTURED_FEATURES,
    PRE_CALL_MARKET_FEATURES,
    build_compressed_dense_bundle,
)
from run_afterhours_transfer_expert_selection import summarize_significance
from run_afterhours_transfer_factor_expert_integration import aligned_keyed_predictions
from run_afterhours_transfer_qa_signal_benchmark import (
    choose_gate_threshold,
    load_joined_rows_with_qa_and_aligned,
    parse_quantiles,
)
from run_afterhours_transfer_role_text_signal_benchmark import build_shared_role_lsa_bundle
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle
from run_offhours_shock_ablations import regime_label
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets
from dj30_qc_utils import write_csv, write_json

MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_QNA_SEM = "residual_pre_call_market_plus_a4_plus_qna_lsa_observability_gate"
MODEL_QUESTION_SEM = "residual_pre_call_market_plus_a4_plus_question_role_lsa_observability_gate"
MODEL_ANSWER_SEM = "residual_pre_call_market_plus_a4_plus_answer_role_lsa_observability_gate"
MODEL_QNA_AUDIO = "residual_pre_call_market_plus_a4_plus_qna_lsa_plus_aligned_audio_svd_observability_gate"
MODEL_QUESTION_AUDIO = "residual_pre_call_market_plus_a4_plus_question_role_lsa_plus_aligned_audio_svd_observability_gate"
MODEL_SELECTED = "validation_selected_role_semantic_core"
MODEL_HARD_ABSTENTION = "agreement_pre_only_abstention"

SELECTION_MODELS = [MODEL_PRE_ONLY, MODEL_QNA_AUDIO, MODEL_QUESTION_AUDIO]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark question-role semantics as a replacement semantic core in the matched after-hours transfer shell."
    )
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv"),
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("results/features_real/event_text_audio_features.csv"),
    )
    parser.add_argument(
        "--old-audio-csv",
        type=Path,
        default=Path("results/audio_real/event_real_audio_features.csv"),
    )
    parser.add_argument(
        "--aligned-audio-csv",
        type=Path,
        default=Path("results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv"),
    )
    parser.add_argument(
        "--qa-csv",
        type=Path,
        default=Path("results/qa_benchmark_features_v2_real/qa_benchmark_features.csv"),
    )
    parser.add_argument(
        "--hard-reference-predictions-csv",
        type=Path,
        default=Path(
            "results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/"
            "afterhours_transfer_agreement_signal_benchmark_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--hard-reference-summary-json",
        type=Path,
        default=Path(
            "results/afterhours_transfer_agreement_signal_benchmark_role_aware_audio_lsa4_real/"
            "afterhours_transfer_agreement_signal_benchmark_summary.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_role_semantic_expert_benchmark_role_aware_audio_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="after_hours")
    parser.add_argument("--exclude-html-flags", default="fail")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min-test-events", type=int, default=3)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=4)
    parser.add_argument("--aligned-compressed-components", type=int, default=8)
    parser.add_argument("--role-max-features", type=int, default=4000)
    parser.add_argument("--role-term-limit", type=int, default=12)
    parser.add_argument("--gate-feature", default="a4_strict_row_share")
    parser.add_argument("--gate-quantiles", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def constant_prior(length: int, value: float) -> np.ndarray:
    return np.full(length, float(value), dtype=float)


def load_reference_predictions(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    quantiles = parse_quantiles(args.gate_quantiles)
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    base_rows, coverage = load_joined_rows_with_qa_and_aligned(
        args.panel_csv,
        args.features_csv,
        args.old_audio_csv,
        args.aligned_audio_csv,
        args.qa_csv,
        "aligned_audio__",
    )

    rows = []
    for row in base_rows:
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        if html_flag in exclude_html_flags:
            continue
        reg = regime_label(row)
        if reg not in include_regimes:
            continue
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        item["_regime"] = reg
        rows.append(item)

    if not rows:
        raise SystemExit("no rows available after joining and filtering")

    aligned_feature_names = [
        key for key in rows[0].keys() if key.startswith("aligned_audio__") and rows[0].get(key, "") != ""
    ]
    if not aligned_feature_names:
        raise SystemExit("no aligned audio features found in joined rows")

    candidate_tickers = sorted({row["ticker"] for row in rows if row["_year"] > args.val_year and row.get("ticker")})
    if not candidate_tickers:
        raise SystemExit("no candidate held-out tickers found")

    reference_pred_rows = load_reference_predictions(args.hard_reference_predictions_csv.resolve())
    reference_summary = json.loads(args.hard_reference_summary_json.resolve().read_text())

    prediction_rows = []
    selection_rows = []
    overall_y: list[float] = []
    overall_preds = {
        MODEL_PRE_ONLY: [],
        MODEL_QNA_SEM: [],
        MODEL_QUESTION_SEM: [],
        MODEL_ANSWER_SEM: [],
        MODEL_QNA_AUDIO: [],
        MODEL_QUESTION_AUDIO: [],
        MODEL_SELECTED: [],
    }
    selection_counts: Counter[str] = Counter()
    ticker_summary = {}
    skipped = {}
    bundle_meta = None

    model_specs = {
        MODEL_PRE_ONLY: ["pre_call_market"],
        MODEL_QNA_SEM: ["pre_call_market", "a4", "qna_lsa"],
        MODEL_QUESTION_SEM: ["pre_call_market", "a4", "question_role_lsa"],
        MODEL_ANSWER_SEM: ["pre_call_market", "a4", "answer_role_lsa"],
        MODEL_QNA_AUDIO: ["pre_call_market", "a4", "qna_lsa", "aligned_audio_svd"],
        MODEL_QUESTION_AUDIO: ["pre_call_market", "a4", "question_role_lsa", "aligned_audio_svd"],
    }

    for ticker in candidate_tickers:
        train_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] <= args.train_end_year]
        val_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] == args.val_year]
        test_rows = [row for row in rows if row["ticker"] == ticker and row["_year"] > args.val_year]

        if len(test_rows) < args.min_test_events or not train_rows or not val_rows:
            skipped[ticker] = {"train": len(train_rows), "val": len(val_rows), "test": len(test_rows)}
            continue

        global_prior = float(np.mean([row["_target"] for row in train_rows]))
        train_prior = constant_prior(len(train_rows), global_prior)
        val_prior = constant_prior(len(val_rows), global_prior)
        test_prior = constant_prior(len(test_rows), global_prior)
        train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
        val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
        test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)

        role_bundle = build_shared_role_lsa_bundle(
            train_rows,
            val_rows,
            test_rows,
            args.role_max_features,
            args.min_df,
            args.lsa_components,
            args.role_term_limit,
        )
        bundles = {
            "pre_call_market": build_dense_bundle(train_rows, val_rows, test_rows, PRE_CALL_MARKET_FEATURES),
            "a4": build_dense_bundle(train_rows, val_rows, test_rows, A4_STRUCTURED_FEATURES),
            "qna_lsa": build_text_lsa_bundle(
                train_rows,
                val_rows,
                test_rows,
                text_col="qna_text",
                max_features=args.max_features,
                min_df=args.min_df,
                lsa_components=args.lsa_components,
            ),
            "question_role_lsa": {
                "train": role_bundle["question_train"],
                "val": role_bundle["question_val"],
                "test": role_bundle["question_test"],
            },
            "answer_role_lsa": {
                "train": role_bundle["answer_train"],
                "val": role_bundle["answer_val"],
                "test": role_bundle["answer_test"],
            },
            "aligned_audio_svd": build_compressed_dense_bundle(
                train_rows,
                val_rows,
                test_rows,
                aligned_feature_names,
                args.aligned_compressed_components,
                prefix="aligned_audio_svd",
            ),
        }

        if bundle_meta is None:
            bundle_meta = {
                "qna_lsa_components": int(bundles["qna_lsa"]["train"].shape[1]),
                "question_role_lsa_components": int(role_bundle["question_train"].shape[1]),
                "answer_role_lsa_components": int(role_bundle["answer_train"].shape[1]),
                "role_shared_lsa_explained_variance_ratio_sum": role_bundle["explained_variance_ratio_sum"],
                "aligned_audio_svd": {
                    "input_feature_count": len(aligned_feature_names),
                    "n_components": bundles["aligned_audio_svd"]["n_components"],
                    "explained_variance_ratio_sum": bundles["aligned_audio_svd"]["explained_variance_ratio_sum"],
                },
            }

        val_scores = np.asarray([float(row.get(args.gate_feature) or 0.0) for row in val_rows], dtype=float)
        test_scores = np.asarray([float(row.get(args.gate_feature) or 0.0) for row in test_rows], dtype=float)

        val_preds = {}
        test_preds = {}
        gate_meta = {}
        for model_name, bundle_names in model_specs.items():
            train_x = np.hstack([bundles[name]["train"] for name in bundle_names])
            val_x = np.hstack([bundles[name]["val"] for name in bundle_names])
            test_x = np.hstack([bundles[name]["test"] for name in bundle_names])
            _, best_model, val_pred = fit_residual_ridge(
                train_x,
                train_prior,
                train_y,
                val_x,
                val_prior,
                val_y,
                alphas,
            )
            val_pred_np = np.asarray(val_pred, dtype=float)
            test_pred_np = np.asarray(test_prior + best_model.predict(test_x), dtype=float)
            if model_name == MODEL_PRE_ONLY:
                val_preds[model_name] = val_pred_np
                test_preds[model_name] = test_pred_np
                continue
            threshold, threshold_meta = choose_gate_threshold(
                val_rows,
                val_prior,
                val_pred_np,
                args.gate_feature,
                quantiles,
            )
            val_preds[model_name] = np.where(val_scores >= threshold, val_pred_np, val_prior)
            test_preds[model_name] = np.where(test_scores >= threshold, test_pred_np, test_prior)
            gate_meta[model_name] = threshold_meta

        validation_metrics = {name: metrics(val_y, val_preds[name]) for name in model_specs}
        test_metrics = {name: metrics(test_y, test_preds[name]) for name in model_specs}
        selected_model = max(SELECTION_MODELS, key=lambda model_name: validation_metrics[model_name]["r2"])
        selected_pred = test_preds[selected_model]
        selection_counts[selected_model] += 1

        ticker_summary[ticker] = {
            "n_test": len(test_rows),
            "selected_model": selected_model,
            "validation_r2": {model_name: validation_metrics[model_name]["r2"] for model_name in model_specs},
            "test_r2": {model_name: test_metrics[model_name]["r2"] for model_name in model_specs},
            "gate_meta": gate_meta,
        }

        selection_rows.append(
            {
                "ticker": ticker,
                "n_test": len(test_rows),
                "selected_model": selected_model,
                **{f"val_r2__{name}": validation_metrics[name]["r2"] for name in model_specs},
                **{f"test_r2__{name}": test_metrics[name]["r2"] for name in model_specs},
                f"test_r2__{MODEL_SELECTED}": metrics(test_y, selected_pred)["r2"],
            }
        )

        for row, pred_pre, pred_qna_sem, pred_question_sem, pred_answer_sem, pred_qna_audio, pred_question_audio, pred_selected in zip(
            test_rows,
            test_preds[MODEL_PRE_ONLY],
            test_preds[MODEL_QNA_SEM],
            test_preds[MODEL_QUESTION_SEM],
            test_preds[MODEL_ANSWER_SEM],
            test_preds[MODEL_QNA_AUDIO],
            test_preds[MODEL_QUESTION_AUDIO],
            selected_pred,
        ):
            prediction_rows.append(
                {
                    "event_key": row["event_key"],
                    "ticker": row["ticker"],
                    "year": row["_year"],
                    "target": row["_target"],
                    MODEL_PRE_ONLY: float(pred_pre),
                    MODEL_QNA_SEM: float(pred_qna_sem),
                    MODEL_QUESTION_SEM: float(pred_question_sem),
                    MODEL_ANSWER_SEM: float(pred_answer_sem),
                    MODEL_QNA_AUDIO: float(pred_qna_audio),
                    MODEL_QUESTION_AUDIO: float(pred_question_audio),
                    MODEL_SELECTED: float(pred_selected),
                    "selected_model": selected_model,
                }
            )

        overall_y.extend(test_y.tolist())
        for model_name in overall_preds:
            source = selected_pred if model_name == MODEL_SELECTED else test_preds[model_name]
            overall_preds[model_name].extend(source.tolist())

    if not overall_y:
        raise SystemExit("no evaluated held-out test rows were produced")

    overall_y_np = np.asarray(overall_y, dtype=float)
    overall_metrics = {model_name: metrics(overall_y_np, np.asarray(preds, dtype=float)) for model_name, preds in overall_preds.items()}

    keys = [row["event_key"] for row in prediction_rows]
    ref_target = aligned_keyed_predictions(reference_pred_rows, keys, "target")
    ref_hard = aligned_keyed_predictions(reference_pred_rows, keys, MODEL_HARD_ABSTENTION)
    if not np.allclose(ref_target, overall_y_np):
        raise SystemExit("reference hard-abstention predictions do not align with the current test rows")

    significance = {
        f"{MODEL_QUESTION_SEM}__vs__{MODEL_QNA_SEM}": summarize_significance(
            overall_y_np,
            np.asarray(overall_preds[MODEL_QNA_SEM], dtype=float),
            np.asarray(overall_preds[MODEL_QUESTION_SEM], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_QUESTION_AUDIO}__vs__{MODEL_QNA_AUDIO}": summarize_significance(
            overall_y_np,
            np.asarray(overall_preds[MODEL_QNA_AUDIO], dtype=float),
            np.asarray(overall_preds[MODEL_QUESTION_AUDIO], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_SELECTED}__vs__{MODEL_PRE_ONLY}": summarize_significance(
            overall_y_np,
            np.asarray(overall_preds[MODEL_PRE_ONLY], dtype=float),
            np.asarray(overall_preds[MODEL_SELECTED], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_SELECTED}__vs__{MODEL_QNA_AUDIO}": summarize_significance(
            overall_y_np,
            np.asarray(overall_preds[MODEL_QNA_AUDIO], dtype=float),
            np.asarray(overall_preds[MODEL_SELECTED], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_SELECTED}__vs__{MODEL_QUESTION_AUDIO}": summarize_significance(
            overall_y_np,
            np.asarray(overall_preds[MODEL_QUESTION_AUDIO], dtype=float),
            np.asarray(overall_preds[MODEL_SELECTED], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_SELECTED}__vs__{MODEL_HARD_ABSTENTION}": summarize_significance(
            overall_y_np,
            ref_hard,
            np.asarray(overall_preds[MODEL_SELECTED], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_QUESTION_AUDIO}__vs__{MODEL_HARD_ABSTENTION}": summarize_significance(
            overall_y_np,
            ref_hard,
            np.asarray(overall_preds[MODEL_QUESTION_AUDIO], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
    }

    summary = {
        "config": {
            "panel_csv": str(args.panel_csv.resolve()),
            "features_csv": str(args.features_csv.resolve()),
            "old_audio_csv": str(args.old_audio_csv.resolve()),
            "aligned_audio_csv": str(args.aligned_audio_csv.resolve()),
            "qa_csv": str(args.qa_csv.resolve()),
            "hard_reference_predictions_csv": str(args.hard_reference_predictions_csv.resolve()),
            "hard_reference_summary_json": str(args.hard_reference_summary_json.resolve()),
            "target_variant": args.target_variant,
            "include_regimes": sorted(include_regimes),
            "exclude_html_flags": sorted(exclude_html_flags),
            "train_end_year": args.train_end_year,
            "val_year": args.val_year,
            "alphas": alphas,
            "gate_feature": args.gate_feature,
            "gate_quantiles": quantiles,
            "lsa_components": args.lsa_components,
            "role_max_features": args.role_max_features,
        },
        "coverage": coverage,
        "candidate_tickers": candidate_tickers,
        "bundle_meta": bundle_meta,
        "reference": {
            "hard_abstention_summary_best_family": reference_summary.get("best_family"),
            "hard_abstention_test_full_metrics": reference_summary.get("reference", {}).get("test_full_hard_abstention"),
            MODEL_HARD_ABSTENTION: metrics(overall_y_np, ref_hard),
        },
        "overall_metrics": {
            **overall_metrics,
            MODEL_HARD_ABSTENTION: metrics(overall_y_np, ref_hard),
        },
        "selection_counts": dict(selection_counts),
        "skipped": skipped,
        "ticker_summary": ticker_summary,
        "significance": significance,
    }

    write_csv(output_dir / "afterhours_transfer_role_semantic_expert_benchmark_predictions.csv", prediction_rows)
    write_csv(output_dir / "afterhours_transfer_role_semantic_expert_benchmark_selection.csv", selection_rows)
    write_json(output_dir / "afterhours_transfer_role_semantic_expert_benchmark_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
