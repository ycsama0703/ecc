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
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    if user_site_removed and isinstance(user_site, str) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json
from run_afterhours_audio_upgrade_benchmark import (
    A4_STRUCTURED_FEATURES,
    PRE_CALL_MARKET_FEATURES,
    build_compressed_dense_bundle,
)
from run_afterhours_transfer_expert_selection import summarize_significance
from run_afterhours_transfer_qa_signal_benchmark import choose_gate_threshold, constant_prior, parse_quantiles
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle
from run_offhours_shock_ablations import regime_label
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets

RESPONSIVENESS_FACTOR_FEATURES = [
    "qa_pair_answer_forward_rate_mean",
    "qa_bench_coverage_mean",
    "qa_bench_direct_early_score_mean",
    "qa_bench_evasion_score_mean",
    "qa_bench_direct_answer_share",
]

CONTENT_FACTOR_FEATURES = [
    "qa_bench_question_complexity_mean",
    "qa_bench_numeric_rate_mean",
    "qa_bench_numeric_mismatch_share",
    "qa_bench_justification_rate_mean",
    "qa_bench_external_attr_rate_mean",
    "qa_bench_subjective_rate_mean",
]

MODEL_PRE_ONLY = "residual_pre_call_market_only"
MODEL_RESPONSIVENESS_EXPERT = "residual_pre_call_market_plus_a4_plus_responsiveness_factor_observability_gate"
MODEL_CONTENT_EXPERT = "residual_pre_call_market_plus_a4_plus_content_accountability_factor_observability_gate"
MODEL_SELECTED = "validation_selected_compact_factor_expert"
MODEL_HARD_ABSTENTION = "agreement_pre_only_abstention"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark minimal transfer-side integration of distilled factor experts against the current hard-abstention shell."
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
        "--qa-csv",
        type=Path,
        default=Path("results/qa_benchmark_features_v2_real/qa_benchmark_features.csv"),
    )
    parser.add_argument(
        "--aligned-audio-csv",
        type=Path,
        default=Path("results/role_aware_aligned_audio_afterhours_clean_real/event_role_aware_aligned_acoustic_features.csv"),
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
        default=Path("results/afterhours_transfer_factor_expert_integration_role_aware_audio_real"),
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
    parser.add_argument("--gate-feature", default="a4_strict_row_share")
    parser.add_argument("--gate-quantiles", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_prefixed_lookup(path: Path, prefix: str, meta_fields: set[str]) -> dict[str, dict[str, str]]:
    lookup = {}
    for row in load_csv_rows(path.resolve()):
        event_key = row.get("event_key", "")
        if not event_key:
            continue
        item = {}
        for key, value in row.items():
            if key in meta_fields:
                continue
            item[f"{prefix}{key}"] = value
        lookup[event_key] = item
    return lookup


def load_reference_predictions(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def build_joined_rows(
    panel_csv: Path,
    features_csv: Path,
    qa_csv: Path,
    aligned_audio_csv: Path,
    target_variant: str,
    eps: float,
    include_regimes: set[str],
    exclude_html_flags: set[str],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    feature_lookup = {row.get("event_key", ""): row for row in load_csv_rows(features_csv.resolve()) if row.get("event_key")}
    qa_lookup = {row.get("event_key", ""): row for row in load_csv_rows(qa_csv.resolve()) if row.get("event_key")}
    aligned_lookup = load_prefixed_lookup(
        aligned_audio_csv.resolve(),
        prefix="aligned_audio__",
        meta_fields={"event_key", "ticker", "year", "quarter"},
    )

    coverage = {
        "panel_rows": 0,
        "with_features": 0,
        "with_qa": 0,
        "with_aligned_audio": 0,
        "with_all_side_inputs": 0,
        "joined_rows": 0,
    }

    rows = []
    for row in load_csv_rows(panel_csv.resolve()):
        coverage["panel_rows"] += 1
        event_key = row.get("event_key", "")
        year_value = row.get("year", "")
        feature_row = feature_lookup.get(event_key)
        qa_row = qa_lookup.get(event_key)
        aligned_row = aligned_lookup.get(event_key)

        if feature_row is not None:
            coverage["with_features"] += 1
        if qa_row is not None:
            coverage["with_qa"] += 1
        if aligned_row is not None:
            coverage["with_aligned_audio"] += 1
        if feature_row is not None and qa_row is not None and aligned_row is not None:
            coverage["with_all_side_inputs"] += 1

        if not event_key or not year_value or feature_row is None or qa_row is None or aligned_row is None:
            continue

        merged = dict(row)
        merged.update(feature_row)
        merged.update(qa_row)
        merged.update(aligned_row)
        merged["_year"] = int(float(year_value))

        html_flag = (merged.get("html_integrity_flag") or "").strip().lower()
        if html_flag in exclude_html_flags:
            continue
        reg = regime_label(merged)
        if reg not in include_regimes:
            continue
        target_value = derived_targets(merged, eps).get(target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        merged["_target"] = float(target_value)
        rows.append(merged)

    coverage["joined_rows"] = len(rows)
    return rows, coverage


def build_factor_bundle(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
    prefix: str,
) -> dict[str, object]:
    def matrix(rows: list[dict[str, str]]) -> np.ndarray:
        return np.asarray(
            [[safe_float(row.get(feature_name)) or 0.0 for feature_name in feature_names] for row in rows],
            dtype=float,
        )

    train_x = matrix(train_rows)
    val_x = matrix(val_rows)
    test_x = matrix(test_rows)

    scaler = StandardScaler().fit(train_x)
    train_x_scaled = scaler.transform(train_x)
    val_x_scaled = scaler.transform(val_x)
    test_x_scaled = scaler.transform(test_x)

    pca = PCA(n_components=1, random_state=42)
    pca.fit(train_x_scaled)

    train_factor = pca.transform(train_x_scaled)
    val_factor = pca.transform(val_x_scaled)
    test_factor = pca.transform(test_x_scaled)
    loading_rows = [
        {"feature": feature, "loading": float(weight)}
        for feature, weight in sorted(
            zip(feature_names, pca.components_[0].tolist()), key=lambda item: abs(item[1]), reverse=True
        )
    ]
    return {
        "train": train_factor,
        "val": val_factor,
        "test": test_factor,
        "n_components": 1,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_[0]),
        "feature_names": feature_names,
        "prefix": prefix,
        "loading_rows": loading_rows,
    }


def aligned_keyed_predictions(pred_rows: list[dict[str, str]], keys: list[str], column: str) -> np.ndarray:
    lookup = {row["event_key"]: row for row in pred_rows}
    return np.asarray([float(lookup[key][column]) for key in keys], dtype=float)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    quantiles = parse_quantiles(args.gate_quantiles)

    rows, coverage = build_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.qa_csv,
        args.aligned_audio_csv,
        args.target_variant,
        args.eps,
        include_regimes,
        exclude_html_flags,
    )
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
    ticker_summary = {}
    factor_loading_rows = []
    overall_y: list[float] = []
    overall_preds = {
        MODEL_PRE_ONLY: [],
        MODEL_RESPONSIVENESS_EXPERT: [],
        MODEL_CONTENT_EXPERT: [],
        MODEL_SELECTED: [],
    }
    selection_counts = Counter()

    bundle_meta = None
    skipped = {}

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
            "aligned_audio_svd": build_compressed_dense_bundle(
                train_rows,
                val_rows,
                test_rows,
                aligned_feature_names,
                8,
                prefix="aligned_audio_svd",
            ),
            "responsiveness_factor": build_factor_bundle(
                train_rows,
                val_rows,
                test_rows,
                RESPONSIVENESS_FACTOR_FEATURES,
                prefix="responsiveness_factor",
            ),
            "content_factor": build_factor_bundle(
                train_rows,
                val_rows,
                test_rows,
                CONTENT_FACTOR_FEATURES,
                prefix="content_factor",
            ),
        }
        if bundle_meta is None:
            bundle_meta = {
                "responsiveness_factor": {
                    "feature_names": RESPONSIVENESS_FACTOR_FEATURES,
                    "explained_variance_ratio_sum": bundles["responsiveness_factor"]["explained_variance_ratio_sum"],
                },
                "content_factor": {
                    "feature_names": CONTENT_FACTOR_FEATURES,
                    "explained_variance_ratio_sum": bundles["content_factor"]["explained_variance_ratio_sum"],
                },
                "aligned_audio_svd": {
                    "input_feature_count": len(aligned_feature_names),
                    "n_components": bundles["aligned_audio_svd"]["n_components"],
                    "explained_variance_ratio_sum": bundles["aligned_audio_svd"]["explained_variance_ratio_sum"],
                },
                "qna_lsa_components": int(bundles["qna_lsa"]["train"].shape[1]),
            }

        factor_loading_rows.extend(
            [
                {"ticker": ticker, "factor_family": "responsiveness", **row}
                for row in bundles["responsiveness_factor"]["loading_rows"]
            ]
        )
        factor_loading_rows.extend(
            [{"ticker": ticker, "factor_family": "content", **row} for row in bundles["content_factor"]["loading_rows"]]
        )

        model_specs = {
            MODEL_PRE_ONLY: ["pre_call_market"],
            MODEL_RESPONSIVENESS_EXPERT: ["pre_call_market", "a4", "responsiveness_factor"],
            MODEL_CONTENT_EXPERT: ["pre_call_market", "a4", "content_factor"],
        }

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
            val_scores = np.asarray([safe_float(row.get(args.gate_feature)) or 0.0 for row in val_rows], dtype=float)
            test_scores = np.asarray([safe_float(row.get(args.gate_feature)) or 0.0 for row in test_rows], dtype=float)
            val_preds[model_name] = np.where(val_scores >= threshold, val_pred_np, val_prior)
            test_preds[model_name] = np.where(test_scores >= threshold, test_pred_np, test_prior)
            gate_meta[model_name] = threshold_meta

        validation_metrics = {model_name: metrics(val_y, val_preds[model_name]) for model_name in model_specs}
        test_metrics = {model_name: metrics(test_y, test_preds[model_name]) for model_name in model_specs}
        selected_model = max(validation_metrics, key=lambda model_name: validation_metrics[model_name]["r2"])
        selection_counts[selected_model] += 1

        selected_pred = test_preds[selected_model]
        ticker_summary[ticker] = {
            "n_test": len(test_rows),
            "selected_model": selected_model,
            "validation_r2": {model_name: payload["r2"] for model_name, payload in validation_metrics.items()},
            "test_r2": {model_name: payload["r2"] for model_name, payload in test_metrics.items()},
            "gate_meta": gate_meta,
        }

        for row, pred_pre, pred_resp, pred_content, pred_selected in zip(
            test_rows,
            test_preds[MODEL_PRE_ONLY],
            test_preds[MODEL_RESPONSIVENESS_EXPERT],
            test_preds[MODEL_CONTENT_EXPERT],
            selected_pred,
        ):
            prediction_rows.append(
                {
                    "event_key": row["event_key"],
                    "ticker": row["ticker"],
                    "year": row["_year"],
                    "target": row["_target"],
                    MODEL_PRE_ONLY: float(pred_pre),
                    MODEL_RESPONSIVENESS_EXPERT: float(pred_resp),
                    MODEL_CONTENT_EXPERT: float(pred_content),
                    MODEL_SELECTED: float(pred_selected),
                    "selected_model": selected_model,
                }
            )

        selection_rows.append(
            {
                "ticker": ticker,
                "n_test": len(test_rows),
                "selected_model": selected_model,
                "val_r2_pre": validation_metrics[MODEL_PRE_ONLY]["r2"],
                "val_r2_responsiveness": validation_metrics[MODEL_RESPONSIVENESS_EXPERT]["r2"],
                "val_r2_content": validation_metrics[MODEL_CONTENT_EXPERT]["r2"],
                "test_r2_pre": test_metrics[MODEL_PRE_ONLY]["r2"],
                "test_r2_responsiveness": test_metrics[MODEL_RESPONSIVENESS_EXPERT]["r2"],
                "test_r2_content": test_metrics[MODEL_CONTENT_EXPERT]["r2"],
                "test_r2_selected": metrics(test_y, selected_pred)["r2"],
            }
        )

        overall_y.extend(test_y.tolist())
        overall_preds[MODEL_PRE_ONLY].extend(test_preds[MODEL_PRE_ONLY].tolist())
        overall_preds[MODEL_RESPONSIVENESS_EXPERT].extend(test_preds[MODEL_RESPONSIVENESS_EXPERT].tolist())
        overall_preds[MODEL_CONTENT_EXPERT].extend(test_preds[MODEL_CONTENT_EXPERT].tolist())
        overall_preds[MODEL_SELECTED].extend(selected_pred.tolist())

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
        f"{MODEL_SELECTED}__vs__{MODEL_PRE_ONLY}": summarize_significance(
            overall_y_np,
            np.asarray(overall_preds[MODEL_PRE_ONLY], dtype=float),
            np.asarray(overall_preds[MODEL_SELECTED], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_SELECTED}__vs__{MODEL_RESPONSIVENESS_EXPERT}": summarize_significance(
            overall_y_np,
            np.asarray(overall_preds[MODEL_RESPONSIVENESS_EXPERT], dtype=float),
            np.asarray(overall_preds[MODEL_SELECTED], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
        f"{MODEL_SELECTED}__vs__{MODEL_CONTENT_EXPERT}": summarize_significance(
            overall_y_np,
            np.asarray(overall_preds[MODEL_CONTENT_EXPERT], dtype=float),
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
        f"{MODEL_PRE_ONLY}__vs__{MODEL_HARD_ABSTENTION}": summarize_significance(
            overall_y_np,
            ref_hard,
            np.asarray(overall_preds[MODEL_PRE_ONLY], dtype=float),
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        ),
    }

    summary = {
        "config": {
            "panel_csv": str(args.panel_csv.resolve()),
            "features_csv": str(args.features_csv.resolve()),
            "qa_csv": str(args.qa_csv.resolve()),
            "aligned_audio_csv": str(args.aligned_audio_csv.resolve()),
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

    write_csv(output_dir / "afterhours_transfer_factor_expert_integration_predictions.csv", prediction_rows)
    write_csv(output_dir / "afterhours_transfer_factor_expert_integration_selection.csv", selection_rows)
    write_csv(output_dir / "afterhours_transfer_factor_expert_integration_factor_loadings.csv", factor_loading_rows)
    write_json(output_dir / "afterhours_transfer_factor_expert_integration_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
