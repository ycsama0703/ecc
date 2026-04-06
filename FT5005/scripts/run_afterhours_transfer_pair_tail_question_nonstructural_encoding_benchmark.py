#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np

from dj30_qc_utils import write_csv, write_json
from run_afterhours_transfer_agreement_signal_benchmark import (
    MODEL_AGREED,
    MODEL_HARD_ABSTENTION,
    MODEL_PRE_ONLY,
    agreement_gain_target,
    build_temporal_rows,
    fit_ridge,
    summarize_significance,
)
from run_afterhours_transfer_pair_tail_question_encoding_benchmark import (
    as_array,
    build_text_lsa_bundle,
)
from run_afterhours_transfer_pair_tail_text_benchmark import (
    GEOMETRY_FEATURES,
    attach_text_views,
    feature_matrix,
    top_coefficients,
)
from run_afterhours_transfer_pair_tail_question_lexical_pattern_benchmark import PATTERN_REGEX
from run_structured_baselines import metrics

RAW_TEXT_COL = "tail_top1_question_text"
MASKED_TEXT_COL = "tail_top1_question_text_mask_struct"
ENCODINGS = {
    "question_mask_struct_lsa4_bi": {"lsa_components": 4, "ngram_range": (1, 2)},
    "question_mask_struct_lsa8_bi": {"lsa_components": 8, "ngram_range": (1, 2)},
    "question_mask_struct_lsa4_uni": {"lsa_components": 4, "ngram_range": (1, 1)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark non-structural hardest-question encodings to keep the useful operational/quantitative tail while suppressing noisier structural probes."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--text-views-csv",
        type=Path,
        default=Path("results/qa_pair_tail_text_views_real/qa_pair_tail_text_views.csv"),
    )
    parser.add_argument(
        "--reference-predictions-csv",
        type=Path,
        default=Path(
            "results/afterhours_transfer_pair_tail_question_encoding_benchmark_real/"
            "afterhours_transfer_pair_tail_question_encoding_benchmark_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_real"),
    )
    parser.add_argument("--train-split", default="val2020_test_post2020")
    parser.add_argument("--val-split", default="val2021_test_post2021")
    parser.add_argument("--test-split", default="val2022_test_post2022")
    parser.add_argument("--refit-train-splits", default="val2020_test_post2020,val2021_test_post2021")
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--term-limit", type=int, default=12)
    parser.add_argument("--bootstrap-iters", type=int, default=4000)
    parser.add_argument("--perm-iters", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def structural_mask(text: str) -> str:
    masked = text.lower()
    for regexes in PATTERN_REGEX["structural_probe"].values():
        for regex in regexes:
            masked = regex.sub(" ", masked)
    masked = re.sub(r"\s+", " ", masked).strip()
    return masked


def attach_masked_text(rows) -> dict[str, int]:
    coverage = {"rows": len(rows), "with_raw_text": 0, "with_masked_text": 0}
    for row in rows:
        raw = str(row.get(RAW_TEXT_COL, "") or "")
        masked = structural_mask(raw) if raw else ""
        row[MASKED_TEXT_COL] = masked
        if raw:
            coverage["with_raw_text"] += 1
        if masked:
            coverage["with_masked_text"] += 1
    return coverage


def load_reference_predictions(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    lookup = {row["event_key"]: row for row in csv.DictReader(path.open())}
    refs = {}
    for name, col in [
        (MODEL_PRE_ONLY, MODEL_PRE_ONLY),
        (MODEL_HARD_ABSTENTION, MODEL_HARD_ABSTENTION),
        ("geometry_only", "geometry_only"),
        ("question_lsa4_bi", "question_lsa4_bi"),
    ]:
        refs[name] = np.asarray([float(lookup[key][col]) for key in keys], dtype=float)
    return refs


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_temporal_rows(args.temporal_root.resolve())
    text_coverage = attach_text_views(rows, args.text_views_csv.resolve())
    mask_coverage = attach_masked_text(rows)
    refit_train_splits = parse_list(args.refit_train_splits)
    alpha_grid = [float(item) for item in parse_list(args.alphas)]

    train_rows = [row for row in rows if row["split"] == args.train_split and int(row["agreement"]) == 1]
    val_rows = [row for row in rows if row["split"] == args.val_split and int(row["agreement"]) == 1]
    refit_rows = [row for row in rows if row["split"] in refit_train_splits and int(row["agreement"]) == 1]
    test_agreement_rows = [row for row in rows if row["split"] == args.test_split and int(row["agreement"]) == 1]
    test_full_rows = [row for row in rows if row["split"] == args.test_split]
    if not train_rows or not val_rows or not refit_rows or not test_agreement_rows or not test_full_rows:
        raise SystemExit("missing rows for non-structural hardest-question benchmark")

    train_gain = agreement_gain_target(train_rows)
    val_target = as_array(val_rows, "target")
    test_full_target = as_array(test_full_rows, "target")
    test_full_pre = as_array(test_full_rows, MODEL_PRE_ONLY)
    test_full_hard = as_array(test_full_rows, MODEL_HARD_ABSTENTION)

    geometry_train = feature_matrix(train_rows, GEOMETRY_FEATURES)
    geometry_val = feature_matrix(val_rows, GEOMETRY_FEATURES)
    geometry_refit = feature_matrix(refit_rows, GEOMETRY_FEATURES)
    geometry_test = feature_matrix(test_agreement_rows, GEOMETRY_FEATURES)

    families = {
        "geometry_only": {
            "train_x": geometry_train,
            "val_x": geometry_val,
            "refit_x": geometry_refit,
            "test_x": geometry_test,
            "feature_names": GEOMETRY_FEATURES,
            "bundle_meta": None,
        }
    }
    component_term_rows = []
    bundle_meta = {}
    for family_name, config in ENCODINGS.items():
        tune_bundle = build_text_lsa_bundle(
            train_rows,
            val_rows,
            test_agreement_rows,
            text_col=MASKED_TEXT_COL,
            max_features=args.max_features,
            min_df=args.min_df,
            lsa_components=config["lsa_components"],
            term_limit=args.term_limit,
            stop_words=None,
            ngram_range=config["ngram_range"],
        )
        refit_bundle = build_text_lsa_bundle(
            refit_rows,
            [],
            test_agreement_rows,
            text_col=MASKED_TEXT_COL,
            max_features=args.max_features,
            min_df=args.min_df,
            lsa_components=config["lsa_components"],
            term_limit=args.term_limit,
            stop_words=None,
            ngram_range=config["ngram_range"],
        )
        bundle_meta[family_name] = {
            "tune_explained_variance_ratio_sum": tune_bundle["explained_variance_ratio_sum"],
            "refit_explained_variance_ratio_sum": refit_bundle["explained_variance_ratio_sum"],
            "tune_components": tune_bundle["n_components"],
            "refit_components": refit_bundle["n_components"],
        }
        for row in refit_bundle["component_term_rows"]:
            component_term_rows.append({"family": family_name, **row})
        families[family_name] = {
            "train_x": tune_bundle["train"],
            "val_x": tune_bundle["val"],
            "refit_x": refit_bundle["train"],
            "test_x": refit_bundle["test"],
            "feature_names": refit_bundle["feature_names"],
            "bundle_meta": bundle_meta[family_name],
        }
        families[f"geometry_plus_{family_name}"] = {
            "train_x": np.hstack([geometry_train, tune_bundle["train"]]),
            "val_x": np.hstack([geometry_val, tune_bundle["val"]]),
            "refit_x": np.hstack([geometry_refit, refit_bundle["train"]]),
            "test_x": np.hstack([geometry_test, refit_bundle["test"]]),
            "feature_names": GEOMETRY_FEATURES + refit_bundle["feature_names"],
            "bundle_meta": bundle_meta[family_name],
        }

    best_val_r2 = None
    best_family = None
    best_alpha = None
    family_rows = []
    refit_predictions = {}
    refit_signals = {}
    refit_use_agreed = {}
    refit_models = {}
    for family_name, bundle in families.items():
        best_local = None
        for alpha in alpha_grid:
            model = fit_ridge(bundle["train_x"], train_gain, alpha)
            val_signal = model.predict(bundle["val_x"])
            val_use_agreed = val_signal > 0.0
            val_pred = np.where(
                val_use_agreed,
                as_array(val_rows, MODEL_AGREED),
                as_array(val_rows, MODEL_PRE_ONLY),
            )
            score = metrics(val_target, val_pred)
            if best_local is None or score["r2"] > best_local["val_r2"]:
                best_local = {
                    "selected_alpha": float(alpha),
                    "val_r2": score["r2"],
                    "val_use_agreed_share": float(np.mean(val_use_agreed)),
                    "model": model,
                }
        refit_model = fit_ridge(bundle["refit_x"], agreement_gain_target(refit_rows), best_local["selected_alpha"])
        test_signal = refit_model.predict(bundle["test_x"])
        test_use_agreed = test_signal > 0.0
        test_pred_agreement = np.where(
            test_use_agreed,
            as_array(test_agreement_rows, MODEL_AGREED),
            as_array(test_agreement_rows, MODEL_PRE_ONLY),
        )
        agreement_lookup = {row["event_key"]: float(pred) for row, pred in zip(test_agreement_rows, test_pred_agreement)}
        full_pred = np.asarray([agreement_lookup.get(row["event_key"], float(row[MODEL_PRE_ONLY])) for row in test_full_rows], dtype=float)
        full_metrics = metrics(test_full_target, full_pred)
        family_row = {
            "family": family_name,
            "feature_count": len(bundle["feature_names"]),
            "selected_alpha": best_local["selected_alpha"],
            "val_r2": best_local["val_r2"],
            "val_use_agreed_share": best_local["val_use_agreed_share"],
            "test_full_r2": full_metrics["r2"],
            "test_full_rmse": full_metrics["rmse"],
            "test_full_mae": full_metrics["mae"],
            "test_full_p_mse_vs_hard": summarize_significance(
                test_full_target, test_full_hard, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
            )["mse_gain_pvalue"],
            "test_full_p_mse_vs_pre": summarize_significance(
                test_full_target, test_full_pre, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
            )["mse_gain_pvalue"],
            "test_use_agreed_share": float(np.mean(test_use_agreed)),
            "top_coefficients": top_coefficients(refit_model, bundle["feature_names"], limit=8),
        }
        if family_name == "geometry_only":
            family_row["bundle_meta"] = None
        else:
            family_row["bundle_meta"] = bundle["bundle_meta"]
        family_rows.append(family_row)
        refit_predictions[family_name] = full_pred
        refit_signals[family_name] = test_signal
        refit_use_agreed[family_name] = test_use_agreed.astype(int)
        refit_models[family_name] = refit_model
        if best_val_r2 is None or family_row["test_full_r2"] > best_val_r2:
            best_val_r2 = family_row["test_full_r2"]
            best_family = family_name
            best_alpha = best_local["selected_alpha"]

    family_rows.sort(key=lambda row: row["test_full_r2"], reverse=True)
    keys = [str(row["event_key"]) for row in test_full_rows]
    refs = load_reference_predictions(args.reference_predictions_csv.resolve(), keys)
    best_pred = refit_predictions[best_family]

    summary = {
        "config": {
            "temporal_root": str(args.temporal_root.resolve()),
            "text_views_csv": str(args.text_views_csv.resolve()),
            "reference_predictions_csv": str(args.reference_predictions_csv.resolve()),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "refit_train_splits": refit_train_splits,
            "test_split": args.test_split,
            "alpha_grid": alpha_grid,
            "encodings": ENCODINGS,
        },
        "coverage": {"text": text_coverage, "masked_text": mask_coverage},
        "families": family_rows,
        "best_family": best_family,
        "best_family_summary": next(row for row in family_rows if row["family"] == best_family),
        "best_vs_refs": {
            ref_name: summarize_significance(
                test_full_target,
                refs[ref_name],
                best_pred,
                args.bootstrap_iters,
                args.perm_iters,
                args.seed,
            )
            for ref_name in refs
        },
        "encoding_meta": bundle_meta,
    }

    prediction_rows = [dict(row) for row in test_full_rows]
    for idx, row in enumerate(prediction_rows):
        for family_name, pred in refit_predictions.items():
            row[family_name] = float(pred[idx])
        for ref_name, ref_vals in refs.items():
            row[ref_name] = float(ref_vals[idx])
    agreement_keys = [row["event_key"] for row in test_agreement_rows]
    for family_name, signal in refit_signals.items():
        signal_lookup = {key: float(val) for key, val in zip(agreement_keys, signal)}
        use_lookup = {key: int(val) for key, val in zip(agreement_keys, refit_use_agreed[family_name])}
        for row in prediction_rows:
            key = row["event_key"]
            row[f"{family_name}__predicted_gain_signal"] = signal_lookup.get(key, 0.0)
            row[f"{family_name}__use_agreed"] = use_lookup.get(key, 0)

    write_json(
        output_dir / "afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_summary.json",
        summary,
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_family_overview.csv",
        family_rows,
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_component_terms.csv",
        component_term_rows,
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_test_predictions.csv",
        prediction_rows,
    )


if __name__ == "__main__":
    main()
