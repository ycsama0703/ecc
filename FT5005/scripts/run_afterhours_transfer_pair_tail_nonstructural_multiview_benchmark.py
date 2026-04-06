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
from run_afterhours_transfer_pair_tail_text_benchmark import attach_text_views, top_coefficients
from run_afterhours_transfer_pair_tail_question_lexical_pattern_benchmark import PATTERN_REGEX
from run_structured_baselines import metrics

VIEW_COLS = {
    "question": "tail_top1_question_text",
    "answer": "tail_top1_answer_text",
    "qa": "tail_top1_qa_text",
}
MASKED_COLS = {
    name: f"{col}_mask_struct" for name, col in VIEW_COLS.items()
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compact non-structural multi-view local pair representations."
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
            "results/afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_real/"
            "afterhours_transfer_pair_tail_question_nonstructural_encoding_benchmark_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_real"),
    )
    parser.add_argument("--train-split", default="val2020_test_post2020")
    parser.add_argument("--val-split", default="val2021_test_post2021")
    parser.add_argument("--test-split", default="val2022_test_post2022")
    parser.add_argument("--refit-train-splits", default="val2020_test_post2020,val2021_test_post2021")
    parser.add_argument("--alphas", default="0.01,0.1,1,10,100,1000")
    parser.add_argument("--max-features", type=int, default=4000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=4)
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


def attach_masked_views(rows) -> dict[str, int]:
    coverage = {"rows": len(rows)}
    for view, raw_col in VIEW_COLS.items():
        masked_col = MASKED_COLS[view]
        coverage[f"with_{view}_raw"] = 0
        coverage[f"with_{view}_masked"] = 0
        for row in rows:
            raw = str(row.get(raw_col, "") or "")
            masked = structural_mask(raw) if raw else ""
            row[masked_col] = masked
            if raw:
                coverage[f"with_{view}_raw"] += 1
            if masked:
                coverage[f"with_{view}_masked"] += 1
    return coverage


def load_reference_predictions(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    lookup = {row["event_key"]: row for row in csv.DictReader(path.open())}
    refs = {}
    for name, col in [
        (MODEL_PRE_ONLY, MODEL_PRE_ONLY),
        (MODEL_HARD_ABSTENTION, MODEL_HARD_ABSTENTION),
        ("question_lsa4_bi", "question_lsa4_bi"),
        ("question_mask_struct_lsa4_bi", "question_mask_struct_lsa4_bi"),
    ]:
        refs[name] = np.asarray([float(lookup[key][col]) for key in keys], dtype=float)
    return refs


def make_single_bundle(train_rows, val_rows, test_rows, *, text_col: str, args: argparse.Namespace):
    return build_text_lsa_bundle(
        train_rows,
        val_rows,
        test_rows,
        text_col=text_col,
        max_features=args.max_features,
        min_df=args.min_df,
        lsa_components=args.lsa_components,
        term_limit=args.term_limit,
        stop_words=None,
        ngram_range=(1, 2),
    )


def build_families(train_rows, val_rows, refit_rows, test_rows, args: argparse.Namespace):
    families = {}
    component_term_rows = []

    tune_bundles = {}
    refit_bundles = {}
    for view, masked_col in MASKED_COLS.items():
        tune = make_single_bundle(train_rows, val_rows, test_rows, text_col=masked_col, args=args)
        refit = make_single_bundle(refit_rows, [], test_rows, text_col=masked_col, args=args)
        tune_bundles[view] = tune
        refit_bundles[view] = refit
        for row in refit["component_term_rows"]:
            component_term_rows.append({"family": f"{view}_mask_struct_lsa4_bi", **row})

    families["question_mask_struct_lsa4_bi"] = {
        "train_x": tune_bundles["question"]["train"],
        "val_x": tune_bundles["question"]["val"],
        "refit_x": refit_bundles["question"]["train"],
        "test_x": refit_bundles["question"]["test"],
        "feature_names": refit_bundles["question"]["feature_names"],
    }
    families["answer_mask_struct_lsa4_bi"] = {
        "train_x": tune_bundles["answer"]["train"],
        "val_x": tune_bundles["answer"]["val"],
        "refit_x": refit_bundles["answer"]["train"],
        "test_x": refit_bundles["answer"]["test"],
        "feature_names": refit_bundles["answer"]["feature_names"],
    }
    families["qa_mask_struct_lsa4_bi"] = {
        "train_x": tune_bundles["qa"]["train"],
        "val_x": tune_bundles["qa"]["val"],
        "refit_x": refit_bundles["qa"]["train"],
        "test_x": refit_bundles["qa"]["test"],
        "feature_names": refit_bundles["qa"]["feature_names"],
    }
    families["question_plus_answer_mask_struct_lsa4_bi"] = {
        "train_x": np.hstack([tune_bundles["question"]["train"], tune_bundles["answer"]["train"]]),
        "val_x": np.hstack([tune_bundles["question"]["val"], tune_bundles["answer"]["val"]]),
        "refit_x": np.hstack([refit_bundles["question"]["train"], refit_bundles["answer"]["train"]]),
        "test_x": np.hstack([refit_bundles["question"]["test"], refit_bundles["answer"]["test"]]),
        "feature_names": refit_bundles["question"]["feature_names"] + refit_bundles["answer"]["feature_names"],
    }
    families["question_plus_qa_mask_struct_lsa4_bi"] = {
        "train_x": np.hstack([tune_bundles["question"]["train"], tune_bundles["qa"]["train"]]),
        "val_x": np.hstack([tune_bundles["question"]["val"], tune_bundles["qa"]["val"]]),
        "refit_x": np.hstack([refit_bundles["question"]["train"], refit_bundles["qa"]["train"]]),
        "test_x": np.hstack([refit_bundles["question"]["test"], refit_bundles["qa"]["test"]]),
        "feature_names": refit_bundles["question"]["feature_names"] + refit_bundles["qa"]["feature_names"],
    }
    return families, component_term_rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_temporal_rows(args.temporal_root.resolve())
    text_coverage = attach_text_views(rows, args.text_views_csv.resolve())
    mask_coverage = attach_masked_views(rows)
    refit_train_splits = parse_list(args.refit_train_splits)
    alpha_grid = [float(item) for item in parse_list(args.alphas)]

    train_rows = [row for row in rows if row["split"] == args.train_split and int(row["agreement"]) == 1]
    val_rows = [row for row in rows if row["split"] == args.val_split and int(row["agreement"]) == 1]
    refit_rows = [row for row in rows if row["split"] in refit_train_splits and int(row["agreement"]) == 1]
    test_agreement_rows = [row for row in rows if row["split"] == args.test_split and int(row["agreement"]) == 1]
    test_full_rows = [row for row in rows if row["split"] == args.test_split]
    if not train_rows or not val_rows or not refit_rows or not test_agreement_rows or not test_full_rows:
        raise SystemExit("missing rows for non-structural multiview benchmark")

    families, component_term_rows = build_families(train_rows, val_rows, refit_rows, test_agreement_rows, args)

    train_gain = agreement_gain_target(train_rows)
    val_target = as_array(val_rows, "target")
    test_full_target = as_array(test_full_rows, "target")
    test_full_pre = as_array(test_full_rows, MODEL_PRE_ONLY)
    test_full_hard = as_array(test_full_rows, MODEL_HARD_ABSTENTION)

    family_rows = []
    prediction_rows = [dict(row) for row in test_full_rows]
    family_predictions = {}

    for family_name, payload in families.items():
        best_alpha = None
        best_val_r2 = None
        best_val_payload = None
        for alpha in alpha_grid:
            model = fit_ridge(payload["train_x"], train_gain, alpha)
            val_signal = model.predict(payload["val_x"])
            val_pred = np.where(val_signal > 0.0, as_array(val_rows, MODEL_AGREED), as_array(val_rows, MODEL_PRE_ONLY))
            score = metrics(val_target, val_pred)
            if best_val_r2 is None or score["r2"] > best_val_r2:
                best_val_r2 = score["r2"]
                best_alpha = float(alpha)
                best_val_payload = {
                    "val_r2": score["r2"],
                    "val_use_agreed_share": float(np.mean(val_signal > 0.0)),
                }

        model = fit_ridge(payload["refit_x"], agreement_gain_target(refit_rows), best_alpha)
        test_signal = model.predict(payload["test_x"])
        test_agreement_pred = np.where(
            test_signal > 0.0,
            as_array(test_agreement_rows, MODEL_AGREED),
            as_array(test_agreement_rows, MODEL_PRE_ONLY),
        )
        pred_lookup = {row["event_key"]: float(pred) for row, pred in zip(test_agreement_rows, test_agreement_pred)}
        signal_lookup = {row["event_key"]: float(sig) for row, sig in zip(test_agreement_rows, test_signal)}
        use_lookup = {row["event_key"]: int(sig > 0.0) for row, sig in zip(test_agreement_rows, test_signal)}
        full_pred = np.asarray([pred_lookup.get(row["event_key"], float(row[MODEL_PRE_ONLY])) for row in test_full_rows], dtype=float)
        family_predictions[family_name] = full_pred

        for row in prediction_rows:
            key = row["event_key"]
            row[family_name] = pred_lookup.get(key, float(row[MODEL_PRE_ONLY]))
            row[f"{family_name}__predicted_gain_signal"] = signal_lookup.get(key, 0.0)
            row[f"{family_name}__use_agreed"] = use_lookup.get(key, 0)

        test_full_metrics = metrics(test_full_target, full_pred)
        family_rows.append(
            {
                "family": family_name,
                "feature_count": len(payload["feature_names"]),
                "selected_alpha": best_alpha,
                "val_r2": best_val_payload["val_r2"],
                "val_use_agreed_share": best_val_payload["val_use_agreed_share"],
                "test_full_r2": test_full_metrics["r2"],
                "test_full_rmse": test_full_metrics["rmse"],
                "test_full_mae": test_full_metrics["mae"],
                "test_full_p_mse_vs_hard": summarize_significance(
                    test_full_target, test_full_hard, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
                )["mse_gain_pvalue"],
                "test_full_p_mse_vs_pre": summarize_significance(
                    test_full_target, test_full_pre, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
                )["mse_gain_pvalue"],
                "test_use_agreed_share": float(np.mean(test_signal > 0.0)),
                "top_coefficients": top_coefficients(model, payload["feature_names"]),
            }
        )

    family_rows.sort(key=lambda row: row["test_full_r2"], reverse=True)
    best_family = family_rows[0]["family"]
    keys = [str(row["event_key"]) for row in test_full_rows]
    refs = load_reference_predictions(args.reference_predictions_csv.resolve(), keys)
    best_vs_refs = {
        name: summarize_significance(
            test_full_target, ref_vals, family_predictions[best_family], args.bootstrap_iters, args.perm_iters, args.seed
        )
        for name, ref_vals in refs.items()
    }

    for idx, row in enumerate(prediction_rows):
        for name, ref_vals in refs.items():
            row[name] = float(ref_vals[idx])

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
            "lsa_components": args.lsa_components,
        },
        "coverage": {"text": text_coverage, "masked_text": mask_coverage},
        "families": family_rows,
        "best_family": best_family,
        "best_family_summary": next(row for row in family_rows if row["family"] == best_family),
        "best_vs_refs": best_vs_refs,
    }

    write_json(
        output_dir / "afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_summary.json",
        summary,
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_family_overview.csv",
        family_rows,
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_component_terms.csv",
        component_term_rows,
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_nonstructural_multiview_benchmark_test_predictions.csv",
        prediction_rows,
    )


if __name__ == "__main__":
    main()
