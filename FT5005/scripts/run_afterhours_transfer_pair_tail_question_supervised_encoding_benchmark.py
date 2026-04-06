#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
import site
import sys
from pathlib import Path

import numpy as np

user_site = site.getusersitepackages()
user_site_removed = False
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)
    user_site_removed = True

try:
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
except ModuleNotFoundError:
    if user_site_removed and isinstance(user_site, str) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.feature_extraction.text import TfidfVectorizer

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
from run_afterhours_transfer_pair_tail_question_encoding_benchmark import as_array
from run_afterhours_transfer_pair_tail_question_lexical_pattern_benchmark import PATTERN_REGEX
from run_afterhours_transfer_pair_tail_text_benchmark import attach_text_views, top_coefficients
from run_structured_baselines import metrics

RAW_TEXT_COL = "tail_top1_question_text"
MASKED_TEXT_COL = "tail_top1_question_text_mask_struct"
FAMILIES = {
    "question_mask_struct_pls1_bi": {"n_components": 1, "ngram_range": (1, 2)},
    "question_mask_struct_pls2_bi": {"n_components": 2, "ngram_range": (1, 2)},
    "question_mask_struct_pls4_bi": {"n_components": 4, "ngram_range": (1, 2)},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compact supervised hardest-question encodings on structurally masked local question text."
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
        default=Path("results/afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_real"),
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


def build_text_pls_bundle(
    train_rows,
    val_rows,
    test_rows,
    *,
    text_col: str,
    max_features: int,
    min_df: int,
    n_components: int,
    term_limit: int,
    ngram_range,
):
    train_corpus = [str(row.get(text_col, "")) for row in train_rows]
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, ngram_range=ngram_range)
    train_tfidf = vectorizer.fit_transform(train_corpus).toarray()
    capped_components = max(1, min(n_components, train_tfidf.shape[0] - 1, train_tfidf.shape[1]))
    pls = PLSRegression(n_components=capped_components, scale=True)
    y = agreement_gain_target(train_rows).reshape(-1, 1)
    pls.fit(train_tfidf, y)
    train_x = pls.transform(train_tfidf)

    def transform(rows):
        if not rows:
            return np.zeros((0, capped_components), dtype=float)
        corpus = [str(row.get(text_col, "")) for row in rows]
        tfidf = vectorizer.transform(corpus).toarray()
        return pls.transform(tfidf)

    val_x = transform(val_rows)
    test_x = transform(test_rows)
    feature_names = [f"{text_col}__pls_{idx+1}" for idx in range(capped_components)]
    vocab = vectorizer.get_feature_names_out()
    weight_matrix = getattr(pls, "x_weights_", None)
    component_term_rows = []
    if weight_matrix is not None:
        for component_idx in range(weight_matrix.shape[1]):
            weights = weight_matrix[:, component_idx]
            order = np.argsort(weights)
            for rank, vocab_idx in enumerate(order[-term_limit:][::-1], start=1):
                component_term_rows.append(
                    {
                        "component": component_idx + 1,
                        "direction": "positive",
                        "rank": rank,
                        "term": vocab[vocab_idx],
                        "weight": float(weights[vocab_idx]),
                    }
                )
            for rank, vocab_idx in enumerate(order[:term_limit], start=1):
                component_term_rows.append(
                    {
                        "component": component_idx + 1,
                        "direction": "negative",
                        "rank": rank,
                        "term": vocab[vocab_idx],
                        "weight": float(weights[vocab_idx]),
                    }
                )
    return {
        "train": train_x,
        "val": val_x,
        "test": test_x,
        "feature_names": feature_names,
        "n_components": capped_components,
        "component_term_rows": component_term_rows,
    }


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
        raise SystemExit("missing rows for supervised question encoding benchmark")

    train_gain = agreement_gain_target(train_rows)
    val_target = as_array(val_rows, "target")
    test_full_target = as_array(test_full_rows, "target")
    test_full_pre = as_array(test_full_rows, MODEL_PRE_ONLY)
    test_full_hard = as_array(test_full_rows, MODEL_HARD_ABSTENTION)

    families = {}
    component_term_rows = []
    bundle_meta = {}
    for family_name, config in FAMILIES.items():
        tune_bundle = build_text_pls_bundle(
            train_rows,
            val_rows,
            test_agreement_rows,
            text_col=MASKED_TEXT_COL,
            max_features=args.max_features,
            min_df=args.min_df,
            n_components=config["n_components"],
            term_limit=args.term_limit,
            ngram_range=config["ngram_range"],
        )
        refit_bundle = build_text_pls_bundle(
            refit_rows,
            [],
            test_agreement_rows,
            text_col=MASKED_TEXT_COL,
            max_features=args.max_features,
            min_df=args.min_df,
            n_components=config["n_components"],
            term_limit=args.term_limit,
            ngram_range=config["ngram_range"],
        )
        bundle_meta[family_name] = {"refit_components": refit_bundle["n_components"]}
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

        full_metrics = metrics(test_full_target, full_pred)
        family_rows.append(
            {
                "family": family_name,
                "feature_count": len(payload["feature_names"]),
                "selected_alpha": best_alpha,
                "val_r2": best_val_payload["val_r2"],
                "val_use_agreed_share": best_val_payload["val_use_agreed_share"],
                "test_full_r2": full_metrics["r2"],
                "test_full_rmse": full_metrics["rmse"],
                "test_full_mae": full_metrics["mae"],
                "test_full_p_mse_vs_hard": summarize_significance(
                    test_full_target, test_full_hard, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
                )["mse_gain_pvalue"],
                "test_full_p_mse_vs_pre": summarize_significance(
                    test_full_target, test_full_pre, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
                )["mse_gain_pvalue"],
                "test_use_agreed_share": float(np.mean(test_signal > 0.0)),
                "top_coefficients": top_coefficients(model, payload["feature_names"]),
                "bundle_meta": payload["bundle_meta"],
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
            "families": FAMILIES,
        },
        "coverage": {"text": text_coverage, "masked_text": mask_coverage},
        "families": family_rows,
        "best_family": best_family,
        "best_family_summary": next(row for row in family_rows if row["family"] == best_family),
        "best_vs_refs": best_vs_refs,
        "bundle_meta": bundle_meta,
    }

    write_json(
        output_dir / "afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_summary.json",
        summary,
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_family_overview.csv",
        family_rows,
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_component_terms.csv",
        component_term_rows,
    )
    write_csv(
        output_dir / "afterhours_transfer_pair_tail_question_supervised_encoding_benchmark_test_predictions.csv",
        prediction_rows,
    )


if __name__ == "__main__":
    main()
