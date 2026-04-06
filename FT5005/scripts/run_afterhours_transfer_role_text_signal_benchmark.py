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
user_site_removed = False
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)
    user_site_removed = True

try:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
except ModuleNotFoundError:
    if user_site_removed and isinstance(user_site, str) and user_site not in sys.path:
        sys.path.insert(0, user_site)
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

from dj30_qc_utils import load_csv_rows, write_csv, write_json
from run_afterhours_transfer_agreement_signal_benchmark import (
    MODEL_AGREED,
    MODEL_HARD_ABSTENTION,
    MODEL_PRE_ONLY,
    agreement_gain_target,
    build_temporal_rows,
    fit_ridge,
    summarize_significance,
)
from run_dense_multimodal_ablation_baselines import normalize_text
from run_structured_baselines import metrics

GEOMETRY_FEATURES = [
    "agreed_choose_qa",
    "agreed_minus_pre_pred",
    "qa_minus_pre_pred",
    "sem_minus_pre_pred",
    "selected_minus_pre_pred",
    "pair_minus_pre_pred",
    "logistic_minus_pre_pred",
    "abs_pair_minus_logistic_pred",
]

SPLITS = [
    "val2020_test_post2020",
    "val2021_test_post2021",
    "val2022_test_post2022",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark compact role-specific Q/A text signals for agreement-side transfer refinement."
    )
    parser.add_argument(
        "--temporal-root",
        type=Path,
        default=Path("results/afterhours_transfer_router_temporal_confirmation_role_aware_audio_lsa4_real"),
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("results/features_real/event_text_audio_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_role_text_signal_benchmark_lsa4_real"),
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


def as_array(rows: list[dict[str, float | int | str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def feature_matrix(rows: list[dict[str, float | int | str]], feature_names: list[str]) -> np.ndarray:
    return np.asarray([[float(row.get(name, 0.0)) for name in feature_names] for row in rows], dtype=float)


def zscore_fit_transform(train_x: np.ndarray, other_xs: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray, np.ndarray]:
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1.0
    train_z = (train_x - means) / stds
    other_zs = [(matrix - means) / stds for matrix in other_xs]
    return train_z, other_zs, means, stds


def attach_role_text(rows: list[dict[str, float | int | str]], features_csv: Path) -> dict[str, int]:
    feature_lookup = {row["event_key"]: row for row in load_csv_rows(features_csv.resolve()) if row.get("event_key")}
    coverage = {
        "temporal_rows": len(rows),
        "with_feature_row": 0,
        "with_question_text": 0,
        "with_answer_text": 0,
        "with_both_role_texts": 0,
    }
    for row in rows:
        feature_row = feature_lookup.get(str(row["event_key"]))
        question_text = ""
        answer_text = ""
        if feature_row is not None:
            coverage["with_feature_row"] += 1
            question_text = normalize_text(feature_row.get("question_text", ""))
            answer_text = normalize_text(feature_row.get("answer_text", ""))
            if question_text:
                coverage["with_question_text"] += 1
            if answer_text:
                coverage["with_answer_text"] += 1
            if question_text and answer_text:
                coverage["with_both_role_texts"] += 1
        row["question_text"] = question_text
        row["answer_text"] = answer_text
    return coverage


def build_shared_role_lsa_bundle(
    train_rows: list[dict[str, float | int | str]],
    val_rows: list[dict[str, float | int | str]],
    test_rows: list[dict[str, float | int | str]],
    max_features: int,
    min_df: int,
    lsa_components: int,
    term_limit: int,
) -> dict[str, object]:
    train_question = [str(row.get("question_text", "")) for row in train_rows]
    train_answer = [str(row.get("answer_text", "")) for row in train_rows]
    train_corpus = train_question + train_answer

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, ngram_range=(1, 2))
    train_tfidf = vectorizer.fit_transform(train_corpus)
    n_components = max(2, min(lsa_components, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_role_x = svd.fit_transform(train_tfidf)

    train_question_x = train_role_x[: len(train_rows)]
    train_answer_x = train_role_x[len(train_rows) :]

    def role_transform(rows: list[dict[str, float | int | str]]) -> tuple[np.ndarray, np.ndarray]:
        if not rows:
            empty = np.zeros((0, n_components), dtype=float)
            return empty, empty
        question = [str(row.get("question_text", "")) for row in rows]
        answer = [str(row.get("answer_text", "")) for row in rows]
        role_x = svd.transform(vectorizer.transform(question + answer))
        return role_x[: len(rows)], role_x[len(rows) :]

    val_question_x, val_answer_x = role_transform(val_rows)
    test_question_x, test_answer_x = role_transform(test_rows)

    train_question_z, [val_question_z, test_question_z], _, _ = zscore_fit_transform(
        train_question_x, [val_question_x, test_question_x]
    )
    train_answer_z, [val_answer_z, test_answer_z], _, _ = zscore_fit_transform(
        train_answer_x, [val_answer_x, test_answer_x]
    )

    train_gap = train_answer_z - train_question_z
    val_gap = val_answer_z - val_question_z
    test_gap = test_answer_z - test_question_z

    feature_names_question = [f"question_role_lsa_{idx+1}" for idx in range(n_components)]
    feature_names_answer = [f"answer_role_lsa_{idx+1}" for idx in range(n_components)]
    feature_names_gap = [f"qa_role_gap_lsa_{idx+1}" for idx in range(n_components)]

    vocab = vectorizer.get_feature_names_out()
    component_term_rows = []
    for component_idx, weights in enumerate(svd.components_):
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
        "question_train": train_question_z,
        "question_val": val_question_z,
        "question_test": test_question_z,
        "answer_train": train_answer_z,
        "answer_val": val_answer_z,
        "answer_test": test_answer_z,
        "gap_train": train_gap,
        "gap_val": val_gap,
        "gap_test": test_gap,
        "question_feature_names": feature_names_question,
        "answer_feature_names": feature_names_answer,
        "gap_feature_names": feature_names_gap,
        "n_components": n_components,
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        "component_term_rows": component_term_rows,
    }


def top_coefficients(model, feature_names: list[str], limit: int = 8) -> list[dict[str, float | str]]:
    ridge = model.named_steps["ridge"]
    return [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in sorted(zip(feature_names, ridge.coef_.tolist()), key=lambda item: abs(item[1]), reverse=True)[
            :limit
        ]
    ]


def main() -> None:
    args = parse_args()
    temporal_root = args.temporal_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_temporal_rows(temporal_root)
    coverage = attach_role_text(rows, args.features_csv.resolve())

    refit_train_splits = parse_list(args.refit_train_splits)
    alpha_grid = [float(item) for item in parse_list(args.alphas)]

    train_rows = [row for row in rows if row["split"] == args.train_split and int(row["agreement"]) == 1]
    val_rows = [row for row in rows if row["split"] == args.val_split and int(row["agreement"]) == 1]
    refit_rows = [row for row in rows if row["split"] in refit_train_splits and int(row["agreement"]) == 1]
    test_agreement_rows = [row for row in rows if row["split"] == args.test_split and int(row["agreement"]) == 1]
    test_full_rows = [row for row in rows if row["split"] == args.test_split]
    if not train_rows or not val_rows or not refit_rows or not test_agreement_rows or not test_full_rows:
        raise SystemExit("missing train/val/refit/test rows for role-text benchmark")

    train_gain = agreement_gain_target(train_rows)
    val_target = as_array(val_rows, "target")
    test_agreement_target = as_array(test_agreement_rows, "target")
    test_full_target = as_array(test_full_rows, "target")
    test_full_pre = as_array(test_full_rows, MODEL_PRE_ONLY)
    test_full_agreed = as_array(test_full_rows, MODEL_AGREED)
    test_full_hard = as_array(test_full_rows, MODEL_HARD_ABSTENTION)

    geometry_train = feature_matrix(train_rows, GEOMETRY_FEATURES)
    geometry_val = feature_matrix(val_rows, GEOMETRY_FEATURES)
    geometry_refit = feature_matrix(refit_rows, GEOMETRY_FEATURES)
    geometry_test = feature_matrix(test_agreement_rows, GEOMETRY_FEATURES)

    tune_bundle = build_shared_role_lsa_bundle(
        train_rows,
        val_rows,
        test_agreement_rows,
        args.max_features,
        args.min_df,
        args.lsa_components,
        args.term_limit,
    )
    refit_bundle = build_shared_role_lsa_bundle(
        refit_rows,
        [],
        test_agreement_rows,
        args.max_features,
        args.min_df,
        args.lsa_components,
        args.term_limit,
    )

    families = {
        "geometry_only": {
            "train_x": geometry_train,
            "val_x": geometry_val,
            "refit_x": geometry_refit,
            "test_x": geometry_test,
            "feature_names": GEOMETRY_FEATURES,
        },
        "question_role_lsa": {
            "train_x": tune_bundle["question_train"],
            "val_x": tune_bundle["question_val"],
            "refit_x": refit_bundle["question_train"],
            "test_x": refit_bundle["question_test"],
            "feature_names": tune_bundle["question_feature_names"],
        },
        "answer_role_lsa": {
            "train_x": tune_bundle["answer_train"],
            "val_x": tune_bundle["answer_val"],
            "refit_x": refit_bundle["answer_train"],
            "test_x": refit_bundle["answer_test"],
            "feature_names": tune_bundle["answer_feature_names"],
        },
        "qa_role_gap_lsa": {
            "train_x": tune_bundle["gap_train"],
            "val_x": tune_bundle["gap_val"],
            "refit_x": refit_bundle["gap_train"],
            "test_x": refit_bundle["gap_test"],
            "feature_names": tune_bundle["gap_feature_names"],
        },
        "geometry_plus_answer_role": {
            "train_x": np.hstack([geometry_train, tune_bundle["answer_train"]]),
            "val_x": np.hstack([geometry_val, tune_bundle["answer_val"]]),
            "refit_x": np.hstack([geometry_refit, refit_bundle["answer_train"]]),
            "test_x": np.hstack([geometry_test, refit_bundle["answer_test"]]),
            "feature_names": GEOMETRY_FEATURES + tune_bundle["answer_feature_names"],
        },
        "geometry_plus_role_gap": {
            "train_x": np.hstack([geometry_train, tune_bundle["gap_train"]]),
            "val_x": np.hstack([geometry_val, tune_bundle["gap_val"]]),
            "refit_x": np.hstack([geometry_refit, refit_bundle["gap_train"]]),
            "test_x": np.hstack([geometry_test, refit_bundle["gap_test"]]),
            "feature_names": GEOMETRY_FEATURES + tune_bundle["gap_feature_names"],
        },
        "geometry_plus_dual_role": {
            "train_x": np.hstack([geometry_train, tune_bundle["question_train"], tune_bundle["answer_train"]]),
            "val_x": np.hstack([geometry_val, tune_bundle["question_val"], tune_bundle["answer_val"]]),
            "refit_x": np.hstack([geometry_refit, refit_bundle["question_train"], refit_bundle["answer_train"]]),
            "test_x": np.hstack([geometry_test, refit_bundle["question_test"], refit_bundle["answer_test"]]),
            "feature_names": GEOMETRY_FEATURES
            + tune_bundle["question_feature_names"]
            + tune_bundle["answer_feature_names"],
        },
    }

    family_rows = []
    tuning_rows = []
    prediction_rows = [dict(row) for row in test_full_rows]
    family_predictions = {
        MODEL_PRE_ONLY: test_full_pre,
        MODEL_AGREED: test_full_agreed,
        MODEL_HARD_ABSTENTION: test_full_hard,
    }
    family_summaries = {}
    best_family_name = None
    best_family_r2 = None
    best_factor_only_name = None
    best_factor_only_r2 = None
    best_geometry_plus_name = None
    best_geometry_plus_r2 = None

    for family_name, payload in families.items():
        train_x = payload["train_x"]
        val_x = payload["val_x"]
        refit_x = payload["refit_x"]
        test_agreement_x = payload["test_x"]
        feature_names = payload["feature_names"]

        best_alpha = None
        best_val_r2 = None
        best_val_payload = None
        for alpha in alpha_grid:
            model = fit_ridge(train_x, train_gain, alpha)
            val_signal = model.predict(val_x)
            val_pred = np.where(val_signal > 0.0, as_array(val_rows, MODEL_AGREED), as_array(val_rows, MODEL_PRE_ONLY))
            score = metrics(val_target, val_pred)
            row = {
                "family": family_name,
                "alpha": float(alpha),
                "val_r2": score["r2"],
                "val_rmse": score["rmse"],
                "val_mae": score["mae"],
                "val_use_agreed_share": float(np.mean(val_signal > 0.0)),
                "feature_count": len(feature_names),
            }
            tuning_rows.append(row)
            if best_val_r2 is None or score["r2"] > best_val_r2:
                best_val_r2 = score["r2"]
                best_alpha = float(alpha)
                best_val_payload = row

        assert best_alpha is not None and best_val_payload is not None
        model = fit_ridge(refit_x, agreement_gain_target(refit_rows), best_alpha)
        test_signal = model.predict(test_agreement_x)
        test_agreement_pred = np.where(
            test_signal > 0.0,
            as_array(test_agreement_rows, MODEL_AGREED),
            as_array(test_agreement_rows, MODEL_PRE_ONLY),
        )
        pred_lookup = {row["event_key"]: float(pred) for row, pred in zip(test_agreement_rows, test_agreement_pred)}
        signal_lookup = {row["event_key"]: float(signal) for row, signal in zip(test_agreement_rows, test_signal)}
        use_lookup = {row["event_key"]: int(signal > 0.0) for row, signal in zip(test_agreement_rows, test_signal)}
        full_pred = np.asarray([pred_lookup.get(row["event_key"], float(row[MODEL_PRE_ONLY])) for row in test_full_rows], dtype=float)
        family_predictions[family_name] = full_pred

        for row in prediction_rows:
            event_key = row["event_key"]
            row[family_name] = pred_lookup.get(event_key, float(row[MODEL_PRE_ONLY]))
            row[f"{family_name}__predicted_gain_signal"] = signal_lookup.get(event_key, 0.0)
            row[f"{family_name}__use_agreed"] = use_lookup.get(event_key, 0)

        test_agreement_metrics = metrics(test_agreement_target, test_agreement_pred)
        test_full_metrics = metrics(test_full_target, full_pred)
        sig_vs_hard = summarize_significance(
            test_full_target, test_full_hard, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
        )
        sig_vs_pre = summarize_significance(
            test_full_target, test_full_pre, full_pred, args.bootstrap_iters, args.perm_iters, args.seed
        )

        family_row = {
            "family": family_name,
            "feature_count": len(feature_names),
            "selected_alpha": best_alpha,
            "val_r2": best_val_payload["val_r2"],
            "val_use_agreed_share": best_val_payload["val_use_agreed_share"],
            "test_agreement_r2": test_agreement_metrics["r2"],
            "test_agreement_rmse": test_agreement_metrics["rmse"],
            "test_full_r2": test_full_metrics["r2"],
            "test_full_rmse": test_full_metrics["rmse"],
            "test_full_mae": test_full_metrics["mae"],
            "test_full_p_mse_vs_hard": sig_vs_hard["mse_gain_pvalue"],
            "test_full_p_mse_vs_pre": sig_vs_pre["mse_gain_pvalue"],
            "test_use_agreed_share": float(np.mean(test_signal > 0.0)),
        }
        family_rows.append(family_row)
        family_summaries[family_name] = {
            "feature_names": feature_names,
            "selected_alpha": best_alpha,
            "best_validation": best_val_payload,
            "test_agreement_metrics": test_agreement_metrics,
            "test_full_metrics": test_full_metrics,
            "significance_vs_hard": sig_vs_hard,
            "significance_vs_pre": sig_vs_pre,
            "coef_rows": top_coefficients(model, feature_names),
        }

        if best_family_r2 is None or test_full_metrics["r2"] > best_family_r2:
            best_family_r2 = test_full_metrics["r2"]
            best_family_name = family_name
        if family_name.startswith("geometry_plus") or family_name == "geometry_only":
            if best_geometry_plus_r2 is None or test_full_metrics["r2"] > best_geometry_plus_r2:
                best_geometry_plus_r2 = test_full_metrics["r2"]
                best_geometry_plus_name = family_name
        else:
            if best_factor_only_r2 is None or test_full_metrics["r2"] > best_factor_only_r2:
                best_factor_only_r2 = test_full_metrics["r2"]
                best_factor_only_name = family_name

    best_vs_geometry = None
    if best_family_name is not None and best_family_name != "geometry_only":
        best_vs_geometry = summarize_significance(
            test_full_target,
            family_predictions["geometry_only"],
            family_predictions[best_family_name],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        )

    summary = {
        "config": {
            "temporal_root": str(temporal_root),
            "features_csv": str(args.features_csv.resolve()),
            "train_split": args.train_split,
            "val_split": args.val_split,
            "refit_train_splits": refit_train_splits,
            "test_split": args.test_split,
            "alpha_grid": alpha_grid,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "lsa_components": args.lsa_components,
        },
        "coverage": coverage,
        "split_sizes": {
            "train_agreement_size": len(train_rows),
            "val_agreement_size": len(val_rows),
            "refit_agreement_size": len(refit_rows),
            "test_agreement_size": len(test_agreement_rows),
            "test_full_size": len(test_full_rows),
        },
        "bundle_meta": {
            "tune_shared_role_lsa": {
                "n_components": tune_bundle["n_components"],
                "explained_variance_ratio_sum": tune_bundle["explained_variance_ratio_sum"],
            },
            "refit_shared_role_lsa": {
                "n_components": refit_bundle["n_components"],
                "explained_variance_ratio_sum": refit_bundle["explained_variance_ratio_sum"],
            },
        },
        "reference": {
            "validation_pre_r2": metrics(val_target, as_array(val_rows, MODEL_PRE_ONLY))["r2"],
            "validation_agreed_r2": metrics(val_target, as_array(val_rows, MODEL_AGREED))["r2"],
            "test_full_pre": metrics(test_full_target, test_full_pre),
            "test_full_agreed": metrics(test_full_target, test_full_agreed),
            "test_full_hard_abstention": metrics(test_full_target, test_full_hard),
        },
        "families": family_rows,
        "best_family": best_family_name,
        "best_family_summary": family_summaries.get(best_family_name),
        "best_factor_only_family": best_factor_only_name,
        "best_factor_only_family_summary": family_summaries.get(best_factor_only_name),
        "best_geometry_plus_family": best_geometry_plus_name,
        "best_geometry_plus_family_summary": family_summaries.get(best_geometry_plus_name),
        "best_family_vs_geometry_only": best_vs_geometry,
        "family_summaries": family_summaries,
    }

    write_csv(output_dir / "afterhours_transfer_role_text_signal_benchmark_overview.csv", family_rows)
    write_csv(output_dir / "afterhours_transfer_role_text_signal_benchmark_tuning.csv", tuning_rows)
    write_csv(output_dir / "afterhours_transfer_role_text_signal_benchmark_test_predictions.csv", prediction_rows)
    write_csv(
        output_dir / "afterhours_transfer_role_text_signal_benchmark_component_terms.csv",
        refit_bundle["component_term_rows"],
    )
    write_json(output_dir / "afterhours_transfer_role_text_signal_benchmark_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
