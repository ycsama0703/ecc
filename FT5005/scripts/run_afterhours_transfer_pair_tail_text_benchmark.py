#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
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

TEXT_FAMILIES = {
    "tail_question_top1_lsa": "tail_top1_question_text",
    "tail_answer_top1_lsa": "tail_top1_answer_text",
    "tail_qa_top1_lsa": "tail_top1_qa_text",
    "tail_qa_top2_lsa": "tail_top2_qa_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark hardest-pair local semantic text views for agreement-side transfer refinement."
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
            "results/afterhours_transfer_pair_tail_factor_benchmark_lsa4_real/"
            "afterhours_transfer_pair_tail_factor_benchmark_test_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_pair_tail_text_benchmark_lsa4_real"),
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


def zscore_fit_transform(train_x: np.ndarray, other_xs: list[np.ndarray]):
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1.0
    train_z = (train_x - means) / stds
    other_zs = [(matrix - means) / stds for matrix in other_xs]
    return train_z, other_zs


def attach_text_views(rows, text_views_csv: Path):
    lookup = {row["event_key"]: row for row in load_csv_rows(text_views_csv.resolve()) if row.get("event_key")}
    coverage = {"temporal_rows": len(rows), "with_text_view_row": 0}
    for family, col in TEXT_FAMILIES.items():
        coverage[f"with_{family}"] = 0
    for row in rows:
        text_row = lookup.get(str(row["event_key"]))
        if text_row is not None:
            coverage["with_text_view_row"] += 1
        for family, col in TEXT_FAMILIES.items():
            text = normalize_text(text_row.get(col, "") if text_row else "")
            row[col] = text
            if text:
                coverage[f"with_{family}"] += 1
    return coverage


def build_text_lsa_bundle(train_rows, val_rows, test_rows, text_col: str, max_features: int, min_df: int, lsa_components: int, term_limit: int):
    train_corpus = [str(row.get(text_col, "")) for row in train_rows]
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, ngram_range=(1, 2))
    train_tfidf = vectorizer.fit_transform(train_corpus)
    n_components = max(2, min(lsa_components, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_x = svd.fit_transform(train_tfidf)

    def transform(rows):
        if not rows:
            return np.zeros((0, n_components), dtype=float)
        corpus = [str(row.get(text_col, "")) for row in rows]
        return svd.transform(vectorizer.transform(corpus))

    val_x = transform(val_rows)
    test_x = transform(test_rows)
    train_z, [val_z, test_z] = zscore_fit_transform(train_x, [val_x, test_x])
    feature_names = [f"{text_col}_lsa_{idx+1}" for idx in range(n_components)]
    vocab = vectorizer.get_feature_names_out()
    component_term_rows = []
    for component_idx, weights in enumerate(svd.components_):
        order = np.argsort(weights)
        for rank, vocab_idx in enumerate(order[-term_limit:][::-1], start=1):
            component_term_rows.append({
                "component": component_idx + 1,
                "direction": "positive",
                "rank": rank,
                "term": vocab[vocab_idx],
                "weight": float(weights[vocab_idx]),
            })
        for rank, vocab_idx in enumerate(order[:term_limit], start=1):
            component_term_rows.append({
                "component": component_idx + 1,
                "direction": "negative",
                "rank": rank,
                "term": vocab[vocab_idx],
                "weight": float(weights[vocab_idx]),
            })
    return {
        "train": train_z,
        "val": val_z,
        "test": test_z,
        "feature_names": feature_names,
        "n_components": n_components,
        "explained_variance_ratio_sum": float(np.sum(svd.explained_variance_ratio_)),
        "component_term_rows": component_term_rows,
    }


def top_coefficients(model, feature_names: list[str], limit: int = 8):
    ridge = model.named_steps["ridge"]
    return [
        {"feature": feature, "coefficient": float(coef)}
        for feature, coef in sorted(zip(feature_names, ridge.coef_.tolist()), key=lambda item: abs(item[1]), reverse=True)[:limit]
    ]


def load_reference_predictions(path: Path, keys: list[str]) -> dict[str, np.ndarray]:
    lookup = {row["event_key"]: row for row in csv.DictReader(path.open())}
    refs = {}
    for name, col in [
        (MODEL_PRE_ONLY, MODEL_PRE_ONLY),
        (MODEL_HARD_ABSTENTION, MODEL_HARD_ABSTENTION),
        ("geometry_only", "geometry_only"),
        ("pair_tail_factor_route", "prediction"),
    ]:
        refs[name] = np.asarray([float(lookup[key][col]) for key in keys], dtype=float)
    return refs


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_temporal_rows(args.temporal_root.resolve())
    coverage = attach_text_views(rows, args.text_views_csv.resolve())
    refit_train_splits = parse_list(args.refit_train_splits)
    alpha_grid = [float(item) for item in parse_list(args.alphas)]

    train_rows = [row for row in rows if row["split"] == args.train_split and int(row["agreement"]) == 1]
    val_rows = [row for row in rows if row["split"] == args.val_split and int(row["agreement"]) == 1]
    refit_rows = [row for row in rows if row["split"] in refit_train_splits and int(row["agreement"]) == 1]
    test_agreement_rows = [row for row in rows if row["split"] == args.test_split and int(row["agreement"]) == 1]
    test_full_rows = [row for row in rows if row["split"] == args.test_split]
    if not train_rows or not val_rows or not refit_rows or not test_agreement_rows or not test_full_rows:
        raise SystemExit("missing rows for pair-tail text benchmark")

    train_gain = agreement_gain_target(train_rows)
    val_target = as_array(val_rows, "target")
    test_full_target = as_array(test_full_rows, "target")
    test_full_pre = as_array(test_full_rows, MODEL_PRE_ONLY)
    test_full_agreed = as_array(test_full_rows, MODEL_AGREED)
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
    for family_name, text_col in TEXT_FAMILIES.items():
        tune_bundle = build_text_lsa_bundle(train_rows, val_rows, test_agreement_rows, text_col, args.max_features, args.min_df, args.lsa_components, args.term_limit)
        refit_bundle = build_text_lsa_bundle(refit_rows, [], test_agreement_rows, text_col, args.max_features, args.min_df, args.lsa_components, args.term_limit)
        bundle_meta[family_name] = {
            "tune_n_components": tune_bundle["n_components"],
            "tune_explained_variance_ratio_sum": tune_bundle["explained_variance_ratio_sum"],
            "refit_n_components": refit_bundle["n_components"],
            "refit_explained_variance_ratio_sum": refit_bundle["explained_variance_ratio_sum"],
        }
        for row in tune_bundle["component_term_rows"]:
            component_term_rows.append({"family": family_name, **row})
        families[family_name] = {
            "train_x": tune_bundle["train"],
            "val_x": tune_bundle["val"],
            "refit_x": refit_bundle["train"],
            "test_x": refit_bundle["test"],
            "feature_names": tune_bundle["feature_names"],
            "bundle_meta": bundle_meta[family_name],
        }
        geom_name = f"geometry_plus_{family_name}"
        families[geom_name] = {
            "train_x": np.hstack([geometry_train, tune_bundle["train"]]),
            "val_x": np.hstack([geometry_val, tune_bundle["val"]]),
            "refit_x": np.hstack([geometry_refit, refit_bundle["train"]]),
            "test_x": np.hstack([geometry_test, refit_bundle["test"]]),
            "feature_names": GEOMETRY_FEATURES + tune_bundle["feature_names"],
            "bundle_meta": bundle_meta[family_name],
        }

    family_rows = []
    tuning_rows = []
    prediction_rows = [dict(row) for row in test_full_rows]
    family_predictions = {MODEL_PRE_ONLY: test_full_pre, MODEL_AGREED: test_full_agreed, MODEL_HARD_ABSTENTION: test_full_hard}
    family_summaries = {}
    best_family_name = None
    best_family_r2 = None

    for family_name, payload in families.items():
        train_x = payload["train_x"]
        val_x = payload["val_x"]
        refit_x = payload["refit_x"]
        test_x = payload["test_x"]
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

        model = fit_ridge(refit_x, agreement_gain_target(refit_rows), best_alpha)
        test_signal = model.predict(test_x)
        test_agreement_pred = np.where(test_signal > 0.0, as_array(test_agreement_rows, MODEL_AGREED), as_array(test_agreement_rows, MODEL_PRE_ONLY))
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

        test_full_metrics = metrics(test_full_target, full_pred)
        sig_vs_hard = summarize_significance(test_full_target, test_full_hard, full_pred, args.bootstrap_iters, args.perm_iters, args.seed)
        sig_vs_pre = summarize_significance(test_full_target, test_full_pre, full_pred, args.bootstrap_iters, args.perm_iters, args.seed)
        family_row = {
            "family": family_name,
            "feature_count": len(feature_names),
            "selected_alpha": best_alpha,
            "val_r2": best_val_payload["val_r2"],
            "val_use_agreed_share": best_val_payload["val_use_agreed_share"],
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
            "test_full_metrics": test_full_metrics,
            "significance_vs_hard": sig_vs_hard,
            "significance_vs_pre": sig_vs_pre,
            "coef_rows": top_coefficients(model, feature_names),
            "bundle_meta": payload["bundle_meta"],
        }
        if best_family_r2 is None or test_full_metrics["r2"] > best_family_r2:
            best_family_r2 = test_full_metrics["r2"]
            best_family_name = family_name

    best_vs_geometry = None
    if best_family_name != "geometry_only":
        best_vs_geometry = summarize_significance(
            test_full_target,
            family_predictions["geometry_only"],
            family_predictions[best_family_name],
            args.bootstrap_iters,
            args.perm_iters,
            args.seed,
        )

    keys = [str(row["event_key"]) for row in test_full_rows]
    refs = load_reference_predictions(args.reference_predictions_csv.resolve(), keys)
    pair_tail_vs_ref = {
        name: summarize_significance(test_full_target, ref_pred, family_predictions[best_family_name], args.bootstrap_iters, args.perm_iters, args.seed)
        for name, ref_pred in refs.items()
    }

    summary = {
        "config": {
            "temporal_root": str(args.temporal_root.resolve()),
            "text_views_csv": str(args.text_views_csv.resolve()),
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
        "bundle_meta": bundle_meta,
        "reference": {
            "test_full_pre": metrics(test_full_target, test_full_pre),
            "test_full_agreed": metrics(test_full_target, test_full_agreed),
            "test_full_hard_abstention": metrics(test_full_target, test_full_hard),
        },
        "families": family_rows,
        "best_family": best_family_name,
        "best_family_summary": family_summaries.get(best_family_name),
        "best_vs_geometry": best_vs_geometry,
        "best_vs_previous_refs": pair_tail_vs_ref,
    }

    ref_lookup = {key: {name: float(refs[name][idx]) for name in refs} for idx, key in enumerate(keys)}
    for row in prediction_rows:
        event_key = row["event_key"]
        for name, value in ref_lookup[event_key].items():
            row[name] = value

    write_json(output_dir / "afterhours_transfer_pair_tail_text_benchmark_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_pair_tail_text_benchmark_family_overview.csv", family_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_text_benchmark_tuning.csv", tuning_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_text_benchmark_test_predictions.csv", prediction_rows)
    write_csv(output_dir / "afterhours_transfer_pair_tail_text_component_terms.csv", component_term_rows)


if __name__ == "__main__":
    main()
