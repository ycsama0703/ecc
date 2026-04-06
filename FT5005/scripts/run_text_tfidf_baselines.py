#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import site
import sys
from pathlib import Path

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json


TARGET_SCALE = 1_000_000.0


STRUCTURED_FEATURES = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "pre_60m_volume_sum",
    "within_call_rv",
    "within_call_vw_rv",
    "within_call_volume_sum",
    "call_duration_min",
    "scheduled_hour_et",
    "revenue_surprise_pct",
    "ebitda_surprise_pct",
    "eps_gaap_surprise_pct",
    "analyst_eps_norm_num_est",
    "analyst_eps_norm_std",
    "analyst_revenue_num_est",
    "analyst_revenue_std",
    "analyst_net_income_num_est",
    "analyst_net_income_std",
    "a1_component_count",
    "a1_question_count",
    "a1_answer_count",
    "a1_qna_component_share",
    "a1_total_text_words",
    "a1_unique_speaker_count",
    "a2_paragraph_count",
    "a2_visible_word_count",
    "a2_size_ratio_vs_group",
    "a2_text_ratio_vs_group",
    "a4_kept_rows_for_duration",
    "a4_median_match_score",
    "a4_strict_row_share",
    "a4_broad_row_share",
    "a4_hard_fail_rows",
]


EXTRA_DENSE_FEATURES = [
    "full_word_count",
    "qna_word_count",
    "question_word_count",
    "answer_word_count",
    "presenter_word_count",
    "operator_word_count",
    "full_sentence_count",
    "qna_sentence_count",
    "question_mark_count",
    "digit_char_share",
    "qna_word_share",
    "presenter_word_share",
    "operator_word_share",
    "answer_to_question_word_ratio",
    "avg_words_per_question",
    "avg_words_per_answer",
    "speaker_turn_concentration",
    "speaker_name_count_from_text",
    "question_mark_per_1k_words",
    "guidance_term_rate_full",
    "guidance_term_rate_qna",
    "guidance_term_rate_presenter",
    "guidance_term_rate_answer",
    "guidance_term_rate_question",
    "uncertainty_term_rate_full",
    "uncertainty_term_rate_qna",
    "uncertainty_term_rate_presenter",
    "uncertainty_term_rate_answer",
    "uncertainty_term_rate_question",
    "positive_term_rate_full",
    "positive_term_rate_qna",
    "positive_term_rate_presenter",
    "positive_term_rate_answer",
    "positive_term_rate_question",
    "negative_term_rate_full",
    "negative_term_rate_qna",
    "negative_term_rate_presenter",
    "negative_term_rate_answer",
    "negative_term_rate_question",
    "qa_pair_count",
    "qa_pair_overlap_mean",
    "qa_pair_overlap_median",
    "qa_pair_low_overlap_share",
    "qa_pair_question_words_mean",
    "qa_pair_answer_words_mean",
    "qa_pair_answer_digit_rate_mean",
    "qa_pair_answer_hedge_rate_mean",
    "qa_pair_answer_assertive_rate_mean",
    "qa_pair_answer_forward_rate_mean",
    "qa_multi_part_question_share",
    "qa_evasive_proxy_share",
    "qna_vs_presenter_uncertainty_gap",
    "qna_vs_presenter_guidance_gap",
    "answer_vs_question_uncertainty_gap",
    "answer_vs_question_negative_gap",
    "a4_strict_segment_count",
    "a4_strict_duration_sum_sec",
    "a4_strict_duration_mean_sec",
    "a4_strict_duration_median_sec",
    "a4_strict_duration_std_sec",
    "a4_strict_gap_mean_sec",
    "a4_strict_gap_max_sec",
    "a4_strict_span_sec",
    "a4_strict_high_conf_share",
    "a4_strict_overlap_warn_share",
    "a4_broad_segment_count",
    "a4_broad_duration_sum_sec",
    "a4_broad_duration_mean_sec",
    "a4_broad_duration_median_sec",
    "a4_broad_duration_std_sec",
    "a4_broad_gap_mean_sec",
    "a4_broad_gap_max_sec",
    "a4_broad_span_sec",
    "a4_broad_high_conf_share",
    "a4_broad_overlap_warn_share",
    "has_audio_file",
    "a3_file_size_bytes",
    "a3_log_file_size",
    "a3_bytes_per_call_sec",
    "a3_bytes_per_strict_span_sec",
]


TEXT_COLUMNS = {
    "qna": "qna_text",
    "full": "full_text",
    "answer": "answer_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sparse TF-IDF and combined dense+sparse baselines."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--sequence-csv", type=Path, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/text_baselines_real")
    )
    parser.add_argument("--target-col", default="post_call_60m_rv")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--sequence-pca-components", type=int, default=20)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def load_joined_rows(
    panel_csv: Path, features_csv: Path, target_col: str, sequence_csv: Path | None = None
) -> list[dict[str, str]]:
    feature_lookup = {}
    for row in load_csv_rows(features_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            feature_lookup[event_key] = row

    sequence_lookup = {}
    if sequence_csv is not None:
        for row in load_csv_rows(sequence_csv.resolve()):
            event_key = row.get("event_key", "")
            if event_key:
                sequence_lookup[event_key] = row

    rows = []
    for row in load_csv_rows(panel_csv.resolve()):
        event_key = row.get("event_key", "")
        target_value = safe_float(row.get(target_col))
        year_value = safe_float(row.get("year"))
        feature_row = feature_lookup.get(event_key)
        if not event_key or target_value is None or year_value is None or feature_row is None:
            continue
        merged = dict(row)
        merged.update(feature_row)
        if sequence_lookup:
            merged.update(sequence_lookup.get(event_key, {}))
        merged["_target"] = target_value
        merged["_year"] = int(year_value)
        rows.append(merged)
    return rows


def split_rows(rows: list[dict[str, str]], train_end_year: int, val_year: int) -> dict[str, list[dict[str, str]]]:
    split = {"train": [], "val": [], "test": []}
    for row in rows:
        year_value = row["_year"]
        if year_value <= train_end_year:
            split["train"].append(row)
        elif year_value == val_year:
            split["val"].append(row)
        else:
            split["test"].append(row)
    return split


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    baseline = float(np.mean(y_true))
    denom = float(np.sum((y_true - baseline) ** 2))
    r2 = 0.0 if denom == 0 else float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def transform_target(values: np.ndarray) -> np.ndarray:
    return np.log1p(values * TARGET_SCALE)


def inverse_target(values: np.ndarray) -> np.ndarray:
    return np.expm1(values) / TARGET_SCALE


def build_dense_matrix(
    rows: list[dict[str, str]], feature_names: list[str], medians: dict[str, float] | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    if medians is None:
        medians = {}
        for feature in feature_names:
            values = [safe_float(row.get(feature)) for row in rows]
            clean = [value for value in values if value is not None and math.isfinite(value)]
            medians[feature] = float(np.median(clean)) if clean else 0.0

    matrix = []
    for row in rows:
        vector = []
        for feature in feature_names:
            value = safe_float(row.get(feature))
            if value is None or not math.isfinite(value):
                value = medians[feature]
            vector.append(value)
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        for level in ("pass", "warn", "fail"):
            vector.append(1.0 if html_flag == level else 0.0)
        matrix.append(vector)
    return np.asarray(matrix, dtype=float), medians


def standardize_dense(
    train_x: np.ndarray, other_xs: list[np.ndarray]
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1.0
    outputs = [(train_x - means) / stds]
    for matrix in other_xs:
        outputs.append((matrix - means) / stds)
    return outputs, means, stds


def fit_and_select_alpha(
    train_x,
    train_y_t: np.ndarray,
    val_x,
    val_y: np.ndarray,
    alphas: list[float],
) -> tuple[float, Ridge, np.ndarray]:
    best_alpha = None
    best_model = None
    best_val_pred = None
    best_val_rmse = None

    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(train_x, train_y_t)
        val_pred = inverse_target(model.predict(val_x))
        val_rmse = metrics(val_y, val_pred)["rmse"]
        if best_val_rmse is None or val_rmse < best_val_rmse:
            best_alpha = alpha
            best_model = model
            best_val_pred = val_pred
            best_val_rmse = val_rmse

    return best_alpha, best_model, best_val_pred


def top_coefficients(model: Ridge, feature_names: list[str], limit: int = 15) -> dict[str, list[dict[str, float]]]:
    if model is None or not hasattr(model, "coef_"):
        return {"positive": [], "negative": []}
    coef = np.asarray(model.coef_).ravel()
    if coef.size != len(feature_names):
        return {"positive": [], "negative": []}
    order = np.argsort(coef)
    negative = [
        {"feature": feature_names[idx], "coef": round(float(coef[idx]), 6)}
        for idx in order[:limit]
    ]
    positive = [
        {"feature": feature_names[idx], "coef": round(float(coef[idx]), 6)}
        for idx in order[-limit:][::-1]
    ]
    return {"positive": positive, "negative": negative}


def build_dense_bundle(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
):
    train_dense, medians = build_dense_matrix(train_rows, feature_names)
    val_dense, _ = build_dense_matrix(val_rows, feature_names, medians)
    test_dense, _ = build_dense_matrix(test_rows, feature_names, medians)
    (train_z, val_z, test_z), _, _ = standardize_dense(train_dense, [val_dense, test_dense])
    return {
        "train": sparse.csr_matrix(train_z),
        "val": sparse.csr_matrix(val_z),
        "test": sparse.csr_matrix(test_z),
        "feature_names": feature_names + [
            "html_integrity_flag=pass",
            "html_integrity_flag=warn",
            "html_integrity_flag=fail",
        ],
    }


def build_pca_dense_bundle(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    feature_names: list[str],
    n_components: int,
    prefix: str,
):
    train_dense, medians = build_dense_matrix(train_rows, feature_names)
    val_dense, _ = build_dense_matrix(val_rows, feature_names, medians)
    test_dense, _ = build_dense_matrix(test_rows, feature_names, medians)
    (train_z, val_z, test_z), _, _ = standardize_dense(train_dense, [val_dense, test_dense])

    max_components = min(
        n_components,
        train_z.shape[0],
        train_z.shape[1],
    )
    if max_components <= 0:
        return None
    pca = PCA(n_components=max_components)
    train_pca = pca.fit_transform(train_z)
    val_pca = pca.transform(val_z)
    test_pca = pca.transform(test_z)
    return {
        "train": sparse.csr_matrix(train_pca),
        "val": sparse.csr_matrix(val_pca),
        "test": sparse.csr_matrix(test_pca),
        "feature_names": [f"{prefix}_pca_{idx}" for idx in range(max_components)],
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }


def build_text_bundle(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    text_column: str,
    max_features: int,
    min_df: int,
):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        lowercase=True,
    )
    train_text = [normalize_text(row.get(text_column, "")) for row in train_rows]
    val_text = [normalize_text(row.get(text_column, "")) for row in val_rows]
    test_text = [normalize_text(row.get(text_column, "")) for row in test_rows]
    train_x = vectorizer.fit_transform(train_text)
    val_x = vectorizer.transform(val_text)
    test_x = vectorizer.transform(test_text)
    feature_names = vectorizer.get_feature_names_out().tolist()
    return {
        "train": train_x,
        "val": val_x,
        "test": test_x,
        "feature_names": feature_names,
    }


def combine_bundles(*bundles):
    combined = {
        "train": sparse.hstack([bundle["train"] for bundle in bundles], format="csr"),
        "val": sparse.hstack([bundle["val"] for bundle in bundles], format="csr"),
        "test": sparse.hstack([bundle["test"] for bundle in bundles], format="csr"),
        "feature_names": [],
    }
    for bundle in bundles:
        combined["feature_names"].extend(bundle["feature_names"])
    return combined


def infer_prefixed_dense_features(rows: list[dict[str, str]], prefix: str) -> list[str]:
    keys = set()
    for row in rows:
        for key in row:
            if key.startswith(prefix):
                keys.add(key)
    return sorted(keys)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_joined_rows(args.panel_csv, args.features_csv, args.target_col, args.sequence_csv)
    split = split_rows(rows, args.train_end_year, args.val_year)
    train_rows = split["train"]
    val_rows = split["val"]
    test_rows = split["test"]

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_y_t = transform_target(train_y)

    dense_structured = build_dense_bundle(train_rows, val_rows, test_rows, STRUCTURED_FEATURES)
    dense_extra = build_dense_bundle(train_rows, val_rows, test_rows, EXTRA_DENSE_FEATURES)
    strict_sequence_feature_names = infer_prefixed_dense_features(rows, "strict_")
    dense_strict_sequence = (
        build_dense_bundle(train_rows, val_rows, test_rows, strict_sequence_feature_names)
        if strict_sequence_feature_names
        else None
    )
    dense_strict_sequence_pca = (
        build_pca_dense_bundle(
            train_rows,
            val_rows,
            test_rows,
            strict_sequence_feature_names,
            args.sequence_pca_components,
            "strict_sequence",
        )
        if strict_sequence_feature_names
        else None
    )
    qna_text = build_text_bundle(
        train_rows, val_rows, test_rows, TEXT_COLUMNS["qna"], args.max_features, args.min_df
    )
    full_text = build_text_bundle(
        train_rows, val_rows, test_rows, TEXT_COLUMNS["full"], args.max_features, args.min_df
    )

    bundles = {
        "structured_only": dense_structured,
        "qna_tfidf_only": qna_text,
        "full_tfidf_only": full_text,
        "structured_plus_extra": combine_bundles(dense_structured, dense_extra),
        "structured_plus_qna_tfidf": combine_bundles(dense_structured, qna_text),
        "structured_plus_qna_plus_extra": combine_bundles(dense_structured, dense_extra, qna_text),
    }
    if dense_strict_sequence is not None:
        bundles["structured_plus_extra_plus_strict_sequence"] = combine_bundles(
            dense_structured, dense_extra, dense_strict_sequence
        )
        bundles["structured_plus_qna_plus_extra_plus_strict_sequence"] = combine_bundles(
            dense_structured, dense_extra, dense_strict_sequence, qna_text
        )
    if dense_strict_sequence_pca is not None:
        bundles["structured_plus_extra_plus_strict_sequence_pca"] = combine_bundles(
            dense_structured, dense_extra, dense_strict_sequence_pca
        )
        bundles["structured_plus_qna_plus_extra_plus_strict_sequence_pca"] = combine_bundles(
            dense_structured, dense_extra, dense_strict_sequence_pca, qna_text
        )

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]
    summary = {
        "target_col": args.target_col,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "config": {
            "alphas": alphas,
            "max_features": args.max_features,
            "min_df": args.min_df,
            "sequence_pca_components": args.sequence_pca_components,
        },
        "models": {},
    }
    if dense_strict_sequence_pca is not None:
        summary["sequence_pca"] = {
            "components": len(dense_strict_sequence_pca["feature_names"]),
            "explained_variance_ratio_sum": round(
                dense_strict_sequence_pca["explained_variance_ratio_sum"], 6
            ),
        }
    prediction_rows = []

    for model_name, bundle in bundles.items():
        best_alpha, model, val_pred = fit_and_select_alpha(
            bundle["train"], train_y_t, bundle["val"], val_y, alphas
        )
        test_pred = inverse_target(model.predict(bundle["test"]))
        summary["models"][model_name] = {
            "best_alpha": best_alpha,
            "feature_count": int(bundle["train"].shape[1]),
            "val": metrics(val_y, val_pred),
            "test": metrics(test_y, test_pred),
            "top_coefficients": top_coefficients(model, bundle["feature_names"]),
        }
        for split_name, split_rows_, y_true_arr, preds in [
            (f"{model_name}_val", val_rows, val_y, val_pred),
            (f"{model_name}_test", test_rows, test_y, test_pred),
        ]:
            for row, truth, pred in zip(split_rows_, y_true_arr, preds):
                prediction_rows.append(
                    {
                        "model_split": split_name,
                        "event_key": row["event_key"],
                        "year": row["year"],
                        "ticker": row["ticker"],
                        "y_true": truth,
                        "y_pred": pred,
                    }
                )

    write_json(output_dir / "text_tfidf_baseline_summary.json", summary)
    write_csv(output_dir / "text_tfidf_baseline_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
