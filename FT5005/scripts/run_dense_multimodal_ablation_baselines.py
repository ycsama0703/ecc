#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import site
import sys
from pathlib import Path

user_site = site.getusersitepackages()
if isinstance(user_site, str) and user_site in sys.path:
    sys.path.remove(user_site)

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json
from run_structured_baselines import inverse_target, metrics, transform_target
from run_text_tfidf_baselines import EXTRA_DENSE_FEATURES, STRUCTURED_FEATURES


TEXT_FEATURE_SPECS = {
    "qna_lsa": "qna_text",
    "full_lsa": "full_text",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dense multimodal ablation baselines with semantic text and real audio features."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/dense_multimodal_baselines_real"),
    )
    parser.add_argument("--target-col", default="post_call_60m_rv")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--lsa-components", type=int, default=64)
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def load_joined_rows(
    panel_csv: Path, features_csv: Path, audio_csv: Path, target_col: str, qa_csv: Path | None = None
) -> list[dict[str, str]]:
    feature_lookup = {}
    for row in load_csv_rows(features_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            feature_lookup[event_key] = row

    audio_lookup = {}
    for row in load_csv_rows(audio_csv.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            audio_lookup[event_key] = row

    qa_lookup = {}
    if qa_csv is not None:
        for row in load_csv_rows(qa_csv.resolve()):
            event_key = row.get("event_key", "")
            if event_key:
                qa_lookup[event_key] = row

    rows = []
    for row in load_csv_rows(panel_csv.resolve()):
        event_key = row.get("event_key", "")
        target_value = safe_float(row.get(target_col))
        year_value = safe_float(row.get("year"))
        feature_row = feature_lookup.get(event_key)
        audio_row = audio_lookup.get(event_key)
        qa_row = qa_lookup.get(event_key) if qa_lookup else None
        if (
            not event_key
            or target_value is None
            or year_value is None
            or feature_row is None
            or audio_row is None
            or (qa_lookup and qa_row is None)
        ):
            continue
        merged = dict(row)
        merged.update(feature_row)
        merged.update(audio_row)
        if qa_row is not None:
            merged.update(qa_row)
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


def infer_audio_feature_names(rows: list[dict[str, str]]) -> list[str]:
    excluded = {"event_key", "audio_path"}
    names = set()
    for row in rows:
        for key, value in row.items():
            if key in excluded:
                continue
            if key.startswith("audio_") or key.startswith("chunk_") or key.startswith("mfcc_") or key == "has_real_audio":
                if safe_float(value) is not None:
                    names.add(key)
    return sorted(names)


def infer_aligned_audio_feature_names(
    rows: list[dict[str, str]],
    prefix: str = "aligned_audio__",
    agg_suffixes: tuple[str, ...] = ("winsor_mean",),
    include_summary: bool = True,
) -> list[str]:
    summary_fields = {
        f"{prefix}aligned_audio_sentence_count",
        f"{prefix}aligned_audio_duration_sum_sec",
        f"{prefix}aligned_audio_duration_mean_sec",
        f"{prefix}aligned_audio_duration_median_sec",
    }
    suffix_tokens = tuple(f"_{suffix}" for suffix in agg_suffixes)
    names = set()
    for row in rows:
        for key, value in row.items():
            if not key.startswith(prefix):
                continue
            if include_summary and key in summary_fields and safe_float(value) is not None:
                names.add(key)
                continue
            if suffix_tokens and key.endswith(suffix_tokens) and safe_float(value) is not None:
                names.add(key)
    return sorted(names)


def infer_prefixed_feature_names(rows: list[dict[str, str]], prefix: str) -> list[str]:
    names = set()
    for row in rows:
        for key, value in row.items():
            if key.startswith(prefix) and safe_float(value) is not None:
                names.add(key)
    return sorted(names)


def build_dense_matrix(rows: list[dict[str, str]], feature_names: list[str], medians: dict[str, float] | None = None):
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
        if feature_names and "html_integrity_flag" not in feature_names:
            html_flag = (row.get("html_integrity_flag") or "").strip().lower()
            for level in ("pass", "warn", "fail"):
                vector.append(1.0 if html_flag == level else 0.0)
        matrix.append(vector)
    extra_names = []
    if feature_names and "html_integrity_flag" not in feature_names:
        extra_names = ["html_integrity_flag=pass", "html_integrity_flag=warn", "html_integrity_flag=fail"]
    return np.asarray(matrix, dtype=float), medians, extra_names


def standardize(train_x: np.ndarray, other_xs: list[np.ndarray]):
    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds[stds == 0] = 1.0
    outputs = [(train_x - means) / stds]
    for matrix in other_xs:
        outputs.append((matrix - means) / stds)
    return outputs


def build_text_lsa_bundle(
    train_rows: list[dict[str, str]],
    val_rows: list[dict[str, str]],
    test_rows: list[dict[str, str]],
    text_col: str,
    max_features: int,
    min_df: int,
    lsa_components: int,
):
    train_texts = [normalize_text(row.get(text_col, "")) for row in train_rows]
    val_texts = [normalize_text(row.get(text_col, "")) for row in val_rows]
    test_texts = [normalize_text(row.get(text_col, "")) for row in test_rows]

    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, ngram_range=(1, 2))
    train_tfidf = vectorizer.fit_transform(train_texts)
    val_tfidf = vectorizer.transform(val_texts)
    test_tfidf = vectorizer.transform(test_texts)

    n_components = max(2, min(lsa_components, train_tfidf.shape[0] - 1, train_tfidf.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    train_x = svd.fit_transform(train_tfidf)
    val_x = svd.transform(val_tfidf)
    test_x = svd.transform(test_tfidf)
    train_z, val_z, test_z = standardize(train_x, [val_x, test_x])

    feature_names = [f"{text_col}_lsa_{idx+1}" for idx in range(n_components)]
    return {
        "train": train_z,
        "val": val_z,
        "test": test_z,
        "feature_names": feature_names,
    }


def fit_and_select_alpha(
    train_x: np.ndarray,
    train_y_t: np.ndarray,
    val_x: np.ndarray,
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
    coef = np.asarray(model.coef_).ravel()
    if coef.size != len(feature_names):
        return {"positive": [], "negative": []}
    order = np.argsort(coef)
    negative = [{"feature": feature_names[idx], "coef": round(float(coef[idx]), 6)} for idx in order[:limit]]
    positive = [{"feature": feature_names[idx], "coef": round(float(coef[idx]), 6)} for idx in order[-limit:][::-1]]
    return {"positive": positive, "negative": negative}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.audio_csv,
        args.target_col,
        args.qa_csv,
    )
    split = split_rows(rows, args.train_end_year, args.val_year)
    train_rows = split["train"]
    val_rows = split["val"]
    test_rows = split["test"]

    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    dense_bundles = {}
    for name, features in {
        "structured": STRUCTURED_FEATURES,
        "extra": EXTRA_DENSE_FEATURES,
        "audio": infer_audio_feature_names(rows),
        "qa_benchmark": infer_prefixed_feature_names(rows, "qa_bench_"),
    }.items():
        if not features:
            continue
        train_x, medians, extra_names = build_dense_matrix(train_rows, features)
        val_x, _, _ = build_dense_matrix(val_rows, features, medians)
        test_x, _, _ = build_dense_matrix(test_rows, features, medians)
        train_z, val_z, test_z = standardize(train_x, [val_x, test_x])
        dense_bundles[name] = {
            "train": train_z,
            "val": val_z,
            "test": test_z,
            "feature_names": features + extra_names,
        }

    text_bundles = {
        name: build_text_lsa_bundle(
            train_rows,
            val_rows,
            test_rows,
            text_col=text_col,
            max_features=args.max_features,
            min_df=args.min_df,
            lsa_components=args.lsa_components,
        )
        for name, text_col in TEXT_FEATURE_SPECS.items()
    }

    train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
    val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)
    test_y = np.asarray([row["_target"] for row in test_rows], dtype=float)
    train_y_t = transform_target(train_y)

    model_specs = {
        "structured_only": ["structured"],
        "structured_plus_extra": ["structured", "extra"],
        "audio_real_only": ["audio"],
        "structured_plus_audio_real": ["structured", "audio"],
        "qa_benchmark_only": ["qa_benchmark"],
        "structured_plus_qa_benchmark": ["structured", "qa_benchmark"],
        "qna_lsa_only": ["qna_lsa"],
        "full_lsa_only": ["full_lsa"],
        "structured_plus_qna_lsa": ["structured", "qna_lsa"],
        "structured_plus_extra_plus_audio_real": ["structured", "extra", "audio"],
        "structured_plus_extra_plus_qa_benchmark": ["structured", "extra", "qa_benchmark"],
        "structured_plus_extra_plus_qa_benchmark_plus_audio_real": ["structured", "extra", "qa_benchmark", "audio"],
        "structured_plus_extra_plus_qna_lsa": ["structured", "extra", "qna_lsa"],
        "structured_plus_extra_plus_qna_lsa_plus_qa_benchmark": ["structured", "extra", "qna_lsa", "qa_benchmark"],
        "structured_plus_extra_plus_qna_lsa_plus_audio_real": ["structured", "extra", "qna_lsa", "audio"],
        "structured_plus_extra_plus_qna_lsa_plus_qa_benchmark_plus_audio_real": [
            "structured",
            "extra",
            "qna_lsa",
            "qa_benchmark",
            "audio",
        ],
    }

    model_specs = {
        name: bundle_names
        for name, bundle_names in model_specs.items()
        if all(bundle_name in dense_bundles or bundle_name in text_bundles for bundle_name in bundle_names)
    }

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
            "lsa_components": args.lsa_components,
        },
        "models": {},
    }
    prediction_rows = []

    for model_name, bundle_names in model_specs.items():
        train_parts = []
        val_parts = []
        test_parts = []
        feature_names = []
        for bundle_name in bundle_names:
            bundle = dense_bundles.get(bundle_name) or text_bundles.get(bundle_name)
            train_parts.append(bundle["train"])
            val_parts.append(bundle["val"])
            test_parts.append(bundle["test"])
            feature_names.extend(bundle["feature_names"])

        train_x = np.hstack(train_parts)
        val_x = np.hstack(val_parts)
        test_x = np.hstack(test_parts)

        best_alpha, best_model, pred_val = fit_and_select_alpha(
            train_x,
            train_y_t,
            val_x,
            val_y,
            alphas,
        )
        pred_test = inverse_target(best_model.predict(test_x))

        summary["models"][model_name] = {
            "best_alpha": best_alpha,
            "feature_count": int(train_x.shape[1]),
            "val": metrics(val_y, pred_val),
            "test": metrics(test_y, pred_test),
            "top_coefficients": top_coefficients(best_model, feature_names),
        }

        for split_name, split_rows_, preds in [
            (f"{model_name}_val", val_rows, pred_val),
            (f"{model_name}_test", test_rows, pred_test),
        ]:
            for row, pred in zip(split_rows_, preds):
                prediction_rows.append(
                    {
                        "model_split": split_name,
                        "event_key": row["event_key"],
                        "ticker": row["ticker"],
                        "year": row["year"],
                        "y_true": row["_target"],
                        "y_pred": float(pred),
                    }
                )

    write_json(output_dir / "dense_multimodal_baseline_summary.json", summary)
    write_csv(output_dir / "dense_multimodal_baseline_predictions.csv", prediction_rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
