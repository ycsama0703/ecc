#!/usr/bin/env python3
"""
Run a minimal benchmark suite on the processed panel and fixed split.

Current benchmarks:
    - market_only
    - market_plus_controls
    - xgboost_market_controls
    - lightgbm_market_controls
    - random_forest_market_controls
    - tfidf_elasticnet
    - compact_qa_baseline
    - finbert_pooled
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.market_prior_model import MarketPriorModel


TARGET_COLUMN = "shock_minus_pre"
EVENT_ID_COLUMN = "event_id"
REPO_ROOT = Path(__file__).resolve().parent.parent

STRICT_PRECALL_MARKET_FEATURES = list(MarketPriorModel.MARKET_FEATURES)
STRICT_PRECALL_CONTROL_FEATURES = list(MarketPriorModel.CONTROL_FEATURES)


def load_panel(panel_path: Path) -> pd.DataFrame:
    if panel_path.suffix == ".parquet":
        return pd.read_parquet(panel_path)
    if panel_path.suffix == ".csv":
        return pd.read_csv(panel_path)
    raise ValueError(f"Unsupported panel format: {panel_path}")


def load_split(split_path: Path) -> pd.DataFrame:
    if split_path.suffix == ".parquet":
        split_df = pd.read_parquet(split_path)
    elif split_path.suffix == ".csv":
        split_df = pd.read_csv(split_path)
    else:
        raise ValueError(f"Unsupported split format: {split_path}")

    required = {EVENT_ID_COLUMN, "train_flag", "val_flag", "test_flag"}
    missing = required.difference(split_df.columns)
    if missing:
        raise ValueError(f"Split file missing columns: {sorted(missing)}")
    return split_df


def split_data(panel: pd.DataFrame, split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = panel.merge(
        split_df[[EVENT_ID_COLUMN, "train_flag", "val_flag", "test_flag"]],
        on=EVENT_ID_COLUMN,
        how="inner",
    )
    train = merged[merged["train_flag"] == 1].copy()
    val = merged[merged["val_flag"] == 1].copy()
    test = merged[merged["test_flag"] == 1].copy()
    return train, val, test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    residuals = y_true - y_pred
    mse = float(np.mean(residuals ** 2))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    spearman, spearman_p = scipy_stats.spearmanr(y_true, y_pred)
    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "spearman": float(spearman),
        "spearman_p": float(spearman_p),
        "n": int(len(y_true)),
    }


def save_predictions(df: pd.DataFrame, preds: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            EVENT_ID_COLUMN: df[EVENT_ID_COLUMN].values,
            TARGET_COLUMN: df[TARGET_COLUMN].values,
            "y_hat": preds,
        }
    )
    out.to_csv(path, index=False)


def resolve_existing_path(*candidates: str | None) -> Path:
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate)
        if candidate_path.exists():
            return candidate_path

        normalized = candidate.replace("\\", "/")
        if normalized.startswith("/mnt/") and len(normalized) > 6:
            drive_letter = normalized[5]
            remainder = normalized[6:].lstrip("/")
            windows_like = Path(f"{drive_letter.upper()}:/{remainder}")
            if windows_like.exists():
                return windows_like

        rel_candidate = REPO_ROOT / normalized
        if rel_candidate.exists():
            return rel_candidate

    raise FileNotFoundError(f"Could not resolve any candidate path: {candidates}")


def build_transcript_text(a1_path: str | None, a1_relpath: str | None, headline: str) -> str:
    text_parts = [headline or ""]
    resolved_path = resolve_existing_path(a1_path, a1_relpath)
    with resolved_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    for component in payload.get("components") or []:
        text = str(component.get("text") or "").strip()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def prepare_text_corpora(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[list[str], list[str], list[str]]:
    def build(df: pd.DataFrame) -> list[str]:
        return [
            build_transcript_text(
                a1_path=row.get("a1_abspath"),
                a1_relpath=row.get("a1_relpath"),
                headline=str(row.get("headline") or ""),
            )
            for _, row in df.iterrows()
        ]

    return build(train_df), build(val_df), build(test_df)


def run_market_benchmark(
    name: str,
    feature_columns: list[str],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tune: bool,
    params: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    model = MarketPriorModel(
        params=params,
        feature_columns=feature_columns,
        tune=tune,
    )
    model.fit(
        train_df,
        train_df[TARGET_COLUMN].values,
        X_val=val_df,
        y_val=val_df[TARGET_COLUMN].values,
    )
    val_pred = model.predict(val_df)
    test_pred = model.predict(test_df)
    meta = {
        "model_name": name,
        "feature_columns": feature_columns,
        "best_params": model.best_params,
        "best_val_mse": model.best_val_mse,
    }
    return val_pred, test_pred, meta


def prepare_numeric_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = train_df[feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    X_val = val_df[feature_columns].apply(pd.to_numeric, errors="coerce").copy()
    X_test = test_df[feature_columns].apply(pd.to_numeric, errors="coerce").copy()

    medians = X_train.median(axis=0, numeric_only=True).fillna(0.0)
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)
    return X_train, X_val, X_test


def run_xgboost_tabular(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"xgboost is required for {model_name}. Install it before running this benchmark."
        ) from exc

    X_train, X_val, X_test = prepare_numeric_features(train_df, val_df, test_df, feature_columns)
    y_train = train_df[TARGET_COLUMN].values.astype(np.float64)
    y_val = val_df[TARGET_COLUMN].values.astype(np.float64)

    grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": [200, 400],
    }

    best_model = None
    best_params = None
    best_score = None

    for max_depth, learning_rate, n_estimators in product(
        grid["max_depth"], grid["learning_rate"], grid["n_estimators"]
    ):
        model = XGBRegressor(
            objective="reg:squarederror",
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = float(np.mean((y_val - val_pred) ** 2))
        if best_score is None or score < best_score:
            best_model = model
            best_params = {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
            }
            best_score = score

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    meta = {
        "model_name": model_name,
        "best_params": best_params,
        "best_val_mse": best_score,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    return val_pred, test_pred, meta


def run_lightgbm_tabular(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"lightgbm is required for {model_name}. Install it before running this benchmark."
        ) from exc

    X_train, X_val, X_test = prepare_numeric_features(train_df, val_df, test_df, feature_columns)
    y_train = train_df[TARGET_COLUMN].values.astype(np.float64)
    y_val = val_df[TARGET_COLUMN].values.astype(np.float64)

    grid = {
        "num_leaves": [15, 31, 63],
        "learning_rate": [0.03, 0.05, 0.1],
        "n_estimators": [200, 400],
    }

    best_model = None
    best_params = None
    best_score = None

    for num_leaves, learning_rate, n_estimators in product(
        grid["num_leaves"], grid["learning_rate"], grid["n_estimators"]
    ):
        model = LGBMRegressor(
            objective="regression",
            num_leaves=num_leaves,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = float(np.mean((y_val - val_pred) ** 2))
        if best_score is None or score < best_score:
            best_model = model
            best_params = {
                "num_leaves": num_leaves,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
            }
            best_score = score

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    meta = {
        "model_name": model_name,
        "best_params": best_params,
        "best_val_mse": best_score,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    return val_pred, test_pred, meta


def run_random_forest_tabular(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        from sklearn.ensemble import RandomForestRegressor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            f"scikit-learn is required for {model_name}. Install it before running this benchmark."
        ) from exc

    X_train, X_val, X_test = prepare_numeric_features(train_df, val_df, test_df, feature_columns)
    y_train = train_df[TARGET_COLUMN].values.astype(np.float64)
    y_val = val_df[TARGET_COLUMN].values.astype(np.float64)

    grid = {
        "n_estimators": [300, 600],
        "max_depth": [4, 6, None],
        "min_samples_leaf": [1, 3, 5],
    }

    best_model = None
    best_params = None
    best_score = None

    for n_estimators, max_depth, min_samples_leaf in product(
        grid["n_estimators"],
        grid["max_depth"],
        grid["min_samples_leaf"],
    ):
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = float(np.mean((y_val - val_pred) ** 2))
        if best_score is None or score < best_score:
            best_model = model
            best_params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
            }
            best_score = score

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    meta = {
        "model_name": model_name,
        "best_params": best_params,
        "best_val_mse": best_score,
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
    }
    return val_pred, test_pred, meta


def run_tfidf_elasticnet(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import ElasticNet
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "scikit-learn is required for tfidf_elasticnet. "
            "Install it with `pip install scikit-learn` or your conda equivalent."
        ) from exc

    train_texts, val_texts, test_texts = prepare_text_corpora(train_df, val_df, test_df)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    grid = {
        "alpha": [0.0005, 0.001, 0.005, 0.01, 0.05],
        "l1_ratio": [0.1, 0.5, 0.9],
    }

    best_model = None
    best_params = None
    best_score = None

    y_train = train_df[TARGET_COLUMN].values.astype(np.float64)
    y_val = val_df[TARGET_COLUMN].values.astype(np.float64)

    for alpha, l1_ratio in product(grid["alpha"], grid["l1_ratio"]):
        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            fit_intercept=True,
            max_iter=5000,
            random_state=42,
        )
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = float(np.mean((y_val - val_pred) ** 2))
        if best_score is None or score < best_score:
            best_model = model
            best_params = {"alpha": alpha, "l1_ratio": l1_ratio}
            best_score = score

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    meta = {
        "model_name": "tfidf_elasticnet",
        "best_params": best_params,
        "best_val_mse": best_score,
        "vocab_size": int(len(vectorizer.vocabulary_)),
    }
    return val_pred, test_pred, meta


def run_compact_qa_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "scikit-learn is required for compact_qa_baseline. "
            "Install it with `pip install scikit-learn` or your conda equivalent."
        ) from exc

    qa_columns = sorted(col for col in train_df.columns if col.startswith("qa_embedding_"))
    if not qa_columns:
        raise ValueError("No qa_embedding_* columns found in panel.")

    X_train = train_df[qa_columns].values.astype(np.float64)
    X_val = val_df[qa_columns].values.astype(np.float64)
    X_test = test_df[qa_columns].values.astype(np.float64)
    y_train = train_df[TARGET_COLUMN].values.astype(np.float64)
    y_val = val_df[TARGET_COLUMN].values.astype(np.float64)

    candidates = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        candidates.append(
            (
                {"model_family": "ridge", "alpha": alpha},
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=alpha)),
                    ]
                ),
            )
        )
    for alpha, l1_ratio in product([0.0005, 0.001, 0.005, 0.01], [0.1, 0.5, 0.9]):
        candidates.append(
            (
                {"model_family": "elasticnet", "alpha": alpha, "l1_ratio": l1_ratio},
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            ElasticNet(
                                alpha=alpha,
                                l1_ratio=l1_ratio,
                                fit_intercept=True,
                                max_iter=5000,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            )
        )

    best_model = None
    best_params = None
    best_score = None

    for params, model in candidates:
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = float(np.mean((y_val - val_pred) ** 2))
        if best_score is None or score < best_score:
            best_model = model
            best_params = params
            best_score = score

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    meta = {
        "model_name": "compact_qa_baseline",
        "best_params": best_params,
        "best_val_mse": best_score,
        "feature_count": len(qa_columns),
    }
    return val_pred, test_pred, meta


def infer_torch_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_finbert_embeddings(
    texts: list[str],
    model_name: str,
    device: str,
    max_length: int,
    max_chunks: int,
    pooling: str,
) -> np.ndarray:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "transformers and torch are required for finbert_pooled. "
            "Install them in your environment before running this benchmark."
        ) from exc

    resolved_device = infer_torch_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(resolved_device)
    model.eval()

    chunk_token_budget = max(8, max_length - 2)
    embeddings = []

    with torch.no_grad():
        for text in texts:
            token_ids = tokenizer.encode(text or "", add_special_tokens=False)
            if not token_ids:
                token_ids = tokenizer.encode("[EMPTY]", add_special_tokens=False)
            token_chunks = [
                token_ids[idx : idx + chunk_token_budget]
                for idx in range(0, len(token_ids), chunk_token_budget)
            ][:max_chunks]

            chunk_embeddings = []
            for chunk in token_chunks:
                encoded = tokenizer.prepare_for_model(
                    chunk,
                    add_special_tokens=True,
                    truncation=True,
                    max_length=max_length,
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )
                input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long, device=resolved_device).unsqueeze(0)
                attention_mask = torch.tensor(
                    encoded["attention_mask"],
                    dtype=torch.long,
                    device=resolved_device,
                ).unsqueeze(0)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                hidden = outputs.last_hidden_state
                if pooling == "cls":
                    pooled = hidden[:, 0, :]
                else:
                    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                    pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

                chunk_embeddings.append(pooled.squeeze(0).detach().cpu().numpy())

            doc_embedding = np.mean(np.vstack(chunk_embeddings), axis=0)
            embeddings.append(doc_embedding)

    return np.vstack(embeddings).astype(np.float64)


def run_finbert_pooled(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str,
    device: str,
    max_length: int,
    max_chunks: int,
    pooling: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    try:
        from sklearn.linear_model import ElasticNet, Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "scikit-learn is required for finbert_pooled. "
            "Install it with `pip install scikit-learn` or your conda equivalent."
        ) from exc

    train_texts, val_texts, test_texts = prepare_text_corpora(train_df, val_df, test_df)
    X_train = build_finbert_embeddings(
        texts=train_texts,
        model_name=model_name,
        device=device,
        max_length=max_length,
        max_chunks=max_chunks,
        pooling=pooling,
    )
    X_val = build_finbert_embeddings(
        texts=val_texts,
        model_name=model_name,
        device=device,
        max_length=max_length,
        max_chunks=max_chunks,
        pooling=pooling,
    )
    X_test = build_finbert_embeddings(
        texts=test_texts,
        model_name=model_name,
        device=device,
        max_length=max_length,
        max_chunks=max_chunks,
        pooling=pooling,
    )
    y_train = train_df[TARGET_COLUMN].values.astype(np.float64)
    y_val = val_df[TARGET_COLUMN].values.astype(np.float64)

    candidates = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        candidates.append(
            (
                {"model_family": "ridge", "alpha": alpha},
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("model", Ridge(alpha=alpha)),
                    ]
                ),
            )
        )
    for alpha, l1_ratio in product([0.0005, 0.001, 0.005], [0.1, 0.5]):
        candidates.append(
            (
                {"model_family": "elasticnet", "alpha": alpha, "l1_ratio": l1_ratio},
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            ElasticNet(
                                alpha=alpha,
                                l1_ratio=l1_ratio,
                                fit_intercept=True,
                                max_iter=5000,
                                random_state=42,
                            ),
                        ),
                    ]
                ),
            )
        )

    best_model = None
    best_params = None
    best_score = None
    for params, model in candidates:
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        score = float(np.mean((y_val - val_pred) ** 2))
        if best_score is None or score < best_score:
            best_model = model
            best_params = params
            best_score = score

    val_pred = best_model.predict(X_val)
    test_pred = best_model.predict(X_test)
    meta = {
        "model_name": "finbert_pooled",
        "best_params": best_params,
        "best_val_mse": best_score,
        "encoder_model": model_name,
        "device": infer_torch_device(device),
        "max_length": max_length,
        "max_chunks": max_chunks,
        "pooling": pooling,
        "embedding_dim": int(X_train.shape[1]),
    }
    return val_pred, test_pred, meta


def benchmark_row(model_name: str, split_name: str, metrics: dict, meta: dict) -> dict:
    return {
        "model": model_name,
        "split": split_name,
        "mse": metrics["mse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "spearman": metrics["spearman"],
        "spearman_p": metrics["spearman_p"],
        "n": metrics["n"],
        "best_val_mse": meta.get("best_val_mse"),
        "best_params": json.dumps(meta.get("best_params"), ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run minimal benchmark baselines.")
    parser.add_argument("--panel", type=Path, required=True, help="Processed panel path.")
    parser.add_argument("--split", type=Path, required=True, help="Split file path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks"),
        help="Benchmark output directory.",
    )
    parser.add_argument("--split-version", type=str, default="v1")
    parser.add_argument("--run-id", type=str, default="bench01")
    parser.add_argument(
        "--tune-market-baselines",
        action="store_true",
        help="Tune market baselines on validation set.",
    )
    parser.add_argument(
        "--strict-precall",
        action="store_true",
        help="Use only decision-time-safe pre-call features and skip post-start text/ECC baselines.",
    )
    parser.add_argument(
        "--finbert-model",
        type=str,
        default="ProsusAI/finbert",
        help="Hugging Face model name for FinBERT embedding extraction.",
    )
    parser.add_argument(
        "--finbert-device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for FinBERT embedding extraction.",
    )
    parser.add_argument(
        "--finbert-max-length",
        type=int,
        default=256,
        help="Per-chunk token length for FinBERT inputs.",
    )
    parser.add_argument(
        "--finbert-max-chunks",
        type=int,
        default=4,
        help="Maximum transcript chunks encoded per document.",
    )
    parser.add_argument(
        "--finbert-pooling",
        type=str,
        default="mean",
        choices=["mean", "cls"],
        help="Pooling strategy within each FinBERT chunk.",
    )
    args = parser.parse_args()

    panel = load_panel(args.panel.resolve())
    split_df = load_split(args.split.resolve())
    train_df, val_df, test_df = split_data(panel, split_df)
    if args.strict_precall:
        market_only_name = "market_precall_only"
        market_controls_name = "market_precall_plus_controls"
        xgb_name = "xgboost_precall_controls"
        lgb_name = "lightgbm_precall_controls"
        rf_name = "random_forest_precall_controls"
        market_features = STRICT_PRECALL_MARKET_FEATURES
        market_controls_features = STRICT_PRECALL_MARKET_FEATURES + STRICT_PRECALL_CONTROL_FEATURES
    else:
        market_only_name = "market_only"
        market_controls_name = "market_plus_controls"
        xgb_name = "xgboost_market_controls"
        lgb_name = "lightgbm_market_controls"
        rf_name = "random_forest_market_controls"
        market_features = MarketPriorModel.MARKET_FEATURES
        market_controls_features = MarketPriorModel.MARKET_FEATURES + MarketPriorModel.CONTROL_FEATURES

    benchmarks = [
        (
            market_only_name,
            lambda: run_market_benchmark(
                name=market_only_name,
                feature_columns=market_features,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                tune=args.tune_market_baselines,
                params=None,
            ),
        ),
        (
            market_controls_name,
            lambda: run_market_benchmark(
                name=market_controls_name,
                feature_columns=market_controls_features,
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                tune=args.tune_market_baselines,
                params=None,
            ),
        ),
        (
            xgb_name,
            lambda: run_xgboost_tabular(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_columns=market_controls_features,
                model_name=xgb_name,
            ),
        ),
        (
            lgb_name,
            lambda: run_lightgbm_tabular(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_columns=market_controls_features,
                model_name=lgb_name,
            ),
        ),
        (
            rf_name,
            lambda: run_random_forest_tabular(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                feature_columns=market_controls_features,
                model_name=rf_name,
            ),
        ),
    ]

    if not args.strict_precall:
        benchmarks.extend(
            [
                (
                    "tfidf_elasticnet",
                    lambda: run_tfidf_elasticnet(
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                    ),
                ),
                (
                    "compact_qa_baseline",
                    lambda: run_compact_qa_baseline(
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                    ),
                ),
                (
                    "finbert_pooled",
                    lambda: run_finbert_pooled(
                        train_df=train_df,
                        val_df=val_df,
                        test_df=test_df,
                        model_name=args.finbert_model,
                        device=args.finbert_device,
                        max_length=args.finbert_max_length,
                        max_chunks=args.finbert_max_chunks,
                        pooling=args.finbert_pooling,
                    ),
                ),
            ]
        )

    rows = []
    predictions_dir = args.output_dir.resolve() / "predictions"
    metrics_dir = args.output_dir.resolve() / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    meta_summary = {
        "run_id": args.run_id,
        "split_version": args.split_version,
        "panel_path": str(args.panel.resolve()),
        "split_path": str(args.split.resolve()),
        "strict_precall": args.strict_precall,
        "benchmarks": {},
    }

    for name, runner in benchmarks:
        val_pred, test_pred, meta = runner()
        val_metrics = compute_metrics(val_df[TARGET_COLUMN].values, val_pred)
        test_metrics = compute_metrics(test_df[TARGET_COLUMN].values, test_pred)

        rows.append(benchmark_row(name, "val", val_metrics, meta))
        rows.append(benchmark_row(name, "test", test_metrics, meta))

        save_predictions(
            val_df,
            val_pred,
            predictions_dir / f"{name}_val_{args.split_version}_{args.run_id}.csv",
        )
        save_predictions(
            test_df,
            test_pred,
            predictions_dir / f"{name}_test_{args.split_version}_{args.run_id}.csv",
        )
        meta_summary["benchmarks"][name] = meta

    results_path = metrics_dir / f"benchmark_results_{args.split_version}_{args.run_id}.csv"
    pd.DataFrame(rows).to_csv(results_path, index=False)

    summary_path = metrics_dir / f"benchmark_summary_{args.split_version}_{args.run_id}.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(meta_summary, handle, indent=2, ensure_ascii=False)

    print("Benchmark run complete")
    print(f"  output: {results_path}")


if __name__ == "__main__":
    main()
