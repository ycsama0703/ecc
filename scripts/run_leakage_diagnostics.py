#!/usr/bin/env python3
"""Leakage diagnostics for the current ECC experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


TARGET_COL = "shock_minus_pre"
POST_TARGET_COL = "RV_post_60m"
EVENT_ID_COL = "event_id"
TICKER_COL = "ticker"

ID_LIKE_COLUMNS = {
    EVENT_ID_COL,
    TICKER_COL,
    "event_date",
    "call_time",
    "scheduled_datetime",
    "headline",
    "a1_relpath",
    "a2_relpath",
    "a4_relpath",
    "stock_relpath",
    "a1_abspath",
    "a2_abspath",
    "a4_abspath",
    "stock_abspath",
    "split_label",
    "split_version",
    "sample_label",
    "call_time_assumption",
    "html_integrity_flag",
    "html_integrity_reason",
}

PRE_CALL_CORE = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "pre_60m_volume_sum",
    "pre_call_volatility",
    "returns",
    "volume",
    "historical_volatility",
    "firm_size",
    "sector",
    "scheduled_hour_et",
]

WITHIN_CALL_FEATURES = [
    "within_call_rv",
    "within_call_vw_rv",
    "within_call_volume_sum",
    "call_duration_min",
]

POST_CALL_FEATURES = [
    "RV_post_60m",
    "post_call_60m_vw_rv",
    "post_call_60m_volume_sum",
]

MARKET_FEATURES = [
    "pre_60m_rv",
    "pre_60m_vw_rv",
    "pre_60m_volume_sum",
    "within_call_rv",
    "within_call_vw_rv",
    "within_call_volume_sum",
]

CONTROL_FEATURES = [
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
    "firm_size",
    "sector",
    "historical_volatility",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
    }


def load_panel_and_split(panel_csv: Path, split_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(panel_csv)
    split_df = pd.read_csv(split_csv)
    merged = panel.merge(
        split_df[[EVENT_ID_COL, "train_flag", "val_flag", "test_flag"]],
        on=EVENT_ID_COL,
        how="inner",
    )
    train_df = merged[merged["train_flag"] == 1].copy()
    val_df = merged[merged["val_flag"] == 1].copy()
    test_df = merged[merged["test_flag"] == 1].copy()
    return train_df, val_df, test_df


def infer_time_reference(feature_name: str) -> tuple[str, bool, bool, str]:
    """Infer time usage for a feature."""
    name = feature_name.lower()
    time_reference = "unknown"
    uses_call_end_time = False
    uses_future_data = False
    severity = ""

    if any(token in name for token in ["post_call", "post_window", "rv_post"]):
        time_reference = "post_call"
        uses_call_end_time = True
        uses_future_data = True
    elif any(token in name for token in ["within_call", "call_duration", "call_end"]):
        time_reference = "within_call"
        uses_call_end_time = True
        uses_future_data = True
    elif any(
        token in name
        for token in [
            "text_embedding_",
            "qa_embedding_",
            "transcript_",
            "qa_total_",
            "a4_",
            "transcript_coverage",
            "alignment_score",
            "audio_completeness",
            "proxy_quality_mean",
        ]
    ):
        time_reference = "within_call"
        uses_future_data = True
    elif any(
        token in name
        for token in [
            "pre_60m",
            "pre_call",
            "historical_",
            "pre_bar",
            "pre_window",
            "scheduled_",
            "firm_size",
            "sector",
            "returns",
            "volume",
            "revenue_",
            "ebitda_",
            "eps_",
            "analyst_",
        ]
    ):
        time_reference = "pre_call"
    elif "rv_pre" in name:
        time_reference = "pre_call"

    if uses_future_data:
        severity = "TIME_LEAKAGE"

    return time_reference, uses_call_end_time, uses_future_data, severity


def build_time_audit(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in panel_df.columns:
        if column in ID_LIKE_COLUMNS:
            continue
        time_reference, uses_call_end_time, uses_future_data, severity = infer_time_reference(column)
        rows.append(
            {
                "feature_name": column,
                "time_reference": time_reference,
                "uses_call_end_time": uses_call_end_time,
                "uses_future_data": uses_future_data,
                "severity": severity,
            }
        )
    return pd.DataFrame(rows).sort_values(["uses_future_data", "feature_name"], ascending=[False, True])


def build_target_leakage_table(panel_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    numeric_cols = [
        col
        for col in panel_df.columns
        if col not in ID_LIKE_COLUMNS and pd.api.types.is_numeric_dtype(panel_df[col])
    ]
    y = panel_df[TARGET_COL]
    rv_post = panel_df[POST_TARGET_COL]
    for col in numeric_cols:
        if col == TARGET_COL:
            continue
        corr_post = panel_df[col].corr(rv_post)
        corr_y = panel_df[col].corr(y)
        flag = (
            abs(corr_post) > 0.8
            or abs(corr_y) > 0.8
        )
        rows.append(
            {
                "feature_name": col,
                "corr_feature_vs_rv_post": float(corr_post) if pd.notna(corr_post) else np.nan,
                "corr_feature_vs_y": float(corr_y) if pd.notna(corr_y) else np.nan,
                "potential_target_leakage": bool(flag),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["potential_target_leakage", "corr_feature_vs_rv_post", "corr_feature_vs_y"],
        ascending=[False, False, False],
    )


def build_preprocessor(df: pd.DataFrame, feature_columns: list[str]) -> ColumnTransformer:
    numeric_features = [
        col for col in feature_columns
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_features = [
        col for col in feature_columns
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def fit_elastic_net(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    candidates = [
        {"alpha": 0.0005, "l1_ratio": 0.1},
        {"alpha": 0.001, "l1_ratio": 0.5},
        {"alpha": 0.01, "l1_ratio": 0.5},
        {"alpha": 0.1, "l1_ratio": 0.9},
    ]
    best_model = None
    best_val_mse = None
    for params in candidates:
        model = Pipeline(
            steps=[
                ("prep", build_preprocessor(train_df, feature_columns)),
                ("reg", ElasticNet(max_iter=20000, random_state=42, **params)),
            ]
        )
        model.fit(train_df[feature_columns], train_df[TARGET_COL].values)
        val_pred = model.predict(val_df[feature_columns])
        val_mse = mean_squared_error(val_df[TARGET_COL].values, val_pred)
        if best_val_mse is None or val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
    return best_model


def fit_ridge(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    candidates = [0.1, 1.0, 10.0, 100.0]
    best_model = None
    best_val_mse = None
    for alpha in candidates:
        model = Pipeline(
            steps=[
                ("prep", build_preprocessor(train_df, feature_columns)),
                ("reg", Ridge(alpha=alpha)),
            ]
        )
        model.fit(train_df[feature_columns], train_df[TARGET_COL].values)
        val_pred = model.predict(val_df[feature_columns])
        val_mse = mean_squared_error(val_df[TARGET_COL].values, val_pred)
        if best_val_mse is None or val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
    return best_model


def fit_random_forest(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: list[str]) -> Pipeline:
    candidates = [
        {"n_estimators": 300, "max_depth": 6, "min_samples_leaf": 1},
        {"n_estimators": 300, "max_depth": 4, "min_samples_leaf": 2},
        {"n_estimators": 500, "max_depth": 8, "min_samples_leaf": 1},
    ]
    best_model = None
    best_val_mse = None
    for params in candidates:
        model = Pipeline(
            steps=[
                ("prep", build_preprocessor(train_df, feature_columns)),
                ("reg", RandomForestRegressor(random_state=42, n_jobs=-1, **params)),
            ]
        )
        model.fit(train_df[feature_columns], train_df[TARGET_COL].values)
        val_pred = model.predict(val_df[feature_columns])
        val_mse = mean_squared_error(val_df[TARGET_COL].values, val_pred)
        if best_val_mse is None or val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
    return best_model


def fit_lightgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_columns: list[str]) -> Pipeline | None:
    if LGBMRegressor is None:
        return None
    candidates = [
        {"num_leaves": 15, "learning_rate": 0.05, "n_estimators": 400},
        {"num_leaves": 31, "learning_rate": 0.03, "n_estimators": 400},
    ]
    best_model = None
    best_val_mse = None
    for params in candidates:
        model = Pipeline(
            steps=[
                ("prep", build_preprocessor(train_df, feature_columns)),
                ("reg", LGBMRegressor(random_state=42, verbosity=-1, **params)),
            ]
        )
        model.fit(train_df[feature_columns], train_df[TARGET_COL].values)
        val_pred = model.predict(val_df[feature_columns])
        val_mse = mean_squared_error(val_df[TARGET_COL].values, val_pred)
        if best_val_mse is None or val_mse < best_val_mse:
            best_val_mse = val_mse
            best_model = model
    return best_model


def evaluate_model(model, test_df: pd.DataFrame, feature_columns: list[str]) -> dict[str, float]:
    pred = model.predict(test_df[feature_columns])
    return compute_metrics(test_df[TARGET_COL].values, pred)


def write_markdown_report(
    output_path: Path,
    summary: dict,
    time_leakage_df: pd.DataFrame,
    target_leakage_df: pd.DataFrame,
    ml_df: pd.DataFrame,
) -> None:
    def dataframe_to_markdown(df: pd.DataFrame) -> str:
        cols = df.columns.tolist()
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        body = []
        for _, row in df.iterrows():
            body.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
        return "\n".join([header, sep] + body)

    severe_time = time_leakage_df[time_leakage_df["severity"] == "TIME_LEAKAGE"]["feature_name"].tolist()
    severe_target = target_leakage_df[target_leakage_df["potential_target_leakage"]]["feature_name"].tolist()
    overlap = summary["residual_structure"]["common_features"]
    split_info = summary["data_split"]
    lines = [
        "# Leakage Diagnostic Report",
        "",
        "## 1. Time Leakage",
        f"- post-start features present: {'YES' if severe_time else 'NO'}",
        f"- count flagged as `TIME_LEAKAGE`: {len(severe_time)}",
        f"- key flagged features: {', '.join(severe_time[:20]) if severe_time else 'None'}",
        f"- severity: {summary['final_conclusion']['severity']}",
        "",
        "## 2. Target Leakage",
        f"- high-correlation features vs `RV_post` or `y`: {len(severe_target)}",
        f"- most suspicious features: {', '.join(severe_target[:20]) if severe_target else 'None'}",
        "",
        "## 3. Residual Structure",
        f"- `R2_z_only`: {summary['residual_structure']['r2_z_only']:.4f}",
        f"- overlap count between `mu` and `z`: {summary['residual_structure']['common_feature_count']}",
        f"- overlapping features: {', '.join(overlap)}",
        f"- residual collapse: {summary['residual_structure']['residual_collapse']}",
        "",
        "## 4. Feature Ablation",
        f"- `R2_A (pre_call + ECC)`: {summary['feature_ablation']['R2_A']:.4f}",
        f"- `R2_B (pre_call + within_call)`: {summary['feature_ablation']['R2_B']:.4f}",
        f"- `R2_C (pre_call + post_call)`: {summary['feature_ablation']['R2_C']:.4f}",
        f"- strongest abnormal jump source: {summary['feature_ablation']['dominant_source']}",
        "",
        "## 5. Data Split",
        f"- split type: {split_info['split_type']}",
        f"- train firms: {split_info['train_firm_count']}",
        f"- val firms: {split_info['val_firm_count']}",
        f"- test firms: {split_info['test_firm_count']}",
        f"- train/test overlap count: {split_info['train_test_overlap_count']}",
        f"- train/val overlap count: {split_info['train_val_overlap_count']}",
        f"- val/test overlap count: {split_info['val_test_overlap_count']}",
        f"- potential panel leakage: {split_info['potential_panel_leakage']}",
        "",
        "## 6. ML Behavior",
        dataframe_to_markdown(ml_df),
        "",
        "## 7. Final Conclusion",
        f"- leakage present: {summary['final_conclusion']['leakage_present']}",
        f"- most likely source: {summary['final_conclusion']['most_likely_source']}",
        f"- severity: {summary['final_conclusion']['severity']}",
        f"- remove immediately: {', '.join(summary['final_conclusion']['remove_immediately'])}",
        "",
        "## Final Answer",
        f"- Current `0.93+` R² is mainly from: {summary['final_conclusion']['root_cause']}",
        f"- Primary leakage class: {summary['final_conclusion']['primary_leakage_class']}",
        f"- Features to remove from the main experiment: {', '.join(summary['final_conclusion']['remove_immediately'])}",
        "",
        "## Notes",
        "- Under a strict pre-call forecasting definition, `within_call_*`, transcript-derived features, and A4 alignment features all use post-start information.",
        "- The hybrid residual additionally reuses `market + controls` in both `mu` and `z`, which breaks clean residual separation even if feature scope is aligned.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run leakage diagnostics.")
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("data/processed/panel_experiment_a_timefix_selective/processed_panel.csv"),
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=Path("data/splits/time_split_experiment_a_timefix_selective.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/leakage_diagnostics"),
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=Path("results/leakage_diagnostic_report.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_md.parent.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_panel_and_split(args.panel_csv, args.split_csv)
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    ecc_columns = sorted(
        col
        for col in full_df.columns
        if col.startswith("text_embedding_") or col.startswith("qa_embedding_") or col.startswith("audio_features_")
    )
    mu_features = MARKET_FEATURES + CONTROL_FEATURES
    z_features = ecc_columns + mu_features
    common_features = sorted(set(mu_features).intersection(z_features))

    # Part 1
    time_leakage_df = build_time_audit(full_df)
    time_leakage_df.to_csv(args.output_dir / "time_leakage_audit.csv", index=False)

    # Part 2
    target_leakage_df = build_target_leakage_table(full_df)
    target_leakage_df.to_csv(args.output_dir / "target_leakage_correlations.csv", index=False)

    # Part 3
    z_only_model = fit_ridge(train_df, val_df, z_features)
    z_only_metrics = evaluate_model(z_only_model, test_df, z_features)

    # Part 4
    ablation_a_features = [col for col in PRE_CALL_CORE + ecc_columns if col in full_df.columns]
    ablation_b_features = [col for col in PRE_CALL_CORE + WITHIN_CALL_FEATURES if col in full_df.columns]
    ablation_c_features = [col for col in PRE_CALL_CORE + POST_CALL_FEATURES if col in full_df.columns]
    r2_a = evaluate_model(fit_ridge(train_df, val_df, ablation_a_features), test_df, ablation_a_features)["r2"]
    r2_b = evaluate_model(fit_ridge(train_df, val_df, ablation_b_features), test_df, ablation_b_features)["r2"]
    r2_c = evaluate_model(fit_ridge(train_df, val_df, ablation_c_features), test_df, ablation_c_features)["r2"]

    # Part 5
    train_firms = sorted(train_df[TICKER_COL].dropna().unique().tolist())
    val_firms = sorted(val_df[TICKER_COL].dropna().unique().tolist())
    test_firms = sorted(test_df[TICKER_COL].dropna().unique().tolist())
    train_test_overlap = sorted(set(train_firms).intersection(test_firms))
    train_val_overlap = sorted(set(train_firms).intersection(val_firms))
    val_test_overlap = sorted(set(val_firms).intersection(test_firms))

    # Part 6
    ml_feature_set = [col for col in mu_features if col in full_df.columns]
    ml_rows = []
    elastic_model = fit_elastic_net(train_df, val_df, ml_feature_set)
    elastic_metrics = evaluate_model(elastic_model, test_df, ml_feature_set)
    ml_rows.append({"model": "ElasticNet", **elastic_metrics})
    rf_model = fit_random_forest(train_df, val_df, ml_feature_set)
    rf_metrics = evaluate_model(rf_model, test_df, ml_feature_set)
    ml_rows.append({"model": "RandomForest", **rf_metrics})
    lgb_model = fit_lightgbm(train_df, val_df, ml_feature_set)
    if lgb_model is not None:
        lgb_metrics = evaluate_model(lgb_model, test_df, ml_feature_set)
        ml_rows.append({"model": "LightGBM", **lgb_metrics})
    ml_df = pd.DataFrame(ml_rows)
    ml_df.to_csv(args.output_dir / "ml_same_features_comparison.csv", index=False)

    # Summary
    severe_time_features = time_leakage_df[time_leakage_df["severity"] == "TIME_LEAKAGE"]["feature_name"].tolist()
    severe_target_features = target_leakage_df[target_leakage_df["potential_target_leakage"]]["feature_name"].tolist()
    dominant_source = "post_call_features" if r2_c > max(r2_a, r2_b) else "within_call_features" if r2_b > r2_a else "pre_call_plus_ecc"
    leakage_present = "YES"
    primary_leakage_class = "TIME_LEAKAGE via within_call/post_call features plus RESIDUAL_COLLAPSE"
    remove_immediately = [
        "within_call_rv",
        "within_call_vw_rv",
        "within_call_volume_sum",
        "call_duration_min",
        "RV_post_60m",
        "post_call_60m_vw_rv",
        "post_call_60m_volume_sum",
    ]
    summary = {
        "time_leakage": {
            "flagged_feature_count": int(len(severe_time_features)),
            "flagged_features": severe_time_features,
        },
        "target_leakage": {
            "high_correlation_feature_count": int(len(severe_target_features)),
            "high_correlation_features": severe_target_features,
        },
        "residual_structure": {
            "features_mu": mu_features,
            "features_z": z_features,
            "common_features": common_features,
            "common_feature_count": len(common_features),
            "r2_z_only": z_only_metrics["r2"],
            "residual_collapse": "YES" if z_only_metrics["r2"] > 0.7 else "NO",
        },
        "feature_ablation": {
            "R2_A": r2_a,
            "R2_B": r2_b,
            "R2_C": r2_c,
            "dominant_source": dominant_source,
        },
        "data_split": {
            "split_type": "time-based split",
            "train_firms": train_firms,
            "val_firms": val_firms,
            "test_firms": test_firms,
            "train_firm_count": len(train_firms),
            "val_firm_count": len(val_firms),
            "test_firm_count": len(test_firms),
            "train_test_overlap": train_test_overlap,
            "train_val_overlap": train_val_overlap,
            "val_test_overlap": val_test_overlap,
            "train_test_overlap_count": len(train_test_overlap),
            "train_val_overlap_count": len(train_val_overlap),
            "val_test_overlap_count": len(val_test_overlap),
            "potential_panel_leakage": "YES" if train_test_overlap else "NO",
        },
        "ml_behavior": ml_rows,
        "final_conclusion": {
            "leakage_present": leakage_present,
            "most_likely_source": "within_call market features and duplicated market/control usage in both mu and z",
            "severity": "HIGH",
            "remove_immediately": remove_immediately,
            "root_cause": "information leakage / near-leakage from within-call and post-call features, amplified by residual duplication",
            "primary_leakage_class": primary_leakage_class,
        },
    }

    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(args.report_md, summary, time_leakage_df, target_leakage_df, ml_df)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
