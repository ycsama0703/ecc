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

from dj30_qc_utils import write_csv, write_json
from run_dense_multimodal_ablation_baselines import build_text_lsa_bundle, load_joined_rows
from run_prior_residual_ridge_baselines import build_dense_bundle, fit_residual_ridge
from run_structured_baselines import metrics
from run_target_variant_experiments import derived_targets
from run_text_tfidf_baselines import STRUCTURED_FEATURES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a harder unseen-ticker stress test on the off-hours shock setting."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--features-csv", type=Path, required=True)
    parser.add_argument("--audio-csv", type=Path, required=True)
    parser.add_argument("--qa-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/unseen_ticker_stress_real"),
    )
    parser.add_argument("--target-variant", default="shock_minus_pre")
    parser.add_argument("--include-regimes", default="pre_market,after_hours")
    parser.add_argument("--exclude-html-flags", default="")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--min-test-events", type=int, default=3)
    parser.add_argument("--with-qna-lsa", action="store_true")
    return parser.parse_args()


def regime_label(row: dict[str, str]) -> str:
    hour = float(row.get("scheduled_hour_et", 0.0))
    if hour < 9.5:
        return "pre_market"
    if hour < 16.0:
        return "market_hours"
    return "after_hours"


def constant_prior(length: int, value: float) -> np.ndarray:
    return np.full(length, float(value), dtype=float)


def ticker_metrics(rows: list[dict[str, str]], preds: np.ndarray) -> dict[str, float]:
    y_true = np.asarray([row["_target"] for row in rows], dtype=float)
    out = metrics(y_true, preds)
    out["n"] = int(len(rows))
    return out


def summarize_by_ticker(rows: list[dict[str, str]], prediction_lookup: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    grouped = {}
    for ticker in sorted({row["ticker"] for row in rows}):
        ticker_rows = [row for row in rows if row["ticker"] == ticker]
        out = {}
        for model_name, preds in prediction_lookup.items():
            ticker_preds = np.asarray(
                [pred for row, pred in zip(rows, preds, strict=True) if row["ticker"] == ticker],
                dtype=float,
            )
            out[model_name] = ticker_metrics(ticker_rows, ticker_preds)
        grouped[ticker] = out
    return grouped


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}
    alphas = [float(item) for item in args.alphas.split(",") if item.strip()]

    base_rows = load_joined_rows(
        args.panel_csv,
        args.features_csv,
        args.audio_csv,
        "post_call_60m_rv",
        args.qa_csv,
    )

    rows = []
    for row in base_rows:
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        if html_flag in exclude_html_flags:
            continue
        regime = regime_label(row)
        if regime not in include_regimes:
            continue
        target_value = derived_targets(row, args.eps).get(args.target_variant)
        if target_value is None or not math.isfinite(target_value):
            continue
        item = dict(row)
        item["_target"] = float(target_value)
        item["_regime"] = regime
        rows.append(item)

    candidate_tickers = sorted(
        {
            row["ticker"]
            for row in rows
            if row["_year"] > args.val_year and row.get("ticker")
        }
    )

    prediction_rows = []
    overall_rows = []
    skipped = {}

    for ticker in candidate_tickers:
        train_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] <= args.train_end_year]
        val_rows = [row for row in rows if row["ticker"] != ticker and row["_year"] == args.val_year]
        test_rows = [row for row in rows if row["ticker"] == ticker and row["_year"] > args.val_year]

        if len(test_rows) < args.min_test_events or not train_rows or not val_rows:
            skipped[ticker] = {
                "train": len(train_rows),
                "val": len(val_rows),
                "test": len(test_rows),
            }
            continue

        global_prior = float(np.mean([row["_target"] for row in train_rows]))
        train_prior = constant_prior(len(train_rows), global_prior)
        val_prior = constant_prior(len(val_rows), global_prior)
        test_prior = constant_prior(len(test_rows), global_prior)

        structured_bundle = build_dense_bundle(train_rows, val_rows, test_rows, STRUCTURED_FEATURES)
        train_y = np.asarray([row["_target"] for row in train_rows], dtype=float)
        val_y = np.asarray([row["_target"] for row in val_rows], dtype=float)

        structured_alpha, structured_model, _ = fit_residual_ridge(
            structured_bundle["train"],
            train_prior,
            train_y,
            structured_bundle["val"],
            val_prior,
            val_y,
            alphas,
        )
        structured_test_pred = test_prior + structured_model.predict(structured_bundle["test"])
        structured_qna_test_pred = None
        structured_qna_alpha = None
        if args.with_qna_lsa:
            qna_lsa_bundle = build_text_lsa_bundle(
                train_rows,
                val_rows,
                test_rows,
                text_col="qna_text",
                max_features=8000,
                min_df=2,
                lsa_components=64,
            )
            structured_qna_alpha, structured_qna_model, _ = fit_residual_ridge(
                np.hstack([structured_bundle["train"], qna_lsa_bundle["train"]]),
                train_prior,
                train_y,
                np.hstack([structured_bundle["val"], qna_lsa_bundle["val"]]),
                val_prior,
                val_y,
                alphas,
            )
            structured_qna_test_pred = test_prior + structured_qna_model.predict(
                np.hstack([structured_bundle["test"], qna_lsa_bundle["test"]])
            )

        prior_test_pred = test_prior
        for idx, row in enumerate(test_rows):
            out_row = {
                "event_key": row["event_key"],
                "ticker": ticker,
                "year": row["_year"],
                "regime": row["_regime"],
                "target": row["_target"],
                "prior_only": float(prior_test_pred[idx]),
                "residual_structured_only": float(structured_test_pred[idx]),
            }
            if structured_qna_test_pred is not None:
                out_row["residual_structured_plus_qna_lsa"] = float(structured_qna_test_pred[idx])
            prediction_rows.append(out_row)
            overall_rows.append(row)

    if not prediction_rows:
        raise SystemExit("no eligible held-out tickers for unseen-ticker stress test")

    ordered_prediction_rows = sorted(
        prediction_rows,
        key=lambda item: (item["ticker"], item["year"], item["event_key"]),
    )
    write_csv(output_dir / "unseen_ticker_predictions.csv", ordered_prediction_rows)

    y_true = np.asarray([row["target"] for row in ordered_prediction_rows], dtype=float)
    prediction_lookup = {
        "prior_only": np.asarray([row["prior_only"] for row in ordered_prediction_rows], dtype=float),
        "residual_structured_only": np.asarray(
            [row["residual_structured_only"] for row in ordered_prediction_rows],
            dtype=float,
        ),
    }
    if args.with_qna_lsa:
        prediction_lookup["residual_structured_plus_qna_lsa"] = np.asarray(
            [row["residual_structured_plus_qna_lsa"] for row in ordered_prediction_rows],
            dtype=float,
        )

    ticker_summary = {}
    for ticker in sorted({row["ticker"] for row in ordered_prediction_rows}):
        ticker_rows = [row for row in ordered_prediction_rows if row["ticker"] == ticker]
        ticker_summary[ticker] = {
            model_name: metrics(
                np.asarray([row["target"] for row in ticker_rows], dtype=float),
                np.asarray([row[model_name] for row in ticker_rows], dtype=float),
            )
            | {"n": len(ticker_rows)}
            for model_name in prediction_lookup
        }

    summary = {
        "target_variant": args.target_variant,
        "include_regimes": sorted(include_regimes),
        "exclude_html_flags": sorted(exclude_html_flags),
        "with_qna_lsa": bool(args.with_qna_lsa),
        "candidate_tickers": len(candidate_tickers),
        "evaluated_tickers": len(ticker_summary),
        "skipped_tickers": skipped,
        "overall_test_size": len(ordered_prediction_rows),
        "overall": {
            model_name: metrics(y_true, preds)
            for model_name, preds in prediction_lookup.items()
        },
        "median_ticker_r2": {
            model_name: float(
                np.median([payload[model_name]["r2"] for payload in ticker_summary.values()])
            )
            for model_name in prediction_lookup
        },
        "by_ticker": ticker_summary,
    }
    write_json(output_dir / "unseen_ticker_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
