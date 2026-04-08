#!/usr/bin/env python3
"""
Build per-ticker test metrics for a finished experiment run.

Outputs:
    - per_ticker_test_metrics_<split_version>_<run_id>.csv
    - per_ticker_test_metrics_<split_version>_<run_id>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy import stats as scipy_stats


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str | None) -> float:
    return float(value) if value not in (None, "") else float("nan")


def regression_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    n = len(y_true)
    if n == 0:
        return {
            "n": 0,
            "mse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
        }

    residuals = [yt - yp for yt, yp in zip(y_true, y_pred)]
    mse = sum(r * r for r in residuals) / n
    mae = sum(abs(r) for r in residuals) / n
    y_mean = sum(y_true) / n
    ss_res = sum(r * r for r in residuals)
    ss_tot = sum((y - y_mean) ** 2 for y in y_true)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if n >= 2:
        spearman, spearman_p = scipy_stats.spearmanr(y_true, y_pred)
        spearman = float(spearman)
        spearman_p = float(spearman_p)
    else:
        spearman = float("nan")
        spearman_p = float("nan")

    return {
        "n": n,
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "spearman": spearman,
        "spearman_p": spearman_p,
    }


def selective_metrics(y_true: list[float], y_pred: list[float], accept: list[int]) -> dict:
    n_total = len(y_true)
    n_accepted = sum(accept)
    coverage = n_accepted / n_total if n_total else float("nan")

    accepted_true = [y for y, a in zip(y_true, accept) if a]
    accepted_pred = [p for p, a in zip(y_pred, accept) if a]
    accepted_summary = regression_metrics(accepted_true, accepted_pred)

    return {
        "coverage": coverage,
        "n_accepted": n_accepted,
        "accepted_mse": accepted_summary["mse"],
        "accepted_mae": accepted_summary["mae"],
        "accepted_r2": accepted_summary["r2"],
        "accepted_spearman": accepted_summary["spearman"],
    }


def round_or_nan(value: float, digits: int = 10) -> float | None:
    if value is None or math.isnan(value):
        return None
    return round(value, digits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-ticker metrics for a finished experiment.")
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("data/processed/panel/processed_panel.csv"),
        help="Processed panel CSV used for the run.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Experiment output directory, e.g. outputs/main_experiment_cov30.",
    )
    parser.add_argument(
        "--split-version",
        type=str,
        default="v1",
        help="Split version tag used in filenames.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="Run identifier used in filenames.",
    )
    args = parser.parse_args()

    panel_rows = read_csv(args.panel_csv.resolve())
    event_to_ticker = {row["event_id"]: row["ticker"] for row in panel_rows}

    pred_dir = args.experiment_dir.resolve() / "predictions"
    metrics_dir = args.experiment_dir.resolve() / "metrics"
    final_path = pred_dir / f"final_main_test_{args.split_version}_{args.run_id}.csv"
    prior_path = pred_dir / f"market_prior_test_{args.split_version}_{args.run_id}.csv"

    final_rows = read_csv(final_path)
    prior_rows = read_csv(prior_path)
    prior_by_event = {row["event_id"]: row for row in prior_rows}

    grouped = defaultdict(list)
    for row in final_rows:
        event_id = row["event_id"]
        ticker = event_to_ticker.get(event_id)
        if not ticker:
            continue
        prior_row = prior_by_event.get(event_id)
        if prior_row is None:
            continue
        grouped[ticker].append(
            {
                "event_id": event_id,
                "y_true": safe_float(row["shock_minus_pre"]),
                "final_pred": safe_float(row["y_hat"]),
                "prior_pred": safe_float(prior_row["mu_hat"]),
                "accept": int(float(row["accept"])),
                "risk": safe_float(row["risk"]),
                "alpha": safe_float(row["alpha"]),
            }
        )

    output_rows = []
    for ticker in sorted(grouped):
        rows = grouped[ticker]
        y_true = [row["y_true"] for row in rows]
        final_pred = [row["final_pred"] for row in rows]
        prior_pred = [row["prior_pred"] for row in rows]
        accept = [row["accept"] for row in rows]
        risk = [row["risk"] for row in rows]
        alpha = [row["alpha"] for row in rows]

        final_metrics = regression_metrics(y_true, final_pred)
        prior_metrics = regression_metrics(y_true, prior_pred)
        sel_metrics = selective_metrics(y_true, final_pred, accept)

        output_rows.append(
            {
                "ticker": ticker,
                "n_test": final_metrics["n"],
                "final_mse": round_or_nan(final_metrics["mse"]),
                "final_mae": round_or_nan(final_metrics["mae"]),
                "final_r2": round_or_nan(final_metrics["r2"]),
                "final_spearman": round_or_nan(final_metrics["spearman"], 8),
                "prior_mse": round_or_nan(prior_metrics["mse"]),
                "prior_mae": round_or_nan(prior_metrics["mae"]),
                "prior_r2": round_or_nan(prior_metrics["r2"]),
                "prior_spearman": round_or_nan(prior_metrics["spearman"], 8),
                "mse_gain_vs_prior": round_or_nan(prior_metrics["mse"] - final_metrics["mse"]),
                "r2_gain_vs_prior": round_or_nan(final_metrics["r2"] - prior_metrics["r2"]),
                "coverage": round_or_nan(sel_metrics["coverage"], 8),
                "n_accepted": sel_metrics["n_accepted"],
                "accepted_mse": round_or_nan(sel_metrics["accepted_mse"]),
                "accepted_mae": round_or_nan(sel_metrics["accepted_mae"]),
                "accepted_r2": round_or_nan(sel_metrics["accepted_r2"]),
                "accepted_spearman": round_or_nan(sel_metrics["accepted_spearman"], 8),
                "mean_alpha": round_or_nan(sum(alpha) / len(alpha), 8),
                "mean_risk": round_or_nan(sum(risk) / len(risk)),
            }
        )

    csv_path = metrics_dir / f"per_ticker_test_metrics_{args.split_version}_{args.run_id}.csv"
    json_path = metrics_dir / f"per_ticker_test_metrics_{args.split_version}_{args.run_id}.json"
    write_csv(csv_path, output_rows)

    summary = {
        "run_id": args.run_id,
        "split_version": args.split_version,
        "experiment_dir": str(args.experiment_dir.resolve()),
        "panel_csv": str(args.panel_csv.resolve()),
        "ticker_count": len(output_rows),
        "tickers": [row["ticker"] for row in output_rows],
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Per-ticker metrics build complete")
    print(f"  tickers: {len(output_rows)}")
    print(f"  output: {csv_path}")


if __name__ == "__main__":
    main()
