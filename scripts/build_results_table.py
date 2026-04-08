#!/usr/bin/env python3
"""
Build a unified experiment results table that goes beyond R^2.

Outputs:
    - comprehensive_results_<split_version>_<main_run_id>.csv
    - comprehensive_results_<split_version>_<main_run_id>.md
    - comprehensive_results_<split_version>_<main_run_id>.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
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


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def safe_float(value) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def round_or_none(value: float | None, digits: int = 10) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return round(float(value), digits)


def format_value(value: float | None, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    value = float(value)
    if value == 0:
        return "0"
    if abs(value) >= 1e-3:
        return f"{value:.{digits}f}"
    return f"{value:.3e}"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def compute_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    n = len(y_true)
    if n == 0:
        return {
            "mse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "spearman": float("nan"),
            "spearman_p": float("nan"),
            "n": 0,
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
        "mse": float(mse),
        "mae": float(mae),
        "r2": float(r2),
        "spearman": spearman,
        "spearman_p": spearman_p,
        "n": n,
    }


def build_markdown(rows: list[dict], metadata: dict) -> str:
    lines = []
    lines.append("# Comprehensive Results")
    lines.append("")
    lines.append(f"- split_version: `{metadata['split_version']}`")
    lines.append(f"- main_run_id: `{metadata['main_run_id']}`")
    lines.append(f"- benchmark_run_id: `{metadata['bench_run_id']}`")
    lines.append(f"- prior_reference_model: `{metadata['prior_reference_model']}`")
    lines.append("")
    lines.append(
        "| Group | Model | Full-set R² | nMAE | Rel. MAE Improve | Coverage | Accepted-set R² | Accepted-set MSE | Gain over prior (ΔMSE) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {group} | {model} | {r2} | {nmae} | {rel_mae} | {coverage} | {accepted_r2} | {accepted_mse} | {gain} |".format(
                group=row["group"],
                model=row["model"],
                r2=format_value(safe_float(row["full_set_r2"]), 4),
                nmae=format_value(safe_float(row["normalized_mae"]), 4),
                rel_mae=format_value(safe_float(row["relative_mae_improvement"]), 4),
                coverage=format_value(safe_float(row["coverage"]), 4),
                accepted_r2=format_value(safe_float(row["accepted_set_r2"]), 4),
                accepted_mse=format_value(safe_float(row["accepted_set_mse"]), 6),
                gain=format_value(safe_float(row["gain_over_prior_mse"]), 6),
            )
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- `nMAE = MAE / std(y_test)` in the shared test split.")
    lines.append("- `Rel. MAE Improve = (MAE_prior - MAE_model) / MAE_prior`; positive is better.")
    lines.append("- For non-selective baselines, `Coverage=1.0` and accepted-set metrics collapse to full-set metrics.")
    lines.append("- `Gain over prior (ΔMSE)` is `prior_mse - model_mse`, so positive values are better.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a unified experiment results table.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--split-version", type=str, required=True)
    parser.add_argument("--main-run-id", type=str, required=True)
    parser.add_argument("--bench-run-id", type=str, required=True)
    parser.add_argument(
        "--prior-reference-model",
        type=str,
        default="market_plus_controls",
        help="Benchmark row used as the prior reference for gain-over-prior.",
    )
    parser.add_argument(
        "--ours-label",
        type=str,
        default="prec_selective",
        help="Display name for the main experiment row.",
    )
    args = parser.parse_args()

    experiment_dir = args.experiment_dir.resolve()
    benchmark_dir = args.benchmark_dir.resolve()
    metrics_dir = experiment_dir / "metrics"
    pred_dir = experiment_dir / "predictions"
    benchmark_metrics_dir = benchmark_dir / "metrics"

    main_metrics_path = metrics_dir / f"main_metrics_{args.split_version}_{args.main_run_id}.json"
    benchmark_results_path = benchmark_metrics_dir / f"benchmark_results_{args.split_version}_{args.bench_run_id}.csv"
    prior_pred_path = pred_dir / f"market_prior_test_{args.split_version}_{args.main_run_id}.csv"

    main_metrics = load_json(main_metrics_path)
    benchmark_rows = read_csv(benchmark_results_path)
    prior_test_rows = read_csv(prior_pred_path)
    final_test_path = pred_dir / f"final_main_test_{args.split_version}_{args.main_run_id}.csv"
    final_test_rows = read_csv(final_test_path)
    family_metrics_path = metrics_dir / f"prior_correction_family_{args.split_version}_{args.main_run_id}.csv"

    benchmark_test_rows = [row for row in benchmark_rows if row.get("split") == "test"]
    benchmark_test_by_model = {row["model"]: row for row in benchmark_test_rows}

    reference_row = benchmark_test_by_model.get(args.prior_reference_model)
    if reference_row is None:
        raise FileNotFoundError(
            f"Could not find benchmark test row for prior reference model '{args.prior_reference_model}'."
        )
    reference_mse = float(reference_row["mse"])
    reference_mae = float(reference_row["mae"])

    prior_true = [float(row["shock_minus_pre"]) for row in prior_test_rows]
    prior_pred = [float(row["mu_hat"]) for row in prior_test_rows]
    prior_mse_from_predictions = sum((yt - yp) ** 2 for yt, yp in zip(prior_true, prior_pred)) / max(len(prior_true), 1)
    y_test_std = float(np.std(prior_true, ddof=0)) if prior_true else float("nan")

    group_by_model = {
        "market_only": "market_baseline",
        "market_plus_controls": "market_baseline",
        "market_precall_only": "market_baseline",
        "market_precall_plus_controls": "market_baseline",
        "market_precall_controls_no_surprise": "market_baseline",
        "xgboost_market_controls": "machine_learning",
        "lightgbm_market_controls": "machine_learning",
        "random_forest_market_controls": "machine_learning",
        "xgboost_precall_controls": "machine_learning",
        "lightgbm_precall_controls": "machine_learning",
        "random_forest_precall_controls": "machine_learning",
        "xgboost_precall_controls_no_surprise": "machine_learning",
        "lightgbm_precall_controls_no_surprise": "machine_learning",
        "random_forest_precall_controls_no_surprise": "machine_learning",
        "tfidf_elasticnet": "ecc_baseline",
        "compact_qa_baseline": "ecc_baseline",
        "finbert_pooled": "ecc_baseline",
        "prior_only": "prior_correction",
        "prior_plus_z_no_gate": "prior_correction",
        "prior_plus_alpha_z_gate_only": "prior_correction",
        args.ours_label: "prior_correction",
    }

    rows = []
    for row in sorted(benchmark_test_rows, key=lambda item: item["model"]):
        mse = float(row["mse"])
        mae = float(row["mae"])
        rows.append(
            {
                "group": group_by_model.get(row["model"], "other"),
                "model": row["model"],
                "family": "benchmark",
                "full_set_r2": round_or_none(float(row["r2"]), 8),
                "full_set_mse": round_or_none(mse),
                "full_set_mae": round_or_none(mae),
                "normalized_mae": round_or_none(mae / y_test_std if y_test_std > 0 else float("nan"), 8),
                "relative_mae_improvement": round_or_none(
                    (reference_mae - mae) / reference_mae if reference_mae > 0 else float("nan"),
                    8,
                ),
                "spearman": round_or_none(safe_float(row.get("spearman")), 8),
                "spearman_p": round_or_none(safe_float(row.get("spearman_p")), 8),
                "coverage": 1.0,
                "accepted_set_r2": round_or_none(float(row["r2"]), 8),
                "accepted_set_mse": round_or_none(mse),
                "aurc": None,
                "gain_over_prior_mse": round_or_none(reference_mse - mse),
                "n_test": int(float(row["n"])),
            }
        )

    if family_metrics_path.exists():
        family_rows = [row for row in read_csv(family_metrics_path) if row.get("split") == "test"]
        for row in family_rows:
            rows.append(
                {
                    "group": group_by_model.get(row["model"], "other"),
                    "model": row["model"],
                    "family": "prior_correction_family",
                    "full_set_r2": round_or_none(float(row["r2"]), 8),
                    "full_set_mse": round_or_none(float(row["mse"])),
                    "full_set_mae": round_or_none(float(row["mae"])),
                    "normalized_mae": round_or_none(
                        float(row["mae"]) / y_test_std if y_test_std > 0 else float("nan"),
                        8,
                    ),
                    "relative_mae_improvement": round_or_none(
                        (reference_mae - float(row["mae"])) / reference_mae if reference_mae > 0 else float("nan"),
                        8,
                    ),
                    "spearman": round_or_none(safe_float(row.get("spearman")), 8),
                    "spearman_p": round_or_none(safe_float(row.get("spearman_p")), 8),
                    "coverage": round_or_none(float(row["coverage"]), 8),
                    "accepted_set_r2": round_or_none(safe_float(row.get("accepted_r2")), 8),
                    "accepted_set_mse": round_or_none(safe_float(row.get("accepted_mse"))),
                    "aurc": round_or_none(safe_float(row.get("aurc"))),
                    "gain_over_prior_mse": round_or_none(float(row["gain_over_prior_mse"])),
                    "n_test": int(float(row["n"])),
                }
            )
    else:
        prior_y = [float(row["shock_minus_pre"]) for row in final_test_rows]
        prior_pred = [float(row["mu_hat"]) for row in final_test_rows]
        plus_z_pred = [
            float(row["y_hat_plus_z"]) if "y_hat_plus_z" in row else float(row["mu_hat"]) + float(row["z_hat"])
            for row in final_test_rows
        ]
        plus_alpha_pred = [
            float(row["y_hat_plus_alpha_z"])
            if "y_hat_plus_alpha_z" in row
            else float(row["mu_hat"]) + float(row["alpha"]) * float(row["z_hat"])
            for row in final_test_rows
        ]
        selective_pred = [float(row["y_hat"]) for row in final_test_rows]
        accept = [int(float(row["accept"])) for row in final_test_rows]

        family_specs = [
            ("prior_only", prior_pred, [1] * len(prior_y), None),
            ("prior_plus_z_no_gate", plus_z_pred, [1] * len(prior_y), None),
            ("prior_plus_alpha_z_gate_only", plus_alpha_pred, [1] * len(prior_y), None),
            (args.ours_label, selective_pred, accept, main_metrics.get("test_aurc")),
        ]
        prior_metrics = compute_metrics(prior_y, prior_pred)
        for name, pred, accept_flags, aurc in family_specs:
            full_metrics = compute_metrics(prior_y, pred)
            accepted_true = [y for y, a in zip(prior_y, accept_flags) if a]
            accepted_pred = [p for p, a in zip(pred, accept_flags) if a]
            accepted_metrics = compute_metrics(accepted_true, accepted_pred)
            rows.append(
                {
                    "group": group_by_model.get(name, "other"),
                    "model": name,
                    "family": "prior_correction_family",
                    "full_set_r2": round_or_none(full_metrics["r2"], 8),
                    "full_set_mse": round_or_none(full_metrics["mse"]),
                    "full_set_mae": round_or_none(full_metrics["mae"]),
                    "normalized_mae": round_or_none(
                        full_metrics["mae"] / y_test_std if y_test_std > 0 else float("nan"),
                        8,
                    ),
                    "relative_mae_improvement": round_or_none(
                        (reference_mae - full_metrics["mae"]) / reference_mae if reference_mae > 0 else float("nan"),
                        8,
                    ),
                    "spearman": round_or_none(full_metrics["spearman"], 8),
                    "spearman_p": round_or_none(full_metrics["spearman_p"], 8),
                    "coverage": round_or_none(sum(accept_flags) / max(len(accept_flags), 1), 8),
                    "accepted_set_r2": round_or_none(accepted_metrics["r2"], 8),
                    "accepted_set_mse": round_or_none(accepted_metrics["mse"]),
                    "aurc": round_or_none(safe_float(aurc)),
                    "gain_over_prior_mse": round_or_none(prior_metrics["mse"] - full_metrics["mse"]),
                    "n_test": full_metrics["n"],
                }
            )

    ordering = {
        "market_only": 10,
        "market_plus_controls": 20,
        "market_precall_only": 15,
        "market_precall_plus_controls": 25,
        "market_precall_controls_no_surprise": 27,
        "xgboost_market_controls": 30,
        "xgboost_precall_controls": 35,
        "xgboost_precall_controls_no_surprise": 37,
        "lightgbm_market_controls": 40,
        "lightgbm_precall_controls": 45,
        "lightgbm_precall_controls_no_surprise": 47,
        "random_forest_market_controls": 50,
        "random_forest_precall_controls": 55,
        "random_forest_precall_controls_no_surprise": 57,
        "tfidf_elasticnet": 60,
        "compact_qa_baseline": 70,
        "finbert_pooled": 80,
        "prior_only": 90,
        "prior_plus_z_no_gate": 100,
        "prior_plus_alpha_z_gate_only": 110,
        args.ours_label: 120,
    }
    rows.sort(key=lambda item: (ordering.get(item["model"], 999), item["model"]))

    summary = {
        "split_version": args.split_version,
        "main_run_id": args.main_run_id,
        "bench_run_id": args.bench_run_id,
        "experiment_dir": str(experiment_dir),
        "benchmark_dir": str(benchmark_dir),
        "prior_reference_model": args.prior_reference_model,
        "prior_reference_mse": reference_mse,
        "prior_reference_mae": reference_mae,
        "prior_test_mse_from_predictions": prior_mse_from_predictions,
        "y_test_std": y_test_std,
        "rows": rows,
    }

    csv_path = metrics_dir / f"comprehensive_results_{args.split_version}_{args.main_run_id}.csv"
    md_path = metrics_dir / f"comprehensive_results_{args.split_version}_{args.main_run_id}.md"
    json_path = metrics_dir / f"comprehensive_results_{args.split_version}_{args.main_run_id}.json"

    write_csv(csv_path, rows)
    md_path.write_text(build_markdown(rows, summary), encoding="utf-8")
    write_json(json_path, summary)

    print("Comprehensive results build complete")
    print(f"  output_csv: {csv_path}")
    print(f"  output_md: {md_path}")
    print(f"  output_json: {json_path}")


if __name__ == "__main__":
    main()
