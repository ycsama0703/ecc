#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dj30_qc_utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose how the question-role transfer family differs from hard abstention: "
            "where it keeps the agreed expert, where it vetoes to pre-only, and whether those vetoes help."
        )
    )
    parser.add_argument(
        "--slice-root",
        type=Path,
        default=Path("results/afterhours_transfer_question_role_slice_diagnostics_lsa4_real"),
    )
    parser.add_argument(
        "--role-benchmark-root",
        type=Path,
        default=Path("results/afterhours_transfer_role_text_signal_benchmark_lsa4_real"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_question_role_gate_diagnostics_lsa4_real"),
    )
    parser.add_argument("--top-count", type=int, default=10)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def mean_bool(values: list[bool]) -> float:
    return float(np.mean(values)) if values else 0.0


def mean_float(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median_float(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def top_rows(rows: list[dict[str, object]], key: str, top_count: int, reverse: bool = True) -> list[dict[str, object]]:
    ordered = sorted(rows, key=lambda row: float(row[key]), reverse=reverse)
    return ordered[:top_count]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_events = {
        row["row_key"]: row
        for row in load_rows(
            args.slice_root / "afterhours_transfer_question_role_slice_diagnostics_events.csv"
        )
    }
    role_predictions = {
        row["row_key"]: row
        for row in load_rows(
            args.role_benchmark_root / "afterhours_transfer_role_text_signal_benchmark_test_predictions.csv"
        )
    }

    event_rows = []
    by_state: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_component: dict[str, list[dict[str, object]]] = defaultdict(list)
    by_ticker: dict[str, list[dict[str, object]]] = defaultdict(list)

    for row_key, event in slice_events.items():
        prediction = role_predictions.get(row_key)
        if prediction is None:
            raise SystemExit(f"missing role benchmark prediction for {row_key}")

        agreement = int(float(prediction["agreement"]))
        use_agreed = int(float(prediction["question_role_lsa__use_agreed"]))
        if agreement == 0:
            state = "disagreement_auto_pre"
        elif use_agreed == 1:
            state = "agreement_keep_agreed"
        else:
            state = "agreement_veto_to_pre"

        event_payload = {
            "row_key": row_key,
            "event_key": event["event_key"],
            "ticker": event["ticker"],
            "year": int(event["year"]),
            "state": state,
            "agreement": agreement,
            "question_role_use_agreed": use_agreed,
            "dominant_component_label": event["dominant_component_label"],
            "dominant_component": int(event["dominant_component"]),
            "dominant_sign": int(event["dominant_sign"]),
            "dominant_abs_score": float(event["dominant_abs_score"]),
            "question_role_mse_gain_vs_pre": float(event["question_role_mse_gain_vs_pre"]),
            "question_role_mse_gain_vs_geometry": float(event["question_role_mse_gain_vs_geometry"]),
            "question_role_mse_gain_vs_hard": float(event["question_role_mse_gain_vs_hard"]),
            "hard_abstention_se": float(event["hard_abstention_se"]),
            "pre_call_market_se": float(event["pre_call_market_se"]),
            "question_role_se": float(event["question_role_se"]),
            "question_text_preview": event["question_text_preview"],
        }
        event_rows.append(event_payload)
        by_state[state].append(event_payload)
        by_component[event_payload["dominant_component_label"]].append(event_payload)
        by_ticker[event_payload["ticker"]].append(event_payload)

    state_rows = []
    for state in ["disagreement_auto_pre", "agreement_keep_agreed", "agreement_veto_to_pre"]:
        rows = by_state.get(state, [])
        state_rows.append(
            {
                "state": state,
                "event_count": len(rows),
                "share": float(len(rows) / len(event_rows)) if event_rows else 0.0,
                "mean_question_role_mse_gain_vs_pre": mean_float(
                    [float(row["question_role_mse_gain_vs_pre"]) for row in rows]
                ),
                "mean_question_role_mse_gain_vs_hard": mean_float(
                    [float(row["question_role_mse_gain_vs_hard"]) for row in rows]
                ),
                "win_share_vs_pre": mean_bool(
                    [float(row["question_role_mse_gain_vs_pre"]) > 0 for row in rows]
                ),
                "win_share_vs_hard": mean_bool(
                    [float(row["question_role_mse_gain_vs_hard"]) > 0 for row in rows]
                ),
                "mean_hard_abstention_se": mean_float([float(row["hard_abstention_se"]) for row in rows]),
                "median_hard_abstention_se": median_float([float(row["hard_abstention_se"]) for row in rows]),
            }
        )

    component_rows = []
    for label, rows in sorted(
        by_component.items(),
        key=lambda item: (
            -sum(1 for row in item[1] if row["state"] == "agreement_veto_to_pre"),
            item[0],
        ),
    ):
        component_rows.append(
            {
                "dominant_component_label": label,
                "event_count": len(rows),
                "agreement_event_count": int(sum(1 for row in rows if row["agreement"] == 1)),
                "veto_event_count": int(sum(1 for row in rows if row["state"] == "agreement_veto_to_pre")),
                "veto_share_within_agreement": (
                    float(
                        sum(1 for row in rows if row["state"] == "agreement_veto_to_pre")
                        / max(1, sum(1 for row in rows if row["agreement"] == 1))
                    )
                ),
                "mean_gain_vs_hard": mean_float([float(row["question_role_mse_gain_vs_hard"]) for row in rows]),
                "win_share_vs_hard": mean_bool(
                    [float(row["question_role_mse_gain_vs_hard"]) > 0 for row in rows]
                ),
                "mean_hard_abstention_se": mean_float([float(row["hard_abstention_se"]) for row in rows]),
            }
        )

    ticker_rows = []
    for ticker, rows in sorted(
        by_ticker.items(),
        key=lambda item: (
            -sum(1 for row in item[1] if row["state"] == "agreement_veto_to_pre"),
            item[0],
        ),
    ):
        ticker_rows.append(
            {
                "ticker": ticker,
                "event_count": len(rows),
                "agreement_event_count": int(sum(1 for row in rows if row["agreement"] == 1)),
                "veto_event_count": int(sum(1 for row in rows if row["state"] == "agreement_veto_to_pre")),
                "veto_share_within_agreement": (
                    float(
                        sum(1 for row in rows if row["state"] == "agreement_veto_to_pre")
                        / max(1, sum(1 for row in rows if row["agreement"] == 1))
                    )
                ),
                "mean_gain_vs_hard": mean_float([float(row["question_role_mse_gain_vs_hard"]) for row in rows]),
                "mean_gain_vs_pre": mean_float([float(row["question_role_mse_gain_vs_pre"]) for row in rows]),
                "win_share_vs_hard": mean_bool(
                    [float(row["question_role_mse_gain_vs_hard"]) > 0 for row in rows]
                ),
            }
        )

    agreement_veto_rows = by_state["agreement_veto_to_pre"]
    summary = {
        "counts": {
            "total_events": len(event_rows),
            "agreement_events": len(by_state["agreement_keep_agreed"]) + len(by_state["agreement_veto_to_pre"]),
            "disagreement_events": len(by_state["disagreement_auto_pre"]),
            "keep_agreed_events": len(by_state["agreement_keep_agreed"]),
            "veto_to_pre_events": len(by_state["agreement_veto_to_pre"]),
        },
        "states": state_rows,
        "component_summaries": component_rows,
        "ticker_summaries": ticker_rows,
        "top_veto_improvements_vs_hard": top_rows(
            agreement_veto_rows,
            "question_role_mse_gain_vs_hard",
            args.top_count,
            reverse=True,
        ),
        "top_veto_failures_vs_hard": top_rows(
            agreement_veto_rows,
            "question_role_mse_gain_vs_hard",
            args.top_count,
            reverse=False,
        ),
    }

    write_json(output_dir / "afterhours_transfer_question_role_gate_diagnostics_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_question_role_gate_diagnostics_states.csv", state_rows)
    write_csv(output_dir / "afterhours_transfer_question_role_gate_diagnostics_components.csv", component_rows)
    write_csv(output_dir / "afterhours_transfer_question_role_gate_diagnostics_tickers.csv", ticker_rows)
    write_csv(output_dir / "afterhours_transfer_question_role_gate_diagnostics_events.csv", event_rows)


if __name__ == "__main__":
    main()
