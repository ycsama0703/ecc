#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

import numpy as np

from dj30_qc_utils import write_csv, write_json


csv.field_size_limit(1024 * 1024 * 128)


FEATURE_SPECS = [
    ("panel", "a4_strict_row_share"),
    ("panel", "a4_median_match_score"),
    ("features", "a4_strict_high_conf_share"),
    ("features", "qa_pair_count"),
    ("features", "qa_pair_low_overlap_share"),
    ("features", "qa_pair_answer_forward_rate_mean"),
    ("features", "qa_pair_answer_assertive_rate_mean"),
    ("features", "qa_pair_answer_hedge_rate_mean"),
    ("features", "qa_multi_part_question_share"),
    ("features", "qa_evasive_proxy_share"),
    ("qa", "qa_bench_direct_answer_share"),
    ("qa", "qa_bench_direct_early_score_mean"),
    ("qa", "qa_bench_coverage_mean"),
    ("qa", "qa_bench_evasion_score_mean"),
    ("qa", "qa_bench_high_evasion_share"),
    ("qa", "qa_bench_nonresponse_share"),
    ("panel", "analyst_eps_norm_num_est"),
    ("panel", "analyst_eps_norm_std"),
    ("panel", "analyst_revenue_num_est"),
    ("panel", "analyst_revenue_std"),
    ("panel", "revenue_surprise_pct"),
    ("panel", "eps_gaap_surprise_pct"),
    ("panel", "ebitda_surprise_pct"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose the remaining latest-window miss cases of the current hard-abstention "
            "transfer shell against existing observability / QA / analyst signals."
        )
    )
    parser.add_argument(
        "--gate-root",
        type=Path,
        default=Path("results/afterhours_transfer_question_role_gate_diagnostics_lsa4_real"),
    )
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("results/audio_sentence_aligned_afterhours_clean_real/panel_subset_afterhours_clean.csv"),
    )
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("results/features_real/event_text_audio_features.csv"),
    )
    parser.add_argument(
        "--qa-csv",
        type=Path,
        default=Path("results/qa_benchmark_features_v2_real/qa_benchmark_features.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/afterhours_transfer_hard_abstention_miss_diagnostics_lsa4_real"),
    )
    parser.add_argument("--top-quantile", type=float, default=0.75)
    parser.add_argument("--top-count", type=int, default=12)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def truncate_text(text: str, limit: int = 220) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def median(values: list[float]) -> float:
    return float(np.median(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gate_events = load_rows((args.gate_root / "afterhours_transfer_question_role_gate_diagnostics_events.csv").resolve())
    event_keys = {row["event_key"] for row in gate_events}

    panel_lookup = {row["event_key"]: row for row in load_rows(args.panel_csv.resolve()) if row.get("event_key") in event_keys}
    feature_lookup = {
        row["event_key"]: row for row in load_rows(args.features_csv.resolve()) if row.get("event_key") in event_keys
    }
    qa_lookup = {row["event_key"]: row for row in load_rows(args.qa_csv.resolve()) if row.get("event_key") in event_keys}

    hard_ses = np.asarray([float(row["hard_abstention_se"]) for row in gate_events], dtype=float)
    threshold = float(np.quantile(hard_ses, args.top_quantile))

    enriched_rows = []
    for row in gate_events:
        event_key = row["event_key"]
        panel_row = panel_lookup.get(event_key, {})
        feature_row = feature_lookup.get(event_key, {})
        qa_row = qa_lookup.get(event_key, {})
        hard_se = float(row["hard_abstention_se"])
        target = {
            "row_key": row["row_key"],
            "event_key": event_key,
            "ticker": row["ticker"],
            "year": int(row["year"]),
            "state": row["state"],
            "dominant_component_label": row["dominant_component_label"],
            "hard_abstention_se": hard_se,
            "hard_abstention_abs_error": math.sqrt(hard_se),
            "pre_call_market_se": float(row["pre_call_market_se"]),
            "question_role_se": float(row["question_role_se"]),
            "question_role_mse_gain_vs_hard": float(row["question_role_mse_gain_vs_hard"]),
            "question_role_mse_gain_vs_pre": float(row["question_role_mse_gain_vs_pre"]),
            "top_miss": int(hard_se >= threshold),
            "question_text_preview": truncate_text(row.get("question_text_preview", "")),
        }
        for source, col in FEATURE_SPECS:
            lookup = {"panel": panel_row, "features": feature_row, "qa": qa_row}[source]
            target[col] = safe_float(lookup.get(col))
        target["abs_revenue_surprise_pct"] = abs(target["revenue_surprise_pct"]) if target["revenue_surprise_pct"] is not None else None
        target["abs_eps_gaap_surprise_pct"] = (
            abs(target["eps_gaap_surprise_pct"]) if target["eps_gaap_surprise_pct"] is not None else None
        )
        target["abs_ebitda_surprise_pct"] = (
            abs(target["ebitda_surprise_pct"]) if target["ebitda_surprise_pct"] is not None else None
        )
        enriched_rows.append(target)

    top_rows = [row for row in enriched_rows if row["top_miss"] == 1]
    rest_rows = [row for row in enriched_rows if row["top_miss"] == 0]

    feature_rows = []
    for _, col in FEATURE_SPECS + [
        ("derived", "abs_revenue_surprise_pct"),
        ("derived", "abs_eps_gaap_surprise_pct"),
        ("derived", "abs_ebitda_surprise_pct"),
    ]:
        top_values = [float(row[col]) for row in top_rows if row.get(col) is not None]
        rest_values = [float(row[col]) for row in rest_rows if row.get(col) is not None]
        if len(top_values) < 3 or len(rest_values) < 3:
            continue
        full = np.asarray(top_values + rest_values, dtype=float)
        std = float(full.std())
        if std == 0.0:
            std = 1.0
        effect = float((np.mean(top_values) - np.mean(rest_values)) / std)
        feature_rows.append(
            {
                "feature": col,
                "top_miss_mean": mean(top_values),
                "rest_mean": mean(rest_values),
                "top_miss_median": median(top_values),
                "rest_median": median(rest_values),
                "effect_size": effect,
                "abs_effect_size": abs(effect),
                "top_miss_count": len(top_values),
                "rest_count": len(rest_values),
                "direction": "higher_in_top_miss" if effect > 0 else "lower_in_top_miss",
            }
        )
    feature_rows.sort(key=lambda row: float(row["abs_effect_size"]), reverse=True)

    state_counter_top = Counter(row["state"] for row in top_rows)
    state_counter_rest = Counter(row["state"] for row in rest_rows)
    state_rows = []
    for state in sorted(set(state_counter_top) | set(state_counter_rest)):
        top_count = state_counter_top[state]
        rest_count = state_counter_rest[state]
        total_state = top_count + rest_count
        state_rows.append(
            {
                "state": state,
                "top_miss_count": top_count,
                "rest_count": rest_count,
                "top_miss_share_within_state": float(top_count / total_state) if total_state else 0.0,
                "state_share_within_top_miss": float(top_count / max(1, len(top_rows))),
                "state_share_within_rest": float(rest_count / max(1, len(rest_rows))),
            }
        )

    dominant_counter_top = Counter(row["dominant_component_label"] for row in top_rows)
    dominant_counter_rest = Counter(row["dominant_component_label"] for row in rest_rows)
    dominant_rows = []
    for label in sorted(set(dominant_counter_top) | set(dominant_counter_rest)):
        top_count = dominant_counter_top[label]
        rest_count = dominant_counter_rest[label]
        total_count = top_count + rest_count
        dominant_rows.append(
            {
                "dominant_component_label": label,
                "top_miss_count": top_count,
                "rest_count": rest_count,
                "top_miss_share_within_component": float(top_count / total_count) if total_count else 0.0,
            }
        )
    dominant_rows.sort(key=lambda row: (-int(row["top_miss_count"]), row["dominant_component_label"]))

    ticker_counter_top = Counter(row["ticker"] for row in top_rows)
    ticker_counter_rest = Counter(row["ticker"] for row in rest_rows)
    ticker_rows = []
    for ticker in sorted(set(ticker_counter_top) | set(ticker_counter_rest)):
        top_count = ticker_counter_top[ticker]
        rest_count = ticker_counter_rest[ticker]
        total_count = top_count + rest_count
        ticker_rows.append(
            {
                "ticker": ticker,
                "top_miss_count": top_count,
                "rest_count": rest_count,
                "top_miss_share_within_ticker": float(top_count / total_count) if total_count else 0.0,
            }
        )
    ticker_rows.sort(key=lambda row: (-int(row["top_miss_count"]), row["ticker"]))

    top_miss_events = sorted(top_rows, key=lambda row: float(row["hard_abstention_se"]), reverse=True)[: args.top_count]

    summary = {
        "config": {"top_quantile": float(args.top_quantile), "top_miss_threshold_hard_se": threshold},
        "counts": {"total_events": len(enriched_rows), "top_miss_events": len(top_rows), "rest_events": len(rest_rows)},
        "overall": {
            "top_miss_mean_hard_abstention_se": mean([float(row["hard_abstention_se"]) for row in top_rows]),
            "rest_mean_hard_abstention_se": mean([float(row["hard_abstention_se"]) for row in rest_rows]),
            "top_miss_mean_abs_error": mean([float(row["hard_abstention_abs_error"]) for row in top_rows]),
            "rest_mean_abs_error": mean([float(row["hard_abstention_abs_error"]) for row in rest_rows]),
        },
        "top_feature_shifts": feature_rows[:15],
        "state_summaries": state_rows,
        "dominant_component_summaries": dominant_rows[:12],
        "ticker_summaries": ticker_rows[:12],
        "top_miss_events": top_miss_events,
    }

    write_json(output_dir / "afterhours_transfer_hard_abstention_miss_diagnostics_summary.json", summary)
    write_csv(output_dir / "afterhours_transfer_hard_abstention_miss_feature_shifts.csv", feature_rows)
    write_csv(output_dir / "afterhours_transfer_hard_abstention_miss_states.csv", state_rows)
    write_csv(output_dir / "afterhours_transfer_hard_abstention_miss_components.csv", dominant_rows)
    write_csv(output_dir / "afterhours_transfer_hard_abstention_miss_tickers.csv", ticker_rows)
    write_csv(output_dir / "afterhours_transfer_hard_abstention_miss_events.csv", top_miss_events)


if __name__ == "__main__":
    main()
