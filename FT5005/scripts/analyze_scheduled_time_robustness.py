#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from build_intraday_targets import aggregate_window, compute_returns
from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute event windows under scheduled-time offsets to measure target sensitivity."
    )
    parser.add_argument("--targets-csv", type=Path, required=True)
    parser.add_argument("--d-dir", type=Path, required=True)
    parser.add_argument(
        "--offset-minutes",
        default="-5,-2,0,2,5",
        help="Comma-separated list of scheduled-time offsets in minutes.",
    )
    parser.add_argument("--pre-call-minutes", type=int, default=60)
    parser.add_argument("--post-call-minutes", type=int, default=60)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/robustness_real")
    )
    return parser.parse_args()


def parse_target_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in load_csv_rows(path.resolve()):
        if not row.get("event_key") or not row.get("ticker"):
            continue
        rows.append(row)
    return rows


def load_ticker_bars(d_path: Path) -> dict[dt.date, list[dict]]:
    rows_by_date = defaultdict(list)
    with d_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = dt.datetime.fromisoformat(row["DateTime"])
            rows_by_date[timestamp.date()].append({**row, "_timestamp": timestamp})

    return {
        event_date: compute_returns(sorted(items, key=lambda row: row["_timestamp"]))
        for event_date, items in rows_by_date.items()
    }


def safe_corr(values_a: list[float], values_b: list[float]) -> float:
    if len(values_a) < 2 or len(values_b) < 2:
        return 0.0
    mean_a = statistics.mean(values_a)
    mean_b = statistics.mean(values_b)
    centered_a = [value - mean_a for value in values_a]
    centered_b = [value - mean_b for value in values_b]
    denom_a = math.sqrt(sum(value * value for value in centered_a))
    denom_b = math.sqrt(sum(value * value for value in centered_b))
    if denom_a == 0 or denom_b == 0:
        return 0.0
    return sum(a * b for a, b in zip(centered_a, centered_b)) / (denom_a * denom_b)


def summarize_against_base(rows: list[dict], metric_name: str) -> dict[str, dict]:
    base_lookup = {
        row["event_key"]: safe_float(row.get(metric_name))
        for row in rows
        if int(row["offset_minutes"]) == 0
    }
    base_bar_lookup = {
        row["event_key"]: row.get(metric_name.replace("_rv", "_bar_count"), "")
        for row in rows
        if int(row["offset_minutes"]) == 0
    }

    summary = {}
    offsets = sorted({int(row["offset_minutes"]) for row in rows})
    for offset in offsets:
        offset_rows = [row for row in rows if int(row["offset_minutes"]) == offset]
        paired = []
        unchanged_bars = 0
        compared = 0
        for row in offset_rows:
            event_key = row["event_key"]
            base_value = base_lookup.get(event_key)
            current_value = safe_float(row.get(metric_name))
            if base_value is None or current_value is None:
                continue
            paired.append((base_value, current_value))
            compared += 1
            if row.get(metric_name.replace("_rv", "_bar_count"), "") == base_bar_lookup.get(event_key, ""):
                unchanged_bars += 1

        if not paired:
            summary[str(offset)] = {
                "paired_events": 0,
                "corr_vs_base": 0.0,
                "mean_abs_diff_vs_base": 0.0,
                "median_abs_diff_vs_base": 0.0,
                "unchanged_bar_count_share": 0.0,
            }
            continue

        base_values = [item[0] for item in paired]
        current_values = [item[1] for item in paired]
        abs_diffs = [abs(a - b) for a, b in paired]
        summary[str(offset)] = {
            "paired_events": len(paired),
            "corr_vs_base": round(float(safe_corr(base_values, current_values)), 6),
            "mean_abs_diff_vs_base": round(float(statistics.mean(abs_diffs)), 8),
            "median_abs_diff_vs_base": round(float(statistics.median(abs_diffs)), 8),
            "unchanged_bar_count_share": round(unchanged_bars / max(compared, 1), 6),
        }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = parse_target_rows(args.targets_csv)
    offsets = [int(item.strip()) for item in args.offset_minutes.split(",") if item.strip()]
    ticker_bars = {}
    output_rows = []

    for d_path in sorted(args.d_dir.resolve().glob("*.csv")):
        ticker_bars[d_path.stem.upper()] = load_ticker_bars(d_path)

    for row in target_rows:
        ticker = row["ticker"].upper()
        bars_by_date = ticker_bars.get(ticker)
        if bars_by_date is None:
            continue

        scheduled_dt = dt.datetime.fromisoformat(row["scheduled_datetime"])
        call_end_dt = dt.datetime.fromisoformat(row["call_end_datetime"])
        event_date = scheduled_dt.date()
        day_rows = bars_by_date.get(event_date, [])

        for offset_minutes in offsets:
            start_dt = scheduled_dt + dt.timedelta(minutes=offset_minutes)
            end_dt = call_end_dt + dt.timedelta(minutes=offset_minutes)
            pre_start = start_dt - dt.timedelta(minutes=args.pre_call_minutes)
            post_end = end_dt + dt.timedelta(minutes=args.post_call_minutes)

            pre_rows = [bar for bar in day_rows if pre_start <= bar["_timestamp"] < start_dt]
            call_rows = [bar for bar in day_rows if start_dt <= bar["_timestamp"] < end_dt]
            post_rows = [bar for bar in day_rows if end_dt <= bar["_timestamp"] < post_end]

            pre_stats = aggregate_window(pre_rows)
            call_stats = aggregate_window(call_rows)
            post_stats = aggregate_window(post_rows)
            output_rows.append(
                {
                    "event_key": row["event_key"],
                    "ticker": ticker,
                    "year": row.get("year", ""),
                    "quarter": row.get("quarter", ""),
                    "offset_minutes": str(offset_minutes),
                    "scheduled_datetime_offset": start_dt.isoformat(sep=" "),
                    "call_end_datetime_offset": end_dt.isoformat(sep=" "),
                    "pre_60m_bar_count": pre_stats["bar_count"],
                    "pre_60m_rv": pre_stats["rv"],
                    "within_call_bar_count": call_stats["bar_count"],
                    "within_call_rv": call_stats["rv"],
                    "post_call_60m_bar_count": post_stats["bar_count"],
                    "post_call_60m_rv": post_stats["rv"],
                }
            )

    summary = {
        "offset_minutes": offsets,
        "num_events": len({row["event_key"] for row in output_rows}),
        "num_rows": len(output_rows),
        "post_call_60m_rv": summarize_against_base(output_rows, "post_call_60m_rv"),
        "within_call_rv": summarize_against_base(output_rows, "within_call_rv"),
    }

    write_csv(output_dir / "scheduled_time_offset_targets.csv", output_rows)
    write_json(output_dir / "scheduled_time_offset_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
