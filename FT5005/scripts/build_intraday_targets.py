#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import datetime as dt
import math
from collections import Counter, defaultdict
from pathlib import Path
import re

from dj30_qc_utils import (
    build_event_path_lookup,
    detect_filename_metadata,
    iter_files,
    load_csv_rows,
    load_json,
    normalize_text,
    parse_a2_scheduled_metadata,
    safe_float,
    text_similarity,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build event-level intraday targets using A2 scheduled time, A4 duration, and D 5-minute bars."
    )
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--d-dir", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/targets"))
    parser.add_argument("--broad-match-score", type=float, default=75.0)
    parser.add_argument("--broad-text-similarity", type=float, default=0.65)
    parser.add_argument("--post-call-minutes", type=int, default=60)
    parser.add_argument("--pre-call-minutes", type=int, default=60)
    return parser.parse_args()


def broad_pass_a4_row(row: dict, broad_match_score: float, broad_text_similarity: float) -> bool:
    official_text = normalize_text(row.get("official_text", ""))
    if not official_text:
        return False

    start_sec = safe_float(row.get("start_sec"))
    end_sec = safe_float(row.get("end_sec"))
    if start_sec is None or end_sec is None or start_sec < 0 or end_sec <= start_sec:
        return False

    match_score = safe_float(row.get("match_score"))
    similarity = text_similarity(official_text.lower(), normalize_text(row.get("asr_matched_text", "")).lower())

    return (match_score is not None and match_score >= broad_match_score) or similarity >= broad_text_similarity


def compute_call_duration_seconds(
    a4_path: Path,
    broad_match_score: float,
    broad_text_similarity: float,
) -> tuple[float | None, int]:
    max_end_sec = None
    kept_rows = 0
    for row in load_csv_rows(a4_path):
        if not broad_pass_a4_row(row, broad_match_score, broad_text_similarity):
            continue
        end_sec = safe_float(row.get("end_sec"))
        if end_sec is None:
            continue
        kept_rows += 1
        max_end_sec = end_sec if max_end_sec is None else max(max_end_sec, end_sec)
    return max_end_sec, kept_rows


def compute_returns(rows: list[dict]) -> list[dict]:
    previous_close = None
    enriched = []
    for row in rows:
        close = safe_float(row["Close"])
        volume = safe_float(row["Volume"]) or 0.0
        log_return = None
        if close is not None and close > 0 and previous_close is not None and previous_close > 0:
            log_return = math.log(close / previous_close)
        if close is not None and close > 0:
            previous_close = close
        enriched.append(
            {
                **row,
                "close_float": close,
                "volume_float": volume,
                "log_return": log_return,
            }
        )
    return enriched


def aggregate_window(rows: list[dict]) -> dict:
    returns = [row["log_return"] for row in rows if row["log_return"] is not None]
    rv = sum(value * value for value in returns)
    vw_rv = sum((row["log_return"] ** 2) * math.log1p(row["volume_float"]) for row in rows if row["log_return"] is not None)
    volume_sum = sum(row["volume_float"] for row in rows)
    return {
        "bar_count": len(rows),
        "rv": round(rv, 8),
        "vw_rv": round(vw_rv, 8),
        "abs_return_sum": round(sum(abs(value) for value in returns), 8),
        "volume_sum": round(volume_sum, 4),
        "first_bar": rows[0]["DateTime"] if rows else "",
        "last_bar": rows[-1]["DateTime"] if rows else "",
    }


def build_event_metadata(
    a2_dir: Path,
    a4_dir: Path,
    a1_dir: Path | None,
    broad_match_score: float,
    broad_text_similarity: float,
) -> tuple[list[dict], dict]:
    a4_lookup, _ = build_event_path_lookup(iter_files(a4_dir.resolve(), [".csv"]))

    a1_lookup = {}
    a1_duplicates = {}
    if a1_dir is not None:
        a1_lookup, a1_duplicates = build_event_path_lookup(iter_files(a1_dir.resolve(), [".json"]))

    events = []
    counts = Counter()
    for a2_path in iter_files(a2_dir.resolve(), [".html", ".htm"]):
        meta = detect_filename_metadata(a2_path)
        event_key = meta["event_key"]
        counts["a2_total"] += 1
        if not event_key:
            counts["missing_event_key"] += 1
            continue

        a4_path = a4_lookup.get(event_key)
        if a4_path is None:
            counts["missing_a4"] += 1
            continue

        raw_html = a2_path.read_text(encoding="utf-8", errors="ignore")
        scheduled_meta = parse_a2_scheduled_metadata(raw_html)
        scheduled_iso = scheduled_meta["scheduled_datetime_iso"]
        a1_path = a1_lookup.get(event_key)
        if not scheduled_iso and a1_path is not None and scheduled_meta["scheduled_time_text"]:
            fallback_date = parse_a1_headline_date(a1_path)
            if fallback_date is not None:
                fallback_dt = dt.datetime.strptime(
                    f"{fallback_date.isoformat()} {scheduled_meta['scheduled_time_text']}",
                    "%Y-%m-%d %I:%M %p",
                )
                scheduled_iso = fallback_dt.isoformat(sep=" ")
                counts["scheduled_time_recovered_from_a1"] += 1
        if not scheduled_iso:
            counts["missing_scheduled_time"] += 1
            continue

        duration_sec, kept_rows = compute_call_duration_seconds(
            a4_path, broad_match_score=broad_match_score, broad_text_similarity=broad_text_similarity
        )
        if duration_sec is None:
            counts["missing_a4_duration"] += 1
            continue

        scheduled_dt = dt.datetime.fromisoformat(scheduled_iso)
        call_end_dt = scheduled_dt + dt.timedelta(seconds=duration_sec)
        events.append(
            {
                "event_key": event_key,
                "ticker": meta["ticker"],
                "year": meta["year"],
                "quarter": meta["quarter"],
                "a2_path": str(a2_path.resolve()),
                "a4_path": str(a4_path.resolve()),
                "a1_path": str(a1_path.resolve()) if a1_path is not None else "",
                "scheduled_datetime": scheduled_dt,
                "call_end_datetime": call_end_dt,
                "call_duration_sec": round(duration_sec, 4),
                "a4_kept_rows_for_duration": kept_rows,
                "header_paragraph": scheduled_meta["header_paragraph"],
            }
        )
        counts["usable_events"] += 1

    summary = dict(counts)
    if a1_duplicates:
        summary["a1_duplicate_event_keys"] = a1_duplicates
    return events, summary


def parse_a1_headline_date(a1_path: Path) -> dt.date | None:
    try:
        headline = normalize_text(load_json(a1_path).get("headline", ""))
    except Exception:
        return None

    match = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})$", headline)
    if not match:
        return None

    date_text = match.group(1)
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            return dt.datetime.strptime(date_text, fmt).date()
        except ValueError:
            continue
    return None


def build_targets_for_ticker(
    ticker: str,
    ticker_events: list[dict],
    d_path: Path,
    pre_call_minutes: int,
    post_call_minutes: int,
) -> list[dict]:
    events_by_date = {event["scheduled_datetime"].date(): event for event in ticker_events}
    rows_by_date = defaultdict(list)

    with d_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = dt.datetime.fromisoformat(row["DateTime"])
            event = events_by_date.get(timestamp.date())
            if event is None:
                continue
            rows_by_date[timestamp.date()].append({**row, "_timestamp": timestamp})

    output_rows = []
    for event_date, event in sorted(events_by_date.items()):
        day_rows = sorted(rows_by_date.get(event_date, []), key=lambda row: row["_timestamp"])
        day_rows = compute_returns(day_rows)

        scheduled_dt = event["scheduled_datetime"]
        call_end_dt = event["call_end_datetime"]
        pre_start = scheduled_dt - dt.timedelta(minutes=pre_call_minutes)
        post_end = call_end_dt + dt.timedelta(minutes=post_call_minutes)

        pre_rows = [row for row in day_rows if pre_start <= row["_timestamp"] < scheduled_dt]
        call_rows = [row for row in day_rows if scheduled_dt <= row["_timestamp"] < call_end_dt]
        post_rows = [row for row in day_rows if call_end_dt <= row["_timestamp"] < post_end]

        pre_stats = aggregate_window(pre_rows)
        call_stats = aggregate_window(call_rows)
        post_stats = aggregate_window(post_rows)

        output_rows.append(
            {
                "event_key": event["event_key"],
                "ticker": ticker,
                "year": event["year"],
                "quarter": event["quarter"],
                "scheduled_datetime": scheduled_dt.isoformat(sep=" "),
                "call_end_datetime": call_end_dt.isoformat(sep=" "),
                "call_duration_sec": event["call_duration_sec"],
                "a4_kept_rows_for_duration": event["a4_kept_rows_for_duration"],
                "day_bar_count": len(day_rows),
                "day_first_bar": day_rows[0]["DateTime"] if day_rows else "",
                "day_last_bar": day_rows[-1]["DateTime"] if day_rows else "",
                "extended_hours_present": bool(
                    day_rows
                    and (
                        day_rows[0]["_timestamp"].time() < dt.time(9, 30)
                        or day_rows[-1]["_timestamp"].time() > dt.time(16, 0)
                    )
                ),
                "pre_60m_bar_count": pre_stats["bar_count"],
                "pre_60m_rv": pre_stats["rv"],
                "pre_60m_vw_rv": pre_stats["vw_rv"],
                "pre_60m_volume_sum": pre_stats["volume_sum"],
                "within_call_bar_count": call_stats["bar_count"],
                "within_call_rv": call_stats["rv"],
                "within_call_vw_rv": call_stats["vw_rv"],
                "within_call_volume_sum": call_stats["volume_sum"],
                "within_call_first_bar": call_stats["first_bar"],
                "within_call_last_bar": call_stats["last_bar"],
                "post_call_60m_bar_count": post_stats["bar_count"],
                "post_call_60m_rv": post_stats["rv"],
                "post_call_60m_vw_rv": post_stats["vw_rv"],
                "post_call_60m_volume_sum": post_stats["volume_sum"],
                "post_call_60m_first_bar": post_stats["first_bar"],
                "post_call_60m_last_bar": post_stats["last_bar"],
            }
        )

    return output_rows


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    events, metadata_summary = build_event_metadata(
        args.a2_dir,
        args.a4_dir,
        args.a1_dir,
        broad_match_score=args.broad_match_score,
        broad_text_similarity=args.broad_text_similarity,
    )

    events_by_ticker = defaultdict(list)
    for event in events:
        events_by_ticker[event["ticker"]].append(event)

    target_rows = []
    tickers_missing_in_d = []
    for ticker, ticker_events in sorted(events_by_ticker.items()):
        d_path = args.d_dir / f"{ticker}.csv"
        if not d_path.exists():
            tickers_missing_in_d.append(ticker)
            continue
        target_rows.extend(
            build_targets_for_ticker(
                ticker,
                ticker_events,
                d_path,
                pre_call_minutes=args.pre_call_minutes,
                post_call_minutes=args.post_call_minutes,
            )
        )

    write_csv(output_dir / "event_intraday_targets.csv", target_rows)
    write_json(
        output_dir / "event_intraday_targets_summary.json",
        {
            "event_metadata_summary": metadata_summary,
            "num_events_with_targets": len(target_rows),
            "tickers_missing_in_d": tickers_missing_in_d,
            "events_with_post_call_bars": sum(1 for row in target_rows if row["post_call_60m_bar_count"] > 0),
            "events_with_within_call_bars": sum(1 for row in target_rows if row["within_call_bar_count"] > 0),
            "extended_hours_events": sum(1 for row in target_rows if row["extended_hours_present"]),
        },
    )

    print(
        {
            "usable_event_metadata": metadata_summary.get("usable_events", 0),
            "num_events_with_targets": len(target_rows),
            "tickers_missing_in_d": tickers_missing_in_d,
        }
    )


if __name__ == "__main__":
    main()
