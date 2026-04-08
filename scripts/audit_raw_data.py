#!/usr/bin/env python3
"""
Audit raw PREC input data under data/raw.

This script inspects the minimum raw assets currently needed by the
root-level PREC pipeline:
    - A1 JSON transcripts
    - A4 aligned timestamp files
    - D 5-minute stock bars

Outputs:
    - summary.json
    - event_coverage.csv
    - ticker_coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


EVENT_RE = re.compile(r"^(?P<ticker>[A-Z0-9.\-]+)_(?P<year>\d{4})Q(?P<quarter>[1-4])$")
ALIGNED_RE = re.compile(
    r"^(?P<ticker>[A-Z0-9.\-]+)_(?P<year>\d{4})Q(?P<quarter>[1-4])_aligned$"
)


@dataclass(frozen=True)
class EventKey:
    ticker: str
    year: int
    quarter: int

    @property
    def event_id(self) -> str:
        return f"{self.ticker}_{self.year}Q{self.quarter}"


def parse_event_key(stem: str, aligned: bool = False) -> EventKey | None:
    pattern = ALIGNED_RE if aligned else EVENT_RE
    match = pattern.match(stem)
    if match is None:
        return None
    return EventKey(
        ticker=match.group("ticker"),
        year=int(match.group("year")),
        quarter=int(match.group("quarter")),
    )


def safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def find_single_subdir(parent: Path, preferred_name: str) -> Path:
    preferred = parent / preferred_name
    if preferred.exists():
        return preferred

    subdirs = [path for path in parent.iterdir() if path.is_dir() and not path.name.startswith(".")]
    if len(subdirs) == 1:
        return subdirs[0]

    raise FileNotFoundError(
        f"Could not locate expected directory under {parent}. "
        f"Tried {preferred_name} and single-subdir fallback."
    )


def scan_a1_json(json_root: Path, raw_root: Path) -> tuple[dict[EventKey, dict], dict]:
    events: dict[EventKey, dict] = {}
    invalid_files = []
    duplicate_keys = Counter()

    for path in sorted(json_root.rglob("*.json")):
        if path.name.startswith("."):
            continue
        key = parse_event_key(path.stem)
        if key is None:
            invalid_files.append(safe_relpath(path, raw_root))
            continue

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        components = payload.get("components") or []
        question_count = 0
        answer_count = 0
        speaker_names = set()
        for component in components:
            component_type = str(component.get("componenttypename") or "")
            if component_type == "Question":
                question_count += 1
            elif component_type == "Answer":
                answer_count += 1

            speaker_name = component.get("personname")
            if speaker_name:
                speaker_names.add(str(speaker_name))

        record = {
            "event_id": key.event_id,
            "ticker": key.ticker,
            "year": key.year,
            "quarter": key.quarter,
            "a1_path": safe_relpath(path, raw_root),
            "headline": payload.get("headline"),
            "event_date": payload.get("mostimportantdate"),
            "transcript_creation_date": payload.get("transcriptcreationdate"),
            "component_count": len(components),
            "question_count": question_count,
            "answer_count": answer_count,
            "speaker_count": len(speaker_names),
        }

        if key in events:
            duplicate_keys[key.event_id] += 1
        events[key] = record

    summary = {
        "file_count": len(events),
        "invalid_name_files": invalid_files,
        "duplicate_event_ids": dict(duplicate_keys),
    }
    return events, summary


def scan_a4_aligned(a4_root: Path, raw_root: Path) -> tuple[dict[EventKey, dict], dict]:
    events: dict[EventKey, dict] = {}
    invalid_files = []
    duplicate_keys = Counter()

    for path in sorted(a4_root.rglob("*_aligned.csv")):
        if path.name.startswith("."):
            continue
        key = parse_event_key(path.stem, aligned=True)
        if key is None:
            invalid_files.append(safe_relpath(path, raw_root))
            continue

        row_count = 0
        non_empty_match_score = 0
        non_empty_start_end = 0
        overall_tfidf_values = []

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            for row in reader:
                row_count += 1
                if (row.get("match_score") or "").strip():
                    non_empty_match_score += 1
                if (row.get("start_sec") or "").strip() and (row.get("end_sec") or "").strip():
                    non_empty_start_end += 1
                tfidf_value = (row.get("overall_TFIDF") or "").strip()
                if tfidf_value:
                    try:
                        overall_tfidf_values.append(float(tfidf_value))
                    except ValueError:
                        pass

        record = {
            "event_id": key.event_id,
            "ticker": key.ticker,
            "year": key.year,
            "quarter": key.quarter,
            "a4_path": safe_relpath(path, raw_root),
            "aligned_sentence_count": row_count,
            "match_score_coverage": (non_empty_match_score / row_count) if row_count else 0.0,
            "timestamp_coverage": (non_empty_start_end / row_count) if row_count else 0.0,
            "mean_overall_tfidf": (
                sum(overall_tfidf_values) / len(overall_tfidf_values)
                if overall_tfidf_values
                else None
            ),
            "columns_present": ",".join(fieldnames),
        }

        if key in events:
            duplicate_keys[key.event_id] += 1
        events[key] = record

    summary = {
        "file_count": len(events),
        "invalid_name_files": invalid_files,
        "duplicate_event_ids": dict(duplicate_keys),
    }
    return events, summary


def scan_stock_bars(stock_root: Path, raw_root: Path) -> tuple[dict[str, dict], dict]:
    ticker_records: dict[str, dict] = {}
    invalid_files = []

    for path in sorted(stock_root.glob("*.csv")):
        if path.name.startswith("."):
            continue
        ticker = path.stem.upper()
        row_count = 0
        min_dt = None
        max_dt = None
        header = []

        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            header = reader.fieldnames or []
            for row in reader:
                row_count += 1
                dt_value = (row.get("DateTime") or "").strip()
                if dt_value:
                    if min_dt is None or dt_value < min_dt:
                        min_dt = dt_value
                    if max_dt is None or dt_value > max_dt:
                        max_dt = dt_value

        if row_count == 0:
            invalid_files.append(safe_relpath(path, raw_root))

        ticker_records[ticker] = {
            "ticker": ticker,
            "stock_path": safe_relpath(path, raw_root),
            "stock_row_count": row_count,
            "stock_min_datetime": min_dt,
            "stock_max_datetime": max_dt,
            "stock_columns_present": ",".join(header),
        }

    summary = {
        "ticker_file_count": len(ticker_records),
        "empty_stock_files": invalid_files,
    }
    return ticker_records, summary


def build_event_coverage(
    a1_events: dict[EventKey, dict],
    a4_events: dict[EventKey, dict],
    stock_tickers: dict[str, dict],
) -> list[dict]:
    rows = []
    all_keys = sorted(
        set(a1_events) | set(a4_events),
        key=lambda item: (item.ticker, item.year, item.quarter),
    )

    for key in all_keys:
        a1 = a1_events.get(key)
        a4 = a4_events.get(key)
        has_a1 = a1 is not None
        has_a4 = a4 is not None
        has_stock = key.ticker in stock_tickers

        missing_reasons = []
        if not has_a1:
            missing_reasons.append("missing_a1")
        if not has_a4:
            missing_reasons.append("missing_a4")
        if not has_stock:
            missing_reasons.append("missing_stock")

        row = {
            "event_id": key.event_id,
            "ticker": key.ticker,
            "year": key.year,
            "quarter": key.quarter,
            "has_a1_json": int(has_a1),
            "has_a4_aligned": int(has_a4),
            "has_stock_ticker": int(has_stock),
            "usable_for_min_pipeline": int(has_a1 and has_a4 and has_stock),
            "missing_reasons": ",".join(missing_reasons),
            "event_date": a1.get("event_date") if a1 else None,
            "headline": a1.get("headline") if a1 else None,
            "a1_path": a1.get("a1_path") if a1 else None,
            "a4_path": a4.get("a4_path") if a4 else None,
            "component_count": a1.get("component_count") if a1 else None,
            "question_count": a1.get("question_count") if a1 else None,
            "answer_count": a1.get("answer_count") if a1 else None,
            "speaker_count": a1.get("speaker_count") if a1 else None,
            "aligned_sentence_count": a4.get("aligned_sentence_count") if a4 else None,
            "match_score_coverage": a4.get("match_score_coverage") if a4 else None,
            "timestamp_coverage": a4.get("timestamp_coverage") if a4 else None,
            "mean_overall_tfidf": a4.get("mean_overall_tfidf") if a4 else None,
        }
        rows.append(row)

    return rows


def build_ticker_coverage(
    event_rows: list[dict],
    stock_tickers: dict[str, dict],
) -> list[dict]:
    a1_counts = Counter()
    a4_counts = Counter()
    usable_counts = Counter()

    for row in event_rows:
        ticker = row["ticker"]
        if row["has_a1_json"]:
            a1_counts[ticker] += 1
        if row["has_a4_aligned"]:
            a4_counts[ticker] += 1
        if row["usable_for_min_pipeline"]:
            usable_counts[ticker] += 1

    all_tickers = sorted(set(a1_counts) | set(a4_counts) | set(stock_tickers))
    rows = []
    for ticker in all_tickers:
        stock_record = stock_tickers.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "a1_event_count": a1_counts[ticker],
                "a4_event_count": a4_counts[ticker],
                "usable_event_count": usable_counts[ticker],
                "has_stock_file": int(ticker in stock_tickers),
                "stock_row_count": stock_record.get("stock_row_count"),
                "stock_min_datetime": stock_record.get("stock_min_datetime"),
                "stock_max_datetime": stock_record.get("stock_max_datetime"),
                "stock_path": stock_record.get("stock_path"),
            }
        )

    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit raw PREC data coverage.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing copied raw experiment data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/audit/raw_data"),
        help="Directory for audit outputs.",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()

    a1_root = find_single_subdir(data_root / "A1.ECC_Text_Json_DJ30", "ECC_Text_Json_DOW30")
    a4_root = find_single_subdir(data_root / "A4.ECC_Timestamp_DJ30", "ECC_Timestamp_DOW30")
    stock_root = data_root / "D.Stock_5min_DJ30"
    if not stock_root.exists():
        raise FileNotFoundError(f"Missing stock directory: {stock_root}")

    a1_events, a1_summary = scan_a1_json(a1_root, data_root)
    a4_events, a4_summary = scan_a4_aligned(a4_root, data_root)
    stock_tickers, stock_summary = scan_stock_bars(stock_root, data_root)

    event_rows = build_event_coverage(a1_events, a4_events, stock_tickers)
    ticker_rows = build_ticker_coverage(event_rows, stock_tickers)

    usable_events = [row for row in event_rows if row["usable_for_min_pipeline"]]
    a1_only = [row["event_id"] for row in event_rows if row["has_a1_json"] and not row["has_a4_aligned"]]
    a4_only = [row["event_id"] for row in event_rows if row["has_a4_aligned"] and not row["has_a1_json"]]
    missing_stock = [row["event_id"] for row in event_rows if not row["has_stock_ticker"]]
    missing_event_dates = [row["event_id"] for row in event_rows if row["has_a1_json"] and not row["event_date"]]

    summary = {
        "data_root": str(data_root),
        "output_dir": str(output_dir),
        "a1_summary": a1_summary,
        "a4_summary": a4_summary,
        "stock_summary": stock_summary,
        "event_counts": {
            "a1_event_count": len(a1_events),
            "a4_event_count": len(a4_events),
            "union_event_count": len(event_rows),
            "usable_event_count": len(usable_events),
            "a1_without_a4_count": len(a1_only),
            "a4_without_a1_count": len(a4_only),
            "missing_stock_count": len(missing_stock),
            "missing_event_date_count": len(missing_event_dates),
        },
        "ticker_counts": {
            "a1_ticker_count": len({key.ticker for key in a1_events}),
            "a4_ticker_count": len({key.ticker for key in a4_events}),
            "stock_ticker_count": len(stock_tickers),
            "usable_ticker_count": len({row["ticker"] for row in usable_events}),
        },
        "samples": {
            "a1_without_a4": a1_only[:25],
            "a4_without_a1": a4_only[:25],
            "missing_stock": missing_stock[:25],
            "missing_event_dates": missing_event_dates[:25],
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_csv(output_dir / "event_coverage.csv", event_rows)
    write_csv(output_dir / "ticker_coverage.csv", ticker_rows)

    print("Raw data audit complete")
    print(f"  data_root: {data_root}")
    print(f"  output_dir: {output_dir}")
    print(f"  A1 events: {len(a1_events)}")
    print(f"  A4 events: {len(a4_events)}")
    print(f"  stock tickers: {len(stock_tickers)}")
    print(f"  usable min-pipeline events: {len(usable_events)}")
    print(f"  A1 without A4: {len(a1_only)}")
    print(f"  A4 without A1: {len(a4_only)}")
    print(f"  missing stock ticker: {len(missing_stock)}")


if __name__ == "__main__":
    main()
