#!/usr/bin/env python3
"""
Build an event-level manifest from raw-data audit outputs.

This manifest is the bridge between:
    raw assets (A1 / A4 / D)
and
    later panel / target / split builders.

Inputs:
    - outputs/audit/raw_data/event_coverage.csv
    - outputs/audit/raw_data/ticker_coverage.csv

Outputs:
    - all_events.csv
    - min_pipeline_events.csv
    - manifest_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


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


def parse_event_date(value: str | None) -> tuple[str | None, str | None]:
    """Parse event date into ISO date and sortable timestamp."""
    if not value:
        return None, None

    value = value.strip()
    if not value:
        return None, None

    for fmt in ("%Y-%m-%d", "%b-%d-%Y", "%b %d, %Y"):
        try:
            dt = datetime.strptime(value, fmt)
            return dt.date().isoformat(), dt.strftime("%Y%m%d")
        except ValueError:
            continue

    return value, None


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def build_event_key(ticker: str, year: str, quarter: str) -> str:
    if ticker and year and quarter:
        return f"{ticker}_{year}{quarter}"
    return ""


def detect_filename_metadata(path: Path) -> dict[str, str]:
    stem = path.stem.upper()
    tokens = [token for token in re.split(r"[^A-Z0-9]+", stem) if token]
    ticker = ""
    for token in tokens:
        if re.fullmatch(r"[A-Z]{1,5}", token):
            ticker = token
            break
    quarter_match = re.search(r"(20\d{2})[^A-Z0-9]?Q([1-4])", stem)
    if not quarter_match:
        quarter_match = re.search(r"Q([1-4])[^A-Z0-9]?(20\d{2})", stem)

    year = ""
    quarter = ""
    if quarter_match:
        groups = quarter_match.groups()
        if len(groups[0]) == 4:
            year = groups[0]
            quarter = f"Q{groups[1]}"
        else:
            year = groups[1]
            quarter = f"Q{groups[0]}"

    return {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "event_key": build_event_key(ticker, year, quarter),
    }


def build_event_path_lookup(path: Path, suffixes: tuple[str, ...]) -> dict[str, Path]:
    lookup: dict[str, list[Path]] = {}
    for file_path in sorted(path.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in suffixes:
            continue
        event_key = detect_filename_metadata(file_path)["event_key"]
        if event_key:
            lookup.setdefault(event_key, []).append(file_path)

    resolved = {}
    for event_key, candidates in lookup.items():
        ranked = sorted(
            candidates,
            key=lambda candidate: (
                candidate.stem.endswith("_2"),
                len(candidate.stem),
                candidate.name,
            ),
        )
        resolved[event_key] = ranked[0]
    return resolved


def maybe_relpath(path: Path, root: Path) -> str | None:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return None


def add_paths(
    row: dict,
    data_root: Path,
    stock_info: dict | None,
    a2_path: Path | None,
) -> dict:
    out = dict(row)

    a1_rel = out.get("a1_path") or ""
    a4_rel = out.get("a4_path") or ""
    stock_rel = (stock_info or {}).get("stock_path") or ""

    out["a1_relpath"] = a1_rel or None
    out["a4_relpath"] = a4_rel or None
    out["stock_relpath"] = stock_rel or None
    out["a1_abspath"] = str((data_root / a1_rel).resolve()) if a1_rel else None
    out["a4_abspath"] = str((data_root / a4_rel).resolve()) if a4_rel else None
    out["stock_abspath"] = str((data_root / stock_rel).resolve()) if stock_rel else None
    out["a2_relpath"] = maybe_relpath(a2_path, data_root) if a2_path else None
    out["a2_abspath"] = str(a2_path.resolve()) if a2_path else None
    return out


def build_manifest_rows(
    event_rows: list[dict],
    ticker_rows: list[dict],
    data_root: Path,
    a2_lookup: dict[str, Path],
) -> list[dict]:
    ticker_lookup = {row["ticker"]: row for row in ticker_rows}
    manifest_rows = []

    for row in event_rows:
        event_date_iso, event_date_sort = parse_event_date(row.get("event_date"))
        stock_info = ticker_lookup.get(row["ticker"])
        event_key = build_event_key(row["ticker"], row["year"], f"Q{row['quarter']}")
        a2_path = a2_lookup.get(event_key)

        manifest_row = {
            "event_id": row["event_id"],
            "ticker": row["ticker"],
            "year": parse_int(row["year"]),
            "quarter": parse_int(row["quarter"]),
            "fiscal_period": f"{row['year']}Q{row['quarter']}",
            "event_date": event_date_iso,
            "event_date_sort_key": event_date_sort,
            "headline": row.get("headline") or None,
            "has_a1_json": parse_int(row["has_a1_json"]),
            "has_a4_aligned": parse_int(row["has_a4_aligned"]),
            "has_stock_ticker": parse_int(row["has_stock_ticker"]),
            "usable_for_min_pipeline": parse_int(row["usable_for_min_pipeline"]),
            "missing_reasons": row.get("missing_reasons") or None,
            "component_count": parse_int(row.get("component_count")),
            "question_count": parse_int(row.get("question_count")),
            "answer_count": parse_int(row.get("answer_count")),
            "speaker_count": parse_int(row.get("speaker_count")),
            "aligned_sentence_count": parse_int(row.get("aligned_sentence_count")),
            "match_score_coverage": parse_float(row.get("match_score_coverage")),
            "timestamp_coverage": parse_float(row.get("timestamp_coverage")),
            "mean_overall_tfidf": parse_float(row.get("mean_overall_tfidf")),
            "stock_row_count": parse_int((stock_info or {}).get("stock_row_count")),
            "stock_min_datetime": (stock_info or {}).get("stock_min_datetime") or None,
            "stock_max_datetime": (stock_info or {}).get("stock_max_datetime") or None,
            "has_a2_html": 1 if a2_path else 0,
            "a1_path": row.get("a1_path") or None,
            "a4_path": row.get("a4_path") or None,
            "a2_path": str(a2_path.resolve()) if a2_path else None,
        }

        manifest_row = add_paths(manifest_row, data_root, stock_info, a2_path)
        manifest_rows.append(manifest_row)

    manifest_rows.sort(
        key=lambda item: (
            item["event_date_sort_key"] or "99999999",
            item["ticker"],
            item["quarter"] or 0,
        )
    )

    for idx, row in enumerate(manifest_rows, start=1):
        row["event_order"] = idx

    return manifest_rows


def summarise_manifest(rows: list[dict]) -> dict:
    usable_rows = [row for row in rows if row["usable_for_min_pipeline"] == 1]
    missing_reason_counts = Counter()
    usable_by_ticker = Counter()

    for row in rows:
        reasons = (row.get("missing_reasons") or "").split(",")
        for reason in reasons:
            reason = reason.strip()
            if reason:
                missing_reason_counts[reason] += 1

    for row in usable_rows:
        usable_by_ticker[row["ticker"]] += 1

    return {
        "all_event_count": len(rows),
        "min_pipeline_event_count": len(usable_rows),
        "ticker_count_all": len({row["ticker"] for row in rows}),
        "ticker_count_min_pipeline": len({row["ticker"] for row in usable_rows}),
        "min_event_date": next((row["event_date"] for row in rows if row["event_date"]), None),
        "max_event_date": next((row["event_date"] for row in reversed(rows) if row["event_date"]), None),
        "missing_reason_counts": dict(missing_reason_counts),
        "top_usable_tickers": [
            {"ticker": ticker, "usable_event_count": count}
            for ticker, count in usable_by_ticker.most_common(15)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build event-level manifest from audit outputs.")
    parser.add_argument(
        "--audit-dir",
        type=Path,
        default=Path("outputs/audit/raw_data"),
        help="Directory containing event_coverage.csv and ticker_coverage.csv.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory for copied raw experiment data.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/event_manifest"),
        help="Directory for manifest outputs.",
    )
    parser.add_argument(
        "--a2-dir",
        type=Path,
        default=Path("data/raw/A2.ECC_Text_html_DJ30"),
        help="Directory containing A2 HTML files used for timing recovery.",
    )
    args = parser.parse_args()

    audit_dir = args.audit_dir.resolve()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    a2_dir = args.a2_dir.resolve()

    event_coverage_path = audit_dir / "event_coverage.csv"
    ticker_coverage_path = audit_dir / "ticker_coverage.csv"
    if not event_coverage_path.exists():
        raise FileNotFoundError(f"Missing audit file: {event_coverage_path}")
    if not ticker_coverage_path.exists():
        raise FileNotFoundError(f"Missing audit file: {ticker_coverage_path}")

    event_rows = read_csv(event_coverage_path)
    ticker_rows = read_csv(ticker_coverage_path)
    a2_lookup = build_event_path_lookup(a2_dir, (".html", ".htm")) if a2_dir.exists() else {}

    manifest_rows = build_manifest_rows(event_rows, ticker_rows, data_root, a2_lookup)
    min_pipeline_rows = [row for row in manifest_rows if row["usable_for_min_pipeline"] == 1]
    summary = {
        "audit_dir": str(audit_dir),
        "data_root": str(data_root),
        "a2_dir": str(a2_dir),
        "output_dir": str(output_dir),
        "summary": summarise_manifest(manifest_rows),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "all_events.csv", manifest_rows)
    write_csv(output_dir / "min_pipeline_events.csv", min_pipeline_rows)
    with (output_dir / "manifest_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Event manifest build complete")
    print(f"  audit_dir: {audit_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  all events: {len(manifest_rows)}")
    print(f"  min-pipeline events: {len(min_pipeline_rows)}")


if __name__ == "__main__":
    main()
