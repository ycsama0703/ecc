#!/usr/bin/env python3
"""
Build a fiscal-year split aligned with the older FT5005 experiment scripts.

Default policy:
    - train: fiscal_year <= 2021
    - val: fiscal_year == 2022
    - test: fiscal_year > 2022
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


EVENT_ID_YEAR_RE = re.compile(r"_(20\d{2})Q([1-4])$")


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


def infer_fiscal_year(row: dict) -> int:
    for key in ("fiscal_year", "year"):
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    event_id = row.get("event_id", "")
    match = EVENT_ID_YEAR_RE.search(event_id)
    if not match:
        raise ValueError(f"Could not infer fiscal year from row: event_id={event_id}")
    return int(match.group(1))


def infer_fiscal_quarter(row: dict) -> int | None:
    for key in ("fiscal_quarter", "quarter"):
        value = row.get(key)
        if value not in (None, ""):
            return int(float(value))
    event_id = row.get("event_id", "")
    match = EVENT_ID_YEAR_RE.search(event_id)
    return int(match.group(2)) if match else None


def split_label_for_year(fiscal_year: int, train_end_year: int, val_year: int) -> str:
    if fiscal_year <= train_end_year:
        return "train"
    if fiscal_year == val_year:
        return "val"
    return "test"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fiscal-year split.")
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--split-version", type=str, default="year_v1")
    parser.add_argument("--train-end-year", type=int, default=2021)
    parser.add_argument("--val-year", type=int, default=2022)
    args = parser.parse_args()

    panel_rows = read_csv(args.panel_csv.resolve())
    split_rows = []
    year_counts = Counter()
    split_counts = Counter()

    for row in panel_rows:
        fiscal_year = infer_fiscal_year(row)
        fiscal_quarter = infer_fiscal_quarter(row)
        split_label = split_label_for_year(
            fiscal_year=fiscal_year,
            train_end_year=args.train_end_year,
            val_year=args.val_year,
        )
        split_counts[split_label] += 1
        year_counts[fiscal_year] += 1
        split_rows.append(
            {
                "event_id": row["event_id"],
                "ticker": row.get("ticker", ""),
                "event_date": row.get("event_date", ""),
                "fiscal_year": fiscal_year,
                "fiscal_quarter": fiscal_quarter or "",
                "split_version": args.split_version,
                "split_label": split_label,
                "train_flag": 1 if split_label == "train" else 0,
                "val_flag": 1 if split_label == "val" else 0,
                "test_flag": 1 if split_label == "test" else 0,
            }
        )

    write_csv(args.output_csv.resolve(), split_rows)
    summary = {
        "panel_csv": str(args.panel_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "split_version": args.split_version,
        "train_end_year": args.train_end_year,
        "val_year": args.val_year,
        "total_rows": len(split_rows),
        "train_rows": split_counts.get("train", 0),
        "val_rows": split_counts.get("val", 0),
        "test_rows": split_counts.get("test", 0),
        "year_counts": dict(sorted(year_counts.items())),
    }
    args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.resolve().open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Year split build complete")
    print(f"  rows: {len(split_rows)}")
    print(
        "  split sizes:"
        f" train={split_counts.get('train', 0)}"
        f" val={split_counts.get('val', 0)}"
        f" test={split_counts.get('test', 0)}"
    )
    print(f"  output: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
