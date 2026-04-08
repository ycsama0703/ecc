#!/usr/bin/env python3
"""
Build a strict chronological train/val/test split from the processed panel.

Current policy:
    - sort by event_date, then ticker, then event_id
    - assign whole event dates to the same split
    - target row ratios approximately, while preserving chronology
"""

from __future__ import annotations

import argparse
import csv
import json
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


def sort_panel_rows(rows: list[dict]) -> list[dict]:
    return sorted(
        rows,
        key=lambda row: (
            row["event_date"],
            row.get("ticker", ""),
            row.get("event_id", ""),
        ),
    )


def group_rows_by_date(rows: list[dict]) -> list[tuple[str, list[dict]]]:
    grouped: list[tuple[str, list[dict]]] = []
    current_date = None
    current_rows: list[dict] = []

    for row in rows:
        event_date = row["event_date"]
        if current_date is None:
            current_date = event_date
        if event_date != current_date:
            grouped.append((current_date, current_rows))
            current_date = event_date
            current_rows = []
        current_rows.append(row)

    if current_rows:
        grouped.append((current_date, current_rows))
    return grouped


def choose_split_label(
    assigned_rows: int,
    total_rows: int,
    train_target: int,
    val_target: int,
) -> str:
    if assigned_rows < train_target:
        return "train"
    if assigned_rows < train_target + val_target:
        return "val"
    return "test"


def build_split_rows(
    panel_rows: list[dict],
    split_version: str,
    train_ratio: float,
    val_ratio: float,
) -> tuple[list[dict], dict]:
    if not panel_rows:
        raise ValueError("Panel is empty; cannot build time split.")

    sorted_rows = sort_panel_rows(panel_rows)
    grouped_by_date = group_rows_by_date(sorted_rows)

    total_rows = len(sorted_rows)
    train_target = max(1, int(round(total_rows * train_ratio)))
    val_target = max(1, int(round(total_rows * val_ratio)))

    assigned_rows = 0
    split_rows: list[dict] = []
    boundary_dates = {"train_end_date": None, "val_end_date": None, "test_start_date": None}

    for event_date, date_rows in grouped_by_date:
        split_label = choose_split_label(
            assigned_rows=assigned_rows,
            total_rows=total_rows,
            train_target=train_target,
            val_target=val_target,
        )

        if split_label == "train":
            boundary_dates["train_end_date"] = event_date
        elif split_label == "val":
            if boundary_dates["test_start_date"] is None:
                boundary_dates["val_end_date"] = event_date
        else:
            if boundary_dates["test_start_date"] is None:
                boundary_dates["test_start_date"] = event_date

        for row in date_rows:
            split_rows.append(
                {
                    "event_id": row["event_id"],
                    "ticker": row.get("ticker", ""),
                    "event_date": row["event_date"],
                    "split_version": split_version,
                    "split_label": split_label,
                    "train_flag": 1 if split_label == "train" else 0,
                    "val_flag": 1 if split_label == "val" else 0,
                    "test_flag": 1 if split_label == "test" else 0,
                }
            )
        assigned_rows += len(date_rows)

    split_counts = Counter(row["split_label"] for row in split_rows)
    date_counts = Counter()
    for event_date, date_rows in grouped_by_date:
        split_for_date = next(
            row["split_label"] for row in split_rows if row["event_date"] == event_date
        )
        date_counts[split_for_date] += 1

    summary = {
        "split_version": split_version,
        "total_rows": total_rows,
        "total_dates": len(grouped_by_date),
        "train_ratio_target": train_ratio,
        "val_ratio_target": val_ratio,
        "test_ratio_target": round(1.0 - train_ratio - val_ratio, 8),
        "train_rows": int(split_counts.get("train", 0)),
        "val_rows": int(split_counts.get("val", 0)),
        "test_rows": int(split_counts.get("test", 0)),
        "train_dates": int(date_counts.get("train", 0)),
        "val_dates": int(date_counts.get("val", 0)),
        "test_dates": int(date_counts.get("test", 0)),
        "min_event_date": sorted_rows[0]["event_date"],
        "max_event_date": sorted_rows[-1]["event_date"],
        **boundary_dates,
    }
    return split_rows, summary


def validate_split_rows(split_rows: list[dict]) -> None:
    event_ids = [row["event_id"] for row in split_rows]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("Duplicate event_id detected in split output.")

    for row in split_rows:
        flag_sum = int(row["train_flag"]) + int(row["val_flag"]) + int(row["test_flag"])
        if flag_sum != 1:
            raise ValueError(f"Invalid split flags for event_id={row['event_id']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a strict chronological split file.")
    parser.add_argument(
        "--panel-csv",
        type=Path,
        default=Path("data/processed/panel/processed_panel.csv"),
        help="Processed panel CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/splits/time_split_v1.csv"),
        help="Output split CSV.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("data/splits/time_split_v1_summary.json"),
        help="Output summary JSON.",
    )
    parser.add_argument(
        "--split-version",
        type=str,
        default="v1",
        help="Split version tag.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Target train ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Target validation ratio.",
    )
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive.")
    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be less than 1.")

    panel_rows = read_csv(args.panel_csv.resolve())
    split_rows, summary = build_split_rows(
        panel_rows=panel_rows,
        split_version=args.split_version,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    validate_split_rows(split_rows)

    output_csv = args.output_csv.resolve()
    summary_json = args.summary_json.resolve()
    write_csv(output_csv, split_rows)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Time split build complete")
    print(f"  rows: {summary['total_rows']}")
    print(
        "  split sizes: "
        f"train={summary['train_rows']}  val={summary['val_rows']}  test={summary['test_rows']}"
    )
    print(
        "  split dates: "
        f"train_end={summary['train_end_date']}  "
        f"val_end={summary['val_end_date']}  "
        f"test_start={summary['test_start_date']}"
    )
    print(f"  output: {output_csv}")


if __name__ == "__main__":
    main()
