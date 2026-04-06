#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


META_FIELDS = {"event_key", "ticker", "year", "quarter"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two event-level aligned-audio views into one benchmark-ready table."
    )
    parser.add_argument("--base-csv", type=Path, required=True)
    parser.add_argument("--extra-csv", type=Path, required=True)
    parser.add_argument("--extra-prefix", default="extra_")
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {row["event_key"]: row for row in reader if row.get("event_key")}
        fieldnames = reader.fieldnames or []
    return fieldnames, rows


def main() -> None:
    args = parse_args()
    base_fields, base_rows = load_rows(args.base_csv)
    extra_fields, extra_rows = load_rows(args.extra_csv)

    merged_rows = []
    extra_feature_names = [field for field in extra_fields if field not in META_FIELDS]
    for event_key, base_row in sorted(base_rows.items()):
        extra_row = extra_rows.get(event_key)
        if extra_row is None:
            continue
        out_row = dict(base_row)
        for field in extra_feature_names:
            out_row[f"{args.extra_prefix}{field}"] = extra_row.get(field, "")
        merged_rows.append(out_row)

    output_fields = list(base_fields) + [f"{args.extra_prefix}{field}" for field in extra_feature_names]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.resolve().open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(merged_rows)

    summary = {
        "base_csv": str(args.base_csv.resolve()),
        "extra_csv": str(args.extra_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "num_rows": len(merged_rows),
        "base_feature_count": len([field for field in base_fields if field not in META_FIELDS]),
        "extra_feature_count": len(extra_feature_names),
        "extra_prefix": args.extra_prefix,
    }
    summary_path = args.output_csv.with_name(args.output_csv.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
