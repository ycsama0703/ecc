#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VALID_SUBSETS = {"question", "answer", "presenter", "qa"}
META_FIELDS = ["event_key", "ticker", "year", "quarter"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a benchmark-ready aligned-audio view for one role-aware subset."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--subset", choices=sorted(VALID_SUBSETS), required=True)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to a file next to the input using the subset name.",
    )
    return parser.parse_args()


def default_output_path(input_csv: Path, subset: str) -> Path:
    return input_csv.resolve().parent / f"event_aligned_acoustic_{subset}_view.csv"


def main() -> None:
    args = parse_args()
    output_csv = args.output_csv.resolve() if args.output_csv is not None else default_output_path(args.input_csv, args.subset)

    with args.input_csv.resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    feature_columns = [name for name in fieldnames if name.startswith(f"{args.subset}_") and name.endswith("_winsor_mean")]
    output_rows = []
    nonzero_events = 0
    for row in rows:
        out_row = {field: row.get(field, "") for field in META_FIELDS}
        out_row["aligned_audio_sentence_count"] = row.get(f"{args.subset}_row_count", "")
        out_row["aligned_audio_duration_sum_sec"] = row.get(f"{args.subset}_duration_sum_sec", "")
        for feature_name in feature_columns:
            renamed = feature_name[len(args.subset) + 1 :]
            out_row[renamed] = row.get(feature_name, "")
        try:
            if float(out_row["aligned_audio_sentence_count"] or 0) > 0:
                nonzero_events += 1
        except ValueError:
            pass
        output_rows.append(out_row)

    output_fieldnames = META_FIELDS + ["aligned_audio_sentence_count", "aligned_audio_duration_sum_sec"] + [
        feature_name[len(args.subset) + 1 :] for feature_name in feature_columns
    ]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "subset": args.subset,
        "input_csv": str(args.input_csv.resolve()),
        "output_csv": str(output_csv),
        "num_events": len(output_rows),
        "nonzero_events": nonzero_events,
        "feature_column_count": len(feature_columns),
    }
    summary_path = output_csv.with_name(output_csv.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
