#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from dj30_qc_utils import detect_filename_metadata, iter_files, write_csv, write_json


DATASET_SUFFIXES = {
    "A1": [".json"],
    "A2": [".html", ".htm"],
    "A3": [".mp3", ".wav", ".m4a"],
    "A4": [".csv"],
    "D": [".csv"],
}


def scan_dataset(dataset_name: str, root: Path) -> list[dict]:
    rows = []
    files = iter_files(root, DATASET_SUFFIXES[dataset_name])
    for file_path in files:
        meta = detect_filename_metadata(file_path)
        rows.append(
            {
                "dataset": dataset_name,
                "path": str(file_path.resolve()),
                "stem": file_path.stem,
                "suffix": file_path.suffix.lower(),
                "size_bytes": file_path.stat().st_size,
                "ticker": meta["ticker"],
                "year": meta["year"],
                "quarter": meta["quarter"],
                "event_key": meta["event_key"],
            }
        )
    return rows


def summarise_manifest(rows: list[dict]) -> dict:
    counts_by_dataset = defaultdict(int)
    event_keys_by_dataset = defaultdict(set)
    tickers_by_dataset = defaultdict(set)

    for row in rows:
        dataset = row["dataset"]
        counts_by_dataset[dataset] += 1
        if row["event_key"]:
            event_keys_by_dataset[dataset].add(row["event_key"])
        if row["ticker"]:
            tickers_by_dataset[dataset].add(row["ticker"])

    pairwise_overlap = {}
    ordered_datasets = ["A1", "A2", "A3", "A4"]
    for left_idx, left in enumerate(ordered_datasets):
        for right in ordered_datasets[left_idx + 1 :]:
            pairwise_overlap[f"{left}_{right}"] = len(
                event_keys_by_dataset[left] & event_keys_by_dataset[right]
            )

    full_overlap = set.intersection(
        *(event_keys_by_dataset[name] for name in ordered_datasets if event_keys_by_dataset[name])
    ) if all(event_keys_by_dataset[name] for name in ordered_datasets) else set()

    return {
        "counts_by_dataset": dict(counts_by_dataset),
        "parsed_event_keys_by_dataset": {
            dataset: len(keys) for dataset, keys in event_keys_by_dataset.items()
        },
        "tickers_by_dataset": {
            dataset: sorted(tickers) for dataset, tickers in tickers_by_dataset.items()
        },
        "pairwise_event_key_overlap": pairwise_overlap,
        "full_A1_A2_A3_A4_event_key_overlap": len(full_overlap),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan the DJ30 package and build a file-level manifest with heuristic event keys."
    )
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument(
        "--a4-path",
        type=Path,
        required=True,
        help="Either the A4 directory or a single A4 CSV file.",
    )
    parser.add_argument("--d-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qc"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows = []
    rows.extend(scan_dataset("A1", args.a1_dir))
    rows.extend(scan_dataset("A2", args.a2_dir))
    rows.extend(scan_dataset("A3", args.a3_dir))
    rows.extend(scan_dataset("A4", args.a4_path))
    rows.extend(scan_dataset("D", args.d_dir))

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "file_manifest.csv", rows)
    write_json(output_dir / "file_manifest_summary.json", summarise_manifest(rows))

    print(f"Wrote {len(rows)} file manifest rows to {output_dir / 'file_manifest.csv'}")


if __name__ == "__main__":
    main()

