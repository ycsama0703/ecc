#!/usr/bin/env python3
"""
Filter a processed panel into a cleaner experiment subset.

Current intended use:
    Experiment C = after_hours + exclude html_integrity_flag=fail
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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


def regime_label(row: dict) -> str:
    hour_text = row.get("scheduled_hour_et")
    hour = float(hour_text) if hour_text not in ("", None) else 0.0
    if hour < 9.5:
        return "pre_market"
    if hour < 16.0:
        return "market_hours"
    return "after_hours"


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter a processed panel into a cleaner subset.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--include-regimes", type=str, default="after_hours")
    parser.add_argument("--exclude-html-flags", type=str, default="fail")
    args = parser.parse_args()

    include_regimes = {item.strip() for item in args.include_regimes.split(",") if item.strip()}
    exclude_html_flags = {item.strip().lower() for item in args.exclude_html_flags.split(",") if item.strip()}

    rows = read_csv(args.input_csv.resolve())
    kept = []
    dropped = []
    for row in rows:
        html_flag = (row.get("html_integrity_flag") or "").strip().lower()
        regime = regime_label(row)
        reasons = []
        if regime not in include_regimes:
            reasons.append(f"regime={regime}")
        if html_flag in exclude_html_flags:
            reasons.append(f"html_flag={html_flag}")
        if reasons:
            dropped.append({**row, "drop_reasons": ",".join(reasons)})
            continue
        kept.append(row)

    write_csv(args.output_csv.resolve(), kept)
    dropped_csv = args.output_csv.resolve().with_name(args.output_csv.stem + "_dropped.csv")
    write_csv(dropped_csv, dropped)

    summary = {
        "input_csv": str(args.input_csv.resolve()),
        "output_csv": str(args.output_csv.resolve()),
        "summary_json": str(args.summary_json.resolve()),
        "include_regimes": sorted(include_regimes),
        "exclude_html_flags": sorted(exclude_html_flags),
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "dropped_rows": len(dropped),
        "kept_tickers": len({row["ticker"] for row in kept}),
        "html_integrity_flag_counts_kept": dict(Counter((row.get("html_integrity_flag") or "") for row in kept)),
        "regime_counts_kept": dict(Counter(regime_label(row) for row in kept)),
        "drop_reason_counts": dict(
            Counter(reason for row in dropped for reason in row["drop_reasons"].split(",") if reason)
        ),
        "dropped_csv": str(dropped_csv),
    }
    args.summary_json.resolve().parent.mkdir(parents=True, exist_ok=True)
    with args.summary_json.resolve().open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Filtered panel complete")
    print(f"  kept rows: {len(kept)}")
    print(f"  dropped rows: {len(dropped)}")
    print(f"  output: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
