#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from build_sentence_aligned_sequence_features import (
    load_a1_sentence_units,
    load_a4_grouped_rows,
    load_panel_event_keys,
    match_rows_to_sentence_units,
)
from dj30_qc_utils import load_csv_rows, safe_float, write_csv, write_json


ROLE_SUBSETS = {
    "question": {"question"},
    "answer": {"answer"},
    "presenter": {"presenter"},
    "qa": {"question", "answer"},
}

META_FIELDS = {
    "event_key",
    "ticker",
    "year",
    "quarter",
    "sentence_id",
    "row_index_within_event",
    "source_file",
    "source_stem",
    "audio_path",
    "processed_audio_path",
    "start_sec",
    "end_sec",
    "duration_sec",
    "match_score",
    "overall_TFIDF",
    "strict_pass",
    "broad_pass",
    "overlap_warn_flag",
    "hard_fail_flag",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build role-aware event-level acoustic aggregates from sentence-level aligned audio."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--sentence-audio-csv", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--a4-row-qc-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/role_aware_aligned_audio_real"),
    )
    parser.add_argument("--winsorize-quantile", type=float, default=0.01)
    return parser.parse_args()


def infer_feature_columns(rows: list[dict[str, str]]) -> list[str]:
    names = set()
    for row in rows:
        for key, value in row.items():
            if key in META_FIELDS:
                continue
            if safe_float(value) is not None:
                names.add(key)
    return sorted(names)


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = min(max(q, 0.0), 1.0) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def winsorize(values: list[float], q: float) -> list[float]:
    if not values:
        return []
    if q <= 0.0:
        return list(values)
    ordered = sorted(values)
    low = quantile(ordered, q)
    high = quantile(ordered, 1.0 - q)
    return [min(max(value, low), high) for value in values]


def group_sentence_audio_rows(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped = defaultdict(list)
    for row in load_csv_rows(path.resolve()):
        event_key = row.get("event_key", "")
        if event_key:
            grouped[event_key].append(row)
    for rows in grouped.values():
        rows.sort(
            key=lambda row: (
                safe_float(row.get("row_index_within_event")) or 0.0,
                safe_float(row.get("start_sec")) or 0.0,
            )
        )
    return grouped


def build_role_lookup(
    panel_csv: Path,
    a1_dir: Path,
    a4_dir: Path,
    a4_row_qc_csv: Path,
) -> dict[tuple[str, str], dict[str, str | float]]:
    panel_rows = load_panel_event_keys(panel_csv.resolve())
    event_keys = {row["event_key"] for row in panel_rows}
    units_lookup = load_a1_sentence_units(a1_dir.resolve(), event_keys)
    a4_lookup = load_a4_grouped_rows(a4_dir.resolve(), a4_row_qc_csv.resolve(), event_keys)

    role_lookup = {}
    for event_key in sorted(event_keys):
        matched = match_rows_to_sentence_units(a4_lookup.get(event_key, []), units_lookup.get(event_key, []))
        for row in matched:
            sentence_id = row.get("sentence_id", "")
            if not sentence_id:
                continue
            role_lookup[(event_key, sentence_id)] = {
                "role": row.get("_role", "unknown"),
                "align_score": row.get("_align_score", 0.0),
            }
    return role_lookup


def aggregate_subset(
    rows: list[dict[str, str]],
    feature_columns: list[str],
    subset_name: str,
    winsor_q: float,
) -> dict[str, str]:
    out = {
        f"{subset_name}_row_count": str(len(rows)),
        f"{subset_name}_duration_sum_sec": f"{sum(max(safe_float(row.get('duration_sec')) or 0.0, 0.0) for row in rows):.6f}",
    }
    if rows:
        align_scores = [safe_float(row.get("_align_score")) for row in rows]
        align_scores = [score for score in align_scores if score is not None and math.isfinite(score)]
        out[f"{subset_name}_align_score_mean"] = (
            f"{statistics.mean(align_scores):.6f}" if align_scores else ""
        )
    else:
        out[f"{subset_name}_align_score_mean"] = ""

    for feature in feature_columns:
        values = [safe_float(row.get(feature)) for row in rows]
        clean = [value for value in values if value is not None and math.isfinite(value)]
        out[f"{subset_name}_{feature}_winsor_mean"] = (
            f"{statistics.mean(winsorize(clean, winsor_q)):.6f}" if clean else ""
        )
    return out


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sentence_rows_by_event = group_sentence_audio_rows(args.sentence_audio_csv)
    feature_columns = infer_feature_columns([row for rows in sentence_rows_by_event.values() for row in rows[:1]])
    role_lookup = build_role_lookup(args.panel_csv, args.a1_dir, args.a4_dir, args.a4_row_qc_csv)

    labeled_rows = []
    event_rows = []
    role_counts = defaultdict(int)

    for event_key in sorted(sentence_rows_by_event):
        event_rows_raw = []
        for row in sentence_rows_by_event[event_key]:
            role_payload = role_lookup.get((event_key, row.get("sentence_id", "")), {})
            labeled = dict(row)
            labeled["_role"] = str(role_payload.get("role", "unknown"))
            labeled["_align_score"] = str(role_payload.get("align_score", 0.0))
            labeled_rows.append(labeled)
            event_rows_raw.append(labeled)
            role_counts[labeled["_role"]] += 1

        if not event_rows_raw:
            continue

        head = event_rows_raw[0]
        out_row = {
            "event_key": event_key,
            "ticker": head.get("ticker", ""),
            "year": head.get("year", ""),
            "quarter": head.get("quarter", ""),
            "aligned_audio_sentence_count": str(len(event_rows_raw)),
            "mapped_row_count": str(sum(1 for row in event_rows_raw if row["_role"] != "unknown")),
            "mapped_row_share": f"{sum(1 for row in event_rows_raw if row['_role'] != 'unknown') / max(len(event_rows_raw), 1):.6f}",
        }
        for subset_name, role_set in ROLE_SUBSETS.items():
            subset_rows = [row for row in event_rows_raw if row["_role"] in role_set]
            out_row.update(aggregate_subset(subset_rows, feature_columns, subset_name, args.winsorize_quantile))
        event_rows.append(out_row)

    summary = {
        "num_events": len(event_rows),
        "num_sentence_rows": len(labeled_rows),
        "feature_column_count": len(feature_columns),
        "winsorize_quantile": args.winsorize_quantile,
        "role_counts": dict(sorted(role_counts.items())),
        "mapped_row_share_mean": round(
            statistics.mean(
                safe_float(row.get("mapped_row_share")) or 0.0 for row in event_rows
            ),
            6,
        )
        if event_rows
        else 0.0,
    }

    write_csv(output_dir / "sentence_role_labeled_acoustic_features.csv", labeled_rows)
    write_csv(output_dir / "event_role_aware_aligned_acoustic_features.csv", event_rows)
    write_json(output_dir / "role_aware_aligned_acoustic_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
