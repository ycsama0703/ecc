#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dj30_qc_utils import (
    build_event_path_lookup,
    detect_filename_metadata,
    iter_files,
    load_csv_rows,
    load_json,
    normalize_text,
    text_similarity,
    token_f1,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedily map A4 timed sentences onto A1 component-level transcript rows."
    )
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/mapping"))
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--event-key", default="")
    parser.add_argument("--forward-window", type=int, default=6)
    parser.add_argument("--include-text", action="store_true")
    return parser.parse_args()


def load_a1_components(path: Path) -> list[dict]:
    payload = load_json(path)
    components = payload.get("components", [])

    def order_key(component: dict) -> tuple[int, int]:
        raw = component.get("componentorder")
        try:
            return (0, int(raw))
        except Exception:
            return (1, 0)

    cleaned = []
    for component in sorted(components, key=order_key):
        text = normalize_text(component.get("text", ""))
        if not text:
            continue
        cleaned.append(
            {
                "componentid": component.get("componentid", ""),
                "componentorder": component.get("componentorder", ""),
                "componenttypename": component.get("componenttypename", ""),
                "personname": component.get("personname", ""),
                "companyofperson": component.get("companyofperson", ""),
                "text": component.get("text", ""),
                "normalized_text": text.lower(),
            }
        )
    return cleaned


def score_component(official_text: str, component: dict) -> tuple[float, bool]:
    sentence = normalize_text(official_text).lower()
    component_text = component["normalized_text"]
    if not sentence or not component_text:
        return 0.0, False

    contains = sentence in component_text
    score = max(
        text_similarity(sentence, component_text),
        token_f1(sentence, component_text),
    )
    if contains:
        score = max(score, 0.99)
    return score, contains


def confidence_label(score: float, contains: bool) -> str:
    if contains or score >= 0.97:
        return "high"
    if score >= 0.78:
        return "medium"
    if score >= 0.55:
        return "low"
    return "unresolved"


def map_event(
    a1_path: Path,
    a4_path: Path,
    forward_window: int,
    include_text: bool,
) -> tuple[list[dict], dict]:
    components = load_a1_components(a1_path)
    a4_rows = load_csv_rows(a4_path)
    mapping_rows = []
    current_idx = 0

    for row_idx, row in enumerate(a4_rows, start=1):
        official_text = normalize_text(row.get("official_text", ""))
        if not official_text:
            continue

        start_idx = max(0, current_idx - 1)
        end_idx = min(len(components), current_idx + forward_window + 1)
        candidates = list(range(start_idx, end_idx)) or list(range(len(components)))

        best_idx = -1
        best_score = -1.0
        best_contains = False
        for component_idx in candidates:
            score, contains = score_component(official_text, components[component_idx])
            if component_idx == current_idx:
                score += 0.005
            if score > best_score:
                best_idx = component_idx
                best_score = score
                best_contains = contains

        component = components[best_idx] if 0 <= best_idx < len(components) else {}
        if best_idx >= current_idx:
            current_idx = best_idx

        record = {
            "event_key": detect_filename_metadata(a1_path)["event_key"],
            "a1_path": str(a1_path.resolve()),
            "a4_path": str(a4_path.resolve()),
            "a4_row_index": row_idx,
            "sentence_id": row.get("sentence_id", ""),
            "official_text_length": len(official_text),
            "componentid": component.get("componentid", ""),
            "componentorder": component.get("componentorder", ""),
            "componenttypename": component.get("componenttypename", ""),
            "personname": component.get("personname", ""),
            "companyofperson": component.get("companyofperson", ""),
            "component_text_length": len(component.get("text", "")),
            "mapping_score": round(best_score, 6) if best_score >= 0 else "",
            "contains_flag": best_contains,
            "mapping_confidence": confidence_label(best_score, best_contains),
        }
        if include_text:
            record["official_text"] = official_text
            record["component_text"] = component.get("text", "")
        mapping_rows.append(record)

    summary = {
        "event_key": detect_filename_metadata(a1_path)["event_key"],
        "num_a1_components": len(components),
        "num_a4_rows": len(a4_rows),
        "num_mapped_rows": len(mapping_rows),
        "confidence_counts": dict(Counter(row["mapping_confidence"] for row in mapping_rows)),
    }
    return mapping_rows, summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    a1_lookup, a1_duplicates = build_event_path_lookup(iter_files(args.a1_dir.resolve(), [".json"]))
    a4_lookup, _ = build_event_path_lookup(iter_files(args.a4_dir.resolve(), [".csv"]))

    common_event_keys = sorted(set(a1_lookup) & set(a4_lookup))
    if args.event_key:
        common_event_keys = [key for key in common_event_keys if key == args.event_key]
    if args.max_events and args.max_events > 0:
        common_event_keys = common_event_keys[: args.max_events]

    mapping_rows = []
    summaries = []
    for event_key in common_event_keys:
        rows, summary = map_event(
            a1_lookup[event_key],
            a4_lookup[event_key],
            forward_window=args.forward_window,
            include_text=args.include_text,
        )
        mapping_rows.extend(rows)
        summaries.append(summary)

    write_csv(output_dir / "a1_a4_sentence_component_map.csv", mapping_rows)
    write_json(
        output_dir / "a1_a4_sentence_component_map_summary.json",
        {
            "num_common_events": len(common_event_keys),
            "events": summaries,
            "aggregate_confidence_counts": dict(
                Counter(row["mapping_confidence"] for row in mapping_rows)
            ),
        },
    )

    print(
        json.dumps(
        {
            "num_common_events": len(common_event_keys),
            "num_mapping_rows": len(mapping_rows),
            "a1_duplicate_event_keys": a1_duplicates,
            "aggregate_confidence_counts": dict(
                Counter(row["mapping_confidence"] for row in mapping_rows)
            ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
