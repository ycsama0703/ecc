#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from dj30_qc_utils import (
    build_event_path_lookup,
    iter_files,
    load_csv_rows,
    load_json,
    normalize_event_key_text,
    normalize_text,
    safe_float,
    token_f1,
    write_csv,
    write_json,
)


LEXICONS = {
    "guidance": {
        "guidance",
        "outlook",
        "forecast",
        "forecasts",
        "expect",
        "expects",
        "expected",
        "anticipate",
        "plan",
        "plans",
    },
    "uncertainty": {
        "uncertain",
        "uncertainty",
        "risk",
        "risks",
        "headwind",
        "headwinds",
        "pressure",
        "challenging",
        "volatile",
        "volatility",
        "cautious",
    },
    "negative": {
        "weak",
        "weaker",
        "decline",
        "declined",
        "soft",
        "pressure",
        "risk",
        "lower",
        "challenge",
        "challenging",
    },
}


ROLE_LABELS = ["question", "answer", "presenter", "operator", "unknown"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fixed-length role-aware sequence features from A4 rows and A1 component types."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--a4-row-qc-csv", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=12)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/sequence_real")
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def classify_role(component_type: str) -> str:
    value = normalize_text(component_type).lower()
    if value == "question":
        return "question"
    if value == "answer":
        return "answer"
    if value == "presenter speech":
        return "presenter"
    if "operator" in value:
        return "operator"
    return "unknown"


def bool_from_text(value: str) -> bool:
    return normalize_text(value).lower() in {"true", "1", "yes"}


def load_panel_event_keys(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in load_csv_rows(path.resolve()):
        event_key = normalize_event_key_text(row.get("event_key", ""))
        if not event_key:
            continue
        row["event_key"] = event_key
        rows.append(row)
    return rows


def load_a1_components(a1_dir: Path, event_keys: set[str]) -> dict[str, list[dict]]:
    path_lookup, _ = build_event_path_lookup(iter_files(a1_dir.resolve(), [".json"]))
    output = {}
    for event_key in sorted(event_keys):
        path = path_lookup.get(event_key)
        if path is None:
            continue
        payload = load_json(path)
        components = []
        for idx, component in enumerate(payload.get("components", [])):
            text = normalize_text(component.get("text", ""))
            if not text:
                continue
            comp_type = normalize_text(component.get("componenttypename", ""))
            norm_text = normalize_text(text).lower()
            components.append(
                {
                    "component_index": idx,
                    "component_type": comp_type,
                    "role": classify_role(comp_type),
                    "text": text,
                    "norm_text": norm_text,
                }
            )
        output[event_key] = components
    return output


def load_a4_grouped_rows(
    a4_dir: Path, qc_path: Path, event_keys: set[str]
) -> dict[str, list[dict]]:
    qc_lookup = defaultdict(dict)
    for row in load_csv_rows(qc_path.resolve()):
        event_key = normalize_event_key_text(row.get("event_id", ""))
        if event_key not in event_keys:
            continue
        sentence_id = normalize_text(row.get("sentence_id", ""))
        if sentence_id:
            qc_lookup[event_key][sentence_id] = row

    raw_lookup, _ = build_event_path_lookup(iter_files(a4_dir.resolve(), [".csv"]))
    grouped = defaultdict(list)
    for event_key in sorted(event_keys):
        raw_path = raw_lookup.get(event_key)
        if raw_path is None:
            continue
        for raw_row in load_csv_rows(raw_path):
            sentence_id = normalize_text(
                raw_row.get("sentence_id", "") or raw_row.get("\ufeffsentence_id", "")
            )
            qc_row = qc_lookup[event_key].get(sentence_id, {})
            row = {
                "sentence_id": sentence_id,
                "official_text": raw_row.get("official_text", ""),
                "match_score": raw_row.get("match_score", ""),
                "start_sec": raw_row.get("start_sec", ""),
                "end_sec": raw_row.get("end_sec", ""),
                "strict_pass": qc_row.get("strict_pass", ""),
                "broad_pass": qc_row.get("broad_pass", ""),
            }
            row["_event_key"] = event_key
            row["_start_sec"] = safe_float(row.get("start_sec"))
            row["_end_sec"] = safe_float(row.get("end_sec"))
            grouped[event_key].append(row)

    for event_key in list(grouped):
        grouped[event_key].sort(key=lambda item: (item["_start_sec"] or 0.0, item["_end_sec"] or 0.0))
    return grouped


def component_score(sentence_text: str, component: dict) -> float:
    sentence_norm = normalize_text(sentence_text).lower()
    if not sentence_norm or not component["norm_text"]:
        return 0.0
    if sentence_norm in component["norm_text"]:
        return 1.0
    if component["norm_text"] in sentence_norm:
        return 0.85
    return token_f1(sentence_norm, component["norm_text"])


def match_rows_to_components(rows: list[dict], components: list[dict]) -> list[dict]:
    if not components:
        matched = []
        for row in rows:
            matched.append({**row, "_role": "unknown", "_component_match_score": 0.0})
        return matched

    matched = []
    pointer = 0
    last_good_idx = 0

    for row in rows:
        sentence_text = normalize_text(row.get("official_text", ""))
        search_start = max(0, pointer - 1)
        search_end = min(len(components), max(pointer + 8, search_start + 1))
        best_idx = None
        best_score = -1.0

        for idx in range(search_start, search_end):
            score = component_score(sentence_text, components[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_score < 0.12:
            fallback_end = min(len(components), last_good_idx + 30)
            for idx in range(max(0, last_good_idx - 1), fallback_end):
                score = component_score(sentence_text, components[idx])
                if score > best_score:
                    best_score = score
                    best_idx = idx

        role = "unknown"
        if best_idx is not None and best_score >= 0.12:
            pointer = max(pointer, best_idx)
            last_good_idx = best_idx
            role = components[best_idx]["role"]

        matched.append(
            {
                **row,
                "_role": role,
                "_component_match_score": round(float(best_score), 6) if best_score > -1 else 0.0,
            }
        )

    return matched


def build_sequence_features_for_subset(rows: list[dict], num_bins: int, subset_prefix: str) -> dict[str, str]:
    starts = [row["_start_sec"] for row in rows if row["_start_sec"] is not None]
    ends = [row["_end_sec"] for row in rows if row["_end_sec"] is not None]
    if not rows or not starts or not ends:
        features = {
            f"{subset_prefix}_row_count": "0",
            f"{subset_prefix}_mapped_row_share": "0.000000",
            f"{subset_prefix}_high_conf_map_share": "0.000000",
            f"{subset_prefix}_span_sec": "",
        }
        for bin_idx in range(num_bins):
            for name in [
                "segment_count",
                "question_share",
                "answer_share",
                "presenter_share",
                "operator_share",
                "unknown_share",
                "duration_mean_sec",
                "match_score_mean",
                "component_map_score_mean",
                "uncertainty_rate_all",
                "guidance_rate_answer",
                "negative_rate_answer",
                "question_mark_rate_question",
            ]:
                features[f"{subset_prefix}_bin_{bin_idx}_{name}"] = ""
        return features

    span_start = min(starts)
    span_end = max(ends)
    span_sec = max(span_end - span_start, 1e-6)

    bins = []
    for _ in range(num_bins):
        bins.append(
            {
                "segment_count": 0,
                "duration_sum": 0.0,
                "match_score_sum": 0.0,
                "match_score_count": 0,
                "component_map_score_sum": 0.0,
                "role_counts": defaultdict(int),
                "all_tokens": [],
                "answer_tokens": [],
                "question_question_marks": 0,
                "question_rows": 0,
            }
        )

    matched_rows = 0
    high_conf_rows = 0
    for row in rows:
        start_sec = row["_start_sec"]
        end_sec = row["_end_sec"]
        if start_sec is None or end_sec is None:
            continue
        mid_sec = (start_sec + end_sec) / 2.0
        rel_pos = (mid_sec - span_start) / span_sec
        rel_pos = min(max(rel_pos, 0.0), 0.999999)
        bin_idx = min(num_bins - 1, int(rel_pos * num_bins))
        bucket = bins[bin_idx]
        bucket["segment_count"] += 1
        bucket["duration_sum"] += max(end_sec - start_sec, 0.0)
        role = row.get("_role", "unknown")
        bucket["role_counts"][role] += 1

        sentence_text = normalize_text(row.get("official_text", ""))
        tokens = tokenize(sentence_text)
        bucket["all_tokens"].extend(tokens)
        if role == "answer":
            bucket["answer_tokens"].extend(tokens)
        if role == "question":
            bucket["question_rows"] += 1
            bucket["question_question_marks"] += sentence_text.count("?")

        match_score = safe_float(row.get("match_score"))
        if match_score is not None:
            bucket["match_score_sum"] += match_score
            bucket["match_score_count"] += 1

        comp_score = safe_float(row.get("_component_match_score"))
        if comp_score is not None:
            bucket["component_map_score_sum"] += comp_score
            matched_rows += 1 if comp_score >= 0.12 else 0
            high_conf_rows += 1 if comp_score >= 0.7 else 0

    features = {
        f"{subset_prefix}_row_count": str(len(rows)),
        f"{subset_prefix}_mapped_row_share": f"{matched_rows / max(len(rows), 1):.6f}",
        f"{subset_prefix}_high_conf_map_share": f"{high_conf_rows / max(len(rows), 1):.6f}",
        f"{subset_prefix}_span_sec": f"{span_sec:.6f}",
    }

    for bin_idx, bucket in enumerate(bins):
        seg_count = bucket["segment_count"]
        features[f"{subset_prefix}_bin_{bin_idx}_segment_count"] = str(seg_count)
        for role in ROLE_LABELS:
            features[f"{subset_prefix}_bin_{bin_idx}_{role}_share"] = (
                f"{bucket['role_counts'][role] / max(seg_count, 1):.6f}" if seg_count else ""
            )
        features[f"{subset_prefix}_bin_{bin_idx}_duration_mean_sec"] = (
            f"{bucket['duration_sum'] / seg_count:.6f}" if seg_count else ""
        )
        features[f"{subset_prefix}_bin_{bin_idx}_match_score_mean"] = (
            f"{bucket['match_score_sum'] / bucket['match_score_count']:.6f}"
            if bucket["match_score_count"]
            else ""
        )
        features[f"{subset_prefix}_bin_{bin_idx}_component_map_score_mean"] = (
            f"{bucket['component_map_score_sum'] / seg_count:.6f}" if seg_count else ""
        )

        all_tokens = bucket["all_tokens"]
        answer_tokens = bucket["answer_tokens"]
        features[f"{subset_prefix}_bin_{bin_idx}_uncertainty_rate_all"] = (
            f"{1000.0 * sum(token in LEXICONS['uncertainty'] for token in all_tokens) / max(len(all_tokens), 1):.6f}"
            if all_tokens
            else ""
        )
        features[f"{subset_prefix}_bin_{bin_idx}_guidance_rate_answer"] = (
            f"{1000.0 * sum(token in LEXICONS['guidance'] for token in answer_tokens) / max(len(answer_tokens), 1):.6f}"
            if answer_tokens
            else ""
        )
        features[f"{subset_prefix}_bin_{bin_idx}_negative_rate_answer"] = (
            f"{1000.0 * sum(token in LEXICONS['negative'] for token in answer_tokens) / max(len(answer_tokens), 1):.6f}"
            if answer_tokens
            else ""
        )
        features[f"{subset_prefix}_bin_{bin_idx}_question_mark_rate_question"] = (
            f"{bucket['question_question_marks'] / max(bucket['question_rows'], 1):.6f}"
            if bucket["question_rows"]
            else ""
        )

    return features


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_rows = load_panel_event_keys(args.panel_csv)
    event_keys = {row["event_key"] for row in panel_rows}
    a1_components = load_a1_components(args.a1_dir, event_keys)
    a4_grouped = load_a4_grouped_rows(args.a4_dir, args.a4_row_qc_csv, event_keys)

    output_rows = []
    mapping_shares = {"strict": [], "broad": []}
    high_conf_shares = {"strict": [], "broad": []}

    for panel_row in panel_rows:
        event_key = panel_row["event_key"]
        components = a1_components.get(event_key, [])
        event_rows = a4_grouped.get(event_key, [])
        matched_rows = match_rows_to_components(event_rows, components)

        strict_rows = [row for row in matched_rows if bool_from_text(row.get("strict_pass", ""))]
        broad_rows = [row for row in matched_rows if bool_from_text(row.get("broad_pass", ""))]

        row = {
            "event_key": event_key,
            "ticker": panel_row.get("ticker", ""),
            "year": panel_row.get("year", ""),
            "quarter": panel_row.get("quarter", ""),
        }
        row.update(build_sequence_features_for_subset(strict_rows, args.num_bins, "strict"))
        row.update(build_sequence_features_for_subset(broad_rows, args.num_bins, "broad"))
        output_rows.append(row)

        for subset in ("strict", "broad"):
            mapping_value = safe_float(row.get(f"{subset}_mapped_row_share"))
            high_conf_value = safe_float(row.get(f"{subset}_high_conf_map_share"))
            if mapping_value is not None:
                mapping_shares[subset].append(mapping_value)
            if high_conf_value is not None:
                high_conf_shares[subset].append(high_conf_value)

    summary = {
        "num_events": len(output_rows),
        "num_bins": args.num_bins,
        "strict_mapped_row_share_mean": round(sum(mapping_shares["strict"]) / max(len(mapping_shares["strict"]), 1), 6),
        "broad_mapped_row_share_mean": round(sum(mapping_shares["broad"]) / max(len(mapping_shares["broad"]), 1), 6),
        "strict_high_conf_map_share_mean": round(sum(high_conf_shares["strict"]) / max(len(high_conf_shares["strict"]), 1), 6),
        "broad_high_conf_map_share_mean": round(sum(high_conf_shares["broad"]) / max(len(high_conf_shares["broad"]), 1), 6),
    }

    write_csv(output_dir / "role_aware_sequence_features.csv", output_rows)
    write_json(output_dir / "role_aware_sequence_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
