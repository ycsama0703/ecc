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
    normalize_event_key_text,
    normalize_text,
    safe_float,
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
        description="Build weakly supervised section and role sequence features directly from A4 text."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--a4-dir", type=Path, required=True)
    parser.add_argument("--a4-row-qc-csv", type=Path, required=True)
    parser.add_argument("--num-bins", type=int, default=12)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/sequence_weak_real")
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def bool_from_text(value: str) -> bool:
    return normalize_text(value).lower() in {"true", "1", "yes"}


def load_panel_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in load_csv_rows(path.resolve()):
        event_key = normalize_event_key_text(row.get("event_key", ""))
        if not event_key:
            continue
        row["event_key"] = event_key
        rows.append(row)
    return rows


def load_a4_grouped_rows(a4_dir: Path, qc_path: Path, event_keys: set[str]) -> dict[str, list[dict]]:
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
            row["_start_sec"] = safe_float(row.get("start_sec"))
            row["_end_sec"] = safe_float(row.get("end_sec"))
            grouped[event_key].append(row)
    for event_key in list(grouped):
        grouped[event_key].sort(key=lambda item: (item["_start_sec"] or 0.0, item["_end_sec"] or 0.0))
    return grouped


def is_operator_text(text: str) -> bool:
    value = normalize_text(text).lower()
    if not value:
        return False
    patterns = [
        "welcome to",
        "our first question",
        "the next question",
        "please go ahead",
        "that does conclude",
        "that concludes",
        "thank you for your participation",
        "you may disconnect",
        "conference call",
        "question-and-answer session",
        "open the call to questions",
        "open the call for questions",
    ]
    return any(pattern in value for pattern in patterns)


def find_qna_start(rows: list[dict]) -> int:
    for idx, row in enumerate(rows):
        value = normalize_text(row.get("official_text", "")).lower()
        if any(
            phrase in value
            for phrase in [
                "open the call to questions",
                "open the call for questions",
                "question-and-answer session",
                "our first question",
                "the next question",
            ]
        ):
            return idx
    return max(0, int(0.45 * len(rows)))


def questionish_score(text: str) -> float:
    value = normalize_text(text).lower()
    tokens = tokenize(value)
    score = 0.0
    score += 2.0 if "?" in value else 0.0
    score += sum(
        phrase in value
        for phrase in [
            "could you",
            "can you",
            "would you",
            "what",
            "how",
            "why",
            "when",
            "one for",
            "two questions",
            "help us understand",
            "talk about",
        ]
    )
    score += 0.2 * sum(token in {"what", "how", "why", "when"} for token in tokens)
    return score


def answerish_score(text: str) -> float:
    value = normalize_text(text).lower()
    score = 0.0
    starts = [
        "yes",
        "yeah",
        "sure",
        "so",
        "well",
        "thanks",
        "thank you",
        "let me",
        "i'll",
        "we",
        "our",
    ]
    score += 2.0 if any(value.startswith(prefix) for prefix in starts) else 0.0
    score += sum(
        phrase in value
        for phrase in [
            "let me start",
            "we think",
            "we expect",
            "we believe",
            "first of all",
            "as i said",
            "as we said",
        ]
    )
    return score


def assign_weak_roles(rows: list[dict]) -> list[dict]:
    if not rows:
        return []
    qna_start = find_qna_start(rows)
    assigned = [{**row, "_role": "presenter"} for row in rows]
    current_block = []

    def flush_block(block_rows: list[dict]):
        if not block_rows:
            return
        seen_question = False
        switched = False
        for row in block_rows:
            text = row["official_text"]
            q_score = questionish_score(text)
            a_score = answerish_score(text)
            role = "question"
            if switched or (seen_question and a_score >= q_score):
                role = "answer"
                switched = True
            else:
                role = "question"
                seen_question = True
            row["_role"] = role

    for idx, row in enumerate(assigned):
        text = row["official_text"]
        if is_operator_text(text):
            row["_role"] = "operator"
            if idx >= qna_start:
                flush_block(current_block)
                current_block = []
            continue
        if idx < qna_start:
            row["_role"] = "presenter"
        else:
            current_block.append(row)

    flush_block(current_block)
    return assigned


def build_sequence_features(rows: list[dict], num_bins: int, subset_prefix: str) -> dict[str, str]:
    starts = [row["_start_sec"] for row in rows if row["_start_sec"] is not None]
    ends = [row["_end_sec"] for row in rows if row["_end_sec"] is not None]
    if not rows or not starts or not ends:
        features = {
            f"{subset_prefix}_row_count": "0",
            f"{subset_prefix}_mapped_row_share": "1.000000",
            f"{subset_prefix}_high_conf_map_share": "1.000000",
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
                "role_counts": defaultdict(int),
                "all_tokens": [],
                "answer_tokens": [],
                "question_marks": 0,
                "question_rows": 0,
            }
        )

    for row in rows:
        start_sec = row["_start_sec"]
        end_sec = row["_end_sec"]
        if start_sec is None or end_sec is None:
            continue
        mid_sec = (start_sec + end_sec) / 2.0
        rel_pos = min(max((mid_sec - span_start) / span_sec, 0.0), 0.999999)
        bin_idx = min(num_bins - 1, int(rel_pos * num_bins))
        bucket = bins[bin_idx]
        bucket["segment_count"] += 1
        bucket["duration_sum"] += max(end_sec - start_sec, 0.0)
        role = row.get("_role", "unknown")
        bucket["role_counts"][role] += 1
        text = normalize_text(row.get("official_text", ""))
        tokens = tokenize(text)
        bucket["all_tokens"].extend(tokens)
        if role == "answer":
            bucket["answer_tokens"].extend(tokens)
        if role == "question":
            bucket["question_rows"] += 1
            bucket["question_marks"] += text.count("?")
        match_score = safe_float(row.get("match_score"))
        if match_score is not None:
            bucket["match_score_sum"] += match_score
            bucket["match_score_count"] += 1

    features = {
        f"{subset_prefix}_row_count": str(len(rows)),
        f"{subset_prefix}_mapped_row_share": "1.000000",
        f"{subset_prefix}_high_conf_map_share": "1.000000",
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
            f"{bucket['match_score_sum'] / bucket['match_score_count']:.6f}" if bucket["match_score_count"] else ""
        )
        features[f"{subset_prefix}_bin_{bin_idx}_component_map_score_mean"] = (
            "1.000000" if seg_count else ""
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
            f"{bucket['question_marks'] / max(bucket['question_rows'], 1):.6f}" if bucket["question_rows"] else ""
        )
    return features


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_rows = load_panel_rows(args.panel_csv)
    event_keys = {row["event_key"] for row in panel_rows}
    a4_lookup = load_a4_grouped_rows(args.a4_dir, args.a4_row_qc_csv, event_keys)

    rows = []
    for panel_row in panel_rows:
        event_key = panel_row["event_key"]
        assigned = assign_weak_roles(a4_lookup.get(event_key, []))
        strict_rows = [row for row in assigned if bool_from_text(row.get("strict_pass", ""))]
        broad_rows = [row for row in assigned if bool_from_text(row.get("broad_pass", ""))]
        out = {
            "event_key": event_key,
            "ticker": panel_row.get("ticker", ""),
            "year": panel_row.get("year", ""),
            "quarter": panel_row.get("quarter", ""),
        }
        out.update(build_sequence_features(strict_rows, args.num_bins, "strict"))
        out.update(build_sequence_features(broad_rows, args.num_bins, "broad"))
        rows.append(out)

    summary = {"num_events": len(rows), "num_bins": args.num_bins, "weak_roles": True}
    write_csv(output_dir / "weak_section_sequence_features.csv", rows)
    write_json(output_dir / "weak_section_sequence_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
