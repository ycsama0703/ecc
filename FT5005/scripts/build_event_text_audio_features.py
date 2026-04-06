#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from dj30_qc_utils import (
    build_event_path_lookup,
    iter_files,
    load_csv_rows,
    load_json,
    normalize_event_key_text,
    normalize_text,
    safe_float,
    split_sentences,
    token_f1,
    write_csv,
    write_json,
)


LEXICONS = {
    "guidance": [
        "guidance",
        "outlook",
        "forecast",
        "forecasts",
        "expect",
        "expects",
        "expected",
        "expecting",
        "anticipate",
        "anticipates",
        "anticipated",
        "trend",
        "trends",
    ],
    "uncertainty": [
        "uncertain",
        "uncertainty",
        "volatile",
        "volatility",
        "risk",
        "risks",
        "headwind",
        "headwinds",
        "pressure",
        "challenging",
        "cautious",
        "macro",
        "slowdown",
    ],
    "positive": [
        "strong",
        "growth",
        "improve",
        "improved",
        "improvement",
        "confidence",
        "confident",
        "opportunity",
        "record",
        "resilient",
        "momentum",
        "benefit",
    ],
    "negative": [
        "weak",
        "weaker",
        "decline",
        "declined",
        "soft",
        "slower",
        "pressure",
        "uncertain",
        "risk",
        "lower",
        "down",
        "challenge",
        "challenging",
    ],
}

PAIR_LEXICONS = {
    "hedge": [
        "may",
        "might",
        "could",
        "would",
        "approximately",
        "around",
        "roughly",
        "likely",
        "maybe",
        "perhaps",
        "believe",
        "think",
        "assume",
        "potentially",
    ],
    "assertive": [
        "will",
        "definitely",
        "certainly",
        "clear",
        "clearly",
        "confident",
        "committed",
        "commitment",
        "strongly",
        "firmly",
    ],
    "forward": [
        "guidance",
        "outlook",
        "expect",
        "expects",
        "expected",
        "forecast",
        "forecasts",
        "next",
        "future",
        "pipeline",
        "plan",
        "plans",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build event-level text fields and lightweight timing/audio-proxy features."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--a4-row-qc-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/features_real")
    )
    return parser.parse_args()


def bool_from_text(value: str) -> bool:
    return normalize_text(value).lower() in {"true", "1", "yes"}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", (text or "").lower())


def count_lexicon_terms(tokens: list[str], terms: list[str]) -> int:
    term_set = set(terms)
    return sum(1 for token in tokens if token in term_set)


def safe_ratio(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def summarize_numeric(values: list[float]) -> dict[str, float]:
    clean = [value for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "count": len(clean),
        "mean": round(float(sum(clean) / len(clean)), 4),
        "median": round(float(statistics.median(clean)), 4),
        "min": round(float(min(clean)), 4),
        "max": round(float(max(clean)), 4),
    }


def load_panel_rows(path: Path) -> list[dict[str, str]]:
    rows = []
    for row in load_csv_rows(path.resolve()):
        event_key = normalize_event_key_text(row.get("event_key", ""))
        if not event_key:
            continue
        row["event_key"] = event_key
        rows.append(row)
    return rows


def build_a1_event_features(a1_dir: Path, event_keys: set[str]) -> dict[str, dict[str, str]]:
    path_lookup, _ = build_event_path_lookup(iter_files(a1_dir.resolve(), [".json"]))
    features = {}

    for event_key in sorted(event_keys):
        path = path_lookup.get(event_key)
        if path is None:
            continue
        payload = load_json(path)
        components = payload.get("components", [])

        text_buckets = {
            "full": [],
            "qna": [],
            "question": [],
            "answer": [],
            "presenter": [],
            "operator": [],
            "unknown_qna": [],
        }
        speaker_turn_counts = Counter()
        speaker_names = set()
        qa_pairs = []
        pending_question = None
        pending_question_marks = 0

        for component in components:
            text = normalize_text(component.get("text", ""))
            if not text:
                continue
            component_type = normalize_text(component.get("componenttypename", ""))
            speaker = normalize_text(component.get("personname", ""))
            if speaker:
                speaker_names.add(speaker)
                speaker_turn_counts[speaker] += 1

            text_buckets["full"].append(text)
            if component_type == "Question":
                text_buckets["question"].append(text)
                text_buckets["qna"].append(text)
                pending_question = text
                pending_question_marks = text.count("?")
            elif component_type == "Answer":
                text_buckets["answer"].append(text)
                text_buckets["qna"].append(text)
                if pending_question is not None:
                    qa_pairs.append(
                        {
                            "question_text": pending_question,
                            "answer_text": text,
                            "question_marks": pending_question_marks,
                        }
                    )
                    pending_question = None
                    pending_question_marks = 0
            elif component_type == "Presenter Speech":
                text_buckets["presenter"].append(text)
            elif "Operator" in component_type:
                text_buckets["operator"].append(text)
                if "Question and Answer" in component_type:
                    text_buckets["qna"].append(text)
            elif component_type == "Unknown Question and Answer Message":
                text_buckets["unknown_qna"].append(text)
                text_buckets["qna"].append(text)

        joined = {name: "\n".join(items) for name, items in text_buckets.items()}
        full_tokens = tokenize(joined["full"])
        qna_tokens = tokenize(joined["qna"])
        question_tokens = tokenize(joined["question"])
        answer_tokens = tokenize(joined["answer"])
        presenter_tokens = tokenize(joined["presenter"])
        operator_tokens = tokenize(joined["operator"])
        presenter_word_count = len(presenter_tokens)

        full_text = joined["full"]
        qna_text = joined["qna"]

        full_word_count = len(full_tokens)
        qna_word_count = len(qna_tokens)
        question_word_count = len(question_tokens)
        answer_word_count = len(answer_tokens)
        operator_word_count = len(operator_tokens)

        top_turn_share = 0.0
        if speaker_turn_counts:
            top_turn_share = max(speaker_turn_counts.values()) / sum(speaker_turn_counts.values())

        sentence_count = len(split_sentences(full_text))
        qna_sentence_count = len(split_sentences(qna_text))
        question_mark_count = full_text.count("?")
        digit_count = sum(char.isdigit() for char in full_text)

        event_features = {
            "event_key": event_key,
            "full_text": full_text,
            "qna_text": qna_text,
            "question_text": joined["question"],
            "answer_text": joined["answer"],
            "presenter_text": joined["presenter"],
            "operator_text": joined["operator"],
            "full_word_count": str(full_word_count),
            "qna_word_count": str(qna_word_count),
            "question_word_count": str(question_word_count),
            "answer_word_count": str(answer_word_count),
            "presenter_word_count": str(presenter_word_count),
            "operator_word_count": str(operator_word_count),
            "full_sentence_count": str(sentence_count),
            "qna_sentence_count": str(qna_sentence_count),
            "question_mark_count": str(question_mark_count),
            "digit_char_share": f"{safe_ratio(digit_count, max(len(full_text), 1)):.6f}",
            "qna_word_share": f"{safe_ratio(qna_word_count, max(full_word_count, 1)):.6f}",
            "presenter_word_share": f"{safe_ratio(presenter_word_count, max(full_word_count, 1)):.6f}",
            "operator_word_share": f"{safe_ratio(operator_word_count, max(full_word_count, 1)):.6f}",
            "answer_to_question_word_ratio": f"{safe_ratio(answer_word_count, max(question_word_count, 1)):.6f}",
            "avg_words_per_question": f"{safe_ratio(question_word_count, max(len(text_buckets['question']), 1)):.6f}",
            "avg_words_per_answer": f"{safe_ratio(answer_word_count, max(len(text_buckets['answer']), 1)):.6f}",
            "speaker_turn_concentration": f"{top_turn_share:.6f}",
            "speaker_name_count_from_text": str(len(speaker_names)),
            "question_mark_per_1k_words": f"{1000.0 * safe_ratio(question_mark_count, max(full_word_count, 1)):.6f}",
        }

        for lexicon_name, terms in LEXICONS.items():
            event_features[f"{lexicon_name}_term_rate_full"] = (
                f"{1000.0 * safe_ratio(count_lexicon_terms(full_tokens, terms), max(full_word_count, 1)):.6f}"
            )
            event_features[f"{lexicon_name}_term_rate_qna"] = (
                f"{1000.0 * safe_ratio(count_lexicon_terms(qna_tokens, terms), max(qna_word_count, 1)):.6f}"
            )
            event_features[f"{lexicon_name}_term_rate_presenter"] = (
                f"{1000.0 * safe_ratio(count_lexicon_terms(presenter_tokens, terms), max(presenter_word_count, 1)):.6f}"
            )
            event_features[f"{lexicon_name}_term_rate_answer"] = (
                f"{1000.0 * safe_ratio(count_lexicon_terms(answer_tokens, terms), max(answer_word_count, 1)):.6f}"
            )
            event_features[f"{lexicon_name}_term_rate_question"] = (
                f"{1000.0 * safe_ratio(count_lexicon_terms(question_tokens, terms), max(question_word_count, 1)):.6f}"
            )

        pair_overlaps = []
        pair_question_words = []
        pair_answer_words = []
        pair_digit_rates = []
        pair_hedge_rates = []
        pair_assertive_rates = []
        pair_forward_rates = []
        multi_part_share_hits = 0
        evasive_hits = 0

        for pair in qa_pairs:
            question_text = pair["question_text"]
            answer_text = pair["answer_text"]
            question_tokens_pair = tokenize(question_text)
            answer_tokens_pair = tokenize(answer_text)
            q_words = len(question_tokens_pair)
            a_words = len(answer_tokens_pair)
            overlap = token_f1(question_text, answer_text)
            hedge_rate = 1000.0 * safe_ratio(
                count_lexicon_terms(answer_tokens_pair, PAIR_LEXICONS["hedge"]), max(a_words, 1)
            )
            assertive_rate = 1000.0 * safe_ratio(
                count_lexicon_terms(answer_tokens_pair, PAIR_LEXICONS["assertive"]), max(a_words, 1)
            )
            forward_rate = 1000.0 * safe_ratio(
                count_lexicon_terms(answer_tokens_pair, PAIR_LEXICONS["forward"]), max(a_words, 1)
            )
            digit_rate = 1000.0 * safe_ratio(
                sum(token.isdigit() for token in answer_tokens_pair), max(a_words, 1)
            )

            pair_overlaps.append(overlap)
            pair_question_words.append(q_words)
            pair_answer_words.append(a_words)
            pair_digit_rates.append(digit_rate)
            pair_hedge_rates.append(hedge_rate)
            pair_assertive_rates.append(assertive_rate)
            pair_forward_rates.append(forward_rate)

            if pair["question_marks"] >= 2:
                multi_part_share_hits += 1
            if overlap < 0.15 and hedge_rate >= 15.0:
                evasive_hits += 1

        def mean_or_zero(values: list[float]) -> float:
            return float(statistics.mean(values)) if values else 0.0

        def median_or_zero(values: list[float]) -> float:
            return float(statistics.median(values)) if values else 0.0

        qa_pair_count = len(qa_pairs)
        event_features.update(
            {
                "qa_pair_count": str(qa_pair_count),
                "qa_pair_overlap_mean": f"{mean_or_zero(pair_overlaps):.6f}",
                "qa_pair_overlap_median": f"{median_or_zero(pair_overlaps):.6f}",
                "qa_pair_low_overlap_share": f"{safe_ratio(sum(1 for value in pair_overlaps if value < 0.15), max(qa_pair_count, 1)):.6f}",
                "qa_pair_question_words_mean": f"{mean_or_zero(pair_question_words):.6f}",
                "qa_pair_answer_words_mean": f"{mean_or_zero(pair_answer_words):.6f}",
                "qa_pair_answer_digit_rate_mean": f"{mean_or_zero(pair_digit_rates):.6f}",
                "qa_pair_answer_hedge_rate_mean": f"{mean_or_zero(pair_hedge_rates):.6f}",
                "qa_pair_answer_assertive_rate_mean": f"{mean_or_zero(pair_assertive_rates):.6f}",
                "qa_pair_answer_forward_rate_mean": f"{mean_or_zero(pair_forward_rates):.6f}",
                "qa_multi_part_question_share": f"{safe_ratio(multi_part_share_hits, max(qa_pair_count, 1)):.6f}",
                "qa_evasive_proxy_share": f"{safe_ratio(evasive_hits, max(qa_pair_count, 1)):.6f}",
                "qna_vs_presenter_uncertainty_gap": (
                    f"{safe_float(event_features['uncertainty_term_rate_qna']) - safe_float(event_features['uncertainty_term_rate_presenter']):.6f}"
                ),
                "qna_vs_presenter_guidance_gap": (
                    f"{safe_float(event_features['guidance_term_rate_qna']) - safe_float(event_features['guidance_term_rate_presenter']):.6f}"
                ),
                "answer_vs_question_uncertainty_gap": (
                    f"{safe_float(event_features['uncertainty_term_rate_answer']) - safe_float(event_features['uncertainty_term_rate_question']):.6f}"
                ),
                "answer_vs_question_negative_gap": (
                    f"{safe_float(event_features['negative_term_rate_answer']) - safe_float(event_features['negative_term_rate_question']):.6f}"
                ),
            }
        )

        features[event_key] = event_features

    return features


def build_a4_timing_features(path: Path, event_keys: set[str]) -> dict[str, dict[str, str]]:
    grouped_rows = defaultdict(list)
    for row in load_csv_rows(path.resolve()):
        event_key = normalize_event_key_text(row.get("event_id", ""))
        if event_key not in event_keys:
            continue
        grouped_rows[event_key].append(row)

    features = {}
    for event_key in sorted(event_keys):
        rows = grouped_rows.get(event_key, [])
        if not rows:
            continue

        strict_rows = [row for row in rows if bool_from_text(row.get("strict_pass", ""))]
        broad_rows = [row for row in rows if bool_from_text(row.get("broad_pass", ""))]

        def summarize_rows(selected_rows: list[dict[str, str]], prefix: str) -> dict[str, str]:
            starts = [safe_float(row.get("start_sec")) for row in selected_rows]
            ends = [safe_float(row.get("end_sec")) for row in selected_rows]
            durations = [safe_float(row.get("duration_sec")) for row in selected_rows]
            starts = [value for value in starts if value is not None]
            ends = [value for value in ends if value is not None]
            durations = [value for value in durations if value is not None]

            ordered = []
            for row in selected_rows:
                start_sec = safe_float(row.get("start_sec"))
                end_sec = safe_float(row.get("end_sec"))
                if start_sec is None or end_sec is None:
                    continue
                ordered.append((start_sec, end_sec, row))
            ordered.sort(key=lambda item: (item[0], item[1]))

            gaps = []
            high_conf = 0
            overlap_warn = 0
            prev_end = None
            for start_sec, end_sec, row in ordered:
                match_score = safe_float(row.get("match_score"))
                if match_score is not None and match_score >= 95.0:
                    high_conf += 1
                if bool_from_text(row.get("overlap_warn_flag", "")):
                    overlap_warn += 1
                if prev_end is not None:
                    gaps.append(max(0.0, start_sec - prev_end))
                prev_end = max(prev_end, end_sec) if prev_end is not None else end_sec

            span_sec = 0.0
            if starts and ends:
                span_sec = max(ends) - min(starts)

            return {
                f"{prefix}_segment_count": str(len(selected_rows)),
                f"{prefix}_duration_sum_sec": f"{sum(durations):.6f}" if durations else "",
                f"{prefix}_duration_mean_sec": f"{statistics.mean(durations):.6f}" if durations else "",
                f"{prefix}_duration_median_sec": f"{statistics.median(durations):.6f}" if durations else "",
                f"{prefix}_duration_std_sec": f"{statistics.pstdev(durations):.6f}" if len(durations) > 1 else "",
                f"{prefix}_gap_mean_sec": f"{statistics.mean(gaps):.6f}" if gaps else "",
                f"{prefix}_gap_max_sec": f"{max(gaps):.6f}" if gaps else "",
                f"{prefix}_span_sec": f"{span_sec:.6f}" if span_sec else "",
                f"{prefix}_high_conf_share": f"{safe_ratio(high_conf, max(len(selected_rows), 1)):.6f}",
                f"{prefix}_overlap_warn_share": f"{safe_ratio(overlap_warn, max(len(selected_rows), 1)):.6f}",
            }

        row = {"event_key": event_key}
        row.update(summarize_rows(strict_rows, "a4_strict"))
        row.update(summarize_rows(broad_rows, "a4_broad"))
        features[event_key] = row

    return features


def build_a3_audio_features(
    a3_dir: Path, panel_rows: dict[str, dict[str, str]], event_keys: set[str]
) -> dict[str, dict[str, str]]:
    path_lookup, _ = build_event_path_lookup(iter_files(a3_dir.resolve(), [".mp3"]))
    features = {}

    for event_key in sorted(event_keys):
        path = path_lookup.get(event_key)
        panel_row = panel_rows[event_key]
        call_duration_sec = safe_float(panel_row.get("call_duration_sec"))
        strict_duration = safe_float(panel_row.get("a4_strict_span_sec"))
        if path is None:
            features[event_key] = {
                "event_key": event_key,
                "has_audio_file": "0",
                "a3_file_size_bytes": "",
                "a3_log_file_size": "",
                "a3_bytes_per_call_sec": "",
                "a3_bytes_per_strict_span_sec": "",
            }
            continue

        size_bytes = path.stat().st_size
        features[event_key] = {
            "event_key": event_key,
            "has_audio_file": "1",
            "a3_file_size_bytes": str(size_bytes),
            "a3_log_file_size": f"{math.log1p(size_bytes):.6f}",
            "a3_bytes_per_call_sec": (
                f"{safe_ratio(size_bytes, call_duration_sec):.6f}" if call_duration_sec else ""
            ),
            "a3_bytes_per_strict_span_sec": (
                f"{safe_ratio(size_bytes, strict_duration):.6f}" if strict_duration else ""
            ),
        }

    return features


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_rows_list = load_panel_rows(args.panel_csv)
    panel_rows = {row["event_key"]: row for row in panel_rows_list}
    event_keys = set(panel_rows)

    a1_features = build_a1_event_features(args.a1_dir, event_keys)
    a4_features = build_a4_timing_features(args.a4_row_qc_csv, event_keys)
    merged_for_audio = {}
    for event_key, panel_row in panel_rows.items():
        merged = dict(panel_row)
        merged.update(a4_features.get(event_key, {}))
        merged_for_audio[event_key] = merged
    a3_features = build_a3_audio_features(args.a3_dir, merged_for_audio, event_keys)

    rows = []
    for event_key in sorted(event_keys):
        row = {"event_key": event_key}
        row.update(a1_features.get(event_key, {"event_key": event_key}))
        row.update(a4_features.get(event_key, {"event_key": event_key}))
        row.update(a3_features.get(event_key, {"event_key": event_key}))
        row["ticker"] = panel_rows[event_key].get("ticker", "")
        row["year"] = panel_rows[event_key].get("year", "")
        row["quarter"] = panel_rows[event_key].get("quarter", "")
        rows.append(row)

    summary = {
        "num_panel_events": len(panel_rows),
        "num_rows_written": len(rows),
        "num_with_a1_text": sum(1 for row in rows if normalize_text(row.get("full_text", ""))),
        "num_with_audio_file": sum(1 for row in rows if row.get("has_audio_file") == "1"),
        "full_word_count": summarize_numeric(
            [safe_float(row.get("full_word_count")) for row in rows]
        ),
        "qna_word_count": summarize_numeric(
            [safe_float(row.get("qna_word_count")) for row in rows]
        ),
        "a4_strict_segment_count": summarize_numeric(
            [safe_float(row.get("a4_strict_segment_count")) for row in rows]
        ),
        "a4_strict_span_sec": summarize_numeric(
            [safe_float(row.get("a4_strict_span_sec")) for row in rows]
        ),
        "a3_file_size_bytes": summarize_numeric(
            [safe_float(row.get("a3_file_size_bytes")) for row in rows]
        ),
    }

    write_csv(output_dir / "event_text_audio_features.csv", rows)
    write_json(output_dir / "event_text_audio_features_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
