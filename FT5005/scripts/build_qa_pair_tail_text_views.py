#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build_qa_benchmark_features import (
    CERTAINTY_MARKERS,
    FORWARD_MARKERS,
    HEDGE_MARKERS,
    JUSTIFICATION_MARKERS,
    NONRESPONSE_PHRASES,
    WH_WORDS,
    content_tokens,
    count_terms,
    direct_answer_flag,
    earliest_overlap_position,
    evasion_score,
    has_numeric_cue,
    opening_restatement_flag,
    phrase_hit,
    safe_rate,
    short_evasive_flag,
    tokenize,
)
from build_qa_pair_tail_features import severity_score
from dj30_qc_utils import (
    build_event_path_lookup,
    iter_files,
    load_csv_rows,
    load_json,
    normalize_event_key_text,
    normalize_text,
    token_f1,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hardest-pair Q/A text views using pair-tail severity ranking."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qa_pair_tail_text_views_real"))
    return parser.parse_args()


def clean_text(text: str) -> str:
    return normalize_text(text)


def pair_record(question_text: str, answer_text: str) -> dict[str, object]:
    q_tokens = content_tokens(question_text)
    answer_all_tokens = tokenize(answer_text)
    a_tokens = content_tokens(answer_text)
    coverage_value = token_f1(question_text, answer_text)
    overlap_pos = earliest_overlap_position(q_tokens, answer_all_tokens)
    direct_flag = direct_answer_flag(question_text, answer_text, q_tokens, answer_all_tokens)
    direct_early = direct_flag * (1.0 - overlap_pos)
    hedge_rate = safe_rate(count_terms(answer_all_tokens, HEDGE_MARKERS), len(answer_all_tokens))
    certainty_rate = safe_rate(count_terms(answer_all_tokens, CERTAINTY_MARKERS), len(answer_all_tokens))
    justification_rate = safe_rate(count_terms(answer_all_tokens, JUSTIFICATION_MARKERS), len(answer_all_tokens))
    forward_rate = safe_rate(count_terms(answer_all_tokens, FORWARD_MARKERS), len(answer_all_tokens))
    nonresponse_flag = 1.0 if phrase_hit(answer_text, NONRESPONSE_PHRASES) else 0.0
    restatement_flag = opening_restatement_flag(answer_text)
    numeric_token_count = sum(token.isdigit() or bool(re.match(r"^\d", token)) for token in answer_all_tokens)
    numeric_mismatch_flag = 1.0 if has_numeric_cue(question_text, tokenize(question_text)) and numeric_token_count == 0 else 0.0
    short_flag = short_evasive_flag(direct_flag, coverage_value, q_tokens, a_tokens)
    evasion = evasion_score(
        direct_flag=direct_flag,
        nonresponse_flag=nonresponse_flag,
        deflection_flag=restatement_flag,
        delay_share=overlap_pos,
        coverage_value=coverage_value,
        hedge_rate=hedge_rate,
        numeric_mismatch_flag=numeric_mismatch_flag,
        short_flag=short_flag,
    )
    question_complexity = float(
        len(question_text.split())
        + 5 * question_text.count("?")
        + 3 * sum(token in WH_WORDS for token in tokenize(question_text))
    )
    severity = severity_score(question_complexity, evasion, coverage_value, direct_early)
    return {
        "question_text": clean_text(question_text),
        "answer_text": clean_text(answer_text),
        "qa_text": f"Q: {clean_text(question_text)} A: {clean_text(answer_text)}",
        "severity": float(severity),
        "coverage": float(coverage_value),
        "direct_early": float(direct_early),
        "evasion": float(evasion),
        "question_complexity": float(question_complexity),
        "forward_rate": float(forward_rate),
        "certainty_rate": float(certainty_rate),
        "justification_rate": float(justification_rate),
    }


def concat_text(items: list[dict[str, object]], key: str, top_k: int) -> str:
    return " ".join(str(item.get(key, "")) for item in items[:top_k] if str(item.get(key, "")))


def mean(items: list[dict[str, object]], key: str, top_k: int) -> float:
    vals = [float(item[key]) for item in items[:top_k] if key in item]
    return float(sum(vals) / len(vals)) if vals else 0.0


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    event_keys = {
        normalize_event_key_text(row.get("event_key", ""))
        for row in load_csv_rows(args.panel_csv.resolve())
        if normalize_event_key_text(row.get("event_key", ""))
    }
    path_lookup, _ = build_event_path_lookup(iter_files(args.a1_dir.resolve(), [".json"]))

    rows = []
    detail_rows = []
    for event_key in sorted(event_keys):
        path = path_lookup.get(event_key)
        if path is None:
            continue
        payload = load_json(path)
        components = payload.get("components", [])
        pending_question = None
        pairs = []
        for component in components:
            text = normalize_text(component.get("text", ""))
            ctype = normalize_text(component.get("componenttypename", ""))
            if not text:
                continue
            if ctype == "Question":
                pending_question = text
            elif ctype == "Answer" and pending_question is not None:
                pairs.append(pair_record(pending_question, text))
                pending_question = None

        pairs.sort(key=lambda item: float(item["severity"]), reverse=True)
        top1 = pairs[:1]
        top2 = pairs[:2]
        row = {
            "event_key": event_key,
            "qa_pair_count": str(len(pairs)),
            "tail_top1_question_text": concat_text(top1, "question_text", 1),
            "tail_top1_answer_text": concat_text(top1, "answer_text", 1),
            "tail_top1_qa_text": concat_text(top1, "qa_text", 1),
            "tail_top2_question_text": concat_text(top2, "question_text", 2),
            "tail_top2_answer_text": concat_text(top2, "answer_text", 2),
            "tail_top2_qa_text": concat_text(top2, "qa_text", 2),
            "tail_top1_severity": f"{mean(top1, 'severity', 1):.6f}",
            "tail_top2_severity_mean": f"{mean(top2, 'severity', 2):.6f}",
            "tail_top1_evasion": f"{mean(top1, 'evasion', 1):.6f}",
            "tail_top2_evasion_mean": f"{mean(top2, 'evasion', 2):.6f}",
            "tail_top1_direct_early": f"{mean(top1, 'direct_early', 1):.6f}",
            "tail_top2_direct_early_mean": f"{mean(top2, 'direct_early', 2):.6f}",
            "tail_top1_coverage": f"{mean(top1, 'coverage', 1):.6f}",
            "tail_top2_coverage_mean": f"{mean(top2, 'coverage', 2):.6f}",
        }
        rows.append(row)
        for rank, item in enumerate(pairs[:5], start=1):
            detail_rows.append(
                {
                    "event_key": event_key,
                    "rank": rank,
                    **{k: v for k, v in item.items() if k != 'qa_text'},
                }
            )

    summary = {
        "num_events": len(rows),
        "num_events_with_top1_text": sum(1 for row in rows if row["tail_top1_question_text"]),
        "avg_pair_count": float(sum(float(row["qa_pair_count"]) for row in rows) / len(rows)) if rows else 0.0,
        "avg_top1_severity": float(sum(float(row["tail_top1_severity"]) for row in rows) / len(rows)) if rows else 0.0,
        "avg_top2_severity_mean": float(sum(float(row["tail_top2_severity_mean"]) for row in rows) / len(rows)) if rows else 0.0,
    }

    write_csv(output_dir / 'qa_pair_tail_text_views.csv', rows)
    write_csv(output_dir / 'qa_pair_tail_text_top_pairs.csv', detail_rows)
    write_json(output_dir / 'qa_pair_tail_text_views_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
