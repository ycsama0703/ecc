#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from build_qa_benchmark_features import (
    CERTAINTY_MARKERS,
    DEFLECTION_PHRASES,
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
    summarize,
    tokenize,
)
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

LOW_OVERLAP_THRESHOLD = 0.15
HIGH_EVASION_THRESHOLD = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build pair-tail Q&A features that summarize worst-case and dispersion structure within each event."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/qa_pair_tail_features_real"))
    return parser.parse_args()


def share(values: list[float]) -> float:
    return float(sum(values) / max(len(values), 1)) if values else 0.0


def std(values: list[float]) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(v)]
    if len(clean) < 2:
        return 0.0
    return float(statistics.pstdev(clean))


def top_mean(values: list[float], k: int) -> float:
    clean = sorted((float(v) for v in values if v is not None and math.isfinite(v)), reverse=True)
    if not clean:
        return 0.0
    return float(sum(clean[:k]) / min(k, len(clean)))


def bottom_mean(values: list[float], k: int) -> float:
    clean = sorted((float(v) for v in values if v is not None and math.isfinite(v)))
    if not clean:
        return 0.0
    return float(sum(clean[:k]) / min(k, len(clean)))


def first_mean(values: list[float], k: int) -> float:
    clean = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not clean:
        return 0.0
    return float(sum(clean[:k]) / min(k, len(clean)))


def severity_score(question_complexity: float, evasion: float, coverage_value: float, direct_early: float) -> float:
    difficulty = math.log1p(max(question_complexity, 0.0))
    answerability_gap = max(0.0, 1.0 - min(max(coverage_value, 0.0), 1.0))
    direct_gap = max(0.0, 1.0 - min(max(direct_early, 0.0), 1.0))
    return float(difficulty * (0.45 * evasion + 0.35 * answerability_gap + 0.20 * direct_gap))


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
    pair_rows = []

    for event_key in sorted(event_keys):
        path = path_lookup.get(event_key)
        if path is None:
            continue
        payload = load_json(path)
        components = payload.get("components", [])
        qa_pairs = []
        pending_question = None
        for component in components:
            text = normalize_text(component.get("text", ""))
            ctype = normalize_text(component.get("componenttypename", ""))
            if not text:
                continue
            if ctype == "Question":
                pending_question = text
            elif ctype == "Answer" and pending_question is not None:
                qa_pairs.append((pending_question, text))
                pending_question = None

        pair_metrics = []
        for pair_idx, (question_text, answer_text) in enumerate(qa_pairs, start=1):
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
            numeric_token_count = sum(token.isdigit() or bool(__import__('re').match(r"^\d", token)) for token in answer_all_tokens)
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
            sev = severity_score(question_complexity, evasion, coverage_value, direct_early)
            low_overlap_flag = 1.0 if coverage_value < LOW_OVERLAP_THRESHOLD else 0.0
            high_evasion_flag = 1.0 if evasion >= HIGH_EVASION_THRESHOLD else 0.0
            pair_metrics.append(
                {
                    "event_key": event_key,
                    "pair_index": pair_idx,
                    "coverage": float(coverage_value),
                    "direct_early": float(direct_early),
                    "delay_share": float(overlap_pos),
                    "evasion": float(evasion),
                    "question_complexity": question_complexity,
                    "severity": sev,
                    "low_overlap_flag": low_overlap_flag,
                    "high_evasion_flag": high_evasion_flag,
                    "nonresponse_flag": nonresponse_flag,
                    "short_evasive_flag": short_flag,
                    "numeric_mismatch_flag": numeric_mismatch_flag,
                    "forward_rate": float(forward_rate),
                    "hedge_rate": float(hedge_rate),
                    "certainty_rate": float(certainty_rate),
                    "justification_rate": float(justification_rate),
                }
            )

        coverage_vals = [row["coverage"] for row in pair_metrics]
        direct_early_vals = [row["direct_early"] for row in pair_metrics]
        delay_vals = [row["delay_share"] for row in pair_metrics]
        evasion_vals = [row["evasion"] for row in pair_metrics]
        complexity_vals = [row["question_complexity"] for row in pair_metrics]
        severity_vals = [row["severity"] for row in pair_metrics]
        low_overlap_flags = [row["low_overlap_flag"] for row in pair_metrics]
        high_evasion_flags = [row["high_evasion_flag"] for row in pair_metrics]
        nonresponse_flags = [row["nonresponse_flag"] for row in pair_metrics]
        short_flags = [row["short_evasive_flag"] for row in pair_metrics]
        numeric_mismatch_flags = [row["numeric_mismatch_flag"] for row in pair_metrics]
        forward_rates = [row["forward_rate"] for row in pair_metrics]
        hedge_rates = [row["hedge_rate"] for row in pair_metrics]
        certainty_rates = [row["certainty_rate"] for row in pair_metrics]
        justification_rates = [row["justification_rate"] for row in pair_metrics]

        coverage_mean, coverage_median = summarize(coverage_vals)
        direct_mean, direct_median = summarize(direct_early_vals)
        evasion_mean, evasion_median = summarize(evasion_vals)
        severity_mean, severity_median = summarize(severity_vals)

        rows.append(
            {
                "event_key": event_key,
                "qa_tail_pair_count": str(len(pair_metrics)),
                "qa_tail_coverage_mean": f"{coverage_mean:.6f}",
                "qa_tail_coverage_median": f"{coverage_median:.6f}",
                "qa_tail_min_coverage": f"{min(coverage_vals) if coverage_vals else 0.0:.6f}",
                "qa_tail_bottom2_coverage_mean": f"{bottom_mean(coverage_vals, 2):.6f}",
                "qa_tail_coverage_std": f"{std(coverage_vals):.6f}",
                "qa_tail_direct_early_mean": f"{direct_mean:.6f}",
                "qa_tail_direct_early_median": f"{direct_median:.6f}",
                "qa_tail_min_direct_early": f"{min(direct_early_vals) if direct_early_vals else 0.0:.6f}",
                "qa_tail_bottom2_direct_early_mean": f"{bottom_mean(direct_early_vals, 2):.6f}",
                "qa_tail_first2_direct_early_mean": f"{first_mean(direct_early_vals, 2):.6f}",
                "qa_tail_direct_early_std": f"{std(direct_early_vals):.6f}",
                "qa_tail_evasion_mean": f"{evasion_mean:.6f}",
                "qa_tail_evasion_median": f"{evasion_median:.6f}",
                "qa_tail_max_evasion": f"{max(evasion_vals) if evasion_vals else 0.0:.6f}",
                "qa_tail_top2_evasion_mean": f"{top_mean(evasion_vals, 2):.6f}",
                "qa_tail_first2_evasion_mean": f"{first_mean(evasion_vals, 2):.6f}",
                "qa_tail_evasion_std": f"{std(evasion_vals):.6f}",
                "qa_tail_high_evasion_share": f"{share(high_evasion_flags):.6f}",
                "qa_tail_low_overlap_share": f"{share(low_overlap_flags):.6f}",
                "qa_tail_nonresponse_share": f"{share(nonresponse_flags):.6f}",
                "qa_tail_short_evasive_share": f"{share(short_flags):.6f}",
                "qa_tail_numeric_mismatch_share": f"{share(numeric_mismatch_flags):.6f}",
                "qa_tail_max_delay_share": f"{max(delay_vals) if delay_vals else 0.0:.6f}",
                "qa_tail_top2_delay_share_mean": f"{top_mean(delay_vals, 2):.6f}",
                "qa_tail_question_complexity_max": f"{max(complexity_vals) if complexity_vals else 0.0:.6f}",
                "qa_tail_top2_complexity_mean": f"{top_mean(complexity_vals, 2):.6f}",
                "qa_tail_severity_mean": f"{severity_mean:.6f}",
                "qa_tail_severity_median": f"{severity_median:.6f}",
                "qa_tail_severity_max": f"{max(severity_vals) if severity_vals else 0.0:.6f}",
                "qa_tail_top2_severity_mean": f"{top_mean(severity_vals, 2):.6f}",
                "qa_tail_first2_severity_mean": f"{first_mean(severity_vals, 2):.6f}",
                "qa_tail_severity_std": f"{std(severity_vals):.6f}",
                "qa_tail_forward_rate_top2_mean": f"{top_mean(forward_rates, 2):.6f}",
                "qa_tail_hedge_rate_top2_mean": f"{top_mean(hedge_rates, 2):.6f}",
                "qa_tail_certainty_rate_top2_mean": f"{top_mean(certainty_rates, 2):.6f}",
                "qa_tail_justification_rate_top2_mean": f"{top_mean(justification_rates, 2):.6f}",
            }
        )
        pair_rows.extend(pair_metrics)

    summary = {
        "num_events": len(rows),
        "num_events_with_pairs": sum(1 for row in rows if float(row["qa_tail_pair_count"]) > 0),
        "avg_pair_count": float(statistics.mean(float(row["qa_tail_pair_count"]) for row in rows)) if rows else 0.0,
        "avg_max_evasion": float(statistics.mean(float(row["qa_tail_max_evasion"]) for row in rows)) if rows else 0.0,
        "avg_top2_severity": float(statistics.mean(float(row["qa_tail_top2_severity_mean"]) for row in rows)) if rows else 0.0,
    }

    write_csv(output_dir / 'qa_pair_tail_features.csv', rows)
    write_csv(output_dir / 'qa_pair_tail_pair_metrics.csv', pair_rows)
    write_json(output_dir / 'qa_pair_tail_features_summary.json', summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
