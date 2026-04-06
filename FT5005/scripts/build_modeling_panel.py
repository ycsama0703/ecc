#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from dj30_qc_utils import (
    build_event_key,
    build_event_path_lookup,
    iter_files,
    load_csv_rows,
    load_json,
    normalize_event_key_text,
    normalize_text,
    safe_float,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a baseline-ready event panel from targets, analyst tables, and transcript QC summaries."
    )
    parser.add_argument("--targets-csv", type=Path, required=True)
    parser.add_argument("--c1-csv", type=Path, required=True)
    parser.add_argument("--c2-csv", type=Path, required=True)
    parser.add_argument("--a1-dir", type=Path, required=True)
    parser.add_argument("--a2-qc-csv", type=Path, required=True)
    parser.add_argument("--a4-event-qc-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/panel_real"))
    return parser.parse_args()


def to_float_str(value) -> str:
    return "" if value is None else str(value)


def scheduled_hour_value(scheduled_datetime: str) -> str:
    value = (scheduled_datetime or "").strip()
    if not value:
        return ""
    try:
        time_part = value.split(" ", 1)[1]
        pieces = time_part.split(":")
        hour = int(pieces[0])
        minute = int(pieces[1]) if len(pieces) > 1 else 0
        second = int(float(pieces[2])) if len(pieces) > 2 else 0
    except Exception:
        return ""
    return to_float_str(round(hour + minute / 60.0 + second / 3600.0, 6))


def event_key_from_c1_row(row: dict[str, str]) -> str:
    ticker = normalize_text(row.get("ticker", "") or row.get("SP_TICKER", "")).upper()
    year = normalize_text(row.get("fiscalyear", ""))
    quarter_raw = normalize_text(row.get("fiscalquarter", ""))
    if not ticker or not year or not quarter_raw:
        return ""
    try:
        year_int = int(float(year))
        quarter_int = int(float(quarter_raw))
    except ValueError:
        return ""
    return build_event_key(ticker, str(year_int), f"Q{quarter_int}")


def event_key_from_c2_row(row: dict[str, str]) -> str:
    ticker = normalize_text(row.get("ticker", "")).upper()
    year = normalize_text(row.get("year", ""))
    quarter_raw = normalize_text(row.get("quarter_num", ""))
    if not ticker or not year or not quarter_raw:
        return ""
    try:
        year_int = int(float(year))
        quarter_int = int(float(quarter_raw))
    except ValueError:
        return ""
    return build_event_key(ticker, str(year_int), f"Q{quarter_int}")


def build_c1_lookup(path: Path) -> dict[str, dict]:
    lookup = {}
    for row in load_csv_rows(path):
        event_key = event_key_from_c1_row(row)
        if not event_key:
            continue
        revenue_actual = safe_float(row.get("CIQ Revenue Actual ($000)"))
        revenue_estimate = safe_float(row.get("CIQ Revenue Estimate ($000)"))
        ebitda_actual = safe_float(row.get("CIQ EBITDA Actual ($000)"))
        ebitda_estimate = safe_float(row.get("CIQ EBITDA Estimate ($000)"))
        eps_actual = safe_float(row.get("CIQ EPS (GAAP) Actual ($)"))
        eps_estimate = safe_float(row.get("CIQ EPS (GAAP) Estimate ($)"))

        def surprise(actual, estimate):
            if actual is None or estimate is None:
                return None, None
            diff = actual - estimate
            pct = None if estimate == 0 else diff / abs(estimate)
            return diff, pct

        revenue_diff, revenue_pct = surprise(revenue_actual, revenue_estimate)
        ebitda_diff, ebitda_pct = surprise(ebitda_actual, ebitda_estimate)
        eps_diff, eps_pct = surprise(eps_actual, eps_estimate)

        lookup[event_key] = {
            "earnings_announce_date": row.get("Earnings Announce Date MM/dd/yyyy", ""),
            "revenue_actual": to_float_str(revenue_actual),
            "revenue_estimate": to_float_str(revenue_estimate),
            "revenue_surprise": to_float_str(revenue_diff),
            "revenue_surprise_pct": to_float_str(revenue_pct),
            "ebitda_actual": to_float_str(ebitda_actual),
            "ebitda_estimate": to_float_str(ebitda_estimate),
            "ebitda_surprise": to_float_str(ebitda_diff),
            "ebitda_surprise_pct": to_float_str(ebitda_pct),
            "eps_gaap_actual": to_float_str(eps_actual),
            "eps_gaap_estimate": to_float_str(eps_estimate),
            "eps_gaap_surprise": to_float_str(eps_diff),
            "eps_gaap_surprise_pct": to_float_str(eps_pct),
        }
    return lookup


def build_c2_lookup(path: Path) -> dict[str, dict]:
    lookup = {}
    for row in load_csv_rows(path):
        event_key = event_key_from_c2_row(row)
        if not event_key:
            continue
        lookup[event_key] = {
            "analyst_eps_norm_mean": row.get("eps_norm_estimate_CIQ_EPS_Normalized_Est\n$", ""),
            "analyst_eps_norm_num_est": row.get(
                "eps_norm_stddev_estimate_CIQ_EPS_Normalized_Est_-_#_of_Est\nactual", ""
            ),
            "analyst_eps_norm_std": row.get(
                "eps_norm_stddev_estimate_CIQ_EPS_Normalized_Est_-_Std_Dev\n$", ""
            ),
            "analyst_revenue_mean": row.get("revenue_estimate_CIQ_Revenue_Est\n$000", ""),
            "analyst_revenue_median": row.get("revenue_estimate_CIQ_Revenue_Est_Med\n$000", ""),
            "analyst_revenue_num_est": row.get(
                "revenue_stddev_estimate_CIQ_Revenue_Est_-_#_of_Est\nactual", ""
            ),
            "analyst_revenue_std": row.get(
                "revenue_stddev_estimate_CIQ_Revenue_Est_-_Std_Dev\n$000", ""
            ),
            "analyst_net_income_mean": row.get("net_income_estimate_CIQ_Net_Income_Est\n$000", ""),
            "analyst_net_income_num_est": row.get(
                "net_income_stddev_estimate_CIQ_Net_Income_Est_-_#_of_Est\nactual", ""
            ),
            "analyst_net_income_std": row.get(
                "net_income_stddev_estimate_CIQ_Net_Income_Est_-_Std_Dev\n$000", ""
            ),
        }
    return lookup


def build_a1_stats_lookup(a1_dir: Path) -> tuple[dict[str, dict], dict[str, list[str]]]:
    paths, duplicates = build_event_path_lookup(iter_files(a1_dir.resolve(), [".json"]))
    lookup = {}
    for event_key, path in paths.items():
        payload = load_json(path)
        components = payload.get("components", [])
        component_types = Counter()
        speaker_names = set()
        management_speakers = set()
        total_chars = 0
        total_words = 0
        for component in components:
            component_type = normalize_text(component.get("componenttypename", ""))
            component_types[component_type] += 1
            text = normalize_text(component.get("text", ""))
            total_chars += len(text)
            total_words += len(text.split())
            personname = normalize_text(component.get("personname", ""))
            companyofperson = normalize_text(component.get("companyofperson", ""))
            if personname:
                speaker_names.add(personname)
                if companyofperson:
                    management_speakers.add(personname)

        question_count = component_types.get("Question", 0)
        answer_count = component_types.get("Answer", 0)
        qa_operator_count = component_types.get("Question and Answer Operator Message", 0)
        presenter_count = component_types.get("Presenter Speech", 0)
        presentation_operator_count = component_types.get("Presentation Operator Message", 0)
        unknown_qa_count = component_types.get("Unknown Question and Answer Message", 0)
        qna_components = (
            question_count + answer_count + qa_operator_count + unknown_qa_count
        )

        lookup[event_key] = {
            "a1_component_count": len(components),
            "a1_question_count": question_count,
            "a1_answer_count": answer_count,
            "a1_qa_operator_count": qa_operator_count,
            "a1_presenter_speech_count": presenter_count,
            "a1_presentation_operator_count": presentation_operator_count,
            "a1_unknown_qa_count": unknown_qa_count,
            "a1_qna_component_share": "" if not components else round(qna_components / len(components), 6),
            "a1_unique_speaker_count": len(speaker_names),
            "a1_management_speaker_count": len(management_speakers),
            "a1_total_text_chars": total_chars,
            "a1_total_text_words": total_words,
            "a1_headline": payload.get("headline", ""),
        }
    return lookup, duplicates


def build_a2_qc_lookup(path: Path) -> dict[str, dict]:
    lookup = {}
    for row in load_csv_rows(path):
        event_key = normalize_event_key_text(row.get("event_key", ""))
        if not event_key:
            continue
        lookup[event_key] = {
            "html_integrity_flag": row.get("html_integrity_flag", ""),
            "html_integrity_reason": row.get("html_integrity_reason", ""),
            "a2_paragraph_count": row.get("paragraph_count", ""),
            "a2_visible_text_length": row.get("visible_text_length", ""),
            "a2_visible_word_count": row.get("visible_word_count", ""),
            "a2_size_ratio_vs_group": row.get("size_ratio_vs_group", ""),
            "a2_text_ratio_vs_group": row.get("text_ratio_vs_group", ""),
        }
    return lookup


def build_a4_qc_lookup(path: Path) -> dict[str, dict]:
    lookup = {}
    for row in load_csv_rows(path):
        event_key = normalize_event_key_text(row.get("event_id", ""))
        if not event_key:
            continue
        lookup[event_key] = {
            "a4_total_rows": row.get("total_rows", ""),
            "a4_strict_rows": row.get("strict_rows", ""),
            "a4_broad_rows": row.get("broad_rows", ""),
            "a4_strict_row_share": row.get("strict_row_share", ""),
            "a4_broad_row_share": row.get("broad_row_share", ""),
            "a4_strict_span_sec": row.get("strict_span_sec", ""),
            "a4_broad_span_sec": row.get("broad_span_sec", ""),
            "a4_median_match_score": row.get("median_match_score", ""),
            "a4_min_match_score": row.get("min_match_score", ""),
            "a4_hard_fail_rows": row.get("hard_fail_rows", ""),
            "a4_overlap_warn_rows": row.get("overlap_warn_rows", ""),
            "a4_qc_subset": row.get("qc_subset", ""),
        }
    return lookup


def build_panel_row(
    target_row: dict[str, str],
    c1_lookup: dict[str, dict],
    c2_lookup: dict[str, dict],
    a1_lookup: dict[str, dict],
    a2_lookup: dict[str, dict],
    a4_lookup: dict[str, dict],
) -> dict:
    event_key = target_row["event_key"]

    row = {
        **target_row,
        "scheduled_hour_et": scheduled_hour_value(target_row.get("scheduled_datetime", "")),
        "call_duration_min": to_float_str(
            None if not target_row.get("call_duration_sec") else float(target_row["call_duration_sec"]) / 60.0
        ),
    }
    row.update(c1_lookup.get(event_key, {}))
    row.update(c2_lookup.get(event_key, {}))
    row.update(a1_lookup.get(event_key, {}))
    row.update(a2_lookup.get(event_key, {}))
    row.update(a4_lookup.get(event_key, {}))
    row["has_c1"] = event_key in c1_lookup
    row["has_c2"] = event_key in c2_lookup
    row["has_a1"] = event_key in a1_lookup
    row["has_a2_qc"] = event_key in a2_lookup
    row["has_a4_qc"] = event_key in a4_lookup
    return row


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    target_rows = load_csv_rows(args.targets_csv.resolve())
    c1_lookup = build_c1_lookup(args.c1_csv.resolve())
    c2_lookup = build_c2_lookup(args.c2_csv.resolve())
    a1_lookup, a1_duplicates = build_a1_stats_lookup(args.a1_dir.resolve())
    a2_lookup = build_a2_qc_lookup(args.a2_qc_csv.resolve())
    a4_lookup = build_a4_qc_lookup(args.a4_event_qc_csv.resolve())

    panel_rows = [
        build_panel_row(row, c1_lookup, c2_lookup, a1_lookup, a2_lookup, a4_lookup)
        for row in target_rows
    ]

    write_csv(output_dir / "event_modeling_panel.csv", panel_rows)
    write_json(
        output_dir / "event_modeling_panel_summary.json",
        {
            "num_target_rows": len(target_rows),
            "num_panel_rows": len(panel_rows),
            "join_counts": {
                "has_c1": sum(1 for row in panel_rows if row["has_c1"]),
                "has_c2": sum(1 for row in panel_rows if row["has_c2"]),
                "has_a1": sum(1 for row in panel_rows if row["has_a1"]),
                "has_a2_qc": sum(1 for row in panel_rows if row["has_a2_qc"]),
                "has_a4_qc": sum(1 for row in panel_rows if row["has_a4_qc"]),
            },
            "html_integrity_counts": dict(
                Counter(row.get("html_integrity_flag", "") for row in panel_rows)
            ),
            "a4_qc_subset_counts": dict(
                Counter(row.get("a4_qc_subset", "") for row in panel_rows)
            ),
            "a1_duplicate_event_keys": a1_duplicates,
        },
    )

    print(
        json.dumps(
            {
                "num_panel_rows": len(panel_rows),
                "join_counts": {
                    "has_c1": sum(1 for row in panel_rows if row["has_c1"]),
                    "has_c2": sum(1 for row in panel_rows if row["has_c2"]),
                    "has_a1": sum(1 for row in panel_rows if row["has_a1"]),
                    "has_a2_qc": sum(1 for row in panel_rows if row["has_a2_qc"]),
                    "has_a4_qc": sum(1 for row in panel_rows if row["has_a4_qc"]),
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
