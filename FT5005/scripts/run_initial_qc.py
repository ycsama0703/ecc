#!/usr/bin/env python3

from __future__ import annotations

import argparse
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from dj30_qc_utils import (
    count_visible_paragraphs,
    detect_filename_metadata,
    infer_a4_event_id,
    iter_files,
    load_csv_rows,
    normalize_text,
    safe_float,
    strip_html_visible_text,
    text_similarity,
    token_f1,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 0 QC over A2 HTML files and A4 timed sentence rows."
    )
    parser.add_argument("--a2-dir", type=Path, required=True)
    parser.add_argument(
        "--a4-path",
        type=Path,
        required=True,
        help="Either the A4 directory or a single A4 CSV file.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/qc"))
    parser.add_argument("--strict-match-score", type=float, default=90.0)
    parser.add_argument("--broad-match-score", type=float, default=75.0)
    parser.add_argument("--strict-block-score", type=float, default=85.0)
    parser.add_argument("--broad-block-score", type=float, default=70.0)
    parser.add_argument("--strict-text-similarity", type=float, default=0.85)
    parser.add_argument("--broad-text-similarity", type=float, default=0.65)
    parser.add_argument("--strict-min-rows", type=int, default=20)
    parser.add_argument("--broad-min-rows", type=int, default=10)
    parser.add_argument("--strict-min-span-sec", type=float, default=600.0)
    parser.add_argument("--broad-min-span-sec", type=float, default=300.0)
    return parser.parse_args()


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def ratio_or_none(value: float, baseline: float | None) -> float | None:
    if baseline in (None, 0):
        return None
    return value / baseline


def run_a2_qc(a2_dir: Path) -> tuple[list[dict], dict]:
    html_files = iter_files(a2_dir, [".html", ".htm"])
    rows = []
    for file_path in html_files:
        raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
        visible_text = strip_html_visible_text(raw_html)
        meta = detect_filename_metadata(file_path)
        rows.append(
            {
                "path": str(file_path.resolve()),
                "stem": file_path.stem,
                "ticker": meta["ticker"],
                "year": meta["year"],
                "quarter": meta["quarter"],
                "event_key": meta["event_key"],
                "size_bytes": file_path.stat().st_size,
                "visible_text_length": len(visible_text),
                "paragraph_count": count_visible_paragraphs(raw_html),
                "visible_word_count": len(visible_text.split()),
            }
        )

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["ticker"] or "__GLOBAL__"].append(row)

    global_size_median = median_or_none([row["size_bytes"] for row in rows])
    global_text_median = median_or_none([row["visible_text_length"] for row in rows])
    global_paragraph_median = median_or_none([row["paragraph_count"] for row in rows])

    for ticker, ticker_rows in grouped.items():
        size_median = median_or_none([row["size_bytes"] for row in ticker_rows]) or global_size_median
        text_median = (
            median_or_none([row["visible_text_length"] for row in ticker_rows]) or global_text_median
        )
        paragraph_median = (
            median_or_none([row["paragraph_count"] for row in ticker_rows]) or global_paragraph_median
        )
        for row in ticker_rows:
            size_ratio = ratio_or_none(row["size_bytes"], size_median)
            text_ratio = ratio_or_none(row["visible_text_length"], text_median)
            paragraph_ratio = ratio_or_none(row["paragraph_count"], paragraph_median)

            flag = "pass"
            reason = ""
            if row["visible_text_length"] == 0 or row["paragraph_count"] == 0:
                flag = "fail"
                reason = "empty_visible_text_or_paragraphs"
            elif any(
                ratio is not None and ratio < threshold
                for ratio, threshold in (
                    (size_ratio, 0.35),
                    (text_ratio, 0.35),
                    (paragraph_ratio, 0.30),
                )
            ):
                flag = "fail"
                reason = "strong_downward_outlier"
            elif any(
                ratio is not None and ratio < threshold
                for ratio, threshold in (
                    (size_ratio, 0.65),
                    (text_ratio, 0.65),
                    (paragraph_ratio, 0.60),
                )
            ):
                flag = "warn"
                reason = "mild_downward_outlier"

            row["size_ratio_vs_group"] = round(size_ratio, 4) if size_ratio is not None else ""
            row["text_ratio_vs_group"] = round(text_ratio, 4) if text_ratio is not None else ""
            row["paragraph_ratio_vs_group"] = (
                round(paragraph_ratio, 4) if paragraph_ratio is not None else ""
            )
            row["html_integrity_flag"] = flag
            row["html_integrity_reason"] = reason

    summary = {
        "total_a2_files": len(rows),
        "html_integrity_flag_counts": dict(Counter(row["html_integrity_flag"] for row in rows)),
    }
    return rows, summary


def parse_a4_rows(a4_path: Path) -> list[dict]:
    csv_files = iter_files(a4_path, [".csv"])
    output_rows = []
    last_start_by_event = {}
    row_index_by_event = Counter()

    for csv_path in csv_files:
        for row in load_csv_rows(csv_path):
            event_id = infer_a4_event_id(row, csv_path.stem)
            row_index_by_event[event_id] += 1
            row_index = row_index_by_event[event_id]

            official_text = normalize_text(row.get("official_text", ""))
            asr_text = normalize_text(row.get("asr_matched_text", ""))
            start_sec = safe_float(row.get("start_sec"))
            end_sec = safe_float(row.get("end_sec"))
            match_score = safe_float(row.get("match_score"))
            block_avg_score = safe_float(row.get("block_avg_score"))
            overall_tfidf = safe_float(row.get("overall_TFIDF"))

            hard_fail_reasons = []
            overlap_warn = False
            if start_sec is None or end_sec is None:
                hard_fail_reasons.append("missing_time")
            else:
                if start_sec < 0 or end_sec < 0:
                    hard_fail_reasons.append("negative_time")
                if end_sec <= start_sec:
                    hard_fail_reasons.append("non_positive_duration")

            previous_start = last_start_by_event.get(event_id)
            if previous_start is not None and start_sec is not None and start_sec < previous_start:
                hard_fail_reasons.append("non_monotonic_start")
            if previous_start is not None and start_sec is not None and start_sec == previous_start:
                overlap_warn = True
            if start_sec is not None:
                last_start_by_event[event_id] = start_sec

            similarity = text_similarity(official_text, asr_text)
            overlap = token_f1(official_text, asr_text)

            output_rows.append(
                {
                    "source_file": str(csv_path.resolve()),
                    "source_stem": csv_path.stem,
                    "event_id": event_id,
                    "row_index_within_event": row_index,
                    "sentence_id": row.get("sentence_id", ""),
                    "start_sec": "" if start_sec is None else start_sec,
                    "end_sec": "" if end_sec is None else end_sec,
                    "duration_sec": "" if start_sec is None or end_sec is None else round(end_sec - start_sec, 4),
                    "match_score": "" if match_score is None else match_score,
                    "block_avg_score": "" if block_avg_score is None else block_avg_score,
                    "overall_TFIDF": "" if overall_tfidf is None else overall_tfidf,
                    "official_text_length": len(official_text),
                    "asr_text_length": len(asr_text),
                    "text_similarity": round(similarity, 6),
                    "token_f1": round(overlap, 6),
                    "hard_fail_reason": "|".join(hard_fail_reasons),
                    "hard_fail_flag": bool(hard_fail_reasons),
                    "overlap_warn_flag": overlap_warn,
                }
            )

    return output_rows


def classify_a4_rows(rows: list[dict], args: argparse.Namespace) -> None:
    for row in rows:
        match_score = safe_float(row["match_score"])
        block_avg_score = safe_float(row["block_avg_score"])
        similarity = safe_float(row["text_similarity"]) or 0.0
        has_official_text = row["official_text_length"] > 0
        hard_fail = bool(row["hard_fail_flag"])

        strict_pass = (
            has_official_text
            and not hard_fail
            and match_score is not None
            and block_avg_score is not None
            and match_score >= args.strict_match_score
            and block_avg_score >= args.strict_block_score
            and similarity >= args.strict_text_similarity
        )

        broad_pass = has_official_text and not hard_fail and (
            (
                match_score is not None
                and match_score >= args.broad_match_score
                and (block_avg_score is None or block_avg_score >= args.broad_block_score)
            )
            or similarity >= args.broad_text_similarity
        )

        row["strict_pass"] = strict_pass
        row["broad_pass"] = broad_pass


def summarise_a4_events(rows: list[dict], args: argparse.Namespace) -> tuple[list[dict], dict]:
    by_event = defaultdict(list)
    for row in rows:
        by_event[row["event_id"]].append(row)

    event_rows = []
    for event_id, event_items in by_event.items():
        strict_items = [row for row in event_items if row["strict_pass"]]
        broad_items = [row for row in event_items if row["broad_pass"]]
        match_scores = [safe_float(row["match_score"]) for row in event_items]
        match_scores = [score for score in match_scores if score is not None]

        def span_seconds(items: list[dict]) -> float:
            starts = [safe_float(row["start_sec"]) for row in items]
            ends = [safe_float(row["end_sec"]) for row in items]
            starts = [value for value in starts if value is not None]
            ends = [value for value in ends if value is not None]
            if not starts or not ends:
                return 0.0
            return max(ends) - min(starts)

        strict_span = span_seconds(strict_items)
        broad_span = span_seconds(broad_items)

        if len(strict_items) >= args.strict_min_rows and strict_span >= args.strict_min_span_sec:
            subset = "strict"
        elif len(broad_items) >= args.broad_min_rows and broad_span >= args.broad_min_span_sec:
            subset = "broad"
        else:
            subset = "reject"

        event_rows.append(
            {
                "event_id": event_id,
                "source_files": "|".join(sorted({row["source_stem"] for row in event_items})),
                "total_rows": len(event_items),
                "strict_rows": len(strict_items),
                "broad_rows": len(broad_items),
                "strict_row_share": round(len(strict_items) / len(event_items), 4) if event_items else 0.0,
                "broad_row_share": round(len(broad_items) / len(event_items), 4) if event_items else 0.0,
                "strict_span_sec": round(strict_span, 4),
                "broad_span_sec": round(broad_span, 4),
                "median_match_score": round(statistics.median(match_scores), 4) if match_scores else "",
                "min_match_score": round(min(match_scores), 4) if match_scores else "",
                "hard_fail_rows": sum(1 for row in event_items if row["hard_fail_flag"]),
                "overlap_warn_rows": sum(1 for row in event_items if row["overlap_warn_flag"]),
                "qc_subset": subset,
            }
        )

    summary = {
        "total_a4_events": len(event_rows),
        "total_a4_rows": len(rows),
        "row_level_counts": {
            "strict_pass_rows": sum(1 for row in rows if row["strict_pass"]),
            "broad_pass_rows": sum(1 for row in rows if row["broad_pass"]),
            "hard_fail_rows": sum(1 for row in rows if row["hard_fail_flag"]),
        },
        "event_subset_counts": dict(Counter(row["qc_subset"] for row in event_rows)),
    }
    return event_rows, summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    a2_rows, a2_summary = run_a2_qc(args.a2_dir.resolve())
    write_csv(output_dir / "a2_html_qc.csv", a2_rows)

    a4_rows = parse_a4_rows(args.a4_path.resolve())
    classify_a4_rows(a4_rows, args)
    a4_event_rows, a4_summary = summarise_a4_events(a4_rows, args)
    write_csv(output_dir / "a4_row_qc.csv", a4_rows)
    write_csv(output_dir / "a4_event_qc.csv", a4_event_rows)

    summary = {
        "a2": a2_summary,
        "a4": a4_summary,
        "thresholds": {
            "strict_match_score": args.strict_match_score,
            "broad_match_score": args.broad_match_score,
            "strict_block_score": args.strict_block_score,
            "broad_block_score": args.broad_block_score,
            "strict_text_similarity": args.strict_text_similarity,
            "broad_text_similarity": args.broad_text_similarity,
            "strict_min_rows": args.strict_min_rows,
            "broad_min_rows": args.broad_min_rows,
            "strict_min_span_sec": args.strict_min_span_sec,
            "broad_min_span_sec": args.broad_min_span_sec,
        },
    }
    write_json(output_dir / "initial_qc_summary.json", summary)

    print(f"Wrote QC outputs to {output_dir}")


if __name__ == "__main__":
    main()

