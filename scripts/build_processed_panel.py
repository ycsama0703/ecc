#!/usr/bin/env python3
"""
Build a minimal processed panel for the root-level PREC pipeline.

Current scope:
    - consumes event manifest built from A1 + A4 + D
    - assumes after-hours calls start at 16:00 on event_date
    - builds market features, target, lightweight ECC text features,
      and proxy features

This is an MVP panel builder intended to make the current pipeline
trainable on the currently available raw assets.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import html
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta
from difflib import SequenceMatcher
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?")

FORWARD_WORDS = {
    "expect",
    "guidance",
    "guide",
    "outlook",
    "forecast",
    "future",
    "quarter",
    "year",
    "demand",
}
RISK_WORDS = {
    "risk",
    "uncertain",
    "uncertainty",
    "pressure",
    "volatility",
    "headwind",
    "challenging",
    "macro",
}

SECTOR_CODE = {
    "AAPL": 1.0,   # Information Technology
    "AMGN": 2.0,   # Health Care
    "AMZN": 3.0,   # Consumer Discretionary
    "AXP": 4.0,    # Financials
    "BA": 5.0,     # Industrials
    "CAT": 5.0,
    "CSCO": 1.0,
    "DIS": 6.0,    # Communication Services / media proxy
    "GS": 4.0,
    "HD": 3.0,
    "HON": 5.0,
    "IBM": 1.0,
    "JNJ": 2.0,
    "JPM": 4.0,
    "KO": 7.0,     # Consumer Staples
    "MCD": 3.0,
    "MMM": 5.0,
    "MRK": 2.0,
    "MSFT": 1.0,
    "NKE": 3.0,
    "NVDA": 1.0,
    "PG": 7.0,
    "SHW": 8.0,    # Materials
    "TRV": 4.0,
    "UNH": 2.0,
    "V": 4.0,
    "VZ": 6.0,
    "WMT": 7.0,
}


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


def parse_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def keyword_rate(tokens: list[str], lexicon: set[str]) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if token in lexicon)
    return hits / len(tokens)


def numeric_rate(tokens: list[str]) -> float:
    if not tokens:
        return 0.0
    hits = sum(1 for token in tokens if any(char.isdigit() for char in token))
    return hits / len(tokens)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))


def parse_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_event_key(ticker: str, year: str, quarter_num: str) -> str:
    quarter_num = str(quarter_num).strip().replace("Q", "")
    if ticker and year and quarter_num:
        return f"{ticker}_{year}Q{quarter_num}"
    return ""


def normalize_text(text: str) -> str:
    text = text or ""
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def strip_html_visible_text(raw_html: str) -> str:
    text = raw_html or ""
    text = re.sub(r"(?is)<(script|style)\b.*?>.*?</\1>", " ", text)
    text = re.sub(r"(?is)<br\s*/?>", "\n", text)
    text = re.sub(r"(?is)</p\s*>", "\n", text)
    text = re.sub(r"(?is)</div\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    return normalize_text(text)


def extract_html_paragraphs(raw_html: str) -> list[str]:
    paragraphs = []
    for match in re.finditer(r"(?is)<p[^>]*>(.*?)</p>", raw_html or ""):
        paragraph = strip_html_visible_text(match.group(1))
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def count_visible_paragraphs(raw_html: str) -> int:
    paragraph_tags = len(re.findall(r"(?i)<p\b", raw_html or ""))
    if paragraph_tags:
        return paragraph_tags
    visible_text = strip_html_visible_text(raw_html or "")
    return len([line for line in re.split(r"[\r\n]+", visible_text) if line.strip()])


def parse_a2_scheduled_metadata(raw_html: str) -> dict[str, str]:
    paragraphs = extract_html_paragraphs(raw_html)
    header = paragraphs[0] if paragraphs else ""
    pattern = re.compile(
        r"(?P<date>[A-Za-z]+\s+\d{1,2},\s+\d{4})\s+"
        r"(?P<time>\d{1,2}:\d{2}\s+[AP]M)\s+"
        r"(?P<tz>ET|EST|EDT)\b"
    )
    match = pattern.search(header)

    scheduled_iso = ""
    scheduled_date_text = match.group("date") if match else ""
    scheduled_time_text = match.group("time") if match else ""
    scheduled_tz_text = match.group("tz") if match else ""
    if match:
        try:
            scheduled_dt = datetime.strptime(
                f"{scheduled_date_text} {scheduled_time_text}",
                "%B %d, %Y %I:%M %p",
            )
            scheduled_iso = scheduled_dt.isoformat(sep=" ")
        except ValueError:
            scheduled_iso = ""

    if not scheduled_time_text:
        time_only_match = re.search(r"(?P<time>\d{1,2}:\d{2}\s+[AP]M)\s+(?P<tz>ET|EST|EDT)\b", header)
        if time_only_match:
            scheduled_time_text = time_only_match.group("time")
            scheduled_tz_text = time_only_match.group("tz")

    return {
        "header_paragraph": header,
        "scheduled_date_text": scheduled_date_text,
        "scheduled_time_text": scheduled_time_text,
        "scheduled_tz_text": scheduled_tz_text,
        "scheduled_datetime_iso": scheduled_iso,
    }


def text_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, normalize_text(text_a), normalize_text(text_b)).ratio()


def broad_pass_a4_row(
    row: dict,
    broad_match_score: float,
    broad_text_similarity: float,
) -> bool:
    official_text = normalize_text(row.get("official_text", ""))
    if not official_text:
        return False

    start_sec = parse_float(row.get("start_sec"))
    end_sec = parse_float(row.get("end_sec"))
    if start_sec is None or end_sec is None or start_sec < 0 or end_sec <= start_sec:
        return False

    match_score = parse_float(row.get("match_score"))
    similarity = text_similarity(
        official_text.lower(),
        normalize_text(row.get("asr_matched_text", "")).lower(),
    )

    return (match_score is not None and match_score >= broad_match_score) or similarity >= broad_text_similarity


def compute_call_duration_seconds(
    a4_path: Path,
    broad_match_score: float,
    broad_text_similarity: float,
) -> tuple[float | None, int]:
    max_end_sec = None
    kept_rows = 0
    for row in read_csv(a4_path):
        if not broad_pass_a4_row(row, broad_match_score, broad_text_similarity):
            continue
        end_sec = parse_float(row.get("end_sec"))
        if end_sec is None:
            continue
        kept_rows += 1
        max_end_sec = end_sec if max_end_sec is None else max(max_end_sec, end_sec)
    return max_end_sec, kept_rows


def resolve_call_timing(
    manifest_row: dict,
    fallback_call_time: time,
    timing_mode: str,
    broad_match_score: float,
    broad_text_similarity: float,
) -> dict:
    event_date = datetime.fromisoformat(manifest_row["event_date"]).date()
    fallback_call_dt = datetime.combine(event_date, fallback_call_time)
    timing = {
        "scheduled_datetime": fallback_call_dt,
        "call_end_datetime": fallback_call_dt,
        "call_duration_sec": 0.0,
        "a4_kept_rows_for_duration": 0,
        "timing_source": "assumed_event_date_16_00",
    }

    if timing_mode != "actual":
        return timing

    a2_abspath = manifest_row.get("a2_abspath")
    a4_abspath = manifest_row.get("a4_abspath")
    if not a2_abspath or not a4_abspath:
        return timing

    try:
        raw_html = Path(a2_abspath).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return timing

    scheduled_meta = parse_a2_scheduled_metadata(raw_html)
    scheduled_iso = scheduled_meta["scheduled_datetime_iso"]
    if not scheduled_iso and scheduled_meta["scheduled_time_text"]:
        try:
            fallback_dt = datetime.strptime(
                f"{event_date.isoformat()} {scheduled_meta['scheduled_time_text']}",
                "%Y-%m-%d %I:%M %p",
            )
            scheduled_iso = fallback_dt.isoformat(sep=" ")
        except ValueError:
            scheduled_iso = ""
    if not scheduled_iso:
        return timing

    duration_sec, kept_rows = compute_call_duration_seconds(
        Path(a4_abspath),
        broad_match_score=broad_match_score,
        broad_text_similarity=broad_text_similarity,
    )
    if duration_sec is None:
        return timing

    scheduled_dt = datetime.fromisoformat(scheduled_iso)
    return {
        "scheduled_datetime": scheduled_dt,
        "call_end_datetime": scheduled_dt + timedelta(seconds=duration_sec),
        "call_duration_sec": float(duration_sec),
        "a4_kept_rows_for_duration": int(kept_rows),
        "timing_source": "a2_scheduled_plus_a4_duration",
    }


def scheduled_hour_value(scheduled_datetime: datetime) -> float:
    return round(
        scheduled_datetime.hour
        + scheduled_datetime.minute / 60.0
        + scheduled_datetime.second / 3600.0,
        6,
    )


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
    return build_event_key(ticker, str(year_int), str(quarter_int))


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
    return build_event_key(ticker, str(year_int), str(quarter_int))


def build_c1_lookup(path: Path) -> dict[str, dict]:
    lookup = {}
    for row in read_csv(path):
        event_key = event_key_from_c1_row(row)
        if not event_key:
            continue
        revenue_actual = parse_float(row.get("CIQ Revenue Actual ($000)"))
        revenue_estimate = parse_float(row.get("CIQ Revenue Estimate ($000)"))
        ebitda_actual = parse_float(row.get("CIQ EBITDA Actual ($000)"))
        ebitda_estimate = parse_float(row.get("CIQ EBITDA Estimate ($000)"))
        eps_actual = parse_float(row.get("CIQ EPS (GAAP) Actual ($)"))
        eps_estimate = parse_float(row.get("CIQ EPS (GAAP) Estimate ($)"))

        def surprise(actual: float | None, estimate: float | None) -> tuple[float | None, float | None]:
            if actual is None or estimate is None:
                return None, None
            diff = actual - estimate
            pct = None if estimate == 0 else diff / abs(estimate)
            return diff, pct

        revenue_diff, revenue_pct = surprise(revenue_actual, revenue_estimate)
        ebitda_diff, ebitda_pct = surprise(ebitda_actual, ebitda_estimate)
        eps_diff, eps_pct = surprise(eps_actual, eps_estimate)

        lookup[event_key] = {
            "revenue_actual": revenue_actual,
            "revenue_estimate": revenue_estimate,
            "revenue_surprise": revenue_diff,
            "revenue_surprise_pct": revenue_pct,
            "ebitda_actual": ebitda_actual,
            "ebitda_estimate": ebitda_estimate,
            "ebitda_surprise": ebitda_diff,
            "ebitda_surprise_pct": ebitda_pct,
            "eps_gaap_actual": eps_actual,
            "eps_gaap_estimate": eps_estimate,
            "eps_gaap_surprise": eps_diff,
            "eps_gaap_surprise_pct": eps_pct,
        }
    return lookup


def build_c2_lookup(path: Path) -> dict[str, dict]:
    lookup = {}
    for row in read_csv(path):
        event_key = event_key_from_c2_row(row)
        if not event_key:
            continue
        lookup[event_key] = {
            "analyst_eps_norm_mean": parse_float(row.get("eps_norm_estimate_CIQ_EPS_Normalized_Est\n$")),
            "analyst_eps_norm_num_est": parse_float(
                row.get("eps_norm_stddev_estimate_CIQ_EPS_Normalized_Est_-_#_of_Est\nactual")
            ),
            "analyst_eps_norm_std": parse_float(
                row.get("eps_norm_stddev_estimate_CIQ_EPS_Normalized_Est_-_Std_Dev\n$")
            ),
            "analyst_revenue_mean": parse_float(row.get("revenue_estimate_CIQ_Revenue_Est\n$000")),
            "analyst_revenue_median": parse_float(row.get("revenue_estimate_CIQ_Revenue_Est_Med\n$000")),
            "analyst_revenue_num_est": parse_float(
                row.get("revenue_stddev_estimate_CIQ_Revenue_Est_-_#_of_Est\nactual")
            ),
            "analyst_revenue_std": parse_float(
                row.get("revenue_stddev_estimate_CIQ_Revenue_Est_-_Std_Dev\n$000")
            ),
            "analyst_net_income_mean": parse_float(row.get("net_income_estimate_CIQ_Net_Income_Est\n$000")),
            "analyst_net_income_num_est": parse_float(
                row.get("net_income_stddev_estimate_CIQ_Net_Income_Est_-_#_of_Est\nactual")
            ),
            "analyst_net_income_std": parse_float(
                row.get("net_income_stddev_estimate_CIQ_Net_Income_Est_-_Std_Dev\n$000")
            ),
        }
    return lookup


def median_or_none(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def ratio_or_none(value: float, baseline: float | None) -> float | None:
    if baseline in (None, 0):
        return None
    return value / baseline


def detect_filename_metadata(path: Path) -> dict[str, str]:
    stem = path.stem.upper()
    tokens = [token for token in re.split(r"[^A-Z0-9]+", stem) if token]

    ticker = ""
    for token in tokens:
        if re.fullmatch(r"[A-Z]{1,5}", token):
            ticker = token
            break

    year = ""
    quarter = ""
    quarter_match = re.search(r"(20\d{2})[^A-Z0-9]?Q([1-4])", stem)
    if not quarter_match:
        quarter_match = re.search(r"Q([1-4])[^A-Z0-9]?(20\d{2})", stem)
    if quarter_match:
        groups = quarter_match.groups()
        if len(groups[0]) == 4:
            year = groups[0]
            quarter = f"Q{groups[1]}"
        else:
            year = groups[1]
            quarter = f"Q{groups[0]}"

    return {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "event_key": f"{ticker}_{year}{quarter}" if ticker and year and quarter else "",
    }


def build_a2_qc_lookup(a2_dir: Path) -> dict[str, dict]:
    html_files = sorted(
        file_path
        for file_path in a2_dir.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in {".html", ".htm"}
    )
    rows = []
    for file_path in html_files:
        raw_html = file_path.read_text(encoding="utf-8", errors="ignore")
        visible_text = strip_html_visible_text(raw_html)
        meta = detect_filename_metadata(file_path)
        rows.append(
            {
                "path": str(file_path.resolve()),
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

    lookup = {}
    for ticker_rows in grouped.values():
        size_median = median_or_none([row["size_bytes"] for row in ticker_rows]) or global_size_median
        text_median = median_or_none([row["visible_text_length"] for row in ticker_rows]) or global_text_median
        paragraph_median = median_or_none([row["paragraph_count"] for row in ticker_rows]) or global_paragraph_median
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

            lookup[row["event_key"]] = {
                "html_integrity_flag": flag,
                "html_integrity_reason": reason,
                "a2_paragraph_count": row["paragraph_count"],
                "a2_visible_word_count": row["visible_word_count"],
                "a2_size_ratio_vs_group": round(size_ratio, 4) if size_ratio is not None else None,
                "a2_text_ratio_vs_group": round(text_ratio, 4) if text_ratio is not None else None,
            }
    return lookup


def extract_text_feature_bundle(a1_path: Path) -> dict:
    payload = parse_json(a1_path)
    components = payload.get("components") or []

    all_texts = []
    qa_texts = []
    question_texts = []
    answer_texts = []
    operator_components = 0
    management_components = 0
    speaker_names = set()

    for component in components:
        component_type = str(component.get("componenttypename") or "")
        text = str(component.get("text") or "").strip()
        personname = str(component.get("personname") or "").strip()
        companyofperson = str(component.get("companyofperson") or "").strip()

        if personname:
            speaker_names.add(personname)
        if companyofperson:
            management_components += 1
        if "Operator" in component_type:
            operator_components += 1

        if text:
            all_texts.append(text)
            if component_type in {"Question", "Answer"}:
                qa_texts.append(text)
            if component_type == "Question":
                question_texts.append(text)
            elif component_type == "Answer":
                answer_texts.append(text)

    all_tokens = tokenize(" ".join(all_texts))
    qa_tokens = tokenize(" ".join(qa_texts))
    question_tokens = tokenize(" ".join(question_texts))
    answer_tokens = tokenize(" ".join(answer_texts))

    total_words = len(all_tokens)
    qa_words = len(qa_tokens)
    question_words = len(question_tokens)
    answer_words = len(answer_tokens)
    component_count = max(len(components), 1)
    question_count = max(len(question_texts), 0)
    answer_count = max(len(answer_texts), 0)

    text_features = {
        "text_embedding_0": round(math.log1p(total_words), 8),
        "text_embedding_1": round(safe_divide(len(set(all_tokens)), total_words), 8),
        "text_embedding_2": round(safe_divide(question_count, component_count), 8),
        "text_embedding_3": round(safe_divide(answer_count, component_count), 8),
        "text_embedding_4": round(safe_divide(len(speaker_names), component_count), 8),
        "text_embedding_5": round(safe_divide(total_words, component_count), 8),
        "text_embedding_6": round(safe_divide(operator_components, component_count), 8),
        "text_embedding_7": round(keyword_rate(all_tokens, FORWARD_WORDS), 8),
        "text_embedding_8": round(keyword_rate(all_tokens, RISK_WORDS), 8),
        "text_embedding_9": round(numeric_rate(all_tokens), 8),
    }

    qa_features = {
        "qa_embedding_0": round(math.log1p(qa_words), 8),
        "qa_embedding_1": round(math.log1p(question_words), 8),
        "qa_embedding_2": round(math.log1p(answer_words), 8),
        "qa_embedding_3": round(safe_divide(question_words, answer_words + 1.0), 8),
        "qa_embedding_4": round(safe_divide(question_words, max(question_count, 1)), 8),
        "qa_embedding_5": round(safe_divide(answer_words, max(answer_count, 1)), 8),
        "qa_embedding_6": round(keyword_rate(qa_tokens, FORWARD_WORDS), 8),
        "qa_embedding_7": round(keyword_rate(qa_tokens, RISK_WORDS), 8),
        "qa_embedding_8": round(numeric_rate(qa_tokens), 8),
        "qa_embedding_9": round(safe_divide(question_count, question_count + answer_count), 8),
    }

    raw_stats = {
        "transcript_total_words": total_words,
        "transcript_total_components": len(components),
        "transcript_question_components": question_count,
        "transcript_answer_components": answer_count,
        "transcript_unique_speakers": len(speaker_names),
        "transcript_management_component_share": round(
            safe_divide(management_components, component_count), 8
        ),
        "qa_total_words": qa_words,
    }

    return {**raw_stats, **text_features, **qa_features}


def load_stock_rows(stock_path: Path) -> list[dict]:
    rows = []
    previous_close = None
    with stock_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = datetime.fromisoformat(row["DateTime"])
            close = float(row["Close"])
            volume = float(row["Volume"])
            log_return = None
            if previous_close is not None and previous_close > 0 and close > 0:
                log_return = math.log(close / previous_close)
            previous_close = close
            rows.append(
                {
                    "timestamp": timestamp,
                    "close": close,
                    "volume": volume,
                    "dollar_volume": close * volume,
                    "log_return": log_return,
                }
            )
    return rows


def aggregate_window(rows: list[dict]) -> dict:
    returns = [row["log_return"] for row in rows if row["log_return"] is not None]
    rv = sum(value * value for value in returns)
    vw_rv = sum(
        (row["log_return"] ** 2) * math.log1p(row["volume"])
        for row in rows
        if row["log_return"] is not None
    )
    return_sum = sum(returns)
    abs_return_sum = sum(abs(value) for value in returns)
    volume_sum = sum(row["volume"] for row in rows)
    dollar_volume_sum = sum(row["dollar_volume"] for row in rows)
    return {
        "bar_count": len(rows),
        "rv": rv,
        "vw_rv": vw_rv,
        "return_sum": return_sum,
        "abs_return_sum": abs_return_sum,
        "volume_sum": volume_sum,
        "dollar_volume_sum": dollar_volume_sum,
        "first_bar": rows[0]["timestamp"].isoformat(sep=" ") if rows else None,
        "last_bar": rows[-1]["timestamp"].isoformat(sep=" ") if rows else None,
    }


def slice_rows_between(
    stock_rows: list[dict],
    timestamps: list[datetime],
    start_dt: datetime,
    end_dt: datetime,
) -> list[dict]:
    start_idx = bisect.bisect_left(timestamps, start_dt)
    end_idx = bisect.bisect_left(timestamps, end_dt)
    return stock_rows[start_idx:end_idx]


def collect_historical_rows(
    stock_rows: list[dict],
    timestamps: list[datetime],
    before_dt: datetime,
    lookback_bars: int,
) -> list[dict]:
    end_idx = bisect.bisect_left(timestamps, before_dt)
    historical_rows = []

    for idx in range(end_idx - 1, -1, -1):
        row = stock_rows[idx]
        if row["log_return"] is None:
            continue
        historical_rows.append(row)
        if len(historical_rows) >= lookback_bars:
            break

    historical_rows.reverse()
    return historical_rows


def derive_proxy_features(manifest_row: dict) -> dict:
    transcript_coverage = clamp01(parse_float(manifest_row.get("timestamp_coverage")))
    alignment_score = clamp01(parse_float(manifest_row.get("mean_overall_tfidf")))
    audio_completeness = clamp01(parse_float(manifest_row.get("match_score_coverage")))

    return {
        "transcript_coverage": round(transcript_coverage, 8),
        "alignment_score": round(alignment_score, 8),
        "audio_completeness": round(audio_completeness, 8),
        "proxy_quality_mean": round(
            (transcript_coverage + alignment_score + audio_completeness) / 3.0, 8
        ),
    }


def build_panel_for_ticker(
    ticker: str,
    ticker_rows: list[dict],
    a2_qc_lookup: dict[str, dict],
    c1_lookup: dict[str, dict],
    c2_lookup: dict[str, dict],
    call_time: time,
    timing_mode: str,
    broad_match_score: float,
    broad_text_similarity: float,
    pre_call_minutes: int,
    post_call_minutes: int,
    min_window_bars: int,
    historical_lookback_bars: int,
) -> tuple[list[dict], list[dict]]:
    stock_rows = load_stock_rows(Path(ticker_rows[0]["stock_abspath"]))
    timestamps = [row["timestamp"] for row in stock_rows]
    panel_rows = []
    dropped_rows = []

    for manifest_row in ticker_rows:
        event_id = manifest_row["event_id"]
        timing = resolve_call_timing(
            manifest_row=manifest_row,
            fallback_call_time=call_time,
            timing_mode=timing_mode,
            broad_match_score=broad_match_score,
            broad_text_similarity=broad_text_similarity,
        )
        call_dt = timing["scheduled_datetime"]
        call_end_dt = timing["call_end_datetime"]
        pre_start = call_dt - timedelta(minutes=pre_call_minutes)
        post_end = call_end_dt + timedelta(minutes=post_call_minutes)
        event_key = build_event_key(
            ticker=manifest_row["ticker"],
            year=str(manifest_row["year"]),
            quarter_num=str(manifest_row["quarter"]),
        )

        pre_rows = slice_rows_between(stock_rows, timestamps, pre_start, call_dt)
        call_rows = slice_rows_between(stock_rows, timestamps, call_dt, call_end_dt)
        post_rows = slice_rows_between(stock_rows, timestamps, call_end_dt, post_end)
        historical_rows = collect_historical_rows(
            stock_rows=stock_rows,
            timestamps=timestamps,
            before_dt=pre_start,
            lookback_bars=historical_lookback_bars,
        )

        pre_stats = aggregate_window(pre_rows)
        call_stats = aggregate_window(call_rows)
        post_stats = aggregate_window(post_rows)
        historical_stats = aggregate_window(historical_rows)

        drop_reasons = []
        if pre_stats["bar_count"] < min_window_bars:
            drop_reasons.append("insufficient_pre_bars")
        if post_stats["bar_count"] < min_window_bars:
            drop_reasons.append("insufficient_post_bars")
        if historical_stats["bar_count"] == 0:
            drop_reasons.append("missing_historical_context")

        if drop_reasons:
            dropped_rows.append(
                    {
                        "event_id": event_id,
                        "ticker": ticker,
                        "event_date": manifest_row["event_date"],
                        "scheduled_datetime": call_dt.isoformat(sep=" "),
                        "call_end_datetime": call_end_dt.isoformat(sep=" "),
                        "timing_source": timing["timing_source"],
                        "pre_bar_count": pre_stats["bar_count"],
                        "post_bar_count": post_stats["bar_count"],
                        "historical_bar_count": historical_stats["bar_count"],
                        "drop_reasons": ",".join(drop_reasons),
                    }
            )
            continue

        text_features = extract_text_feature_bundle(Path(manifest_row["a1_abspath"]))
        proxy_features = derive_proxy_features(manifest_row)
        a2_qc_features = a2_qc_lookup.get(event_key, {})
        analyst_features = {
            **c1_lookup.get(event_key, {}),
            **c2_lookup.get(event_key, {}),
        }

        shock_minus_pre = post_stats["rv"] - pre_stats["rv"]
        historical_dollar_volume_mean = safe_divide(
            historical_stats["dollar_volume_sum"], historical_stats["bar_count"]
        )

        row = {
            "event_id": event_id,
            "ticker": ticker,
            "event_date": manifest_row["event_date"],
            "call_time": call_dt.time().isoformat(),
            "scheduled_datetime": call_dt.isoformat(sep=" "),
            "call_end_datetime": call_end_dt.isoformat(sep=" "),
            "call_duration_sec": round(timing["call_duration_sec"], 4),
            "call_duration_min": round(timing["call_duration_sec"] / 60.0, 6),
            "scheduled_hour_et": scheduled_hour_value(call_dt),
            "a4_kept_rows_for_duration": timing["a4_kept_rows_for_duration"],
            "sample_label": (
                "after_hours_timefix" if timing["timing_source"] == "a2_scheduled_plus_a4_duration" else "after_hours_assumed"
            ),
            "call_time_assumption": (
                None if timing["timing_source"] == "a2_scheduled_plus_a4_duration" else f"event_date_{call_time.isoformat()}"
            ),
            "timing_source": timing["timing_source"],
            "split_label": "",
            "shock_minus_pre": round(shock_minus_pre, 10),
            "RV_pre_60m": round(pre_stats["rv"], 10),
            "RV_post_60m": round(post_stats["rv"], 10),
            "pre_60m_rv": round(pre_stats["rv"], 10),
            "pre_60m_vw_rv": round(pre_stats["vw_rv"], 10),
            "pre_60m_volume_sum": round(pre_stats["volume_sum"], 4),
            "within_call_bar_count": call_stats["bar_count"],
            "within_call_rv": round(call_stats["rv"], 10),
            "within_call_vw_rv": round(call_stats["vw_rv"], 10),
            "within_call_volume_sum": round(call_stats["volume_sum"], 4),
            "within_call_first_bar": call_stats["first_bar"],
            "within_call_last_bar": call_stats["last_bar"],
            "post_call_60m_bar_count": post_stats["bar_count"],
            "post_call_60m_vw_rv": round(post_stats["vw_rv"], 10),
            "post_call_60m_volume_sum": round(post_stats["volume_sum"], 4),
            "pre_call_volatility": round(pre_stats["rv"], 10),
            "returns": round(pre_stats["return_sum"], 10),
            "volume": round(pre_stats["volume_sum"], 4),
            "firm_size": round(math.log1p(historical_dollar_volume_mean), 8),
            "sector": SECTOR_CODE.get(ticker, 0.0),
            "historical_volatility": round(historical_stats["rv"], 10),
            "pre_bar_count": pre_stats["bar_count"],
            "post_bar_count": post_stats["bar_count"],
            "historical_bar_count": historical_stats["bar_count"],
            "pre_window_first_bar": pre_stats["first_bar"],
            "pre_window_last_bar": pre_stats["last_bar"],
            "post_window_first_bar": post_stats["first_bar"],
            "post_window_last_bar": post_stats["last_bar"],
            "headline": manifest_row["headline"],
            "a1_relpath": manifest_row["a1_relpath"],
            "a2_relpath": manifest_row.get("a2_relpath"),
            "a4_relpath": manifest_row["a4_relpath"],
            "stock_relpath": manifest_row["stock_relpath"],
            "a1_abspath": manifest_row["a1_abspath"],
            "a2_abspath": manifest_row.get("a2_abspath"),
            "a4_abspath": manifest_row["a4_abspath"],
            "stock_abspath": manifest_row["stock_abspath"],
            "match_score_coverage_raw": manifest_row["match_score_coverage"],
            "timestamp_coverage_raw": manifest_row["timestamp_coverage"],
            "mean_overall_tfidf_raw": manifest_row["mean_overall_tfidf"],
            **a2_qc_features,
            **proxy_features,
            **text_features,
            **analyst_features,
        }
        panel_rows.append(row)

    return panel_rows, dropped_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a minimal processed PREC panel.")
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("data/processed/event_manifest/min_pipeline_events.csv"),
        help="Event manifest CSV to consume.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/panel"),
        help="Directory for processed panel outputs.",
    )
    parser.add_argument(
        "--c1-csv",
        type=Path,
        default=Path("data/raw/C.Analyst/C1.Surprise_DJ30.csv"),
        help="Analyst surprise CSV used for Experiment B controls.",
    )
    parser.add_argument(
        "--c2-csv",
        type=Path,
        default=Path("data/raw/C.Analyst/C2.AnalystForecast_DJ30.csv"),
        help="Analyst forecast CSV used for Experiment B controls.",
    )
    parser.add_argument(
        "--a2-dir",
        type=Path,
        default=Path("data/raw/A2.ECC_Text_html_DJ30"),
        help="A2 HTML directory used to derive html integrity QC flags.",
    )
    parser.add_argument(
        "--call-time",
        type=str,
        default="16:00:00",
        help="Assumed call start time in HH:MM[:SS] on event_date.",
    )
    parser.add_argument(
        "--timing-mode",
        choices=["assumed", "actual"],
        default="assumed",
        help="Use assumed event_date call time or recover timing from A2+A4 where available.",
    )
    parser.add_argument(
        "--pre-call-minutes",
        type=int,
        default=60,
        help="Minutes in the pre-call market window.",
    )
    parser.add_argument(
        "--post-call-minutes",
        type=int,
        default=60,
        help="Minutes in the post-call market window.",
    )
    parser.add_argument(
        "--min-window-bars",
        type=int,
        default=8,
        help="Minimum number of 5-minute bars required in both pre and post windows.",
    )
    parser.add_argument(
        "--historical-lookback-bars",
        type=int,
        default=78,
        help="Lookback bars used to compute historical volatility and liquidity controls.",
    )
    parser.add_argument(
        "--broad-match-score",
        type=float,
        default=75.0,
        help="A4 match_score threshold used when estimating call duration from aligned rows.",
    )
    parser.add_argument(
        "--broad-text-similarity",
        type=float,
        default=0.65,
        help="A4 text-similarity threshold used when estimating call duration from aligned rows.",
    )
    args = parser.parse_args()

    manifest_rows = read_csv(args.manifest_csv.resolve())
    usable_rows = [row for row in manifest_rows if row["usable_for_min_pipeline"] == "1"]
    a2_qc_lookup = build_a2_qc_lookup(args.a2_dir.resolve()) if args.a2_dir.resolve().exists() else {}
    c1_lookup = build_c1_lookup(args.c1_csv.resolve()) if args.c1_csv.resolve().exists() else {}
    c2_lookup = build_c2_lookup(args.c2_csv.resolve()) if args.c2_csv.resolve().exists() else {}

    call_time = time.fromisoformat(args.call_time)
    grouped = defaultdict(list)
    for row in usable_rows:
        grouped[row["ticker"]].append(row)

    panel_rows = []
    dropped_rows = []
    for ticker, ticker_rows in sorted(grouped.items()):
        ticker_panel_rows, ticker_dropped_rows = build_panel_for_ticker(
            ticker=ticker,
            ticker_rows=ticker_rows,
            a2_qc_lookup=a2_qc_lookup,
            c1_lookup=c1_lookup,
            c2_lookup=c2_lookup,
            call_time=call_time,
            timing_mode=args.timing_mode,
            broad_match_score=args.broad_match_score,
            broad_text_similarity=args.broad_text_similarity,
            pre_call_minutes=args.pre_call_minutes,
            post_call_minutes=args.post_call_minutes,
            min_window_bars=args.min_window_bars,
            historical_lookback_bars=args.historical_lookback_bars,
        )
        panel_rows.extend(ticker_panel_rows)
        dropped_rows.extend(ticker_dropped_rows)

    panel_rows.sort(key=lambda row: (row["event_date"], row["ticker"], row["event_id"]))

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_path = output_dir / "processed_panel.csv"
    dropped_path = output_dir / "dropped_events.csv"
    summary_path = output_dir / "processed_panel_summary.json"

    write_csv(panel_path, panel_rows)
    write_csv(dropped_path, dropped_rows)

    summary = {
        "manifest_csv": str(args.manifest_csv.resolve()),
        "output_dir": str(output_dir),
        "timing_mode": args.timing_mode,
        "call_time_assumption": f"event_date_{call_time.isoformat()}",
        "pre_call_minutes": args.pre_call_minutes,
        "post_call_minutes": args.post_call_minutes,
        "min_window_bars": args.min_window_bars,
        "historical_lookback_bars": args.historical_lookback_bars,
        "broad_match_score": args.broad_match_score,
        "broad_text_similarity": args.broad_text_similarity,
        "c1_csv": str(args.c1_csv.resolve()),
        "c2_csv": str(args.c2_csv.resolve()),
        "a2_dir": str(args.a2_dir.resolve()),
        "a2_qc_coverage": sum(1 for row in panel_rows if row.get("html_integrity_flag")),
        "c1_coverage": sum(1 for row in panel_rows if row.get("revenue_surprise_pct") is not None),
        "c2_coverage": sum(1 for row in panel_rows if row.get("analyst_eps_norm_num_est") is not None),
        "manifest_usable_events": len(usable_rows),
        "panel_event_count": len(panel_rows),
        "dropped_event_count": len(dropped_rows),
        "dropped_reason_counts": dict(
            Counter(
                reason
                for row in dropped_rows
                for reason in row["drop_reasons"].split(",")
                if reason
            )
        ),
        "panel_ticker_count": len({row["ticker"] for row in panel_rows}),
        "timing_source_counts": dict(Counter(row["timing_source"] for row in panel_rows)),
        "html_integrity_flag_counts": dict(Counter((row.get("html_integrity_flag") or "") for row in panel_rows)),
        "min_event_date": panel_rows[0]["event_date"] if panel_rows else None,
        "max_event_date": panel_rows[-1]["event_date"] if panel_rows else None,
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print("Processed panel build complete")
    print(f"  manifest usable events: {len(usable_rows)}")
    print(f"  panel rows: {len(panel_rows)}")
    print(f"  dropped rows: {len(dropped_rows)}")
    print(f"  output: {panel_path}")


if __name__ == "__main__":
    main()
