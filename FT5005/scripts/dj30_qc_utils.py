#!/usr/bin/env python3

from __future__ import annotations

import csv
import datetime as dt
import html
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable


csv.field_size_limit(1024 * 1024 * 128)


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


def split_sentences(text: str) -> list[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", normalized)
    return [part.strip() for part in parts if part.strip()]


def safe_float(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def token_f1(text_a: str, text_b: str) -> float:
    tokens_a = re.findall(r"[A-Za-z0-9]+", (text_a or "").lower())
    tokens_b = re.findall(r"[A-Za-z0-9]+", (text_b or "").lower())
    if not tokens_a or not tokens_b:
        return 0.0
    counter_a = Counter(tokens_a)
    counter_b = Counter(tokens_b)
    overlap = sum(min(counter_a[token], counter_b[token]) for token in counter_a)
    precision = overlap / sum(counter_a.values())
    recall = overlap / sum(counter_b.values())
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def text_similarity(text_a: str, text_b: str) -> float:
    return SequenceMatcher(None, normalize_text(text_a), normalize_text(text_b)).ratio()


def detect_filename_metadata(path: Path) -> dict[str, str]:
    stem = path.stem
    upper_stem = stem.upper()
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", upper_stem) if token]

    ticker = ""
    for token in tokens:
        if re.fullmatch(r"[A-Z]{1,5}", token):
            ticker = token
            break

    year = ""
    quarter = ""
    patterns = [
        re.search(r"(20\d{2})[^A-Za-z0-9]?Q([1-4])", upper_stem),
        re.search(r"Q([1-4])[^A-Za-z0-9]?(20\d{2})", upper_stem),
        re.search(r"([1-4])Q[^A-Za-z0-9]?(20\d{2})", upper_stem),
    ]
    for match in patterns:
        if match:
            groups = match.groups()
            if len(groups[0]) == 4:
                year = groups[0]
                quarter = f"Q{groups[1]}"
            else:
                year = groups[1]
                quarter = f"Q{groups[0]}"
            break

    return {
        "ticker": ticker,
        "year": year,
        "quarter": quarter,
        "event_key": build_event_key(ticker, year, quarter),
    }


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
        date_text = scheduled_date_text
        time_text = scheduled_time_text
        try:
            scheduled_dt = dt.datetime.strptime(
                f"{date_text} {time_text}", "%B %d, %Y %I:%M %p"
            )
            scheduled_iso = scheduled_dt.isoformat(sep=" ")
        except ValueError:
            scheduled_iso = ""

    # Some incomplete Seeking Alpha pages keep only a malformed placeholder date
    # like "January 1, 0000" but still expose the useful scheduled time at the end.
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
        "paragraph_count": str(len(paragraphs)),
    }


def build_event_key(ticker: str, year: str, quarter: str) -> str:
    if ticker and year and quarter:
        return f"{ticker}_{year}_{quarter}"
    return ""


def normalize_event_key_text(text: str) -> str:
    value = normalize_text(text).upper()
    if not value:
        return ""

    patterns = [
        re.search(r"([A-Z]{1,5})[_-](20\d{2})Q([1-4])(?:[_-]ALIGNED)?$", value),
        re.search(r"([A-Z]{1,5})[_-](20\d{2})[_-]Q([1-4])(?:[_-]ALIGNED)?$", value),
        re.search(r"([A-Z]{1,5})[_-](20\d{2})Q([1-4])", value),
        re.search(r"([A-Z]{1,5})[_-](20\d{2})[_-]Q([1-4])", value),
    ]
    for match in patterns:
        if match:
            ticker, year, quarter_num = match.groups()
            return build_event_key(ticker, year, f"Q{quarter_num}")

    return value


def iter_files(path: Path, suffixes: Iterable[str]) -> list[Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    if path.is_file():
        return [path] if path.suffix.lower() in suffix_set else []
    return sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in suffix_set
    )


def build_event_path_lookup(paths: Iterable[Path]) -> tuple[dict[str, Path], dict[str, list[str]]]:
    grouped = {}
    duplicates = {}
    for path in paths:
        event_key = detect_filename_metadata(path)["event_key"]
        if not event_key:
            continue
        grouped.setdefault(event_key, []).append(path)

    lookup = {}
    for event_key, candidates in grouped.items():
        ranked = sorted(
            candidates,
            key=lambda path: (
                path.stem.endswith("_2"),
                len(path.stem),
                path.name,
            ),
        )
        lookup[event_key] = ranked[0]
        if len(ranked) > 1:
            duplicates[event_key] = [path.name for path in ranked]

    return lookup, duplicates


def infer_a4_event_id(row: dict[str, str], source_stem: str) -> str:
    candidate_columns = [
        "event_id",
        "transcriptid",
        "transcript_id",
        "call_id",
        "callid",
        "file_name",
        "filename",
        "file",
        "source_file",
    ]
    for column in candidate_columns:
        value = normalize_text(row.get(column, ""))
        if value:
            return value

    sentence_id = normalize_text(row.get("sentence_id", ""))
    if sentence_id:
        for delimiter in ("__", "_", "-"):
            if delimiter in sentence_id:
                head, tail = sentence_id.rsplit(delimiter, 1)
                if tail.isdigit() and head:
                    return head

    return source_stem


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
