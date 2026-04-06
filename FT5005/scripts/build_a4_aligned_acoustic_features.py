#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from dj30_qc_utils import (
    build_event_path_lookup,
    iter_files,
    load_csv_rows,
    normalize_event_key_text,
    safe_float,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract A4-aligned sentence-level acoustic features from A3 audio with openSMILE."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument("--a4-row-qc-csv", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/audio_sentence_aligned_real"),
    )
    parser.add_argument(
        "--filter-mode",
        choices=["all", "strict", "broad"],
        default="strict",
        help="Which A4 QC slice to extract.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["eGeMAPSv02", "ComParE_2016"],
        default="eGeMAPSv02",
    )
    parser.add_argument(
        "--feature-level",
        choices=["Functionals", "LowLevelDescriptors"],
        default="Functionals",
    )
    parser.add_argument("--min-duration-sec", type=float, default=0.5)
    parser.add_argument("--winsorize-quantile", type=float, default=0.01)
    parser.add_argument(
        "--wav-cache-dir",
        type=Path,
        default=None,
        help="Cache full-event WAV conversions here before sentence-level extraction.",
    )
    return parser.parse_args()


def load_panel_event_lookup(path: Path) -> dict[str, dict[str, str]]:
    lookup = {}
    for row in load_csv_rows(path.resolve()):
        event_key = normalize_event_key_text(row.get("event_key", ""))
        if not event_key:
            continue
        item = dict(row)
        item["event_key"] = event_key
        lookup[event_key] = item
    return lookup


def should_keep_row(row: dict[str, str], filter_mode: str, min_duration_sec: float) -> tuple[bool, str]:
    duration_sec = safe_float(row.get("duration_sec"))
    if duration_sec is None or duration_sec < min_duration_sec:
        return False, "too_short"
    start_sec = safe_float(row.get("start_sec"))
    end_sec = safe_float(row.get("end_sec"))
    if start_sec is None or end_sec is None or end_sec <= start_sec:
        return False, "bad_timing"
    if filter_mode == "strict" and row.get("strict_pass", "").lower() != "true":
        return False, "strict_filter"
    if filter_mode == "broad" and row.get("broad_pass", "").lower() != "true":
        return False, "broad_filter"
    return True, ""


def load_extraction_rows(
    qc_path: Path,
    panel_lookup: dict[str, dict[str, str]],
    filter_mode: str,
    min_duration_sec: float,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows = []
    skip_reasons = defaultdict(int)
    for row in load_csv_rows(qc_path.resolve()):
        event_key = normalize_event_key_text(row.get("event_id", "") or row.get("source_stem", ""))
        if event_key not in panel_lookup:
            skip_reasons["event_not_in_panel"] += 1
            continue
        keep, reason = should_keep_row(row, filter_mode, min_duration_sec)
        if not keep:
            skip_reasons[reason] += 1
            continue
        item = dict(row)
        item["event_key"] = event_key
        item["ticker"] = panel_lookup[event_key].get("ticker", "")
        item["year"] = panel_lookup[event_key].get("year", "")
        item["quarter"] = panel_lookup[event_key].get("quarter", "")
        rows.append(item)
    rows.sort(
        key=lambda row: (
            row["event_key"],
            safe_float(row.get("row_index_within_event")) or 0.0,
            safe_float(row.get("start_sec")) or 0.0,
        )
    )
    return rows, dict(skip_reasons)


def build_smile(feature_set_name: str, feature_level_name: str):
    try:
        import opensmile
    except Exception as exc:  # pragma: no cover - env-specific dependency
        raise SystemExit(
            "openSMILE is required for build_a4_aligned_acoustic_features.py. "
            "Install the 'opensmile' package in the runtime environment."
        ) from exc

    feature_set = getattr(opensmile.FeatureSet, feature_set_name)
    feature_level = getattr(opensmile.FeatureLevel, feature_level_name)
    return opensmile.Smile(feature_set=feature_set, feature_level=feature_level)


def convert_audio_to_wav(source_path: Path, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    wav_path = cache_dir / f"{source_path.stem}.wav"
    if wav_path.exists():
        return wav_path
    try:
        import audiofile
    except Exception as exc:  # pragma: no cover - env-specific dependency
        raise SystemExit(
            "audiofile is required for WAV caching in build_a4_aligned_acoustic_features.py."
        ) from exc
    converted = audiofile.convert_to_wav(str(source_path), outfile=str(wav_path), overwrite=False)
    return Path(converted)


def infer_feature_columns(frame) -> list[str]:
    columns = []
    for column in frame.columns:
        text = str(column)
        if text not in {"sentence_id", "event_key", "ticker", "year", "quarter"}:
            columns.append(text)
    return columns


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = min(max(q, 0.0), 1.0) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def winsorize(values: list[float], q: float) -> list[float]:
    if not values:
        return []
    if q <= 0.0:
        return list(values)
    ordered = sorted(values)
    low = quantile(ordered, q)
    high = quantile(ordered, 1.0 - q)
    return [min(max(value, low), high) for value in values]


def weighted_mean(values: list[float], weights: list[float]) -> float:
    total = sum(weights)
    if not values or total <= 0:
        return 0.0
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def aggregate_event_rows(rows: list[dict[str, str]], feature_columns: list[str], winsor_q: float) -> dict[str, str]:
    out = {
        "event_key": rows[0]["event_key"],
        "ticker": rows[0].get("ticker", ""),
        "year": rows[0].get("year", ""),
        "quarter": rows[0].get("quarter", ""),
        "aligned_audio_sentence_count": str(len(rows)),
    }
    durations = [max(safe_float(row.get("duration_sec")) or 0.0, 0.0) for row in rows]
    out["aligned_audio_duration_sum_sec"] = f"{sum(durations):.6f}"
    out["aligned_audio_duration_mean_sec"] = f"{statistics.mean(durations):.6f}" if durations else ""
    out["aligned_audio_duration_median_sec"] = (
        f"{statistics.median(durations):.6f}" if durations else ""
    )

    for feature in feature_columns:
        values = [safe_float(row.get(feature)) for row in rows]
        clean = [value for value in values if value is not None and math.isfinite(value)]
        if not clean:
            out[f"{feature}_mean"] = ""
            out[f"{feature}_median"] = ""
            out[f"{feature}_std"] = ""
            out[f"{feature}_winsor_mean"] = ""
            out[f"{feature}_duration_weighted_mean"] = ""
            continue
        out[f"{feature}_mean"] = f"{statistics.mean(clean):.6f}"
        out[f"{feature}_median"] = f"{statistics.median(clean):.6f}"
        out[f"{feature}_std"] = f"{statistics.pstdev(clean):.6f}" if len(clean) > 1 else "0.000000"
        winsorized = winsorize(clean, winsor_q)
        out[f"{feature}_winsor_mean"] = f"{statistics.mean(winsorized):.6f}"

        paired = [
            (safe_float(row.get(feature)), max(safe_float(row.get("duration_sec")) or 0.0, 0.0))
            for row in rows
        ]
        paired = [(value, weight) for value, weight in paired if value is not None and math.isfinite(value)]
        if paired:
            p_values = [value for value, _ in paired]
            p_weights = [weight if weight > 0 else 1e-6 for _, weight in paired]
            out[f"{feature}_duration_weighted_mean"] = f"{weighted_mean(p_values, p_weights):.6f}"
        else:
            out[f"{feature}_duration_weighted_mean"] = ""

    return out


def summarize_sentence_rows(
    extracted_rows: list[dict[str, str]],
    failures: list[dict[str, str]],
    skip_reasons: dict[str, int],
    feature_columns: list[str],
    filter_mode: str,
    feature_set_name: str,
    feature_level_name: str,
) -> dict[str, object]:
    event_keys = sorted({row["event_key"] for row in extracted_rows})
    duration_values = [safe_float(row.get("duration_sec")) for row in extracted_rows]
    duration_values = [value for value in duration_values if value is not None and math.isfinite(value)]
    return {
        "filter_mode": filter_mode,
        "feature_set": feature_set_name,
        "feature_level": feature_level_name,
        "num_sentence_rows": len(extracted_rows),
        "num_event_keys": len(event_keys),
        "feature_column_count": len(feature_columns),
        "skip_reasons": skip_reasons,
        "num_failures": len(failures),
        "failures": failures[:200],
        "sentence_duration_mean_sec": round(statistics.mean(duration_values), 4) if duration_values else 0.0,
        "sentence_duration_median_sec": round(statistics.median(duration_values), 4) if duration_values else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_cache_dir = (
        args.wav_cache_dir.resolve()
        if args.wav_cache_dir is not None
        else (output_dir / "wav_cache").resolve()
    )

    panel_lookup = load_panel_event_lookup(args.panel_csv)
    audio_lookup, duplicate_audio_candidates = build_event_path_lookup(
        iter_files(args.a3_dir.resolve(), [".mp3"])
    )
    target_rows, skip_reasons = load_extraction_rows(
        args.a4_row_qc_csv,
        panel_lookup,
        args.filter_mode,
        args.min_duration_sec,
    )

    smile = build_smile(args.feature_set, args.feature_level)
    sentence_rows = []
    failures = []
    feature_columns: list[str] = []

    for index, row in enumerate(target_rows, start=1):
        event_key = row["event_key"]
        audio_path = audio_lookup.get(event_key)
        if audio_path is None:
            failures.append(
                {
                    "event_key": event_key,
                    "sentence_id": row.get("sentence_id", ""),
                    "reason": "missing_audio",
                }
            )
            continue
        processed_audio_path = audio_path
        if audio_path.suffix.lower() != ".wav":
            try:
                processed_audio_path = convert_audio_to_wav(audio_path, wav_cache_dir)
            except Exception as exc:  # pragma: no cover - conversion edge cases
                failures.append(
                    {
                        "event_key": event_key,
                        "sentence_id": row.get("sentence_id", ""),
                        "reason": "wav_cache_failed",
                        "error": str(exc),
                    }
                )
                continue
        start_sec = safe_float(row.get("start_sec"))
        end_sec = safe_float(row.get("end_sec"))
        try:
            frame = smile.process_file(str(processed_audio_path), start=start_sec, end=end_sec)
        except Exception as exc:  # pragma: no cover - runtime extraction edge cases
            failures.append(
                {
                    "event_key": event_key,
                    "sentence_id": row.get("sentence_id", ""),
                    "reason": "extract_failed",
                    "error": str(exc),
                }
            )
            continue

        if frame.empty:
            failures.append(
                {
                    "event_key": event_key,
                    "sentence_id": row.get("sentence_id", ""),
                    "reason": "empty_frame",
                }
            )
            continue

        if not feature_columns:
            feature_columns = infer_feature_columns(frame)

        feature_row = {str(key): value for key, value in frame.iloc[0].to_dict().items()}
        out_row = {
            "event_key": event_key,
            "ticker": row.get("ticker", ""),
            "year": row.get("year", ""),
            "quarter": row.get("quarter", ""),
            "sentence_id": row.get("sentence_id", ""),
            "row_index_within_event": row.get("row_index_within_event", ""),
            "source_file": row.get("source_file", ""),
            "source_stem": row.get("source_stem", ""),
            "audio_path": str(audio_path),
            "processed_audio_path": str(processed_audio_path),
            "start_sec": row.get("start_sec", ""),
            "end_sec": row.get("end_sec", ""),
            "duration_sec": row.get("duration_sec", ""),
            "match_score": row.get("match_score", ""),
            "overall_TFIDF": row.get("overall_TFIDF", ""),
            "strict_pass": row.get("strict_pass", ""),
            "broad_pass": row.get("broad_pass", ""),
            "overlap_warn_flag": row.get("overlap_warn_flag", ""),
            "hard_fail_flag": row.get("hard_fail_flag", ""),
        }
        for feature_name in feature_columns:
            value = feature_row.get(feature_name)
            out_row[feature_name] = "" if value is None else f"{float(value):.6f}"
        sentence_rows.append(out_row)

        if index % 250 == 0 or index == len(target_rows):
            print(f"processed {index}/{len(target_rows)} sentence rows", flush=True)

    event_rows = []
    grouped_rows = defaultdict(list)
    for row in sentence_rows:
        grouped_rows[row["event_key"]].append(row)
    for event_key in sorted(grouped_rows):
        event_rows.append(
            aggregate_event_rows(
                grouped_rows[event_key],
                feature_columns,
                args.winsorize_quantile,
            )
        )

    write_csv(output_dir / "sentence_aligned_acoustic_features.csv", sentence_rows)
    write_csv(output_dir / "event_aligned_acoustic_features.csv", event_rows)
    write_json(
        output_dir / "aligned_acoustic_features_summary.json",
        {
            **summarize_sentence_rows(
                sentence_rows,
                failures,
                skip_reasons,
                feature_columns,
                args.filter_mode,
                args.feature_set,
                args.feature_level,
            ),
            "duplicate_audio_candidates": duplicate_audio_candidates,
            "num_event_rows": len(event_rows),
            "wav_cache_dir": str(wav_cache_dir),
        },
    )
    print(
        json.dumps(
            {
                "num_sentence_rows": len(sentence_rows),
                "num_event_rows": len(event_rows),
                "num_failures": len(failures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
