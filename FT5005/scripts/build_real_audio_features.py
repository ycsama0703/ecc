#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from dj30_qc_utils import build_event_path_lookup, iter_files, load_csv_rows, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract real call-level audio features from A3 mp3 files."
    )
    parser.add_argument("--panel-csv", type=Path, required=True)
    parser.add_argument("--a3-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/audio_real")
    )
    parser.add_argument("--chunk-seconds", type=float, default=20.0)
    parser.add_argument("--num-chunks", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=6)
    return parser.parse_args()


def zero_crossing_rate(waveform: np.ndarray) -> float:
    if waveform.size <= 1:
        return 0.0
    crossings = (waveform[1:] >= 0) != (waveform[:-1] >= 0)
    return float(np.mean(crossings))


def summarize_chunk(
    waveform: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if waveform.size == 0:
        return {}

    abs_wave = np.abs(waveform)
    rms = float(np.sqrt(np.mean(np.square(waveform)) + 1e-12))
    spectrum = np.abs(np.fft.rfft(waveform))
    freq = np.fft.rfftfreq(waveform.size, d=1.0 / sample_rate)
    spec_sum = float(np.sum(spectrum)) + 1e-12
    centroid = float(np.sum(freq * spectrum) / spec_sum)
    bandwidth = float(np.sqrt(np.sum(((freq - centroid) ** 2) * spectrum) / spec_sum))
    cumulative = np.cumsum(spectrum)
    rolloff_threshold = 0.85 * cumulative[-1] if cumulative.size else 0.0
    rolloff_idx = int(np.searchsorted(cumulative, rolloff_threshold)) if cumulative.size else 0
    rolloff_hz = float(freq[min(rolloff_idx, max(len(freq) - 1, 0))]) if freq.size else 0.0
    flatness = float(np.exp(np.mean(np.log(spectrum + 1e-12))) / (np.mean(spectrum) + 1e-12))
    low_band = float(np.sum(spectrum[freq < 300.0]) / spec_sum)
    mid_band = float(np.sum(spectrum[(freq >= 300.0) & (freq < 3000.0)]) / spec_sum)
    high_band = float(np.sum(spectrum[freq >= 3000.0]) / spec_sum)

    features = {
        "chunk_rms": rms,
        "chunk_abs_mean": float(np.mean(abs_wave)),
        "chunk_std": float(np.std(waveform)),
        "chunk_silence_share": float(np.mean(abs_wave < 0.01)),
        "chunk_zcr": zero_crossing_rate(waveform),
        "chunk_max_abs": float(np.max(abs_wave)),
        "chunk_dynamic_range": float(np.quantile(waveform, 0.95) - np.quantile(waveform, 0.05)),
        "chunk_spectral_centroid_hz": centroid,
        "chunk_spectral_bandwidth_hz": bandwidth,
        "chunk_spectral_rolloff85_hz": rolloff_hz,
        "chunk_spectral_flatness": flatness,
        "chunk_low_band_share": low_band,
        "chunk_mid_band_share": mid_band,
        "chunk_high_band_share": high_band,
    }
    return features


def aggregate_chunk_features(chunk_features: list[dict[str, float]]) -> dict[str, float]:
    if not chunk_features:
        return {}
    keys = sorted(chunk_features[0].keys())
    output = {}
    for key in keys:
        values = [item[key] for item in chunk_features if key in item and math.isfinite(item[key])]
        if not values:
            output[f"{key}_mean"] = 0.0
            output[f"{key}_std"] = 0.0
            continue
        output[f"{key}_mean"] = float(np.mean(values))
        output[f"{key}_std"] = float(np.std(values))
    return output


def chunk_offsets(total_frames: int, chunk_frames: int, num_chunks: int) -> list[int]:
    if total_frames <= chunk_frames or num_chunks <= 1:
        return [0]
    max_offset = total_frames - chunk_frames
    offsets = np.linspace(0, max_offset, num_chunks)
    seen = []
    for value in offsets:
        offset = int(round(float(value)))
        if not seen or offset != seen[-1]:
            seen.append(offset)
    return seen


def extract_event_features(audio_path: Path, chunk_seconds: float, num_chunks: int) -> dict[str, float]:
    with sf.SoundFile(str(audio_path)) as audio_file:
        sample_rate = int(audio_file.samplerate)
        num_frames = int(len(audio_file))
        num_channels = int(audio_file.channels)
        duration_sec = float(num_frames / sample_rate) if sample_rate else 0.0
        chunk_frames = max(1, int(round(chunk_seconds * sample_rate)))
        offsets = chunk_offsets(num_frames, chunk_frames, num_chunks)

        chunk_feature_rows: list[dict[str, float]] = []
        for offset in offsets:
            audio_file.seek(offset)
            chunk = audio_file.read(frames=chunk_frames, dtype="float32", always_2d=True)
            if chunk.size == 0:
                continue
            waveform = np.mean(chunk, axis=1)
            chunk_feature_rows.append(summarize_chunk(waveform, sample_rate))

    file_size_bytes = float(audio_path.stat().st_size)
    metadata = {
        "audio_duration_sec": duration_sec,
        "audio_sample_rate": float(sample_rate),
        "audio_num_channels": float(num_channels),
        "audio_num_frames": float(num_frames),
        "audio_file_size_bytes": file_size_bytes,
        "audio_bitrate_kbps_est": 0.0 if duration_sec <= 0 else float(file_size_bytes * 8.0 / duration_sec / 1000.0),
        "audio_chunk_count": float(len(chunk_feature_rows)),
    }
    metadata.update(aggregate_chunk_features(chunk_feature_rows))
    return metadata


def process_audio_event(task: tuple[str, str, float, int]) -> dict[str, object]:
    event_key, audio_path_text, chunk_seconds, num_chunks = task
    audio_path = Path(audio_path_text)
    row = {"event_key": event_key, "has_real_audio": 1.0, "audio_path": str(audio_path)}
    try:
        row.update(
            extract_event_features(
                audio_path=audio_path,
                chunk_seconds=chunk_seconds,
                num_chunks=num_chunks,
            )
        )
        return {"row": row, "failure": None}
    except Exception as exc:  # pragma: no cover - corrupted mp3 edge case
        row["has_real_audio"] = 0.0
        row["audio_error"] = str(exc)
        return {"row": row, "failure": {"event_key": event_key, "audio_path": str(audio_path), "error": str(exc)}}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_rows = list(load_csv_rows(args.panel_csv.resolve()))
    event_keys = sorted({row.get("event_key", "") for row in panel_rows if row.get("event_key")})

    audio_lookup, duplicates = build_event_path_lookup(iter_files(args.a3_dir.resolve(), [".mp3"]))

    rows = []
    matched = 0
    failures = []
    unmatched_rows = []
    tasks = []
    for event_key in event_keys:
        audio_path = audio_lookup.get(event_key)
        if audio_path is None:
            unmatched_rows.append({"event_key": event_key, "has_real_audio": 0.0})
        else:
            matched += 1
            tasks.append((event_key, str(audio_path), args.chunk_seconds, args.num_chunks))

    with cf.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_audio_event, task) for task in tasks]
        for index, future in enumerate(cf.as_completed(futures), start=1):
            result = future.result()
            rows.append(result["row"])
            if result["failure"] is not None:
                failures.append(result["failure"])
            if index % 25 == 0 or index == len(tasks):
                print(f"processed {index}/{len(tasks)} matched audio events", flush=True)

    rows.extend(unmatched_rows)
    rows.sort(key=lambda item: item["event_key"])

    write_csv(output_dir / "event_real_audio_features.csv", rows)
    write_json(
        output_dir / "event_real_audio_features_summary.json",
        {
            "requested_event_keys": len(event_keys),
            "matched_audio_files": matched,
            "missing_audio_files": len(event_keys) - matched,
            "audio_failures": failures,
            "duplicate_audio_candidates": duplicates,
            "config": {
                "chunk_seconds": args.chunk_seconds,
                "num_chunks": args.num_chunks,
            },
        },
    )
    print(
        json.dumps(
            {
                "requested_event_keys": len(event_keys),
                "matched_audio_files": matched,
                "missing_audio_files": len(event_keys) - matched,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
