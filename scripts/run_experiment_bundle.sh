#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_experiment_bundle.sh [options]

Runs the full experiment bundle:
  1. build_event_manifest.py
  2. build_processed_panel.py
  3. build_time_split.py
  4. training/fit_full_pipeline.py
  5. scripts/run_benchmarks.py
  6. scripts/build_per_ticker_metrics.py
  7. scripts/build_results_table.py

Core options:
  --tag TAG                         Experiment tag used in default output paths.
  --split-version VERSION           Split version tag.
  --main-run-id RUN_ID              Run id for the full pipeline.
  --bench-run-id RUN_ID             Run id for benchmarks.
  --python BIN                      Python executable to use. Default: python

Data / panel options:
  --audit-dir DIR                   Default: outputs/audit/raw_data
  --data-root DIR                   Default: data/raw
  --a2-dir DIR                      Default: data/raw/A2.ECC_Text_html_DJ30
  --c1-csv PATH                     Default: data/raw/C.Analyst/C1.Surprise_DJ30.csv
  --c2-csv PATH                     Default: data/raw/C.Analyst/C2.AnalystForecast_DJ30.csv
  --manifest-dir DIR                Default: data/processed/event_manifest
  --panel-dir DIR                   Default: data/processed/panel_<tag>
  --timing-mode MODE                assumed | actual. Default: actual
  --call-time HH:MM:SS              Default: 16:00:00
  --pre-call-minutes N              Default: 60
  --post-call-minutes N             Default: 60
  --min-window-bars N               Default: 8
  --historical-lookback-bars N      Default: 78
  --broad-match-score X             Default: 75.0
  --broad-text-similarity X         Default: 0.65

Split / evaluation options:
  --split-csv PATH                  Default: data/splits/time_split_<tag>.csv
  --split-summary-json PATH         Default: data/splits/time_split_<tag>_summary.json
  --train-ratio X                   Default: 0.70
  --val-ratio X                     Default: 0.15
  --experiment-dir DIR              Default: outputs/<tag>
  --benchmark-dir DIR               Default: outputs/benchmarks_<tag>
  --ecc-feature-mode MODE           ecc_only only. Default: ecc_only
  --abstention-min-coverage X       Default: 0.30
  --abstention-target-coverage X    Optional explicit target coverage for kappa selection
  --coverage-sweep-targets LIST     Default: 0.2,0.4,0.6,0.8,unconstrained

Benchmark / FinBERT options:
  --finbert-device DEVICE           Default: cuda
  --finbert-max-length N            Default: 256
  --finbert-max-chunks N            Default: 4
  --finbert-pooling MODE            cls | mean. Default: mean

Skip flags:
  --skip-manifest
  --skip-panel
  --skip-split
  --skip-main
  --skip-benchmarks
  --skip-per-ticker
  --skip-results-table

Example:
  bash scripts/run_experiment_bundle.sh \
    --tag experiment_a_timefix \
    --split-version timefix_a_v1 \
    --main-run-id mainA01 \
    --bench-run-id benchA01 \
    --timing-mode actual
EOF
}

TAG="experiment_a_timefix"
SPLIT_VERSION="timefix_a_v1"
MAIN_RUN_ID="mainA01"
BENCH_RUN_ID="benchA01"
PYTHON_BIN="python"

AUDIT_DIR="outputs/audit/raw_data"
DATA_ROOT="data/raw"
A2_DIR="data/raw/A2.ECC_Text_html_DJ30"
C1_CSV="data/raw/C.Analyst/C1.Surprise_DJ30.csv"
C2_CSV="data/raw/C.Analyst/C2.AnalystForecast_DJ30.csv"
MANIFEST_DIR="data/processed/event_manifest"
PANEL_DIR=""
TIMING_MODE="actual"
CALL_TIME="16:00:00"
PRE_CALL_MINUTES="60"
POST_CALL_MINUTES="60"
MIN_WINDOW_BARS="8"
HISTORICAL_LOOKBACK_BARS="78"
BROAD_MATCH_SCORE="75.0"
BROAD_TEXT_SIMILARITY="0.65"

SPLIT_CSV=""
SPLIT_SUMMARY_JSON=""
TRAIN_RATIO="0.70"
VAL_RATIO="0.15"
EXPERIMENT_DIR=""
BENCHMARK_DIR=""
ECC_FEATURE_MODE="ecc_only"
ABSTENTION_MIN_COVERAGE="0.30"
ABSTENTION_TARGET_COVERAGE=""
COVERAGE_SWEEP_TARGETS="0.2,0.4,0.6,0.8,unconstrained"

FINBERT_DEVICE="cuda"
FINBERT_MAX_LENGTH="256"
FINBERT_MAX_CHUNKS="4"
FINBERT_POOLING="mean"

SKIP_MANIFEST=0
SKIP_PANEL=0
SKIP_SPLIT=0
SKIP_MAIN=0
SKIP_BENCHMARKS=0
SKIP_PER_TICKER=0
SKIP_RESULTS_TABLE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="$2"; shift 2 ;;
    --split-version) SPLIT_VERSION="$2"; shift 2 ;;
    --main-run-id) MAIN_RUN_ID="$2"; shift 2 ;;
    --bench-run-id) BENCH_RUN_ID="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --audit-dir) AUDIT_DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --a2-dir) A2_DIR="$2"; shift 2 ;;
    --c1-csv) C1_CSV="$2"; shift 2 ;;
    --c2-csv) C2_CSV="$2"; shift 2 ;;
    --manifest-dir) MANIFEST_DIR="$2"; shift 2 ;;
    --panel-dir) PANEL_DIR="$2"; shift 2 ;;
    --timing-mode) TIMING_MODE="$2"; shift 2 ;;
    --call-time) CALL_TIME="$2"; shift 2 ;;
    --pre-call-minutes) PRE_CALL_MINUTES="$2"; shift 2 ;;
    --post-call-minutes) POST_CALL_MINUTES="$2"; shift 2 ;;
    --min-window-bars) MIN_WINDOW_BARS="$2"; shift 2 ;;
    --historical-lookback-bars) HISTORICAL_LOOKBACK_BARS="$2"; shift 2 ;;
    --broad-match-score) BROAD_MATCH_SCORE="$2"; shift 2 ;;
    --broad-text-similarity) BROAD_TEXT_SIMILARITY="$2"; shift 2 ;;
    --split-csv) SPLIT_CSV="$2"; shift 2 ;;
    --split-summary-json) SPLIT_SUMMARY_JSON="$2"; shift 2 ;;
    --train-ratio) TRAIN_RATIO="$2"; shift 2 ;;
    --val-ratio) VAL_RATIO="$2"; shift 2 ;;
    --experiment-dir) EXPERIMENT_DIR="$2"; shift 2 ;;
    --benchmark-dir) BENCHMARK_DIR="$2"; shift 2 ;;
    --ecc-feature-mode) ECC_FEATURE_MODE="$2"; shift 2 ;;
    --abstention-min-coverage) ABSTENTION_MIN_COVERAGE="$2"; shift 2 ;;
    --abstention-target-coverage) ABSTENTION_TARGET_COVERAGE="$2"; shift 2 ;;
    --coverage-sweep-targets) COVERAGE_SWEEP_TARGETS="$2"; shift 2 ;;
    --finbert-device) FINBERT_DEVICE="$2"; shift 2 ;;
    --finbert-max-length) FINBERT_MAX_LENGTH="$2"; shift 2 ;;
    --finbert-max-chunks) FINBERT_MAX_CHUNKS="$2"; shift 2 ;;
    --finbert-pooling) FINBERT_POOLING="$2"; shift 2 ;;
    --skip-manifest) SKIP_MANIFEST=1; shift ;;
    --skip-panel) SKIP_PANEL=1; shift ;;
    --skip-split) SKIP_SPLIT=1; shift ;;
    --skip-main) SKIP_MAIN=1; shift ;;
    --skip-benchmarks) SKIP_BENCHMARKS=1; shift ;;
    --skip-per-ticker) SKIP_PER_TICKER=1; shift ;;
    --skip-results-table) SKIP_RESULTS_TABLE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "$ECC_FEATURE_MODE" != "ecc_only" ]]; then
  echo "Main experiment bundle only supports --ecc-feature-mode ecc_only." >&2
  echo "Hybrid ecc_plus_market_controls is now appendix-only and excluded from the clean mainline." >&2
  exit 1
fi

if [[ -z "$PANEL_DIR" ]]; then
  PANEL_DIR="data/processed/panel_${TAG}"
fi
if [[ -z "$SPLIT_CSV" ]]; then
  SPLIT_CSV="data/splits/time_split_${TAG}.csv"
fi
if [[ -z "$SPLIT_SUMMARY_JSON" ]]; then
  SPLIT_SUMMARY_JSON="data/splits/time_split_${TAG}_summary.json"
fi
if [[ -z "$EXPERIMENT_DIR" ]]; then
  EXPERIMENT_DIR="outputs/${TAG}"
fi
if [[ -z "$BENCHMARK_DIR" ]]; then
  BENCHMARK_DIR="outputs/benchmarks_${TAG}"
fi

run_step() {
  echo
  echo "==> $1"
  shift
  "$@"
}

echo "Experiment bundle config"
echo "  python: $PYTHON_BIN"
echo "  tag: $TAG"
echo "  split_version: $SPLIT_VERSION"
echo "  main_run_id: $MAIN_RUN_ID"
echo "  bench_run_id: $BENCH_RUN_ID"
echo "  panel_dir: $PANEL_DIR"
echo "  split_csv: $SPLIT_CSV"
echo "  experiment_dir: $EXPERIMENT_DIR"
echo "  benchmark_dir: $BENCHMARK_DIR"
echo "  timing_mode: $TIMING_MODE"
echo "  ecc_feature_mode: $ECC_FEATURE_MODE"
echo "  abstention_target_coverage: ${ABSTENTION_TARGET_COVERAGE:-<none>}"

if [[ "$SKIP_MANIFEST" -eq 0 ]]; then
  run_step "Build event manifest" \
    "$PYTHON_BIN" scripts/build_event_manifest.py \
      --audit-dir "$AUDIT_DIR" \
      --data-root "$DATA_ROOT" \
      --a2-dir "$A2_DIR" \
      --output-dir "$MANIFEST_DIR"
fi

if [[ "$SKIP_PANEL" -eq 0 ]]; then
  run_step "Build processed panel" \
    "$PYTHON_BIN" scripts/build_processed_panel.py \
      --manifest-csv "$MANIFEST_DIR/min_pipeline_events.csv" \
      --output-dir "$PANEL_DIR" \
      --c1-csv "$C1_CSV" \
      --c2-csv "$C2_CSV" \
      --timing-mode "$TIMING_MODE" \
      --call-time "$CALL_TIME" \
      --pre-call-minutes "$PRE_CALL_MINUTES" \
      --post-call-minutes "$POST_CALL_MINUTES" \
      --min-window-bars "$MIN_WINDOW_BARS" \
      --historical-lookback-bars "$HISTORICAL_LOOKBACK_BARS" \
      --broad-match-score "$BROAD_MATCH_SCORE" \
      --broad-text-similarity "$BROAD_TEXT_SIMILARITY"
fi

if [[ "$SKIP_SPLIT" -eq 0 ]]; then
  run_step "Build time split" \
    "$PYTHON_BIN" scripts/build_time_split.py \
      --panel-csv "$PANEL_DIR/processed_panel.csv" \
      --output-csv "$SPLIT_CSV" \
      --summary-json "$SPLIT_SUMMARY_JSON" \
      --split-version "$SPLIT_VERSION" \
      --train-ratio "$TRAIN_RATIO" \
      --val-ratio "$VAL_RATIO"
fi

if [[ "$SKIP_MAIN" -eq 0 ]]; then
  MAIN_ARGS=(
    "$PYTHON_BIN" training/fit_full_pipeline.py
      --panel "$PANEL_DIR/processed_panel.csv"
      --split "$SPLIT_CSV"
      --output-dir "$EXPERIMENT_DIR"
      --split-version "$SPLIT_VERSION"
      --run-id "$MAIN_RUN_ID"
      --tune-market-prior
      --ecc-feature-mode "$ECC_FEATURE_MODE"
      --abstention-min-coverage "$ABSTENTION_MIN_COVERAGE"
      --coverage-sweep-targets "$COVERAGE_SWEEP_TARGETS"
  )
  if [[ -n "$ABSTENTION_TARGET_COVERAGE" ]]; then
    MAIN_ARGS+=(--abstention-target-coverage "$ABSTENTION_TARGET_COVERAGE")
  fi
  run_step "Run full pipeline" \
    "${MAIN_ARGS[@]}"
fi

if [[ "$SKIP_BENCHMARKS" -eq 0 ]]; then
  run_step "Run benchmarks" \
    "$PYTHON_BIN" scripts/run_benchmarks.py \
      --panel "$PANEL_DIR/processed_panel.csv" \
      --split "$SPLIT_CSV" \
      --output-dir "$BENCHMARK_DIR" \
      --split-version "$SPLIT_VERSION" \
      --run-id "$BENCH_RUN_ID" \
      --tune-market-baselines \
      --finbert-device "$FINBERT_DEVICE" \
      --finbert-max-length "$FINBERT_MAX_LENGTH" \
      --finbert-max-chunks "$FINBERT_MAX_CHUNKS" \
      --finbert-pooling "$FINBERT_POOLING"
fi

if [[ "$SKIP_PER_TICKER" -eq 0 ]]; then
  run_step "Build per-ticker metrics" \
    "$PYTHON_BIN" scripts/build_per_ticker_metrics.py \
      --panel-csv "$PANEL_DIR/processed_panel.csv" \
      --experiment-dir "$EXPERIMENT_DIR" \
      --split-version "$SPLIT_VERSION" \
      --run-id "$MAIN_RUN_ID"
fi

if [[ "$SKIP_RESULTS_TABLE" -eq 0 ]]; then
  run_step "Build comprehensive results table" \
    "$PYTHON_BIN" scripts/build_results_table.py \
      --experiment-dir "$EXPERIMENT_DIR" \
      --benchmark-dir "$BENCHMARK_DIR" \
      --split-version "$SPLIT_VERSION" \
      --main-run-id "$MAIN_RUN_ID" \
      --bench-run-id "$BENCH_RUN_ID"
fi

echo
echo "Experiment bundle complete"
echo "  main outputs: $EXPERIMENT_DIR"
echo "  benchmark outputs: $BENCHMARK_DIR"
