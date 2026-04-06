# Data Restore From Google Drive (2026-04-06)

## Bottom line

The GitHub repository does not contain:

- the raw DJ30 package from Google Drive
- or the rebuilt core intermediate artifacts such as:
  - `results/panel_real/event_modeling_panel.csv`
  - `results/features_real/event_text_audio_features.csv`

But the repository **does** contain the processing pipeline needed to rebuild them.

So the current blocker is not missing methodology.
The blocker is that the raw Drive package is not present on this machine.

## Original source package

The original Drive package matches the repository data inventory:

- `A1.ECC_Text_Json_DJ30`
- `A2.ECC_Text_html_DJ30`
- `A3.ECC_Audio_DJ30`
- `A4.ECC_Timestamp_DJ30`
- `B3.Compustat Index Constituents`
- `C1.Surprise_DJ30`
- `C2.AnalystForecast_DJ30`
- `D.Stock_5min_DJ30`

Reference:

- [data_inventory.md](F:/FT5005/docs/data_inventory.md)

## Recommended local raw-data layout

Place the restored Drive package under:

- `F:\FT5005\data\raw\dj30_shared\`

Recommended structure:

- `F:\FT5005\data\raw\dj30_shared\A1.ECC_Text_Json_DJ30\`
- `F:\FT5005\data\raw\dj30_shared\A2.ECC_Text_html_DJ30\`
- `F:\FT5005\data\raw\dj30_shared\A3.ECC_Audio_DJ30\`
- `F:\FT5005\data\raw\dj30_shared\A4.ECC_Timestamp_DJ30\`
- `F:\FT5005\data\raw\dj30_shared\B.Compustat\`
- `F:\FT5005\data\raw\dj30_shared\C.Analyst\`
- `F:\FT5005\data\raw\dj30_shared\D.Stock_5min_DJ30\`

For the current modeling mainline, the minimum required sets are:

- `A1`
- `A2`
- `A3`
- `A4`
- `C1`
- `C2`
- `D`

## Rebuild chain

The repository now supports this reconstruction flow:

1. scan raw files
2. run initial `A2/A4` QC
3. build intraday targets from `A2 + A4 + D`
4. build the modeling panel from targets + `C1/C2/A1/A2-QC/A4-QC`
5. build event-level text/audio features
6. build `Q&A` benchmark features
7. run modeling experiments

## Script-by-script rebuild

### Step 1. File manifest

Script:

- `scripts/build_event_manifest.py`

Expected outputs:

- `results/qc_real/file_manifest.csv`
- `results/qc_real/file_manifest_summary.json`

### Step 2. Initial QC

Script:

- `scripts/run_initial_qc.py`

Expected outputs:

- `results/qc_real/a2_html_qc.csv`
- `results/qc_real/a4_row_qc.csv`
- `results/qc_real/a4_event_qc.csv`
- `results/qc_real/initial_qc_summary.json`

### Step 3. Intraday targets

Script:

- `scripts/build_intraday_targets.py`

Expected outputs:

- `results/targets_real/event_intraday_targets.csv`
- `results/targets_real/event_intraday_targets_summary.json`

### Step 4. Modeling panel

Script:

- `scripts/build_modeling_panel.py`

Expected outputs:

- `results/panel_real/event_modeling_panel.csv`
- `results/panel_real/event_modeling_panel_summary.json`

### Step 5. Event text/audio features

Script:

- `scripts/build_event_text_audio_features.py`

Expected outputs:

- `results/features_real/event_text_audio_features.csv`
- `results/features_real/event_text_audio_features_summary.json`

### Step 6. Q&A benchmark features

Script:

- `scripts/build_qa_benchmark_features.py`

Expected outputs:

- `results/qa_benchmark_features_real/qa_benchmark_features.csv`
- `results/qa_benchmark_features_real/qa_benchmark_features_summary.json`

## Why the current machine looks "data-empty"

On this machine right now:

- the raw Drive folders are missing under `F:\FT5005\data\...`
- `results/panel_real/event_modeling_panel.csv` is missing
- `results/features_real/event_text_audio_features.csv` is missing

So this is not a repository-without-processing.
It is a repository-without-the-restored-data package and rebuilt core artifacts.

## Fastest next step

Once the Drive folders are restored locally, run:

1. `scripts/restore_drive_pipeline.ps1`
2. `scripts/run_i_pog_qa_benchmark_suite.py`

The PowerShell helper rebuilds:

- `qc_real`
- `targets_real`
- `panel_real`
- `features_real`
- `qa_benchmark_features_real`

Then the benchmark suite can immediately produce the current `I-POG-QA` main table and ablation outputs.
