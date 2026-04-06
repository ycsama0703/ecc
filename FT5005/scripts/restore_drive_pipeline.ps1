param(
    [string]$RepoRoot = "F:\FT5005",
    [string]$RawRoot = "F:\FT5005\data\raw\dj30_shared",
    [string]$PythonExe = "python",
    [switch]$RunQaBenchmark = $true
)

$ErrorActionPreference = "Stop"

function Resolve-ExistingPath {
    param(
        [string[]]$Candidates,
        [string]$Label
    )
    foreach ($candidate in $Candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw "Missing required path for $Label. Checked: $($Candidates -join '; ')"
}

$scriptsDir = Join-Path $RepoRoot "scripts"
$resultsDir = Join-Path $RepoRoot "results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

$a1Dir = Resolve-ExistingPath -Candidates @(
    (Join-Path $RawRoot "A1.ECC_Text_Json_DJ30")
) -Label "A1"

$a2Dir = Resolve-ExistingPath -Candidates @(
    (Join-Path $RawRoot "A2.ECC_Text_html_DJ30")
) -Label "A2"

$a3Dir = Resolve-ExistingPath -Candidates @(
    (Join-Path $RawRoot "A3.ECC_Audio_DJ30")
) -Label "A3"

$a4Path = Resolve-ExistingPath -Candidates @(
    (Join-Path $RawRoot "A4.ECC_Timestamp_DJ30"),
    (Join-Path $RawRoot "A4.ECC_Timestamp_DJ30.csv")
) -Label "A4"

$dDir = Resolve-ExistingPath -Candidates @(
    (Join-Path $RawRoot "D.Stock_5min_DJ30")
) -Label "D"

$c1Csv = Resolve-ExistingPath -Candidates @(
    (Join-Path $RawRoot "C.Analyst\C1.Surprise_DJ30.csv"),
    (Join-Path $RawRoot "C.Analyst\Surprise_DJ30.csv"),
    (Join-Path $RawRoot "C1.Surprise_DJ30.csv")
) -Label "C1"

$c2Csv = Resolve-ExistingPath -Candidates @(
    (Join-Path $RawRoot "C.Analyst\C2.AnalystForecast_DJ30.csv"),
    (Join-Path $RawRoot "C.Analyst\AnalystForecast_DJ30.csv"),
    (Join-Path $RawRoot "C2.AnalystForecast_DJ30.csv")
) -Label "C2"

$qcDir = Join-Path $resultsDir "qc_real"
$targetsDir = Join-Path $resultsDir "targets_real"
$panelDir = Join-Path $resultsDir "panel_real"
$featuresDir = Join-Path $resultsDir "features_real"
$qaDir = Join-Path $resultsDir "qa_benchmark_features_real"

New-Item -ItemType Directory -Force -Path $qcDir, $targetsDir, $panelDir, $featuresDir | Out-Null
if ($RunQaBenchmark) {
    New-Item -ItemType Directory -Force -Path $qaDir | Out-Null
}

Write-Host "[1/6] Building file manifest..."
& $PythonExe (Join-Path $scriptsDir "build_event_manifest.py") `
    --a1-dir $a1Dir `
    --a2-dir $a2Dir `
    --a3-dir $a3Dir `
    --a4-path $a4Path `
    --d-dir $dDir `
    --output-dir $qcDir

Write-Host "[2/6] Running initial QC..."
& $PythonExe (Join-Path $scriptsDir "run_initial_qc.py") `
    --a2-dir $a2Dir `
    --a4-path $a4Path `
    --output-dir $qcDir

Write-Host "[3/6] Building intraday targets..."
& $PythonExe (Join-Path $scriptsDir "build_intraday_targets.py") `
    --a2-dir $a2Dir `
    --a4-dir $a4Path `
    --d-dir $dDir `
    --a1-dir $a1Dir `
    --output-dir $targetsDir

Write-Host "[4/6] Building modeling panel..."
& $PythonExe (Join-Path $scriptsDir "build_modeling_panel.py") `
    --targets-csv (Join-Path $targetsDir "event_intraday_targets.csv") `
    --c1-csv $c1Csv `
    --c2-csv $c2Csv `
    --a1-dir $a1Dir `
    --a2-qc-csv (Join-Path $qcDir "a2_html_qc.csv") `
    --a4-event-qc-csv (Join-Path $qcDir "a4_event_qc.csv") `
    --output-dir $panelDir

Write-Host "[5/6] Building event text/audio features..."
& $PythonExe (Join-Path $scriptsDir "build_event_text_audio_features.py") `
    --panel-csv (Join-Path $panelDir "event_modeling_panel.csv") `
    --a1-dir $a1Dir `
    --a3-dir $a3Dir `
    --a4-row-qc-csv (Join-Path $qcDir "a4_row_qc.csv") `
    --output-dir $featuresDir

if ($RunQaBenchmark) {
    Write-Host "[6/6] Building Q&A benchmark features..."
    & $PythonExe (Join-Path $scriptsDir "build_qa_benchmark_features.py") `
        --panel-csv (Join-Path $panelDir "event_modeling_panel.csv") `
        --a1-dir $a1Dir `
        --output-dir $qaDir
}

Write-Host "Drive restore pipeline completed."
Write-Host "Panel: $(Join-Path $panelDir 'event_modeling_panel.csv')"
Write-Host "Features: $(Join-Path $featuresDir 'event_text_audio_features.csv')"
if ($RunQaBenchmark) {
    Write-Host "QA features: $(Join-Path $qaDir 'qa_benchmark_features.csv')"
}
