#Requires -Version 7.0
<#
.SYNOPSIS
    Evaluate the CREAMpy NetworkPlatformModel against analytical tests
    and M-Pesa Kenya calibration data.

.DESCRIPTION
    Three-phase evaluation:

    Phase 1 — Analytical validation (always runs)
        Eight closed-form degenerate-case tests:
        pre-launch zeroes, critical mass gate, never-crossed gate,
        monotonicity, ceiling enforcement, sigma=0 decoupling,
        ptrs proportional scaling, N_p0 seed effect.

    Phase 2 — M-Pesa Kenya calibration (always runs)
        Grid search over p_p, q_p, sigma, S_crit, p_f, q_f against
        the bundled synthetic M-Pesa Kenya dataset (2007-2020).
        Reports best-fit parameters and combined RMSE.

    Phase 3 — Optional GSMA real data calibration
        If -GsmaFile is supplied, repeats calibration against the real
        GSMA Global Mobile Money Excel workbook (freely available at
        gsma.com/mobilefordevelopment/resources/global-mobile-money-dataset).
        Extracts Kenya rows automatically.

    Output:
        eval_network_platform_report_<timestamp>.md

.PARAMETER GsmaFile
    Path to the GSMA Global Mobile Money Excel file (optional).
    If supplied, Phase 3 runs after Phase 2.

.PARAMETER RepoRoot
    Root of the CREAMpy repository. Defaults to the parent of this script.

.PARAMETER OutputDir
    Directory for output files. Default: <RepoRoot>\eval\output

.EXAMPLE
    # Analytical + synthetic calibration only:
    .\eval\eval_network_platform.ps1

    # Full evaluation with real GSMA data:
    .\eval\eval_network_platform.ps1 -GsmaFile "C:\Downloads\GSMA_MobileMoney.xlsx"
#>
param(
    [string]$GsmaFile  = "",
    [string]$RepoRoot  = "",
    [string]$OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'

# ── Resolve paths ──────────────────────────────────────────────────────────────
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $RepoRoot)  { $RepoRoot  = Split-Path -Parent $ScriptDir }
if (-not $OutputDir) { $OutputDir = Join-Path $RepoRoot "eval\output" }

$Ts         = Get-Date -Format 'yyyyMMdd-HHmmss'
$ReportPath = Join-Path $OutputDir "eval_network_platform_report_$Ts.md"
$PythonEval = Join-Path $ScriptDir "eval_network_platform.py"
$SyntheticData = Join-Path $ScriptDir "data\mpesa_kenya_synthetic.csv"

$null = New-Item -ItemType Directory -Path $OutputDir -Force

# ── Helper functions ──────────────────────────────────────────────────────────
function Write-Phase { param([int]$N,[string]$M)
    Write-Host "`n══ Phase $N — $M ══" -ForegroundColor Cyan }
function Write-OK    { param([string]$M) Write-Host "  OK   $M" -ForegroundColor Green }
function Write-Fail  { param([string]$M) Write-Host "  FAIL $M" -ForegroundColor Red }
function Write-Info  { param([string]$M) Write-Host "       $M" }

# ── Pre-flight checks ─────────────────────────────────────────────────────────
Write-Host "`nCREAMpy NetworkPlatformModel evaluation harness" -ForegroundColor White
Write-Info "Repo root:  $RepoRoot"
Write-Info "Output dir: $OutputDir"

# Python availability
try   { $pyVersion = (python --version 2>&1) -replace 'Python ', ''
        Write-OK "Python $pyVersion found" }
catch { Write-Fail "Python not found on PATH — install Python 3.9+ and retry"; exit 1 }

# Minimum version check (3.9)
$pyMajor, $pyMinor = $pyVersion.Split('.')[0,1] | ForEach-Object { [int]$_ }
if ($pyMajor -lt 3 -or ($pyMajor -eq 3 -and $pyMinor -lt 9)) {
    Write-Fail "Python 3.9+ required, found $pyVersion"; exit 1
}

# Eval script
if (-not (Test-Path $PythonEval)) {
    Write-Fail "Eval script not found: $PythonEval"; exit 1
}
Write-OK "Eval script found"

# Synthetic data
if (-not (Test-Path $SyntheticData)) {
    Write-Fail "Synthetic data not found: $SyntheticData"; exit 1
}
Write-OK "Synthetic M-Pesa data found"

# ── Phase 1: Analytical validation ───────────────────────────────────────────
Write-Phase 1 "Analytical validation (8 degenerate-case tests)"

$p1Out = python $PythonEval 2>&1
$p1Out | ForEach-Object { Write-Info $_ }

# Initialise before conditional so Set-StrictMode never sees undefined vars
$p1Passed = 0
$p1Failed = 1   # default to failure; overwritten on successful parse

$p1PassLine = $p1Out | Select-String -Pattern '(\d+) passed / (\d+) failed' |
              Select-Object -Last 1
if ($p1PassLine -and $p1PassLine.Line -match '(\d+) passed / (\d+) failed') {
    $p1Passed = [int]$Matches[1]
    $p1Failed = [int]$Matches[2]
    if ($p1Failed -eq 0) { Write-OK "All $p1Passed analytical tests passed" }
    else                  { Write-Fail "$p1Failed analytical tests FAILED" }
} else {
    Write-Fail "Could not parse test results from output"
    $p1Failed = 1
}

# ── Phase 2: Synthetic M-Pesa calibration ─────────────────────────────────────
Write-Phase 2 "M-Pesa Kenya calibration (synthetic data, 2007-2020)"

$p2ReportPath = Join-Path $OutputDir "calibration_synthetic_$Ts.md"
$p2Out = python $PythonEval --calibrate --data $SyntheticData --report $p2ReportPath 2>&1
$p2Out | ForEach-Object { Write-Info $_ }

if (Test-Path $p2ReportPath) { Write-OK "Calibration report: $p2ReportPath" }
else                          { Write-Fail "Calibration report not written" }

$rmse_line = $p2Out | Select-String -Pattern 'Combined RMSE' | Select-Object -Last 1
if ($rmse_line) {
    if ($rmse_line.Line -match '(\d+\.\d+)') {
        $rmse_val = [double]$Matches[1]
        if ($rmse_val -lt 0.10) { Write-OK "RMSE = $rmse_val (good portfolio-level fit)" }
        elseif ($rmse_val -lt 0.20) { Write-Info "RMSE = $rmse_val (acceptable)" }
        else                         { Write-Fail "RMSE = $rmse_val (poor fit — review parameters)" }
    }
}

# ── Phase 3: Optional GSMA real data calibration ──────────────────────────────
$p3Skipped = $true
if ($GsmaFile) {
    Write-Phase 3 "GSMA real data calibration"
    if (-not (Test-Path $GsmaFile)) {
        Write-Fail "GSMA file not found: $GsmaFile"
    } else {
        Write-OK "GSMA file: $GsmaFile"

        # Extract Kenya rows from GSMA Excel using Python + openpyxl
        $extractScript = Join-Path $OutputDir "extract_gsma.py"
        $kenyaCsv      = Join-Path $OutputDir "gsma_kenya_extracted.csv"
        [System.IO.File]::WriteAllText($extractScript, @"
import sys, csv
try:
    import openpyxl
except ImportError:
    print("ERROR: openpyxl not installed. Run: pip install openpyxl")
    sys.exit(1)

gsma_path  = sys.argv[1]
output_csv = sys.argv[2]

wb = openpyxl.load_workbook(gsma_path, data_only=True, read_only=True)
print(f"Sheets: {wb.sheetnames}")

# GSMA workbook typically has a sheet per metric or a combined sheet.
# Strategy: scan all sheets for rows where first column contains 'Kenya'.
kenya_rows = {}  # year -> {agents, accounts}

for sheet_name in wb.sheetnames:
    ws = wb[sheet_name]
    headers = None
    for row in ws.iter_rows(values_only=True):
        if headers is None:
            if row and any(str(v).lower() in ('year','date','metric') for v in row if v):
                headers = [str(v).strip() if v else '' for v in row]
            continue
        if row and any(str(v).lower() == 'kenya' for v in row if v):
            year_val = None
            agents_val = None
            accounts_val = None
            for i, v in enumerate(row):
                h = headers[i].lower() if i < len(headers) else ''
                if 'year' in h or 'date' in h:
                    try: year_val = int(str(v)[:4])
                    except: pass
                if 'agent' in h:
                    try: agents_val = float(v)
                    except: pass
                if 'account' in h or 'customer' in h:
                    try: accounts_val = float(v)
                    except: pass
            if year_val and agents_val:
                if year_val not in kenya_rows:
                    kenya_rows[year_val] = {}
                if agents_val:
                    kenya_rows[year_val]['agents'] = agents_val
                if accounts_val:
                    kenya_rows[year_val]['accounts'] = accounts_val

if not kenya_rows:
    print("WARNING: No Kenya rows found. Check GSMA sheet structure.")
    sys.exit(1)

with open(output_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['year', 'agent_outlets', 'registered_accounts_M', 'active_accounts_M', 'notes'])
    for yr in sorted(kenya_rows):
        agents   = kenya_rows[yr].get('agents', 0)
        accounts = kenya_rows[yr].get('accounts', 0)
        w.writerow([yr, agents, accounts / 1e6, accounts / 1e6 * 0.6, 'GSMA extracted'])

print(f"Extracted {len(kenya_rows)} Kenya rows to {output_csv}")
"@, [System.Text.Encoding]::UTF8)

        $extractOut = python $extractScript $GsmaFile $kenyaCsv 2>&1
        $extractOut | ForEach-Object { Write-Info $_ }

        if (Test-Path $kenyaCsv) {
            Write-OK "Kenya data extracted: $kenyaCsv"
            $p3ReportPath = Join-Path $OutputDir "calibration_gsma_$Ts.md"
            $p3Out = python $PythonEval --calibrate --data $kenyaCsv --report $p3ReportPath 2>&1
            $p3Out | ForEach-Object { Write-Info $_ }
            if (Test-Path $p3ReportPath) { Write-OK "GSMA calibration report: $p3ReportPath" }
            $p3Skipped = $false
        } else {
            Write-Fail "Kenya extraction failed — check GSMA file structure"
        }
    }
}

if ($p3Skipped) {
    Write-Phase 3 "GSMA real data (skipped)"
    Write-Info "Pass -GsmaFile to enable real-data calibration."
    Write-Info "Download from: gsma.com/mobilefordevelopment/resources/global-mobile-money-dataset"
}

# ── Summary ───────────────────────────────────────────────────────────────────
Write-Host "`n══ Summary ══" -ForegroundColor Cyan
Write-Info "Output dir: $OutputDir"
if ($p1Failed -eq 0) { Write-OK  "Phase 1: PASS ($p1Passed analytical tests)" }
else                  { Write-Fail "Phase 1: $p1Failed FAILURES" }
Write-OK  "Phase 2: calibration complete — see $p2ReportPath"
if ($p3Skipped) { Write-Info "Phase 3: skipped" }
else            { Write-OK   "Phase 3: GSMA calibration complete" }

exit ($p1Failed -gt 0 ? 1 : 0)
