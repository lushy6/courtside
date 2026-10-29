# Restart the phase 7 lanes after a reboot. Run at logon by the "Courtside phase 7 lanes" scheduled task:
#
#   powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File scripts\phase7_restart.ps1
#
# A lane that is already running, or whose eval/run.py outlived its lane, is left alone, so running this
# twice never starts a second lane on the same provider (they would share one daily usage counter).
# Existing logs are renamed with a timestamp first, because Start-Process truncates its redirect targets.
# Every decision is appended to eval/results/_phase7_logs/restart.log.

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$logDir = 'eval\results\_phase7_logs'
New-Item -ItemType Directory -Force $logDir | Out-Null

function Write-RestartLog([string]$msg) {
    $line = '[{0}] {1}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss zzz'), $msg
    Add-Content -Path "$logDir\restart.log" -Value $line -Encoding utf8
}

$procs = @(Get-CimInstance Win32_Process -Filter "Name like 'python%'" | Where-Object { $_.CommandLine })
$stamp = Get-Date -Format 'yyyy-MM-dd-HHmm'

foreach ($lane in 'gemini', 'groq') {
    $laneProc = $procs | Where-Object { $_.CommandLine -match "phase7\.py\s+run\s+--lane\s+$lane\b" }
    $runProc = $procs | Where-Object { $_.CommandLine -match "eval\\run\.py.*--provider\s+$lane\b" }
    if ($laneProc) {
        Write-RestartLog "$lane`: lane already running (pid $(@($laneProc)[0].ProcessId)), skipped"
        continue
    }
    if ($runProc) {
        Write-RestartLog "$lane`: eval/run.py still running without its lane (pid $(@($runProc)[0].ProcessId)), skipped; restart by hand once it exits"
        continue
    }
    foreach ($kind in 'out', 'err') {
        $log = "$logDir\$lane.$kind.log"
        if (Test-Path $log) { Rename-Item $log "$lane.$kind.$stamp.log" }
    }
    $p = Start-Process .\.venv\Scripts\python.exe -ArgumentList 'scripts\phase7.py', 'run', '--lane', $lane `
        -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$logDir\$lane.out.log" -RedirectStandardError "$logDir\$lane.err.log"
    Write-RestartLog "$lane`: started lane (pid $($p.Id))"
}
