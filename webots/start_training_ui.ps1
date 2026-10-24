$ErrorActionPreference = 'Stop'
$dashboardRoot = $PSScriptRoot
$dashboardUrl = 'http://127.0.0.1:8765'
try {
    $null = Invoke-RestMethod "$dashboardUrl/api/status" -TimeoutSec 2
} catch {
    $trainingPython = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python313\python.exe'
    if (-not (Test-Path -LiteralPath $trainingPython)) { throw 'Training Python 3.13 was not found. Run training_ui.py with your SB3 Python environment.' }
    $dashboardScript = Join-Path $dashboardRoot 'training_ui.py'
    $dashboardLogs = Join-Path $dashboardRoot 'runs'
    Start-Process -FilePath $trainingPython -ArgumentList @('-u', ('"' + $dashboardScript + '"')) -WorkingDirectory $dashboardRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $dashboardLogs 'training_ui.log') -RedirectStandardError (Join-Path $dashboardLogs 'training_ui_errors.log')
    $dashboardReady = $false
    for ($dashboardAttempt = 0; $dashboardAttempt -lt 30; $dashboardAttempt++) {
        Start-Sleep -Milliseconds 200
        try { $null = Invoke-RestMethod "$dashboardUrl/api/status" -TimeoutSec 1; $dashboardReady = $true; break } catch { }
    }
    if (-not $dashboardReady) { throw 'Dashboard did not start. See runs/training_ui_errors.log.' }
}
Start-Process $dashboardUrl
