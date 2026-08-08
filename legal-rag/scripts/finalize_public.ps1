param(
    [Parameter(Mandatory = $true)]
    [int]$TargetProcessId
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$questionsPath = Join-Path $projectRoot "data\questions\public-official.json"
$answersPath = Join-Path $projectRoot "data\outputs\full-public\internal-results.json"
$submissionPath = Join-Path $projectRoot "data\outputs\full-public\submission.json"

Wait-Process -Id $TargetProcessId
Set-Location $projectRoot

& $pythonPath -m app.cli.main submit `
    --questions $questionsPath `
    --answers $answersPath `
    --output $submissionPath
if ($LASTEXITCODE -ne 0) {
    throw "Submission formatting failed with exit code $LASTEXITCODE."
}

& $pythonPath -m app.cli.main validate-submission $submissionPath `
    --questions $questionsPath
if ($LASTEXITCODE -ne 0) {
    throw "Submission validation failed with exit code $LASTEXITCODE."
}
