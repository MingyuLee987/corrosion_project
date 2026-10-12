$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:OMP_NUM_THREADS = '2'
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $taskPython)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Python environment creation failed' }
}
& $taskPython -c "import numpy, pandas, sklearn"
if ($LASTEXITCODE -ne 0) {
    & $taskPython -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
}
& $taskPython -B corrosion_re_expand.py
if ($LASTEXITCODE -ne 0) { throw 'Analysis failed' }
& $taskPython -B evaluate_ml.py
if ($LASTEXITCODE -ne 0) { throw 'ML evaluation failed' }
Write-Host 'Completed: results/RE_expanded'
Write-Host 'ML evaluation: results/ML_evaluation'
