param([ValidateSet('Train','Predict','Check','PairTrain','PairPredict','V2Audit','V2Train','V2Predict')][string]$Mode='Train')
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
try {
 $env:PYTHONUTF8='1'
 $env:PYTHONIOENCODING='utf-8'
 $taskPython=Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
 if (!(Test-Path -LiteralPath $taskPython)) {
  $taskBase=$null
  $taskBundled=Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
  if (Test-Path -LiteralPath $taskBundled) {$taskBase=$taskBundled}
  foreach ($taskName in @('python','python3')) {
   if ($taskBase) {break}
   $taskCmd=Get-Command $taskName -ErrorAction SilentlyContinue
   if ($taskCmd) {
    $taskProbe=& $taskCmd.Source -c "import sys; print((3,12) <= sys.version_info < (3,15))" 2>$null
    if ($LASTEXITCODE -eq 0 -and $taskProbe -eq 'True') {$taskBase=$taskCmd.Source;break}
   }
  }
  if (!$taskBase) {
   $taskBundled=Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
   if (Test-Path -LiteralPath $taskBundled) {$taskBase=$taskBundled}
  }
  if (!$taskBase) {throw 'Python 3.12~3.14을 설치한 후 다시 실행하세요. 설치 시 Add Python to PATH를 선택하세요.'}
  Write-Host '폴더 전용 Python 환경 준비 중...'
  & $taskBase -m venv (Join-Path $PSScriptRoot '.venv')
  if ($LASTEXITCODE -ne 0) {throw 'Python 환경 생성 실패'}
 }
 $taskReady=& $taskPython -c "import importlib.util as u,importlib.metadata as m; print(all(u.find_spec(x) is not None for x in ['numpy','scipy','sklearn','joblib']) and m.version('scikit-learn')=='1.9.1')"
 if ($LASTEXITCODE -ne 0 -or $taskReady -ne 'True') {
  Write-Host '첫 실행: 필요한 라이브러리 설치 중 (인터넷 필요)...'
  & $taskPython -m pip install -r (Join-Path $PSScriptRoot 'requirements.txt')
  if ($LASTEXITCODE -ne 0) {throw '라이브러리 설치 실패. 인터넷 연결과 오류 메시지를 확인하세요.'}
 }
 if ($Mode -eq 'Train') {
  & $taskPython prepare.py
  if ($LASTEXITCODE -ne 0) {throw '학습 데이터 점검 실패'}
  & $taskPython train.py --data results/current_input.json --out results/retrained
  if ($LASTEXITCODE -eq 0) {Copy-Item -LiteralPath 'results/retrained/exploratory_model.joblib' -Destination 'models/exploratory_model.joblib' -Force}
 } elseif ($Mode -eq 'Predict') {& $taskPython predict.py}
 elseif ($Mode -eq 'PairTrain') {& $taskPython paired_benchmark.py}
 elseif ($Mode -eq 'PairPredict') {& $taskPython predict_paired.py}
 elseif ($Mode -eq 'V2Audit') {
  & $taskPython build_v2_31.py
  if ($LASTEXITCODE -ne 0) {throw '31컬럼 데이터 구성 실패'}
  & $taskPython corrosion_v2.py audit
 }
 elseif ($Mode -eq 'V2Train') {
  & $taskPython build_v2_31.py
  if ($LASTEXITCODE -ne 0) {throw '31컬럼 데이터 구성 실패'}
  & $taskPython corrosion_v2.py benchmark
 }
 elseif ($Mode -eq 'V2Predict') {& $taskPython corrosion_v2.py predict}
 else {& $taskPython check.py}
 if ($LASTEXITCODE -ne 0) {throw '실행 실패. 위 오류 메시지를 확인하세요.'}
 Write-Host '완료. results 폴더에서 결과를 확인하세요.'
} catch {Write-Host $_ -ForegroundColor Red;exit 1}


