# 품질 보류와 추가 튜닝 결과

원본460행에서 원문 Table6의 전류밀도·부식속도 환산이 서로 맞지 않는 일반강2행을 보류했다. 오류가 큰 예측값을 보고 제외한 것이 아니다. 두 행의 전류값 자체가 틀렸다고 확정할 수 없어 원본과 보류 파일을 유지한다.

활성 자료는 `data/training_RE_clean.csv`의458행이다. 두 보류 행은 `data/targets_quarantined.csv`, 판단 근거와 원문 위치는 `decisions.json`에 저장했다. 중복 record_id 및 비양수 목표는 없었다. 희토류 검증47행은 한 행도 제외하지 않았다.

| 실험 | 중첩 log10 R² | log10 MAE | 2배 | 3배 | 5배 | 10배 |
|---|---:|---:|---:|---:|---:|---:|
| 전체460행·같은4개 후보 | −0.0660 | 0.7161 | 14.9% | 31.9% | 61.7% | 74.5% |
| 품질 보류 후458행·같은4개 후보 | **−0.0651** | 0.7174 | 14.9% | 31.9% | 59.6% | 74.5% |
| 458행·규제/커널 범위 확장8개 후보 | −0.2993 | 0.7201 | 23.4% | 42.6% | 61.7% | 80.9% |

두 행 보류의 R² 변화는 약0.0009로 작으며, MAE와5배 적중률은 개선되지 않았다. 이 두 행이 낮은 성능의 주원인이라는 근거는 없다. 규제 강도를 바꾼 모델은 배수 적중률이 상승했으나 일부 큰 오차가 증가해 R²가 악화했다. R²·적중률·평균 로그 오차는 서로 다른 지표이므로 좋은 숫자만 선택하지 않는다.

추가 튜닝에서 최고 고정 설정(C=3, gamma=scale, 보조가중치0.1)의 논문 제외 탐색 R²는0.1033이었다. 같은 검증 자료로 설정을 선택한 수치이므로 중첩 검증R²=−0.2993과 구분한다. 학습R²=0.9829도 새 논문 정확도가 아니다. 기본335행 모델은 교체하지 않았다.

점수가 낮은 Ma19FSS/UCS23 같은 희토류 논문을 사후 제거하면 평가대상이 달라져 정확도가 부풀 수 있다. 적법한 개선은 원문 단위 재확인, 같은 시험범위의 독립 자료 확보, 사전 정의한 환경별 모델과 전체 평가의 병행이다. 검증 오차를 근거로 일반강 보조행을 선택할 때도 선택 과정은 내부 학습·검증에서만 이루어져야 한다.

```powershell
.venv/Scripts/python.exe -X utf8 -B clean_training_quality.py
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --semantic --conservative --input data/training_RE_clean.csv --output results/quality_cleanup/clean458
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --semantic --refine --input data/training_RE_clean.csv --output results/quality_cleanup/refined458
```
