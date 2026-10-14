# 일반강 보조 자료 확대와 희토류 예측 검증

수집·통합 자료를 335→412행으로 늘렸다. 기존 수집 일반강 66행을 새로 통합하고 이번에 42CrMo4 4행, WAAM2209 3행, AISI1010 4행을 새로 전사하여 총 77행을 더했다. 새로 찾은 77행의 희토류 논문이라는 의미는 아니다. 최신 통합 자료는 `data/training_RE_auxiliary_max.csv`이다.

추가 11행은 단위·시편 조건을 확인했다. 42CrMo4는 GDOES 조성 및 열처리·XRD 조직, WAAM2209는 원재료 와이어 조성 및 ferrite/austenite 조직, AISI1010은 C만 보고된 조성과 PDP25°C·사전노출 온도를 구분해 기록했다. WAAM의 본문/초록 solution 시간 불일치는 결측으로 두었다. 일반강의 RE=0은 의도적 첨가 미보고를 뜻하며 잔류 RE 실측 0을 뜻하지 않는다.

## 같은 희토류 47행의 중첩 논문 검증

| 실험 | 전체 자료 | log10 R² | log10 MAE | 2배 적중률 | 3배 적중률 | 5배 적중률 | 10배 적중률 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 335_reference | 335 | 0.0587 | 0.6641 | 9/47 (19.1%) | 15/47 (31.9%) | 26/47 (55.3%) | 39/47 (83.0%) |
| 335_matched_MAE | 335 | -0.0561 | 0.7070 | 9/47 (19.1%) | 13/47 (27.7%) | 25/47 (53.2%) | 37/47 (78.7%) |
| 401_all_auxiliary_MAE | 401 | -0.1732 | 0.7608 | 9/47 (19.1%) | 13/47 (27.7%) | 21/47 (44.7%) | 36/47 (76.6%) |
| 412_environment_focused | 412 | -0.3449 | 0.8209 | 5/47 (10.6%) | 12/47 (25.5%) | 21/47 (44.7%) | 34/47 (72.3%) |
| 412_SVR_Ridge_transfer | 412 | -0.2049 | 0.7419 | 10/47 (21.3%) | 15/47 (31.9%) | 27/47 (57.4%) | 36/47 (76.6%) |

335_matched_MAE와401_all_auxiliary_MAE는 동일 12개 후보·동일 내부 MAE 선택 기준으로 데이터 추가의 영향을 비교했다. 412_environment_focused는 13개 후보와 내부 R² 선택을 사용하는 별도 탐색으로, 차이를 자료 추가만의 효과로 해석하지 않는다. NaCl 보조 자료 우선 후보도 모든 희토류 연구의 대조군을 보존하고, 검증 논문은 대조군까지 통째로 제외한다.

전체 후보 중 권장 저장 모델은 `335_reference`이다. 확대 모델은 기준의 R²와 MAE를 동시에 개선하지 못하여 기존 기준 모델을 유지했다.

412_SVR_Ridge_transfer는 기존 SVR과 Ridge(alpha10/100), 전체 보조 자료와 NaCl 보조 자료 우선, 보조 가중치0.1/1을 12개 후보로 비교한 별도 탐색이다. 다른 후보 탐색과 동일 희토류 47행을 평가하였다.

자료 수를 늘리는 것만으로 정확도 상승을 보장할 수 없다. 독립 시편 수, 일반강/스테인리스강 차이, 시험환경 차이 및 결측 조성 때문에 관련 없는 자료의 추가가 오차를 늘릴 수 있다. RE 함량·조성·공정·기지 조직과 시험 환경을 함께 확인한 자료를 우선 수집하고 논문별 균형 가중치를 유지한다. 이번 결과는 반복 탐색 결과이며 독립된 신규 문헌의 검증을 대체하지 않는다.

## 사용

`python predict_phase_model.py input.csv output.csv --model results/RE_auxiliary_final/model.joblib`

재현: `python build_auxiliary_training.py`, `python add_ordinary_steel_batch.py`, `python tune_phase_model.py --latest --svr-only`, `python tune_phase_model.py --latest --improve --accuracy`, `python tune_phase_model.py --expanded --improve --accuracy`, `python tune_phase_model.py --improve --focused`, `python report_auxiliary_expansion.py`.
