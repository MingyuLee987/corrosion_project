# NaCl 단독 환경 실험 (2026-10-14)

원본458행을 보존하고 NaCl 단독 용액128행(희토류30행·12편)을 선별했다. 혼합산/염, CO2포화, 모르타르/공극용액을 제외했다. NaCl로 분류되지만 아세트산/NaOH로 pH조절한9행도 제외했다. 제외330행은 excluded_rows.csv에 이유와 함께 보존했다. 기체 N2 탈기 또는 공기 조건은 포함한다. 기존 메타데이터 기반 선별이므로 모든 기존 원문을 재감사한 결과는 아니다.

컬럼명·순서196개를 유지해 원문 확인된 일반강2편11행을 추가했다. 최종139행이다. 추가분은 희토류 첨가 시료가 아니며 희토류 검증30행은 그대로다. nA/cm2는1e-9, uA/cm2는1e-6을 곱했다. 표 평균만 전사하고 반복3회를 가상3행으로 늘리지 않았다. 실온의 수치 온도나 미보고 원소를 추정하지 않았다. 원문 PDF와 표 시각검증 이미지를 sources에 보존했다.

## 동일 NaCl30행 비교

| 실험 | 검증 R² | log10 MAE | 2배 이내 | 3배 이내 | 5배 이내 | 10배 이내 |
|---|---:|---:|---:|---:|---:|---:|
| 전체458행 학습·동일NaCl30행 평가 | -0.0465 | 0.7104 | 4/30 (13.3%) | 11/30 (36.7%) | 18/30 (60.0%) | 22/30 (73.3%) |
| NaCl128행 학습 | 0.1101 | 0.5833 | 11/30 (36.7%) | 16/30 (53.3%) | 21/30 (70.0%) | 25/30 (83.3%) |
| NaCl139행 학습·11행 추가 | 0.1322 | 0.5712 | 12/30 (40.0%) | 15/30 (50.0%) | 21/30 (70.0%) | 26/30 (86.7%) |

SVR semantic_matrix, C=.1/.3, 보조문헌 가중치=.1/1의 동일4후보를 사용했다. 바깥 검증은 해당 논문 전체(대조군 포함)를 제외하고, 안쪽 논문분할에서 설정을 선택한다. 모든 데이터 변환은 학습 내에서 적합한다. 목표는 log10(icorr_A_cm2). 전체458행 모델의 동일30행 값은 기존 nested 예측에서 해당 행만 추출한 값이다. 기존 설정선택에는 다른 환경의 희토류도 포함되어 있어 완전히 동일한 모델선택 모집단의 비교는 아니다. 전체47행 R²=-0.0651과30행 점수는 직접 전후 개선율로 비교하지 않는다. 새11행 전후128/139 비교는 같은30행·같은 후보·같은 검증 규칙이다.

## 추가 문헌 및 대기

18편의 이번 검토 결과·중복·제외·대기 사유는 literature_screening.csv에 기록했다. 새로운 RE 관련5행(2편)은 기존196컬럼으로 data/NaCl_RE_pending.csv에 전사했다. Q355B2026은 초록 전류값은 있으나 전체 조성/최종RE 분석이 없어 보류했다. RE3042026은 조성 범위 중간값(.005/.010 wt%)과 원문 범위를 보존했으나 icorr가 확인되지 않아 보류했다. 부동태전류, pitting potential, EIS 저항을 icorr로 대체하지 않는다. 본문 접근 제한과 목표값 미확인이 남아 있어 이5행은 학습/검증에 넣지 않았다.

원문 추가:
- [LPBF304L2024](https://doi.org/10.1016/j.jmrt.2024.03.025): 6행, Table1/2/4; Sec2.1/2.4.
- [9Cr2024](https://doi.org/10.3390/ma17204960): 5행, Sec2.1/2.4; Table1.

새 희토류 독립 검증은 수행하지 않았다. 기본335행 예측모델은 교체하지 않았다. 새139행 모델은 results/NaCl_only/expanded/model_all139.joblib이며 전용 실험용이다.

```powershell
.venv/Scripts/python.exe -X utf8 -B build_nacl_dataset.py
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --semantic --conservative --input data/training_RE_NaCl.csv --output results/NaCl_only/baseline
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --semantic --conservative --input data/training_RE_NaCl_expanded.csv --output results/NaCl_only/expanded
.venv/Scripts/python.exe -X utf8 -B report_nacl_results.py
```
