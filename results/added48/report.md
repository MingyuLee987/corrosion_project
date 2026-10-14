# 신규 48행 추가 후 SVR 검증

기존 412행에 일반강 조건별 부식전류밀도 48행을 추가하여 460행을 만들었다. 출처는 [Abd et al., International Journal of Corrosion (2024), DOI 10.1155/ijco/5531986](https://onlinelibrary.wiley.com/doi/full/10.1155/ijco/5531986)이다. 원문 Tables1,2,5,6과 Sec2.8을 확인하였다.

48행은 같은 기본 조성에서 6개 처리 상태 × 4개 시험 온도 × 2개 환경을 조합한 보고 결과다. 독립적인 48개 조성이나 논문이 아니다. 반복시험 횟수는 행으로 늘리지 않았다. Cr/Mo/Ni의 원문 대시는 결측으로 유지했고, CuO/Al2O3는 담금질 매체로 기록했으며 강의 Cu/Al 조성으로 간주하지 않았다.

Table6의 CuO55°C와 CuO_T55°C는 보고된 icorr와 부식속도의 환산 관계가 일치하지 않는다. 원문 icorr를 그대로 저장하고 quality_flags에 문제를 표시하였다. 부식속도에서 전류를 역산하여 교체하지 않았다. pH는 CO2 주입 전후 용액 값이며 각 온도별 측정값으로 단정하지 않는다.

## 같은 희토류 47행의 검증

동일한 4개 RBF-SVR 후보(C=0.1/0.3, 보조 가중치0.25/1), 동일 내부 모델 선택 기준, 중첩 논문 제외 검증을 적용했다. 검증 논문 전체와 그 대조군을 학습에서 제외하고 전처리는 학습 분할에서만 수행했다. 추가 48행은 보조 자료이며 RE 검증 행수는 변하지 않았다.

| 지표 | 추가 전 412행 | 추가 후 460행 |
|---|---:|---:|
| 중첩 검증 log10 R² | −0.1307 | **−0.1678** |
| 학습 RE log10 R² | 0.6993 | 0.8498 |
| 중첩 log10 MAE | 0.7247 | 0.7431 |
| 2배 이내 | 10/47 (21.3%) | 9/47 (19.1%) |
| 3배 이내 | 15/47 (31.9%) | 14/47 (29.8%) |
| 5배 이내 | 27/47 (57.4%) | 24/47 (51.1%) |
| 10배 이내 | 36/47 (76.6%) | 34/47 (72.3%) |

이번 자료 추가는 희토류 일반화 성능을 개선하지 못했다. 학습 R² 상승은 검증 성능 상승이 아니다. 기존 README의 412행 R²=−0.2049는 SVR·Ridge 12개 후보 탐색 결과로, 이번 동일 4개 SVR 후보의 추가 전 점수와 구분한다. 기본 예측 모델은 기존 335행 기준 모델로 유지하며, 신규 460행 모델은 실험 결과로 보존한다.

최신 데이터: `data/training_RE_auxiliary_460.csv`. 추가 행: `data/auxiliary_additions_48.csv`. 원문 표 전사·단위·위치: `sources/literature/NanofluidSteel2024_verified_tables.json`. 검증 결과와 개별 예측은 이 폴더의 `baseline412/`, `expanded460/`에 저장했다.

```powershell
.venv/Scripts/python.exe -X utf8 -B add_auxiliary_48.py
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --svr-only --input data/training_RE_auxiliary_max.csv --output results/added48/baseline412
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --svr-only --input data/training_RE_auxiliary_460.csv --output results/added48/expanded460
```
