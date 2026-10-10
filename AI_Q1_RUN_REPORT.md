# 부식 전류밀도 예측 모델 실행 기록 — 2026-10-09

## 실제 채택 모델과 Q1 AI 논문

이 프로젝트에서 현재 사용할 수 있는 모델은 **같은 시험 조건의 무첨가 대조군 `i_corr` 측정값이 있는 경우**의 SVR이다. 51개 기록 중 대조군과 엄격하게 짝지을 수 있는 33개 희토류 첨가 시편, 11편의 논문으로 검증했다. 각 회차에서 논문 한 편 전체를 빼고 모델과 하이퍼파라미터 후보를 나머지 논문만으로 선택했다. 선택된 SVR을 전체 33쌍에 다시 학습해 `models/paired_model.joblib`에 저장했다. 이 최종 학습 성능을 검증 성능으로 보고하지 않는다.

**방법 참고 논문:** Ahmad Fares Al-Nouti and Zaher Mundher Yaseen, “Integration of hybrid machine learning model with differential evolution algorithm for underground Pipeline corrosion prediction,” *Engineering Applications of Artificial Intelligence* **158**, 111511 (2025), [DOI:10.1016/j.engappai.2025.111511](https://doi.org/10.1016/j.engappai.2025.111511). 이 논문은 부식 예측 후보로 SVR과 XGBoost 등을 사용한다. 해당 저널은 [SCImago의 2025 AI 분야 Q1](https://www.akaturk.com/journals/24182?lang=en)으로 확인된다. 이 프로젝트는 **SVR을 부식 회귀 후보로 쓰는 방법**을 참고했다. 논문의 지중 배관 데이터, 차분진화 최적화, 특징 선택 결과를 그대로 재현하지 않았다. 논문에 보고된 `R²=0.8407`도 이 프로젝트의 `i_corr` 성능이 아니다.

**평가 설계 참고 논문:** Janosh Riebesell 외, “A framework to evaluate machine learning crystal stability predictions,” *Nature Machine Intelligence* **7**, 836–847 (2025), [DOI:10.1038/s42256-025-01055-1](https://doi.org/10.1038/s42256-025-01055-1). [SCImago Q1 저널](https://www.scimagojr.com/journalrank.php?openaccess=true&page=2&total_size=10160&type=j)이다. 이 논문이 강조하는 사용 상황에 맞춘 검증 원칙을 적용해, 새 논문 예측과 대조군 측정 후 예측을 별도 과제로 평가하고 논문 단위로 분리했다. 해당 논문이 부식 데이터나 이 프로젝트의 SVR 구조를 제시한 것은 아니다.

## 실행 결과

| 예측 과제와 방법 | 자료 | 논문 단위 검증의 log10(`i_corr`) R² | MAE, log10 단위 | 판단 |
|---|---:|---:|---:|---|
| 새 논문에 대한 절대값 예측, 기존 18개 후보 중첩 선택 | 51건, 11편 | -1.175 | 1.259 | 사용 권장 불가 |
| 위 후보에 CatBoost 3개 추가 | 51건, 11편 | -1.308 | 1.297 | 개선 없음 |
| TabPFN-2, 고정된 transfer 컬럼 | 51건, 11편 | -0.337 | 0.872 | 단독 진단만 개선, 여전히 낮음 |
| 기존 18개와 TabPFN-2 2개 컬럼 묶음을 내부 검증으로 선택 | 51건, 11편 | -1.297 | 1.294 | 채택하지 않음 |
| 대조군 측정값을 변경 없이 사용 | 33쌍, 11편 | 0.521 | 0.365 | 대조군 기준선 |
| 대조군 측정값 + 중첩 선택 SVR | 33쌍, 11편 | **0.683** | **0.266** | 현재 채택 모델 |
| 대조군 측정값 + TabPFN-2 | 33쌍, 11편 | 0.471 | 0.448 | 채택하지 않음 |

SVR의 예측식은 `log10(i_corr_variant) = log10(i_corr_control) + predicted_delta`이고, 학습 목표는 `delta = log10(i_corr_variant / i_corr_control)`이다. **첨가 효과인 `delta` 자체의 R²는 -0.126**이다. 따라서 절대값 R² 0.683의 상당 부분은 대조군 측정값에서 얻은 정보이며, 첨가 효과를 정확히 설명했다고 해석하면 안 된다. 0.266 log10 MAE는 대략 1.84배의 평균 배수 오차에 해당한다. `La24Q355` 한 논문의 평균 delta 오차가 1.237 log10으로 가장 크다. 원본 조건과 수치의 출처를 재확인할 필요가 있지만, 결과를 높이려고 검증 집합에서 임의로 제거하지 않았다.

TabPFN-2 모델 자체의 원 논문은 Noah Hollmann 외, “Accurate predictions on small data with a tabular foundation model,” *Nature* **637**, 319–326 (2025), [DOI:10.1038/s41586-024-08328-6](https://doi.org/10.1038/s41586-024-08328-6)이다. **Nature는 AI 전문 저널이 아니므로 이를 Q1 AI 저널 논문이라고 표시하지 않는다.** 공개 v2 가중치로 CPU에서 실행했고, 성능이 채택 SVR보다 낮았다.

## 컬럼 설계와 0.8에 접근하는 실험

한 행은 **한 논문에서 특정 강재, 제조 및 열처리, 표면 상태, 노출 이력, 전해질, 전기화학 측정 조건으로 얻은 한 시편의 `i_corr`**로 정의한다. 같은 시편의 목표값에서 계산한 `Rct`, Tafel 기울기, 시험 뒤 측정값은 사전 예측 입력으로 넣지 않는다.

1. **논문 및 근거:** `reference_id`, DOI, 표/그림 번호, 원래 단위, 변환식, 값의 불확실성. 모델 입력이 아닌 검증 분할과 감사를 위한 컬럼이다.
2. **조성:** `C, Si, Mn, Cr, Ni, Mo, Cu, S, P`의 실측 wt%, 강종, 희토류 `La/Ce` 등 원소별 실측 wt%, 총 RE, 첨가량의 명목값과 분석값 구분. 미보고는 0과 구분한다.
3. **제조 및 표면:** 주조/압연, 냉각, 최종 열처리 온도와 시간, 연마 정도, 조도, 코팅, 조직·개재물의 **시험 전 측정값**. 논문에서 미보고인 항목은 결측으로 보존한다.
4. **시험 환경:** 전해질 종류와 실제 농도, pH, 온도, 용존 산소, 사전 노출 시간과 매질. NaCl·NaHSO3·황산을 하나의 농도 컬럼으로 합치지 않는다.
5. **측정 방식:** 분극 속도, OCP 안정화 시간, 기준 전극, Tafel 외삽 범위, 반복 횟수. 논문 사이의 측정 편차를 진단하기 위해 수집한다.
6. **대조군 경로:** 같은 논문의 무첨가 대조군이 같은 제조·표면·환경·측정 조건에서 **미리 측정된 경우에만** `control_icorr_A_cm2`를 입력한다. 예측 대상 시편의 값이나 그로부터 파생된 컬럼을 넣지 않는다.

다음 실험은 적용 범위를 미리 정하고, 독립 논문을 더 수집해 같은 논문 단위 검증을 반복하는 것이다. 0.8은 **보장된 목표가 아니며**, 현재 자료의 검증 결과로는 달성했다고 말할 수 없다. 특히 현재 `temperature_C`, `Cr_wt_pct`, `Ni_wt_pct`의 보고 결측이 많아 모델 교체보다 원문에서 조건을 보완하는 작업이 우선이다. 자료를 늘릴 때에는 검증용 논문을 별도로 고정하고, 추가 자료 수집 중 그 논문의 결과를 보고 특징이나 제외 기준을 바꾸지 않는다.

## 재현

```powershell
.\.venv\Scripts\python.exe paired_benchmark.py
.\.venv\Scripts\python.exe tabpfn_paper_benchmark.py
.\.venv\Scripts\python.exe tabpfn_nested_extension.py
.\.venv\Scripts\python.exe tabpfn_paired_benchmark.py
```

채택 모델 결과는 `results/paired_benchmark.json`, 논문별 예측은 `results/paired_heldout_predictions.csv`에서 확인한다. TabPFN 진단 결과는 `results/tabpfn_*.json`에 있다. TabPFN 재현에는 `torch`, `tabpfn`과 공식 TabPFN-2 가중치 다운로드가 필요하다.
