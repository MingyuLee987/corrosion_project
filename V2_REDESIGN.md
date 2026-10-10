# i_corr 예측 파이프라인 v2 — 2026-10-09

## 목표와 현재 도달점

기존 93컬럼 원본을 보존하고, `build_v2_31.py`가 **정확히 31컬럼인 `data/v2_curated_31.csv`**를 만든다. `corrosion_v2.py`는 이 31컬럼 표만 읽어 **자료 점검 → 논문 단위 검증 → 최종 학습 → 새 조건 예측**을 실행한다. 이 버전은 대조군 `i_corr`를 미리 측정할 수 있는 시험을 대상으로 한다. 11편, 33개 대조군·첨가 시편 쌍으로 중첩 논문 제외 검증했을 때 절대값 `log10(i_corr)` R²는 **0.653**, MAE는 **0.277**이다. 고정 SVR 단독 진단값은 R² **0.682**이지만, 같은 논문들을 보고 모델을 고른 이력이 있으므로 **0.653을 모델 선택까지 포함한 주 결과**로 제시한다. 첨가 효과 `log10(i_corr_variant/i_corr_control)` 자체의 R²는 **-0.233**이다. 0.8 이상은 달성하지 못했다.

### 31컬럼 구성

| 역할 | 컬럼 | 개수 |
|---|---|---:|
| 시편과 출처 | `record_id`, `reference_id`, `doi`, `sample_id`, `material_class` | 5 |
| 합금 조성 | `C`, `Si`, `Mn`, `Cr`, `Ni`, `Mo`, `Cu`, `RE`의 wt%, `RE_element_cat`, `composition_basis` | 10 |
| 제조·표면·환경·측정 | `processing_state_cat`, `surface_state_cat`, `electrolyte_family`, `NaCl_wt_pct`, `Cl_mol_L`, `H2SO4_mol_L`, `NaHSO3_mol_L`, `temperature_C`, `pre_exposure_h`, `gas_condition_cat`, `TP_scan_rate_mV_s` | 11 |
| 목표값과 원문 추적 | `raw_icorr_value`, `raw_icorr_unit`, `icorr_A_cm2`, `target_locator`, `quality_flags` | 5 |
| **합계** | | **31** |

이 31개가 **DB의 컬럼 수**다. 모델은 그중 예측 전에 알 수 있고 현재 자료량으로 다룰 수 있는 15개 조건 컬럼을 사용한다. `steel_family`는 `material_class`에서 실행 중 파생하며 DB 컬럼을 추가하지 않는다. DOI와 시편 ID는 논문 분리·원문 추적용이고 예측 특징으로 사용하지 않는다. `build_v2_31.py`는 원본 값을 그대로 복사하며 결측 채우기나 단위 재환산을 수행하지 않는다. 사용한 컬럼의 정확한 순서는 `data/v2_curated_31_manifest.json`에 기록한다.

## 단계별 설계

### 1. PDF → 근거가 붙은 원자료

`V2_EXTRACTION_PROMPT.md`는 LLM이 시편, 원 단위, 표·그림·페이지, 실측/명목 조성, 대조군 매칭을 함께 제출하도록 정의한다. 추출값은 사람이 원문을 확인한 다음에만 원본에 반영하고 31컬럼 표를 다시 만든다. `corrosion_v2.py audit`는 31컬럼 표의 결측을 계산하고 `results/v2/source_review_queue.csv`에 근거 재확인이 필요한 행을 우선순위로 정리한다. **LLM을 실제로 호출하거나 새 논문 수치를 자동 편입한 단계는 아직 수행하지 않았다.**

### 2. 고정된 예측 컬럼

| 종류 | v2 모델 입력 | 이유 |
|---|---|
| 대조군 | 같은 시험 조건의 `control_icorr_A_cm2` | 절대 부식 수준을 실측으로 보정; 미리 측정한 경우에만 사용 |
| 희토류 | `RE_wt_pct`, `RE_element_cat` | 원소와 첨가량을 분리; 수치는 `log1p(ppm)` 변환 |
| 강재 | `C_wt_pct`, `Cr_wt_pct`, `Ni_wt_pct`, `steel_family` | 현재 보고율과 재료 구분을 고려한 작은 입력 집합 |
| 전해질 | `electrolyte_family`, `NaCl_wt_pct`, `Cl_mol_L`, `H2SO4_mol_L`, `NaHSO3_mol_L` | 서로 다른 용액을 한 농도 컬럼으로 섞지 않음 |
| 노출·상태 | `temperature_C`, `pre_exposure_h`, `processing_state_cat`, `surface_state_cat` | 부식 조건이 달라지는 주요 원인 |

`pre_exposure_h`는 `log1p` 변환한다. 결측은 훈련 분할 안에서만 중앙값으로 대치하고 결측 지시자를 추가한다. 범주는 훈련 분할 안에서만 인코딩한다. `Ecorr_V`, `log_icorr`, 실험 후 조직·개재물 수치, 논문 ID·DOI는 예측 입력에서 제외한다. `pH`는 51행 전부 미보고라 현재 모델에 넣지 않았다. 원문에서 수집되면 먼저 자료 점검을 거쳐 스키마 변경을 사전 선언해야 한다.

### 3. 두 종류의 점수 구분

`delta = log10(i_corr_variant/i_corr_control)`을 학습하고 `predicted_i_corr = control_i_corr × 10^predicted_delta`로 계산한다. **절대값 R²와 첨가 효과 delta R²를 함께 보고**하여 대조군 측정값만으로 점수가 높아 보이는 착시를 막는다.

모델 후보는 `control_only`, 전체 훈련 delta의 중앙값, Ridge, SVR이다. 논문 한 편을 바깥 시험 집합으로 제외하고, 남은 논문들을 다시 하나씩 제외한 평균 논문별 MAE로 모델을 선택한다. 한 논문에 시편이 많아도 SVR/Ridge 학습에서 한 논문의 총 가중치가 같도록 한다. 최종 모델은 전체 자료의 논문별 내부 검증으로 선택한 SVR이며 `models/v2_paired.joblib`에 저장된다.

### 4. 적용 범위와 다음 자료 수집

새 예측 때 필수 입력인 무첨가 대조군 측정값이 없으면 오류로 멈춘다. 학습 때 보지 못한 범주나 수치 범위를 벗어난 입력은 `applicability_warnings`에 표시한다. 현재 논문별 오차의 90% 분위수는 기술통계로만 저장하며, 검증된 예측 구간이라고 주장하지 않는다.

R² 0.8에 접근하려면 먼저 한 강재 계열과 한 전해질·측정 프로토콜을 적용 범위로 **사전 정의**하고, 새 독립 논문을 추가해야 한다. 특히 현재 온도와 Ni 조성은 각각 15/51건만 보고된다. 이미 시험한 논문을 보고 나서 불리한 조건을 제외하거나 하이퍼파라미터를 바꾸면 0.8을 실제 새 논문 성능으로 해석할 수 없다. 추가 논문을 모을 때 검증용 논문을 먼저 고정한다.

표면 특징의 다음 후보는 분극 시험 전 실측 조도 `pretest_surface_roughness_Ra_um`이다. 현재 `surface_state_cat`은 논문마다 일정하고, 기존 51건에는 검증된 Ra가 없으므로 **현 모델 입력에는 추가하지 않았다**. `data/surface_roughness_review.csv`에 측정 시점·원문 위치를 포함한 수집 칸을 마련했다. 선정 근거와 채택 조건은 `MODEL_IMPROVEMENT.md`의 표면 컬럼 절에 기록했다.

## 실제 참고한 Q1 AI 저널 논문

1. **회귀 후보와 물리 조건별 특징 설계:** Al-Nouti & Yaseen, “Integration of hybrid machine learning model with differential evolution algorithm for underground Pipeline corrosion prediction,” *Engineering Applications of Artificial Intelligence* **158**, 111511 (2025), [DOI](https://doi.org/10.1016/j.engappai.2025.111511). [2025 Artificial Intelligence 분야 Q1 확인](https://www.akaturk.com/journals/24182?lang=en). 이 논문의 SVR을 후보로 사용했다. 배관 부식 깊이 데이터, 차분진화 최적화, 보고된 R²는 재현하거나 차용하지 않았다.
2. **사용 상황에 맞는 평가:** Riebesell 외, “A framework to evaluate machine learning crystal stability predictions,” *Nature Machine Intelligence* **7**, 836–847 (2025), [DOI](https://doi.org/10.1038/s42256-025-01055-1). [SCImago Q1 확인](https://www.scimagojr.com/journalrank.php?openaccess=true&page=2&total_size=10160&type=j). 본 프로젝트에서는 논문 전체를 제외한 검증과 예측 과제 구분에 참고했다. 논문의 소재 안정성 모델을 부식 모델로 재현한 것은 아니다.
3. **향후 문헌 추출의 사람 검토 단계:** “Predicting new research directions in materials science using large language models and concept graphs,” *Nature Machine Intelligence* **8** (2026), [DOI](https://doi.org/10.1038/s42256-026-01206-y). 추출 → 사람 교정 → 다시 학습하는 절차를 `V2_EXTRACTION_PROMPT.md`와 검토 목록 설계의 참고로 삼았다. 이 논문의 개념 그래프나 LLM을 현재 코드에서 실행하지는 않았다.

## 실행 방법

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1 -Mode V2Audit
powershell -ExecutionPolicy Bypass -File run.ps1 -Mode V2Train
```

새 조건은 `v2_prediction_input.csv`에 한 행씩 작성한 다음 아래 명령을 실행한다.

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1 -Mode V2Predict
```

결과: `results/v2/data_audit.json`, `source_review_queue.csv`, `benchmark.json`, `heldout_predictions.csv`, `new_predictions.csv`. `results/v2/example_input_training_record.csv`와 `example_prediction_training_record.csv`는 입출력 형식 확인용 **기존 학습 행**이며 외부 검증 결과가 아니다.
