# 강재 부식전류밀도 예측

<!-- composition CV -->
19입력 GBR의 시험 28행을 제외하고 116행에서 4분할 교차검증했다. **행 단위 검증 R²: 전체 0.8217·희토류 0.7782**, **논문 단위 검증 R²: 전체 −0.2428·희토류 −0.4125**이다. 검증 예측을 모아 계산한 R²이며, 입력 선택에 사용한 개발용 검증으로 독립 중첩 검증은 아니다. [검증 방법·시험과의 차이·적중률](results/composition_inputs/cross_validation/report.md).
<!-- /composition CV -->

<!-- composition inputs -->
조성 입력을 확장한 비교에서 **Si·Mn·P·S·Cu를 추가한 19입력 GBR**를 학습 내 논문별 CV로 선택했다. 같은 시험 행에서 전체 R²는 **0.8406 → 0.8573**, 희토류 R²는 **0.8052 → 0.9597**이고 희토류 2배 이내 적중은 **6/6**이다. 단, 희토류 시험은 6행이며 학습 내 논문별 CV R²는 **−0.4125**로 새 문헌 일반화는 여전히 부족하다. [조성 컬럼·결측률·비교 결과](results/composition_inputs/report.md).
<!-- /composition inputs -->


<!-- RE new literature -->
NaCl 자료에 원 논문 수치가 확인된 **5행(희토류 첨가 3행·대조군 2행)**을 추가하여 **144행·196컬럼**으로 재학습했다. 약 20행 목표에는 아직 도달하지 못했다. 기존 시험 28행을 고정한 전체 시험 R²는 **0.8392 → 0.8406**, 희토류 시험 6행의 R²는 **0.8173 → 0.8052**였다. 희토류 5배 이내 적중은 5/6 → 6/6이지만 2배 이내 적중은 5/6 → 4/6이므로 전반적 개선으로 해석하지 않는다. [추가 자료](data/training_RE_NaCl_REexpanded.csv), [출처·보류 문헌·학습과 시험 결과](results/RE_NaCl_new_literature/report.md)를 확인할 수 있다.
<!-- /RE new literature -->


<!-- train-test experiment -->
NaCl139행에서 학습내4분할 검증으로54후보를 비교한 뒤 고정80:20 시험을 평가했다. 선택된 GBR의 **전체 학습/시험 R²=0.9458/0.8392**, **희토류 학습/시험 R²=0.9797/0.8173**이다. 희토류 시험은6행뿐이며 같은 논문의 시료가 학습과 시험에 함께 존재할 수 있다. 별도 논문전체 제외 시험에서는 희토류4행 R²=−9.4356이므로 새논문 예측 문제는 해결되지 않았다. [학습·시험 상세 결과와 사용법](results/train_test_optimization/report.md)에 함께 기록했다. 기본모델은 유지한다.
<!-- /train-test experiment -->


<!-- NaCl-only experiment -->
NaCl 단독 환경으로 제한한 자료128행에 일반강 원문2편11행을 추가해 **139행·기존196컬럼**을 만들었다. 희토류30행·12편의 논문 단위 중첩 검증 R²는 **0.1101 → 0.1322**다. [NaCl 현재 실험 자료](data/training_RE_NaCl_expanded.csv), [같은30행 비교·적중률·재현방법](results/NaCl_only/report.md), [문헌18편 검토 기록](results/NaCl_only/literature_screening.csv)을 참고한다. 새희토류5행은 [확인 대기](data/NaCl_RE_pending.csv)로 분리했으며 학습에는 쓰지 않았다. 기본 예측모델은 유지했다.
<!-- /NaCl-only experiment -->


품질 점검에서 원문 내 수치 환산이 불일치하는 일반강2행을 보류했다. [현재 품질 보류 후458행](data/training_RE_clean.csv)과 [보류 행](data/targets_quarantined.csv)을 분리했다. 희토류 검증47행은 유지했으며 중첩R²는−0.0660→−0.0651로 거의 같았다. 추가 튜닝은 배수 적중률을 올렸지만 R²가−0.2993으로 악화해 기본 모델을 교체하지 않았다. [품질 판단·전체 비교](results/quality_cleanup/report.md)를 참고한다.

컬럼의 단위·시험 단계·기지상 의미를 점검하고 근사 염화물 환산값을 제외한 새 설계를 검증했다. 같은 희토류47행에서 중첩R²는 **−0.1678→−0.0660**, 5배 적중률은 **51.1%→61.7%**였다. 입력·후보 설정이 함께 변경된 탐색이며, 여전히 음수R²이므로 기본335행 모델은 유지한다. [컬럼 의미와 사용 기준](results/semantic_audit/column_meanings.md), [수정·검증 결과·다음 우선순위](results/semantic_corrected/report.md)를 참고한다.

최신 추가 실험에서 일반강 48행을 더해 **460행**으로 확대했다. 같은 희토류 47행·같은 4개 SVR 후보의 중첩 검증 R²는 **−0.1307→−0.1678**로 낮아졌다. 학습 RE R²=0.8498은 예측 정확도를 뜻하지 않는다. [최신 460행 데이터](data/training_RE_auxiliary_460.csv)와 [전후 검증·원문·재현 방법](results/added48/report.md)을 참고한다. 기본 예측 모델은 유지했으며, 아래는 이전 412행 분석이다.

문헌의 원소 조성·공정·미세조직으로 `log10(icorr_A_cm2)`를 예측한다. 최신 통합 데이터는 **412행**이며, 희토류 첨가 시편 **47행**을 논문 단위로 검증한다. 검증 논문의 대조군도 학습에서 함께 제외한다.

자료 확대 후 일부 배수 적중률은 상승했지만 검증 R²와 큰 오차가 악화하여, 기본 모델은 335행 기준 SVR로 유지했다. 학습 적합도는 새 시편의 예측 정확도를 뜻하지 않는다.

| 지표 | 기준 SVR | 412행 SVR·Ridge 비교 |
|---|---:|---:|
| 검증 R² | 0.0587 | −0.2049 |
| 2배 이내 | 19.1% | 21.3% |
| 3배 이내 | 31.9% | 31.9% |
| 5배 이내 | 55.3% | 57.4% |
| 10배 이내 | 83.0% | 76.6% |

[상세 비교·한계](results/RE_auxiliary_final/report.md)에 전체 실험을 기록했다. 독립적인 신규 문헌 검증이 필요하다.

## 남긴 데이터

| 파일 | 역할 |
|---|---|
| [training_RE_auxiliary_max.csv](data/training_RE_auxiliary_max.csv) | 최신 412행 통합 자료 |
| [training_RE_collection_latest.csv](data/training_RE_collection_latest.csv) | 채택 모델 재현용 335행 기준 |
| [training_RE_with_auxiliary.csv](data/training_RE_with_auxiliary.csv) | 401행 비교 실험 |
| [process_literature_verified.csv](data/process_literature_verified.csv) | 출처 검증된 보조 66행 |
| [ordinary_steel_additions_20261014.csv](data/ordinary_steel_additions_20261014.csv) | 추가 전사한 일반강 11행 |
| [RE_graph_readings_quarantined.csv](data/RE_graph_readings_quarantined.csv) | 단위 미확정 그래프 판독값 |
| [RE_weld_targets_pending_composition.csv](data/RE_weld_targets_pending_composition.csv) | 최종 RE 조성 확인 대기 |

일반강의 RE=0은 의도적 첨가 미보고를 뜻하며 잔류 함량의 실측 0은 아니다. 조성 범위는 중간값과 원문 범위를 보존한다. 결측·미확정 단위는 임의로 채우지 않는다. 412행은 독립적인 412개 조성이 아니다. 문헌 원본과 검증 근거는 `sources/`에 보존한다.

## 실행

기본 모델 예측:

```powershell
.venv/Scripts/python.exe -X utf8 -B predict_phase_model.py input.csv output.csv
```

`실행.bat` 또는 `run.ps1`은 기준 SVR과 확대 자료 모델을 비교한다. 전체 실험 재현:

```powershell
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --latest --svr-only
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --latest --improve --accuracy
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --expanded --improve --accuracy
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --improve --focused
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --improve --transfer
.venv/Scripts/python.exe -X utf8 -B report_auxiliary_expansion.py
```

`build_auxiliary_training.py`와 `add_ordinary_steel_batch.py`는 데이터 통합, `optimize_process_features.py`는 입력 변환·평가 지표, `composition_values.py`는 조성 범위 변환을 담당한다.

구버전 데이터·일회성 스크립트·이전 실험 결과는 정리했다. [삭제 기록](results/cleanup/project_consolidation_20261014.json)에 경로·크기·해시를 남겼다.
