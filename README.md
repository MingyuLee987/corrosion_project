# 강재 부식전류밀도 예측

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
