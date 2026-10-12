# corrosion_project

희토류 첨가 강재의 부식전류밀도 `i_corr` 예측 실험입니다.

현재 주 분석은 [50행 확장 데이터](data/training_RE_expanded.csv)와 [확장 검증 보고서](RE_EXPANSION_2026-10-12.md)입니다. `.venv/Scripts/python.exe -B corrosion_re_expand.py`로 재현합니다.

49편 엑셀 기반의 [전체 컬럼 수집 결과](results/workbook_mining/collection_report.md)와 [데이터마이닝 수집 엑셀](outputs/workbook_mining/literature_mining_collected.xlsx)은 별도 문헌 수집표입니다. 151행 중 전체 34개 측정·계산 항목이 완비된 행은 0행이므로 현재 학습 데이터에 자동 합치지 않았습니다.

data 폴더의 학습 CSV는 `training_51.csv`(원본 47행)와 `training_RE_expanded.csv`(확장 50행) 두 개입니다. 기준 RE 데이터와 신규 3행은 실행 중 구성하므로 중복 CSV를 저장하지 않습니다. 보조 자료는 `review/`(검토·후보), `sources/`(이전 수집·스냅샷), `time/`(시간 검증 별도 실험), `v2/`(대조군 기반 실험), `templates/`(입력 예시·범주값)에 있습니다.

- [처음 실행 안내](README_먼저읽기.txt)
- [2026-10-09 모델 개선 결과와 한계](MODEL_IMPROVEMENT.md)
- [2026-10-10 부식 논문 기반 확장 실험과 실행 방법](CORROSION_RESEARCH_2026-10-10.md)
- [Q1 AI 논문 근거와 실제 모델 실행 기록](AI_Q1_RUN_REPORT.md)
- [새 파이프라인 v2 설계와 실행 결과](V2_REDESIGN.md)
- [v2 학습용 31컬럼 표](data/v2/v2_curated_31.csv)
- [분극 전 표면 조도 Ra 원문 검토표](data/review/surface_roughness_review.csv)
- 완전 신규 조건 예측: `prediction_input.csv` 작성 후 `새조건_예측.bat`
- 같은 시험의 무첨가 대조군을 알고 있을 때: `paired_prediction_input.csv` 작성 후 `대조군_새조건예측.bat`
- 대조군 모델 논문 단위 검증·재학습: `대조군_모델검증.bat`
