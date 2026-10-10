# corrosion_project

희토류 첨가 강재의 부식전류밀도 `i_corr` 예측 실험입니다.

- [처음 실행 안내](README_먼저읽기.txt)
- [2026-10-09 모델 개선 결과와 한계](MODEL_IMPROVEMENT.md)
- [Q1 AI 논문 근거와 실제 모델 실행 기록](AI_Q1_RUN_REPORT.md)
- [새 파이프라인 v2 설계와 실행 결과](V2_REDESIGN.md)
- [v2 학습용 31컬럼 표](data/v2_curated_31.csv)
- [분극 전 표면 조도 Ra 원문 검토표](data/surface_roughness_review.csv)
- 완전 신규 조건 예측: `prediction_input.csv` 작성 후 `새조건_예측.bat`
- 같은 시험의 무첨가 대조군을 알고 있을 때: `paired_prediction_input.csv` 작성 후 `대조군_새조건예측.bat`
- 대조군 모델 논문 단위 검증·재학습: `대조군_모델검증.bat`
