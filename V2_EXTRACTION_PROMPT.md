# PDF에서 새 실험 행을 만드는 추출 지침

논문을 읽는 LLM에 아래 지침을 전달하고, **한 시편·한 시험 조건·한 `i_corr` 측정마다 한 JSON 객체**를 요청한다. 추출된 값은 곧바로 학습 DB에 넣지 않는다. 사람이 근거를 검토한 후 원본 `data/training_51.csv`에 반영하고, `build_v2_31.py`로 31컬럼 학습 표를 다시 만든다.

```text
대상은 강재의 희토류 첨가와 전기화학 부식전류밀도 i_corr이다.
본문, 표, 그림 설명에서 명시된 값만 추출한다. 추정하거나 빈칸을 0으로 채우지 않는다.
각 JSON 객체에 paper DOI, 시편명, i_corr 원문 수치와 단위,
표·그림 번호 및 페이지, 전해질과 농도, 온도, 노출 시간,
강재 조성의 실측/명목 구분, 희토류 첨가량의 실측/명목 구분,
제조·열처리·표면 상태, 분극 측정 방법, 같은 조건의 무첨가
대조군 시편명을 기록한다. 모든 값에는 원문 근거 위치를 붙인다.
표면 조도 Ra가 **분극 시험 전에** 측정되었다면 μm 단위 수치,
측정 시점(연마 직후/사전 노출 후/분극 직전), 측정 방법을 기록한다.
Sa, Rq, 연마지 grit 번호를 Ra 값으로 바꾸어 쓰지 않는다.
값을 찾지 못하면 null, 원문 사이에 모순이 있으면 conflict로 표시한다.
`i_corr`와 부동태 전류, 부식속도, `E_corr`를 혼동하지 않는다.
μA/cm², mA/cm², A/m² 등을 A/cm²로 환산할 때는 원단위와
환산식을 함께 기록한다. 표 값의 지수 표기와 머리글 배율을 확인한다.
대조군 매칭은 전해질, 처리, 표면, 사전 노출이 같을 때만 제안하고,
다르면 match=false와 그 이유를 출력한다.
```

최소 출력 필드: `doi`, `sample_id`, `target.raw_value`, `target.raw_unit`, `target.A_cm2`, `target_locator`, `control_sample_id`, `control_match`, `material`, `RE_element`, `RE_wt_pct`, `RE_basis`, `electrolyte`, `exposure`, `polarization_method`, `pretest_surface_roughness_Ra_um`, `roughness_measured_when`, `roughness_method`, `evidence`, `conflicts`. 각 `evidence` 항목은 필드명, 페이지/표/그림, 짧은 원문 근거를 포함한다. Ra가 없으면 null로 둔다.

검토자는 `results/v2/source_review_queue.csv`에서 결측과 품질 표시가 많은 행부터 확인한다. 검토 완료 전에 대상 논문의 `i_corr`를 학습 집합에 합치지 않는다. 특히 측정값을 보고 나서 유리한 논문만 채택하는 일을 피하기 위해 포함 기준을 먼저 기록한다.
