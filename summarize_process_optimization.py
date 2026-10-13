"""Report completed exploratory comparisons and unfinished strict data collection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/process_optimization'


def main():
    summary=pd.read_csv(OUT/'feature_ablation_comparison.csv')
    inner=pd.read_csv(OUT/'inner_selection_scores.csv')
    predictions=pd.read_csv(OUT/'feature_ablation_predictions.csv')
    meta=json.loads((OUT/'nested_validation.json').read_text(encoding='utf-8'))
    # Verify all comparisons independently against stored held-out predictions.
    for row in summary.itertuples():
        p=predictions[predictions.feature_set.eq(row.feature_set)&predictions.model.eq(row.model)
                      &predictions.auxiliary_weight.eq(row.auxiliary_weight)]
        assert len(p)==33 and p.record_id.is_unique
        np.testing.assert_allclose(r2_score(p.observed_log10,p.predicted_log10),row.log10_R2,atol=1e-12)
    # Sensitivity after observing Ridge extrapolation: clearly mark as post-hoc.
    selections=inner[inner.model.isin(['svr03','svr1','median'])].sort_values(
        ['log10_R2','paper_balanced_log10_MAE'],ascending=[False,True]).groupby('held_out_paper',sort=False).first().reset_index()
    records=[]
    for choice in selections.itertuples():
        p=predictions[predictions.reference_id.eq(choice.held_out_paper)&predictions.feature_set.eq(choice.feature_set)
            &predictions.model.eq(choice.model)&predictions.auxiliary_weight.eq(choice.auxiliary_weight)].copy()
        assert not p.empty
        records.append(p)
    sensitivity=pd.concat(records,ignore_index=True)
    assert len(sensitivity)==33 and sensitivity.record_id.is_unique
    sensitivity.to_csv(OUT/'posthoc_svr_only_nested_predictions.csv',index=False,encoding='utf-8-sig')
    sensitivity_r2=float(r2_score(sensitivity.observed_log10,sensitivity.predicted_log10))
    best=summary.iloc[0]
    plan=dict(status='exploratory_candidate_not_promoted; strict300_collection_incomplete',
        candidate_feature_set=best.feature_set,model=best.model,auxiliary_weight=float(best.auxiliary_weight),
        observed_candidate_R2=float(best.log10_R2),candidate_parameters=dict(C=.3,epsilon=.1),
        keep=['C_wt_pct','Si_wt_pct','Mn_wt_pct','RE_log1p_ppm','RE_element_cat','electrolyte_family'],
        add=['final_heat_treatment_temp_C','final_heat_treatment_time_min','processing_state_cat','melting_route_cat','cooling_method_cat'],
        remove_from_candidate_inputs=['P_wt_pct','S_wt_pct'],
        retain_in_raw_data=['P_wt_pct','S_wt_pct','P_upper_bound_wt_pct','S_upper_bound_wt_pct'],
        collect_next=['austenitizing_temperature_C','austenitizing_time_min','quench_medium_cat',
                      'partitioning_temperature_C','partitioning_time_min','austenitizing_heating_rate_C_s',
                      'quench_cooling_rate_C_s','rolling_reduction_reported_pct'],
        strict_new_ready_rows=3,verified_value_new_rows=66,target_rows=300,
        nested_all_candidates_R2=meta['nested_metrics']['log10_R2'],
        posthoc_SVR_only_nested_R2=sensitivity_r2,
        caveat='Best fixed-setting result is selected after inspecting82 settings; it is not an unbiased estimate of choosing that setting. SVR-only restriction is also post-hoc.')
    (OUT/'feature_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=summary[summary.model.eq('svr03')&summary.auxiliary_weight.eq(1.)].set_index('feature_set')
    lines=['# 공정 컬럼 비교와 수집 우선순위', '', '2026-10-13', '',
        '**약300행의 완전 확인 데이터 수집은 미완료다.** 새 문헌7편에서 조성표와 연결되는66행을 '
        '추출했지만 모든 수치 시험 조건·핵심 공정을 확인한 신규행은3행이다. '
        '[수집 상태](collection_status.md)에 미확인 항목과 포함 기준을 기록했다.', '',
        '## 먼저 넣고 빼볼 컬럼', '',
        '| 구분 | 컬럼 | 판단 |', '|---|---|---|',
        '| 유지 | C·Si·Mn, RE_log1p_ppm, RE_element_cat, electrolyte_family | 기본 출발점 |',
        '| 우선 추가 | 최종 열처리 온도·시간, 제조 상태, 용해 경로, 냉각 방식 | 이번 동일 설정 비교에서 가장 유망한 블록 |',
        '| 입력에서 잠정 제외 | P·S | 단독 제외 및 공정 추가와의 조합을 검증; 원자료에서는 삭제하지 않음 |',
        '| 다음 수집 | 오스테나이트화 온도·시간, 담금질 매질, 가열·냉각 속도, 압연 감소율, 단계 순서 | 제조공정 설명력을 높일 후보; 아직 효과 미검증 |',
        '| 보류 | 환경/추가 합금 원소를 한꺼번에 추가 | 현재 비교에서는 악화; 결측 보완 후 개별 블록 재검증 |',
        '| 입력 제외 | 논문/시편ID, Ecorr, Tafel기울기, 부식 후 생성물, 추정 주조조건 | 분할용ID/누수 및 미검증 값 |', '',
        '제조공정 조건을 추가한다는 것은 실험에서 어느 열처리가 가장 좋다는 인과적 결론이 아니다. '
        '주조·선행 열처리·최종 열처리·PDP 온도를 구분하고 각 단계의 시간 및 냉각을 수집해야 한다.', '',
        '## 동일한 SVR 설정에서 컬럼 효과', '',
        '314행 정제본을 사용한 논문 전체 제외 예측이며 평가 대상은 희토류 첨가33행/11편이다. '
        'SVR C=0.3, epsilon=0.1 및 보조논문 비중1을 고정했다. 학습에서만 결측 대체·스케일링·인코딩하며 '
        '학습 내 상수/전부 결측 컬럼을 제외한다. R²는 log10(icorr)에서 계산했다.', '',
        '| 입력 | R² | log10 MAE | 논문별 MAE |', '|---|---:|---:|---:|']
    for feature in ['core8','drop_P_S','add_manufacturing','drop_P_S_add_manufacturing','add_protocol','add_environment','add_alloy']:
        r=rows.loc[feature]
        lines.append(f'| {feature} | {r.log10_R2:.4f} | {r.log10_MAE:.4f} | {r.paper_balanced_log10_MAE:.4f} |')
    lines += ['', 'P·S를 빼고 공정5개를 더한 설정은 R² 0.1634, log10 MAE 0.6001이었다. '
        '단순 입력8개 대비 R²는 약0.2273 증가했다. 이 값은82개 설정을 관측한 뒤 발견한 후보 결과다.', '',
        '## 모델·컬럼 선택까지 검증한 결과', '',
        f'- Ridge/SVR/중앙값·입력 블록·보조비중까지 내부에서 선택한 원 중첩검증 R²: **{meta["nested_metrics"]["log10_R2"]:.4f}**.',
        '- GCr23 제외 시 Ridge가 log10(icorr)=4.55까지 외삽해 성능이 크게 무너졌다. 해당 결과를 제거하거나 숨기지 않았다.',
        f'- 이 외삽을 관측한 후 SVR/중앙값으로 제한한 추가 민감도 비교 R²: **{sensitivity_r2:.4f}**. 이 제한 자체가 사후 결정이므로 개선 성능의 독립 증거가 아니다.',
        '- 따라서0.1634를 확정된 최종 예측력으로 주장하거나 모델을 승격하지 않는다. 새 독립 희토류 문헌으로 확인해야 한다.', '',
        '## 원자료 보완', '',
        '기존314행 정제본에서224행의 조건을 대조했고 검증 기록2714건 중2498건은 기존 빈칸의 보완이다. '
        '저자의 철근 연구 학위논문으로216행의 스캔 속도, 사전 부동태화 및 표면 준비 조건을 확인했다. '
        '용액72행의 pH는13.5/8.0으로 확인했으며 모르타르에 같은 수치를 부여하지 않았다. '
        '정제본의 스캔 속도 확보는295행, pH 확보는90행이다. 철근216행은 조성 및 수치 PDP 온도 미확인으로 엄격300행 목표에서 제외한다.', '',
        '## 새 문헌 수집 출처', '',
        '- [AISI304 민감화](https://www.mdpi.com/1996-1944/15/23/8543): 신규 엄격 확인3행.',
        '- [S32101](https://www.mdpi.com/1996-1944/18/9/1921): 조성·전해질·온도와 연결한32행, 제조 이력은 부분 보고.',
        '- [8620 가열속도](https://www.mdpi.com/2624-5558/5/3/16): 공정 및 평균값5행, PDP 수치 온도 미보고.',
        '- [미세합금강 열처리](https://www.electrochemsci.org/papers/vol7/7086677.pdf): 열처리10행, PDP 수치 온도 미보고.',
        '- [51CrV4](https://doi.org/10.47978/TUS.2023.73.04.002): 중복 조건을 제외한6행, PDP 온도·스캔 속도 미보고.',
        '- [SAE9254 Q&P](https://www.mdpi.com/2075-4701/15/5/509):4행, 실온만 보고.',
        '- [X2CrNi12 냉간압연·소둔](https://www.mdpi.com/1996-1944/15/19/6914):6행, PDP 수치 온도 미보고.', '',
        '## 재실행', '',
        '```powershell', '.venv/Scripts/python.exe -X utf8 -B curate_process_data.py',
        '.venv/Scripts/python.exe -X utf8 -B build_verified_literature.py',
        '.venv/Scripts/python.exe -X utf8 -B optimize_process_features.py',
        '.venv/Scripts/python.exe -X utf8 -B summarize_process_optimization.py', '```', '',
        '새66행은 자동 합치지 않았으며 이번 R²에는 사용되지 않았다. 엄격 확인 자료의 규모가 확보되면 '
        '그 집합을 고정하고 같은 검증 방식으로 다시 비교한다. 새 문헌 수집은 계속 필요한 상태다.', '']
    (OUT/'process_feature_report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Verified82 settings; exploratory best R2',best.log10_R2,'; strict300 collection incomplete.')


if __name__=='__main__':
    main()
