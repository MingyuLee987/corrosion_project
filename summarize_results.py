"""Recalculate R2 from saved publication-held-out predictions; no refitting."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'performance_summary'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / 'data/training_PDP_augmented.csv')
    predictions = pd.read_csv(ROOT / 'results/ML_evaluation/heldout_predictions.csv')
    original = pd.read_csv(ROOT / 'results/ML_evaluation/model_comparison.csv')
    re_papers = set(data.iloc[:50].reference_id)
    rows = []
    for keys, group in predictions.groupby(['dataset', 'feature_set', 'model']):
        assert group.record_id.is_unique
        assert np.isfinite(group[['observed_log10', 'predicted_log10']]).all().all()
        np.testing.assert_allclose(group.observed_log10, np.log10(group.icorr_A_cm2))
        for scope, mask in [
            ('all_rows', np.ones(len(group), dtype=bool)),
            ('RE_study_rows', group.reference_id.isin(re_papers)),
            ('RE_positive_only', group.RE_wt_pct.gt(0)),
        ]:
            g = group.loc[mask]
            score = r2_score(g.observed_log10, g.predicted_log10)
            saved = original.loc[
                original.dataset.eq(keys[0]) & original.feature_set.eq(keys[1])
                & original.model.eq(keys[2]) & original.scope.eq(scope)
            ]
            assert len(saved) == 1
            np.testing.assert_allclose(score, saved.log10_R2.iloc[0], atol=1e-12)
            rows.append(dict(dataset=keys[0], feature_set=keys[1], model=keys[2],
                scope=scope, rows=len(g), papers=g.reference_id.nunique(),
                log10_R2=score, raw_icorr_R2=r2_score(g.icorr_A_cm2, 10**g.predicted_log10)))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUT / 'r2_recalculated.csv', index=False, encoding='utf-8-sig')
    coverage = []
    for scope, g in [('all_314', data), ('previous_98', data[data.reference_id.ne('RebarLi2023')]),
                     ('RE_positive_33', data[data.RE_wt_pct.gt(0)])]:
        for col in data.columns:
            values = g[col].replace(['unknown', 'unreported', 'not_reported', ''], np.nan)
            coverage.append(dict(scope=scope, column=col, rows=len(g),
                available_rows=values.notna().sum(), missing_pct=100*values.isna().mean(),
                available_papers=g.loc[values.notna(), 'reference_id'].nunique(),
                distinct_values=values.nunique()))
    pd.DataFrame(coverage).to_csv(OUT / 'column_coverage.csv', index=False, encoding='utf-8-sig')
    selected = metrics[metrics.scope.eq('RE_positive_only')]
    lines = ['# R² 재계산 및 개선 컬럼 검토', '', '검토일: 2026-10-13', '',
        '저장된 논문 제외 예측값으로 R²를 재계산했다. 재학습이나 모델 변경은 하지 않았다. '
        '기존 ML 비교표의 72개 log10 R²와 모두 일치한다.', '',
        '## 평가 기준', '',
        '- 주 평가 대상: 실제 희토류 첨가 33행, 11개 논문. 98행/314행은 학습 자료 규모다.',
        '- R² = 1 − Σ(y−ŷ)² / Σ(y−평균(y))². 주 지표의 y는 log10(icorr_A_cm2)다.',
        '- 논문별 전체 시편을 함께 제외한 예측을 모아 계산한다. 폴드별 R² 평균이나 상관계수의 제곱이 아니다.',
        '- 음의 R²는 평가 대상의 실제 평균값을 모든 시편에 예측하는 경우보다 제곱오차가 크다는 뜻이다. '
        '이 평균 기준과 학습 자료로 계산하는 중앙값 기준 모델은 서로 다르다.',
        '- 원 단위 icorr R²도 r2_recalculated.csv에 별도로 저장했다. log10 R²와 혼용하지 않는다.', '',
        '## 모델별 희토류 첨가 시편 R²', '',
        '| 모델 | 98행 기본 8개 | 98행 조성·조건 | 314행 기본 8개 | 314행 조성·조건 |',
        '|---|---:|---:|---:|---:|']
    for model in ['median', 'ridge', 'svr', 'random_forest', 'extra_trees', 'hist_gradient_boosting']:
        scores = [selected.loc[selected.model.eq(model) & selected.dataset.eq(d)
                  & selected.feature_set.eq(f), 'log10_R2'].iloc[0]
                  for d in ['previous_98', 'current_314'] for f in ['core8', 'composition_and_conditions']]
        lines.append('| ' + model + ' | ' + ' | '.join(f'{s:.4f}' for s in scores) + ' |')
    lines += ['', '98행 기본 입력 SVR의 R²가 관측 비교 중 가장 높지만 여전히 음수다. '
        '314행 조성·조건 SVR도 중앙값 기준보다 R²와 논문별 MAE가 나쁘다. '
        '이 순위를 보고 선택한 모델의 성능은 새 독립 논문에서 확인해야 한다.', '',
        '## 지금까지의 데이터 확장 비교', '',
        '아래는 입력·모델을 내부 검증에서 선택하는 기존 분석이다. 위 고정 모델별 평가와 다른 절차이므로 수치를 직접 혼합하지 않는다.', '',
        '| 분석 시나리오 | 희토류 첨가 시편 log10 R² | log10 MAE |', '|---|---:|---:|']
    comparison = pd.read_csv(ROOT / 'results/RE_expanded/non_RE_comparison.csv')
    for row in comparison[comparison.scope.eq('RE_positive_only')].itertuples():
        lines.append(f'| {row.scenario} | {row.log10_R2:.4f} | {row.log10_MAE:.4f} |')
    lines += ['', '## 개선 컬럼 우선순위', '',
        '아래 우선순위는 현재 데이터의 결측·입력 구성과 문헌을 바탕으로 정한 수집 및 검증 후보다. '
        '컬럼별 제거/추가 실험을 수행한 중요도 순위가 아니며 R² 상승이 입증된 것은 아니다.', '',
        '| 우선순위 | 컬럼 | 권장 작업 |', '|---|---|---|',
        '| 1 | `Cl_mol_L`, `NaCl_wt_pct`, `pH`, `electrolyte_family` | 현재 확장 입력에서 빠진 Cl_mol_L을 후보로 추가. 용액별 단위·농도 기준을 맞추고 pH 원문을 보완. 혼합수/시멘트 염화물과 자유 용액 농도는 별도 유지. |',
        '| 1 | `temperature_C`, `pre_exposure_h`, `exposure_time_h`, `surface_state_cat` | PDP 온도와 사전 노출 온도 구분. 시간은 사전 노출/침지/PDP 안정화의 의미를 확인한 뒤 각각 사용. 시간 파생값 log1p도 내부 검증 후보. |',
        '| 1 | `RE_log1p_ppm`, `RE_element_cat`, `C_wt_pct`, `Si_wt_pct`, `Mn_wt_pct`, `P_wt_pct`, `S_wt_pct` | 기본 8개를 출발점으로 유지. P/S 상한값은 정확한 함량과 구별하고 원문 함량 보완. |',
        '| 2 | `Cr_wt_pct`, `Ni_wt_pct`, `Mo_wt_pct`, `Cu_wt_pct`, `material_class` | 정확한 조성을 확보한 뒤 합금 원소 블록을 추가해 비교. 강종 표준값이나 미보고 값을 0으로 대입하지 않음. |',
        '| 2 | `TP_scan_rate_mV_s`, `OCP_stabilization_s`, `TP_OCP_stabilization_time_s`, `gas_condition_cat`, `hydrodynamic_condition_cat` | 측정 프로토콜 보정 후보. 두 안정화 시간 컬럼은 의미·출처 일치 확인 후 통합하고 모순 표시. |',
        '| 3 | `processing_state_cat`, `final_heat_treatment_temp_C`, `final_heat_treatment_time_min`, `cooling_method_cat` | 보고된 열처리 온도·시간·냉각 조건을 수집. 추정 주조 방식은 입력 제외. |',
        '| 3 | `O_wt_pct`, `La_wt_pct`, `Ce_wt_pct`, `inclusion_number_density_mm2`, `inclusion_mean_area_um2` | 희토류 기구를 설명할 후보지만 매우 희소. 부식 전 측정값만 사용하고 여러 독립 논문에서 확보 후 평가. |',
        '| 조건부 | `test_specimen_type`, `carbonation_state`, `water_cement_ratio`, `relative_humidity_pct`, `chloride_by_cement_wt_pct` | 모르타르 자료를 계속 사용할 때 매질 차이를 구분. 희토류 자료에도 같은 조건의 관측이 있어야 효과를 확인 가능. |', '',
        '## 후보 컬럼의 실제 확보량', '',
        '숫자는 결측이 아닌 값의 수이며, 정보가 여러 논문에 걸쳐 있는지는 column_coverage.csv의 available_papers로 확인한다.', '',
        '| 컬럼 | 전체 314행 확보 | 희토류 첨가 33행 확보 | 희토류 첨가 결측률 |', '|---|---:|---:|---:|']
    cov = pd.DataFrame(coverage)
    for col in ['pH', 'Cl_mol_L', 'NaCl_wt_pct', 'temperature_C', 'Cr_wt_pct', 'Ni_wt_pct', 'Mo_wt_pct',
                'Cu_wt_pct', 'pre_exposure_h', 'exposure_time_h', 'TP_scan_rate_mV_s', 'OCP_stabilization_s',
                'final_heat_treatment_temp_C', 'O_wt_pct', 'La_wt_pct', 'Ce_wt_pct',
                'inclusion_number_density_mm2', 'inclusion_mean_area_um2']:
        a = cov[cov.scope.eq('all_314') & cov.column.eq(col)].iloc[0]
        b = cov[cov.scope.eq('RE_positive_33') & cov.column.eq(col)].iloc[0]
        lines.append(f'| `{col}` | {a.available_rows}/314 | {b.available_rows}/33 | {b.missing_pct:.1f}% |')
    lines += ['', 'pH는 희토류 첨가 33행에서 전부 결측이다. 현재 값만으로 희토류 영역의 pH 효과를 학습할 수 없다. '
        '개재물 수밀도·평균 면적은 2행만 있어 지금 추가하면 안정적인 중요도 평가가 어렵다. '
        '314행의 조성 결측은 새 철근 216행의 조성 미보고와 연결된다.', '',
        '## 추천 검증 순서', '',
        '1. 98행 기본 8개를 기준으로 환경 블록(염화물 농도·pH·시험 온도·사전 노출)을 추가한다. 우선 원문 결측 보완이 필요하다.',
        '2. 조성 블록(Cr/Ni/Mo/Cu), 측정 조건 블록, 열처리 블록을 하나씩 추가하고 같은 논문 제외 분할에서 비교한다.',
        '3. 컬럼·모델·보조 자료 가중치 선택은 내부 논문 제외 검증에서 수행한다. 외부 희토류 논문을 선택에 사용하지 않는다.',
        '4. 희토류 첨가 33행의 log10 R²와 논문별 동일 비중 MAE를 함께 보고, 새로운 독립 희토류 논문으로 최종 확인한다.', '',
        '`reference_id`, DOI, 시편 ID는 분할·추적 용도로만 사용한다. `icorr_A_cm2`, `log_icorr`와 '
        '부식 후 생성물·동일 PDP 곡선의 Ecorr/Tafel 기울기 등은 사전 부식 예측 입력에서 제외한다.', '',
        '## 문헌 근거', '',
        '- [S275의 매질·pH·열처리 비교](https://www.mdpi.com/2075-4701/12/8/1386): 조건별 부식 차이를 관측한 원 연구. '
        '환경 조건 수집의 근거이며 현재 모델의 컬럼 효과를 입증하지는 않는다.',
        '- [Cr/RE 철근의 염화물·개재물 연구](https://www.nature.com/articles/s41529-026-00746-3): '
        '염화물 수준, Cr/RE 및 개재물의 연관성을 조사한 원 연구. 조성·개재물 컬럼은 이 연구에 근거한 후보다.', '',
        '## 산출물과 재실행', '',
        '- `r2_recalculated.csv`: 24개 설정 × 3개 평가 범위의 log10/원 단위 R².',
        '- `column_coverage.csv`: 전체/98행/희토류 첨가 데이터의 컬럼별 결측률·확보 논문 수.',
        '- 재실행: `.venv/Scripts/python.exe -B summarize_results.py`.', '']
    (OUT / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Verified 72 saved log10 R2 values; wrote summary and coverage to', OUT)


if __name__ == '__main__':
    main()
