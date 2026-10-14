"""Compare the same held-out RE rows and document the NaCl-only experiment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/NaCl_only'

def main():
    d=pd.read_csv(ROOT/'data/training_RE_NaCl.csv')
    ids=set(d.loc[d.RE_wt_pct.gt(0),'record_id'])
    experiments=[('전체458행 학습·동일NaCl30행 평가','results/quality_cleanup/clean458'),
        ('NaCl128행 학습','results/NaCl_only/baseline'),
        ('NaCl139행 학습·11행 추가','results/NaCl_only/expanded')]
    rows=[]
    for label,folder in experiments:
        p=pd.read_csv(ROOT/folder/'predictions.csv')
        p=p.loc[p.record_id.isin(ids)]
        assert set(p.record_id)==ids and p.nested_prediction.notna().all()
        err=np.abs(p.observed_log10-p.nested_prediction)
        r=dict(experiment=label,RE_test_rows=len(p),RE_test_papers=p.reference_id.nunique(),
            log10_R2=r2_score(p.observed_log10,p.nested_prediction),
            log10_MAE=mean_absolute_error(p.observed_log10,p.nested_prediction))
        for k in [2,3,5,10]:
            r[f'within_{k}x_count']=int((err<=np.log10(k)).sum())
            r[f'within_{k}x_pct']=float((err<=np.log10(k)).mean()*100)
        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT/'matched_comparison.csv',index=False,encoding='utf-8-sig')
    ledger=[
        ('10.1016/j.jmrt.2024.03.025','added','6','304L LPBF; composition Table1; nA/cm2 means Table4; N2 purged', 'https://nano.xjtu.edu.cn/202412.pdf'),
        ('10.3390/ma17204960','added','5','9Cr tempered; chemistry Sec2.1; uA/cm2 Table1','https://doi.org/10.3390/ma17204960'),
        ('10.3390/ma18010069','duplicate','0','Ce316L already Zhang2025; no duplicate append','https://www.mdpi.com/1996-1944/18/1/69'),
        ('10.3390/coatings14040413','duplicate_further_graphs_pending','0','EH36 already Coat24; temperature/concentration graphs require validated icorr digitization','https://www.mdpi.com/2079-6412/14/4/413'),
        ('10.1007/s11663-026-04267-x','pending','2','Q355B NaCl abstract currents verified; full chemistry and final RE assay not verified','https://link.springer.com/article/10.1007/s11663-026-04267-x'),
        ('10.1016/j.jmrt.2026.02.234','pending','3','RE304 chemistry verified; numeric icorr not verified; passive current not substituted','https://www.sciencedirect.com/science/article/pii/S2238785426004849'),
        ('10.1080/00084433.2026.2656834','pending_full_text','0','Ce microalloy steel NaCl; abstract compositions Ce0/.015/.045; target table unavailable','https://www.tandfonline.com/doi/full/10.1080/00084433.2026.2656834'),
        ('10.1016/j.jre.2020.06.015','pending_full_text','0','Ce HSLA NaCl; no validated chemistry+icorr table retrieved','https://www.sciencedirect.com/science/article/abs/pii/S1002072120303045'),
        ('10.1515/rams-2022-0284','pending_methods_tables','0','Hyperduplex RE; icorr mentioned; exact electrolyte and full table not retrieved','https://www.degruyterbrill.com/document/doi/10.1515/rams-2022-0284/html'),
        ('10.1016/j.jmst.2021.03.014','pending_target','0','13Cr4Ni RE chemistry retrieved; numeric icorr not verified','https://www.jmst.org/EN/10.1016/j.jmst.2021.03.014'),
        ('10.16490/j.cnki.issn.1001-3660.2020.08.025','pending_full_chemistry','0','A572 RE=.0047; abstract icorr2.77e-6; full chemistry unavailable','https://www.surface-techj.com/CN/10.16490/j.cnki.issn.1001-3660.2020.08.025'),
        ('10.3390/ma18051048','not_compatible_target','0','Ce bridge study examined: EIS/weight loss; no numeric icorr table','https://www.mdpi.com/1996-1944/18/5/1048'),
        ('10.3390/ma17051022','pending_full_chemistry','0','Uncoated press-hardened2currents verified; full chemistry not provided; AlSicoated excluded','https://www.mdpi.com/1996-1944/17/5/1022'),
        ('10.1038/s41529-025-00680-w','separate_method_pending','0','API microalloy NaCl LPR-derived icorr; method needs harmonization before PDP integration','https://www.nature.com/articles/s41529-025-00680-w'),
        ('10.3390/met10091174','excluded_environment','0','Wet-dryRE NaHSO3; not NaCl','https://www.mdpi.com/2075-4701/10/9/1174'),
        ('10.1007/s11663-025-03792-5','excluded_environment','0','NaCl+Na2S mixed electrolyte','https://research.birmingham.ac.uk/en/publications/effect-of-rare-earth-metal-alloying-on-pitting-corrosion-resistan/'),
        ('10.1016/j.mtcomm.2025.111799','pending_method_target','0','RE N80 NaCl Raman corrosion; no confirmed PDPicorr label','https://www.sciencedirect.com/science/article/abs/pii/S2352492825003113'),
        ('10.3390/met15020195','excluded_inhibitor','0','Rare earth is dissolved inhibitor, not alloy RE','https://www.mdpi.com/2075-4701/15/2/195')]
    pd.DataFrame(ledger,columns=['doi','decision','new_rows_or_pending','reason','source_url']).to_csv(OUT/'literature_screening.csv',index=False,encoding='utf-8-sig')
    headers='| 실험 | 검증 R² | log10 MAE | 2배 이내 | 3배 이내 | 5배 이내 | 10배 이내 |\n|---|---:|---:|---:|---:|---:|---:|'
    table='\n'.join(f"| {r['experiment']} | {r['log10_R2']:.4f} | {r['log10_MAE']:.4f} | "+' | '.join(f"{r[f'within_{k}x_count']}/30 ({r[f'within_{k}x_pct']:.1f}%)" for k in [2,3,5,10])+' |' for r in rows)
    report=f'''# NaCl 단독 환경 실험 (2026-10-14)

원본458행을 보존하고 NaCl 단독 용액128행(희토류30행·12편)을 선별했다. 혼합산/염, CO2포화, 모르타르/공극용액을 제외했다. NaCl로 분류되지만 아세트산/NaOH로 pH조절한9행도 제외했다. 제외330행은 excluded_rows.csv에 이유와 함께 보존했다. 기체 N2 탈기 또는 공기 조건은 포함한다. 기존 메타데이터 기반 선별이므로 모든 기존 원문을 재감사한 결과는 아니다.

컬럼명·순서196개를 유지해 원문 확인된 일반강2편11행을 추가했다. 최종139행이다. 추가분은 희토류 첨가 시료가 아니며 희토류 검증30행은 그대로다. nA/cm2는1e-9, uA/cm2는1e-6을 곱했다. 표 평균만 전사하고 반복3회를 가상3행으로 늘리지 않았다. 실온의 수치 온도나 미보고 원소를 추정하지 않았다. 원문 PDF와 표 시각검증 이미지를 sources에 보존했다.

## 동일 NaCl30행 비교

{headers}
{table}

SVR semantic_matrix, C=.1/.3, 보조문헌 가중치=.1/1의 동일4후보를 사용했다. 바깥 검증은 해당 논문 전체(대조군 포함)를 제외하고, 안쪽 논문분할에서 설정을 선택한다. 모든 데이터 변환은 학습 내에서 적합한다. 목표는 log10(icorr_A_cm2). 전체458행 모델의 동일30행 값은 기존 nested 예측에서 해당 행만 추출한 값이다. 기존 설정선택에는 다른 환경의 희토류도 포함되어 있어 완전히 동일한 모델선택 모집단의 비교는 아니다. 전체47행 R²=-0.0651과30행 점수는 직접 전후 개선율로 비교하지 않는다. 새11행 전후128/139 비교는 같은30행·같은 후보·같은 검증 규칙이다.

## 추가 문헌 및 대기

18편의 이번 검토 결과·중복·제외·대기 사유는 literature_screening.csv에 기록했다. 새로운 RE 관련5행(2편)은 기존196컬럼으로 data/NaCl_RE_pending.csv에 전사했다. Q355B2026은 초록 전류값은 있으나 전체 조성/최종RE 분석이 없어 보류했다. RE3042026은 조성 범위 중간값(.005/.010 wt%)과 원문 범위를 보존했으나 icorr가 확인되지 않아 보류했다. 부동태전류, pitting potential, EIS 저항을 icorr로 대체하지 않는다. 본문 접근 제한과 목표값 미확인이 남아 있어 이5행은 학습/검증에 넣지 않았다.

원문 추가:
- [LPBF304L2024](https://doi.org/10.1016/j.jmrt.2024.03.025): 6행, Table1/2/4; Sec2.1/2.4.
- [9Cr2024](https://doi.org/10.3390/ma17204960): 5행, Sec2.1/2.4; Table1.

새 희토류 독립 검증은 수행하지 않았다. 기본335행 예측모델은 교체하지 않았다. 새139행 모델은 results/NaCl_only/expanded/model_all139.joblib이며 전용 실험용이다.

```powershell
.venv/Scripts/python.exe -X utf8 -B build_nacl_dataset.py
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --semantic --conservative --input data/training_RE_NaCl.csv --output results/NaCl_only/baseline
.venv/Scripts/python.exe -X utf8 -B tune_phase_model.py --semantic --conservative --input data/training_RE_NaCl_expanded.csv --output results/NaCl_only/expanded
.venv/Scripts/python.exe -X utf8 -B report_nacl_results.py
```
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    readme=ROOT/'README.md'
    text=readme.read_text(encoding='utf-8')
    marker='<!-- NaCl-only experiment -->'
    block=f'''{marker}
NaCl 단독 환경으로 제한한 자료128행에 일반강 원문2편11행을 추가해 **139행·기존196컬럼**을 만들었다. 희토류30행·12편의 논문 단위 중첩 검증 R²는 **0.1101 → 0.1322**다. [NaCl 현재 실험 자료](data/training_RE_NaCl_expanded.csv), [같은30행 비교·적중률·재현방법](results/NaCl_only/report.md), [문헌18편 검토 기록](results/NaCl_only/literature_screening.csv)을 참고한다. 새희토류5행은 [확인 대기](data/NaCl_RE_pending.csv)로 분리했으며 학습에는 쓰지 않았다. 기본 예측모델은 유지했다.
<!-- /NaCl-only experiment -->

'''
    if marker not in text:
        lines=text.splitlines(keepends=True)
        text=lines[0]+'\n'+block+''.join(lines[1:])
        readme.write_text(text,encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=='__main__':main()
