"""Source-located RE additions without a journal-quartile filter."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from composition_values import apply_composition_values

ROOT=Path(__file__).resolve().parent

def main():
    rows=[]
    table=json.loads((ROOT/'sources/literature/HRB400_RE_2025_tables.json').read_text(encoding='utf-8'))
    assert [float(r[2]) for r in table[3][1:]]==[7.612,4.937,4.727]
    for sample,c,si,mn,cu,cr,v,re,current in [
        ('0RE',.11,.38,1.29,.28,.92,.022,0.,7.612),
        ('0.003RE',.10,.39,1.30,.28,.89,.023,.003,4.937),
        ('0.010RE',.11,.42,1.29,.29,.90,.027,.010,4.727)]:
        rows.append(dict(reference_id='HRB400_RE_2025',sample_id=sample,C_wt_pct=c,Si_wt_pct=si,
            Mn_wt_pct=mn,Cu_wt_pct=cu,Cr_wt_pct=cr,V_wt_pct=v,RE_wt_pct=re,RE_element_cat='La+Ce',
            La_Ce_addition_mass_ratio=3.,icorr_A_cm2=current*1e-6,electrolyte_family='NaCl',NaCl_wt_pct=3.5,
            temperature_C=25.,TP_scan_rate_mV_s=1.,TP_reference_electrode_cat='SCE',
            processing_state_cat='hot_rolled',melting_route_cat='vacuum_induction',cooling_method_cat='air',
            rolling_soak_temperature_C=1050.,rolling_soak_time_min=40.,rolling_start_temperature_C=1050.,
            rolling_finish_temperature_min_C=950.,rolling_finish_temperature_max_C=990.,
            cooling_bed_entry_temperature_C=950.,surface_state_cat='polished',
            target_locator='PDF printed p62 Table4; composition/process printed p59 Tables1-2 and Sec1',
            source_url='https://www.specialsteeljournal.com/zh/article/doi/10.20057/j.1003-8620.2024-00280/',
            doi='10.20057/j.1003-8620.2024-00280',journal='Special Steel',
            source_file='sources/HRB400_RE_2025.pdf',source_verification='PDF_tables_visually_checked',
            quality_flags='P,S and OCP stabilization unreported. 168h is weight-loss exposure, not verified PDP pre-exposure. RE table chemistry; assay method unspecified.',
            strict300_eligible=False))
    # Publisher-hosted PDF parsed by the web tool, Tables1-2 and Sec1.
    # PDF image retrieval unavailable: explicitly track text-only verification.
    chemistry=[(.047,.24,1.80,.011,.0050,.28,.46,.24,.034,0.,[7.99e-6,3.76e-5,7.1e-5]),
               (.042,.20,1.88,.012,.0048,.27,.43,.25,.031,.008,[1.12e-5,4.23e-6,2.68e-6]),
               (.042,.23,1.90,.012,.0047,.27,.46,.24,.032,.15,[3.31e-5,6.19e-6,8.11e-6])]
    for sample,(c,si,mn,p,s,cr,ni,mo,al,re,currents) in enumerate(chemistry):
        for days,current in zip([1,28,53],currents):
            rows.append(dict(reference_id='Y_X100_2018',sample_id=f'{sample}_{days}d',C_wt_pct=c,Si_wt_pct=si,
                Mn_wt_pct=mn,P_wt_pct=p,S_wt_pct=s,Cr_wt_pct=cr,Ni_wt_pct=ni,Mo_wt_pct=mo,
                Al_wt_pct=al,Cu_wt_pct=.2,RE_wt_pct=re,RE_element_cat='Y',icorr_A_cm2=current,
                electrolyte_family='NaCl',NaCl_wt_pct=3.5,pre_exposure_h=days*24.,
                temperature_description='room_temperature_no_numeric_value',TP_scan_rate_mV_s=.5,
                OCP_stabilization_s=1200.,TP_reference_electrode_cat='SCE',processing_state_cat='quenched_tempered',
                melting_route_cat='vacuum_induction',final_heat_treatment_temp_C=650.,
                austenitizing_temperature_C=935.,rolling_soak_temperature_C=1200.,rolling_soak_time_min=120.,
                surface_state_cat='acetone_cleaned',
                target_locator='Publisher PDF printed pp102-103 Tables1-2; Sec1.1-1.2',
                source_url='https://cdn.sciengine.com/doi/pdf/4A040ADEAFCF45B4BD8F9C554DCBF122',
                doi='10.16533/j.cnki.15-1099/tf.201803014',journal='Chinese Rare Earths',
                source_verification='publisher_PDF_web_text_tables_checked; image_QA_unavailable',
                quality_flags='QT hold times/quench medium and numeric PDP temperature unreported; repeated exposure states are not independent steels; entire paper grouped. Y=0.15 is published Table1, not silently corrected.',
                strict300_eligible=False))
    d=apply_composition_values(pd.DataFrame(rows))
    d['record_id']='RENEW_'+d.reference_id+'_'+d.sample_id
    d['material_class']='low alloy steel'
    d['log_icorr']=np.log10(d.icorr_A_cm2)
    d['journal_quartile_filter']='none_user_authorized'
    d['source_sha256']=d.source_file.map(lambda f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() if pd.notna(f) else '')
    d.to_csv(ROOT/'data/RE_all_journals_additions.csv',index=False,encoding='utf-8-sig')
    original=pd.read_csv(ROOT/'data/training_process_curated.csv')
    assert not set(d.record_id)&set(original.record_id)
    combined=pd.concat([original,d],ignore_index=True)
    combined.to_csv(ROOT/'data/training_RE_all_journals.csv',index=False,encoding='utf-8-sig')
    print(dict(new_rows=len(d),new_RE_positive=int(d.RE_wt_pct.gt(0).sum()),combined_rows=len(combined)))

if __name__=='__main__':main()
