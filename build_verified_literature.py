"""Transcribe checked literature rows; keep strict readiness separate from row count."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd
from composition_values import apply_composition_values

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/process_optimization'
LIT = ROOT / 'sources/literature'
ROWS = []


def add(reference, sample, current, chemistry, process, test, locator, url, file, **extra):
    row = dict(record_id=f'PROC_{reference}_{sample}',reference_id=reference,sample_id=sample,
        material_class='steel',RE_wt_pct=0.,RE_element_cat='none',RE_content_status='no_intentional_RE_addition_reported_not_assayed',
        icorr_A_cm2=current,raw_icorr_unit='A/cm2',log_icorr=float(np.log10(current)),
        source_url=url,source_sha256=hashlib.sha256((ROOT/file).read_bytes()).hexdigest(),
        target_locator=locator,target_method='PDP_Tafel',composition_basis='source_reported_scalar_chemistry',
        origin='process_literature_collection_2026_10_13',**chemistry,**process,**test,**extra)
    ROWS.append(row)


def main():
    # XRF chemistry and representative PDP curves; 10^-3 mA/cm2 = 10^-6 A/cm2.
    chemistry=dict(C_wt_pct=.027,Si_wt_pct=.38,Mn_wt_pct=1.4,P_wt_pct=.031,S_wt_pct=.004,
                   Cr_wt_pct=18.,Ni_wt_pct=8.01,N_wt_pct=.075)
    for sample,temp,current,pickled in [('S20',20.,12.1,False),('S50',50.,24.,False),('SP50',50.,80.5,True)]:
        add('Sensitized304_2022',sample,current*1e-6,chemistry,
            dict(processing_state_cat='sensitized',final_heat_treatment_temp_C=650.,
                 final_heat_treatment_time_min=2400.,cooling_method_cat='slow_air',
                 surface_state_cat='pickled' if pickled else 'heat_tinted',
                 pickling_temperature_C=50. if pickled else np.nan,pickling_time_min=10. if pickled else np.nan,
                 heat_treatment_sequence='650C/40h/slow_air; acid_pickling_50C_10min' if pickled else '650C/40h/slow_air'),
            dict(temperature_C=temp,electrolyte_family='NaCl_HCl',NaCl_mol_L=.9,HCl_mol_L=.1,Cl_mol_L=1.,
                 pH=1.1,TP_scan_rate_mV_s=1.,OCP_stabilization_s=600.,TP_reference_electrode_cat='SCE'),
            'Tables1,3,5; Sec2; Table5 '+sample,'https://www.mdpi.com/1996-1944/15/23/8543',
            'sources/literature/Sensitized304_2022.html',doi='10.3390/ma15238543',
            composition_verification='reported_XRF',process_completeness='quantitative_heat_and_cooling_complete',
            strict300_eligible=True,target_statistic='representative_curve; >=3 repeats not individually published')
    # Supplier scalar chemistry is published; no assay method or rolling reduction is given.
    chemistry=dict(C_wt_pct=.031,Si_wt_pct=.68,Mn_wt_pct=4.71,P_wt_pct=.03,S_wt_pct=.02,
                   Cr_wt_pct=21.4,Ni_wt_pct=1.56,Mo_wt_pct=.19,Cu_wt_pct=.33,N_wt_pct=.23,Co_wt_pct=.058)
    conditions=[
        ('H2SO4',3,[661.4,1042.,1277.,1378.],dict(H2SO4_mol_L=1.)),
        ('HCl',4,[15.3,17.3,20.4,367.9],dict(HCl_mol_L=.1)),
        ('NaCl',5,[.3,1.,11.2,16.1],dict(NaCl_wt_pct=3.5)),
        ('HCl_NaCl',6,[1.6,23.5,337.,379.],dict(HCl_mol_L=.1,NaCl_wt_pct=3.5)),
        ('black_liquor',7,[.12,.55,.41,.48],{}),
        ('green_liquor',8,[19.7,29.,38.1,155.],{}),
        ('white_liquor',9,[122.,562.,612.,1695.],{}),
        ('ClO2_bleaching',10,[.01,.03,.07,.17],{}),
    ]
    text=(LIT/'S32101_2025.md').read_text(encoding='utf-8')
    for family,table,currents,concentration in conditions:
        assert f'**Table {table}.' in text
        block=text[text.rfind(f'**Table {table}.'):]
        block=block.split(f'**Table {table+1}.')[0]
        target_line=next(line for line in block.splitlines() if line.startswith('| icorr ('))
        checked=[float(re.search(r'[0-9.]+',cell).group()) for cell in target_line.split('|')[2:6]]
        np.testing.assert_array_equal(checked,currents)
        for temp,current in zip([20.,30.,40.,50.],currents):
            add('S32101_2025',f'{family}_{int(temp)}C',current*1e-6,chemistry,
                dict(processing_state_cat='cold_rolled',surface_state_cat='polished',
                     process_completeness='cold_rolled_state_known; reduction_and_upstream_heat_unreported'),
                dict(temperature_C=temp,electrolyte_family=family,TP_scan_rate_mV_s=.167,
                     OCP_stabilization_s=7200.,TP_reference_electrode_cat='Ag_AgCl',**concentration),
                f'Tables1,{table}; Sec2.1-2.2; {temp:g}C column','https://www.mdpi.com/1996-1944/18/9/1921',
                'sources/literature/S32101_2025.md',doi='10.3390/ma18091921',
                composition_verification='reported_scalar_composition_assay_method_unreported',strict300_eligible=False,
                target_statistic='reported_mean; replicates not individually tabulated',
                quality_flags='Typical liquor composition is not exact sample chemistry; liquor concentration columns left missing. OCP time reported as typically 2h.')
    # Five actual published mean results, not 15 fabricated replicate rows.
    chemistry=dict(C_wt_pct=.18,Si_wt_pct=.15,Mn_wt_pct=.7,P_wt_pct=.035,S_wt_pct=.04,
                   Cr_wt_pct=.4,Ni_wt_pct=.4,Mo_wt_pct=.15)
    for sample,current,rate,tempered in [('No-HT',12.15,None,False),('SR',6.36,.3928,False),
                                      ('FR',9.91,58.92,False),('Temp_SR',5.39,.3928,True),('Temp_FR',3.97,58.92,True)]:
        heat=sample!='No-HT'
        add('Heating8620_2024',sample,current*1e-6,chemistry,
            dict(processing_state_cat='quenched_tempered' if tempered else ('quenched' if heat else 'cold_rolled'),
                 final_heat_treatment_temp_C=500. if tempered else (850. if heat else np.nan),
                 final_heat_treatment_time_min=20. if tempered else (40. if heat else np.nan),
                 cooling_method_cat='air' if tempered else ('water' if heat else 'not_applicable'),
                 austenitizing_temperature_C=850. if heat else np.nan,austenitizing_time_min=40. if heat else np.nan,
                 austenitizing_heating_rate_C_s=rate,quench_medium_cat='DI_water' if heat else 'not_applicable',
                 heat_treatment_sequence='850C/40min/DI_water_25C;500C/20min/air' if tempered else ('850C/40min/DI_water_25C' if heat else 'cold_rolled_as_supplied')),
            dict(electrolyte_family='NaCl',NaCl_wt_pct=3.5,TP_scan_rate_mV_s=1.,
                 OCP_stabilization_minimum_s=1800.,TP_reference_electrode_cat='SCE',temperature_description='not_numerically_reported'),
            'Tables1-2; Sec2.1-2.2,2.5','https://www.mdpi.com/2624-5558/5/3/16',
            'sources/literature/Heating8620_2024.html',doi='10.3390/cmd5030016',
            composition_verification='reported_commercial_chemistry_assay_method_unreported',strict300_eligible=False,
            process_completeness='quantitative_heat_complete_but_PDP_temperature_unreported' if heat else 'cold_rolled_history_partial',
            target_statistic='reported_mean_n3',quality_flags='Possible repeated specimens before/after tempering; do not split by sample row.')
    # Carbon steel heat treatments: original PDF Table3. Laser rows have incomplete thermal history and are not transcribed.
    bases={
        'X':[.214,.94,.26,.011,.042,.112,.01,.005],
        'V1':[.21,1.,.28,.031,.033,.122,.001,.003],
        'V2':[.195,1.27,.32,.012,.008,.053,.001,.004],
        'T1':[.214,1.02,.28,.022,.04,.138,.087,.047],
        'T2':[.23,.98,.37,.04,.02,.1,.01,.1],
    }
    for sample,currents in [('X',[19.58,20.35]),('V1',[10.63,15.76]),('V2',[5.368,7.917]),
                            ('T1',[4.375,7.291]),('T2',[9.662,10.52])]:
        chemistry=dict(zip(['C_wt_pct','Mn_wt_pct','Si_wt_pct','P_wt_pct','S_wt_pct','Cr_wt_pct','Ni_wt_pct','Al_wt_pct'],bases[sample]))
        if sample.startswith('V'):
            chemistry['V_wt_pct']=.12 if sample=='V1' else .22
        if sample.startswith('T'):
            chemistry['Ti_wt_pct']=.12 if sample=='T1' else .2
        for temp,current in zip([900.,1200.],currents):
            add('LowC_heat_2012',f'{sample}_{int(temp)}',current*1e-6,chemistry,
                dict(processing_state_cat='quenched_tempered',austenitizing_temperature_C=temp,
                     austenitizing_time_min=60.,final_heat_treatment_temp_C=600.,final_heat_treatment_time_min=60.,
                     cooling_method_cat='air',quench_medium_cat='water',
                     heat_treatment_sequence=f'{temp:g}C/60min/water;600C/60min/air',surface_state_cat='polished'),
                dict(electrolyte_family='CaOH2_NaCl',NaCl_mol_L=.6,Cl_mol_L=.6,pH=12.,
                     OCP_stabilization_s=5400.,TP_scan_rate_mV_s=.5,TP_reference_electrode_cat='SCE',temperature_description='not_reported'),
                'Table1 PDFp2; Sec2.2 PDFp3; Table3 PDFp9; Sec3.3 PDFp8-9',
                'https://www.electrochemsci.org/papers/vol7/7086677.pdf','sources/LowC_heat_2012.pdf',
                composition_verification='reported_experimental_steel_composition',strict300_eligible=False,
                process_completeness='quantitative_heat_complete_but_PDP_temperature_unreported',
                target_statistic='single_published_value',quality_flags='Narrative says equal C, but Table1 differs: tabulated compositions retained.')
    chemistry=dict(C_wt_pct=.564,Si_wt_pct=.299,Mn_wt_pct=.865,P_wt_pct=.0153,S_wt_pct=.0162,
                   Ni_wt_pct=.0381,Cr_wt_pct=1.15,Mo_wt_pct=.0176,Cu_wt_pct=.026,Al_wt_pct=.0306,V_wt_pct=.139)
    cases=[('N',3.5,13.01,'normalized'),('Q',3.5,16.25,'quenched'),('QT',3.5,49.87,'quenched_tempered'),
           ('QT_tap',None,5.24,'quenched_tempered'),('QT_1pct',1.,10.79,'quenched_tempered'),('QT_5pct',5.,650.80,'quenched_tempered')]
    for sample,salt,current,state in cases:
        heat=state!='normalized'; tempered=state=='quenched_tempered'
        add('SpringSteel_2023',sample,current*1e-6,chemistry,
            dict(processing_state_cat=state,austenitizing_temperature_C=850. if heat else np.nan,
                 austenitizing_time_min=15. if heat else np.nan,final_heat_treatment_temp_C=600. if tempered else (850. if heat else np.nan),
                 final_heat_treatment_time_min=60. if tempered else (15. if heat else np.nan),
                 cooling_method_cat='air' if tempered else ('oil' if heat else 'not_reported'),
                 quench_medium_cat='oil' if heat else 'not_applicable',surface_state_cat='polished'),
            dict(electrolyte_family='tap_water' if salt is None else 'NaCl',NaCl_reported_pct=salt,
                 NaCl_concentration_basis='g_per_100mL_water; not established solution mass percent' if salt is not None else 'not_applicable',
                 OCP_stabilization_s=3600.,TP_reference_electrode_cat='SCE'),
            'Table1-2 PDFp2; Table3 PDFp3; Table4 PDFp4; identical QT3.5% row counted once',
            'https://e-university.tu-sofia.bg/e-conf/files/169/paper_10.47978%40TUS.2023.73.04.002.pdf',
            'sources/SpringSteel_2023.pdf',doi='10.47978/TUS.2023.73.04.002',composition_verification='reported_GDS_assay',
            strict300_eligible=False,process_completeness='heat_complete_for_Q_QT_but_PDP_temperature_scan_rate_unreported',
            target_statistic='single_published_value',quality_flags='QT3.5% Ecorr differs between Tables3/4; same icorr deduplicated. Inhibitor-treated rows excluded.')
    chemistry=dict(C_wt_pct=.544,Si_wt_pct=1.338,Mn_wt_pct=.699,P_wt_pct=.009,S_wt_pct=.003,
                   Cr_wt_pct=.658,Mo_wt_pct=.003,Ni_wt_pct=.127,Al_wt_pct=.226,Cu_wt_pct=.150,Ti_wt_pct=.005,N_wt_pct=.003)
    qp_text=(LIT/'QP9254_DOI.md').read_text(encoding='utf-8')
    for qtemp,ptemp,current in [(250.,400.,58.),(250.,300.,151.),(220.,400.,74.),(220.,300.,107.)]:
        label=f'{int(qtemp)}–{int(ptemp)}'
        table_row=[line for line in qp_text.splitlines() if line.startswith('| '+label+' |')][-1]
        assert float(table_row.split('|')[-2].strip())==current
        add('QP9254_2025',label,current*1e-6,chemistry,
            dict(processing_state_cat='quenched_partitioned',austenitizing_temperature_C=900.,austenitizing_time_min=12.,
                 quench_stop_temperature_C=qtemp,quench_holding_time_min=10.,
                 partitioning_temperature_C=ptemp,partitioning_time_min=8.,
                 final_heat_treatment_temp_C=ptemp,final_heat_treatment_time_min=8.,
                 cooling_method_cat='helium_gas',quench_cooling_rate_C_s=100.,final_cooling_rate_C_s=50.,surface_state_cat='polished',
                 heat_treatment_sequence=f'900C/12min; helium_100C_s_to_{qtemp:g}C/10min; {ptemp:g}C/8min;helium_50C_s_to_room'),
            dict(electrolyte_family='H2SO4',H2SO4_mol_L=.1,temperature_description='room_temperature_without_numeric_value',
                 OCP_stabilization_s=3600.,TP_scan_rate_mV_s=1.,TP_reference_electrode_cat='Ag_AgCl'),
            'Tables1,2,5; Sec2.1,2.5; Table5 '+label,'https://www.mdpi.com/2075-4701/15/5/509',
            'sources/literature/QP9254_DOI.md',doi='10.3390/met15050509',
            composition_verification='reported_commercial_scalar_composition',strict300_eligible=False,
            process_completeness='quantitative_QP_schedule_complete; numerical_PDP_temperature_unreported',target_statistic='single_published_value')
    chemistry=dict(C_wt_pct=.02,Si_wt_pct=.28,Mn_wt_pct=1.2,P_wt_pct=.02,S_wt_pct=.01,
                   Cr_wt_pct=11.7,Ni_wt_pct=.68,N_wt_pct=.013)
    for reduction,currents in [(50.,[5.613,3.269,2.433]),(90.,[3.71,2.81,2.01])]:
        for temp,current in zip([720.,740.,770.],currents):
            add('ColdRollX2_2022',f'{int(reduction)}pct_{int(temp)}C',current*1e-6,chemistry,
                dict(processing_state_cat='cold_rolled_annealed',rolling_reduction_reported_pct=reduction,
                     final_heat_treatment_temp_C=temp,final_heat_treatment_time_min=30.,cooling_method_cat='air',surface_state_cat='polished'),
                dict(electrolyte_family='NaCl',NaCl_wt_pct=3.5,OCP_stabilization_s=1800.,
                     TP_scan_rate_mV_s=1.,TP_reference_electrode_cat='SCE',temperature_description='not_reported'),
                'Table1 chemistry; Tables3-4 PDP; Sec2.1,2.3','https://www.mdpi.com/1996-1944/15/19/6914',
                'sources/literature/ColdRollX2_2022.md',doi='10.3390/ma15196914',
                composition_verification='reported_commercial_scalar_composition',strict300_eligible=False,
                process_completeness='rolling_reduction_and_anneal_schedule_complete; PDP_temperature_unreported',
                target_statistic='reported_mean_n3',quality_flags='Reported nominal reductions retained; not recalculated from rounded sheet thicknesses.')
    d=apply_composition_values(pd.DataFrame(ROWS))
    assert len(d)==66 and d.record_id.is_unique and d.icorr_A_cm2.gt(0).all()
    assert d[['C_wt_pct','Si_wt_pct','Mn_wt_pct','P_wt_pct','S_wt_pct']].notna().all().all()
    original_core=json.loads((LIT/'Sensitized304_2022_tables.json').read_text(encoding='utf-8'))[1]
    assay=dict(zip(original_core[0],original_core[1]))
    for element in ['C','Si','Mn','P','S','Cr','Ni','N']:
        np.testing.assert_allclose(d.loc[d.reference_id.eq('Sensitized304_2022'),element+'_wt_pct'],float(assay[element]))
    curves=json.loads((LIT/'Sensitized304_2022_tables.json').read_text(encoding='utf-8'))[5]
    for row in curves[1:]:
        if row[2]=='-':
            continue
        record=d.loc[d.reference_id.eq('Sensitized304_2022') & d.sample_id.eq(row[0].replace(' ',''))]
        assert len(record)==1
        np.testing.assert_allclose(record.icorr_A_cm2,float(row[2].split('±')[0])*1e-6)
    d.to_csv(ROOT/'data/process_literature_verified.csv',index=False,encoding='utf-8-sig')
    d[d.strict300_eligible].to_csv(ROOT/'data/process_literature_strict_ready.csv',index=False,encoding='utf-8-sig')
    summary=d.groupby('reference_id').agg(rows=('record_id','size'),strict_ready=('strict300_eligible','sum')).reset_index()
    summary.to_csv(OUT/'new_literature_summary.csv',index=False,encoding='utf-8-sig')
    print(summary.to_string(index=False))
    print('New source-linked rows:',len(d),'Strict ready:',int(d.strict300_eligible.sum()),'Target 300 NOT reached.')


if __name__=='__main__':
    main()
