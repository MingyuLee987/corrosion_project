"""Keep strict aqueous NaCl rows and append source-verified, unchanged-schema data."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/NaCl_only'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    original = pd.read_csv(ROOT / 'data/training_RE_clean.csv')
    mask = original.electrolyte_family.eq('NaCl') & original.pH_adjustment.isna()
    baseline = original.loc[mask].copy()
    baseline.to_csv(ROOT / 'data/training_RE_NaCl.csv', index=False, encoding='utf-8-sig')
    excluded = original.loc[~mask].copy()
    excluded['NaCl_filter_reason'] = np.where(excluded.electrolyte_family.eq('NaCl'),
        'added acid/base for pH adjustment; strict NaCl-only scope', 'not NaCl-only aqueous electrolyte')
    excluded.to_csv(OUT / 'excluded_rows.csv', index=False, encoding='utf-8-sig')
    additions = []
    def row(ref, sample, target, raw, unit, doi, source, title, **values):
        p = ROOT / source
        r = dict(record_id=f'NACL_{ref}_{sample}', reference_id=ref, sample_id=sample,
            RE_wt_pct=0., RE_element_cat='none', RE_content_status='no_intentional_RE_addition_reported_not_assayed',
            material_class='steel', icorr_A_cm2=target, log_icorr=np.log10(target),
            raw_icorr_value=raw, raw_icorr_unit=unit, electrolyte_family='NaCl',
            NaCl_wt_pct=3.5, electrolyte_type_cat='3.5 wt% NaCl',
            doi=doi, source_url=f'https://doi.org/{doi}', paper_title=title, publication_year=2024,
            source_file=source, source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            icorr_status='reported_table', target_method='PDP_Tafel', target_value_basis='reported_table_mean',
            origin='NaCl_literature_20261014', training_role='ordinary_steel_auxiliary',
            inhibitor_status='none', coating_status='none', composition_basis='reported_sample',
            source_verification='full_paper_composition_methods_target_table_verified',
            phase_verification_status='source_reported', temperature_description='room/ambient temperature; numeric value unreported',
            journal_quartile_filter='not_applied')
        r.update(values)
        unknown = set(r) - set(original.columns)
        if unknown:
            raise ValueError(f'Unexpected columns: {unknown}')
        additions.append(r)
    # Zhang et al., JMRT 29 (2024), author-hosted original PDF: Tables 1,2,4; Sec 2.1,2.4; Fig 5.
    ref='LPBF304L_2024'
    title='The effects of heat treatment and surface state on the corrosion resistance of laser powder bed fusion 304L stainless steel in 3.5 wt% NaCl solution'
    chem=dict(C_wt_pct=.014,N_wt_pct=.013,Si_wt_pct=.065,Mn_wt_pct=.054,P_wt_pct=.027,
        S_wt_pct=.003,Cr_wt_pct=19.07,Ni_wt_pct=9.62,Mo_wt_pct=.83,O_wt_pct=.031)
    for state,temp,grain,targets in [('as_received',None,8.,[21.2,64.5]),('1050',1050.,9.9,[117.1,224.3]),('1200',1200.,46.9,[23.2,1610.7])]:
        for surface,current in zip(['electropolished','sandpaper_ground_2000grit'],targets):
            row(ref,f'{state}_{surface}',current*1e-9,current,'nA/cm2','10.1016/j.jmrt.2024.03.025',
                'sources/LPBF304L_2024.pdf',title,**chem,alloy_system='304L stainless steel',
                processing_state_cat='LPBF_as_received' if temp is None else 'LPBF_annealed_water_quenched',
                final_heat_treatment_temp_C=temp,final_heat_treatment_time_min=None if temp is None else 30.,
                cooling_method_cat=None if temp is None else 'water_quench',
                surface_state_cat='electropolished' if surface=='electropolished' else 'polished',
                surface_preparation_cat=surface,gas_condition_cat='N2_continuously_purged',aeration_cat='deaerated',
                TP_scan_rate_mV_s=.167,TP_start_potential_V=-.3,TP_potential_basis_cat='vs_OCP',
                TP_reference_electrode_cat='SCE',TP_OCP_stabilization_time_s=3600.,
                NaCl_mol_L=.6,Cl_mol_L=.6,Cl_conversion_basis='0.6 M explicitly reported in source',
                matrix_grain_size_um=grain,phase_reported='austenite' if surface=='electropolished' else 'austenite; martensite',
                microstructure_constituents_reported='austenite' if surface=='electropolished' else 'austenite; grinding-induced martensite',
                phase_evidence_method='XRD',phase_source_locator='Figure5; Sec3',
                phase_source_url='https://doi.org/10.1016/j.jmrt.2024.03.025',target_locator='Table4 (nA/cm2); chemistry Table1',
                remarks='Table4 mean only; three tests not expanded into synthetic replicate rows. EP-only grain sizes not verified for ground surface; bulk grain size copied with this caveat. No numeric room temperature imputed; N2 OCP conditioning distinct from passive-film EIS pretreatment.')
    # Li & Bai, Materials 17 (2024), 4960: Sec2.1 composition; Table1 current; Sec2.4 protocol.
    ref='Tempered9Cr_2024'
    title='Effect of Tempering Temperature on the Aqueous Corrosion Resistance of 9Cr Series Heat-Resistant Steel'
    for temp,current in zip([680,700,720,740,760],[.692,.653,.612,.619,.704]):
        row(ref,str(temp),current*1e-6,current,'uA/cm2','10.3390/ma17204960','sources/Tempered9Cr_2024.pdf',title,
            alloy_system='9Cr heat-resistant steel',C_wt_pct=.12,Si_wt_pct=.2,Mn_wt_pct=1.,Cr_wt_pct=9.67,
            Mo_wt_pct=2.,Co_wt_pct=1.32,Ni_wt_pct=.18,V_wt_pct=.3,Nb_wt_pct=.04,N_wt_pct=.01,
            melting_route_cat='VIM',final_heat_treatment_temp_C=temp,final_heat_treatment_time_min=120.,
            austenitizing_temperature_C=1150.,austenitizing_time_min=60.,processing_state_cat='tempered',
            heat_treatment_sequence='1150 C 1 h, air cool; temper 2 h',
            phase_reported='martensite',microstructure_constituents_reported='lath tempered martensite; fine precipitates',
            phase_evidence_method='OM; SEM',phase_source_locator='Sec3.1.2 Figure2',
            phase_source_url='https://doi.org/10.3390/ma17204960',surface_state_cat='surface_untreated',
            TP_scan_rate_mV_s=.5,TP_OCP_stabilization_time_s=1800.,TP_reference_electrode_cat='SCE',
            TP_start_potential_V=-.2,TP_end_potential_V=.3,target_locator='Table1; chemistry Sec2.1; PDP Sec2.4',
            remarks='Reported composition includes B=0.005 wt%, retained in remarks because schema has no B column. Ambient numeric temperature, gas and exact PDP pre-exposure unreported. Humidity-chamber observations are separate; 30-day chamber exposure not assigned to PDP. Three replicates represented by one reported mean.')
    add=pd.DataFrame(additions,columns=original.columns)
    # EP grain measurements apply to EP specimens; leave ground-surface measurement missing.
    add.loc[add.surface_preparation_cat.eq('sandpaper_ground_2000grit'),'matrix_grain_size_um']=np.nan
    add.to_csv(ROOT/'data/NaCl_verified_additions.csv',index=False,encoding='utf-8-sig')
    expanded=pd.concat([baseline,add],ignore_index=True)
    assert expanded.columns.tolist()==original.columns.tolist()
    assert expanded.record_id.is_unique and expanded.icorr_A_cm2.gt(0).all()
    assert not set(add.doi) & set(original.doi.dropna())
    expanded.to_csv(ROOT/'data/training_RE_NaCl_expanded.csv',index=False,encoding='utf-8-sig')
    pending=[]
    for sample,re,current in [('control',0.,4.13e-5),('RE',.007,2.92e-5)]:
        pending.append(dict(record_id=f'PENDING_Q355B_2026_{sample}',reference_id='CeLaQ355B_2026',sample_id=sample,
            alloy_system='Q355B steel',RE_wt_pct=re,RE_nominal_wt_pct=re,RE_element_cat='La+Ce',
            RE_content_status='abstract_addition_amount_not_verified_final_assay',
            icorr_A_cm2=current,log_icorr=np.log10(current),NaCl_wt_pct=3.5,electrolyte_family='NaCl',
            doi='10.1007/s11663-026-04267-x',source_url='https://link.springer.com/article/10.1007/s11663-026-04267-x',
            target_locator='publisher abstract',icorr_status='reported_abstract',primary_eligible=False,
            exclusion_reason='full chemistry and PDP conditions unavailable; RE addition versus final assay not resolved',
            training_role='pending_not_used',remarks='Published 2026-09-08; no chemistry inferred from steel grade. Inclusion mean diameter 2.84/2.16 um; number density 9.18/7.63 mm^-2 reported in abstract.',
            inclusion_number_density_mm2=9.18 if re==0 else 7.63))
    # Chemistry of a new RE study is verified, but passive current is not an icorr label.
    for sample,chem,re,bounds in [
        ('REE-0',[.027,8.113,18.064,.864,.024,.345,.002,.004,.236],0.,'not intentionally added'),
        ('REE-1',[.020,8.14,18.242,.854,.052,.456,.009,.006,.203],.005,'0.004-0.006 wt%'),
        ('REE-2',[.019,8.014,18.349,1.004,.05,.435,.009,.005,.206],.010,'0.009-0.011 wt%')]:
        r=dict(record_id=f'PENDING_RE304_2026_{sample}',reference_id='RE304_2026',sample_id=sample,
            RE_wt_pct=re,RE_element_cat='La+Ce',NaCl_wt_pct=3.5,electrolyte_family='NaCl',
            alloy_system='304 austenitic stainless steel',doi='10.1016/j.jmrt.2026.02.234',
            source_url='https://www.sciencedirect.com/science/article/pii/S2238785426004849',
            composition_basis='Table1; RE range midpoint used; original range preserved in remarks',
            primary_eligible=False,training_role='pending_not_used',
            exclusion_reason='icorr not verified; passive-region current, pitting potential and EIS not substituted',
            remarks=f'Table1 RE={bounds}; Sn=0.003 wt% (no Sn schema column). No assumed icorr.',
            casting_method_cat='continuous_casting',processing_state_cat='hot_rolled',rolled_thickness_mm=20.)
        r.update(dict(zip(['C_wt_pct','Ni_wt_pct','Cr_wt_pct','Mn_wt_pct','Cu_wt_pct','Si_wt_pct','Mo_wt_pct','Al_wt_pct','Co_wt_pct'],chem)))
        pending.append(r)
    pd.DataFrame(pending,columns=original.columns).to_csv(ROOT/'data/NaCl_RE_pending.csv',index=False,encoding='utf-8-sig')
    audit=dict(original_rows=len(original),baseline_rows=len(baseline),excluded_rows=len(excluded),
        columns=len(original.columns),new_rows=len(add),expanded_rows=len(expanded),
        RE_rows=int(baseline.RE_wt_pct.gt(0).sum()),RE_papers=int(baseline.loc[baseline.RE_wt_pct.gt(0),'reference_id'].nunique()),
        filter='electrolyte_family == NaCl and no recorded added acid/base pH adjustment',
        limitations='Existing metadata filter is not a complete original-paper re-audit. N2/air condition permitted; CO2-saturated solutions excluded. No synthetic rows; original file preserved.')
    (OUT/'curation.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(audit,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
