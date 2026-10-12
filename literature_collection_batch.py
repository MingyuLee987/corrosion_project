"""Primary-source extraction batch. Unknown chemistry is never replaced by zero."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

def verified_additions():
    rows=[]
    def add(ref,sample,h,value,url,doi,**kw):
        row=dict(record_id=f'{ref}_{sample}_{h}h',reference_id=ref,sample_id=sample,
            doi=doi,source_url=url,raw_icorr_value=value,raw_icorr_unit='uA/cm2',
            icorr_A_cm2=value*1e-6,pre_exposure_h=h,immersion_time_h=h,
            pre_exposure_type='continuous_immersion',target_method='potentiodynamic_Tafel',
            time_evidence_status='explicit_PDP_link',time_basis='source_reported_pre_PDP_continuous_immersion',
            time_source_url=url,composition_source_url=url,data_role='general_steel',
            validation_role='development',primary_eligible=True,origin='primary_source_collection_batch')
        row.update(kw);rows.append(row)
    url='https://www.mdpi.com/2075-4701/15/11/1249'
    comps={'CS':(.11,.18,.44,np.nan),'5Al':(.91,.42,28.12,4.95),
           '7Al':(.89,.44,28.44,7.13),'9Al':(.88,.41,27.89,9.19)}
    for h,values,sds,ecorr in [(168,[92.33,78.73,73.73,82.46],[4.79,2.16,2.18,4.13],[-.940,-.850,-.805,-.946]),
                                (840,[213.7,47.24,9.89,13.49],[7.79,3.15,.79,3.01],[-.906,-.798,-.768,-.770])]:
        for sample,value,sd,e in zip(comps,values,sds,ecorr,strict=True):
            c,si,mn,al=comps[sample]
            add('FeMnAl25',sample,h,value,url,'10.3390/met15111249',
                C_wt_pct=c,Si_wt_pct=si,Mn_wt_pct=mn,Al_wt_pct=al,
                material_class='carbon steel' if sample=='CS' else 'high manganese steel',
                alloy_system=sample,electrolyte_family='NaCl',
                pre_exposure_electrolyte='35 g/L NaCl',pre_exposure_temperature_C=35,
                temperature_C=25,PDP_measurement_mode_cat='ex_situ',surface_state_cat='rusted',
                TP_scan_rate_mV_s=.01667,TP_start_potential_V=-2,TP_end_potential_V=1,
                TP_reference_electrode_cat='SCE',OCP_stabilization_s=3600,
                Ecorr_V=e,icorr_sd_A_cm2=sd*1e-6,journal='Metals',publication_year=2025,
                paper_title='Effect of Aluminum Content on the Corrosion Behavior of Fe-Mn-Al-C Structural Steels in Marine Environments',
                composition_evidence_status='sample_specific_table',composition_source_locator='Table 1 PDF p3',
                composition_table_locator='Table 1',target_locator='Table 2 PDF p11; Fig 10: 7/35 d full immersion',
                time_source_locator='Sec 2.2 full immersion; Sec 2.4 corroded specimens; Fig 10/Table 2 PDF p11',
                quality_flags='Cr/Ni/Mo reported as merged upper-bound cell, not individual values; RE unreported not zero; CS Al dash not zero; 24h desiccator storage not immersion; retain printed scan rate; mass-loss descaling not assigned to PDP')
    url='https://www.ams.org.cn/article/2020/0412-1961/0412-1961-2020-6-909.shtml'
    for sample,value,e in [('Diamond',.030,-.158),('Al2O3',.039,-.153),('QC',.028,-.146)]:
        add('QC304_20',sample,24,value,url,'10.11900/0412.1961.2019.00309',
            validation_role='external_locked',C_wt_pct=.05,Mo_wt_pct=.11,Si_wt_pct=.60,
            Mn_wt_pct=1.21,Ni_wt_pct=8.11,Cr_wt_pct=17.13,
            material_class='austenitic stainless steel',alloy_system='304',
            electrolyte_family='NaCl',pre_exposure_electrolyte='3.5% NaCl; percentage basis unassigned',
            temperature_description='room_temperature',gas_condition_cat='aerated',
            surface_state_cat='passivated',surface_treatment=f'{sample} abrasive;35vol%HNO3 45C30min;Na2CO3 neutralization',
            TP_scan_rate_mV_s=1,TP_start_potential_V=-.3,TP_end_potential_V=1,
            TP_potential_basis_cat='vs_OCP',TP_reference_electrode_cat='AgAgCl_saturated_KCl',
            Ecorr_V=e,journal='Acta Metallurgica Sinica',publication_year=2020,
            paper_title='Passivation Behavior on the Surface of Stainless Steel Reinforced by Quasicrystal-Abrasive via Finite Element Simulation',
            composition_evidence_status='reported_in_text',composition_source_locator='Experimental methods quantitative 304 composition',
            target_locator='Table 3: icorr in 10^-7 A/cm2; Fig 11',
            time_source_locator='Fig 11 caption: polarization after 24 h immersion in 3.5% NaCl',
            quality_flags='External publication locked before modelling; chemical passivation domain shift; RE unreported not zero; Table3 multiplier explicitly applied: .30/.39/.28 x10^-7 A/cm2')
    return pd.DataFrame(rows)

def save_audit():
    rows=[]
    for sample,ce,values in [('1',0,[2.463,2.536,3.482,3.537,2.652]),
                             ('2',.009,[1.357,1.984,2.152,2.409,2.127]),
                             ('3',.034,[2.289,2.359,2.365,3.034,2.209])]:
        for day,value in zip([3,5,7,14,21],values,strict=True):
            rows.append(dict(reference_id='AH36_Ce25',doi='10.13228/j.boyuan.issn1006-9356.20240777',
                sample_id=sample,Ce_wt_pct=ce,pre_exposure_h=day*24,raw_icorr_value=value,
                raw_icorr_unit='A/cm2 (printed)',icorr_A_cm2=np.nan,
                target_locator='Table 3 PDF p111; visually verified',
                review_status='quarantined_unit_conflict',
                exclusion_reason='Printed A/cm2 values contradict polarization curve scale; no multiplier guessed'))
    for sample,ce,value in [('C0',0,18.923),('C1',.023,np.nan),('C2',.044,3.35),('C3',.065,5.98)]:
        rows.append(dict(reference_id='HRB500E24',doi='10.1038/s41598-024-67799-9',
            sample_id=sample,Ce_wt_pct=ce,pre_exposure_h=960,raw_icorr_value=value,
            raw_icorr_unit='mA/cm2 (printed)',icorr_A_cm2=np.nan,
            target_locator='Fig 6 PDF p7; methods and conclusion',review_status='quarantined_unit_conflict',
            exclusion_reason='Current unit conflicts PDP curve scale; C2 3.35 vs6.350 text conflict; C1 numeric target not extracted'))
    pd.DataFrame(rows).to_csv(ROOT/'data/review/time_pending_verification.csv',index=False,encoding='utf-8-sig')
    registry=pd.read_csv(ROOT/'data/review/time_literature_review.csv',encoding='utf-8-sig')
    decisions=[
        ('FeMnAl25','10.3390/met15111249','https://www.mdpi.com/2075-4701/15/11/1249','included_continuous','8 rows: Tables1/2, Fig10; 7/35d full immersion linked to PDP'),
        ('QC304_20','10.11900/0412.1961.2019.00309','https://www.ams.org.cn/article/2020/0412-1961/0412-1961-2020-6-909.shtml','external_locked','3 rows: quantitative composition in prose; Fig11 24h; Table3 explicit multiplier; passivated domain'),
        ('AH36_Ce25','10.13228/j.boyuan.issn1006-9356.20240777','','quarantined_unit_conflict','15 RE steel rows retained pending: printed Table3 A/cm2 contradicts curve scale; no correction guessed'),
        ('HRB500E24','10.1038/s41598-024-67799-9','','quarantined_unit_conflict','960h PDP link verified; Fig6 mA/cm2 conflicts curve scale; C2 text/conclusion 3.35 vs6.350; excluded'),
        ('X65_FSW21','10.11902/1005.4537.2019.237','https://www.jcscp.org/article/2021/1005-4537/1005-4537-2021-41-1-87.shtml','excluded_composition','2/4/6/8h PDP table available but no quantitative steel composition; dissimilar FSW joint cannot inherit grade composition'),
        ('MnNiMo24','10.3390/coatings14050588','','excluded_scope','Bare substrate target only at0d; 14/28d targets coated; grade composition ranges'),
        ('Super13Cr21','10.3389/fmats.2021.802136','','excluded_target','Mass-loss corrosion rate only; no PDP icorr'),
        ('MicroAlloy25','10.1038/s41529-025-00680-w','','excluded_target','LPR/Wagner-Traud current, not PDP/Tafel target'),
        ('DSS_LST','10.1002/adem.202502108','','excluded_composition_target','Standard grade chemistry with upper bounds; no Tafel region; cathodic intersection extrapolation'),
        ('Montmorillonite23','10.3390/ma16186291','','excluded_composition','No quantitative steel chemistry; blank medium .5M vs1M HCl conflict')]
    for ref,doi,url,status,reason in decisions:
        registry=registry.loc[~registry.doi.eq(doi)]
        registry=pd.concat([registry,pd.DataFrame([dict(reference_id=ref,doi=doi,source_url=url or f'https://doi.org/{doi}',review_status=status,decision_reason=reason)])],ignore_index=True)
    registry.to_csv(ROOT/'data/review/time_literature_review.csv',index=False,encoding='utf-8-sig')

if __name__=='__main__': save_audit()
