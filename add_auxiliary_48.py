"""48 reported condition results from one primary paper; repeats not expanded."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
URL='https://onlinelibrary.wiley.com/doi/full/10.1155/ijco/5531986'
VALUES={
 'NaCl':{
 'as_received':[1.5347e-5,2.5683e-5,2.5491e-5,2.445e-5],
 'tap_water':[8.0799e-6,7.6291e-6,4.4398e-6,6.1892e-6],
 'CuO':[5.367e-6,3.54e-6,4.815e-6,8.298e-6],
 'Al2O3':[1.543e-5,6.5239e-6,7.658e-6,9.2515e-6],
 'Al2O3_T':[6.6492e-6,1.2605e-5,7.253e-6,8.2179e-6],
 'CuO_T':[2.936e-6,7.8241e-6,8.292e-6,3.257e-6]},
 'NaCl_CO2':{
 'as_received':[4.44e-5,9.456e-5,.000253,.000364],
 'tap_water':[5.846e-5,.000119,.000307,.000631],
 'CuO':[5.216e-5,9.06e-5,.000248,.0009355],
 'Al2O3':[5.462e-5,.000159,.000144,.000476],
 'Al2O3_T':[5.613e-5,.0001516,.0003767,.000604],
 'CuO_T':[5.67e-5,.000159,.000171,.000486]}}

def main():
 archive=ROOT/'sources/literature/NanofluidSteel2024_verified_tables.json'
 source=dict(source_url=URL,doi='10.1155/ijco/5531986',retrieval_date='2026-10-14',
  retrieval_method='primary publisher full text via web; Tables1,2,5,6 and Sec2.8',
  chemistry_wt_pct=dict(C=.432,Si=.175,Mn=.829,P=.0157,S=.0165),
  table_target_unit='A/cm2',temperatures_C=[25,35,45,55],tables=VALUES,
  caveat='48 condition combinations in one paper and one base chemistry, not 48 independent compositions. Cr/Mo/Ni dashes remain missing.')
 archive.write_text(json.dumps(source,indent=2),encoding='utf-8')
 sha=hashlib.sha256(archive.read_bytes()).hexdigest();rows=[]
 for family,groups in VALUES.items():
  for state,values in groups.items():
   for temp,current in zip([25,35,45,55],values):
    tempered=state.endswith('_T');raw=state=='as_received'
    rows.append(dict(record_id=f'AUX_NanofluidSteel2024_{family}_{state}_{temp}',reference_id='NanofluidSteel2024',
     sample_id=state,material_class='carbon steel',C_wt_pct=.432,Si_wt_pct=.175,Mn_wt_pct=.829,
     P_wt_pct=.0157,S_wt_pct=.0165,RE_wt_pct=0.,RE_element_cat='none',
     RE_content_status='no_intentional_RE_addition_reported_not_assayed',icorr_A_cm2=current,log_icorr=np.log10(current),
     electrolyte_family=family,NaCl_wt_pct=3.5,temperature_C=temp,pH=3.9 if family=='NaCl_CO2' else 6.5,
     gas_condition_cat='CO2_saturated' if family=='NaCl_CO2' else 'CO2_not_added',
     TP_scan_rate_mV_s=1.,OCP_stabilization_s=1800.,TP_reference_electrode_cat='SCE',
     processing_state_cat='as_received' if raw else ('quenched_tempered' if tempered else 'quenched'),
     austenitizing_temperature_C=np.nan if raw else 900.,austenitizing_time_min=np.nan if raw else 60.,
     final_heat_treatment_temp_C=np.nan if raw else (350. if tempered else 900.),
     final_heat_treatment_time_min=np.nan if raw else (30. if tempered else 60.),
     quench_medium_cat=np.nan if raw else state.removesuffix('_T'),
     cooling_method_cat='air' if tempered else np.nan,
     phase_reported='ferrite',phase_verification_status='source_reported',phase_evidence_method='XRD',
     phase_source_locator='Sec3.1 ferrite detected in all samples; additional constituents not assigned',
     source_url=URL,doi='10.1155/ijco/5531986',source_file=str(archive.relative_to(ROOT)),source_sha256=sha,
     target_locator='Table5' if family=='NaCl' else 'Table6',target_value_basis='reported_table',
     training_role='ordinary_steel_auxiliary_added48',strict300_eligible=False,
     quality_flags='OCP measured30min at25C before temperature-specific PDP; pH reported for solutions before/after CO2 bubbling, not separately at each test temperature. Table6 CuO55C and CuO_T55C current/corrosion-rate conversion inconsistent; reported current retained with no inferred correction.'
       if family=='NaCl_CO2' and temp==55 and state in ['CuO','CuO_T'] else 'Cr/Mo/Ni unreported; exact exposure time and assay method unreported. pH from solution preparation, not measured at each test temperature.'))
 new=pd.DataFrame(rows);base=pd.read_csv(ROOT/'data/training_RE_auxiliary_max.csv')
 assert len(base)==412 and len(new)==48
 assert not set(base.reference_id)&set(new.reference_id)
 d=pd.concat([base,new],ignore_index=True)
 assert d.record_id.is_unique and d.icorr_A_cm2.gt(0).all()
 new.to_csv(ROOT/'data/auxiliary_additions_48.csv',index=False,encoding='utf-8-sig')
 d.to_csv(ROOT/'data/training_RE_auxiliary_460.csv',index=False,encoding='utf-8-sig')
 print('Added48 ->',len(d),'rows; RE positive',d.RE_wt_pct.gt(0).sum())

if __name__=='__main__':main()
