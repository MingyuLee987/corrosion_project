from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
 rows=[]
 def add(ref,sample,current,chem,fields,file,doi,locator):
  p=ROOT/file
  rows.append(dict(record_id=f'AUX_{ref}_{sample}',reference_id=ref,sample_id=sample,RE_wt_pct=0.,RE_element_cat='none',
   RE_content_status='no_intentional_RE_addition_reported_not_assayed',icorr_A_cm2=current,log_icorr=np.log10(current),
   target_value_basis='reported_table',source_file=file,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),doi=doi,
   source_url='https://doi.org/'+doi,target_locator=locator,strict300_eligible=False,training_role='additional_ordinary_steel_auxiliary',**chem,**fields))
 chem=dict(C_wt_pct=.427,Si_wt_pct=.214,Mn_wt_pct=.846,P_wt_pct=.0056,S_wt_pct=.0173,Ni_wt_pct=.0407,
  Cr_wt_pct=1.07,Mo_wt_pct=.183,Cu_wt_pct=.167,Al_wt_pct=.0386,Ti_wt_pct=.0089,V_wt_pct=.0106)
 for sample,state,temp,cooling,current in [(1,'normalized',870,'air',41.72),(2,'soft_annealed',720,'air',52.72),
  (3,'spheroidized',735,'furnace',26.91),(4,'fully_annealed',820,'furnace_then_air',20.52)]:
  add('Annealed42CrMo4_2023',str(sample),current*1e-6,chem,dict(material_class='low alloy steel',electrolyte_family='NaCl',
   Cl_mol_L=.6,NaCl_mol_L=.6,temperature_C=20.,gas_condition_cat='naturally_aerated',OCP_stabilization_s=7200.,
   TP_scan_rate_mV_s=.166,TP_reference_electrode_cat='SCE',processing_state_cat=state,final_heat_treatment_temp_C=temp,
   final_heat_treatment_time_min=120.,cooling_method_cat=cooling,phase_reported='ferrite',microstructure_constituents_reported='ferrite+pearlite',
   phase_verification_status='source_reported',phase_evidence_method='OM_SEM_XRD',quality_flags='GDOES composition; furnace-to-air switch 670C for sample4. PDP temperature20+/-2C; reported scan0.166mV/s.'),
   'sources/Annealed42CrMo4_2023.pdf','10.17222/mit.2022.624','Tables1-3; Sec2.5; Sec3 Figures2-4')
 chem=dict(C_wt_pct=.03,Si_wt_pct=.8,Mn_wt_pct=.9,Cr_wt_pct=22.7,Ni_wt_pct=9.,Mo_wt_pct=3.2,N_wt_pct=.13)
 for sample,state,temp,time,current in [('AD','as_deposited',np.nan,np.nan,5.42e-8),('SS','solution_treated',1100.,np.nan,3.61e-7),('SST','solution_tempered',300.,120.,1.05e-7)]:
  add('WAAM2209_2024',sample,current,chem,dict(material_class='stainless steel',electrolyte_family='NaCl',NaCl_wt_pct=3.5,
   temperature_C=25.,TP_scan_rate_mV_s=10/60,TP_reference_electrode_cat='SCE',processing_state_cat=state,
   final_heat_treatment_temp_C=temp,final_heat_treatment_time_min=time,phase_reported='ferrite+austenite',
   microstructure_constituents_reported='ferrite+austenite',phase_verification_status='source_reported',phase_evidence_method='metallography_EBSD',
   quality_flags='Chemistry is commercial raw wire, not deposited assay. Solution hold1h in methods vs2h in abstract; conflicting hold left missing. Post-tempering cooling/OCP not reported.'),
   'sources/WAAM2209_2024.pdf','10.24425/amm.2024.149796','Sec2 wire composition/process; printed p672 Table2; Sec3.1 phases')
 for exposure,current in [(25,2.6),(95,4.2),(180,6.8),(230,7.5)]:
  add('AISI1010Sulfolane_2020',str(exposure),current*1e-9,dict(C_wt_pct=.1),dict(material_class='carbon steel',
   electrolyte_family='sulfolane',processing_state_cat='annealed',temperature_C=25.,pre_exposure_h=96.,
   pre_exposure_temperature_C=exposure,OCP_stabilization_s=3600.,TP_scan_rate_mV_s=10/60,TP_reference_electrode_cat='SCE',
   quality_flags='Only C reported; other chemistry missing. Exposure25/95/180/230C is distinct from PDP25C. Very different solvent; useful only as optional auxiliary.',
   target_method='polarization_Tafel_equation_fit'),
   'sources/literature/AISI1010Sulfolane_2020.md','10.3390/ma13112563','Sec2.1-2.2 and electrochemical parameters table jcorr nA/cm2')
 new=pd.DataFrame(rows)
 base=pd.read_csv(ROOT/'data/training_RE_with_auxiliary.csv')
 assert not set(new.record_id)&set(base.record_id)
 assert not set(new.reference_id)&set(base.reference_id)
 new.to_csv(ROOT/'data/ordinary_steel_additions_20261014.csv',index=False,encoding='utf-8-sig')
 d=pd.concat([base,new],ignore_index=True)
 d.to_csv(ROOT/'data/training_RE_auxiliary_max.csv',index=False,encoding='utf-8-sig')
 print('11 newly transcribed ordinary-steel results; total',len(d))

if __name__=='__main__':main()
