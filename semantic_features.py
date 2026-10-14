"""Only pre-test, unit-aware predictors; original measurements are retained."""
import numpy as np
import pandas as pd

def prepare(data):
 x=data.copy()
 def num(c):return pd.to_numeric(x[c],errors='coerce') if c in x else pd.Series(np.nan,index=x.index)
 # Exact molarity aliases first. No wt%/vol% conversion without density.
 chloride=num('Cl_mol_L')
 basis=x.get('Cl_conversion_basis',pd.Series('',index=x.index)).fillna('')
 chloride=chloride.mask(basis.str.contains('근사|미검증|approx|unverified',case=False,regex=True))
 x['chloride_molar_verified']=chloride.combine_first(num('NaCl_mol_L'))
 x['log_pre_exposure_verified']=np.log1p(num('pre_exposure_h'))
 clocks=[num('OCP_stabilization_s'),num('TP_OCP_stabilization_time_s'),num('TP_OCP_stabilization_min')*60]
 ocp=clocks[0]
 for clock in clocks[1:]:
  both=ocp.notna()&clock.notna()
  if not np.isclose(ocp[both],clock[both]).all():raise ValueError('Conflicting OCP stabilization clocks')
  ocp=ocp.combine_first(clock)
 x['OCP_stabilization_s']=ocp
 x['log_OCP_s']=np.log1p(ocp)
 cr=num('Cr_wt_pct')
 x['high_Cr_indicator']=np.where(cr.notna(),(cr>=10.5).astype(float),np.nan)
 x['RE_log1p_ppm']=np.log1p(num('RE_wt_pct')*10000)
 x['RE_highCr_interaction']=x.RE_log1p_ppm*x.high_Cr_indicator
 phase=x.get('phase_reported',pd.Series('',index=x.index)).fillna('').str.lower()
 # Broad reported matrix types, not paper-specific lists or rust-product phases.
 x['matrix_family_reported']=np.select([
  phase.str.contains('ferrite')&phase.str.contains('austenite'),phase.str.contains('austenite'),
  phase.str.contains('martensite'),phase.str.contains('ferrite')],
  ['ferrite_austenite','austenite','martensite','ferrite'],default='unreported')
 return x

FEATURES={
 'semantic_comp':(['C_wt_pct','Si_wt_pct','Mn_wt_pct','S_wt_pct','Cr_wt_pct','Ni_wt_pct','Mo_wt_pct','Cu_wt_pct','RE_log1p_ppm','high_Cr_indicator','RE_highCr_interaction'],['RE_element_cat','electrolyte_family']),
 'semantic_env':(['C_wt_pct','Cr_wt_pct','Ni_wt_pct','Mo_wt_pct','RE_log1p_ppm','high_Cr_indicator','RE_highCr_interaction','chloride_molar_verified','H2SO4_mol_L','NaHSO3_mol_L','pH','temperature_C','log_pre_exposure_verified'],['RE_element_cat','electrolyte_family','gas_condition_cat','surface_state_cat']),
 'semantic_matrix':(['C_wt_pct','Cr_wt_pct','Ni_wt_pct','Mo_wt_pct','RE_log1p_ppm','high_Cr_indicator','chloride_molar_verified','temperature_C','log_pre_exposure_verified','TP_scan_rate_mV_s','log_OCP_s'],['RE_element_cat','electrolyte_family','matrix_family_reported','gas_condition_cat'])}
