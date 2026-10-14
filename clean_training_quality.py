"""Quarantine source-inconsistent auxiliary targets; never filter by model errors."""
from pathlib import Path
import json,hashlib
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
 source=ROOT/'data/training_RE_auxiliary_460.csv';d=pd.read_csv(source)
 ids=['AUX_NanofluidSteel2024_NaCl_CO2_CuO_55','AUX_NanofluidSteel2024_NaCl_CO2_CuO_T_55']
 assert d.record_id.is_unique and d.icorr_A_cm2.gt(0).all()
 mask=d.record_id.isin(ids);assert mask.sum()==2
 hold=d.loc[mask].copy();hold['quarantine_reason']='Table6 reported current and corrosion-rate conversion inconsistent with same-composition rows; target unresolved; not excluded by prediction error'
 clean=d.loc[~mask].copy()
 assert set(clean.loc[clean.RE_wt_pct.gt(0),'record_id'])==set(d.loc[d.RE_wt_pct.gt(0),'record_id'])
 clean.to_csv(ROOT/'data/training_RE_clean.csv',index=False,encoding='utf-8-sig')
 hold.to_csv(ROOT/'data/targets_quarantined.csv',index=False,encoding='utf-8-sig')
 out=ROOT/'results/quality_cleanup';out.mkdir(exist_ok=True)
 (out/'decisions.json').write_text(json.dumps(dict(original_rows=len(d),clean_rows=len(clean),quarantined_ids=ids,
  RE_evaluation_rows=int(clean.RE_wt_pct.gt(0).sum()),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
  invalid_nonpositive_targets=0,duplicate_record_ids=0,rule='Source inconsistency only. No filtering by residual, R2 contribution, or held-out predictions.',
  source_url='https://onlinelibrary.wiley.com/doi/full/10.1155/ijco/5531986',source_locator='Table6; comparison of icorr and corrosion rate across same alloy'),indent=2),encoding='utf-8')
 print('460 ->458 active rows;2 quarantined; all47 RE evaluation rows retained')

if __name__=='__main__':main()
