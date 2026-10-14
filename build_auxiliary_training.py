"""Integrate source-verified ordinary-steel PDP rows without changing RE targets."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
    base_path=ROOT/'data/training_RE_collection_latest.csv'
    auxiliary_path=ROOT/'data/process_literature_verified.csv'
    base=pd.read_csv(base_path);aux=pd.read_csv(auxiliary_path)
    assert not set(base.record_id)&set(aux.record_id)
    assert not set(base.reference_id)&set(aux.reference_id)
    assert aux.icorr_A_cm2.gt(0).all() and aux.RE_wt_pct.eq(0).all()
    assert aux.source_url.notna().all() and aux.target_locator.notna().all()
    for col in ['OCP_stabilization_s','TP_OCP_stabilization_time_s']:
        if col not in base:base[col]=np.nan
    # This field is explicitly PDP open-circuit stabilization, not immersion time.
    mins=pd.to_numeric(base.get('TP_OCP_stabilization_min'),errors='coerce')*60
    base['OCP_stabilization_s']=pd.to_numeric(base.OCP_stabilization_s,errors='coerce').combine_first(mins)
    base['training_role']='existing_RE_study_or_auxiliary'
    aux['training_role']='additional_ordinary_steel_auxiliary'
    combined=pd.concat([base,aux],ignore_index=True)
    assert combined.record_id.is_unique and np.isfinite(combined.icorr_A_cm2).all()
    combined.to_csv(ROOT/'data/training_RE_with_auxiliary.csv',index=False,encoding='utf-8-sig')
    out=ROOT/'results/RE_auxiliary_expanded';out.mkdir(exist_ok=True)
    manifest={'base_rows':len(base),'additional_auxiliary_rows':len(aux),'combined_rows':len(combined),
        'RE_positive_rows':int(combined.RE_wt_pct.gt(0).sum()),'additional_papers':aux.groupby('reference_id').size().to_dict(),
        'base_sha256':hashlib.sha256(base_path.read_bytes()).hexdigest(),'auxiliary_sha256':hashlib.sha256(auxiliary_path.read_bytes()).hexdigest(),
        'note':'66 previously curated rows newly integrated into this RE model; not 66 newly discovered RE rows. No intentional RE addition is reported, not an assayed residual RE zero.'}
    (out/'data_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__':main()
