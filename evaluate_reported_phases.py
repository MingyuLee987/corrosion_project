from pathlib import Path
import numpy as np
import pandas as pd
import optimize_process_features as o

ROOT=Path(__file__).resolve().parent
def main():
    d=pd.read_csv(ROOT/'data/training_RE_all_journals.csv')
    base='drop_P_S_add_manufacturing'
    numeric,categorical=o.FEATURE_SETS[base]
    o.FEATURE_SETS['manufacturing_plus_phase']=(numeric,categorical+['phase_reported','microstructure_constituents_reported'])
    o.FEATURE_SETS['manufacturing_plus_phase_grain']=(numeric+['matrix_grain_size_um','polygonal_ferrite_fraction_pct'],categorical+['phase_reported','microstructure_constituents_reported'])
    x=o.transform(d);y=np.log10(d.icorr_A_cm2.to_numpy(float));g=d.reference_id.to_numpy()
    papers=np.union1d(g[:50],g[314:]);positive=d.RE_wt_pct.gt(0).to_numpy();idx=np.flatnonzero(positive)
    scores=[];records=[]
    for feature in [base,'manufacturing_plus_phase','manufacturing_plus_phase_grain']:
        pred=np.full(len(d),np.nan)
        for held in papers:
            train=np.flatnonzero(g!=held);test=np.flatnonzero((g==held)&positive)
            pred[test]=o.train_predict(x,y,g,papers,train,test,feature,.25,['svr1'])['svr1']
        scores.append(dict(feature=feature,**o.metrics(y[idx],pred[idx],g[idx])))
        for i in idx:records.append(dict(feature=feature,record_id=d.at[i,'record_id'],reference_id=g[i],observed_log10=y[i],predicted_log10=pred[i]))
    out=ROOT/'results/phase_analysis'
    pd.DataFrame(scores).to_csv(out/'feature_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(scores).to_string(index=False))
if __name__=='__main__':main()
