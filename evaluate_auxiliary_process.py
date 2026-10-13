"""Matched 41-row RE test population; add previously verified process papers."""
from pathlib import Path
import pandas as pd
import numpy as np
import optimize_process_features as o
ROOT=Path(__file__).resolve().parent

def main():
    base=pd.read_csv(ROOT/'data/training_RE_all_journals.csv')
    aux=pd.read_csv(ROOT/'data/process_literature_verified.csv')
    assert not set(base.reference_id)&set(aux.reference_id)
    papers=np.union1d(base.reference_id.iloc[:50].unique(),base.reference_id.iloc[314:].unique())
    rows=[];predictions=[]
    for label,d in [('326_rows',base),('392_rows_add_process66',pd.concat([base,aux],ignore_index=True))]:
        x=o.transform(d);y=np.log10(d.icorr_A_cm2.to_numpy(float));g=d.reference_id.to_numpy()
        positive=d.RE_wt_pct.gt(0).to_numpy();idx=np.flatnonzero(positive)
        for feature in ['core8','drop_P_S_add_manufacturing']:
            for model in ['svr03','svr1']:
                for weight in [.25,1.]:
                    p=np.full(len(d),np.nan)
                    for held in papers:
                        tr=np.flatnonzero(g!=held);te=np.flatnonzero((g==held)&positive)
                        p[te]=o.train_predict(x,y,g,papers,tr,te,feature,weight,[model])[model]
                    rows.append(dict(dataset=label,feature=feature,model=model,auxiliary_weight=weight,rows=len(idx),**o.metrics(y[idx],p[idx],g[idx])))
                    for i in idx:predictions.append(dict(dataset=label,feature=feature,model=model,auxiliary_weight=weight,record_id=d.at[i,'record_id'],reference_id=g[i],observed_log10=y[i],predicted_log10=p[i]))
    out=ROOT/'results/RE_all_journals'
    pd.DataFrame(rows).to_csv(out/'auxiliary_process_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(predictions).to_csv(out/'auxiliary_process_predictions.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(rows).sort_values('log10_R2',ascending=False).to_string(index=False))
if __name__=='__main__':main()
