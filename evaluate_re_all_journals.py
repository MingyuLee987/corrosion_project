"""Frozen first-external check and nested publication-held-out RE evaluation."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import optimize_process_features as o

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/RE_all_journals'
CONFIGS=[(f,m,w) for f in ['core8','drop_P_S_add_manufacturing','drop_P_S','add_protocol']
         for m in ['svr03','svr1'] for w in [.25,1.]]
FROZEN=[('core8','svr03',1.),('drop_P_S_add_manufacturing','svr03',1.)]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data/training_RE_all_journals.csv'
    d=pd.read_csv(path);x=o.transform(d);y=np.log10(d.icorr_A_cm2.to_numpy(float));g=d.reference_id.to_numpy()
    old_papers=np.unique(g[:50]);new_papers=np.unique(g[314:]);papers=np.union1d(old_papers,new_papers)
    positive=d.RE_wt_pct.gt(0).to_numpy()
    def predict(train,test,c):
        return o.train_predict(x,y,g,papers,train,test,c[0],c[2],[c[1]])[c[1]]
    records=[];summaries=[]
    # Freeze both settings from the earlier run; new papers never enter training.
    new=np.flatnonzero(positive & np.isin(g,new_papers));train=np.arange(314)
    for c in FROZEN:
        p=predict(train,new,c)
        summaries.append(dict(evaluation='first_external_new_papers',feature=c[0],model=c[1],weight=c[2],rows=len(new),**o.metrics(y[new],p,g[new])))
        for i,v in zip(new,p):records.append(dict(evaluation='first_external_new_papers',record_id=d.at[i,'record_id'],reference_id=g[i],feature=c[0],model=c[1],weight=c[2],observed_log10=y[i],predicted_log10=v))
    fixed={c:np.full(len(d),np.nan) for c in CONFIGS};nested=np.full(len(d),np.nan);selections=[]
    for held in papers:
        test=np.flatnonzero(positive&(g==held));train=np.flatnonzero(g!=held)
        inner_idx=np.flatnonzero(positive & np.isin(g,papers)&(g!=held));scores={}
        for c in CONFIGS:
            fixed[c][test]=predict(train,test,c)
            ip=np.full(len(d),np.nan)
            for ih in papers[papers!=held]:
                it=np.flatnonzero(positive&(g==ih));tr=np.flatnonzero((g!=held)&(g!=ih))
                ip[it]=predict(tr,it,c)
            scores[c]=o.metrics(y[inner_idx],ip[inner_idx],g[inner_idx])
        best=min(CONFIGS,key=lambda c:(-scores[c]['log10_R2'],scores[c]['paper_balanced_log10_MAE']))
        nested[test]=fixed[best][test];selections.append(dict(held=held,feature=best[0],model=best[1],weight=best[2]))
        print(held,best,flush=True)
    for c,p in fixed.items():
        for label,mask in [('historical33',positive & (np.arange(len(d))<314)),('all_RE41',positive)]:
            idx=np.flatnonzero(mask)
            summaries.append(dict(evaluation=label,feature=c[0],model=c[1],weight=c[2],rows=len(idx),**o.metrics(y[idx],p[idx],g[idx])))
        for i in np.flatnonzero(positive):records.append(dict(evaluation='fixed_publication_heldout',record_id=d.at[i,'record_id'],reference_id=g[i],feature=c[0],model=c[1],weight=c[2],observed_log10=y[i],predicted_log10=p[i]))
    idx=np.flatnonzero(positive)
    summaries.append(dict(evaluation='nested_all_RE41',feature='inner_selected',model='SVR_only',weight=np.nan,rows=len(idx),**o.metrics(y[idx],nested[idx],g[idx])))
    for i in idx:records.append(dict(evaluation='nested_publication_heldout',record_id=d.at[i,'record_id'],reference_id=g[i],feature='inner_selected',model='SVR_only',weight=np.nan,observed_log10=y[i],predicted_log10=nested[i]))
    pd.DataFrame(summaries).to_csv(OUT/'scores.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    (OUT/'validation.json').write_text(json.dumps(dict(input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),target='log10(icorr_A_cm2)',new_rows=12,new_RE_positive=8,total_RE_positive=41,papers=13,journal_quartile_filter='none',selection=selections,limitations=['New Y source text-only verification; visual PDF QA pending','Frozen candidate was selected after prior historical feature exploration','SVR-only search restriction follows prior Ridge failure; this protocol is exploratory','New external test is 8 rows from 2 papers, not a broad deployment guarantee']),indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries).to_string(index=False))

if __name__=='__main__':main()
