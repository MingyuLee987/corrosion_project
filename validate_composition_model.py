"""Out-of-fold validation of the frozen 19-input model, excluding historical test."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GroupKFold
import semantic_features as sf
from optimize_train_test import metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/composition_inputs/cross_validation'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    b=joblib.load(ROOT/'results/composition_inputs/model_train_only.joblib')
    all_data=pd.read_csv(ROOT/'data/training_RE_NaCl_REexpanded.csv')
    d=all_data.loc[all_data.record_id.isin(b['train_ids'])].reset_index(drop=True)
    assert len(d)==116 and not set(d.record_id)&set(b['test_ids'])
    x=sf.prepare(d); y=np.log10(d.icorr_A_cm2.to_numpy()); re=d.RE_wt_pct.gt(0).to_numpy()
    for nums,cats in b['feature_sets'].values():
        for c in nums:x[c]=pd.to_numeric(x[c],errors='coerce')
        for c in cats:x[c]=x[c].replace(['unknown','not_reported','unreported',''],np.nan)
    splits={'random_rows':StratifiedKFold(4,shuffle=True,random_state=17).split(x,re),
        'held_out_papers':GroupKFold(4).split(x,groups=d.reference_id)}
    result={}
    for name,folds in splits.items():
        pred=np.full(len(d),np.nan); assignment=np.full(len(d),-1); scores=[]
        for k,(train,val) in enumerate(folds):
            assert not set(train)&set(val)
            if name=='held_out_papers':assert not set(d.reference_id.iloc[train])&set(d.reference_id.iloc[val])
            model=clone(b['model'])
            # Match feature-selection experiment: remove constants within each training fold.
            model.named_steps['prep'].transformers=[(n,t,[c for c in cols if x.iloc[train][c].nunique()>1])
                for n,t,cols in model.named_steps['prep'].transformers]
            model.fit(x.iloc[train],y[train])
            pred[val]=model.predict(x.iloc[val]);assignment[val]=k+1
            scores.append(dict(fold=k+1,training_rows=len(train),validation_rows=len(val),
                all=metrics(y[val],pred[val]),RE=metrics(y[val][re[val]],pred[val][re[val]])))
        assert np.isfinite(pred).all() and (assignment>0).all()
        result[name]=dict(pooled_all=metrics(y,pred),pooled_RE=metrics(y[re],pred[re]),folds=scores,
            mean_fold_R2_all=float(np.mean([s['all']['log10_R2'] for s in scores])),
            mean_fold_R2_RE=float(np.mean([s['RE']['log10_R2'] for s in scores])))
        pd.DataFrame(dict(record_id=d.record_id,reference_id=d.reference_id,RE_positive=re,
            validation_fold=assignment,observed_log10=y,predicted_log10=pred)).to_csv(OUT/f'{name}_OOF.csv',index=False,encoding='utf-8-sig')
    result['protocol']=dict(rows=len(d),RE_rows=int(re.sum()),folds=4,target='log10(icorr_A_cm2)',
        selection_caveat='19-input model previously selected using the same training-paper folds; these are development CV scores, not independent nested validation. Historical 28-row test excluded. No model/feature tuning in this run.',
        aggregation='Pooled R2 over all out-of-fold predictions; fold mean also stored separately.')
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:v for a,v in r.items() if a!='folds'} for k,r in result.items()},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
