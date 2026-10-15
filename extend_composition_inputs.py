"""Training-only forward addition of measured trace composition columns."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import GroupKFold,StratifiedKFold
import semantic_features as sf
from optimize_train_test import metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/composition_extension'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data/training_RE_NaCl_REexpanded.csv')
    b=joblib.load(ROOT/'results/composition_inputs/model_train_only.joblib')
    nums,cats=next(iter(b['feature_sets'].values()))
    candidates=['N_wt_pct','Al_wt_pct','Nb_wt_pct','V_wt_pct','Ti_wt_pct','Co_wt_pct','O_wt_pct','La_wt_pct','Ce_wt_pct']
    x=sf.prepare(d);y=np.log10(d.icorr_A_cm2.to_numpy());re=d.RE_wt_pct.gt(0).to_numpy()
    train=np.flatnonzero(d.record_id.isin(b['train_ids']));test=np.flatnonzero(d.record_id.isin(b['test_ids']))
    assert len(train)==116 and len(test)==28 and not set(train)&set(test)
    for c in nums+candidates:x[c]=pd.to_numeric(x[c],errors='coerce')
    for c in cats:x[c]=x[c].replace(['unknown','not_reported','unreported',''],np.nan)
    folds=list(GroupKFold(4).split(train,groups=d.reference_id.iloc[train]))
    def fit(idx,extra):
        n=[c for c in nums+extra if x.iloc[idx][c].nunique()>1]
        ca=[c for c in cats if x.iloc[idx][c].nunique()>1]
        prep=ColumnTransformer([
            ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),n),
            ('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='unreported')),
                ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),ca)])
        return Pipeline([('prep',prep),('model',clone(b['model'].named_steps['model']))]).fit(x.iloc[idx],y[idx])
    cache={};history=[]
    def cv(extra):
        key=tuple(extra)
        if key not in cache:
            pred=np.full(len(d),np.nan)
            for a,v in folds:
                assert not set(d.reference_id.iloc[train[a]])&set(d.reference_id.iloc[train[v]])
                pred[train[v]]=fit(train[a],extra).predict(x.iloc[train[v]])
            cache[key]=(metrics(y[train][re[train]],pred[train][re[train]]),pred)
        return cache[key][0]
    extra=[];baseline=cv([]);best=baseline
    # Prespecified cap: at most three additions; no test data used by this search.
    for step in range(1,4):
        rankings=[]
        for c in candidates:
            if c in extra:continue
            score=cv(extra+[c]);rankings.append((score['log10_R2'],c,score))
            history.append(dict(step=step,extra=extra+[c],**score))
        score,c,result=max(rankings,key=lambda r:r[0])
        if score<=best['log10_R2']+1e-6:break
        extra.append(c);best=result
    # Selection frozen before any test predictions are calculated.
    final=fit(train,extra);baseline_model=fit(train,[])
    p=final.predict(x);p0=baseline_model.predict(x)
    rowpred=np.full(len(d),np.nan)
    for a,v in StratifiedKFold(4,shuffle=True,random_state=17).split(train,re[train]):
        rowpred[train[v]]=fit(train[a],extra).predict(x.iloc[train[v]])
    result=dict(added_columns=extra,raw_input_count=len(nums+extra+cats),search_candidates=candidates,
        baseline_paper_cv_RE=baseline,selected_paper_cv_RE=best,
        selected_paper_cv_all=metrics(y[train],cache[tuple(extra)][1][train]),
        selected_row_cv_RE=metrics(y[train][re[train]],rowpred[train][re[train]]),
        selected_row_cv_all=metrics(y[train],rowpred[train]),
        training_all=metrics(y[train],p[train]),training_RE=metrics(y[train][re[train]],p[train][re[train]]),
        baseline_test_all=metrics(y[test],p0[test]),baseline_test_RE=metrics(y[test][re[test]],p0[test][re[test]]),
        selected_test_all=metrics(y[test],p[test]),selected_test_RE=metrics(y[test][re[test]],p[test][re[test]]),
        caveat='Exploratory forward selection using reused development paper CV; selected CV is optimistic, not independent nested validation. Historical test reused and only 6 RE rows. No test-based selection. Missing elements not zero-filled; observed La/Ce only, no inferred fractions. Data remain 144 rows and 196 columns.')
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(history).to_json(OUT/'search_history.json',orient='records',indent=2)
    pd.DataFrame(dict(record_id=d.record_id,reference_id=d.reference_id,partition=np.where(d.record_id.isin(b['test_ids']),'historical_test','train'),
        observed_log10=y,prediction_log10=p,paper_OOF_log10=cache[tuple(extra)][1],row_OOF_log10=rowpred)).to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    features={'trace_extended':(nums+extra,cats)}
    joblib.dump(dict(b,model=final,feature_sets=features,config=['trace_extended',*b['config'][1:]]),OUT/'model_train_only.joblib')
    joblib.dump(dict(b,model=fit(np.arange(len(d)),extra),feature_sets=features,config=['trace_extended',*b['config'][1:]],
        train_ids=d.record_id.tolist(),test_ids=[],caution='All-row experimental refit; reported test score is not independent for this model.'),OUT/'model_all144.joblib')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
