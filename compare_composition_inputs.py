"""Compare composition additions with fixed estimator and historical holdout."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupKFold
import semantic_features as sf
from optimize_train_test import metrics

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/composition_inputs'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'data/training_RE_NaCl_REexpanded.csv')
    b=joblib.load(ROOT/'results/RE_NaCl_new_literature/model_train_only.joblib')
    x=sf.prepare(d); y=np.log10(d.icorr_A_cm2.to_numpy()); re=d.RE_wt_pct.gt(0).to_numpy()
    train=np.flatnonzero(d.record_id.isin(b['train_ids'])); test=np.flatnonzero(d.record_id.isin(b['test_ids']))
    nums,cats=sf.FEATURES['semantic_matrix']; cats=[c for c in cats if c!='electrolyte_family']
    sets={'current':(nums,cats),'main_composition':(nums+['Si_wt_pct','Mn_wt_pct','P_wt_pct','S_wt_pct','Cu_wt_pct'],cats),
        'extended_composition':(nums+['Si_wt_pct','Mn_wt_pct','P_wt_pct','S_wt_pct','Cu_wt_pct',
            'N_wt_pct','Al_wt_pct','Nb_wt_pct','V_wt_pct','Ti_wt_pct','Co_wt_pct','O_wt_pct'],cats)}
    for n,c in sets.values():
        for col in n:x[col]=pd.to_numeric(x[col],errors='coerce')
        for col in c:x[col]=x[col].replace(['unknown','not_reported','unreported',''],np.nan)
    def fit(idx,key):
        n,c=sets[key]; n=[col for col in n if x.iloc[idx][col].nunique()>1]
        c=[col for col in c if x.iloc[idx][col].nunique()>1]
        prep=ColumnTransformer([
            ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),n),
            ('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='unreported')),
                ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),c)])
        return Pipeline([('prep',prep),('model',clone(b['model'].named_steps['model']))]).fit(x.iloc[idx],y[idx])
    folds=list(GroupKFold(4).split(train,groups=d.reference_id.iloc[train]))
    cv=[]
    for key in sets:
        pred=np.full(len(d),np.nan)
        for a,v in folds:pred[train[v]]=fit(train[a],key).predict(x.iloc[train[v]])
        ri=train[re[train]]
        cv.append(dict(feature_set=key,**metrics(y[ri],pred[ri])))
    selected=max(cv,key=lambda r:r['log10_R2'])['feature_set']
    # Select using training-paper CV before observing held-out predictions.
    results=[]
    for key in sets:
        model=fit(train,key); p=model.predict(x)
        results.append(dict(feature_set=key,selected_by_training_cv=key==selected,
            input_columns=[col for name,_,cols in model.named_steps['prep'].transformers_ if name!='remainder' for col in cols],
            training_all=metrics(y[train],p[train]),training_RE=metrics(y[train][re[train]],p[train][re[train]]),
            historical_test_all=metrics(y[test],p[test]),historical_test_RE=metrics(y[test][re[test]],p[test][re[test]])))
        pd.DataFrame(dict(record_id=d.record_id,partition=np.where(d.record_id.isin(b['test_ids']),'historical_test','train'),
            observed_log10=y,predicted_log10=p)).to_csv(OUT/f'{key}_predictions.csv',index=False,encoding='utf-8-sig')
        if key==selected:
            joblib.dump(dict(b,model=model,feature_sets={key:sets[key]},config=[key,*b['config'][1:]]),OUT/'model_train_only.joblib')
            joblib.dump(dict(b,model=fit(np.arange(len(d)),key),feature_sets={key:sets[key]},
                config=[key,*b['config'][1:]],train_ids=d.record_id.tolist(),test_ids=[],
                caution='All-row experimental refit; no independent holdout score applies.'),OUT/'model_all144.joblib')
    coverage=pd.DataFrame([dict(column=c,reported_rows=int(d[c].notna().sum()),total_rows=len(d),
        missing_pct=float(d[c].isna().mean()*100),training_reported_rows=int(d[c].iloc[train].notna().sum()))
        for c in d if c.endswith('_wt_pct')])
    coverage.to_csv(OUT/'composition_coverage.csv',index=False,encoding='utf-8-sig')
    report=dict(selected=selected,training_paper_cv=cv,results=results,training_rows=len(train),test_rows=len(test),
        caveat='Only composition inputs changed; fixed GBR parameters. Training-only paper CV used for selection. Historical random-row test reused, 6 RE rows; exploratory comparison, not independent external validation. Missing compositions are not zero-filled.')
    (OUT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
