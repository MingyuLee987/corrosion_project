"""Phase-aware SVR search, nested paper validation, then fit every available row."""
from pathlib import Path
import json
import hashlib
import joblib
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import optimize_process_features as o

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/phase_tuning'

def main():
    global OUT
    advanced='--algorithms' in sys.argv
    if advanced:OUT=ROOT/'results/phase_algorithms_v2'
    OUT.mkdir(exist_ok=True)
    d=pd.read_csv(ROOT/'data/training_RE_all_journals.csv')
    n,c=o.FEATURE_SETS['drop_P_S_add_manufacturing']
    features={'phase':(n,c+['phase_reported','microstructure_constituents_reported']),
              'phase_grain':(n+['matrix_grain_size_um','polygonal_ferrite_fraction_pct'],c+['phase_reported','microstructure_constituents_reported'])}
    o.FEATURE_SETS.update(features);x=o.transform(d)
    y=np.log10(d.icorr_A_cm2.to_numpy(float));g=d.reference_id.to_numpy()
    papers=np.union1d(g[:50],g[314:]);positive=d.RE_wt_pct.gt(0).to_numpy();idx=np.flatnonzero(positive)
    configs=[(f,C,gamma,w) for f in features for C in [.1,.3,1.,3.] for gamma in ['scale',.01] for w in [.25,1.]]
    if advanced:
        configs=[(f,leaf,kind,.25) for f in features for leaf in [3,8] for kind in ['extra_trees','random_forest']] + [('phase',C,'scale',w) for C in [.1,.3] for w in [.25,1.]]
    def fit(tr,config):
        f,C,gamma,w=config;num,cat=features[f]
        num=[v for v in num if x.iloc[tr][v].nunique()>1];cat=[v for v in cat if x.iloc[tr][v].nunique()>1]
        prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),
            ('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='unreported')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
        est=ExtraTreesRegressor(n_estimators=64,min_samples_leaf=int(C),random_state=42,n_jobs=1) if gamma=='extra_trees' else (RandomForestRegressor(n_estimators=64,min_samples_leaf=int(C),random_state=42,n_jobs=1) if gamma=='random_forest' else SVR(C=C,gamma=gamma,epsilon=.1))
        model=Pipeline([('prep',prep),('svr',est)])
        counts=pd.Series(g[tr]).value_counts();weights=np.array([1/counts[v] for v in g[tr]])*np.where(np.isin(g[tr],papers),1.,w);weights/=weights.mean()
        model.fit(x.iloc[tr],y[tr],svr__sample_weight=weights);return model
    # Every outer/inner split excludes all rows of the held paper, including controls.
    fixed={conf:np.full(len(d),np.nan) for conf in configs}
    nested=np.full(len(d),np.nan);selection=[];cache={}
    def predict(excluded,conf,test):
        key=(tuple(sorted(excluded)),conf)
        if key not in cache:
            tr=np.flatnonzero(~np.isin(g,list(excluded)))
            cache[key]=fit(tr,conf).predict(x)
        return cache[key][test]
    for held in papers:
        test=np.flatnonzero(positive&(g==held));val=np.flatnonzero(positive&(g!=held));scores={}
        for conf in configs:
            fixed[conf][test]=predict({held},conf,test)
            p=np.full(len(d),np.nan)
            for ih in papers[papers!=held]:
                it=np.flatnonzero(positive&(g==ih));p[it]=predict({held,ih},conf,it)
            scores[conf]=o.metrics(y[val],p[val],g[val])
        best=min(configs,key=lambda v:(-scores[v]['log10_R2'],scores[v]['paper_balanced_log10_MAE']))
        nested[test]=fixed[best][test];selection.append(dict(held=held,config=best))
        print(held,best,flush=True)
    rows=[dict(feature=conf[0],C=conf[1],gamma=conf[2],auxiliary_weight=conf[3],**o.metrics(y[idx],fixed[conf][idx],g[idx])) for conf in configs]
    table=pd.DataFrame(rows).sort_values('log10_R2',ascending=False);table.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    best=max(configs,key=lambda conf:o.metrics(y[idx],fixed[conf][idx],g[idx])['log10_R2'])
    model=fit(np.arange(len(d)),best)
    joblib.dump(dict(model=model,config=best,input_columns=list(x.columns),preprocessing='optimize_process_features.transform plus phase feature declarations'),OUT/'model_all326.joblib')
    pred=model.predict(x)
    report=dict(rows_trained=len(d),RE_positive_rows=len(idx),search_configs=len(configs),best_exploratory_config=best,
        best_exploratory=o.metrics(y[idx],fixed[best][idx],g[idx]),nested=o.metrics(y[idx],nested[idx],g[idx]),
        training_all_rows=o.metrics(y,pred,g),training_RE_only=o.metrics(y[idx],pred[idx],g[idx]),
        selections=selection,input_sha256=hashlib.sha256((ROOT/'data/training_RE_all_journals.csv').read_bytes()).hexdigest(),
        caution='Training R2 is resubstitution, not generalization. Search is exploratory after previous experiments; nested splits reduce selection leakage but do not replace independent new-paper validation.')
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(dict(record_id=d.record_id,reference_id=g,observed_log10=y,nested_prediction=nested,best_fixed_prediction=fixed[best],all_rows_training_prediction=pred)).to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({k:v for k,v in report.items() if k!='selections'},indent=2),flush=True)
if __name__=='__main__':main()
