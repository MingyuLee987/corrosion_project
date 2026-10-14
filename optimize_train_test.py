"""Fixed random/paper holdouts; tune on training CV only, prioritize RE error."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import semantic_features as sf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/train_test_optimization'

def metrics(y,p):
    err=np.abs(y-p)
    return dict(n=len(y),log10_R2=float(r2_score(y,p)),log10_MAE=float(mean_absolute_error(y,p)),
        **{f'within_{k}x_pct':float((err<=np.log10(k)).mean()*100) for k in [2,3,5,10]})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'data/training_RE_NaCl_expanded.csv'
    d=pd.read_csv(path); x=sf.prepare(d); y=np.log10(d.icorr_A_cm2.to_numpy()); re=d.RE_wt_pct.gt(0).to_numpy()
    groups=d.reference_id.to_numpy(); indices=np.arange(len(d))
    features={'compact':sf.FEATURES['semantic_matrix'],
        'composition_process':(list(dict.fromkeys(sf.FEATURES['semantic_comp'][0]+[
            'O_wt_pct','W_wt_pct','NaCl_wt_pct','chloride_molar_verified','temperature_C','pH',
            'log_pre_exposure_verified','TP_scan_rate_mV_s','log_OCP_s',
            'final_heat_treatment_temp_C','final_heat_treatment_time_min','matrix_grain_size_um'])),
            ['RE_element_cat','matrix_family_reported','gas_condition_cat','surface_state_cat','processing_state_cat'])}
    for num,cat in features.values():
        for c in num:x[c]=pd.to_numeric(x[c],errors='coerce')
        for c in cat:x[c]=x[c].replace(['unknown','not_reported','unreported',''],np.nan)
    candidates=[]
    for C in [.1,1.,10.,100.]:
        for gamma in ['scale',.01,.1]:candidates.append(('SVR',dict(C=C,gamma=gamma,epsilon=.1)))
    for name in ['RF','ExtraTrees']:
        for leaf in [1,3,6]:candidates.append((name,dict(n_estimators=160,min_samples_leaf=leaf,max_features=1.,random_state=42,n_jobs=1)))
    for depth in [1,2,3]:
        for lr in [.03,.1]:candidates.append(('GBR',dict(n_estimators=120,max_depth=depth,learning_rate=lr,loss='huber',random_state=42)))
    for alpha in [1.,10.,100.]:candidates.append(('Ridge',dict(alpha=alpha)))
    grid=[(f,name,params) for f in features for name,params in candidates]
    def fit(train,config):
        f,name,params=config; num,cat=features[f]
        num=[c for c in num if x.iloc[train][c].nunique()>1]
        cat=[c for c in cat if x.iloc[train][c].nunique()>1]
        prep=ColumnTransformer([
            ('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),
            ('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='unreported')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
        est={'SVR':SVR,'RF':RandomForestRegressor,'ExtraTrees':ExtraTreesRegressor,'GBR':GradientBoostingRegressor,'Ridge':Ridge}[name](**params)
        pipe=Pipeline([('prep',prep),('model',est)])
        # Tune without weighting so independent held-out performance is the selection signal.
        return pipe.fit(x.iloc[train],y[train])
    train,test=train_test_split(indices,test_size=.2,random_state=42,stratify=re)
    gt,ge=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(x,y,groups))
    results={}
    for split,train,test in [('random_rows',train,test),('held_out_papers',gt,ge)]:
        folder=OUT/split;folder.mkdir(exist_ok=True)
        if split=='random_rows':
            folds=[(train[a],train[b]) for a,b in StratifiedKFold(n_splits=4,shuffle=True,random_state=17).split(train,re[train])]
        else:
            assert not set(groups[train])&set(groups[test])
            folds=[(train[a],train[b]) for a,b in GroupKFold(n_splits=4).split(train,groups=groups[train])]
        rankings=[]
        for k,config in enumerate(grid):
            pred=np.full(len(d),np.nan)
            for it,iv in folds:pred[iv]=fit(it,config).predict(x.iloc[iv])
            ri=train[re[train]]
            scores=metrics(y[ri],pred[ri])
            rankings.append(dict(candidate=k,feature=config[0],model=config[1],params=json.dumps(config[2]),**scores))
        ranking=pd.DataFrame(rankings).sort_values(['log10_R2','log10_MAE'],ascending=[False,True])
        ranking.to_csv(folder/'training_cv_candidates.csv',index=False,encoding='utf-8-sig')
        best=grid[int(ranking.iloc[0].candidate)]
        # Test is accessed only after the training-only configuration has been frozen.
        model=fit(train,best); predtrain=model.predict(x.iloc[train]); predtest=model.predict(x.iloc[test])
        result=dict(split=split,seed=42,train_rows=len(train),test_rows=len(test),
            test_RE_papers=int(pd.Series(groups[test][re[test]]).nunique()),selected_config=best,
            training_cv_RE=ranking.iloc[0].to_dict(),
            training_all=metrics(y[train],predtrain),test_all=metrics(y[test],predtest),
            training_RE=metrics(y[train][re[train]],predtrain[re[train]]),test_RE=metrics(y[test][re[test]],predtest[re[test]]),
            caveat='One fixed split; small RE holdout. Historical dataset already explored. Not prospective external validation.',
            input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        (folder/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        preds=np.full(len(d),np.nan);preds[train]=predtrain;preds[test]=predtest
        pd.DataFrame(dict(record_id=d.record_id,reference_id=groups,RE_positive=re,
            partition=np.where(np.isin(indices,test),'test','train'),observed_log10=y,prediction_log10=preds)).to_csv(folder/'predictions.csv',index=False,encoding='utf-8-sig')
        bundle=dict(model=model,config=best,feature_sets=features,input_columns=list(x.columns),
            preprocessing='semantic_only',semantic_transform=True,
            train_ids=d.record_id.iloc[train].tolist(),test_ids=d.record_id.iloc[test].tolist())
        joblib.dump(bundle,folder/'model_train_only.joblib')
        if split=='random_rows':
            full_bundle=dict(bundle,model=fit(indices,best),train_ids=d.record_id.tolist(),test_ids=[],
                caution='All-row deployment refit. Reported holdout score belongs to model_train_only, not this model.')
            joblib.dump(full_bundle,folder/'model_all139.joblib')
        results[split]=result
        print(json.dumps(result,indent=2),flush=True)
    (OUT/'comparison.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

if __name__=='__main__':main()
