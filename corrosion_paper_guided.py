"""Paper-inspired tuning within source-held-out folds; no synthetic targets."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error
from corrosion_time_verified import NUM, CAT, model, MAX_MISSING_FRACTION

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/time_verified/paper_guided'
optuna.logging.set_verbosity(optuna.logging.WARNING)

def estimator(params):
    p=params.copy(); family=p.pop('family')
    if family=='ridge': return Ridge(**p)
    if family=='svr': return SVR(**p)
    if family=='rf': return RandomForestRegressor(n_estimators=150,random_state=42,n_jobs=1,**p)
    return CatBoostRegressor(iterations=150,loss_function='RMSE',verbose=False,
        random_seed=42,thread_count=1,allow_writing_files=False,**p)

def fit_predict(x,y,g,tr,va,params):
    selected=[c for c in x if x.iloc[tr][c].isna().mean()<=MAX_MISSING_FRACTION][:40]
    pipe=model('ridge10',[c for c in selected if c not in CAT],[c for c in selected if c in CAT])
    pipe.named_steps['prep'].transformers[0][1].steps[-1]=('scale',RobustScaler())
    pipe.steps[-1]=('reg',estimator(params))
    counts=pd.Series(g[tr]).value_counts()
    weights=np.array([1/counts[a] for a in g[tr]]);weights/=weights.mean()
    pipe.fit(x.iloc[tr],y[tr],reg__sample_weight=weights)
    return pipe.predict(x.iloc[va]),pipe.predict(x.iloc[tr]),selected

def suggest(trial):
    family=trial.suggest_categorical('family',['ridge','svr','rf','catboost'])
    if family=='ridge': trial.suggest_float('alpha',.1,100,log=True)
    elif family=='svr':
        trial.suggest_float('C',.1,20,log=True)
        trial.suggest_float('epsilon',.05,.5)
        trial.suggest_categorical('gamma',['scale','auto'])
    elif family=='rf':
        trial.suggest_int('max_depth',2,4)
        trial.suggest_int('min_samples_leaf',2,4)
        trial.suggest_float('max_features',.5,1)
    else:
        trial.suggest_int('depth',2,4)
        trial.suggest_float('learning_rate',.01,.1,log=True)
        trial.suggest_float('l2_leaf_reg',3,30,log=True)
    return trial.params

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data=pd.read_csv(ROOT/'data/time/training_time_verified.csv',encoding='utf-8-sig')
    assert data.target_method.eq('potentiodynamic_Tafel').all()
    assert data.immersion_time_h.gt(0).all()
    numeric=NUM+['log1p_immersion_h','pre_exposure_temperature_C','pre_exposure_pressure_MPa','OCP_stabilization_s']
    x=data.reindex(columns=numeric+CAT).replace(['unknown','not_reported','unreported',''],np.nan)
    x['log1p_immersion_h']=np.log1p(data.immersion_time_h)
    for c in numeric: x[c]=pd.to_numeric(x[c],errors='coerce')
    x['RE_wt_pct']=np.log1p(x.RE_wt_pct*10000)
    x['material_class']=x.material_class.str.replace('low alloy steel','low-alloy steel',regex=False)
    y=np.log10(data.icorr_A_cm2.to_numpy());g=data.reference_id.to_numpy()
    papers=np.unique(g);pred=np.empty(len(y));folds=[]
    for held in papers:
        outer=np.flatnonzero(g!=held);test=np.flatnonzero(g==held)
        def objective(trial):
            params=suggest(trial);inner_pred=np.empty(len(outer));train_scores=[]
            for inner in papers[papers!=held]:
                tr=np.flatnonzero((g!=held)&(g!=inner));va=np.flatnonzero(g==inner)
                pv,pt,_=fit_predict(x,y,g,tr,va,params)
                inner_pred[np.isin(outer,va)]=pv
                train_scores.append(r2_score(y[tr],pt))
            cv=r2_score(y[outer],inner_pred)
            return float(cv-.5*max(0,np.mean(train_scores)-cv-.15))
        study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
        study.enqueue_trial({'family':'ridge','alpha':10})
        study.enqueue_trial({'family':'svr','C':.3,'epsilon':.1,'gamma':'scale'})
        study.enqueue_trial({'family':'rf','max_depth':2,'min_samples_leaf':3,'max_features':.7})
        study.enqueue_trial({'family':'catboost','depth':2,'learning_rate':.03,'l2_leaf_reg':10})
        study.optimize(objective,n_trials=24)
        pv,_,cols=fit_predict(x,y,g,outer,test,study.best_params);pred[test]=pv
        folds.append(dict(held_out_paper=held,parameters=study.best_params,inner_objective=study.best_value,
            input_columns=cols,input_missing_fraction={c:float(x.iloc[outer][c].isna().mean()) for c in cols},
            log10_MAE=float(mean_absolute_error(y[test],pv))))
        pd.DataFrame([t.params|{'objective':t.value} for t in study.trials]).to_csv(OUT/f'trials_{held}.csv',index=False)
        print(held,study.best_params,flush=True)
    baseline=json.loads((ROOT/'results/time_verified/benchmark.json').read_text(encoding='utf-8'))['with_time']['metrics']
    metrics=dict(log10_R2=float(r2_score(y,pred)),log10_MAE=float(mean_absolute_error(y,pred)),
        paper_balanced_log10_MAE=float(np.mean([mean_absolute_error(y[g==p],pred[g==p]) for p in papers])))
    result=dict(rows=len(data),papers=len(papers),baseline=baseline,metrics=metrics,
        delta_log10_R2=metrics['log10_R2']-baseline['log10_R2'],folds=folds,
        validation='Nested leave-one-publication-out; 24 TPE trials per outer fold; inner pooled R2 minus train/CV gap penalty; train-only preprocessing and <=20% missing feature selection',
        reference='10.1038/s41529-026-00859-9',GAN_used=False,
        status='Exploratory comparison; outer scores observed during development; requires new publications for confirmation')
    (OUT/'benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    records=data[['record_id','reference_id','immersion_time_h','icorr_A_cm2']].copy()
    records['observed_log10']=y;records['predicted_log10']=pred
    records.to_csv(OUT/'heldout_predictions.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(metrics,indent=2))

if __name__=='__main__': main()
