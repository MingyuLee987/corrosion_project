"""Tune on development papers only, then evaluate the locked publication once."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import r2_score,mean_absolute_error
from corrosion_paper_guided import suggest,fit_predict,NUM,CAT

ROOT=Path(__file__).resolve().parent
def main():
    dev=pd.read_csv(ROOT/'data/time/training_time_verified.csv',encoding='utf-8-sig')
    ext=pd.read_csv(ROOT/'data/time/external_time_verified.csv',encoding='utf-8-sig')
    assert set(dev.doi).isdisjoint(set(ext.doi))
    data=pd.concat([dev,ext],ignore_index=True)
    numeric=NUM+['log1p_immersion_h','pre_exposure_temperature_C','pre_exposure_pressure_MPa','OCP_stabilization_s']
    x=data.reindex(columns=numeric+CAT).replace(['unknown','not_reported','unreported',''],np.nan)
    x['log1p_immersion_h']=np.log1p(data.immersion_time_h)
    for c in numeric:x[c]=pd.to_numeric(x[c],errors='coerce')
    x['RE_wt_pct']=np.log1p(x.RE_wt_pct*10000)
    x['material_class']=x.material_class.str.replace('low alloy steel','low-alloy steel',regex=False)
    y=np.log10(data.icorr_A_cm2.to_numpy());g=data.reference_id.to_numpy()
    tr=np.arange(len(dev));va=np.arange(len(dev),len(data))
    def objective(trial):
        params=suggest(trial);p=np.empty(len(dev));train_scores=[]
        for paper in np.unique(g[tr]):
            a=tr[g[tr]!=paper];b=tr[g[tr]==paper]
            pv,pt,_=fit_predict(x,y,g,a,b,params);p[b]=pv
            train_scores.append(r2_score(y[a],pt))
        cv=r2_score(y[tr],p)
        return float(cv-.5*max(0,np.mean(train_scores)-cv-.15))
    study=optuna.create_study(direction='maximize',sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial({'family':'ridge','alpha':10})
    study.enqueue_trial({'family':'svr','C':.3,'epsilon':.1,'gamma':'scale'})
    study.optimize(objective,n_trials=24)
    pred,_,cols=fit_predict(x,y,g,tr,va,study.best_params)
    out=ROOT/'results/time_verified/external';out.mkdir(parents=True,exist_ok=True)
    result=dict(development_rows=len(dev),external_rows=len(ext),external_papers=ext.reference_id.unique().tolist(),
        parameters=study.best_params,input_columns=cols,log10_R2=float(r2_score(y[va],pred)),
        log10_MAE=float(mean_absolute_error(y[va],pred)),
        validation='External targets unused in selection, imputation, training or tuning; final parameters chosen by development publication CV only',
        limitation='One publication, three similar targets, chemically passivated stainless domain absent from development; R2 unstable and not a broad validation')
    (out/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    records=ext[['record_id','doi','icorr_A_cm2']].copy()
    records['predicted_icorr_A_cm2']=10**pred;records['observed_log10']=y[va];records['predicted_log10']=pred
    records.to_csv(out/'predictions.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
