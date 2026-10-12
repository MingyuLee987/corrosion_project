"""Fixed, low-missingness features for rare-earth steel studies and their controls."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score,mean_absolute_error
from corrosion_time_verified import model

ROOT=Path(__file__).resolve().parent
NUM=['C_wt_pct','Si_wt_pct','Mn_wt_pct','P_wt_pct','S_wt_pct','RE_log1p_ppm']
CAT=['RE_element_cat','electrolyte_family']
FEATURES=NUM+CAT

def load_core():
    d=pd.read_csv(ROOT/'data/training_51.csv',encoding='utf-8-sig')
    # Only publications studying RE additions, including their reported controls.
    refs=['Tang2019','MDPI26LaCe','Zhang2025','Coat24','Ma19FSS','Bridge25','UCS23','GCr23','Al10_26','La24Q355']
    d=d.loc[d.reference_id.isin(refs)].copy()
    d['RE_log1p_ppm']=np.log1p(d.RE_wt_pct*10000)
    d['RE_element_cat']=d.RE_element_cat.replace({'La_Ce':'La+Ce'})
    d['dataset_scope']='RE_addition_studies_and_reported_controls'
    d['time_usage']='auxiliary_only; unverified or different exposure clocks not imputed'
    d['composition_caveat']=d.get('composition_basis','')
    return d

def main():
    d=load_core()
    x=d[FEATURES].copy();y=np.log10(d.icorr_A_cm2.to_numpy());g=d.reference_id.to_numpy()
    assert y.size and np.isfinite(y).all()
    out=ROOT/'results/RE_core';out.mkdir(parents=True,exist_ok=True)
    candidates=['median','ridge10','svr03'];pred=np.empty(len(y));folds=[]
    def predict(name,tr,va):
        pipe=model(name,NUM,CAT)
        if name=='median':pipe.fit(x.iloc[tr],y[tr])
        else:
            counts=pd.Series(g[tr]).value_counts();w=np.array([1/counts[a] for a in g[tr]]);w/=w.mean()
            pipe.fit(x.iloc[tr],y[tr],reg__sample_weight=w)
        return pipe.predict(x.iloc[va])
    for tr,va in LeaveOneGroupOut().split(x,y,g):
        rates=x.iloc[tr].isna().mean();assert rates.max()<=.20
        scores={}
        for name in candidates:
            errors=[]
            for a,b in LeaveOneGroupOut().split(x.iloc[tr],y[tr],g[tr]):
                errors.append(mean_absolute_error(y[tr[b]],predict(name,tr[a],tr[b])))
            scores[name]=float(np.mean(errors))
        best=min(scores,key=scores.get);pred[va]=predict(best,tr,va)
        folds.append(dict(held_out_paper=str(g[va][0]),model=best,inner_paper_MAE=scores,training_missing_fraction=rates.to_dict()))
    result=dict(rows=len(d),papers=len(np.unique(g)),features=FEATURES,
        missing_fraction=x.isna().mean().to_dict(),log10_R2=float(r2_score(y,pred)),
        log10_MAE=float(mean_absolute_error(y,pred)),folds=folds,
        validation='Nested leave-one-publication-out; inner mean publication MAE; training-only imputation; fixed eight raw predictors',
        status='Exploratory: heterogeneous electrolytes/processes and unreported composition elements limit generalization; RE feature association is not causal effect',
        exposure_time_used=False,external_locked_data_used=False)
    (out/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    d[['record_id','reference_id','doi','icorr_A_cm2']].assign(observed_log10=y,predicted_log10=pred).to_csv(out/'heldout_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame({'feature':FEATURES,'missing_rows':x.isna().sum().values,'missing_fraction':x.isna().mean().values}).to_csv(out/'feature_missingness.csv',index=False)
    print(json.dumps({k:result[k] for k in ['rows','papers','features','missing_fraction','log10_R2','log10_MAE']},indent=2))
if __name__=='__main__':main()
