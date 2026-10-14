"""Translate saved nested RE predictions to explicit multiplicative error thresholds."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

ROOT=Path(__file__).resolve().parent
def main():
    path=ROOT/'results/phase_algorithms_v2/predictions.csv';d=pd.read_csv(path)
    d=d[d.nested_prediction.notna()].copy()
    d['absolute_log10_error']=np.abs(d.nested_prediction-d.observed_log10)
    d['error_factor']=10**d.absolute_log10_error
    rows=[]
    for label,col in [('nested_validation','nested_prediction'),('training_RE_only','all_rows_training_prediction')]:
        err=np.abs(d[col]-d.observed_log10)
        for factor in [2,3,5,10]:
            hit=err<=np.log10(factor)
            rows.append(dict(evaluation=label,factor=factor,hits=int(hit.sum()),rows=len(d),hit_rate_pct=100*hit.mean(),paper_balanced_hit_rate_pct=100*hit.groupby(d.reference_id).mean().mean()))
    papers=d.groupby('reference_id').agg(rows=('record_id','size'),log10_MAE=('absolute_log10_error','mean'),median_error_factor=('error_factor','median'),maximum_error_factor=('error_factor','max')).sort_values('log10_MAE',ascending=False)
    out=ROOT/'results/prediction_diagnostics';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'accuracy_thresholds.csv',index=False,encoding='utf-8-sig')
    papers.to_csv(out/'paper_errors.csv',encoding='utf-8-sig')
    d.to_csv(out/'row_errors.csv',index=False,encoding='utf-8-sig')
    summary=dict(rows=len(d),papers=d.reference_id.nunique(),nested_log10_R2=r2_score(d.observed_log10,d.nested_prediction),
        mean_log10_error=float(d.absolute_log10_error.mean()),geometric_mean_error_factor=float(10**d.absolute_log10_error.mean()),
        median_error_factor=float(d.error_factor.median()),underpredict_rows=int((d.nested_prediction<d.observed_log10).sum()),
        definition='factor k hit: observed/k <= predicted <= observed*k; this is threshold coverage, not confidence/probability')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(pd.DataFrame(rows).to_string(index=False));print(papers.to_string());print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
