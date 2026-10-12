"""Exploratory paper-held-out modelling using mandatory verified immersion time.

No paper/sample identifiers, Ecorr, or measured corrosion products enter predictors.
Two feature sets are evaluated under the same nested leave-one-paper-out splits.
No model is promoted to production on the strength of this small dataset.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/time_verified'
NUM = ['C_wt_pct','Si_wt_pct','Mn_wt_pct','Al_wt_pct','Cr_wt_pct','Ni_wt_pct','Mo_wt_pct','Cu_wt_pct',
       'P_wt_pct','S_wt_pct','RE_wt_pct','temperature_C','NaCl_wt_pct','NaCl_g_L','NaHCO3_g_L','Cl_mol_L',
       'H2SO4_mol_L','NaHSO3_mol_L','pH','dissolved_oxygen_mg_L','rotation_rpm',
       'TP_scan_rate_mV_s']
CAT = ['material_class','RE_element_cat','electrolyte_family','gas_condition_cat',
       'hydrodynamic_condition_cat','surface_state_cat','processing_state_cat']
MAX_INPUT_COLUMNS = 40
MAX_MISSING_FRACTION = .20
CANDIDATES = ['median','ridge10','svr03']

def model(name, numeric, categorical):
    if name == 'median':
        return DummyRegressor(strategy='median')
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,
            keep_empty_features=True)),('scale',StandardScaler())]), numeric),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
    return Pipeline([('prep',prep),('reg',Ridge(alpha=10) if name=='ridge10' else SVR(C=.3,epsilon=.1))])

def predict(name,x,y,g,train,test,numeric):
    # Select using training data only, before imputation; unknown labels are missing.
    rates = x.iloc[train].isna().mean()
    priority = (['log1p_immersion_h'] if 'log1p_immersion_h' in numeric else []) + numeric + CAT
    selected = list(dict.fromkeys(c for c in priority if rates.get(c,1) <= MAX_MISSING_FRACTION))[:MAX_INPUT_COLUMNS]
    if not selected:
        raise ValueError('No predictors meet the <=20% missingness requirement in training fold')
    fitted = model(name,[c for c in selected if c in numeric],[c for c in selected if c in CAT])
    if name == 'median':
        fitted.fit(x.iloc[train],y[train])
    else:
        counts = pd.Series(g[train]).value_counts()
        weights = np.asarray([1/counts[p] for p in g[train]])
        fitted.fit(x.iloc[train],y[train],reg__sample_weight=weights)
    return fitted.predict(x.iloc[test])

def score(y,pred,g):
    return dict(log10_R2=float(r2_score(y,pred)),log10_MAE=float(mean_absolute_error(y,pred)),
        paper_balanced_log10_MAE=float(np.mean([mean_absolute_error(y[g==p],pred[g==p]) for p in np.unique(g)])))

def evaluate(data,with_time):
    numeric = NUM+(['log1p_immersion_h'] if with_time else [])
    x = data.reindex(columns=numeric+CAT).copy()
    for c in numeric:
        x[c] = pd.to_numeric(x[c],errors='coerce')
    for c in CAT:
        x[c] = x[c].replace(['not_reported','unknown','unreported',''],np.nan)
    # Prevent legacy label spellings from creating artificial paper distinctions.
    x['material_class'] = x.material_class.str.replace('low alloy steel','low-alloy steel',regex=False)
    x['RE_wt_pct'] = np.log1p(x.RE_wt_pct*10000)
    y = np.log10(data.icorr_A_cm2.to_numpy(float))
    g = data.reference_id.to_numpy()
    papers = np.unique(g)
    pred = np.full(len(data),np.nan)
    baseline = np.full(len(data),np.nan)
    folds = []
    for held in papers:
        train = np.flatnonzero(g!=held)
        test = np.flatnonzero(g==held)
        scores = {}
        for candidate in CANDIDATES:
            errors = []
            for inner in papers[papers!=held]:
                tr = np.flatnonzero((g!=held)&(g!=inner))
                va = np.flatnonzero(g==inner)
                errors.append(mean_absolute_error(y[va],predict(candidate,x,y,g,tr,va,numeric)))
            scores[candidate] = float(np.mean(errors))
        chosen = min(CANDIDATES,key=lambda c:(scores[c],CANDIDATES.index(c)))
        pred[test] = predict(chosen,x,y,g,train,test,numeric)
        baseline[test] = predict('median',x,y,g,train,test,numeric)
        rates = x.iloc[train].isna().mean()
        priority = (['log1p_immersion_h'] if with_time else []) + numeric + CAT
        columns = list(dict.fromkeys(c for c in priority if rates.get(c,1) <= MAX_MISSING_FRACTION))[:MAX_INPUT_COLUMNS]
        folds.append(dict(held_out_paper=held,selected=chosen,input_columns=columns,input_column_count=len(columns),input_missing_fraction={c:float(rates[c]) for c in columns},inner_scores=scores,
                          test_rows=len(test),log10_MAE=float(mean_absolute_error(y[test],pred[test]))))
    result = dict(with_time=with_time,rows=len(data),papers=len(papers),metrics=score(y,pred,g),
        median_baseline=score(y,baseline,g),folds=folds,input_numeric=numeric,input_categorical=CAT,
        validation='Nested leave-one-paper-out; imputation/encoding fitted inside each training fold',
        feature_policy='At most 40 source columns; missingness <=20% in each training fold, before imputation; categorical unknown labels count as missing; selection never uses held-out data',
        status='exploratory; small heterogeneous literature dataset; not production validation')
    records = data[['record_id','reference_id','immersion_time_h','icorr_A_cm2']].copy()
    records['observed_log10'] = y
    records['predicted_log10'] = pred
    records['predicted_icorr_A_cm2'] = 10**pred
    records['median_baseline_log10'] = baseline
    suffix = 'with_time' if with_time else 'without_time'
    records.to_csv(OUT/f'heldout_{suffix}.csv',index=False,encoding='utf-8-sig')
    return result

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    data = pd.read_csv(ROOT/'data/time/training_time_verified.csv',encoding='utf-8-sig')
    if not data.target_method.eq('potentiodynamic_Tafel').all():
        raise ValueError('Only PDP-derived Tafel icorr targets are eligible; mass-loss and other target methods excluded')
    if data.immersion_time_h.isna().any() or not data.immersion_time_h.gt(0).all():
        raise ValueError('Verified positive immersion duration is mandatory')
    if not data.time_evidence_status.eq('explicit_PDP_link').all():
        raise ValueError('Every target must have explicit PDP-linked time evidence')
    if not data.pre_exposure_type.eq('continuous_immersion').all():
        raise ValueError('Noncontinuous exposure belongs in the separate auxiliary dataset')
    if not data.composition_evidence_status.isin(['sample_specific_table','reported_in_text','documented_base_plus_measured_RE']).all() or data.composition_source_locator.isna().any():
        raise ValueError('Source-supported quantitative composition is mandatory')
    if data.reference_id.nunique() < 3:
        result = {'status':'not_run_insufficient_independent_papers', 'rows':len(data), 'papers':int(data.reference_id.nunique()), 'reason':'Nested paper-held-out validation requires at least three independent papers; prior benchmark no longer applies after composition exclusions'}
        (OUT/'benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        for suffix in ['with_time','without_time']:
            pd.DataFrame(columns=['record_id','reference_id','observed_log10','predicted_log10']).to_csv(OUT/f'heldout_{suffix}.csv',index=False,encoding='utf-8-sig')
        print(json.dumps(result)); return
    data['log1p_immersion_h'] = np.log1p(data.immersion_time_h)
    audit = data.reindex(columns=NUM+CAT+['log1p_immersion_h']).replace(['not_reported','unknown','unreported',''],np.nan).isna().mean().rename('missing_fraction').rename_axis('column').reset_index()
    audit['meets_20pct'] = audit.missing_fraction <= MAX_MISSING_FRACTION
    audit.to_csv(OUT/'feature_missingness.csv',index=False,encoding='utf-8-sig')
    results = {label:evaluate(data,flag) for label,flag in [('with_time',True),('without_time',False)]}
    results['RE_model_status'] = 'Only one verified continuous-immersion RE paper: independent paper validation is impossible; no new RE model fitted'
    (OUT/'benchmark.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v['metrics'] for k,v in results.items() if isinstance(v,dict)},indent=2))

if __name__ == '__main__':
    main()
