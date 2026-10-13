"""Nested publication exclusion for process feature blocks and auxiliary weighting."""
from pathlib import Path
import json
import hashlib
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from corrosion_re_expand import NUM, CAT
from composition_values import apply_composition_values

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/process_optimization'
PROCESS_N = ['final_heat_treatment_temp_C', 'final_heat_treatment_time_min']
PROCESS_C = ['processing_state_cat', 'melting_route_cat', 'cooling_method_cat']
ENV_N = ['Cl_mol_L', 'NaCl_wt_pct', 'H2SO4_mol_L', 'pH', 'temperature_C', 'log_pre_exposure_verified']
ENV_C = ['surface_state_cat', 'gas_condition_cat']
PROTOCOL_N = ['TP_scan_rate_mV_s', 'OCP_stabilization_verified_s']
PROTOCOL_C = ['TP_reference_electrode_cat']
ALLOY_N = ['Cr_wt_pct', 'Ni_wt_pct', 'Mo_wt_pct', 'Cu_wt_pct']
LEAN = [c for c in NUM if c not in ['P_wt_pct', 'S_wt_pct']]
FEATURE_SETS = {
    'core8': (NUM, CAT),
    'drop_P_S': (LEAN, CAT),
    'add_manufacturing': (NUM+PROCESS_N, CAT+PROCESS_C),
    'drop_P_S_add_manufacturing': (LEAN+PROCESS_N, CAT+PROCESS_C),
    'add_environment': (NUM+ENV_N, CAT+ENV_C),
    'add_protocol': (NUM+PROTOCOL_N, CAT+PROTOCOL_C),
    'add_alloy': (NUM+ALLOY_N, CAT+['material_class']),
    'add_process_environment': (NUM+PROCESS_N+ENV_N, CAT+PROCESS_C+ENV_C),
    'drop_P_S_add_process_environment': (LEAN+PROCESS_N+ENV_N, CAT+PROCESS_C+ENV_C),
}
MODELS = ['svr03', 'svr1', 'ridge10']
WEIGHTS = [0., .25, 1.]
CONFIGS = [(f, m, w) for f in FEATURE_SETS for m in MODELS for w in WEIGHTS] + [('core8', 'median', 1.)]


def estimator(name):
    if name == 'median':
        return DummyRegressor(strategy='median')
    return Ridge(alpha=10.) if name == 'ridge10' else SVR(C=.3 if name == 'svr03' else 1., epsilon=.1)


def transform(data):
    data = apply_composition_values(data)
    x = data.copy()
    x['RE_log1p_ppm'] = np.log1p(data.RE_wt_pct*10000)
    # Explicitly verified pre-exposure clock; do not mix in generic exposure_time_h.
    x['log_pre_exposure_verified'] = np.log1p(pd.to_numeric(data.pre_exposure_h, errors='coerce'))
    a = pd.to_numeric(data.OCP_stabilization_s, errors='coerce')
    b = pd.to_numeric(data.TP_OCP_stabilization_time_s, errors='coerce')
    assert not (a.notna() & b.notna() & a.ne(b)).any(), 'Conflicting OCP clocks'
    x['OCP_stabilization_verified_s'] = a.combine_first(b)
    for numeric, categorical in FEATURE_SETS.values():
        for c in numeric:
            x[c] = pd.to_numeric(x[c], errors='coerce')
        for c in categorical:
            x[c] = x[c].replace(['unknown', 'unreported', 'not_reported', ''], np.nan)
    x['RE_element_cat'] = x.RE_element_cat.replace({'La_Ce':'La+Ce'})
    x['material_class'] = x.material_class.replace({'low-alloy steel':'low alloy steel'})
    return x


def train_predict(x, y, groups, re_papers, train, test, feature, auxiliary_weight, names):
    if auxiliary_weight == 0:
        train = train[np.isin(groups[train], re_papers)]
    assert not set(groups[train]) & set(groups[test])
    numeric, categorical = FEATURE_SETS[feature]
    # Drop train-all-missing and train-constant predictors for every fold.
    numeric = [c for c in numeric if x.iloc[train][c].nunique() > 1]
    categorical = [c for c in categorical if x.iloc[train][c].nunique() > 1]
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]), numeric),
        ('category',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                             ('encode',OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),categorical)])
    tx = prep.fit_transform(x.iloc[train])
    vx = prep.transform(x.iloc[test])
    counts = pd.Series(groups[train]).value_counts()
    weights = np.array([1/counts[g] for g in groups[train]])
    weights *= np.where(np.isin(groups[train], re_papers), 1., auxiliary_weight)
    weights /= weights.mean()
    predictions = {}
    for name in names:
        est = estimator(name)
        est.fit(tx, y[train], sample_weight=weights)
        predictions[name] = est.predict(vx)
    return predictions


def metrics(y, prediction, groups):
    error = np.abs(y-prediction)
    return dict(log10_R2=float(r2_score(y,prediction)), log10_MAE=float(error.mean()),
        paper_balanced_log10_MAE=float(np.mean([error[groups==p].mean() for p in np.unique(groups)])))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d = pd.read_csv(ROOT/'data/training_process_curated.csv')
    x = transform(d)
    y = np.log10(d.icorr_A_cm2.to_numpy(float))
    groups = d.reference_id.to_numpy()
    re_papers = np.unique(groups[:50])
    positive = d.RE_wt_pct.gt(0).to_numpy()
    assert positive.sum() == 33 and len(re_papers) == 11
    # Cache single-held-paper predictions to reuse in fixed-setting comparisons.
    fixed = {c: np.full(len(d), np.nan) for c in CONFIGS}
    selected = np.full(len(d),np.nan)
    selection_log, inner_log = [], []
    for held in re_papers:
        outer_train = np.flatnonzero(groups != held)
        outer_test = np.flatnonzero((groups == held) & positive)
        inner_validation = np.flatnonzero(np.isin(groups, re_papers) & (groups != held) & positive)
        inner = {c: np.full(len(d),np.nan) for c in CONFIGS}
        for inner_held in re_papers[re_papers != held]:
            train = np.flatnonzero((groups != held) & (groups != inner_held))
            test = np.flatnonzero((groups == inner_held) & positive)
            assert held not in groups[train] and held not in groups[test]
            for feature in FEATURE_SETS:
                for w in WEIGHTS:
                    preds = train_predict(x,y,groups,re_papers,train,test,feature,w,MODELS)
                    for name, pred in preds.items():
                        inner[(feature,name,w)][test] = pred
            inner[('core8','median',1.)][test] = train_predict(x,y,groups,re_papers,train,test,'core8',1.,['median'])['median']
        inner_scores = {}
        for c in CONFIGS:
            assert np.isfinite(inner[c][inner_validation]).all()
            score = metrics(y[inner_validation], inner[c][inner_validation], groups[inner_validation])
            inner_scores[c] = score
            inner_log.append(dict(held_out_paper=held, feature_set=c[0],model=c[1],auxiliary_weight=c[2],**score))
        best = min(CONFIGS, key=lambda c: (-inner_scores[c]['log10_R2'], inner_scores[c]['paper_balanced_log10_MAE'],CONFIGS.index(c)))
        for feature in FEATURE_SETS:
            for w in WEIGHTS:
                preds = train_predict(x,y,groups,re_papers,outer_train,outer_test,feature,w,MODELS)
                for name, pred in preds.items():
                    fixed[(feature,name,w)][outer_test] = pred
        fixed[('core8','median',1.)][outer_test] = train_predict(x,y,groups,re_papers,outer_train,outer_test,'core8',1.,['median'])['median']
        selected[outer_test] = fixed[best][outer_test]
        selection_log.append(dict(held_out_paper=held,feature_set=best[0],model=best[1],auxiliary_weight=best[2],inner_scores=inner_scores[best]))
        print(held, 'selected', best, 'inner_R2', round(inner_scores[best]['log10_R2'],4),flush=True)
    idx = np.flatnonzero(positive)
    summaries, records = [], []
    for c, pred in fixed.items():
        summaries.append(dict(feature_set=c[0],model=c[1],auxiliary_weight=c[2],rows=len(idx),papers=len(re_papers),
                              **metrics(y[idx],pred[idx],groups[idx])))
        for i in idx:
            records.append(dict(record_id=d.at[i,'record_id'],reference_id=groups[i],feature_set=c[0],model=c[1],
                auxiliary_weight=c[2],observed_log10=y[i],predicted_log10=pred[i]))
    summary = pd.DataFrame(summaries).sort_values('log10_R2',ascending=False)
    summary.to_csv(OUT/'feature_ablation_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(records).to_csv(OUT/'feature_ablation_predictions.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(inner_log).to_csv(OUT/'inner_selection_scores.csv',index=False,encoding='utf-8-sig')
    output = d.loc[positive,['record_id','reference_id','icorr_A_cm2']].copy()
    output['observed_log10'] = y[idx]
    output['nested_predicted_log10'] = selected[idx]
    output['median_predicted_log10'] = fixed[('core8','median',1.)][idx]
    output.to_csv(OUT/'nested_heldout_predictions.csv',index=False,encoding='utf-8-sig')
    metadata = dict(target='log10(icorr_A_cm2)',rows=314,RE_evaluated_rows=33,RE_papers=11,
        input_sha256=hashlib.sha256((ROOT/'data/training_process_curated.csv').read_bytes()).hexdigest(),
        selection_objective='Maximize pooled inner RE-positive log10 R2; tie-break with paper-balanced MAE.',
        feature_sets={k:dict(numeric=v[0],categorical=v[1]) for k,v in FEATURE_SETS.items()},
        candidate_models=MODELS,auxiliary_weights=WEIGHTS,configurations=len(CONFIGS),
        nested_metrics=metrics(y[idx],selected[idx],groups[idx]),
        matching_median_metrics=metrics(y[idx],fixed[('core8','median',1.)][idx],groups[idx]),
        folds=selection_log,
        validation='Nested leave-one-RE-publication-out; outer paper absent from every training and inner fold. All transformations train-only. Auxiliary papers training-only; no all-314 validation claim.',
        limitations=['Fixed-setting leaderboard is exploratory, not unbiased validation of picking its top setting.',
                     'These publications were already observed in previous project work; nested estimate is not a new independent external test.',
                     '314 rows retain chemistry and process missingness; source-linked does not mean fully complete.',
                     'No causal or optimal manufacturing recipe can be inferred from these predictions.'],promoted=False)
    (OUT/'nested_validation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print('NESTED',metadata['nested_metrics'],flush=True)
    print(summary.head(10).to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
