"""Literature-informed exploratory extension; leaves production v2 untouched.

All preprocessing and candidate selection are confined to training papers.
Existing papers have been explored previously: independent validation is still needed.
"""
from pathlib import Path
import json
import warnings
import argparse
import hashlib
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import corrosion_v2 as v2

OUT = Path(__file__).resolve().parent / 'results/research_extension'
EXTRAS = ['Si_wt_pct', 'Mn_wt_pct', 'Mo_wt_pct', 'Cu_wt_pct',
          'TP_scan_rate_mV_s', 'gas_condition_cat']

def feature_frame(data):
    data = data.copy()
    if 'steel_family' not in data:
        data['steel_family'] = data.material_class.apply(v2.steel_family)
    x = v2.design(data)
    for col in EXTRAS:
        values = data[col] if col in data else pd.Series(np.nan, index=data.index)
        x[col] = values.fillna('not_reported').astype(str) if col == 'gas_condition_cat' else pd.to_numeric(values, errors='coerce')
    return x

def predict_new(input_path):
    package = joblib.load(OUT/'exploratory_model.joblib')
    data = pd.read_csv(input_path, encoding='utf-8-sig')
    control = pd.to_numeric(data['control_icorr_A_cm2'], errors='coerce').to_numpy(float)
    if not np.isfinite(control).all() or (control <= 0).any():
        raise ValueError('Positive finite matched measured control is required')
    x = feature_frame(data)
    if package['selected'] == 'control_only': delta = np.zeros(len(data))
    elif package['selected'] == 'median_delta': delta = np.full(len(data), package['median_delta'])
    else: delta = package['model'].predict(x)
    messages = []
    for _, row in x.iterrows():
        issues = []
        for col, limits in package['ranges'].items():
            if pd.notna(row[col]) and not limits[0] <= row[col] <= limits[1]: issues.append(f'{col}:outside_training_range')
        for col, categories in package['categories'].items():
            if row[col] not in categories: issues.append(f'{col}:unseen_category')
        messages.append('|'.join(issues))
    data['predicted_delta_log10'] = delta
    data['predicted_icorr_A_cm2'] = control * 10**delta
    data['applicability_warnings'] = messages
    data['model_status'] = package['status']
    data.to_csv(OUT/'new_predictions.csv', index=False, encoding='utf-8-sig')
    print(OUT/'new_predictions.csv')
PANELS = {
    'compact': (['RE_wt_pct', 'C_wt_pct', 'Cr_wt_pct', 'Ni_wt_pct',
                 'temperature_C', 'pre_exposure_h'], ['steel_family', 'RE_element_cat',
                 'electrolyte_family', 'surface_state_cat']),
    'v2': (v2.NUMERIC, v2.CATEGORICAL),
    'expanded': (v2.NUMERIC + ['Si_wt_pct', 'Mn_wt_pct', 'Mo_wt_pct',
                 'Cu_wt_pct', 'TP_scan_rate_mV_s'], v2.CATEGORICAL + ['gas_condition_cat']),
}

def specifications():
    specs = [('control_only', None, None), ('median_delta', None, None)]
    for panel in PANELS:
        for c in [0.03, 0.3, 3.0]:
            specs.append((f'{panel}:svr:{c}', panel, SVR(C=c, epsilon=0.1)))
        for alpha in [1., 10., 100.]:
            specs.append((f'{panel}:ridge:{alpha}', panel, Ridge(alpha=alpha)))
        specs.append((f'{panel}:huber', panel, HuberRegressor(alpha=1., max_iter=1000)))
        specs.append((f'{panel}:gpr', panel, GaussianProcessRegressor(
            kernel=ConstantKernel(0.25)*Matern(length_scale=3., nu=1.5),
            alpha=0.1, optimizer=None, normalize_y=True)))
        for kind, cls in [('extra', ExtraTreesRegressor), ('rf', RandomForestRegressor)]:
            specs.append((f'{panel}:{kind}', panel, cls(n_estimators=60,
                min_samples_leaf=3, max_depth=3, random_state=20261010, n_jobs=1)))
    return specs

def pipeline(panel, estimator):
    nums, cats = PANELS[panel]
    prep = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median',
            add_indicator=True, keep_empty_features=True)), ('scale', StandardScaler())]), nums),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)])
    return Pipeline([('pre', prep), ('regressor', estimator)])

def main():
    warnings.filterwarnings('ignore', category=UserWarning)
    data = v2.load_data()
    pairs = v2.pair_rows(data)
    extras = EXTRAS
    pairs = pairs.merge(data[['record_id']+extras], on='record_id', validate='one_to_one')
    x = feature_frame(pairs)
    y = pairs.delta_log10.to_numpy(float)
    groups = pairs.reference_id.to_numpy(str)
    papers = sorted(set(groups))
    specs = specifications()
    cache = {}
    def predict(spec, train_papers, test_paper):
        key = (spec[0], tuple(sorted(train_papers)), test_paper)
        if key in cache:
            return cache[key]
        tr = np.flatnonzero(np.isin(groups, train_papers))
        te = np.flatnonzero(groups == test_paper)
        assert test_paper not in train_papers and not set(tr).intersection(te)
        if spec[0] == 'control_only': pred = np.zeros(len(te))
        elif spec[0] == 'median_delta': pred = np.full(len(te), np.median(y[tr]))
        else:
            from sklearn.base import clone
            model = pipeline(spec[1], clone(spec[2]))
            kwargs = {} if ':gpr' in spec[0] else {'regressor__sample_weight': v2.group_weights(groups[tr])}
            model.fit(x.iloc[tr], y[tr], **kwargs)
            pred = model.predict(x.iloc[te])
        cache[key] = pred
        return pred
    def select(pool):
        scores = []
        for spec in specs:
            mse = []
            for held in pool:
                pred = predict(spec, [p for p in pool if p != held], held)
                mse.append(float(np.mean((y[groups == held]-pred)**2)))
            scores.append(float(np.mean(mse)))
        return specs[int(np.argmin(scores))], dict(zip([s[0] for s in specs], scores))
    outer = np.full(len(y), np.nan)
    folds = []
    for held in papers:
        pool = [p for p in papers if p != held]
        spec, scores = select(pool)
        pred = predict(spec, pool, held)
        outer[groups == held] = pred
        folds.append({'paper': held, 'selected': spec[0], 'inner_paper_MSE': scores,
                      'outer_MAE': float(mean_absolute_error(y[groups == held], pred))})
        print(held, spec[0], flush=True)
    actual = pairs.control_log10.to_numpy()+y
    predicted = pairs.control_log10.to_numpy()+outer
    selected, scores = select(papers)
    result = {'status': 'exploratory; previously examined papers; requires new independent papers',
        'scope': 'matched measured control required', 'rows': len(y), 'papers': len(papers),
        'candidate_count': len(specs), 'selection': 'mean paper MSE in inner leave-one-paper-out',
        'data_sha256': hashlib.sha256(v2.DATA.read_bytes()).hexdigest(),
        'absolute_log_R2': float(r2_score(actual, predicted)),
        'absolute_log_MAE': float(mean_absolute_error(actual, predicted)),
        'absolute_raw_R2': float(r2_score(10**actual, 10**predicted)),
        'delta_R2': float(r2_score(y, outer)), 'control_only_log_R2': float(r2_score(actual, pairs.control_log10)),
        'final_candidate': selected[0], 'final_inner_scores': scores, 'folds': folds}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'benchmark.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    pairs['predicted_delta_log10'] = outer
    pairs['predicted_icorr_A_cm2'] = 10**predicted
    pairs.to_csv(OUT/'heldout_predictions.csv', index=False, encoding='utf-8-sig')
    from sklearn.base import clone
    model = None
    if selected[1]:
        model = pipeline(selected[1], clone(selected[2]))
        kwargs = {} if ':gpr' in selected[0] else {'regressor__sample_weight': v2.group_weights(groups)}
        model.fit(x, y, **kwargs)
    joblib.dump({'model': model, 'selected': selected[0], 'features': list(x),
        'training_papers': papers, 'median_delta': float(np.median(y)),
        'ranges': {col: [float(x[col].min()), float(x[col].max())] for col in x.select_dtypes(include='number') if x[col].notna().any()},
        'categories': {col: sorted(x[col].unique().tolist()) for col in x.select_dtypes(exclude='number')},
        'scope': result['scope'], 'status': result['status']}, OUT/'exploratory_model.joblib')
    print(json.dumps({k:v for k,v in result.items() if k not in ['folds', 'final_inner_scores']}, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predict', type=Path)
    args = parser.parse_args()
    predict_new(args.predict) if args.predict else main()
