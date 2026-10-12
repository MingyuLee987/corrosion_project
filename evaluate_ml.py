"""Fixed-parameter ML comparison with every test publication excluded in full."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from corrosion_re_expand import NUM, CAT, CONTEXT_NUM, CONTEXT_CAT

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/ML_evaluation'
MODELS = ['median', 'ridge', 'svr', 'random_forest', 'extra_trees', 'hist_gradient_boosting']


def estimator(name):
    return {
        'median': lambda: DummyRegressor(strategy='median'),
        'ridge': lambda: Ridge(alpha=10),
        'svr': lambda: SVR(C=1., epsilon=.1),
        'random_forest': lambda: RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=2),
        'extra_trees': lambda: ExtraTreesRegressor(n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=42, n_jobs=2),
        'hist_gradient_boosting': lambda: HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7, min_samples_leaf=5, l2_regularization=10., early_stopping=False, random_state=42),
    }[name]()


def features(data, numeric, categorical):
    x = data.reindex(columns=numeric+categorical).copy()
    x['RE_log1p_ppm'] = np.log1p(data.RE_wt_pct*10000)
    for c in numeric:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    for c in categorical:
        x[c] = x[c].replace(['unknown','unreported','not_reported',''], np.nan)
    x['RE_element_cat'] = x.RE_element_cat.replace({'La_Ce':'La+Ce'})
    if 'material_class' in x:
        x['material_class'] = x.material_class.replace({'low-alloy steel':'low alloy steel'})
    return x


def scores(data, prediction):
    y = np.log10(data.icorr_A_cm2.to_numpy(float))
    error = np.abs(y-prediction)
    groups = data.reference_id.to_numpy()
    return dict(rows=len(data), papers=int(data.reference_id.nunique()),
        log10_R2=float(r2_score(y,prediction)), log10_MAE=float(mean_absolute_error(y,prediction)),
        log10_RMSE=float(np.sqrt(mean_squared_error(y,prediction))),
        paper_balanced_log10_MAE=float(np.mean([error[groups==p].mean() for p in np.unique(groups)])),
        within_factor2_pct=float(100*np.mean(error<=np.log10(2))),
        within_factor10_pct=float(100*np.mean(error<=1.)),
        geometric_mean_error_factor=float(10**error.mean()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_data = pd.read_csv(ROOT/'data/training_PDP_augmented.csv',encoding='utf-8-sig')
    assert all_data.record_id.is_unique and all_data.icorr_A_cm2.gt(0).all()
    re_studies = all_data.iloc[:50].reference_id.unique()
    assert len(re_studies)==11 and all_data.iloc[:50].RE_wt_pct.gt(0).sum()==33
    summaries, records, folds = [], [], []
    for dataset in ['previous_98','current_314']:
        d = all_data.loc[all_data.reference_id.ne('RebarLi2023')].copy() if dataset=='previous_98' else all_data.copy()
        d = d.reset_index(drop=True)
        assert len(d)==(98 if dataset=='previous_98' else 314)
        groups = d.reference_id.to_numpy()
        y = np.log10(d.icorr_A_cm2.to_numpy(float))
        for kind,numeric,categorical in [('core8',NUM,CAT),('composition_and_conditions',CONTEXT_NUM,CONTEXT_CAT)]:
            x = features(d,numeric,categorical)
            for name in MODELS:
                prediction = np.full(len(d),np.nan)
                for held in np.unique(groups):
                    train = np.flatnonzero(groups!=held)
                    test = np.flatnonzero(groups==held)
                    assert not set(groups[train])&set(groups[test])
                    cats = [c for c in categorical if x.iloc[train][c].notna().any()]
                    prep = ColumnTransformer([
                        ('numeric',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),numeric),
                        ('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats)])
                    pipe = Pipeline([('prep',prep),('model',estimator(name))])
                    counts = pd.Series(groups[train]).value_counts()
                    weights = np.array([1/counts[p] for p in groups[train]])
                    weights /= weights.mean()
                    pipe.fit(x.iloc[train],y[train],model__sample_weight=weights)
                    prediction[test] = pipe.predict(x.iloc[test])
                    folds.append(dict(dataset=dataset,feature_set=kind,model=name,held_out_paper=held,
                        train_rows=len(train),test_rows=len(test),test_log10_MAE=float(mean_absolute_error(y[test],prediction[test])),
                        training_missing_fraction=x.iloc[train].isna().mean().to_dict()))
                assert np.isfinite(prediction).all()
                for scope,mask in [('all_rows',np.ones(len(d),dtype=bool)),
                    ('RE_study_rows',np.isin(groups,re_studies)),('RE_positive_only',d.RE_wt_pct.gt(0).to_numpy())]:
                    summaries.append(dict(dataset=dataset,feature_set=kind,model=name,scope=scope,**scores(d.loc[mask],prediction[mask])))
                output = d[['record_id','reference_id','RE_wt_pct','icorr_A_cm2']].copy()
                output['dataset']=dataset; output['feature_set']=kind; output['model']=name
                output['observed_log10']=y; output['predicted_log10']=prediction
                output['predicted_icorr_A_cm2']=10**prediction
                records.append(output)
                print(dataset,kind,name,'RE_MAE',summaries[-1]['log10_MAE'],flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'model_comparison.csv',index=False,encoding='utf-8-sig')
    pd.concat(records,ignore_index=True).to_csv(OUT/'heldout_predictions.csv',index=False,encoding='utf-8-sig')
    metadata=dict(validation='Leave-one-publication-out. Imputation, scaling and encoding fitted only on training publications. Fixed parameters chosen before this run; no test-based tuning. All replicates of a source remain in one group. Paper-inverse-size sample weights.',
        target='log10(icorr_A_cm2)',random_seed=42,folds=folds,
        numeric_features={'core8':NUM,'composition_and_conditions':CONTEXT_NUM},
        categorical_features={'core8':CAT,'composition_and_conditions':CONTEXT_CAT},
        parameters={name:estimator(name).get_params() for name in MODELS},
        limitations=['Model rankings are exploratory comparisons on previously observed publications, not a nested estimate of selecting the best model.',
            '216 measurements from one source have unknown exact chemistry and different mortar/pore-solution environments.',
            'Ecorr, publication/sample IDs, corrosion-derived observations and inferred casting methods excluded from inputs.',
            'Each dataset/feature set is evaluated against its own matching median baseline.'],promoted=False)
    (OUT/'validation.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.loc[summary.scope.eq('RE_positive_only')].sort_values('paper_balanced_log10_MAE').to_string(index=False))


if __name__=='__main__':
    main()
