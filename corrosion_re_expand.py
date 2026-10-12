"""Auditable RE literature additions and paired publication-held-out comparison.

Run with .venv/Scripts/python.exe corrosion_re_expand.py.
New publication targets never tune a model
when measuring its first external prediction.
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
OUT = ROOT / 'results/RE_expanded'
CANDIDATES = ['median', 'ridge10', 'svr03']


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

def model(name, numeric, categorical):
    if name == 'median':
        return DummyRegressor(strategy='median')
    prep = ColumnTransformer([
        ('numeric', Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,
            keep_empty_features=True)),('scale',StandardScaler())]), numeric),
        ('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),categorical)])
    return Pipeline([('prep',prep),('reg',Ridge(alpha=10) if name=='ridge10' else SVR(C=.3,epsilon=.1))])


def additions(columns):
    url = 'https://www.ams.org.cn/EN/10.11900/0412.1961.2022.00359'
    rows = []
    for sample, ce, nominal, current, potential in [
        ('C1', 0., 0., 8.53e-5, -1.),
        ('C3', .0195, .03, 4.03e-5, -.936),
        ('C5', .0416, .08, 5.07e-5, -.986),
    ]:
        row = dict(record_id=f'NEW_75Cr1_24_{sample}', reference_id='Ce75Cr1_24',
            sample_id=sample, C_wt_pct=.76, Si_wt_pct=.29, Mn_wt_pct=.6938,
            Al_wt_pct=.013, Cr_wt_pct=.51, RE_wt_pct=ce, Ce_wt_pct=ce,
            RE_nominal_wt_pct=nominal, RE_element_cat='none' if ce == 0 else 'Ce',
            RE_log1p_ppm=np.log1p(ce * 10000), material_class='low alloy steel',
            alloy_system='75Cr1-Ce', melting_route_cat='high_frequency_induction',
            RE_addition_stage_cat='during_melting', cooling_method_cat='furnace',
            processing_state_cat='as_cast', surface_state_cat='polished',
            electrolyte_family='NaCl', NaCl_wt_pct=3.5,
            TP_reference_electrode_cat='SCE', icorr_A_cm2=current,
            raw_icorr_value=current, raw_icorr_unit='A/cm2', log_icorr=np.log10(current),
            Ecorr_V=potential, journal='Acta Metallurgica Sinica', publication_year=2024,
            doi='10.11900/0412.1961.2022.00359',
            paper_title='Effect of Ce on Cleanliness, Microstructure, and Pitting Corrosion Resistance of 75Cr1 Steel',
            source_url=url, target_locator='Table 2, PDF p8 (printed p1236); Fig14; Sec2.4',
            composition_basis='documented_base_plus_measured_RE',
            icorr_status='reported', primary_eligible=True, origin='RE_expansion_2026_10_12',
            TP_settings_source_url=url, TP_settings_status='partially_reported',
            gas_condition_cat='not_reported',
            quality_flags='Base slab composition Sec1 PDF p2; residual Ce ICP-AES Sec1 p3. '
                'Non-RE chemistry shared from base slab, not individual remelt measurements. '
                'P unreported; S only graphical and changes with Ce: left missing. '
                'Temperature, scan rate and pre-exposure unreported. '
                '1150C/30min is separate in-situ microscopy, not PDP heat treatment. '
                'C2/C4 have no tabulated icorr and are not invented.',
            remarks='Source-verified printed icorr; exploratory base-chemistry provenance; not time-verified.')
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=columns)


def frame(data):
    x = data.reindex(columns=FEATURES).copy()
    x['RE_log1p_ppm'] = np.log1p(pd.to_numeric(data.RE_wt_pct) * 10000)
    x['RE_element_cat'] = x.RE_element_cat.replace({'La_Ce': 'La+Ce'})
    for c in NUM:
        x[c] = pd.to_numeric(x[c], errors='coerce')
    return x


def fit_predict(name, data, train, test):
    x = frame(data)
    y = np.log10(data.icorr_A_cm2.to_numpy(float))
    pipe = model(name, NUM, CAT)
    if name == 'median':
        pipe.fit(x.iloc[train], y[train])
    else:
        groups = data.reference_id.to_numpy()[train]
        counts = pd.Series(groups).value_counts()
        weights = np.array([1 / counts[g] for g in groups])
        weights /= weights.mean()
        pipe.fit(x.iloc[train], y[train], reg__sample_weight=weights)
    return pipe.predict(x.iloc[test])


def choose(data, train):
    g = data.reference_id.to_numpy()
    y = np.log10(data.icorr_A_cm2.to_numpy(float))
    scores = {}
    for name in CANDIDATES:
        errors = []
        for held in np.unique(g[train]):
            tr = train[g[train] != held]
            va = train[g[train] == held]
            errors.append(mean_absolute_error(y[va], fit_predict(name, data, tr, va)))
        scores[name] = float(np.mean(errors))
    return min(CANDIDATES, key=lambda n: (scores[n], CANDIDATES.index(n))), scores


def evaluate(data, held_papers):
    g = data.reference_id.to_numpy()
    pred = np.full(len(data), np.nan)
    folds = []
    for held in held_papers:
        tr, va = np.flatnonzero(g != held), np.flatnonzero(g == held)
        name, scores = choose(data, tr)
        pred[va] = fit_predict(name, data, tr, va)
        rates = frame(data).iloc[tr].isna().mean()
        folds.append(dict(held_out_paper=held, model=name, inner_paper_MAE=scores,
            training_missing_fraction=rates.to_dict(),
            missingness_policy_pass=bool(rates.max() <= .20)))
        print(held, name, flush=True)
    return pred, folds


def metrics(data, pred):
    valid = np.isfinite(pred)
    y = np.log10(data.icorr_A_cm2.to_numpy(float))[valid]
    g = data.reference_id.to_numpy()[valid]
    p = pred[valid]
    return dict(rows=int(valid.sum()), papers=len(np.unique(g)),
        log10_R2=float(r2_score(y, p)), log10_MAE=float(mean_absolute_error(y, p)),
        paper_balanced_log10_MAE=float(np.mean([mean_absolute_error(y[g == a], p[g == a]) for a in np.unique(g)])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = load_core()
    original = pd.read_csv(ROOT / 'data/training_51.csv', encoding='utf-8-sig')
    assert len(base) == len(original)
    assert set(base.record_id) == set(original.record_id)
    indexed = original.set_index('record_id').loc[base.record_id]
    assert np.allclose(indexed.icorr_A_cm2, base.icorr_A_cm2, rtol=1e-12)
    new = additions(original.columns)
    assert not new.doi.isin(base.doi).any()
    assert new.record_id.is_unique and new.icorr_A_cm2.gt(0).all()
    expanded = pd.concat([base, new], ignore_index=True)
    expanded['RE_log1p_ppm'] = np.log1p(expanded.RE_wt_pct * 10000)
    assert expanded.record_id.is_unique
    expanded.to_csv(ROOT / 'data/training_RE_expanded.csv', index=False, encoding='utf-8-sig')
    # First predict the unseen publication using only prior publications.
    tr = np.arange(len(base))
    test = np.arange(len(base), len(expanded))
    chosen, inner = choose(base, tr)
    external = np.full(len(expanded), np.nan)
    external[test] = fit_predict(chosen, expanded, tr, test)
    base_pred, base_folds = evaluate(base, np.unique(base.reference_id))
    augmented_pred, augmented_folds = evaluate(expanded, np.unique(expanded.reference_id))
    same_rows = augmented_pred[:len(base)]
    before, after = metrics(base, base_pred), metrics(base, same_rows)
    result = dict(features=FEATURES, candidates=CANDIDATES,
        original=before, augmented_on_same_original_rows=after,
        expanded_all_rows=metrics(expanded, augmented_pred),
        first_new_publication_prediction=dict(model=chosen, inner_scores=inner,
            metrics=metrics(expanded, external)),
        delta_log10_R2=after['log10_R2'] - before['log10_R2'],
        delta_log10_MAE=after['log10_MAE'] - before['log10_MAE'],
        base_folds=base_folds, augmented_folds=augmented_folds,
        validation='Nested leave-one-publication-out; identical original test rows for addition comparison; train-only imputation and model choice; inverse-publication-size weights.',
        missingness_policy='Eight fixed inputs retained for an explicitly exploratory sensitivity analysis. Folds exceeding the historical 20% limit are flagged, not represented as policy-compliant.',
        promoted=False, status='Exploratory; outer results now observed; new independent publications required for confirmation.',
        source_pdf_sha256='3643ec6c00596386c143259a72e0cb2eafb37d50bd7d87fe22c2f75c918d8fdf')
    (OUT/'benchmark.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf8')
    records = expanded[['record_id', 'reference_id', 'doi', 'icorr_A_cm2']].copy()
    records['observed_log10'] = np.log10(expanded.icorr_A_cm2)
    records['augmented_heldout_log10'] = augmented_pred
    records['augmented_heldout_icorr_A_cm2'] = 10 ** augmented_pred
    records['original_heldout_log10'] = np.r_[base_pred, np.full(len(new), np.nan)]
    records['first_external_log10'] = external
    records.to_csv(OUT/'heldout_predictions.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame([
        dict(scenario='original_47', **before),
        dict(scenario='augmented_on_same_original_47', **after),
        dict(scenario='expanded_all_50', **result['expanded_all_rows']),
        dict(scenario='new_publication_external_3', **result['first_new_publication_prediction']['metrics'])
    ]).to_csv(OUT/'comparison.csv', index=False, encoding='utf-8-sig')
    print(json.dumps({k: result[k] for k in ['original', 'augmented_on_same_original_rows', 'expanded_all_rows', 'delta_log10_R2', 'delta_log10_MAE']}, indent=2))


if __name__ == '__main__':
    main()
