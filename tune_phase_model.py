"""Phase-aware SVR search, nested paper validation, then fit every available row."""
from pathlib import Path
import json
import hashlib
import joblib
import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import optimize_process_features as o

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/RE_auxiliary_reference'

def main():
    global OUT
    advanced='--algorithms' in sys.argv
    if advanced:OUT=ROOT/'results/phase_algorithms_v2'
    literature='--literature' in sys.argv
    if literature:OUT=ROOT/'results/RE_literature_20261014'
    svr_only='--svr-only' in sys.argv
    if svr_only:OUT=ROOT/('results/RE_SVR_literature_20261014' if literature else 'results/RE_SVR_baseline_20261014')
    OUT.mkdir(exist_ok=True)
    input_path=ROOT/'data/training_RE_collection_latest.csv'
    collection='--collection' in sys.argv
    if collection:
        input_path=ROOT/'data/training_RE_collection_latest.csv'
        OUT=ROOT/'results/RE_100_collection/SVR'
        OUT.mkdir(parents=True,exist_ok=True)
    expanded='--expanded' in sys.argv
    latest='--latest' in sys.argv
    focused='--focused' in sys.argv
    transfer='--transfer' in sys.argv
    if expanded or latest:
        input_path=ROOT/('data/training_RE_with_auxiliary.csv' if expanded else 'data/training_RE_collection_latest.csv')
        OUT=ROOT/('results/RE_auxiliary_expanded' if expanded else 'results/RE_auxiliary_reference')
        OUT.mkdir(parents=True,exist_ok=True)
    if focused:input_path=ROOT/'data/training_RE_auxiliary_max.csv'
    if transfer:input_path=ROOT/'data/training_RE_auxiliary_max.csv'
    if '--input' in sys.argv:input_path=Path(sys.argv[sys.argv.index('--input')+1])
    d=pd.read_csv(input_path)
    n,c=o.FEATURE_SETS['drop_P_S_add_manufacturing']
    features={'phase':(n,c+['phase_reported','microstructure_constituents_reported']),
              'phase_grain':(n+['matrix_grain_size_um','polygonal_ferrite_fraction_pct'],c+['phase_reported','microstructure_constituents_reported'])}
    improve='--improve' in sys.argv
    accuracy='--accuracy' in sys.argv
    semantic='--semantic' in sys.argv
    if semantic:
        import semantic_features as sf
        d=sf.prepare(d)
        features.update(sf.FEATURES)
        for nums,cats in sf.FEATURES.values():
            for col in nums+cats:
                if col not in d:d[col]=np.nan
    if improve:
        OUT=ROOT/'results/RE_prediction_improvement'
        OUT.mkdir(parents=True,exist_ok=True)
        phase=features['phase'][1]
        features.update({
            'alloy_phase':(n+o.ALLOY_N,phase+['material_class']),
            'alloy_environment':(n+o.ALLOY_N+o.ENV_N,phase+['material_class']+o.ENV_C),
            'alloy_environment_protocol':(n+o.ALLOY_N+o.ENV_N+o.PROTOCOL_N,phase+['material_class']+o.ENV_C+o.PROTOCOL_C),
        })
        if accuracy:
            OUT=ROOT/'results/RE_prediction_accuracy'
            OUT.mkdir(parents=True,exist_ok=True)
            features['alloy_core']=(o.LEAN+o.ALLOY_N,o.CAT+['material_class'])
        if expanded or latest:
            OUT=ROOT/('results/RE_auxiliary_expanded' if expanded else 'results/RE_auxiliary_matched_baseline')
            OUT.mkdir(parents=True,exist_ok=True)
        if focused:
            OUT=ROOT/'results/RE_auxiliary_focused'
            OUT.mkdir(parents=True,exist_ok=True)
            features['phase_marine_aux']=features['phase']
        if transfer:
            OUT=ROOT/'results/RE_auxiliary_transfer'
            OUT.mkdir(parents=True,exist_ok=True)
            features['phase_marine_aux']=features['phase']
        needed={col for nums,cats in features.values() for col in nums+cats}
        missing=sorted(needed-set(d.columns))
        if missing:d=pd.concat([d,pd.DataFrame(np.nan,index=d.index,columns=missing)],axis=1)
    if '--output' in sys.argv:
        OUT=Path(sys.argv[sys.argv.index('--output')+1]);OUT.mkdir(parents=True,exist_ok=True)
    o.FEATURE_SETS.update(features);x=o.transform(d)
    y=np.log10(d.icorr_A_cm2.to_numpy(float));g=d.reference_id.to_numpy()
    positive=d.RE_wt_pct.gt(0).to_numpy();idx=np.flatnonzero(positive)
    papers=np.unique(g[positive])
    configs=[(f,C,gamma,w) for f in features for C in [.1,.3,1.,3.] for gamma in ['scale',.01] for w in [.25,1.]]
    if advanced:
        configs=[(f,leaf,kind,.25) for f in features for leaf in [3,8] for kind in ['extra_trees','random_forest']] + [('phase',C,'scale',w) for C in [.1,.3] for w in [.25,1.]]
    if svr_only:configs=[('phase',C,'scale',w) for C in [.1,.3] for w in [.25,1.]]
    if semantic:configs=[(f,C,'scale',w) for f in sf.FEATURES for C in [.1,.3] for w in [.1,1.]]
    if '--conservative' in sys.argv:configs=[('semantic_matrix',C,'scale',w) for C in [.1,.3] for w in [.1,1.]]
    if '--refine' in sys.argv:configs=[('semantic_matrix',C,'scale',w) for C in [.1,.3] for w in [.1,1.]]+ [('semantic_matrix',C,gamma,.1) for C in [1.,3.] for gamma in ['scale',.01]]
    if improve:
        configs=[('phase',C,'scale',w) for C in [.1,.3] for w in [.25,1.]]+[(f,C,'scale',w) for f in ['alloy_phase','alloy_environment','alloy_environment_protocol'] for C in [.3,1.] for w in [.25,1.]]
        if accuracy:
            configs=[('phase',C,'scale',w) for C in [.1,.3] for w in [.25,1.]]+[('phase',.03,'scale',1.),('phase',1.,'scale',1.)]+[(f,.1,'scale',w) for f in ['phase','alloy_core'] for w in [0.,.1]]+[('alloy_core',C,'scale',1.) for C in [.1,.3]]
        if focused:
            configs=[('phase',C,'scale',w) for C in [.1,.3] for w in [0.,1.]]+ [('phase_marine_aux',C,'scale',w) for C in [.1,.3,1.] for w in [.1,.25,1.]]
        if transfer:
            configs=[('phase',C,'scale',w) for C in [.1,.3] for w in [.25,1.]]+[(f,alpha,'ridge',w) for f in ['phase','phase_marine_aux'] for alpha in [10.,100.] for w in [.1,1.]]
    def fit(tr,config):
        f,C,gamma,w=config;num,cat=features[f]
        if f=='phase_marine_aux':tr=tr[np.isin(g[tr],papers) | d.electrolyte_family.iloc[tr].eq('NaCl').to_numpy()]
        if w==0:tr=tr[np.isin(g[tr],papers)]
        num=[v for v in num if x.iloc[tr][v].nunique()>1];cat=[v for v in cat if x.iloc[tr][v].nunique()>1]
        prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),('scale',StandardScaler())]),num),
            ('cat',Pipeline([('impute',SimpleImputer(strategy='constant',fill_value='unreported')),('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cat)])
        est=ExtraTreesRegressor(n_estimators=64,min_samples_leaf=int(C),random_state=42,n_jobs=1) if gamma=='extra_trees' else (RandomForestRegressor(n_estimators=64,min_samples_leaf=int(C),random_state=42,n_jobs=1) if gamma=='random_forest' else SVR(C=C,gamma=gamma,epsilon=.1))
        if gamma=='ridge':est=Ridge(alpha=C)
        model=Pipeline([('prep',prep),('svr',est)])
        counts=pd.Series(g[tr]).value_counts();weights=np.array([1/counts[v] for v in g[tr]])*np.where(np.isin(g[tr],papers),1.,w);weights/=weights.mean()
        model.fit(x.iloc[tr],y[tr],svr__sample_weight=weights);return model
    # Every outer/inner split excludes all rows of the held paper, including controls.
    fixed={conf:np.full(len(d),np.nan) for conf in configs}
    nested=np.full(len(d),np.nan);selection=[];cache={}
    def predict(excluded,conf,test):
        key=(tuple(sorted(excluded)),conf)
        if key not in cache:
            tr=np.flatnonzero(~np.isin(g,list(excluded)))
            cache[key]=fit(tr,conf).predict(x)
        return cache[key][test]
    for held in papers:
        test=np.flatnonzero(positive&(g==held));val=np.flatnonzero(positive&(g!=held));scores={}
        for conf in configs:
            fixed[conf][test]=predict({held},conf,test)
            p=np.full(len(d),np.nan)
            for ih in papers[papers!=held]:
                it=np.flatnonzero(positive&(g==ih));p[it]=predict({held,ih},conf,it)
            scores[conf]=o.metrics(y[val],p[val],g[val])
        best=min(configs,key=lambda v:(scores[v]['log10_MAE'],-scores[v]['log10_R2']) if accuracy else (-scores[v]['log10_R2'],scores[v]['paper_balanced_log10_MAE']))
        nested[test]=fixed[best][test];selection.append(dict(held=held,config=best))
        print(held,best,flush=True)
    rows=[dict(feature=conf[0],C=conf[1],gamma=conf[2],auxiliary_weight=conf[3],**o.metrics(y[idx],fixed[conf][idx],g[idx])) for conf in configs]
    table=pd.DataFrame(rows).sort_values('log10_R2',ascending=False);table.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    best=min(configs,key=lambda conf:o.metrics(y[idx],fixed[conf][idx],g[idx])['log10_MAE']) if accuracy else max(configs,key=lambda conf:o.metrics(y[idx],fixed[conf][idx],g[idx])['log10_R2'])
    model=fit(np.arange(len(d)),best)
    joblib.dump(dict(model=model,config=best,input_columns=list(x.columns),feature_sets=features,semantic_transform=semantic,preprocessing='optimize_process_features.transform plus phase feature declarations'),OUT/f'model_all{len(d)}.joblib')
    pred=model.predict(x)
    report=dict(rows_trained=len(d),RE_positive_rows=len(idx),search_configs=len(configs),best_exploratory_config=best,
        selection_objective='log10_MAE' if accuracy else 'log10_R2',
        best_exploratory=o.metrics(y[idx],fixed[best][idx],g[idx]),nested=o.metrics(y[idx],nested[idx],g[idx]),
        training_all_rows=o.metrics(y,pred,g),training_RE_only=o.metrics(y[idx],pred[idx],g[idx]),
        selections=selection,input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
        actual_final_training_rows=int(np.isin(g,papers).sum() if best[3]==0 else (len(d) if best[0]!='phase_marine_aux' else (np.isin(g,papers)|d.electrolyte_family.eq('NaCl').to_numpy()).sum())),
        caution='Training R2 is resubstitution, not generalization. Search is exploratory after previous experiments; nested splits reduce selection leakage but do not replace independent new-paper validation.')
    if literature:
        frozen=fixed[('phase',.1,'scale',1.)]
        oldidx=idx[idx<326];newidx=idx[idx>=326]
        report['frozen_previous_SVR_historical41']=o.metrics(y[oldidx],frozen[oldidx],g[oldidx])
        report['nested_historical41']=o.metrics(y[oldidx],nested[oldidx],g[oldidx])
        report['new_paper_frozen_external']=o.metrics(y[newidx],predict(set(g[newidx]),('phase',.1,'scale',1.),newidx),g[newidx])
        pd.DataFrame(dict(record_id=d.record_id,reference_id=g,observed_log10=y,frozen_previous_SVR_prediction=frozen)).to_csv(OUT/'frozen_predictions.csv',index=False,encoding='utf-8-sig')
    if collection:
        oldidx=idx[idx<329];newidx=idx[idx>=329]
        report['nested_historical43']=o.metrics(y[oldidx],nested[oldidx],g[oldidx])
        report['new_paper_frozen_external']=o.metrics(y[newidx],predict(set(g[newidx]),('phase',.1,'scale',1.),newidx),g[newidx])
    (OUT/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame(dict(record_id=d.record_id,reference_id=g,observed_log10=y,nested_prediction=nested,best_fixed_prediction=fixed[best],all_rows_training_prediction=pred)).to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    print(json.dumps({k:v for k,v in report.items() if k!='selections'},indent=2),flush=True)
if __name__=='__main__':main()
