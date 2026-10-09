"""Reproducible literature-level nested validation. Never split one paper across train/test.
Usage: python train_oct08.py --data input_snapshot.json --out results
Targets are reported A/cm2; performance is evaluated in log10 and raw units.
"""
import pathlib,sys,json,csv,math,collections,argparse,itertools
HERE=pathlib.Path(__file__).resolve().parent
if (HERE/'ml_packages').exists():sys.path.insert(0,str(HERE/'ml_packages'))
import numpy as np,joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.ensemble import ExtraTreesRegressor,RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error

def writecsv(path,rs):
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
def weights(groups):
 count=collections.Counter(groups);return np.array([len(groups)/len(count)/count[g] for g in groups])
def canon(r):
 r=r.copy();r['RE_log1p_ppm']=math.log1p(10000*r['RE_wt_pct']) if r.get('RE_wt_pct') is not None else None
 a=str(r.get('processing_state_cat') or '').lower()
 r['process_family']='quenched_tempered' if 'temper' in a else 'quenched' if 'quench' in a else 'solution_treated' if 'solution' in a else 'annealed' if 'anneal' in a else 'hot_rolled' if 'roll' in a else 'as_cast' if 'cast' in a else 'not_reported'
 a=str(r.get('surface_state_cat') or '').lower()
 r['surface_family']='rusted' if ('rust' in a or 'pre_expos' in a) else 'polished' if ('polish' in a or 'ground' in a) else 'not_reported'
 return r
procN=['final_heat_treatment_temp_C','final_heat_treatment_time_min'];procC=['melting_route_cat','casting_method_cat','cooling_method_cat','processing_state_cat','surface_state_cat']
envN=['temperature_C','NaCl_wt_pct','Cl_mol_L','H2SO4_mol_L','NaHSO3_mol_L','log_pre_exposure_h'];envC=['electrolyte_family','gas_condition_cat']
panels={
 'compact':(['RE_wt_pct','C_wt_pct','Cr_wt_pct','Ni_wt_pct']+procN+envN,['RE_element_cat']+procC+envC),
 'full':(['RE_wt_pct','C_wt_pct','Si_wt_pct','Mn_wt_pct','Cr_wt_pct','Ni_wt_pct','Mo_wt_pct','Cu_wt_pct','S_wt_pct','P_wt_pct']+procN+envN,['RE_element_cat','material_class']+procC+envC),
 'transfer':(['RE_log1p_ppm','C_wt_pct','Si_wt_pct','Mn_wt_pct','Cr_wt_pct','Ni_wt_pct','Mo_wt_pct','Cu_wt_pct','S_wt_pct']+procN+envN,['RE_element_cat','process_family','surface_family']+envC)
}
configs=[]
for p in panels:
 for name,params in [
  ('GB',dict(n_estimators=100,max_depth=1,learning_rate=.05,loss='huber',random_state=42)),
  ('GB',dict(n_estimators=100,max_depth=2,learning_rate=.05,loss='huber',random_state=42)),
  ('ExtraTrees',dict(n_estimators=80,max_depth=None,min_samples_leaf=1,random_state=42,n_jobs=1)),
  ('RF',dict(n_estimators=80,max_depth=3,min_samples_leaf=2,random_state=42,n_jobs=1)),
  ('Ridge',dict(alpha=100)),('SVR',dict(C=1,gamma=.01))
 ]:configs.append(dict(id=len(configs),panel=p,model=name,params=params))
def make(c):
 ns,cs=panels[c['panel']]
 pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True)),('scale',StandardScaler())]),list(range(len(ns)))),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),list(range(len(ns),len(ns)+len(cs))))],sparse_threshold=0)
 cls={'GB':GradientBoostingRegressor,'ExtraTrees':ExtraTreesRegressor,'RF':RandomForestRegressor,'Ridge':Ridge,'SVR':SVR}[c['model']]
 return Pipeline([('pre',pre),('model',cls(**c['params']))])
def metrics(actual,pred):
 return dict(log_R2=float(r2_score(np.log10(actual),pred)),raw_R2=float(r2_score(actual,10**np.clip(pred,-30,1))),log_MAE=float(mean_absolute_error(np.log10(actual),pred)))
class Experiment:
 def __init__(self,rows):
  self.rows=[canon(r) for r in rows];self.g=np.array([r['reference_id'] for r in rows]);self.raw=np.array([r['icorr_A_cm2'] for r in rows]);self.y=np.log10(self.raw);self.ids=sorted(set(self.g));self.cache={}
  self.X={p:np.array([[float(r[k]) if r.get(k) is not None else np.nan for k in ns]+[str(r.get(k) or 'not_reported') for k in cs] for r in self.rows],dtype=object) for p,(ns,cs) in panels.items()}
 def fit(self,c,tr,te):
  key=(c['id'],tuple(tr),tuple(te))
  if key not in self.cache:
   model=make(c);model.fit(self.X[c['panel']][tr],self.y[tr],model__sample_weight=weights(self.g[tr]));self.cache[key]=model.predict(self.X[c['panel']][te])
  return self.cache[key]
 def logo(self,c,pool=None):
  pool=self.ids if pool is None else pool;ix=np.where(np.isin(self.g,pool))[0];pred=np.full(len(self.rows),np.nan)
  for p in pool:
   tr=np.where(np.isin(self.g,[q for q in pool if q!=p]))[0];te=np.where(self.g==p)[0];assert not set(self.g[tr])&set(self.g[te]);pred[te]=self.fit(c,tr,te)
  return metrics(self.raw[ix],pred[ix]),pred
 def nested(self,label):
  pred=np.empty(len(self.rows));folds=[]
  for held in self.ids:
   pool=[q for q in self.ids if q!=held];scores=[]
   for c in configs:score,_=self.logo(c,pool);scores.append((score['log_R2'],-c['id']))
   score,negative_id=max(scores);c=configs[-negative_id];tr=np.where(self.g!=held)[0];te=np.where(self.g==held)[0]
   pred[te]=self.fit(c,tr,te);folds.append(dict(held_out_paper=held,selected_id=c['id'],selected_model=c['model'],selected_panel=c['panel'],inner_log_R2=score,outer_log_MAE=float(mean_absolute_error(self.y[te],pred[te])),train_papers='|'.join(pool)))
   print(label,held,'selected',c['id'],'outerMAE',round(folds[-1]['outer_log_MAE'],3),flush=True)
  return dict(metrics=metrics(self.raw,pred),folds=folds,predictions=self.predrows(pred))
 def predrows(self,pred):
  return [dict(record_id=r['record_id'],reference_id=r['reference_id'],actual_icorr=float(self.raw[i]),predicted_icorr=float(10**np.clip(pred[i],-30,1)),actual_log=float(self.y[i]),predicted_log=float(pred[i])) for i,r in enumerate(self.rows)]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',type=pathlib.Path,required=True);ap.add_argument('--out',type=pathlib.Path,required=True);args=ap.parse_args();OUT=args.out;OUT.mkdir(parents=True,exist_ok=True);d=json.loads(args.data.read_text(encoding='utf8'));old=Experiment(d['original_rows']);new=Experiment(d['enriched_rows'])
 result=dict(rows=len(new.rows),papers=len(new.ids),configs=configs,panels=panels,protocol='18 candidates, log target; nested leave-one-paper-out; inner pooledlogR2; all preprocessing fit ontraining only; paperweights; no targetderived inputs; retrospective experiment')
 # Fixed before augmentation: old bestpaperGB compact. Old shared9paper tests remain identical.
 fixed=configs[0];before,pbefore=old.logo(fixed);after,pafter=new.logo(fixed)
 shared=np.where(np.isin(new.g,old.ids))[0]
 result['fixed_same9paper_benchmark']=dict(before=before,after=metrics(new.raw[shared],pafter[shared]),config=fixed,scope='Retrospective same34testrecords; each heldpaper excluded. Both before/after use revised compact panel including NaHSO3; differs from the earlier original pipeline; not prospective external validation')
 # Freeze original models trained only34 rows and evaluate new two papers before augmentation.
 frozen={}
 for name in ['interpaper','withinpaper']:
  path=HERE/'baseline_models'/f'{name}_exploratory_model.joblib'
  if not path.exists():continue
  saved=joblib.load(path);ns,cs=saved['columns'];added=[r for r in new.rows if r['reference_id'] not in old.ids];X=np.array([[float(r[k]) if r.get(k) is not None else np.nan for k in ns]+[str(r.get(k) or 'not_reported') for k in cs] for r in added],dtype=object);v=saved['model'].predict(X);actual=np.array([r['icorr_A_cm2'] for r in added]);frozen[name]=dict(metrics=metrics(actual,v),predictions=[dict(record_id=r['record_id'],reference_id=r['reference_id'],actual_icorr=float(actual[i]),predicted_icorr=float(10**v[i]),actual_log=float(np.log10(actual[i])),predicted_log=float(v[i])) for i,r in enumerate(added)])
 result['frozen_original_new17']=frozen
 result['old34_nested18']=old.nested('old34');(OUT/'partial_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 result['new51_nested18']=new.nested('new51');ranking=[]
 for c in configs:
  ms,pred=new.logo(c);ranking.append(dict(**c,**ms))
 result['ranking']=ranking;best=max(ranking,key=lambda r:r['log_R2']);result['best_diagnostic']=best
 c=configs[best['id']];model=make(c);model.fit(new.X[c['panel']],new.y,model__sample_weight=weights(new.g));joblib.dump(dict(model=model,config=c,columns=panels[c['panel']],target='log10_A_cm2',scope='Exploratory; nestedwholepapermetricsinresults'),OUT/'exploratory_model.joblib')
 # Supplemental nominal-RE records tested separately with the same nested selection.
 if d.get('supplementary_rows'):
  supplemental=Experiment(d['enriched_rows']+d['supplementary_rows']);result['supplementary54_nested18']=supplemental.nested('supp54')
 (OUT/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 for name in ['old34_nested18','new51_nested18','supplementary54_nested18']:
  if name in result:writecsv(OUT/(name+'_predictions.csv'),result[name]['predictions']);writecsv(OUT/(name+'_folds.csv'),result[name]['folds'])
 writecsv(OUT/'model_comparison.csv',[{**{k:r[k] for k in ['id','panel','model','log_R2','raw_R2','log_MAE']},'params':json.dumps(r['params'])} for r in ranking])
 print(json.dumps({k:v['metrics'] for k,v in result.items() if isinstance(v,dict) and 'metrics' in v},indent=2),flush=True)
if __name__=='__main__':main()
