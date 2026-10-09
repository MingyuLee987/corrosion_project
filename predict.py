import pathlib,argparse,csv,math
import numpy as np,joblib
from train import canon,writecsv
HERE=pathlib.Path(__file__).resolve().parent
def run(input_path, output_path, model_path):
 bundle=joblib.load(model_path);ns,cs=bundle['columns']
 with input_path.open(encoding='utf-8-sig',newline='') as f:source=list(csv.DictReader(f))
 if not source:raise ValueError('prediction_input.csv에 예측할 행을 입력하세요. 빈칸은 결측값이며 0과 다릅니다.')
 rows=[]
 for raw in source:
  r={k:(None if v=='' else v) for k,v in raw.items()}
  for k in set(ns)|{'RE_wt_pct','pre_exposure_h'}:
   if r.get(k) is not None:r[k]=float(r[k])
  if r.get('pre_exposure_h') is not None:
   if r['pre_exposure_h']<0:raise ValueError('pre_exposure_h must be nonnegative')
   r['log_pre_exposure_h']=math.log1p(r['pre_exposure_h'])
  if r.get('RE_wt_pct') is not None and r['RE_wt_pct']<0:raise ValueError('RE_wt_pct must be nonnegative')
  rows.append(canon(r))
 X=np.array([[float(r[k]) if r.get(k) is not None else np.nan for k in ns]+[str(r.get(k) or 'not_reported') for k in cs] for r in rows],dtype=object)
 predictions=bundle['model'].predict(X)
 writecsv(output_path,[dict(record_id=r.get('record_id',''),prediction_log10_A_cm2=float(p),prediction_A_cm2=float(10**np.clip(p,-30,1)),prediction_uA_cm2=float(1e6*10**np.clip(p,-30,1)),missing_model_inputs='|'.join(k for k in ns+cs if r.get(k) is None),status='탐색용: 새로운 논문에 대한 예측 성능 미확보') for r,p in zip(rows,predictions)])
 print('예측 저장:',output_path)
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--input',type=pathlib.Path,default=HERE/'prediction_input.csv');a.add_argument('--out',type=pathlib.Path,default=HERE/'results/new_predictions.csv');a.add_argument('--model',type=pathlib.Path,default=HERE/'models/exploratory_model.joblib');s=a.parse_args();run(s.input,s.out,s.model)
