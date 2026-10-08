import pathlib,json,csv,math
HERE=pathlib.Path(__file__).resolve().parent
def prepare():
 d=json.loads((HERE/'data/input_snapshot.json').read_text(encoding='utf8'))
 with (HERE/'data/training_51.csv').open(encoding='utf-8-sig',newline='') as f:raw=list(csv.DictReader(f))
 rows=[];ids=set()
 for i,r in enumerate(raw,2):
  if not r.get('record_id') or not r.get('reference_id'):raise ValueError(f'{i}행: record_id/reference_id 필요')
  if r['record_id'] in ids:raise ValueError(f'{i}행: record_id 중복')
  ids.add(r['record_id'])
  for k,v in list(r.items()):
   if v=='':r[k]=None
   else:
    try:r[k]=float(v)
    except (ValueError,TypeError):pass
  for k in ['icorr_A_cm2','RE_wt_pct']:
   v=r.get(k)
   if not isinstance(v,(int,float)) or not math.isfinite(v):raise ValueError(f'{i}행: {k} 유효 숫자 필요')
  if r['icorr_A_cm2']<=0 or r['RE_wt_pct']<0:raise ValueError(f'{i}행: 타겟 양수, RE 0 이상 필요')
  if r.get('pre_exposure_h') is not None:r['log_pre_exposure_h']=math.log1p(r['pre_exposure_h'])
  rows.append(r)
 if len(set(r['reference_id'] for r in rows))<4:raise ValueError('독립 논문이 4개 이상 필요합니다.')
 d['enriched_rows']=rows
 path=HERE/'results/current_input.json';path.parent.mkdir(exist_ok=True)
 path.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
 print(f'학습 입력: {len(rows)}행 / {len(set(r["reference_id"] for r in rows))}논문')
 return d
if __name__=='__main__':prepare()
