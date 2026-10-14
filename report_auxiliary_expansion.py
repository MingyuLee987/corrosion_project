from pathlib import Path
import json,shutil,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def main():
 paths={'335_reference':'RE_auxiliary_reference','335_matched_MAE':'RE_auxiliary_matched_baseline',
        '401_all_auxiliary_MAE':'RE_auxiliary_expanded','412_environment_focused':'RE_auxiliary_focused',
        '412_SVR_Ridge_transfer':'RE_auxiliary_transfer'}
 rows=[];reports={};ids=None
 for name,path in paths.items():
  folder=ROOT/'results'/path;v=json.loads((folder/'validation.json').read_text());reports[name]=v
  p=pd.read_csv(folder/'predictions.csv').dropna(subset=['nested_prediction'])
  assert len(p)==47
  if ids is not None:assert set(p.record_id)==ids
  ids=set(p.record_id);e=np.abs(p.observed_log10-p.nested_prediction)
  row=dict(experiment=name,dataset_rows=v['rows_trained'],RE_test_rows=len(p),R2=v['nested']['log10_R2'],
   MAE=float(e.mean()),median_factor=float(10**e.median()))
  for k in [2,3,5,10]:row[f'hits_{k}']=int((e<=np.log10(k)).sum());row[f'percent_{k}']=100*row[f'hits_{k}']/len(p)
  rows.append(row)
 out=ROOT/'results/RE_auxiliary_final';out.mkdir(exist_ok=True)
 pd.DataFrame(rows).to_csv(out/'matched_comparison.csv',index=False,encoding='utf-8-sig')
 baseline=rows[0]
 focused=min(rows[-2:],key=lambda r:r['MAE'])
 improved=focused['MAE']<baseline['MAE'] and focused['R2']>baseline['R2']
 selected=focused['experiment'] if improved else '335_reference'
 src=ROOT/'results'/paths[selected]/f"model_all{reports[selected]['rows_trained']}.joblib"
 shutil.copy2(src,out/'model.joblib')
 selection=dict(selected_experiment=selected,expanded_model_improved=improved,
  model_sha256=hashlib.sha256((out/'model.joblib').read_bytes()).hexdigest(),
  note='Selection compares repeated exploratory experiments; independent external validation still required. Dataset size differs from fitted subset for environment-focused candidates.')
 (out/'model_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
 lines=['# 일반강 보조 자료 확대와 희토류 예측 검증','',
  '수집·통합 자료를 335→412행으로 늘렸다. 기존 수집 일반강 66행을 새로 통합하고 이번에 42CrMo4 4행, WAAM2209 3행, AISI1010 4행을 새로 전사하여 총 77행을 더했다. 새로 찾은 77행의 희토류 논문이라는 의미는 아니다. 최신 통합 자료는 `data/training_RE_auxiliary_max.csv`이다.','',
  '추가 11행은 단위·시편 조건을 확인했다. 42CrMo4는 GDOES 조성 및 열처리·XRD 조직, WAAM2209는 원재료 와이어 조성 및 ferrite/austenite 조직, AISI1010은 C만 보고된 조성과 PDP25°C·사전노출 온도를 구분해 기록했다. WAAM의 본문/초록 solution 시간 불일치는 결측으로 두었다. 일반강의 RE=0은 의도적 첨가 미보고를 뜻하며 잔류 RE 실측 0을 뜻하지 않는다.','',
  '## 같은 희토류 47행의 중첩 논문 검증','',
  '| 실험 | 전체 자료 | log10 R² | log10 MAE | 2배 적중률 | 3배 적중률 | 5배 적중률 | 10배 적중률 |',
  '|---|---:|---:|---:|---:|---:|---:|---:|']
 for r in rows:
  cells=' | '.join(f"{r[f'hits_{k}']}/47 ({r[f'percent_{k}']:.1f}%)" for k in [2,3,5,10])
  lines.append(f"| {r['experiment']} | {r['dataset_rows']} | {r['R2']:.4f} | {r['MAE']:.4f} | {cells} |")
 lines+=['', '335_matched_MAE와401_all_auxiliary_MAE는 동일 12개 후보·동일 내부 MAE 선택 기준으로 데이터 추가의 영향을 비교했다. 412_environment_focused는 13개 후보와 내부 R² 선택을 사용하는 별도 탐색으로, 차이를 자료 추가만의 효과로 해석하지 않는다. NaCl 보조 자료 우선 후보도 모든 희토류 연구의 대조군을 보존하고, 검증 논문은 대조군까지 통째로 제외한다.','',
  f"전체 후보 중 권장 저장 모델은 `{selected}`이다. {'확대 모델의 중첩 R²와 MAE가 기준보다 개선되었다.' if improved else '확대 모델은 기준의 R²와 MAE를 동시에 개선하지 못하여 기존 기준 모델을 유지했다.'}",
  '', '412_SVR_Ridge_transfer는 기존 SVR과 Ridge(alpha10/100), 전체 보조 자료와 NaCl 보조 자료 우선, 보조 가중치0.1/1을 12개 후보로 비교한 별도 탐색이다. 다른 후보 탐색과 동일 희토류 47행을 평가하였다.',
  '', '자료 수를 늘리는 것만으로 정확도 상승을 보장할 수 없다. 독립 시편 수, 일반강/스테인리스강 차이, 시험환경 차이 및 결측 조성 때문에 관련 없는 자료의 추가가 오차를 늘릴 수 있다. RE 함량·조성·공정·기지 조직과 시험 환경을 함께 확인한 자료를 우선 수집하고 논문별 균형 가중치를 유지한다. 이번 결과는 반복 탐색 결과이며 독립된 신규 문헌의 검증을 대체하지 않는다.',
  '', '## 사용','',
  '`python predict_phase_model.py input.csv output.csv --model results/RE_auxiliary_final/model.joblib`',
  '', '재현: `python build_auxiliary_training.py`, `python add_ordinary_steel_batch.py`, `python tune_phase_model.py --latest --svr-only`, `python tune_phase_model.py --latest --improve --accuracy`, `python tune_phase_model.py --expanded --improve --accuracy`, `python tune_phase_model.py --improve --focused`, `python tune_phase_model.py --improve --transfer`, `python report_auxiliary_expansion.py`.']
 (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 readme=ROOT/'README.md';t=readme.read_text(encoding='utf-8-sig')
 marker='## 일반강 보조 학습을 포함한 최신 분석'
 note=f"{marker}\n\n최신 통합 자료는 **412행**이며, 일반강 보조 자료 **77행**을 추가했다. 희토류 성능은 동일 **47행**을 논문 단위로 검증했다. 확대 모델의 중첩 R²는 **{focused['R2']:.4f}**, 2배/3배 적중률은 **{focused['percent_2']:.1f}%/{focused['percent_3']:.1f}%**다. {'확대 모델을 후보로 저장했다.' if improved else '종합 성능을 개선하지 못하여 기준 모델을 유지했다.'} [전체 비교·자료 출처·모델 사용법](results/RE_auxiliary_final/report.md)을 참고한다. 이하 수치는 이전 실험 기록이다.\n\n"
 if marker in t:
  start=t.index(marker);end=t.index('\n\n',t.index('\n\n',start)+2)+2
  t=t[:start]+note+t[end:]
 else:
  pos=t.index('\n\n')+2;t=t[:pos]+note+t[pos:]
 readme.write_text(t,encoding='utf-8-sig')
 print(json.dumps(rows,indent=2));print(json.dumps(selection,indent=2))

if __name__=='__main__':main()
