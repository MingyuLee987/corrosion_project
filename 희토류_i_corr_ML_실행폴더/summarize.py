import pathlib,sys,json,csv,math,collections
OUT=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import numpy as np,joblib
from train import Experiment,make,configs,weights,writecsv
from prepare import prepare
from predict import run
data=prepare();evaluation=json.loads((OUT/'results/retrained/results.json').read_text(encoding='utf8'))
snapshot=json.loads((OUT/'data/input_snapshot.json').read_text(encoding='utf8'))
fresh=Experiment(data['enriched_rows']);prior=Experiment(snapshot['enriched_rows'])
assert list(fresh.g)==list(prior.g)
np.testing.assert_allclose(fresh.raw,prior.raw,rtol=0,atol=0)
for panel,(ns,cs) in __import__('train').panels.items():
 np.testing.assert_allclose(fresh.X[panel][:,:len(ns)].astype(float),prior.X[panel][:,:len(ns)].astype(float),rtol=0,atol=0,equal_nan=True)
 assert np.array_equal(fresh.X[panel][:,len(ns):],prior.X[panel][:,len(ns):])
config=configs[evaluation['best_diagnostic']['id']]
model=make(config);model.fit(fresh.X[config['panel']],fresh.y,model__sample_weight=weights(fresh.g))
bundle=dict(model=model,config=config,columns=__import__('train').panels[config['panel']],target='log10_A_cm2',scope='Exploratory; nested whole-paper validation weak; final fit all 51 rows')
joblib.dump(bundle,OUT/'models/exploratory_model.joblib')
ns,cs=bundle['columns'];fields=list(dict.fromkeys(['record_id']+ns+cs+['RE_wt_pct','processing_state_cat','surface_state_cat','pre_exposure_h']))
inputfile=OUT/'results/prediction_input_all51.csv'
with inputfile.open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data['enriched_rows'])
run(inputfile,OUT/'results/fitted_model_predictions_all51.csv',OUT/'models/exploratory_model.joblib')
with (OUT/'results/retrained/new51_nested18_predictions.csv').open(encoding='utf-8-sig',newline='') as f:pred=list(csv.DictReader(f))
groups=collections.defaultdict(list)
for r in pred:groups[r['reference_id']].append(r)
paper=[];combined=[]
for ref,rs in groups.items():
 errs=[abs(float(r['actual_log'])-float(r['predicted_log'])) for r in rs]
 paper.append(dict(reference_id=ref,rows=len(rs),log_MAE=float(np.mean(errs)),median_error_factor=float(10**np.median(errs)),max_error_factor=float(10**max(errs))))
 for r in rs:
  combined.append(dict(record_id=r['record_id'],reference_id=ref,actual_uA_cm2=1e6*float(r['actual_icorr']),held_out_prediction_uA_cm2=1e6*float(r['predicted_icorr']),actual_log10_A_cm2=float(r['actual_log']),held_out_prediction_log10_A_cm2=float(r['predicted_log']),absolute_error_factor=10**abs(float(r['actual_log'])-float(r['predicted_log'])),scope='same paper fully excluded from training; nested model selection'))
paper.sort(key=lambda r:r['log_MAE'],reverse=True)
writecsv(OUT/'results/논문별_검증오차.csv',paper)
writecsv(OUT/'results/51행_독립논문검증_실측예측비교.csv',combined)
summary=[]
for key,label,n,p in [('old34_nested18','기존 주 데이터',34,9),('new51_nested18','확장 주 데이터',51,11),('supplementary54_nested18','명목 첨가량 보조 포함',54,12)]:
 m=evaluation[key]['metrics'];summary.append(dict(dataset=label,rows=n,papers=p,**m))
writecsv(OUT/'results/검증성능_요약.csv',summary)
mae=evaluation['new51_nested18']['metrics']['log_MAE']
median=float(np.median([r['absolute_error_factor'] for r in combined]))
audit=dict(date='2026-10-08 Asia/Seoul',fresh_actions=['current CSV preparation and validation','exact model inputs and targets compared to full nested-validation dataset: identical','final selected model freshly fitted on all 51 rows','saved model reloaded and all 51 rows predicted','reports and held-out error tables generated'],model_config=config,rows=51,papers=11,validation_results='full nested leave-one-paper-out results already executed in results/retrained; not recomputed this time because model inputs and targets are unchanged',median_heldout_error_factor=median)
(OUT/'results/이번_실행기록.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
text='''# 희토류-강재 i_corr 실제 실행 결과

2026-10-08 기준. 현재 CSV를 점검하고, 최종 모델을 51행 전체로 다시 학습한 후 저장된 모델을 불러와 51행 예측을 실행했습니다. 아래 성능은 이미 실행 완료한 **논문 단위 중첩 검증** 결과입니다. 현재 데이터의 타겟과 모든 모델 입력이 그 검증에 사용한 데이터와 동일함을 확인했습니다.

## 핵심 결과

'''
text+='| 데이터 | 행/논문 | log10 i_corr R² | 원단위 i_corr R² | log MAE |\n|---|---:|---:|---:|---:|\n'
for r in summary:text+=f'| {r["dataset"]} | {r["rows"]}/{r["papers"]} | {r["log_R2"]:.3f} | {r["raw_R2"]:.3f} | {r["log_MAE"]:.3f} |\n'
text+=f'''\n**추가 논문을 반영해도 새 논문 예측 성능은 개선되지 않았습니다.** 주 데이터 51행의 log R²는 -1.175, 원단위 R²는 -0.158입니다. R² 음수는 해당 전체 검증 타겟의 평균값을 사용하는 예측보다 제곱오차가 크다는 뜻입니다. 로그 기준과 원단위 기준 점수는 서로 다르며, 퍼센트 정확도로 읽으면 안 됩니다.

주 데이터의 논문 제외 검증에서 행별 실측·예측 차이의 중앙값은 약 **{median:.1f}배**입니다. 따라서 현재 모델을 새 합금의 정량 예측에 신뢰할 단계는 아닙니다.

## 무엇을 학습했는가

- 조성: 실제 합금의 RE와 C·Si·Mn·Cr·Ni·Mo·Cu·S·P 등. 3개 컬럼 조합을 비교합니다.
- 제조/공정: 제강·주조·열처리·냉각·압연 상태 및 표면 준비 상태.
- 부식 환경: 전해질 종류/농도, 온도, 가스 조건, 사전 노출시간.
- 타겟: A/cm²로 통일한 i_corr의 log10. 예측을 다시 A/cm²와 µA/cm²로 환산합니다.
- RE의 로그 변환이나 공정 범주 통합을 포함하는 transfer 패널도 비교합니다. 최종 선택된 모델이 실제 사용한 컬럼은 아래에 따로 표시했습니다.
- 미보고값을 실제 0으로 채우지 않습니다. 각 학습 분할 안에서 중앙값 대체와 범주 인코딩을 수행합니다.
- E_corr, Rct, Tafel 기울기, 논문 식별자는 예측 입력에 쓰지 않습니다. 논문 식별자는 분할과 가중치 계산에만 사용합니다.

## 검증 방법

3개 컬럼 조합 × 6개 모델 설정 = 18개 후보를 비교했습니다. 한 논문의 모든 행을 검증용으로 제외하고, 나머지 논문 안에서 다시 논문별 내부 검증으로 후보를 선택합니다. 제외한 논문의 타겟은 후보 선택이나 전처리 적합에 쓰지 않습니다. 모든 바깥 검증 예측을 모아 R²를 계산합니다. 행 수가 많은 논문이 학습을 지배하지 않도록 논문별 가중치를 사용합니다.

`model_comparison.csv`는 같은 검증 결과로 후보를 비교한 탐색 순위표입니다. 그 최고점을 새로운 논문의 최종 성능이라고 보고하면 안 됩니다. 전체 51행으로 최종 모델을 적합한 뒤 그 51행을 다시 예측한 결과도 독립 성능 평가가 아닙니다.

## 현재 저장 모델

'''
text+=f'- 모델: {config["model"]}, 컬럼 패널: {config["panel"]}, 후보 ID: {config["id"]}\n- 수치 입력: '+', '.join(ns)+'\n- 범주 입력: '+', '.join(cs)+'\n'
text+='\n## 논문별 오차\n\n| 제외한 논문 | 행 | log MAE | 실측·예측 차이 중앙값 |\n|---|---:|---:|---:|\n'
for r in paper:text+=f'| {r["reference_id"]} | {r["rows"]} | {r["log_MAE"]:.3f} | {r["median_error_factor"]:.1f}배 |\n'
text+='''\n서로 다른 강종·시험 용액·표면 상태를 함께 다루고 있으며, 핵심 조건이 미보고인 행도 많습니다. 추가된 행만으로 논문 사이의 차이를 학습하기에는 데이터가 부족할 가능성이 있습니다. 이는 실패 원인에 대한 해석이며, 특정 변수의 인과효과를 입증한 결과는 아닙니다.

## 저장 결과

- `51행_독립논문검증_실측예측비교.csv`: 성능 평가에 사용한 실제/예측값, µA/cm², 행별 오차 배수.
- `논문별_검증오차.csv`, `검증성능_요약.csv`: 논문별 오차와 전체 검증 점수.
- `fitted_model_predictions_all51.csv`: 이번에 다시 학습한 저장 모델의 51행 예측. 학습에 사용한 행이므로 독립 검증이 아닙니다.
- `이번_실행기록.json`: 이번 실행 및 기존 검증 결과 재사용 근거.
- `retrained/`: 이미 완료된 전체 중첩 검증 결과 및 각 분할에서 선택한 모델.

## 다음에 필요한 데이터

높은 R²를 목표로 한다면 같은 강종·전해질·표면 상태를 비교할 수 있는 독립 논문을 우선 확보해야 합니다. 실제 잔류 RE 조성, 분극 시험 조건, 수치 i_corr가 함께 보고된 데이터를 수집하고, 새 논문을 통째로 제외하는 검증을 유지해야 합니다. 현재 자료에서 수치를 크게 만들기 위해 논문이 섞이는 무작위 분할로 바꾸는 것은 새 논문 예측 성능을 입증하지 못합니다.
'''
(OUT/'results/결과_정리.md').write_text(text,encoding='utf8')
print(json.dumps(dict(selected_config=config,metrics=summary,median_factor=median,worst_papers=paper[:3]),ensure_ascii=False,indent=2))
