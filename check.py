import pathlib,json,joblib,numpy as np
from prepare import prepare
from train import Experiment,configs,make,weights
from predict import run
HERE=pathlib.Path(__file__).resolve().parent
d=prepare();e=Experiment(d['enriched_rows']);c=configs[0]
tr=np.where(e.g!=e.ids[0])[0];te=np.where(e.g==e.ids[0])[0]
assert not set(e.g[tr]) & set(e.g[te])
model=make(c);model.fit(e.X[c['panel']][tr],e.y[tr],model__sample_weight=weights(e.g[tr]))
assert np.isfinite(model.predict(e.X[c['panel']][te])).all()
run(HERE/'data/example_input.csv',HERE/'results/example_prediction.csv',HERE/'models/exploratory_model.joblib')
print('환경/데이터/논문 분리/학습/저장 모델 예측: PASS')

