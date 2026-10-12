# 부식 문헌 기반 모델 확장 실험 — 2026-10-10

## 결과와 해석

기존 v2의 11편·33개 대조군–첨가 시편 쌍을 그대로 사용했다. 목표값, 대조군 값, 포함 논문을 변경하지 않았다. 3가지 특징 묶음과 6개 회귀 모델군·설정, 2개 기준선을 합쳐 32개 후보를 논문 단위 중첩 검증했다.

| 방법 | log10(i_corr) R² | log10 MAE | 첨가 효과 Δ R² |
|---|---:|---:|---:|
| 대조군 값 그대로 | 0.521 | 0.365 | — |
| 기존 v2 중첩 선택 | 0.653 | 0.277 | -0.233 |
| 이번 확장 중첩 선택 | 0.680 | 0.264 | -0.135 |

이번 결과는 v2보다 개선됐지만, 기존 별도 paired_benchmark.py의 R² 0.683을 넘지는 못했다. R² 0.8은 달성하지 않았다. 이번 **원 단위 i_corr R²는 0.344**다. log 점수를 원 단위 점수처럼 표시하면 안 된다. 첨가 효과 자체의 R²가 음수인 점도 남아 있다.

현재 논문들은 이전 분석에서 이미 검토한 자료다. 이번 중첩 검증은 각 분할의 시험 논문을 후보 선택에서 제외하지만, 연구 전체의 반복 탐색까지 없애지는 못한다. 따라서 결과는 탐색적 개선으로 보고하고 새로운 독립 논문에서 확인해야 한다. 새 모델을 기존 production v2 모델 파일에 덮어쓰지 않았다.

## 실제 구현

- `corrosion_research_extension.py`: compact / 기존 v2 / expanded 특징 묶음.
- expanded는 원자료에 이미 있는 Si, Mn, Mo, Cu, 분극 주사속도, 가스 조건을 추가한다. 원문에 없는 수치를 만들어 넣지 않는다.
- 모델: SVR(C=0.03/0.3/3), Ridge(alpha=1/10/100), Huber, 고정 Matern Gaussian Process, Random Forest, ExtraTrees. 기준선: 대조군 그대로, 훈련 Δ 중앙값.
- 수치 결측 대치·결측 지시자·표준화·범주 인코딩은 각각의 훈련 분할에서만 적합한다.
- SVR·Ridge·Huber·트리 모델은 논문별 총 가중치를 동일하게 한다. GPR은 sklearn에서 sample_weight를 지원하지 않아 행별 적합한다.
- 내부 논문별 MSE 평균으로 후보를 선택한다. R²와 관련된 제곱 오차를 줄이되, 시편이 많은 논문이 모델 선택을 독점하지 않게 한 설계 판단이다.
- 외부 시험 논문의 모든 첨가 시편은 학습·내부 선택에서 제외한다. 대조군 실측값은 이 예측 과제의 허용된 별도 기준값이다.
- 최종 전체 학습 후보: 기존 v2 특징 + SVR(C=0.03, epsilon=0.1). 특징을 많이 추가하는 후보가 최종 선택된 것은 아니다.
- 시험 예측과 원문 목표값을 기존 v2와 record_id로 대조하여 동일한 33행·목표값·대조군 값임을 확인했다.

## 참고 논문과 적용 범위

아래는 방법 검토에 참고한 문헌이다. 서로 다른 부식 목표·재료·분할을 사용하므로 논문 R²를 이 프로젝트의 성능으로 차용하지 않는다.

1. **Al-Nouti & Yaseen, Engineering Applications of Artificial Intelligence (2025)** — [Integration of hybrid machine learning model with differential evolution algorithm for underground Pipeline corrosion prediction](https://doi.org/10.1016/j.engappai.2025.111511). 환경과 배관 변수를 이용한 회귀·특징 선택·SVR 후보 비교. 이번에는 SVR 설정 비교를 확장했고 차분진화 알고리즘은 사용하지 않았다.
2. **Chen et al., Materials (2025)** — [An Integrated Approach Using GA-XGBoost and GMM-RegGAN for Marine Corrosion Prediction Under Small Sample Size](https://pmc.ncbi.nlm.nih.gov/articles/PMC12387397/). 소규모 해양 강재 부식에서 여러 모델과 설정 최적화·합성 데이터 접근을 검토. 이번에는 후보 모델 비교 원칙만 참고했으며 GA나 GAN을 재현하지 않았다.
3. **Subramanian et al., npj Materials Degradation (2026)** — [A supervised machine learning approach with dual GAN augmentation and Bayesian HPO for corrosion prediction of Mg-Li alloys](https://www.nature.com/articles/s41529-026-00859-9). 조성·공정·환경, 목표 로그 변환, 여러 회귀 후보·최적화를 검토. Mg-Li의 물리 기술자를 강재에 그대로 적용하지 않았고 GAN 증강도 실행하지 않았다.
4. **Dong et al., Materials & Design (2026)** — [Enhancing corrosion rate prediction under data scarcity with virtual sample generation and heterogeneous transfer learning](https://doi.org/10.1016/j.matdes.2026.116410). 재료별 제약을 둔 가상 표본·전이학습 접근. 신규 독립 자료와 적합한 source dataset이 필요한 후속 후보이며 이번 코드에는 미구현이다.
5. **Feng et al., Engineering Applications of Artificial Intelligence (2025)** — [The lead-bismuth eutectic corrosion rate prediction and composition optimization of ferritic/martensitic steels by physics-guided neural network](https://doi.org/10.1016/j.engappai.2024.109807). 조성·환경·노출 시간과 물리 과정 구분의 참고. 납–비스무트 산화층 법칙을 수용액 i_corr에 이식하지 않았다.
6. **Liu et al. (2026)** — [Machine Learning Prediction and Interpretable Analysis of Steel Corrosion in Concrete Subjected to Chloride and Sulfate](https://discovery.ucl.ac.uk/id/eprint/10228241/). 염화물·황산염 환경과 여러 모델 비교. 환경 농도를 별도로 표현하는 방향의 참고이며 콘크리트 데이터를 기존 강재 분극 데이터와 합치지 않았다.
7. **Hybrid data-driven machine learning approach for evaluating steel corrosion in concrete using electrical resistivity and documented concrete performance indicators (2025)** — [출판사 원문](https://doi.org/10.1016/j.conbuildmat.2025.142154). GPR·SVM·트리 등 후보와 입력 조합을 비교. 이번 GPR 후보 도입의 참고이며 전기저항·콘크리트 지표는 현재 스키마에 없음.

원문 품질 점검에서는 [La 첨가 탄소강 연구](https://pmc.ncbi.nlm.nih.gov/articles/PMC11433059/)의 목표값 위치와 기존 표의 품질 표지를 대조했다. 큰 오차의 La24Q355는 정확한 Table 2 값으로 이미 정정된 기록이 있어 이번에 제외하지 않았다. 논문의 첨가강 i_corr와 대조군 비율이 다른 자료보다 크게 감소하는 사례라서 추가 원문 조건 확인이 우선이다.

## 실행과 파일

```powershell
.venv/Scripts/python.exe -m pip install -r requirements_extension.txt
.venv/Scripts/python.exe corrosion_research_extension.py
.venv/Scripts/python.exe corrosion_research_extension.py --predict v2_prediction_input.csv
```

결과는 `results/research_extension/benchmark.json`, `heldout_predictions.csv`, `exploratory_model.joblib`, `new_predictions.csv`에 저장한다. 새로운 입력에는 동일 조건의 대조군 실측값이 반드시 필요하다. 기존 v2 입력을 사용할 수 있으며 확장 특징은 미입력 시 결측으로 처리한다.

## 다음 개선의 우선순위

1. 논문별 비교 가능한 조건과 대조군 매칭을 원문에서 재확인한다. 결측 농도·온도·노출시간·조성에 대한 근거를 보완한다.
2. 새 독립 논문을 확보하여 모델·특징·설정 선택에 쓰지 않는 보류 집합으로 고정한다. 기존 후보 CSV의 실험 조건 충돌은 해결 전까지 보류한다.
3. 서로 다른 강재·환경을 무작정 섞지 말고, 사용 범위를 사전 정의한 후 충분한 독립 논문을 확보한다. 성능이 나쁜 기존 논문만 사후 제거하지 않는다.
4. 합성 데이터와 전이학습은 적합한 외부 source 데이터·물리적 제약이 준비된 뒤 훈련 분할 안에서만 시험한다. 현재 11편의 합성 행 증가를 독립 실험 증가로 해석하지 않는다.
