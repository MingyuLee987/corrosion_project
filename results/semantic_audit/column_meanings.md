# 컬럼의 물리적 의미와 검증 개선 설계

## 이번에 바꾼 접근

행 수보다 조성·시험 환경·표면 상태의 비교 가능성을 먼저 점검했다. 모든 희토류 47행을 기존과 동일하게 논문 단위로 제외해 평가하며, 점수가 낮은 논문이나 행을 제거하지 않는다. 연구 식별자·DOI·저널·발행 연도·원문 위치는 모델 입력으로 사용하지 않는다.

## 모델에 사용할 수 있는 입력

| 컬럼 | 의미·단위 | 해석 및 처리 |
|---|---|---|
| C/Si/Mn/P/S/Cr/Ni/Mo/Cu_wt_pct | 시험 강재의 원소 질량백분율 | 예: 0.01은0.01wt%, 100ppm이다. 담금질 용액이나 개재물 EDS 조성과 구분한다. 미보고는0으로 만들지 않는다. |
| RE_wt_pct | 강의 희토류 총 질량백분율 | 실측/명목/무첨가 미보고의 근거를 구분한다. 플럭스 첨가량으로 대체하지 않는다. |
| W_wt_pct 등 동시 합금 원소 | 강재의 W 등 동시 첨가량 | Ma19FSS는 W도 함께 변화한다. RE만의 효과로 귀속할 수 없다. 현재 주요 모델이 W를 쓰지 않았다는 한계가 있다. |
| La_wt_pct/Ce_wt_pct | 개별 희토류의 강재 함량 | 전체 RE와 동시에 중복 정보가 될 수 있다. 이번 모델은 총 RE와 종류를 사용한다. |
| RE_log1p_ppm | log(1+RE_wt_pct×10000) | 스케일 조정값이며 실제 조성 단위가 아니다. |
| RE_element_cat | Ce, La, Y 또는 혼합 종류 | 무첨가와 미확정 종류를 구분한다. 원소 종류 자체가 일정 개선을 보장하지 않는다. |
| high_Cr_indicator | 보고된 Cr≥10.5wt%의 표시 | 조성에 근거한 범주화다. 강종 인증이나 austenite/ferrite 실측 판정이 아니다. Cr미보고이면 결측이다. |
| RE_highCr_interaction | RE_log1p_ppm×high_Cr_indicator | 다른 Cr 영역에서 RE 효과가 달라질 가능성을 시험하는 가설 입력이다. 검증으로 효용을 판단한다. |
| electrolyte_family | 실제 PDP 용액의 종류 | NaCl, NaHSO3, H2SO4_NaCl, NaCl_CO2는 같은 환경이 아니다. 사전 침지액과 구분한다. |
| Cl_mol_L/NaCl_mol_L | 용액의 염화물/NaCl 몰농도 | NaCl만의 염화물 몰농도는1:1이다. 복합염의 NaCl만으로 총 Cl를 단정하지 않는다. |
| chloride_molar_verified | 근사 환산 표시를 제외한 Cl_mol_L 우선, 없으면 NaCl_mol_L | wt%/vol%를 밀도 없이 몰농도로 변환하지 않는다. 근사·미검증으로 명시된 기존10행의 환산값은 새 모델에서 제외한다. 그 외 값도 출처 감사로 원문 근거를 추가 확인해야 한다. |
| NaCl_wt_pct/NaCl_vol_pct | 질량/체적 기준 염 농도 | 서로 교환 불가. 이번 새 설계에서 두 값을 합치거나 같은 수치로 몰농도로 환산하지 않는다. |
| H2SO4_mol_L/NaHSO3_mol_L/pH | 산 종류·농도/수소이온 활성도 척도 | pH는 선형 농도가 아니다. 산을 중성 NaCl과 같은 방식으로 취급하지 않는다. 온도별 측정 여부도 확인해야 한다. |
| temperature_C | PDP 측정 시 용액 온도 | 열처리 온도나 사전 침지 온도가 아니다. 'room temperature'를25°C로 바꾸지 않는다. |
| pre_exposure_h | PDP 전 시편에 실제 수행한 노출시간 | 별도 질량감소 시험의 시간과 연결하지 않는다. 미보고는0시간이 아니다. |
| log_pre_exposure_verified | log(1+pre_exposure_h) | 긴 노출시간 스케일을 줄인다. 노출시간 자체의 의미는 유지한다. |
| OCP_stabilization_s/TP_OCP_stabilization_time_s/TP_OCP_stabilization_min | 분극 전 자연전위 안정화 대기시간 | 초·분을 통일하고 충돌하면 중단한다. 최소 대기시간 컬럼은 정확한 대기시간과 합치지 않는다. |
| log_OCP_s | log(1+안정화 초) | 침지시간과 별개인 측정 프로토콜 입력이다. |
| TP_scan_rate_mV_s | 분극 전위의 변화속도 | 10mV/min=1/6mV/s. 시험 반응과 측정 비정상성에 영향을 줄 수 있다. |
| TP_reference_electrode_cat | 전위 기준 전극 | SCE/AgAgCl 등. 서로 다른 기준의 전위값을 그대로 비교하지 않는다. |
| gas_condition_cat/dissolved_oxygen_mg_L | 산소·CO2 공급 조건/용존산소 | CO2 주입, 공기포화, 탈기, 미보고를 구분한다. 미보고를 공기포화로 보지 않는다. |
| surface_state_cat | PDP 시작 시 표면 상태 | 녹층·산세·밀스케일·연마 표면은 다르다. 부식 후 관찰 표면과 구분한다. |
| final_heat_treatment_temp_C/time_min | 마지막 열처리 온도·시간 | austenitizing, quench, temper 단계 전체를 대체하지 못한다. 이번 환경 중심 설계에서는 제외해 무리한 단계 혼합을 줄인다. |
| austenitizing_temperature_C/time_min | 담금질 전 오스테나이트화 처리 | 시험 온도와 다르다. 단계별 컬럼을 유지한다. |
| quench_medium_cat/cooling_method_cat | 담금질 매체/후속 냉각 방식 | CuO나Al2O3 용액은 강의 Cu/Al 함량이 아니다. |
| phase_reported | 문헌이 보고한 금속 기지 상 | 부식 생성물 α-FeOOH/γ-FeOOH와 다른 정보다. grade 이름만으로 상을 채우지 않는다. |
| microstructure_constituents_reported | ferrite, pearlite, bainite 등 조직 구성 | pearlite는 단일 결정상이 아니라 ferrite·cementite 조직이다. 긴 출처별 문자열이 논문 구분자로 작용할 수 있어 새 모델에서는 직접 사용하지 않는다. |
| matrix_family_reported | 넓은 보고 기지 범주 | ferrite/austenite/martensite/mixed/unreported로 구분. 미보고는 상의 부재가 아니다. 조직 목록의 비포괄성을 고려해야 한다. |
| matrix_grain_size_um | 금속 기지의 입자 크기 | XRD 결정립자 크기nm나개재물 크기와 다르다. 현재 희토류 전체 자료에서 결측이 많아 주요 입력으로 쓰지 않는다. |
| inclusion_number_density_mm2/inclusion_mean_area_um2 | 개재물의 면적당 수/평균 면적 | RE가 산소·황과 반응해 바꾸는 경로에 관련되지만, 대부분 미보고라 주요 입력으로 쓰지 않는다. |

Cr의 보호막 형성 및 Mo의 염화물 환경 역할은 [worldstainless의 합금 원소 설명](https://worldstainless.org/about-stainless/process-and-production/alloying-elements-and-scrap/)에서 확인했다. 보호 효과는 조성만으로 일정하지 않으며 표면·시험 환경도 중요하다. RE가 개재물과 국소부식에 미치는 경로는 [Tang et al.의 실험 연구](https://www.mdpi.com/1996-1944/12/20/3359), 기지 조직과 부식 생성물의 구분은 [UCS 연구](https://www.mdpi.com/2075-4701/13/1/66)에서 대조했다.

## 입력으로 넣으면 안 되는 정보

`icorr_A_cm2`와 `log_icorr`는 목표값이다. `Ecorr_V`·anodic/cathodic_Tafel_V_dec는 같은 분극 측정에서 구한 값이므로 조성·공정만으로 미래 전류를 예측하는 현재 용도에서 제외한다. 부식속도를 icorr로 환산한 값, EIS 저항, 부식 후 녹층의 상 비율을 동일 시편의 전류 예측 입력으로 넣지 않는다. 별도의 시험 전 측정 모델을 만들 때만 사용 가능 여부를 다시 판단한다.

출처·검증 상태·quality_flags·source_sha256·journal·record_id는 감사용이며 설명변수가 아니다. 낮은 점수의 논문을 제거하거나 대조군의 실제 icorr를 같은 논문의 검증 입력으로 제공하지 않는다.

## 현재의 병목과 실험

희토류47행에서 온도12행, 사전노출시간19행, 기지상19행만 보고돼 있다. 기존460행 SVR 오차제곱의22.7%는Ma19FSS,15.8%는UCS23에서 발생한다. UCS의 원문 표 단위는mA/cm²로 확인했으며, 점수를 위해µA로 임의 변경하지 않는다. 큰 오류는 연구 조건 차이·미보고 정보·출처 단위 신뢰성을 확인하는 우선순위를 뜻한다.

새 설계는 조성 중심, 환경 중심, 넓은 기지상·프로토콜 중심의3가지 입력 묶음과 같은 SVR 후보를 중첩 검증한다. 기존47행을 유지해 결과를 비교한다. 개선이 없으면 개선됐다고 보고하지 않는다. 반복 탐색의 점수는 독립 신규 문헌 검증을 대체하지 못한다.
