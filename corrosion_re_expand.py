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


def annotate_casting(data):
    """Keep reported processes separate from explicitly hypothetical candidates.

    Composition constrains compatibility, not the actual manufacturing history.
    Neither candidate processes nor missing pouring temperatures enter FEATURES.
    """
    data = data.copy()
    for column in ['casting_temperature_C', 'casting_method_reported_cat']:
        if column not in data:
            data[column] = np.nan if column.endswith('_C') else ''
    for i, row in data.iterrows():
        existing = str(row.get('casting_method_cat', '')).strip()
        previous = str(row.get('casting_method_status', ''))
        reported = str(row.get('casting_method_reported_cat', '')).strip()
        if reported in ('', 'nan') and existing not in ('', 'nan', 'unknown', 'not_reported') and previous != 'inferred':
            reported = existing
        data.loc[i, 'casting_method_reported_cat'] = reported if reported != 'nan' else ''
        if reported not in ('', 'nan'):
            method, status, confidence = reported, 'reported_in_existing_dataset', 'not_assessed'
            note = 'Existing source-extracted casting method retained; no new inference.'
        else:
            route = str(row.get('melting_route_cat', '')).lower()
            # A compatibility hypothesis, not a reconstruction of the paper.
            method = 'ingot_casting'
            if 'bof' in route:
                method = 'continuous_casting'
            status, confidence = 'inferred', 'low'
            composition = '; '.join(f'{c}={row[c]:g}' for c in ['C_wt_pct', 'Si_wt_pct', 'Mn_wt_pct', 'Cr_wt_pct', 'Ni_wt_pct', 'RE_wt_pct'] if pd.notna(row.get(c)))
            note = (f'INFERRED, not source-reported: {method} is a possible route for '
                    f'{row.get("material_class", "steel")} ({composition}); '
                    f'existing melting route={row.get("melting_route_cat", "not_reported")}. '
                    'Composition does not uniquely determine casting method; ingot, continuous, '
                    'or strip casting alternatives remain possible. Low confidence; not a measured predictor.')
        data.loc[i, 'casting_method_cat'] = method
        data.loc[i, 'casting_method_status'] = status
        data.loc[i, 'casting_method_confidence'] = confidence
        data.loc[i, 'casting_process_remarks'] = note
        temperature = pd.to_numeric(pd.Series([row.get('casting_temperature_C')]), errors='coerce').iloc[0]
        data.loc[i, 'casting_temperature_status'] = 'reported_in_existing_dataset' if pd.notna(temperature) else 'not_reported'
        data.loc[i, 'casting_temperature_remarks'] = ('Pouring temperature; separate from melting, mold, reheating and heat-treatment temperatures.' if pd.notna(temperature) else 'No verified pouring temperature available. Composition alone is insufficient for a numerical estimate; heat-treatment temperature was not substituted.')
        remarks = '' if pd.isna(row.get('remarks')) else str(row.get('remarks', ''))
        marker = ' [CASTING] '
        data.loc[i, 'remarks'] = remarks.split(marker)[0] + marker + note
    return data


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


def non_re_additions():
    """Source-transcribed, uncoated/no-inhibitor PDP rows; no measured targets inferred."""
    rows = []

    def add(reference, doi, title, sample, current, potential, chemistry, locator, **conditions):
        row = dict(record_id=f'NONRE_{reference}_{sample}', reference_id=reference,
            doi=doi, source_url=f'https://doi.org/{doi}', paper_title=title,
            sample_id=sample, RE_wt_pct=0., RE_element_cat='none', RE_log1p_ppm=0.,
            RE_content_status='no_intentional_RE_reported; zero is operational, not residual assay',
            icorr_A_cm2=current * 1e-6, raw_icorr_value=current,
            raw_icorr_unit='uA/cm2', log_icorr=np.log10(current * 1e-6),
            Ecorr_V=potential / 1000, target_method='PDP_Tafel',
            target_locator=locator, material_class='low alloy steel',
            electrolyte_family='NaCl', NaCl_wt_pct=3.5,
            surface_state_cat='polished', inhibitor_status='none', coating_status='none',
            dataset_scope='non_RE_auxiliary_PDP', origin='non_RE_PDP_2026_10_12',
            composition_basis='source_reported', primary_eligible=True,
            icorr_status='reported', remarks='Source-transcribed auxiliary row; no intentional RE addition reported. Residual RE was not assayed; operational RE=0 is not a measured zero.')
        row.update(chemistry)
        row.update(conditions)
        rows.append(row)

    for sample, current, potential, cycles in [('AR', 2.3645, -539, 0),
            ('Q1', 4.8089, -629, 1), ('Q2', 5.0893, -664, 2), ('Q3', 11.086, -692, 3)]:
        add('E34_2013', '10.1155/2013/587514',
            'Influence of Heat Treatment on the Corrosion of Microalloyed Steel in Sodium Chloride Solution',
            sample, current, potential,
            dict(C_wt_pct=.060, Si_wt_pct=.020, Mn_wt_pct=.57, P_wt_pct=.012,
                 S_wt_pct=.008, Al_wt_pct=.055, Nb_wt_pct=.012),
            'Sec2.1 chemistry; Sec2.6.2 exposed area=1cm2; Table4 current in uA divided by 1cm2',
            alloy_system='E34_Nb', temperature_C=25., TP_scan_rate_mV_s=1.,
            TP_reference_electrode_cat='Ag/AgCl', pre_exposure_h=1.,
            gas_condition_cat='air_equilibrated', hydrodynamic_condition_cat='stagnant',
            processing_state_cat='hot_rolled' if cycles == 0 else 'quenched',
            final_heat_treatment_temp_C=np.nan if cycles == 0 else 600.,
            final_heat_treatment_time_min=np.nan if cycles == 0 else 45.,
            cooling_method_cat='not_reported' if cycles == 0 else 'water',
            processing_sequence=f'hot_rolled; repeated_600C_45min_water_quench_cycles={cycles}')

    chemistry = dict(C_wt_pct=.16, Si_wt_pct=.03, Mn_wt_pct=.4,
                     P_wt_pct=.11, S_wt_pct=.1, Cu_wt_pct=.05)
    for medium, samples in [
        ('NaCl', [('OA',4,28,-556),('QA',4,24,-539),('TA',4,33,-537),
                  ('OB',7,21,-648),('QB',7,16,-601),('TB',7,25,-491),
                  ('OC',10,18,-579),('QC',10,16,-475),('TC',10,20,-617)]),
        ('Na2SO4', [('OD',4,52,-543),('QD',4,41,-543),('TD',4,55,-550),
                    ('OE',7,39,-502),('QE',7,39,-408),('TE',7,42,-546),
                    ('OF',10,20,-509),('QF',10,16,-467),('TF',10,21,-547)])]:
        for sample, ph, current, potential in samples:
            state = {'O':'as_received', 'Q':'quenched', 'T':'quenched_tempered'}[sample[0]]
            add('S275_2022', '10.3390/met12081386',
                'Effect of Microstructure on Electrochemical Properties of the EN S275 Mild Steel under Chlorine-Rich and Chlorine-Free Media at Different pHs',
                sample, current, potential, chemistry,
                'Table1 chemistry; Table6 NaCl/Table7 Na2SO4; Sec2.1,2.2,2.5; HT2 only',
                alloy_system='S275', electrolyte_family=medium,
                NaCl_wt_pct=3.5 if medium == 'NaCl' else np.nan,
                Na2SO4_wt_pct=3.5 if medium == 'Na2SO4' else np.nan,
                pH=ph, pH_adjustment='acetic_acid_or_NaOH',
                TP_scan_rate_mV_s=1., TP_OCP_stabilization_time_s=3600.,
                TP_reference_electrode_cat='Ag/AgCl', processing_state_cat=state,
                final_heat_treatment_temp_C={'O':np.nan,'Q':1000.,'T':600.}[sample[0]],
                final_heat_treatment_time_min={'O':np.nan,'Q':45.,'T':30.}[sample[0]],
                cooling_method_cat={'O':'not_reported','Q':'water','T':'air'}[sample[0]],
                quality_flags='P=0.11 and S=0.1 are Table1 printed values, not silently corrected. Test temperature unreported. pH-adjusted electrolyte; different from plain NaCl.',
                processing_sequence={'O':'as_received','Q':'1000C45min;water_quench',
                    'T':'1000C45min;water_quench;600C30min;air_cool'}[sample[0]])

    for speed, current, potential in [(1,38.85,-204.73),(3,55.48,-86.07),
            (5,89.25,-120.97),(7,93.25,-597.27),(9,142.83,-177.61),(11,106.90,-628.08)]:
        add('Flow304L_2022', '10.3390/met12081324',
            'On-Line Measurement and Characterization of Electrochemical Corrosion of 304L Stainless Steel Pipe Wall in High-Speed Cl-Containing Solution',
            f'v{speed}', current, potential,
            dict(C_wt_pct=.024, Si_wt_pct=.53, Cr_wt_pct=18.38, Mn_wt_pct=1.074,
                 P_wt_pct=.034, S_wt_pct=.004, Ni_wt_pct=8.13),
            'Table1 chemistry; Table2 only (not corrosion-rate back-calculation); Sec2.1-2.3',
            material_class='austenitic stainless steel', alloy_system='304L',
            NaCl_wt_pct=4., temperature_C=30., flow_velocity_m_s=float(speed),
            hydrodynamic_condition_cat='flowing', TP_reference_electrode_cat='Ag/AgCl',
            quality_flags='High flow velocities; auxiliary compatibility is model-selected, not assumed.')

    add('BlankMild_2024', '10.3390/coatings14060652',
        'Performance of a Composite Inhibitor on Mild Steel in NaCl Solution: Imidazoline, Sodium Molybdate, and Sodium Dodecylbenzenesulfonate',
        'blank',73.29,np.nan,
        dict(C_wt_pct=.2, Si_wt_pct=.3, Mn_wt_pct=.7, S_wt_pct=.042, P_wt_pct=.042),
        'Table1 chemistry; Table3 row1 as_received/no inhibitors; Sec2.4',
        alloy_system='mild_steel', TP_scan_rate_mV_s=1.,
        TP_reference_electrode_cat='Ag/AgCl', gas_condition_cat='N2_deaerated',
        temperature_description='ambient; numerical temperature not reported',
        quality_flags='Only inhibitor-free control included. N2 pretreatment of solution=24h, not specimen exposure.')
    add('BareLCS_2021','10.1002/nano.202000051',
        'Synergistic effect of graphene polyindole nanocomposite for enhanced corrosion protection of aqueous coating in 3.5% NaCl solution for low carbon steel',
        'bare',127.,-597.,
        dict(C_wt_pct=.049, Mn_wt_pct=.723, Al_wt_pct=.01, Cr_wt_pct=.051,
             P_wt_pct=.028, Mo_wt_pct=.081, V_wt_pct=.033),
        'Sec2.1 chemistry; Sec2.5 conditions; Table3 bare_steel current=1.27e-4 A/cm2',
        alloy_system='low_carbon_steel', hydrodynamic_condition_cat='stagnant',
        temperature_description='room temperature; numerical temperature not reported',
        quality_flags='Bare control only; coated rows excluded. Si and S unreported, left missing.')
    for sample, carbon, silicon, manganese, nickel, copper, currents, potentials in [
        ('Cu075',.020,.19,.74,3.56,.75,[6.8039,3.4432,2.9724,2.6072,2.4321],[-.84813,-.62675,-.5745,-.56769,-.54944]),
        ('Cu125',.019,.21,.73,3.55,1.25,[3.7376,2.8933,2.7949,2.3413,1.2609],[-.85544,-.54231,-.5460,-.50237,-.40243]),
        ('Cu215',.022,.20,.73,3.55,2.15,[2.079,1.8313,1.8802,1.7967,1.1080],[-.86328,-.53506,-.5050,-.47775,-.42931])]:
        for hours,current,potential in zip([24,48,72,96,120],currents,potentials):
            add('CuNiWeather_2025','10.3390/ma18153496',
                'Effect of Cu Content on Corrosion Resistance of 3.5%Ni Weathering Steel in Marine Atmosphere of South China Sea',
                f'{sample}_{hours}h',current*100,potential*1000,
                dict(C_wt_pct=carbon,Si_wt_pct=silicon,Mn_wt_pct=manganese,
                     Ni_wt_pct=nickel,Cu_wt_pct=copper,Mo_wt_pct=.25,
                     P_upper_bound_wt_pct=.01,S_upper_bound_wt_pct=.001),
                'Table1 nominal chemistry; Table4 current units 1e-4 A/cm2; Table5 potential; Sec2.5 PDP at35C',
                alloy_system='3.5Ni_Cu_weathering_steel',composition_basis='source_nominal_table1',
                NaCl_wt_pct=2.,temperature_C=35.,TP_scan_rate_mV_s=1.,
                TP_reference_electrode_cat='SCE',surface_state_cat='rusted',
                processing_state_cat='hot_rolled',pre_exposure_h=float(hours),
                pre_exposure_type='cyclic_wet_dry',pre_exposure_cycle_min=60.,
                pre_exposure_wet_min=12.,
                processing_sequence='1200C60min_homogenize;1150C_start_roll;800C_finish_roll',
                quality_flags='P<0.01 and S<0.001 are upper bounds, not measured exact values. Pre-exposure43-45C and drying70+/-10C differ from PDP35C; not conflated.')
    for days,current,potential in [(5,7.8,-810),(10,27.,-450),(15,39.,-940)]:
        add('L245_2025','10.3390/ma18102270',
            'Investigation of the Corrosion Behavior of L245 Steel in 3.5 wt.% NaCl Solution with Varying Concentrations of Na2S2O3',
            f'NaCl_{days}d',current,potential,
            dict(C_wt_pct=.202,Si_wt_pct=.254,Mn_wt_pct=.396,P_wt_pct=.0189,
                 S_wt_pct=.0139,Cr_wt_pct=.023,Ni_wt_pct=.020,Mo_wt_pct=.018,
                 V_wt_pct=.0014,Nb_wt_pct=.003,Ti_wt_pct=.0025),
            'Table1 L245 measured row; Table2 pureNaCl rows (0 thiosulfate); Sec2.2,2.4',
            alloy_system='L245',pre_exposure_h=float(days*24),pre_exposure_type='immersion',
            pre_exposure_temperature_C=40.,surface_state_cat='rusted',
            TP_scan_rate_mV_s=1.,TP_reference_electrode_cat='SCE',
            TP_OCP_stabilization_time_s=2000.,
            quality_flags='Table2 5d=7.8e-6 used; narrative says7.9e-6. PDP temperature not independently stated; pre-immersion40C not copied into PDP temperature. Only pureNaCl rows included; thiosulfate rows excluded.')
    result = pd.DataFrame(rows)
    assert len(result) == 48 and result.record_id.is_unique
    assert result.icorr_A_cm2.gt(0).all() and result.RE_wt_pct.eq(0).all()
    result = annotate_casting(result)
    public = pd.read_csv(ROOT/'data/non_RE_rebar_PDP216.csv', encoding='utf-8-sig')
    assert len(public) == 216 and public.record_id.is_unique
    assert public.reference_id.eq('RebarLi2023').all()
    assert public.groupby('independent_condition_id').size().eq(3).all()
    assert np.allclose(public.icorr_A_cm2, public.raw_icorr_value * 1e-6)
    assert not public.record_id.isin(result.record_id).any()
    public['casting_method_cat'] = 'not_reported'
    public['casting_method_status'] = 'not_reported'
    public['casting_process_remarks'] = 'Exact chemistry not verified; no casting-process inference made.'
    return pd.concat([result, public], ignore_index=True)


CONTEXT_NUM = NUM + ['Cr_wt_pct', 'Ni_wt_pct', 'Mo_wt_pct', 'Cu_wt_pct',
    'Al_wt_pct', 'NaCl_wt_pct', 'H2SO4_mol_L', 'pH', 'temperature_C',
    'TP_scan_rate_mV_s', 'flow_velocity_m_s', 'water_cement_ratio',
    'relative_humidity_pct', 'chloride_by_cement_wt_pct']
CONTEXT_CAT = CAT + ['material_class', 'processing_state_cat', 'gas_condition_cat',
    'hydrodynamic_condition_cat', 'surface_state_cat', 'test_specimen_type',
    'carbonation_state', 'moisture_condition']


def auxiliary_benchmark(target, historical_pred):
    """Nested target-paper validation. Auxiliary weights selected on RE-positive rows only."""
    auxiliary = non_re_additions()
    target = target.copy()
    target['dataset_scope'] = 'RE_studies_and_reported_controls'
    combined = pd.concat([target, auxiliary], ignore_index=True)
    is_rebar = combined.reference_id.eq('RebarLi2023')
    # Explicitly mark the ordinary aqueous experiments as outside mortar settings.
    combined.loc[~is_rebar, 'test_specimen_type'] = 'aqueous_steel_specimen'
    combined.loc[~is_rebar, 'carbonation_state'] = 'not_applicable'
    combined.loc[~is_rebar, 'moisture_condition'] = 'aqueous_test'
    combined.to_csv(ROOT/'data/training_PDP_augmented.csv', index=False, encoding='utf-8-sig')
    groups = combined.reference_id.to_numpy()
    y = np.log10(combined.icorr_A_cm2.to_numpy(float))
    n = len(target)
    is_aux = np.arange(len(combined)) >= n
    positive = combined.RE_wt_pct.to_numpy(float) > 0
    papers = np.unique(groups[:n])
    frames = {}
    for kind, numeric, categorical in [('core',NUM,CAT),('context',CONTEXT_NUM,CONTEXT_CAT)]:
        x = combined.reindex(columns=numeric + categorical).copy()
        x['RE_log1p_ppm'] = np.log1p(combined.RE_wt_pct * 10000)
        for c in numeric:
            x[c] = pd.to_numeric(x[c], errors='coerce')
        for c in categorical:
            x[c] = x[c].replace(['unknown','not_reported','unreported',''],np.nan)
        x['RE_element_cat'] = x.RE_element_cat.replace({'La_Ce':'La+Ce'})
        if kind == 'context':
            x['material_class'] = x.material_class.replace({'low-alloy steel':'low alloy steel'})
        frames[kind] = (x, numeric, categorical)
    frames['core98'] = frames['core']

    cache = {}
    def predict(spec, excluded):
        key = (spec, tuple(sorted(excluded)))
        if key in cache:
            return cache[key]
        kind, name, strength = spec
        train = np.flatnonzero(~np.isin(groups,list(excluded)) & (~is_aux if strength == 0 else True))
        if kind == 'core98':
            train = train[groups[train] != 'RebarLi2023']
        assert not np.isin(groups[train], list(excluded)).any()
        test = np.flatnonzero(np.isin(groups,list(excluded)) & ~is_aux)
        x, numeric, categorical = frames[kind]
        categorical = [c for c in categorical if x.iloc[train][c].notna().any()]
        # No source identifiers, inferred casting, Ecorr or corrosion-derived measures.
        pipe = model(name, numeric, categorical)
        counts = pd.Series(groups[train]).value_counts()
        weights = np.array([1/counts[g] for g in groups[train]])
        weights *= np.where(is_aux[train], strength, 1.)
        weights /= weights.mean()
        if name == 'median':
            pipe.fit(x.iloc[train], y[train])
        else:
            pipe.fit(x.iloc[train], y[train], reg__sample_weight=weights)
        p = np.full(len(combined), np.nan)
        p[test] = pipe.predict(x.iloc[test])
        cache[key] = p
        return p

    specs = [('core','median',0.)] + [(kind,name,strength)
        for kind in ['core','context'] for name in ['ridge10','svr03']
        for strength in [0.,.25,1.]]
    predictions, fold_rows = {}, []
    for scenario, allowed in [('target_only_fixed_core',[s for s in specs if s[0]=='core' and s[2] == 0]),
                               ('target_only_nested',[s for s in specs if s[2] == 0]),
                               ('auxiliary_nested', specs),
                               ('previous_98_fixed_core',[('core98','median',0.),('core98','ridge10',1.),('core98','svr03',1.)]),
                               ('auxiliary_fixed_core',[('core','median',0.),('core','ridge10',1.),('core','svr03',1.)])]:
        pred = np.full(n, np.nan)
        for held in papers:
            scores = {}
            for spec in allowed:
                errors = []
                for inner in papers[papers != held]:
                    p = predict(spec, {held,inner})
                    mask = (groups == inner) & positive & ~is_aux
                    errors.append(mean_absolute_error(y[mask],p[mask]))
                scores[spec] = float(np.mean(errors))
            selected = min(allowed,key=lambda s:(scores[s],allowed.index(s)))
            p = predict(selected,{held})
            test = np.flatnonzero(groups[:n] == held)
            pred[test] = p[test]
            x = frames[selected[0]][0]
            train = ~np.isin(groups,[held]) & (~is_aux if selected[2] == 0 else True)
            if selected[0] == 'core98':
                train &= groups != 'RebarLi2023'
            fold_rows.append(dict(scenario=scenario, held_out_paper=held,
                feature_set=selected[0], model=selected[1], auxiliary_weight=selected[2],
                inner_RE_positive_paper_MAE=scores[selected],
                input_missing_fraction=x.loc[train].isna().mean().to_dict(),
                inner_scores={str(s):v for s,v in scores.items()}))
            print('nonRE',scenario,held,selected,flush=True)
        predictions[scenario] = pred
    predictions['historical_fixed8'] = historical_pred
    summaries = []
    for scenario,p in predictions.items():
        for scope,mask in [('all_RE_study_rows',np.ones(n,dtype=bool)),
                            ('RE_positive_only',positive[:n])]:
            summaries.append(dict(scenario=scenario,scope=scope,**metrics(target.loc[mask],p[mask])))
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT/'non_RE_comparison.csv',index=False,encoding='utf-8-sig')
    records = target[['record_id','reference_id','RE_wt_pct','icorr_A_cm2']].copy()
    records['observed_log10'] = y[:n]
    for scenario,p in predictions.items():
        records[scenario+'_log10'] = p
    records.to_csv(OUT/'non_RE_heldout_predictions.csv',index=False,encoding='utf-8-sig')
    result = dict(auxiliary_rows=len(auxiliary),auxiliary_papers=int(auxiliary.reference_id.nunique()),
        combined_rows=len(combined),target_RE_positive_rows=int(positive[:n].sum()),
        new_public_dataset=dict(rows=216, independent_conditions=72,
            experimental_replicates_per_condition=3,source_papers=1,
            composition_status='Exact steel chemistry not verified in public appendix; numeric chemistry left missing and imputed within training folds only.',
            environment_status='144 mortar-embedded measurements and 72 simulated pore-solution measurements; distinct electrolyte families, not plain NaCl.',
            dataset_doi='10.5281/zenodo.10150202',source_paper_doi='10.1016/j.conbuildmat.2023.134160',
            source_sha256=str(auxiliary.loc[auxiliary.reference_id.eq('RebarLi2023'),'source_sha256'].iloc[0]),
            weighting='All 216 records are ONE paper; inverse-paper-size weights prevent count from giving 216 independent-paper equivalents.'),
        comparison=summaries,folds=fold_rows,
        validation='Outer held-out RE publication is absent from training and every inner selection fold. Inner mean RE-positive publication MAE selects features/model/auxiliary weight. Auxiliary papers are training-only; no claimed validation on those papers. Zero weight is allowed. All observed outer results are exploratory.',
        context_numeric=CONTEXT_NUM,context_categorical=CONTEXT_CAT,
        exclusions=[dict(doi='10.14456/jmmm.2014.9',reason='Table2 labels current A/cm2 but prints 14.48-22.60; corrosion-rate consistency suggests microamps, not confirmed. Four rows not used.'),
            dict(doi='10.3390/ma17235979',reason='Verified composition but no tabulated PDP icorr recovered; no invented targets.'),
            dict(doi='10.3390/polym16050669',reason='Composition bounds/typical values rather than exact chemistry; not used in this batch.')],
        promoted=False)
    # Paired bootstrap unit is a publication, not repeated rows of the same steel.
    errors = {s:np.array([np.mean(np.abs(y[:n][(groups[:n]==p)&positive[:n]]-
        prediction[(groups[:n]==p)&positive[:n]])) for p in papers])
        for s,prediction in predictions.items()}
    indices = np.random.default_rng(42).integers(0,len(papers),size=(10000,len(papers)))
    result['paired_RE_positive_paper_MAE_deltas'] = {}
    for scenario,baseline in [('auxiliary_fixed_core','target_only_fixed_core'),
                              ('auxiliary_fixed_core','previous_98_fixed_core'),
                              ('auxiliary_nested','target_only_nested'),
                              ('target_only_nested','historical_fixed8')]:
        delta = errors[scenario]-errors[baseline]
        interval = np.quantile(delta[indices].mean(axis=1),[.025,.975])
        result['paired_RE_positive_paper_MAE_deltas'][f'{scenario}_minus_{baseline}'] = dict(
            delta=float(delta.mean()),bootstrap_95_percent_interval=interval.tolist(),
            improving_papers=int((delta<0).sum()),total_papers=len(papers))
    (OUT/'non_RE_benchmark.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(summary.to_string(index=False),flush=True)
    return result


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
    expanded = annotate_casting(expanded)
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
    auxiliary_benchmark(expanded, augmented_pred)
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
