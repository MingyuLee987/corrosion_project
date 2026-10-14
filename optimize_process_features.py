"""Nested publication exclusion for process feature blocks and auxiliary weighting."""
from pathlib import Path
import json
import hashlib
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
NUM=['C_wt_pct','Si_wt_pct','Mn_wt_pct','P_wt_pct','S_wt_pct','RE_log1p_ppm']
CAT=['RE_element_cat','electrolyte_family']
from composition_values import apply_composition_values

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/process_optimization'
PROCESS_N = ['final_heat_treatment_temp_C', 'final_heat_treatment_time_min']
PROCESS_C = ['processing_state_cat', 'melting_route_cat', 'cooling_method_cat']
ENV_N = ['Cl_mol_L', 'NaCl_wt_pct', 'H2SO4_mol_L', 'pH', 'temperature_C', 'log_pre_exposure_verified']
ENV_C = ['surface_state_cat', 'gas_condition_cat']
PROTOCOL_N = ['TP_scan_rate_mV_s', 'OCP_stabilization_verified_s']
PROTOCOL_C = ['TP_reference_electrode_cat']
ALLOY_N = ['Cr_wt_pct', 'Ni_wt_pct', 'Mo_wt_pct', 'Cu_wt_pct']
LEAN = [c for c in NUM if c not in ['P_wt_pct', 'S_wt_pct']]
FEATURE_SETS = {
    'core8': (NUM, CAT),
    'drop_P_S': (LEAN, CAT),
    'add_manufacturing': (NUM+PROCESS_N, CAT+PROCESS_C),
    'drop_P_S_add_manufacturing': (LEAN+PROCESS_N, CAT+PROCESS_C),
    'add_environment': (NUM+ENV_N, CAT+ENV_C),
    'add_protocol': (NUM+PROTOCOL_N, CAT+PROTOCOL_C),
    'add_alloy': (NUM+ALLOY_N, CAT+['material_class']),
    'add_process_environment': (NUM+PROCESS_N+ENV_N, CAT+PROCESS_C+ENV_C),
    'drop_P_S_add_process_environment': (LEAN+PROCESS_N+ENV_N, CAT+PROCESS_C+ENV_C),
}
MODELS = ['svr03', 'svr1', 'ridge10']
WEIGHTS = [0., .25, 1.]
CONFIGS = [(f, m, w) for f in FEATURE_SETS for m in MODELS for w in WEIGHTS] + [('core8', 'median', 1.)]


def estimator(name):
    if name == 'median':
        return DummyRegressor(strategy='median')
    return Ridge(alpha=10.) if name == 'ridge10' else SVR(C=.3 if name == 'svr03' else 1., epsilon=.1)


def transform(data):
    data = apply_composition_values(data)
    x = data.copy()
    x['RE_log1p_ppm'] = np.log1p(data.RE_wt_pct*10000)
    # Explicitly verified pre-exposure clock; do not mix in generic exposure_time_h.
    x['log_pre_exposure_verified'] = np.log1p(pd.to_numeric(data.pre_exposure_h, errors='coerce'))
    a = pd.to_numeric(data.OCP_stabilization_s, errors='coerce')
    b = pd.to_numeric(data.TP_OCP_stabilization_time_s, errors='coerce')
    assert not (a.notna() & b.notna() & a.ne(b)).any(), 'Conflicting OCP clocks'
    x['OCP_stabilization_verified_s'] = a.combine_first(b)
    for numeric, categorical in FEATURE_SETS.values():
        for c in numeric:
            x[c] = pd.to_numeric(x[c], errors='coerce')
        for c in categorical:
            x[c] = x[c].replace(['unknown', 'unreported', 'not_reported', ''], np.nan)
    x['RE_element_cat'] = x.RE_element_cat.replace({'La_Ce':'La+Ce'})
    x['material_class'] = x.material_class.replace({'low-alloy steel':'low alloy steel'})
    return x


def metrics(y, prediction, groups):
    error = np.abs(y-prediction)
    return dict(log10_R2=float(r2_score(y,prediction)), log10_MAE=float(error.mean()),
        paper_balanced_log10_MAE=float(np.mean([error[groups==p].mean() for p in np.unique(groups)])))


