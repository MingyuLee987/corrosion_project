"""Use reported composition intervals as midpoint model inputs, preserving bounds."""
import re
import unicodedata
import numpy as np
import pandas as pd

NUMBER=r'(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
INTERVAL=re.compile(rf'^\s*({NUMBER})\s*(?:-|–|—|~|～|〜|至|to)\s*({NUMBER})\s*(?:wt\.?\s*%|mass\s*%|%)?\s*$',re.I)

def composition_value(raw):
    if pd.isna(raw):return np.nan,np.nan,np.nan,'missing'
    text=unicodedata.normalize('NFKC',str(raw)).strip()
    match=INTERVAL.fullmatch(text)
    if match:
        lower,upper=map(float,match.groups())
        if lower>upper:raise ValueError(f'Reversed composition interval: {raw}')
        return (lower+upper)/2,lower,upper,'reported_range_midpoint'
    try:
        value=float(text.rstrip('%').strip())
        return value,value,value,'reported_scalar'
    except ValueError:
        # Single-sided limits and unreadable source text are not two-sided ranges.
        return np.nan,np.nan,np.nan,'unresolved'

def apply_composition_values(data):
    result=data.copy()
    for column in [c for c in data if c.endswith('_wt_pct')]:
        values=[composition_value(v) for v in data[column]]
        result[column]=[v[0] for v in values]
        if any(v[3]=='reported_range_midpoint' for v in values):
            result[column+'_reported']=data[column]
            result[column+'_lower']=[v[1] for v in values]
            result[column+'_upper']=[v[2] for v in values]
            result[column+'_value_basis']=[v[3] for v in values]
    return result
