"""Apply the locally trained phase model to a CSV using the training transforms."""
from pathlib import Path
import argparse
import joblib
import pandas as pd
import numpy as np
import optimize_process_features as o

ROOT=Path(__file__).resolve().parent
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('input_csv')
    parser.add_argument('output_csv')
    parser.add_argument('--model',default=str(ROOT/'results/phase_tuning/model_all326.joblib'))
    args=parser.parse_args()
    bundle=joblib.load(args.model)
    data=pd.read_csv(args.input_csv)
    for col in bundle['input_columns']:
        if col not in data:data[col]=np.nan
    x=o.transform(data)
    for name,_,columns in bundle['model'].named_steps['prep'].transformers_:
        if name=='cat':
            for col in columns:x[col]=x[col].astype(object)
        if name=='num':
            for col in columns:x[col]=pd.to_numeric(x[col],errors='coerce')
    result=data.copy();result['predicted_log10_icorr']=bundle['model'].predict(x)
    result['predicted_icorr_A_cm2']=10**result.predicted_log10_icorr
    result.to_csv(args.output_csv,index=False,encoding='utf-8-sig')
    print(f'Predicted {len(result)} rows -> {args.output_csv}')
if __name__=='__main__':main()
