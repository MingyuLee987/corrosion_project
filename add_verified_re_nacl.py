"""Append reported RE-alloy NaCl measurements; keep historical test IDs frozen."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
import semantic_features as sf
from optimize_train_test import metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/RE_NaCl_new_literature'

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = pd.read_csv(ROOT/'data/training_RE_NaCl_expanded.csv')
    rows = []
    def add(ref, sample, doi, year, current, re, element, locator, **values):
        r = dict(record_id=f'RENEW_{ref}_{sample}', reference_id=ref, sample_id=sample,
            material_class='steel', RE_wt_pct=re, RE_element_cat=element if re else 'none',
            RE_content_status='reported_sample_composition' if re else 'no_intentional_RE_addition_reported',
            icorr_A_cm2=current, log_icorr=float(np.log10(current)), raw_icorr_value=current,
            raw_icorr_unit='A/cm2', electrolyte_family='NaCl', electrolyte_type_cat='3.5% NaCl',
            NaCl_reported_pct=3.5, doi=doi, publication_year=year, source_url=f'https://doi.org/{doi}',
            icorr_status='reported_numeric', target_method='PDP_Tafel', target_locator=locator,
            target_value_basis='reported_numeric_not_digitized', primary_eligible=True,
            composition_basis='reported_sample', source_verification='primary_full_text_checked',
            inhibitor_status='none', coating_status='none', origin='RE_NaCl_literature_20261014',
            training_role='RE_training' if re else 'matched_no_RE_control',
            TP_reference_electrode_cat='SCE', journal_quartile_filter='not_applied')
        r.update(values)
        assert not set(r)-set(base.columns), set(r)-set(base.columns)
        rows.append(r)
    # Original publisher full text publicly displayed on ResearchGate: Table 1; Sec 2, 3.4.
    title='Effects of rare earth metals on microstructure, mechanical properties, and pitting corrosion of 27% Cr hyper duplex stainless steel'
    url='https://www.researchgate.net/publication/366506894_Effects_of_rare_earth_metals_on_microstructure_mechanical_properties_and_pitting_corrosion_of_27_Cr_hyper_duplex_stainless_steel'
    for sample,chem,current in [
        ('DSS1',[.021,.44,1.15,26.9,6.44,4.58,.88,1.02,.005,.43,0.],5.88e-7),
        ('DSS3',[.022,.39,1.12,27.1,6.41,4.62,.87,.93,.002,.42,.031],9.81e-8)]:
        chemical=dict(zip(['C_wt_pct','Si_wt_pct','Mn_wt_pct','Cr_wt_pct','Ni_wt_pct',
            'Mo_wt_pct','Cu_wt_pct','Co_wt_pct','S_wt_pct','N_wt_pct'],chem[:-1]))
        add('HyperDSS_RE_2022',sample,'10.1515/rams-2022-0284',2022,current,chem[-1],'La+Ce',
            'Sec3.4 numeric icorr; Table1 chemistry; Sec2 testing', **chemical,
            paper_title=title, journal='Reviews on Advanced Materials Science', source_url=url,
            NaCl_wt_pct=3.5, NaCl_concentration_basis='wt%',
            melting_route_cat='VIM', final_heat_treatment_temp_C=1100.,final_heat_treatment_time_min=120.,
            cooling_method_cat='water_quench', processing_state_cat='solution_treated',
            phase_reported='ferrite; austenite', phase_verification_status='source_reported',
            phase_evidence_method='OM',phase_source_locator='Sec3.1 solution-treated microstructure',
            TP_scan_rate_mV_s=20/60,TP_start_potential_V=-.6,TP_end_potential_V=1.2,
            aeration_cat='deaerated', temperature_description='room temperature; numeric value unreported',
            remarks='Feed REM contains Ce and La with trace Nd/Pr; total assayed REM used, individual final RE fractions not inferred. Deaeration gas unreported. DSS2/DSS4 icorr not reported numerically; not included. As-cast sigma phase not assigned to solution-treated matrix.')
    # IOP publisher full text: Table1 chemistry, Table3 targets, Sec2 protocol.
    title='Effect of trace rare earth elements (Ce) on corrosion resistance of high strength steel used for offshore platform'
    url='https://www.researchgate.net/publication/369082815_Effect_of_trace_rare_earth_elements_Ce_on_corrosion_resistance_of_high_strength_steel_used_for_offshore_platform'
    for sample,chem,current,ecorr in [
        ('1',[.078,.211,1.56,.012,.34,.12,.05,.005,.015,0.],1.9255e-4,-.62948),
        ('2',[.066,.186,1.57,.012,.34,.12,.05,.004,.012,.0013],1.7422e-4,-.6086),
        ('3',[.072,.185,1.62,.011,.36,.12,.05,.004,.019,.0019],1.4476e-4,-.59124)]:
        chemical=dict(zip(['C_wt_pct','Si_wt_pct','Mn_wt_pct','P_wt_pct','Cr_wt_pct','Mo_wt_pct',
            'Nb_wt_pct','V_wt_pct','Ti_wt_pct'],chem[:-1]))
        add('EH420_Ce_2023',sample,'10.1088/2053-1591/acc236',2023,current,chem[-1],'Ce',
            'Table3 96h rusted-sample polarization fit; Table1 chemistry; Sec2.2', **chemical,
            Ce_wt_pct=chem[-1],S_upper_bound_wt_pct=.002,Ecorr_V=ecorr,
            paper_title=title,journal='Materials Research Express',source_url=url,alloy_system='EH420',
            NaCl_concentration_basis='percentage reported; wt/vol basis unspecified',
            melting_route_cat='VIM',processing_state_cat='quenched_tempered',rolled_thickness_mm=25.,
            austenitizing_temperature_C=900.,final_heat_treatment_temp_C=600.,
            pre_exposure_h=96.,pre_exposure_type='cyclic_immersion',pre_exposure_electrolyte='3.5% NaCl',
            pre_exposure_cycle_min=60.,pre_exposure_wet_min=15.,surface_state_cat='rusted',
            TP_scan_rate_mV_s=1.,TP_start_potential_V=-.25,TP_end_potential_V=.25,TP_potential_basis_cat='vs_OCP',
            remarks='S <0.002 wt% retained as upper bound, not exact composition. 96h is total wet/dry elapsed time, not continuous wet immersion. Other corrosion periods lack numeric icorr labels. Prior-austenite microscopy reheating time/grain sizes not assigned as final matrix properties. Final matrix phase and quantitative PDP temperature/OCP time unreported.')
    add_df=pd.DataFrame(rows,columns=base.columns)
    assert not set(add_df.doi)&set(base.doi.dropna())
    expanded=pd.concat([base,add_df],ignore_index=True)
    assert expanded.record_id.is_unique and expanded.icorr_A_cm2.gt(0).all()
    assert expanded.columns.tolist()==base.columns.tolist()
    source=OUT/'verified_extraction.json'
    source.write_text(add_df.dropna(axis=1,how='all').to_json(orient='records',indent=2,force_ascii=False),encoding='utf-8')
    add_df['source_file']=str(source.relative_to(ROOT)).replace('\\','/')
    add_df['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    expanded=pd.concat([base,add_df],ignore_index=True)
    add_df.to_csv(ROOT/'data/NaCl_RE_verified_new.csv',index=False,encoding='utf-8-sig')
    expanded.to_csv(ROOT/'data/training_RE_NaCl_REexpanded.csv',index=False,encoding='utf-8-sig')
    bundle=joblib.load(ROOT/'results/train_test_optimization/random_rows/model_train_only.joblib')
    x=sf.prepare(expanded)
    for nums,cats in bundle['feature_sets'].values():
        for c in nums:x[c]=pd.to_numeric(x[c],errors='coerce')
        for c in cats:x[c]=x[c].replace(['unknown','not_reported','unreported',''],np.nan)
    y=np.log10(expanded.icorr_A_cm2.to_numpy()); re=expanded.RE_wt_pct.gt(0).to_numpy()
    test=expanded.record_id.isin(bundle['test_ids']).to_numpy()
    oldtrain=expanded.record_id.isin(bundle['train_ids']).to_numpy()
    new=~(test|oldtrain); train=~test
    assert test.sum()==28 and oldtrain.sum()==111 and new.sum()==5
    old=bundle['model']; before=old.predict(x)
    model=clone(old).fit(x.loc[train],y[train]); after=model.predict(x)
    result=dict(base_rows=len(base),added_rows=int(new.sum()),added_RE_rows=int((new&re).sum()),
        expanded_rows=len(expanded),expanded_RE_rows=int(re.sum()),columns=len(base.columns),
        model_config=bundle['config'],training_rows=int(train.sum()),test_rows=int(test.sum()),
        new_papers_before_retraining=metrics(y[new],before[new]),
        new_RE_before_retraining=metrics(y[new&re],before[new&re]),
        training_all=metrics(y[train],after[train]),training_RE=metrics(y[train&re],after[train&re]),
        historical_test_all_before=metrics(y[test],before[test]),historical_test_all_after=metrics(y[test],after[test]),
        historical_test_RE_before=metrics(y[test&re],before[test&re]),historical_test_RE_after=metrics(y[test&re],after[test&re]),
        caveat='Existing 28 test IDs unchanged. Configuration frozen before adding new papers; no test tuning. Historical test is reused and only 6 RE rows; not new independent validation. New-paper metrics use old model before fitting. Extraction JSON hash identifies curated facts, not an original PDF hash.',
        input_sha256=hashlib.sha256((ROOT/'data/training_RE_NaCl_REexpanded.csv').read_bytes()).hexdigest())
    (OUT/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    pd.DataFrame(dict(record_id=expanded.record_id,reference_id=expanded.reference_id,RE_positive=re,
        partition=np.where(test,'historical_test',np.where(new,'new_training','old_training')),
        observed_log10=y,old_model_log10=before,refit_model_log10=after)).to_csv(OUT/'predictions.csv',index=False,encoding='utf-8-sig')
    joblib.dump(dict(bundle,model=model,train_ids=expanded.loc[train,'record_id'].tolist()),OUT/'model_train_only.joblib')
    full=clone(old).fit(x,y)
    joblib.dump(dict(bundle,model=full,train_ids=expanded.record_id.tolist(),test_ids=[],
        caution='All-row refit; no independent test score applies to this bundle.'),OUT/'model_all144.joblib')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
