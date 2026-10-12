"""Build a source-linked dataset with mandatory, explicit pre-PDP immersion time.

Run with .venv/Scripts/python.exe build_time_verified.py.
The original workbook export and v2 production model are retained for audit.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/time_verified'
DATA = ROOT / 'data'

def new_rows():
    rows = []
    def add(ref, doi, times, values, unit, source, locator, **conditions):
        factors = {'A/cm2': 1, 'uA/cm2': 1e-6, 'mA/cm2': 1e-3, 'nA/cm2': 1e-9}
        for time, value in zip(times, values, strict=True):
            row = dict(record_id=f'{ref}_{conditions.get("sample_id", "steel")}_{time:g}h',
                reference_id=ref, doi=doi, material_class='carbon steel',
                pre_exposure_h=time, pre_exposure_type='continuous_immersion',
                immersion_time_h=time, time_evidence_status='explicit_PDP_link',
                raw_icorr_value=value, raw_icorr_unit=unit, icorr_A_cm2=value*factors[unit],
                target_method='potentiodynamic_Tafel', source_url=source,
                target_locator=locator, time_source_locator=locator,
                data_role='general_steel', composition_basis='reported_sample')
            row.update(conditions)
            rows.append(row)
    add('Sherif14', '10.3390/molecules19079962', [40/60, 12], [15,10], 'uA/cm2',
        'https://www.mdpi.com/1420-3049/19/7/9962', 'Table 1 PDF p4; Sec 2.1; Sec 3.1/3.2',
        sample_id='X65', C_wt_pct=.10, Si_wt_pct=.35, Mn_wt_pct=1.5,
        P_wt_pct=.15, S_wt_pct=.005, N_wt_pct=.015, electrolyte_family='NaCl',
        NaCl_wt_pct=4, pre_exposure_electrolyte='4.0 wt% NaCl',
        gas_condition_cat='aerated', hydrodynamic_condition_cat='stagnant',
        temperature_description='room_temperature', surface_state_cat='ground',
        quality_flags='P=0.15 wt% preserved as printed; RE unreported, not zero; concentration preparation says 40 g/L but label 4 wt%; room temperature not 25C')
    add('Eyu16', '10.3390/ma9090748', [2,4,8], [22.28,8.98,6.04], 'uA/cm2',
        'https://www.mdpi.com/1996-1944/9/9/748', 'Table 3 PDF p7; Sec 2.3; Table 5/Sec 3.1/3.2',
        sample_id='AISI1020', C_wt_pct=.20, Si_wt_pct=.32, Mn_wt_pct=.79,
        Cr_wt_pct=.01, Cu_wt_pct=.01, Ni_wt_pct=.01,
        electrolyte_family='NaHCO3_NaCl', NaHCO3_g_L=5, NaCl_g_L=2,
        Cl_mol_L=2/58.44, Cl_conversion_basis='2 g NaCl per L divided by 58.44 g/mol',
        pre_exposure_electrolyte='5 g/L NaHCO3 + 2 g/L NaCl',
        temperature_C=22, temperature_description='22 +/- 3 C', pH=8.2,
        dissolved_oxygen_mg_L=.8, rotation_rpm=2000, gas_condition_cat='N2_deaerated',
        hydrodynamic_condition_cat='rotating_disk', surface_state_cat='ground',
        TP_scan_rate_mV_s=.167, TP_reference_electrode_cat='SCE',
        quality_flags='RE unreported, not zero; temperature +/-3 C and pH +/-0.1 retained in provenance')
    # Table 1 explicitly reports 0/14/28/49 d, whereas Sec 2.2 lists 7/14/28/49 d.
    # Exclude the disputed 0 d specimen; do not reinterpret it as 7 d.
    add('EH47_21', '10.3390/met11081317', [14*24,28*24,49*24], [1.107e-4,8.093e-5,1.030e-4], 'A/cm2',
        'https://www.mdpi.com/2075-4701/11/8/1317', 'Table 1 PDF p6; Sec 2.1/2.2',
        sample_id='EH47', C_wt_pct=.07, Si_wt_pct=.20, Mn_wt_pct=1.42,
        Mo_wt_pct=.22, Ni_wt_pct=.79, Cr_wt_pct=.07, Al_wt_pct=.03,
        P_wt_pct=.006, S_wt_pct=.001, electrolyte_family='natural_seawater',
        pre_exposure_electrolyte='natural seawater', temperature_description='room_temperature',
        surface_state_cat='rusted', TP_scan_rate_mV_s=.5, TP_reference_electrode_cat='SCE',
        OCP_stabilization_s=60,
        quality_flags='RE unreported, not zero; room temperature not 25C; disputed 0d/7d specimen excluded; natural seawater composition unreported')
    add('C1020_19', '10.1051/matecconf/201927201001', [1,6,7], [.53,.60,.58], 'mA/cm2',
        'https://www.matec-conferences.org/articles/matecconf/pdf/2019/21/matecconf_icfmce2019_01001.pdf',
        'Table 4 PDF p6; Sec 3.3.1/Fig 6', sample_id='C1020_Sa2.5',
        electrolyte_family='NaCl', pre_exposure_electrolyte='3% NaCl (percentage basis not assigned)',
        surface_state_cat='sandblasted_Sa2.5', TP_reference_electrode_cat='SCE',
        composition_basis='grade_reported_quantitative_composition_pending',
        quality_flags='Quantitative composition/temperature/RE unreported; 3% concentration basis not assumed wt%; PDF table obtained through primary-source indexed text')
    # Table 5 is PDP-derived; Tables 3/4 contain other corrosion-rate results.
    # The repeated 80 C / 7 d Table 5 row is one measurement, not two.
    for temp, times, values, potentials in [
        (80, [84,168,336,672], [3.70e-7,7.53e-9,6.47e-9,1.05e-8], [-.541,-.377,-.522,-.427]),
        (40, [168], [1.80e-7], [-.326]),
        (120, [168], [8.54e-7], [-.416]),
    ]:
        for duration, value, potential in zip(times, values, potentials, strict=True):
            add('N80_24', '10.3390/app14020728', [duration], [value], 'A/cm2',
                'https://www.mdpi.com/2076-3417/14/2/728',
                'Table 5 and Fig 8 PDP; Sec 2.3 ex-situ measurement; Table 2 immersion; Table 1 composition',
                sample_id=f'N80_pre{temp}C', alloy_system='N80',
                C_wt_pct=.24, Si_wt_pct=.22, Mn_wt_pct=1.19, P_wt_pct=.013,
                S_wt_pct=.04, Mo_wt_pct=.021, Cr_wt_pct=.036, Ni_wt_pct=.028,
                Ecorr_V=potential, electrolyte_family='simulated_formation_water',
                pre_exposure_electrolyte='CO2-saturated simulated formation water',
                pre_exposure_temperature_C=temp, pre_exposure_pressure_MPa=10,
                temperature_description='ambient_temperature', PDP_measurement_mode_cat='ex_situ',
                surface_state_cat='rusted', Cl_mol_L=26.96/35.45,
                Cl_conversion_basis='Table 2 solution description: 26.96 g/L chloride divided by 35.45 g/mol',
                TP_scan_rate_mV_s=10, TP_start_potential_V=-.5, TP_end_potential_V=.5,
                TP_potential_basis_cat='vs_OCP', TP_reference_electrode_cat='SCE',
                OCP_stabilization_s=3600,
                journal='Applied Sciences', publication_year=2024,
                paper_title='Temperature and Reaction Time’s Effects on N80 Steel Corrosion Behavior in Supercritical CO2 and Formation Water Environments',
                quality_flags='Ex-situ PDP at ambient temperature and pressure; pre-exposure temperature is not PDP temperature; half-immersed two-phase coupon, tested face unspecified; RE unreported not zero; repeated 80C7d Table5 row deduplicated')
    return pd.DataFrame(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(DATA/'training_51.csv', encoding='utf-8-sig')
    retained = source[source.reference_id.eq('Coat24')].copy()
    retained['pre_exposure_h'] = 63*24
    retained['immersion_time_h'] = 63*24
    retained['pre_exposure_type'] = 'continuous_immersion'
    retained['pre_exposure_electrolyte'] = '3.5 wt% NaCl'
    retained['time_evidence_status'] = 'explicit_PDP_link'
    retained['time_source_locator'] = 'Sec 3.4 PDF p7: Tafel after 63 d corrosion; Sec 2.2 immersion protocol'
    retained['data_role'] = 'alloy_RE'
    retained['target_method'] = 'potentiodynamic_Tafel'
    # Legacy Cl conversion is expressly marked approximate and unused.
    retained['Cl_mol_L'] = np.nan
    strict = pd.concat([retained,new_rows()], ignore_index=True)
    # Quantitative composition may be reported in a table, prose, or a
    # documented base steel plus measured additions. Unknown grade-only rows fail.
    eligible = ~strict.reference_id.eq('C1020_19')
    rejected = strict.loc[~eligible].copy()
    rejected['exclusion_reason'] = 'Quantitative steel composition unknown; grade name alone is insufficient'
    rejected.to_csv(OUT/'composition_exclusions.csv',index=False,encoding='utf-8-sig')
    strict = strict.loc[eligible].copy()
    status = {'Eyu16':'sample_specific_table','Sherif14':'reported_in_text','EH47_21':'reported_in_text','Coat24':'documented_base_plus_measured_RE'}
    locator = {'Eyu16':'Table 5: measured mild steel composition (wt%)','Sherif14':'Sec 3.1: X65 quantitative composition','EH47_21':'Sec 2.1: EH47 quantitative composition','Coat24':'Sec 2.1: EH36 base steel remelted with RE; Table 1 base composition; Table 2 measured RE'}
    status['N80_24'] = 'sample_specific_table'
    locator['N80_24'] = 'Table 1: N80 steel quantitative composition (wt%)'
    strict['composition_evidence_status'] = strict.reference_id.map(status)
    strict['composition_source_locator'] = strict.reference_id.map(locator)
    strict['composition_table_locator'] = strict.reference_id.map({'Eyu16':'Table 5','Coat24':'Tables 1 and 2 (base composition plus RE)','N80_24':'Table 1'})
    strict['composition_source_url'] = strict.source_url
    # Keep the time evidence separate from target/composition evidence so each
    # recovered duration can be audited without borrowing another test's clock.
    strict['time_source_url'] = strict.source_url
    strict['time_source_locator'] = strict.reference_id.map({
        'Sherif14': 'Sec 2.1 and Table 1: polarization after 40 min and 12 h immersion',
        'Eyu16': 'Sec 2.3, Fig 9 and Table 3: PDP at 2, 4 and 8 h immersion',
        'EH47_21': 'Sec 2.2 and Table 1: PDP after 14, 28 and 49 d; conflicting 0/7 d excluded',
        'Coat24': 'Sec 3.4 and Fig 5a: Tafel polarization after 63 d corrosion; immersion protocol Sec 2.2',
        'N80_24': 'Sec 2.3: electrochemical measurement after autoclave immersion; Table 2 durations 3.5/7/14/28 d; Table 5 and Fig 8 matched PDP conditions; 1 h OCP wait recorded separately',
    })
    strict['time_basis'] = 'source_reported_pre_PDP_continuous_immersion'
    for col in ['C_wt_pct','Si_wt_pct','Mn_wt_pct','P_wt_pct','S_wt_pct']:
        base = source.loc[source.record_id.eq('RECORR_000010'),col].iloc[0]
        strict.loc[strict.reference_id.eq('Coat24'),col] = base
    strict.loc[strict.reference_id.eq('Coat24'),'composition_basis'] = 'reported_base_steel_plus_measured_RE; final nonRE composition not separately measured'
    from literature_collection_batch import verified_additions, save_audit
    strict['validation_role'] = 'development'
    additions = verified_additions()
    external = additions.loc[additions.validation_role.eq('external_locked')].copy()
    external['log_icorr'] = np.log10(external.icorr_A_cm2)
    external['log_pre_exposure_h'] = np.log1p(external.pre_exposure_h)
    external.to_csv(DATA/'time/external_time_verified.csv',index=False,encoding='utf-8-sig')
    additions.to_csv(DATA/'time/pdp_literature_additions.csv',index=False,encoding='utf-8-sig')
    save_audit()
    strict = pd.concat([strict,additions.loc[additions.validation_role.eq('development')]],ignore_index=True)
    strict['log_icorr'] = np.log10(strict.icorr_A_cm2)
    strict['log_pre_exposure_h'] = np.log1p(strict.pre_exposure_h)
    assert strict.immersion_time_h.notna().all() and (strict.immersion_time_h > 0).all()
    assert not strict.record_id.duplicated().any()
    assert strict.icorr_A_cm2.gt(0).all()
    assert strict.target_method.eq('potentiodynamic_Tafel').all()
    factors = strict.raw_icorr_unit.map({'A/cm2':1, 'uA/cm2':1e-6,'mA/cm2':1e-3})
    np.testing.assert_allclose(strict.icorr_A_cm2, strict.raw_icorr_value*factors, rtol=1e-12)
    strict.to_csv(DATA/'time/training_time_verified.csv', index=False, encoding='utf-8-sig')
    strict[['record_id','reference_id','immersion_time_h','pre_exposure_type',
            'time_evidence_status','time_basis','time_source_locator','time_source_url',
            'target_locator']].to_csv(OUT/'immersion_time_evidence.csv',index=False,encoding='utf-8-sig')
    strict[strict.data_role.eq('alloy_RE')].to_csv(DATA/'time/training_RE_immersion_verified.csv',index=False,encoding='utf-8-sig')

    auxiliary = source[source.reference_id.isin(['Al10_26','Bridge25'])].copy()
    auxiliary['time_evidence_status'] = 'explicit_PDP_link_noncontinuous_exposure'
    auxiliary['data_role'] = 'alloy_RE_auxiliary_noncontinuous'
    mask = auxiliary.reference_id.eq('Bridge25')
    auxiliary.loc[mask,'pre_exposure_h'] = 216
    auxiliary.loc[mask,'pre_exposure_type'] = 'salt_spray'
    auxiliary.loc[mask,'time_source_locator'] = 'Sec 3.4.1/Fig 14: rusted samples after 9 d; Sec 2.2 salt spray'
    auxiliary.loc[~mask,'time_source_locator'] = 'Table 3 p122; pre-exposure Sec 1.2: cycle 60 min, wet 12 min'
    auxiliary['composition_evidence_status'] = 'sample_specific_table'
    auxiliary['composition_table_locator'] = 'Table 1: specimen-specific steel composition'
    auxiliary['composition_source_url'] = auxiliary.source_url
    auxiliary['immersion_time_h'] = np.nan  # Elapsed exposure is not continuous immersion.
    auxiliary.to_csv(DATA/'time/time_verified_noncontinuous_auxiliary.csv',index=False,encoding='utf-8-sig')
    exclusions = source[~source.record_id.isin(strict.record_id)][['record_id','reference_id','doi','target_locator']].copy()
    exclusions['exclusion_reason'] = 'PDP-linked continuous immersion duration not verified; missing is not zero'
    exclusions.loc[exclusions.reference_id.eq('Tang2019'),'exclusion_reason'] = '50 d is separate weight-loss test (Fig 2), not demonstrated pre-PDP immersion (Fig 1/Table 2)'
    exclusions.loc[exclusions.reference_id.eq('Al10_26'),'exclusion_reason'] = 'Cyclic wet/dry elapsed time; retained only in noncontinuous auxiliary dataset'
    exclusions.loc[exclusions.reference_id.eq('Bridge25'),'exclusion_reason'] = 'Salt-spray exposure, not immersion; retained only in noncontinuous auxiliary dataset'
    exclusions.loc[exclusions.reference_id.eq('Coat24'),'exclusion_reason'] = 'No specimen-specific full composition table; only control 0# composition Table1 is eligible'
    exclusions.to_csv(OUT/'original_row_exclusions.csv',index=False,encoding='utf-8-sig')
    manifest = dict(source_rows=len(source), strict_rows=len(strict), strict_papers=int(strict.reference_id.nunique()),
        new_rows=int((~strict.reference_id.eq('Coat24')).sum()), new_papers=int(strict.loc[~strict.reference_id.eq('Coat24'),'reference_id'].nunique()), RE_immersion_rows=int(strict.data_role.eq('alloy_RE').sum()), RE_immersion_papers=int(strict.loc[strict.data_role.eq('alloy_RE'),'reference_id'].nunique()), composition_excluded_rows=len(rejected),
        auxiliary_noncontinuous_rows=len(auxiliary), excluded_original_rows=len(exclusions),
        rule='Source-supported quantitative composition AND positive, explicitly linked continuous pre-PDP immersion durations; no OCP time or separate mass-loss duration substitution',
        target_unit='A/cm2', target_rule='Only PDP/Tafel-derived icorr; exclude gravimetric corrosion rates and separate mass-loss durations', source_original_unchanged=True,
        disputed_new_record='EH47 Table1 day0 excluded because Sec2.2 says day7',
        general_steel_RE_rule='Unreported RE remains missing; never inferred to be 0',
        intended_entrypoint='corrosion_time_verified.py',
        source_counts=strict.groupby('reference_id').size().to_dict())
    manifest['external_locked_rows'] = len(external)
    manifest['external_locked_papers'] = int(external.reference_id.nunique())
    manifest['collection_batch_verified_rows'] = len(additions)
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
