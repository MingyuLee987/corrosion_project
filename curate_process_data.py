"""Apply source-verified process conditions without changing measured targets."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/process_optimization'
THESIS_URL = 'https://harvest.usask.ca/server/api/core/bitstreams/761fa3c0-71d4-43a8-b15f-7c4be2a85e96/content'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data/training_PDP_augmented.csv'
    d = pd.read_csv(source, encoding='utf-8-sig')
    before = d.copy()
    ledger = []
    thesis_hash = hashlib.sha256((ROOT / 'sources/Li_thesis.pdf').read_bytes()).hexdigest()
    assert thesis_hash == '8aaa714911507f2940964e3ba0c9f6836ac07beceae0bb7a1fccb1a0d5c116a0'

    def apply(mask, column, value, locator, note='', url=THESIS_URL, source_hash=thesis_hash):
        if column not in d:
            d[column] = np.nan if isinstance(value, (int, float)) else pd.Series('', index=d.index, dtype=object)
        for i in d.index[mask]:
            old = d.at[i, column]
            if pd.notna(old) and old != '' and old != value:
                raise ValueError(f'Conflicting existing value: {d.at[i, "record_id"]} {column} {old} / {value}')
            d.at[i, column] = value
            ledger.append(dict(record_id=d.at[i, 'record_id'], reference_id=d.at[i, 'reference_id'],
                column=column, previous_value=old, verified_value=value, source_url=url,
                source_sha256=source_hash, locator=locator, note=note))

    rebar = d.reference_id.eq('RebarLi2023')
    solution = rebar & d.test_specimen_type.eq('bare_rebar_in_pore_solution')
    mortar = rebar & d.test_specimen_type.eq('mortar_embedded')
    assert rebar.sum() == 216 and solution.sum() == 72 and mortar.sum() == 144
    apply(solution & d.carbonation_state.eq('non_carbonated'), 'pH', 13.5, 'PDF p84 / printed p44; Table 3.5', 'Prepared solution pH; not a per-sample measured pH.')
    apply(solution & d.carbonation_state.eq('carbonated'), 'pH', 8.0, 'PDF p84 / printed p44; Table 3.6', 'Prepared solution pH; not assigned to mortar.')
    apply(solution, 'TP_scan_rate_mV_s', .166, 'PDF p86 / printed p46; Table 3.7')
    apply(mortar, 'TP_scan_rate_mV_s', .1, 'PDF p86 / printed p46; Table 3.7')
    apply(rebar, 'prepassivation_time_h', 168., 'PDF p82 / printed p42; Sec3.3.1', 'One week before corrosive exposure; distinct from depassivation duration.')
    apply(rebar, 'prepassivation_solution_pH', 13.5, 'PDF p82 / printed p42; Sec3.3.1')
    apply(rebar, 'mill_scale_preserved_cat', 'yes', 'PDF p82 / printed p42; Sec3.3.1', 'Original mill scale retained; loose rust wire-brushed, water rinsed and acetone degreased.')
    apply(rebar, 'surface_preparation_cat', 'mill_scale_retained_wire_brush_water_acetone', 'PDF p82 / printed p42; Sec3.3.1')
    apply(rebar, 'TP_reference_electrode_cat', 'SCE', 'PDF p84-86 / printed p44-46; Sec3.3.2 and Table 3.7')
    apply(rebar, 'TP_scan_protocol_cat', 'two_step_cathodic_then_anodic', 'PDF p91 / printed p51; Sec3.5.3')
    apply(rebar, 'TP_IR_compensation_cat', 'real_time', 'PDF p86 / printed p46; Table 3.7')
    apply(rebar, 'TP_scan_half_range_mV', 150., 'PDF p86 / printed p46; Table 3.7', 'OCP-relative range; not absolute potential.')
    apply(rebar, 'nominal_rebar_diameter_mm', 11.3, 'PDF p82 / printed p42; Sec3.3.1')
    apply(solution, 'scan_branch_recovery_time_h', 4., 'PDF p91 / printed p51; Sec3.5.3', 'Recovery between branches, not initial OCP stabilization.')
    apply(mortar & d.moisture_condition.eq('controlled_relative_humidity'), 'scan_branch_recovery_time_h', 24., 'PDF p91 / printed p51; Sec3.5.3', 'Reported for mortar under air RH conditions; submerged samples not assigned.')
    apply(mortar & d.moisture_condition.eq('submerged'), 'moisture_conditioning_time_h', 168., 'PDF p83 / printed p43; Sec3.3.1; Fig3.13 PDF p101', 'Only seven-day immersion converted. Two months at RH kept as reported text, not invented hours.')
    apply(mortar & d.moisture_condition.eq('controlled_relative_humidity'), 'moisture_conditioning_duration_reported', 'two_months', 'PDF p83 / printed p43; Sec3.3.1')
    apply(mortar, 'mortar_curing_age_before_carbonation_days', 60., 'PDF p83 / printed p43; Sec3.3.1')
    # Process verification for the UCS paper: row labels explicitly distinguish cast and rolled specimens.
    ucs = d.reference_id.eq('UCS23')
    ucs_url = 'https://doi.org/10.3390/met13010066'
    for suffix, state, thickness in [('(C)', 'as_cast', 1.7), ('(R)', 'hot_rolled', 1.0)]:
        mask = ucs & d.sample_id.str.endswith(suffix, na=False)
        apply(mask, 'processing_state_cat', state, 'Sec2.1 Sample Preparation; Fig1', url=ucs_url, source_hash='')
        # Only the RE cast/rolled specimen thicknesses are explicitly tied to selected samples.
        apply(mask & d.sample_id.str.contains('Q235BRE', na=False), 'strip_thickness_mm', thickness,
              'Sec2.1 Sample Preparation', url=ucs_url, source_hash='')
    apply(ucs, 'verified_casting_route_cat', 'twin_roll_strip_casting', 'Sec2.1 Sample Preparation; Fig1', url=ucs_url, source_hash='')
    # EH36 polarization targets are reported after 63 days of prior corrosion.
    eh36 = d.reference_id.eq('Coat24')
    apply(eh36, 'pre_exposure_h', 1512., 'Sec3.4 Electrochemical Experiments Results; 63-day Tafel curves',
          '63 days x 24 h; existing exposure_time_h retained as provenance, not separately duplicated in model.',
          url='https://doi.org/10.3390/coatings14040413', source_hash='')

    # IDs document dependence; never use them as predictors.
    d['measurement_dependence_status'] = 'not_reaudited'
    d.loc[rebar, 'measurement_dependence_status'] = 'three_measurements_per_condition; repeated_conditioning_or_exposure_time_dependence'
    d['curation_status'] = 'existing_source_linked; missing_values_retained'
    d.loc[rebar, 'curation_status'] = 'protocol_verified_in_author_thesis; chemistry_and_numeric_temperature_unverified'
    assert d.record_id.is_unique and len(d) == 314
    np.testing.assert_array_equal(d.icorr_A_cm2, before.icorr_A_cm2)
    np.testing.assert_array_equal(d.record_id, before.record_id)
    d.to_csv(ROOT / 'data/training_process_curated.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(ledger).to_csv(OUT / 'verified_cell_updates.csv', index=False, encoding='utf-8-sig')
    unknown = ['unknown', 'unreported', 'not_reported', '']
    coverage = []
    for scope, g in [('all_314', d), ('RE_positive_33', d[d.RE_wt_pct.gt(0)])]:
        for c in d.columns:
            v = g[c].replace(unknown, np.nan)
            coverage.append(dict(scope=scope, column=c, available_rows=int(v.notna().sum()),
                missing_pct=float(100*v.isna().mean()), available_papers=int(g.loc[v.notna(), 'reference_id'].nunique())))
    pd.DataFrame(coverage).to_csv(OUT / 'curated_column_coverage.csv', index=False, encoding='utf-8-sig')
    audit = []
    for ref, g in d.groupby('reference_id'):
        audit.append(dict(reference_id=ref, rows=len(g), doi=g.doi.iloc[0],
            target_and_source_linked=bool(g.icorr_A_cm2.gt(0).all() and g.source_url.notna().all()),
            base_core_nonmissing_rows=int(g[['C_wt_pct', 'Si_wt_pct', 'Mn_wt_pct', 'P_wt_pct', 'S_wt_pct']].notna().all(axis=1).sum()),
            manufacturing_state_known_rows=int(g.processing_state_cat.replace(unknown,np.nan).notna().sum()),
            numeric_temperature_known_rows=int(g.temperature_C.notna().sum()),
            scan_rate_known_rows=int(g.TP_scan_rate_mV_s.notna().sum()),
            new_verified_cells=sum(x['reference_id']==ref for x in ledger)))
    pd.DataFrame(audit).to_csv(OUT / 'publication_audit.csv', index=False, encoding='utf-8-sig')
    meta = dict(rows=314, papers=int(d.reference_id.nunique()), updated_cells=len(ledger),
        changed_rows=int(pd.DataFrame(ledger).record_id.nunique()), source_csv_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        author_thesis_sha256=thesis_hash, targets_unchanged=True,
        completeness_claim='314 source-linked measured rows, NOT 300 rows with fully verified chemistry and manufacturing history.',
        unresolved=['216 rebar rows lack verified exact steel chemistry and numeric PDP temperature.',
                    'RH sequences and exposure-time measurements reuse specimen groups; rows are not independent specimens.',
                    'Most RE studies still lack numeric pH and manufacturing detail.',
                    'No unreported chemistry, temperature or heat-treatment values were synthesized.'])
    (OUT / 'curation_metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
