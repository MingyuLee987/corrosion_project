"""Rebuild source-linked rows from the 49-paper workbook; never impute measurements.

The supplied workbook is a paper index, not trusted row-wise measurement joins.
New rows are transcribed from cached publisher documents. Existing rows retain
their previous provenance and are explicitly not newly re-audited here.
"""
from pathlib import Path
import json
import re
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/workbook_mining'
COLLECTION = ROOT / 'data/sources/workbook_mining_collected.csv'
SOURCE = ROOT / 'results/workbook_mining_source.json'


def read_index():
    source = json.loads(SOURCE.read_text(encoding='utf8'))
    mining = source['sheets'][2]
    headers = list(mining['rows'][0]['cells'].values())
    letters = [re.sub(r'\d+', '', k) for k in mining['rows'][0]['cells']]
    index = {}
    for row in source['sheets'][3]['rows'][1:]:
        cells = {re.sub(r'\d+', '', k): v for k, v in row['cells'].items()}
        index[int(cells['A'])] = cells
    return source, headers, letters, index


def collect(index):
    rows = []
    def add(paper, sample, chemistry, locator, **extra):
        row = dict(A=paper, B=index[paper]['F'], C=sample,
            record_id=f'WM{paper:02d}_{sample}', doi=index[paper]['D'],
            source_url='https://doi.org/' + index[paper]['D'],
            source_locator=locator, provenance='primary_source_transcription',
            target_status='not_reported', remarks='', RE_total_wt_pct=None,
            electrolyte_concentration_raw=None, processing_sequence=None,
            raw_icorr_value=None, raw_icorr_unit=None)
        row.update(chemistry)
        row.update(extra)
        if row.get('AL') is not None:
            row['target_status'] = 'reported_numeric'
            row.setdefault('raw_icorr_value', row['AL'])
        # Sum only two actually reported analytical element values.
        if row.get('S') is not None and row.get('T') is not None:
            row['X'] = row['S'] + row['T']
        # Original sheet uses PREN=Cr+3.3Mo+16N. Only compute with all inputs.
        if all(row.get(c) is not None for c in ['I', 'K', 'M']):
            row['Z'] = row['I'] + 3.3 * row['K'] + 16 * row['M']
            row['remarks'] += ' PREN calculated as Cr+3.3Mo+16N; not a measurement.'
        rows.append(row)

    # Only original records belonging to the supplied 49-paper bibliography.
    doi_paper = {v['D']: k for k, v in index.items()}
    legacy = pd.read_csv(ROOT / 'data/training_RE_expanded.csv').replace({np.nan: None})
    mapping = dict(D='C_wt_pct', E='Si_wt_pct', F='Mn_wt_pct', G='P_wt_pct',
        H='S_wt_pct', I='Cr_wt_pct', J='Ni_wt_pct', K='Mo_wt_pct', L='Cu_wt_pct',
        M='N_wt_pct', N='Nb_wt_pct', O='V_wt_pct', P='Ti_wt_pct', Q='Al_wt_pct',
        R='Sb_wt_pct', S='Ce_wt_pct', T='La_wt_pct', U='Pr_wt_pct', V='W_wt_pct',
        W='Y_wt_pct', AA='final_heat_treatment_temp_C', AB='final_heat_treatment_time_min',
        AC='cooling_method_cat', AD='electrolyte_family', AH='temperature_C',
        AI='TP_reference_electrode_cat', AJ='TP_scan_rate_mV_s', AK='Ecorr_V', AL='icorr_A_cm2')
    for r in legacy.to_dict('records'):
        if r['doi'] not in doi_paper:
            continue
        chemistry = {k: r.get(v) for k, v in mapping.items()}
        # Resolve only an explicitly identified single RE species from prior data.
        species = r.get('RE_element_cat')
        if species in {'Ce', 'La', 'Pr', 'Y'}:
            letter = {'Ce': 'S', 'La': 'T', 'Pr': 'U', 'Y': 'W'}[species]
            if chemistry.get(letter) is None:
                chemistry[letter] = r.get('RE_wt_pct')
        if species in {'La+Ce', 'La_Ce'}:
            chemistry['X'] = r.get('RE_wt_pct')
        if r.get('electrolyte_family') == 'NaCl':
            chemistry['AE'] = r.get('NaCl_wt_pct')
        add(doi_paper[r['doi']], r['sample_id'], chemistry,
            r.get('target_locator') or 'Prior curated record; consult original source',
            record_id=r['record_id'], provenance='legacy_curated_not_reaudited_in_this_batch',
            RE_total_wt_pct=r.get('RE_wt_pct'),
            remarks=str(r.get('quality_flags') or '') + '; ' + str(r.get('composition_basis') or ''),
            source_url=r.get('source_url'), processing_sequence=r.get('processing_sequence'),
            raw_icorr_value=r.get('raw_icorr_value'), raw_icorr_unit=r.get('raw_icorr_unit'))

    # H13: Table1 nominal chemistry; Sec2 pp2-3; Sec3.3 p10 gives exact Tafel targets.
    for sample, yttrium, current, potential in [
        ('Q', 0, 4.59e-6, -.486), ('YQ', .014, 4.77e-6, -.509),
        ('QT', 0, 4.46e-6, -.594), ('YQT', .014, 5.32e-6, -.554)]:
        c, si, mn, cr, mo, v = (.42, .98, .41, 4.88, 1.20, .92) if yttrium == 0 else (.41, .87, .49, 5.48, 1.39, .97)
        temper = sample in {'QT', 'YQT'}
        add(16, sample, dict(D=c, E=si, F=mn, I=cr, K=mo, O=v, W=yttrium, Y='Balance'),
            'Table1 p3; Sec2 pp2-3; Sec3.3 p10', AA=580 if temper else 1080,
            AB=120 if temper else 30, AC='air' if temper else 'oil',
            AD='NaCl', AH=25, AI='SCE', AJ=.33, AK=potential, AL=current,
            RE_total_wt_pct=yttrium, raw_icorr_value=current, raw_icorr_unit='A/cm2',
            electrolyte_concentration_raw='0.1 mol/L NaCl',
            processing_sequence='1080C/30min oil quench' + ('; 580C/120min air cool' if temper else ''),
            remarks='P/S unreported. Table1 caption nominal; ICP-MS method reported. Separate 20h immersion is not automatically PDP pre-exposure. Molarity not guessed as wt%.')

    # S31254 B/Ce: sample-specific chemical Table1 and polarization Table3.
    compositions = [
        ('0B', [.012,.64,.47,.035,.006,19.18,18.11,6.17,.71,.20,0,0], -.3076,7.4063e-7),
        ('50B', [.012,.61,.49,.017,.005,18.59,18.11,6.12,.58,.20,.005,0], -.2806,5.6066e-7),
        ('50B-20Ce', [.018,.17,1.82,.007,.002,19.84,18.07,6.21,.56,.199,.005,.002], -.2376,4.3216e-7),
        ('50B-50Ce', [.015,.17,1.87,.008,.002,20.08,17.94,6.02,.48,.187,.005,.005], -.1506,1.0656e-7)]
    for sample, values, e, current in compositions:
        chem = dict(zip(['D','E','F','G','H','I','J','K','L','M','B_addition_wt_pct','S'], values))
        add(30, sample, chem, 'Table1 PDF p2; methods pp2-3; Table3/Fig5 PDF p5',
            AA=950, AB=30, AC='water', AD='H2SO4_HCl_FeCl3_CuCl2',
            AF=30, AH=30, AI='SCE', AK=e, AL=current, RE_total_wt_pct=chem['S'],
            raw_icorr_value=current, raw_icorr_unit='A/cm2',
            electrolyte_concentration_raw='11.5wt%H2SO4+1.2wt%HCl+1.0wt%FeCl3+1.0wt%CuCl2',
            processing_sequence='1200C/10h homogenize;hot roll;1220C/2h solution;950C/30min age;water quench',
            remarks='AF=30min explicitly pre-PDP stabilization; prior cathodic depassivation5min separate. Scan rate unreported. B also changes; not causal Ce comparison.')

    # S31254 Ce: printed Table1 positive exponent conflicts with curve; retain raw, no correction.
    chemistry = dict(D=.012,E=.61,F=.93,G=.01,H=.002,I=20.17,J=18.15,K=6.03,L=.72,M=.212,S=.004,Y='Balance')
    for temp, raw, e in zip([None,800,820,840,860,880,900],
            [2.11,2.14,3.25,5.06,9.39,12.42,13.18],[-.148,-.149,-.182,-.249,-.373,-.463,-.534]):
        add(41, 'solution' if temp is None else f'aged_{temp}', chemistry,
            'Sec1 composition and methods PDF p2; Table1/Fig6 PDF p5',
            AA=1250 if temp is None else temp, AB=120 if temp is None else 90,
            AC='water', AD='NaCl', AE=10, AH=25, AI='SCE', AK=e,
            target_status='quarantined_printed_unit_conflict', raw_icorr_value=raw,
            raw_icorr_unit='Table1: Icorr/10^7 A cm^-2 (printed positive exponent)',
            RE_total_wt_pct=.004,
            remarks='Do not infer missing minus sign from graph or bibliography. AL intentionally unavailable until correction verified. Different aging states are separate conditions.')

    # ODS and its comparator: no inferred elemental Y from Y2O3 processing addition.
    for electrolyte, rawconc, values in [
        ('HNO3','1 mol/L HNO3',[('12Cr ODS',1.7,-.133),('X46Cr13',429.8,-.1715)]),
        ('HNO3','4 mol/L HNO3',[('12Cr ODS',6.28,.2016),('X46Cr13',1.96,.2272)]),
        ('NaCl','0.61 mol/L NaCl',[('12Cr ODS',.68,-.3925),('X46Cr13',2.89,-.4502)])]:
        for sample, value, e in values:
            chem = dict(D=.11,E=.02,F=.10,I=12.58,V=1.49,Y='Balance') if sample=='12Cr ODS' else dict(D=.46,E=.29,F=.54,I=13.53,J=.19,Y='Balance')
            add(31, sample+'_'+rawconc.split()[0]+'_'+electrolyte, chem,
                'Table1 PDF p4; PDP methods pp4-5; Table3 p14/Table4 p15',
                AD=electrolyte, AF=120, AH=20, AI='SCE', AJ=3, AK=e, AL=value*1e-6,
                raw_icorr_value=value, raw_icorr_unit='uA/cm2',
                electrolyte_concentration_raw=rawconc,
                remarks='P/S unreported; Y2O3 powder addition is not measured elemental Y. Nominal ODS recipe and measured OES chemistry differ. Comparator is different material. AF explicit 2h OCP soaking.')

    # Y304: Table2 contains Ecorr and Epit, not the workbook's purported icorr.
    chem = dict(D=.0511,E=.4016,F=1.2005,G=.0334,H=.0019,I=18.0614,J=8.048,
        K=.0198,L=.0391,Q=.0132,M=.0377,Y='Balance')
    for y, e in zip([0,.007,.013,.019,.049],[-.31,-.35,-.29,-.30,-.35]):
        add(17, f'Y{y:g}', dict(chem,W=y), 'Table1/methods PDF p2; Table2 p5',
            AD='NaCl', electrolyte_concentration_raw='3.5% NaCl; percentage basis unassigned',
            AH=24.85, AI='SCE', AJ=.38, AK=e, RE_total_wt_pct=y,
            remarks='298K converted to24.85C. Shared starting-steel composition and nominal experimental Y levels, not individual residual measurements. Table2 has no icorr; do not import workbook AL.')

    # Mg-Ce ingots: target is pitting potential/curve slope rather than icorr.
    for sample, vals in [
        ('LM',[.08,.34,1.68,.31,.23,.0081,.004,.010,.015,.0033,None]),
        ('HM',[.07,.38,1.72,.28,.20,.0072,.0039,.011,.010,.0032,None]),
        ('HC',[.06,.42,1.72,.29,.20,.0077,.0035,.011,.011,.0030,.034]),
        ('MC',[.07,.38,1.72,.29,.20,.007,.003,.012,.010,.0028,.014])]:
        add(6, sample, dict(zip(['D','E','F','J','L','G','H','Q','P','M','S'],vals)),
            'Table1 PDF p3; Table5 PDF p10', AD='NaCl',
            remarks='Table5 reports pitting potential and KI, not icorr. Dash Ce not converted to zero.')

    # Q370: numerical composition available, no tabulated PDP current.
    for sample, vals in [
        ('1',[.017,.431,1.051,.027,.021,1.014,.581,.004,.025,0]),
        ('2',[.010,.453,1.049,.022,.018,1.019,.603,.021,.051,.0014]),
        ('3',[.014,.451,1.042,.016,.019,1.015,.615,.018,.054,.0025])]:
        add(11, sample, dict(zip(['D','E','F','G','H','J','L','Q','P','S'], vals)),
            'Table1 PDF p3; results electrochemical section',
            remarks='Standard-grade0# is bounds/ranges and excluded. No numerical icorr table recovered.')

    # Y-modified weld: report measured residual Y, not electrode coating addition.
    for sample, y in [('unmodified',None),('Y_electrode',.0023),('Y2O3_electrode',.0018)]:
        add(2, sample, dict(D=.11,E=.9,F=5.3,G=.019,H=.011,I=19.3,J=9,P=.09,W=y),
            'Table1 PDF pp3-4; Table2 p6', AD='H2SO4', AH=60, AI='AgAgCl',
            electrolyte_concentration_raw='1 mol/L H2SO4',
            remarks='Composition is surfaced-metal common chemistry. Table2 is mass-loss corrosion rate and hydrogen-evolution Tafel constants, not icorr. Coating1%Y is not bulk Y.')

    # HRB400 2026: Ipass must never be placed in Icorr.
    for sample, vals in [
        ('HRB400',[.24,.5,1.5,.006,.06,None,None]),
        ('HRB400-Cr',[.12,.5,1.4,.007,.08,.5,None]),
        ('HRB400-Cr-RE',[.12,.5,1.5,.007,.08,.5,.018])]:
        for concentration in [.5,1,2,3.5]:
            chem=dict(zip(['D','E','F','G','H','I','X'],vals));chem['Y']='Balance'
            add(32, sample+f'_{concentration:g}NaCl', chem,
                'Table3 PDF p12; methods pp11-12; Table1 p3',
                AD='CaOH2_NaCl', AE=concentration, AH=25, AI='SCE', AJ=.33,
                remarks='Table1 Ipass is passive current, not corrosion current. P/S retained as printed (S0.06/0.08wt%). TotalCeLa only; no invented individual fractions.')

    # La Q355B: exact chemistry; study focuses inclusion evolution, no icorr table.
    for sample, values in [
        ('NO1',[.1740,.1346,1.56,.003,.0549,.0179,None]),
        ('NO2',[.1611,.1327,1.47,.0028,.0566,.0203,.0075]),
        ('NO3',[.1657,.1344,1.47,.0023,.0555,.0192,.0184]),
        ('NO4',[.1791,.2079,1.59,.0022,.0554,.0325,.0425])]:
        add(45, sample, dict(zip(['D','E','F','H','I','Q','T'],values)),
            'Table1 PDF p3; corrosion-target screening',
            remarks='P unreported. No tabulated electrochemical icorr; avoid treating thermodynamic coefficients as composition. Dash La not zero.')

    # HRB4002022: common chemistry plus stated additions, no reported icorr table.
    for sample, cr, total in [('BM',None,None),('Cr',.5,None),('RE',.5,.012)]:
        add(20,sample,dict(D=.24,E=.6,F=1.52,G=.02,H=.03,I=cr,X=total,Y='Balance'),
            'Sec2.1 PDF p2; electrochemical methods pp2-3; Table1 EIS',
            AD='CaOH2_NaCl',AE=1,AI='SCE',AJ=.1667,
            remarks='Base chemistry shared plus nominal additions. Table1 reports EIS values, not icorr. Room temperature not assigned25C; 50min stabilization is explicitly pre-EIS, not pre-PDP.')
    # Chromized layers are a separate surface-treatment domain, not bulk RE steels.
    for oxide, e, current in zip([0,2,4,6,8,10],
            [-.2318,-.4362,-.4456,-.4221,-.4631,-.5297],
            [6.094e-9,1.006e-8,5.296e-8,1.822e-8,1.114e-8,2.429e-8]):
        add(33,f'Cr_{oxide}REO',dict(D=.03,F=2,G=.035,I=8,J=11,K=2,M=.10,Y='Balance'),
            'Table1/Table2 PDF p2; processing pp2-3; Table6/Fig8 p8',
            AA=1100,AB=1800,AC='controlled_cooling',AD='NaCl',AE=3.5,
            AI='SCE',AJ=1,AK=e,AL=current,REO_addition_wt_pct=oxide,
            raw_icorr_value=current,raw_icorr_unit='A/cm2',
            remarks='Surface chromizing domain. Y2O3 is pack addition, not measured steelY. '
                'Table1 has TWO S headings with .75/.015 and Cr8/Ni11 inconsistent with grade; '
                'Si/S left unavailable, other printed values retained. Substrate chemistry not layer chemistry.')
    # Additional composition-only and quarantined papers remain useful for collection.
    for sample, c, si, mn, sulfur, ce, raw in [
        ('C0',.248,.711,1.582,.045,None,18.923),
        ('C1',.248,.712,1.582,.045,.023,None),
        ('C2',.247,.711,1.584,.045,.044,6.350),
        ('C3',.248,.711,1.583,.044,.065,5.980)]:
        add(3,sample,dict(D=c,E=si,F=mn,G=.045,H=sulfur,S=ce),
            'Table1 PDF p2; Fig6 PDF p7; methods/conclusion',AD='NaCl',AE=3.5,
            target_status='quarantined_printed_unit_conflict',raw_icorr_value=raw,
            raw_icorr_unit='mA/cm2 as printed in Fig6',
            remarks='Original workbook has shifted chemistry. Source chemistry retained exactly. '
                'Fig6 current unit conflicts curve magnitude; C2 6.350 vs3.35 prose conflict. '
                'C1 numeric target not extracted. 960h corrosion clock not automatically AF minute soak.')
    for sample, c, si, mn, p, al, cr, ni, cu, nb, ce in [
        ('17',.055,.17,1.12,.010,.032,.53,.31,.31,.043,None),
        ('1',.067,.16,1.10,.011,.020,.55,.32,.35,.034,.0006),
        ('3',.065,.15,1.09,.011,.030,.55,.32,.34,.036,.0021)]:
        add(24,sample,dict(D=c,E=si,F=mn,G=p,Q=al,I=cr,J=ni,L=cu,N=nb,S=ce),
            'Table1 PDF p2; methods p3; Fig10 polarization',AD='NaCl',AE=3.5,
            S_upper_bound_wt_pct=.002,
            remarks='S is <0.002, not measured0.002. No tabulated icorr. '
                'Salt-spray exposure and 20min pre-EIS soaking not assigned to PDP time.')
    for sample, values in [
        ('Q235B',[.12,.09,.34,.019,.009,.024,.022,.023,None,None]),
        ('Q235BM',[.12,.08,.34,.020,.010,.023,.022,.023,.018,.0065]),
        ('Q355B',[.14,.23,1.15,.020,.007,.023,.039,.020,None,None]),
        ('Q355BM',[.13,.23,1.15,.020,.007,.023,.039,.020,.011,.0041])]:
        add(43,sample,dict(zip(['D','E','F','G','H','I','Q','L','S','T'],values)),
            'Table1 PDF p2; wet-dry methods and corrosion results',
            remarks='Final average chemical compositions as reported. No numerical PDP icorr table. '
                'Dash controls not treated as analytical zero. Mass-loss target not converted to icorr.')
    # Weld specimen chemistry is Table6, not base-plate chemistry Table3.
    for sample, values, value, e in zip(
            ['REE0','REE1','REE2','REE3','REE4'],
            [[.033,.30,1.59,.23,.013,.030,2.39],
             [.12,.25,.97,.11,.011,.028,1.07],
             [.16,.29,1.44,.17,.014,.031,1.77],
             [.056,.42,1.47,.23,.014,.029,2.41],
             [.036,.33,1.42,.25,.013,.032,2.34]],
            [24.9,13.4,14.0,13.6,16.5],[-.344,-.546,-.544,-.403,-.527]):
        add(36,sample,dict(zip(['D','E','F','K','Q','P','J'],values)),
            'Publisher full text Sec2; Table6 weld chemistry; Table10 polarization',
            source_url='https://onlinelibrary.wiley.com/doi/10.1155/2018/4071352',
            AD='NaCl',AE=3.5,AF=30,AI='SCE',AJ=1,AK=e,AL=value*1e-6,
            raw_icorr_value=value,raw_icorr_unit='uA/cm2',
            remarks='P/S and bulk RE unreported. Ferroalloy addition to powdered flux '
                'is not weld residual Ce/La. No base-plate composition substitution.')
    for sample, oxide, e in [('316L_ref',None,-.220),('316L_SPS',None,-.225),
            ('316L_1Y2O3',1,-.200),('316L_3Y2O3',3,-.235),('316L_5Y2O3',5,None)]:
        add(46,sample,{},'Processing methods; Table4 electrochemistry PDF p10; Table5 EPR',
            AD='artificial_seawater',AH=25,AI='ACLE',AK=e,REO_addition_wt_pct=oxide,
            remarks='No quantitative elemental chemistry table. Y2O3 powder addition kept separate '
                'from bulk Y. Table4 has Ecorr/Eb but no icorr;5Y sample not present in Table4.')
    # Public sources found during the final bibliography pass.
    for sample, raw, e in [('BM',9.31e-6,-.69),('CGHAZ',1.02e-5,-.64)]:
        add(39,sample,dict(D=.05,E=.17,F=1.55,N=.034,Y='Balance'),
            'Table1 PDF p4; Table4/Fig5 PDF pp9-10; electrochemical methods',
            AD='NaCl',AE=3.5,AI='SCE',raw_icorr_value=raw,
            raw_icorr_unit='Table4 prints A cm^2; corrosion-density area exponent unresolved',
            target_status='quarantined_printed_unit_conflict',
            remarks='Al+Ca+Zr+Ti+RE reported only as combined0.04wt%; no individual RE amount. '
                'Preserve exact Table4 values in raw column; canonical area-unit correction not assumed.')
    for sample, vals in [
        ('A1',[.18,.58,15.2,.009,.006,17.51,1.29,.022,.33,0]),
        ('A2',[.18,.56,15.4,.008,.005,17.53,1.27,.025,.33,.007]),
        ('A3',[.18,.57,15.1,.007,.005,17.51,1.28,.026,.33,.016]),
        ('A4',[.18,.58,15.2,.008,.005,17.54,1.27,.021,.33,.023])]:
        add(26,sample,dict(zip(['D','E','F','G','H','I','J','Q','M','S'],vals)),
            'Publisher J-STAGE full text Table1; corrosion-weight-loss methods',
            AD='H2SO4',AE=5,
            source_url='https://www.jstage.jst.go.jp/article/isijinternational/59/12/59_ISIJINT-2019-248/_html/-char/en',
            remarks='Corrosion target is gravimetric rate, not Tafel icorr. No target conversion.')
    for sample, si, mn, cr, ce, la in [('Q355',.21,1.26,.024,None,None),('Q355RE',.22,1.25,.023,.0144,.0064)]:
        add(47,sample,dict(D=.15,E=si,F=mn,G=.014,H=.007,I=cr,J=.013,Q=.049,L=.021,S=ce,T=la),
            'Table1 PDF p2; Sec2.2; rust products Table3',
            remarks='Exact sample chemistry. Paper reports wet-dry mass loss and rust composition; '
                'no tabulated numerical icorr. Dash RE controls not analytical zero.')
    for sample, vals, raw, e in [
        ('H20B',[.066,.19,1.78,.04,.25,.29,.06,.32,.09,0],1.8e-5,-.709),
        ('H21',[.056,.20,1.77,.04,.25,.29,.05,.33,.09,.0057],2.3e-5,-.716),
        ('H22',[.065,.19,1.79,.04,.26,.29,.05,.33,.09,.0164],2.6e-5,-.723),
        ('H23',[.063,.19,1.82,.04,.26,.29,.06,.33,.09,.0263],3.9e-5,-.712),
        ('X100',[.085,.26,1.69,.19,.14,.17,.047,.24,None,None],5.35e-5,-.724)]:
        add(19,sample,dict(zip(['D','E','F','I','L','K','N','J','O','S'],vals)),
            'NETL-hosted author conference manuscript Table1 pp2-3; Table3 p6; Fig3 p7',
            source_url='https://netl.doe.gov/projects/VueConnection/download.aspx?filename=EffectofMinorCeAdditionsonCorrosionBehaviorofExperimentalPipelineSteel_030324.pdf&id=ccdc6afc-4fb1-49ae-9c7c-737d0869e7c5',
            AD='CO2_NaCl',AE=3.5,AF=60,AH=20,AI='SCE',AK=e,
            target_status='quarantined_printed_unit_conflict',
            raw_icorr_value=raw,raw_icorr_unit='A/m2 (Table3 printed)',
            remarks='Table3 A/m2 conflicts current/derivedCR magnitudes; do not silently reinterpret as A/cm2. '
                'P/S/Ti combined column; B/N combined column: no individual values. Author draft has revision annotations; not final published table verification.')
    for sample, chem in [
        ('MnS_A',dict(D=.051,E=.39,F=1.51,G=.039,H=.0215,J=8.27,I=18.3,K=.21,L=.35,Q=.002,M=.0799)),
        ('CeS_B',dict(D=.011,E=.47,G=.033,H=.017,J=9.41,I=17.9,S=.091,M=.0039))]:
        for electrolyte in ['NaCl','Na2SO4']:
            add(18,sample+'_'+electrolyte,chem,'Author CC-BY paper TableI pp1-2; micro-PDP methods p2',
                source_url='https://doi.org/10.1149/2.0051714jes',
                AA=1099.85,AB=30,AC='water',AD=electrolyte,
                electrolyte_concentration_raw='0.1 mol/L '+electrolyte,AI='AgAgCl_3.33M_KCl',AJ=.383,
                remarks='Microscopic anodic/pitting test, not reported Tafel icorr. '
                    'B steel Mn<.05/Mo<.01/Cu<.01/Al<.005 remain unavailable numeric values. '
                    '1373K converted to1099.85C. A steel Ce dash means no addition, not measured zero.')
    for treated, data in [(False,[(0,-.172,6e-7),(3.6,-.168,5.1e-7),(13.1,-.163,3e-7),(25.7,-.056,6.9e-8)]),
            (True,[(0,-1.083,4.6e-6),(4.2,-.300,5.1e-7),(11.2,-.284,1.2e-7),(29.6,-.168,1.3e-8)])]:
        for surface_ce, e, current in data:
            add(42,f'surfaceCe{surface_ce:g}_'+('450C' if treated else 'as_deposited'),
                dict(D=.02,I=20,Q=5,Y='Balance'),'Publisher PDF methods pp1-2; Tables1/2 p2',
                source_url='https://bcc.bas.bg/BCC_Volumes/Volume_43_Number_1_2011/Volume_43_Number_1_2011_PDF/2011-43-1_24.pdf',
                AA=450 if treated else None,AB=120 if treated else None,
                AD='NaCl',AI='SCE',AJ=10,AK=e,AL=current,
                surface_Ce_at_pct=surface_ce,raw_icorr_value=current,raw_icorr_unit='A/cm2',
                electrolyte_concentration_raw='3.5% NaCl; percentage basis not reassigned',
                remarks='OC404 substrate composition only. Surface oxide Ce atomic percent is not '
                    'bulk Ce wt%. Room-temperature PDP not forced to25C. Thermal treatment performed in air; cooling route unreported.')
    for sample, p, ce, la, raw, e in [
        ('Base',.015,.002,.001,4.39,-.56),('RE90',.016,.006,.003,3.94,-.55),
        ('RE170',.015,.011,.006,2.25,-.50),('RE280',.015,.020,.008,3.63,-.54)]:
        add(35,sample,dict(D=.04,E=.23,F=1.19,G=p,H=.004,I=.38,J=.40,Q=.03,L=.31,S=ce,T=la),
            'Publisher PDF Table1 p2; methods p3; Table3/Fig7 p5',
            source_url='https://fsyfh.mat-test.com/cn/article/doi/10.11973/fsyfh-202403006',
            AD='NaCl',AE=3.5,AI='AgAgCl',AJ=.33,AK=e,AL=raw*1e-5,
            RE_total_wt_pct=ce+la,raw_icorr_value=raw,raw_icorr_unit='x10^-5 A/cm2',
            remarks='Base is not RE-free: measuredCe.002+La.001. Reference is AgAgCl, '
                'not workbook bibliographySCE. 25C belongs to separate6%NaCl inclusion-immersion '
                'test; pre-EIS30min OCP not total PDP immersion.')
    for ce in [0,.006,.012,.018]:
        add(27,f'Ce{ce:g}',dict(D=.060,E=.480,F=1.380,G=.020,H=.002,I=18.370,J=7.210,S=ce,Y='Balance'),
            'TMS-hosted conference paper TableI p2; Sec2.2.2 p3; polarization discussion pp6-7',
            source_url='https://www.tms.org/membercontent/pdf2/mst_2019_122_129.pdf',
            AC='water',AD='NaCl',AE=3.5,AH=25,AI='SCE',AJ=.5,
            AK=-.31125 if ce==.012 else None,
            remarks='Shared starting-steel composition; Ce is nominal laboratory level, '
                'not individually verified residual. 1600C is melting, not post-cast heat treatment. '
                'No numerical icorr in text/table; Ecorr -.31125 only explicit .012Ce sample.')
    for sample, vals, total in [
        ('A',[.07,.40,.40,.087,.004,.26,.48,.12,.036],0),
        ('B',[.09,.42,.37,.10,.008,.26,.47,.12,.022],.0016),
        ('C',[.07,.42,.38,.086,.002,.26,.47,.12,.038],.036),
        ('D',[.08,.48,.44,.09,.001,.27,.49,.11,.038],.064)]:
        add(44,sample,dict(zip(['D','E','F','G','H','L','I','J','Q'],vals)),
            'Publisher paper Table1 printed p108; experimental Sec2.4',
            source_url='https://www.anticorr.org.tw/JCCE/paper/V26/260207.pdf',
            AD='NaCl',AE=.05,RE_total_wt_pct=total,
            remarks='Primary publisher PDF indexed text Table1. Individual rare-earth species '
                'unreported; total RE not assigned to Ce/La. No sample-specific numerical '
                'icorr table; no estimated polarization-curve extraction. Direct download endpoint404; indexed primary text used.')
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    source, headers, letters, index = read_index()
    rows = collect(index)
    measured = [c for c in letters if c not in {'A','B','C','AG'}]
    assert len(measured)==34
    for row in rows:
        missing=[headers[letters.index(c)] for c in measured if row.get(c) is None]
        row['missing_measurement_count']=len(missing)
        row['missing_measurement_columns']='; '.join(missing)
        row['all_measurements_complete']=len(missing)==0
        row['eligible_under_requested_all_column_policy']=len(missing)==0 and row.get('AL') is not None
        row['paper_title']=index[row['A']]['B']
    frame=pd.DataFrame(rows)
    assert frame.record_id.is_unique
    meta=['record_id','doi','source_url','source_locator','provenance','target_status',
        'raw_icorr_value','raw_icorr_unit','electrolyte_concentration_raw','processing_sequence',
        'RE_total_wt_pct','B_addition_wt_pct','REO_addition_wt_pct','surface_Ce_at_pct','S_upper_bound_wt_pct','missing_measurement_count',
        'all_measurements_complete','eligible_under_requested_all_column_policy',
        'missing_measurement_columns','remarks']
    df=frame.reindex(columns=letters+meta).rename(columns=dict(zip(letters,headers)))
    df.to_csv(COLLECTION,index=False,encoding='utf-8-sig')
    access_file=OUT/'remaining_access_review.json'
    access={r['paper_number']:r for r in json.loads(access_file.read_text(encoding='utf8'))} if access_file.exists() else {}
    registry=[]
    for paper, info in index.items():
        part=frame.loc[frame.A.eq(paper)]
        primary=part.loc[part.provenance.eq('primary_source_transcription')]
        registry.append(dict(paper_number=paper,doi=info['D'],paper_title=info['B'],
            workbook_status=info.get('E'),collected_rows=len(part),
            newly_transcribed_rows=len(primary),reported_icorr_rows=int(part.AL.notna().sum()),
            full_measurement_rows=int(part.all_measurements_complete.sum()),
            this_batch_status='primary_source_transcribed' if len(primary) else 'prior_dataset_retained' if len(part) else 'not_transcribed_requires_source_review',
            source_access_result=access.get(paper,{}).get('access_status') if not len(part) else 'record_sources_linked',
            workbook_note=info.get('H')))
    review=pd.DataFrame(registry)
    review.to_csv(ROOT/'data/review/workbook_mining_49_review.csv',index=False,encoding='utf-8-sig')
    coverage=pd.DataFrame([dict(column=headers[letters.index(c)],present_rows=int(frame[c].notna().sum()) if c in frame else 0,
        missing_rows=int(frame[c].isna().sum()) if c in frame else len(frame)) for c in measured])
    coverage.to_csv(OUT/'coverage.csv',index=False,encoding='utf-8-sig')
    new=frame.provenance.eq('primary_source_transcription')
    stats=dict(rows=len(frame),papers=frame.A.nunique(),legacy_rows=int((~new).sum()),
        new_rows=int(new.sum()),new_papers=int(frame.loc[new,'A'].nunique()),
        numeric_icorr_rows=int(frame.AL.notna().sum()),new_numeric_icorr_rows=int(frame.loc[new,'AL'].notna().sum()),
        all_34_measurements_complete_rows=int(frame.all_measurements_complete.sum()),
        requested_policy_training_rows=int(frame.eligible_under_requested_all_column_policy.sum()),
        unit_quarantined_rows=int(frame.target_status.str.startswith('quarantined').sum()),
        original_workbook_sha256=hashlib.sha256(Path(source['source']).read_bytes()).hexdigest(),
        original_workbook_unchanged=True,source_raw_populated_rows=len(source['sheets'][2]['rows'])-1,
        policy='No measurement imputation. No guessed absent elements, target exponents, wt% conversions or cross-experiment exposure clocks.',
        nonexhaustive='49 bibliography entries screened for collection status; papers marked not_transcribed require further primary-source access. No claim that every obtainable value has been exhausted.')
    (OUT/'summary.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf8')
    def records(data):
        return data.astype(object).where(pd.notna(data),None).to_dict('records')
    payload=dict(headers=list(df.columns),rows=records(df),review_headers=list(review.columns),
        review_rows=records(review),coverage_headers=list(coverage.columns),coverage_rows=records(coverage),stats=stats)
    (OUT/'workbook_payload.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
