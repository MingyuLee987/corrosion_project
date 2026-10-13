"""Attach sample-specific reported matrix phases/constituents, never infer from chemistry."""
from pathlib import Path
import hashlib
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parent
COLUMNS=['phase_reported','microstructure_constituents_reported','phase_evidence_method',
         'phase_source_url','phase_source_locator','phase_source_sha256','phase_verification_status',
         'polygonal_ferrite_fraction_pct','granular_bainite_fraction_pct','bainitic_ferrite_fraction_pct',
         'matrix_grain_size_um']

def annotate(data):
    d=data.copy()
    for c in COLUMNS:d[c]=np.nan if c.endswith('_pct') or c.endswith('_um') else ''
    d['phase_verification_status']='unreported_or_not_yet_verified'
    def assign(mask,phase,constituents,method,url,locator,file,**numbers):
        for c,value in dict(phase_reported=phase,microstructure_constituents_reported=constituents,
            phase_evidence_method=method,phase_source_url=url,phase_source_locator=locator,
            phase_source_sha256=hashlib.sha256((ROOT/file).read_bytes()).hexdigest() if file else '',
            phase_verification_status='source_reported',**numbers).items():d.loc[mask,c]=value
    assign(d.reference_id.eq('Zhang2025'),'austenite','austenite','metallography',
        'https://www.mdpi.com/1996-1944/18/1/69','Sec3.1 Figure1: all three experimental steels single-phase austenite',
        'sources/literature/Zhang2025.md')
    assign(d.reference_id.eq('Bridge25'),'ferrite','polygonal_ferrite+pearlite','optical_metallography',
        'https://www.mdpi.com/2075-4701/15/1/85','Sec2 Figure1: three test steels',
        'sources/literature/Bridge25.md')
    for sample,grain in [('UCS-Q235BRE(R)',24.),('UCS-Q235B(R)',41.)]:
        assign(d.reference_id.eq('UCS23')&d.sample_id.eq(sample),'ferrite','polygonal_ferrite+pearlite',
            'optical_metallography_ImagePro','https://www.mdpi.com/2075-4701/13/1/66','Sec3.1 Figure2b-c Table2',
            'sources/literature/UCS23.md',polygonal_ferrite_fraction_pct=98.,matrix_grain_size_um=grain)
    assign(d.reference_id.eq('UCS23')&d.sample_id.eq('UCS-SPA-H(R)'),'ferrite',
        'polygonal_ferrite+granular_bainite+bainitic_ferrite','optical_metallography_ImagePro',
        'https://www.mdpi.com/2075-4701/13/1/66','Sec3.1 Figure2d Table2','sources/literature/UCS23.md',
        polygonal_ferrite_fraction_pct=57.,granular_bainite_fraction_pct=32.,bainitic_ferrite_fraction_pct=11.)
    for sample,grain in [('0RE',19.9),('0.003RE',17.5),('0.010RE',15.5)]:
        assign(d.reference_id.eq('HRB400_RE_2025')&d.sample_id.eq(sample),'ferrite','ferrite',
            'optical_metallography','https://www.specialsteeljournal.com/zh/article/doi/10.20057/j.1003-8620.2024-00280/',
            'PDF printed p59 Sec2.1 and p60 Figure2: reported ferrite grain refinement; other constituents not assigned',
            'sources/HRB400_RE_2025.pdf',matrix_grain_size_um=grain)
    # Y paper was verified as publisher PDF text, not a local visually checked PDF.
    assign(d.reference_id.eq('Y_X100_2018'),'ferrite','irregular_polygonal_ferrite+quasi_polygonal_ferrite',
        'metallography_publisher_PDF_text_only','https://cdn.sciengine.com/doi/pdf/4A040ADEAFCF45B4BD8F9C554DCBF122',
        'PDF printed p104 Sec2.2 Figure4: three experimental steels',None)
    return d

def main():
    audit=[]
    for name in ['training_PDP_augmented','training_RE_expanded','training_process_curated',
                 'training_RE_all_journals','RE_all_journals_additions']:
        path=ROOT/'data'/f'{name}.csv';before=pd.read_csv(path);after=annotate(before)
        assert before.record_id.equals(after.record_id)
        np.testing.assert_array_equal(before.icorr_A_cm2,after.icorr_A_cm2)
        after.to_csv(path,index=False,encoding='utf-8-sig')
        audit.append(dict(dataset=name,rows=len(after),verified_phase_rows=int(after.phase_verification_status.eq('source_reported').sum())))
    out=ROOT/'results/phase_analysis';out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(audit).to_csv(out/'coverage.csv',index=False,encoding='utf-8-sig')
    print(pd.DataFrame(audit).to_string(index=False))

if __name__=='__main__':main()
