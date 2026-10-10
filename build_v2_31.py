"""Build the reviewed 31-column modelling table from the 93-column source.

The source file is retained as an audit trail. Values are copied, never guessed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/training_51.csv"
DESTINATION = ROOT / "data/v2_curated_31.csv"
MANIFEST = ROOT / "data/v2_curated_31_manifest.json"

COLUMNS = [
    # Identity and source evidence (5)
    "record_id", "reference_id", "doi", "sample_id", "material_class",
    # Composition (10)
    "C_wt_pct", "Si_wt_pct", "Mn_wt_pct", "Cr_wt_pct", "Ni_wt_pct",
    "Mo_wt_pct", "Cu_wt_pct", "RE_wt_pct", "RE_element_cat", "composition_basis",
    # Manufacture, surface, environment and protocol (11)
    "processing_state_cat", "surface_state_cat", "electrolyte_family",
    "NaCl_wt_pct", "Cl_mol_L", "H2SO4_mol_L", "NaHSO3_mol_L",
    "temperature_C", "pre_exposure_h", "gas_condition_cat", "TP_scan_rate_mV_s",
    # Target and traceability (5)
    "raw_icorr_value", "raw_icorr_unit", "icorr_A_cm2",
    "target_locator", "quality_flags",
]


def main() -> None:
    source = pd.read_csv(SOURCE, encoding="utf-8-sig")
    if len(COLUMNS) != 31 or len(set(COLUMNS)) != 31:
        raise AssertionError("Curated schema must have 31 distinct columns")
    absent = set(COLUMNS) - set(source)
    if absent:
        raise ValueError(f"Missing source columns: {sorted(absent)}")
    curated = source.loc[:, COLUMNS].copy()
    if curated.record_id.isna().any() or curated.record_id.duplicated().any():
        raise ValueError("Every row needs a unique record_id")
    if curated.reference_id.isna().any() or curated.doi.isna().any():
        raise ValueError("Every row needs a paper ID and DOI")
    target = pd.to_numeric(curated.icorr_A_cm2, errors="coerce")
    if target.isna().any() or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Every target must be positive A/cm2")
    # Check a round trip because a later source edit must not silently alter targets.
    curated.to_csv(DESTINATION, index=False, encoding="utf-8-sig")
    reread = pd.read_csv(DESTINATION, encoding="utf-8-sig")
    if not reread.record_id.equals(curated.record_id):
        raise AssertionError("Record IDs changed during export")
    np.testing.assert_allclose(reread.icorr_A_cm2, curated.icorr_A_cm2, rtol=1e-12, atol=0)
    manifest = {
        "source": str(SOURCE.relative_to(ROOT)),
        "output": str(DESTINATION.relative_to(ROOT)),
        "rows": len(curated), "papers": int(curated.reference_id.nunique()),
        "column_count": len(COLUMNS), "columns": COLUMNS,
        "rule": "Direct subset only; no imputation, unit conversion or target changes in this step",
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{DESTINATION}: {len(curated)} rows, {len(COLUMNS)} columns, {manifest['papers']} papers")


if __name__ == "__main__":
    main()
