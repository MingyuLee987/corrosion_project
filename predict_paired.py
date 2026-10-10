"""Predict a RE variant when a matched, measured control is available.

Input CSV must contain record_id, control_record_id, control_icorr_A_cm2,
RE_wt_pct, material_class, RE_element_cat, electrolyte_family, plus any known
model feature columns. The control must have the same test solution, surface,
pre-exposure, processing, and polarization protocol as the variant.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import joblib
import numpy as np

from paired_benchmark import ROOT, features


def run(input_path: Path, output_path: Path, model_path: Path) -> None:
    bundle = joblib.load(model_path)
    with input_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Input CSV has no sample rows")
    required = ("record_id", "control_record_id", "control_icorr_A_cm2",
                "RE_wt_pct", "material_class", "RE_element_cat", "electrolyte_family")
    for number, row in enumerate(rows, start=2):
        missing = [name for name in required if not row.get(name)]
        if missing:
            raise ValueError(f"Row {number}: missing {', '.join(missing)}")
        control = float(row["control_icorr_A_cm2"])
        re_content = float(row["RE_wt_pct"])
        if not math.isfinite(control) or control <= 0:
            raise ValueError(f"Row {number}: control_icorr_A_cm2 must be positive")
        if not math.isfinite(re_content) or re_content <= 0:
            raise ValueError(f"Row {number}: RE_wt_pct must be positive")
    X = np.asarray([features({"variant": row}) for row in rows], dtype=object)
    deltas = bundle["model"].predict(X)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "record_id", "control_record_id", "control_icorr_A_cm2",
            "predicted_delta_log10", "predicted_icorr_A_cm2", "predicted_icorr_uA_cm2",
            "input_warnings", "scope"])
        writer.writeheader()
        for row, values, delta in zip(rows, X, deltas):
            predicted = float(row["control_icorr_A_cm2"]) * 10 ** float(delta)
            warnings = []
            for i, name in enumerate(bundle["numeric_columns"]):
                value = float(values[i])
                limits = bundle["numeric_ranges"][name]
                if not math.isfinite(value):
                    warnings.append(name + "=missing")
                elif limits is not None and (value < limits[0] or value > limits[1]):
                    warnings.append(name + "=outside_training_range")
            offset = len(bundle["numeric_columns"])
            for j, name in enumerate(bundle["categorical_columns"]):
                value = values[offset + j]
                if value not in bundle["known_categories"][name]:
                    warnings.append(name + "=unseen_category")
            writer.writerow({"record_id": row["record_id"],
                             "control_record_id": row["control_record_id"],
                             "control_icorr_A_cm2": row["control_icorr_A_cm2"],
                             "predicted_delta_log10": float(delta),
                             "predicted_icorr_A_cm2": predicted,
                             "predicted_icorr_uA_cm2": predicted * 1e6,
                             "input_warnings": "|".join(warnings),
                             "scope": bundle["scope"]})
    print(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "paired_prediction_input.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "results/paired_new_predictions.csv")
    parser.add_argument("--model", type=Path, default=ROOT / "models/paired_model.joblib")
    args = parser.parse_args()
    run(args.input, args.out, args.model)
