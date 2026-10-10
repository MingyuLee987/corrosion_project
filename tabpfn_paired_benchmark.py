"""TabPFN-2 for measured-control corrosion-current ratios, paper-held-out."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from paired_benchmark import CATEGORICAL, NUMERIC, features, load_pairs, metrics


ROOT = Path(__file__).resolve().parent


def main() -> None:
    pairs = load_pairs(ROOT / "data/training_51.csv")
    control_log = np.array([math.log10(float(pair["control"]["icorr_A_cm2"])) for pair in pairs])
    delta = np.array([pair["delta"] for pair in pairs])
    groups = np.array([pair["paper"] for pair in pairs])
    table = pd.DataFrame([features(pair) for pair in pairs], columns=NUMERIC + CATEGORICAL)
    table.insert(0, "measured_control_log10_icorr", control_log)
    for col in ["measured_control_log10_icorr"] + NUMERIC:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    for col in CATEGORICAL:
        table[col] = table[col].astype("category")
    predicted_delta = np.full(len(pairs), np.nan)
    folds = []
    for paper in sorted(set(groups)):
        train_idx = np.flatnonzero(groups != paper)
        test_idx = np.flatnonzero(groups == paper)
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device="cpu", n_estimators=1, random_state=42,
            show_progress_bar=False)
        model.fit(table.iloc[train_idx], delta[train_idx])
        predicted_delta[test_idx] = model.predict(table.iloc[test_idx])
        folds.append({"held_out_paper": paper, "test_rows": len(test_idx)})
        print(paper, flush=True)
    result = {"model": "TabPFN-2, n_estimators=1, CPU",
              "scope": "Matched-control prediction with one measured paper-specific control per condition",
              "protocol": "Fixed features, leave-one-paper-out, no hyperparameter tuning",
              "model_paper": "Hollmann et al., Nature (2025), doi:10.1038/s41586-024-08328-6",
              "evaluation_paper_Q1_AI_journal": "Riebesell et al., Nature Machine Intelligence (2025), doi:10.1038/s42256-025-01055-1",
              "rows": len(pairs), "papers": len(set(groups)),
              "delta": metrics(delta, predicted_delta),
              "absolute_log": metrics(control_log + delta, control_log + predicted_delta),
              "folds": folds}
    (ROOT / "results/tabpfn_paired_benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"delta": result["delta"], "absolute_log": result["absolute_log"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
