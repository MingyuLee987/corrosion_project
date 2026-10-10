"""Exploratory TabPFN-2 benchmark with one complete paper held out per fold.

Run: .venv/Scripts/python tabpfn_paper_benchmark.py
The transfer feature panel is fixed before testing; no outer-fold result selects it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from train import Experiment, metrics, panels, writecsv


ROOT = Path(__file__).resolve().parent
PANEL = "transfer"


def table(exp: Experiment) -> pd.DataFrame:
    numbers, categories = panels[PANEL]
    frame = pd.DataFrame(exp.X[PANEL], columns=numbers + categories)
    for column in numbers:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in categories:
        frame[column] = frame[column].astype("category")
    return frame


def main() -> None:
    data = json.loads((ROOT / "results/current_input.json").read_text(encoding="utf-8"))
    exp = Experiment(data["enriched_rows"])
    features = table(exp)
    predictions = np.full(len(exp.rows), np.nan)
    folds = []
    for paper in exp.ids:
        train_idx = np.flatnonzero(exp.g != paper)
        test_idx = np.flatnonzero(exp.g == paper)
        assert set(exp.g[train_idx]).isdisjoint(exp.g[test_idx])
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V2, device="cpu", n_estimators=1, random_state=42,
            show_progress_bar=False,
        )
        model.fit(features.iloc[train_idx], exp.y[train_idx])
        predictions[test_idx] = model.predict(features.iloc[test_idx])
        folds.append({"held_out_paper": paper, "test_rows": len(test_idx),
                      "log_MAE": float(np.mean(np.abs(exp.y[test_idx] - predictions[test_idx])))})
        print(paper, folds[-1]["log_MAE"], flush=True)
    result = {
        "model": "TabPFN-2 regressor, n_estimators=1, CPU",
        "model_paper": "Hollmann et al., Nature 637, 319-326 (2025), doi:10.1038/s41586-024-08328-6",
        "evaluation_paper_Q1_AI_journal": "Riebesell et al., Nature Machine Intelligence 7, 836-847 (2025), doi:10.1038/s42256-025-01055-1",
        "protocol": "fixed transfer feature panel; 11 leave-one-paper-out folds; log10 target; no cross-paper leakage; no hyperparameter tuning",
        "feature_panel": PANEL,
        "rows": len(exp.rows), "papers": len(exp.ids),
        "metrics": metrics(exp.raw, predictions), "folds": folds,
    }
    (ROOT / "results/tabpfn_paper_benchmark.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    writecsv(ROOT / "results/tabpfn_paper_predictions.csv", exp.predrows(predictions))
    print(json.dumps(result["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
