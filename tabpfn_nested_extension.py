"""Nested paper-held-out choice among the saved 18 baselines and TabPFN-2.

The outer paper is never used to choose a feature panel or model. The baseline
file is checked against the current 51 records before its inner scores are used.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

from train import Experiment, metrics, panels, writecsv


ROOT = Path(__file__).resolve().parent
PANELS = ("compact", "transfer")


def frame(exp: Experiment, panel: str) -> pd.DataFrame:
    numerical, categorical = panels[panel]
    data = pd.DataFrame(exp.X[panel], columns=numerical + categorical)
    for col in numerical:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in categorical:
        data[col] = data[col].astype("category")
    return data


def predict(exp: Experiment, data: pd.DataFrame, train_papers: list[str],
            test_paper: str) -> np.ndarray:
    train_idx = np.flatnonzero(np.isin(exp.g, train_papers))
    test_idx = np.flatnonzero(exp.g == test_paper)
    model = TabPFNRegressor.create_default_for_version(
        ModelVersion.V2, device="cpu", n_estimators=1, random_state=42,
        show_progress_bar=False,
    )
    model.fit(data.iloc[train_idx], exp.y[train_idx])
    return model.predict(data.iloc[test_idx])


def main() -> None:
    current = json.loads((ROOT / "results/current_input.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results/retrained/results.json").read_text(encoding="utf-8"))
    exp = Experiment(current["enriched_rows"])
    base_rows = prior["new51_nested18"]["predictions"]
    if len(base_rows) != len(exp.rows):
        raise ValueError("Baseline record count changed")
    for row, saved in zip(exp.rows, base_rows):
        if row["record_id"] != saved["record_id"] or row["icorr_A_cm2"] != saved["actual_icorr"]:
            raise ValueError("Baseline records changed")
    baseline_pred = np.array([r["predicted_log"] for r in base_rows])
    baseline_folds = {f["held_out_paper"]: f for f in prior["new51_nested18"]["folds"]}
    tables = {panel: frame(exp, panel) for panel in PANELS}
    outer_pred = baseline_pred.copy()
    folds = []
    for held in exp.ids:
        pool = [p for p in exp.ids if p != held]
        pool_idx = np.flatnonzero(exp.g != held)
        candidate_scores = []
        for panel in PANELS:
            inner_pred = np.full(len(exp.rows), np.nan)
            for inner_held in pool:
                inner_pred[exp.g == inner_held] = predict(
                    exp, tables[panel], [p for p in pool if p != inner_held], inner_held)
            candidate_scores.append((float(r2_score(exp.y[pool_idx], inner_pred[pool_idx])), panel))
        best_score, best_panel = max(candidate_scores)
        prior_score = baseline_folds[held]["inner_log_R2"]
        choice = "baseline_18"
        if best_score > prior_score:
            outer_pred[exp.g == held] = predict(exp, tables[best_panel], pool, held)
            choice = "TabPFN-2_" + best_panel
        folds.append({"held_out_paper": held, "selected": choice,
                      "tabpfn_best_inner_log_R2": best_score,
                      "baseline_best_inner_log_R2": prior_score})
        print(held, choice, flush=True)
    result = {
        "protocol": "Nested leave-one-paper-out, 18 prior candidates plus 2 TabPFN-2 feature panels; log target",
        "model_paper": "Hollmann et al., Nature (2025), doi:10.1038/s41586-024-08328-6",
        "evaluation_paper_Q1_AI_journal": "Riebesell et al., Nature Machine Intelligence (2025), doi:10.1038/s42256-025-01055-1",
        "baseline_metrics": prior["new51_nested18"]["metrics"],
        "extended_metrics": metrics(exp.raw, outer_pred), "folds": folds,
    }
    (ROOT / "results/tabpfn_nested_extension.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    writecsv(ROOT / "results/tabpfn_nested_predictions.csv", exp.predrows(outer_pred))
    print(json.dumps(result["extended_metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
