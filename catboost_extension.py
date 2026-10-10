"""Extend the existing nested paper-held-out benchmark with CatBoost candidates.

Optional research command: python catboost_extension.py
Requires: pip install catboost
The saved 18-candidate inner scores and outer predictions are reused only after
their record IDs and targets are checked against the current input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from train import Experiment, metrics, panels, weights, writecsv


ROOT = Path(__file__).resolve().parent
PARAMS = dict(iterations=250, depth=2, learning_rate=0.04, l2_leaf_reg=8,
              loss_function="RMSE", verbose=False, thread_count=1, random_seed=42,
              allow_writing_files=False)


def predict(experiment: Experiment, panel: str, training_papers: list[str],
            test_paper: str) -> np.ndarray:
    numeric, categorical = panels[panel]
    train_idx = np.flatnonzero(np.isin(experiment.g, training_papers))
    test_idx = np.flatnonzero(experiment.g == test_paper)
    model = CatBoostRegressor(**PARAMS)
    model.fit(experiment.X[panel][train_idx], experiment.y[train_idx],
              cat_features=list(range(len(numeric), len(numeric) + len(categorical))),
              sample_weight=weights(experiment.g[train_idx]))
    return model.predict(experiment.X[panel][test_idx])


def main() -> None:
    current = json.loads((ROOT / "results/current_input.json").read_text(encoding="utf-8"))
    prior = json.loads((ROOT / "results/retrained/results.json").read_text(encoding="utf-8"))
    exp = Experiment(current["enriched_rows"])
    snapshot = json.loads((ROOT / "data/input_snapshot.json").read_text(encoding="utf-8"))
    baseline_input = Experiment(snapshot["enriched_rows"])
    for panel, (numeric, _) in panels.items():
        width = len(numeric)
        np.testing.assert_allclose(exp.X[panel][:, :width].astype(float),
                                   baseline_input.X[panel][:, :width].astype(float),
                                   rtol=0, atol=0, equal_nan=True)
        if not np.array_equal(exp.X[panel][:, width:], baseline_input.X[panel][:, width:]):
            raise ValueError("Current categorical inputs differ from baseline snapshot")
    baseline_rows = prior["new51_nested18"]["predictions"]
    if len(baseline_rows) != len(exp.rows):
        raise ValueError("Baseline benchmark row count differs from current input")
    for row, saved in zip(exp.rows, baseline_rows):
        if row["record_id"] != saved["record_id"] or not np.isclose(
                row["icorr_A_cm2"], saved["actual_icorr"], rtol=0, atol=0):
            raise ValueError("Baseline benchmark inputs differ from current input")
    baseline_pred = np.asarray([r["predicted_log"] for r in baseline_rows])
    previous_folds = {r["held_out_paper"]: r for r in prior["new51_nested18"]["folds"]}
    panels_to_try = ("compact", "full", "transfer")
    outer_pred = baseline_pred.copy()
    folds = []
    for held in exp.ids:
        pool = [paper for paper in exp.ids if paper != held]
        pool_idx = np.flatnonzero(exp.g != held)
        scores = []
        for panel in panels_to_try:
            inner_pred = np.full(len(exp.rows), np.nan)
            for inner_held in pool:
                inner_pred[exp.g == inner_held] = predict(
                    exp, panel, [paper for paper in pool if paper != inner_held], inner_held)
            scores.append((r2_score(exp.y[pool_idx], inner_pred[pool_idx]), panel))
        best_score, best_panel = max(scores)
        previous = previous_folds[held]
        chosen = "baseline_18"
        if best_score > previous["inner_log_R2"]:
            outer_pred[exp.g == held] = predict(exp, best_panel, pool, held)
            chosen = "CatBoost_" + best_panel
        folds.append(dict(held_out_paper=held, selected=chosen,
                          best_catboost_inner_log_R2=float(best_score),
                          baseline_best_inner_log_R2=float(previous["inner_log_R2"])))
        print(held, chosen, flush=True)
    result = dict(scope="Nested leave-one-paper-out; 18 prior candidates plus 3 CatBoost candidates",
                  catboost_params=PARAMS, baseline_metrics=prior["new51_nested18"]["metrics"],
                  extended_metrics=metrics(exp.raw, outer_pred), folds=folds)
    out = ROOT / "results/catboost_extension.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    writecsv(ROOT / "results/catboost_extension_predictions.csv", exp.predrows(outer_pred))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
