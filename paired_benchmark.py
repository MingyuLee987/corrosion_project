"""Paper-held-out benchmark when a matched control i_corr is available.

This is a different prediction task from zero-shot prediction: the held-out
paper supplies one measured control for each matched condition.  No target
variant from that paper is used to fit or select a model.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR


ROOT = Path(__file__).resolve().parent


def load_pairs(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    by_paper: dict[str, list[dict]] = {}
    for row in rows:
        by_paper.setdefault(row["reference_id"], []).append(row)

    pairs = []
    for paper, group in by_paper.items():
        controls = [row for row in group if float(row["RE_wt_pct"]) == 0]
        variants = [row for row in group if float(row["RE_wt_pct"]) > 0]
        for variant in variants:
            # UCS23 includes Q235B and SPA-H controls, with distinct casting
            # states. Only the rolled Q235B variant has an unambiguous match.
            if paper == "UCS23":
                if variant["sample_id"] != "UCS-Q235BRE(R)":
                    continue
                matched = [c for c in controls if c["sample_id"] == "UCS-Q235B(R)"]
            else:
                matched = [c for c in controls if all(
                    c.get(key, "") == variant.get(key, "")
                    for key in ("processing_state_cat", "pre_exposure_h", "electrolyte_family", "surface_state_cat")
                )]
            if len(matched) != 1:
                raise ValueError(f"Ambiguous control for {variant['record_id']}: {len(matched)}")
            control = matched[0]
            pairs.append(dict(paper=paper, variant=variant, control=control,
                              delta=math.log10(float(variant["icorr_A_cm2"]) /
                                               float(control["icorr_A_cm2"]))))
    return pairs


def family(row: dict) -> str:
    material = (row.get("material_class") or "").lower()
    if "stainless" in material:
        return "stainless"
    if "bearing" in material:
        return "bearing"
    if "carbon" in material:
        return "carbon"
    return "low_alloy_or_unspecified"


NUMERIC = ["RE_log1p_ppm", "C_wt_pct", "Cr_wt_pct", "Ni_wt_pct",
           "Cl_mol_L", "NaCl_wt_pct", "H2SO4_mol_L", "NaHSO3_mol_L",
           "log_pre_exposure_h", "temperature_C"]
CATEGORICAL = ["steel_family", "RE_element_cat", "electrolyte_family",
               "surface_state_cat", "processing_state_cat"]


def features(pair: dict) -> list[object]:
    row = pair["variant"]
    values = []
    for name in NUMERIC:
        if name == "RE_log1p_ppm":
            value = math.log1p(float(row["RE_wt_pct"]) * 10000)
        elif name == "log_pre_exposure_h":
            raw = row.get("pre_exposure_h", "")
            value = math.log1p(float(raw)) if raw else np.nan
        else:
            raw = row.get(name, "")
            value = float(raw) if raw else np.nan
        values.append(value)
    for name in CATEGORICAL:
        raw = family(row) if name == "steel_family" else row.get(name, "")
        if name == "RE_element_cat":
            raw = raw.replace("_", "+")
        values.append(raw or "not_reported")
    return values


def model(kind: str) -> Pipeline:
    num = Pipeline([("fill", SimpleImputer(strategy="median", keep_empty_features=True,
                                           add_indicator=True)),
                    ("scale", StandardScaler())])
    pre = ColumnTransformer([("num", num, list(range(len(NUMERIC)))),
                             ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                              list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL))))])
    estimators = {
        "ridge": Ridge(alpha=10),
        "svr": SVR(C=0.3, epsilon=0.1),
        "rf": RandomForestRegressor(n_estimators=100, max_depth=3,
                                    min_samples_leaf=2, random_state=42, n_jobs=1),
        "extra": ExtraTreesRegressor(n_estimators=100, max_depth=4,
                                     min_samples_leaf=2, random_state=42, n_jobs=1),
        "gb": GradientBoostingRegressor(n_estimators=80, max_depth=1,
                                        learning_rate=0.04, loss="huber", random_state=42),
    }
    return Pipeline([("pre", pre), ("model", estimators[kind])])


def metrics(truth: np.ndarray, predicted: np.ndarray) -> dict:
    return {"R2": float(r2_score(truth, predicted)),
            "MAE": float(mean_absolute_error(truth, predicted))}


def paper_weights(groups: np.ndarray) -> np.ndarray:
    counts = Counter(groups)
    return np.asarray([1 / counts[group] for group in groups], dtype=float)


def predict_held_out(kind: str, X: np.ndarray, delta: np.ndarray,
                     groups: np.ndarray, held: str, pool: list[str] | None = None) -> np.ndarray:
    if pool is None:
        pool = sorted(set(groups))
    train = np.flatnonzero(np.isin(groups, [paper for paper in pool if paper != held]))
    test = np.flatnonzero(groups == held)
    fitted = model(kind)
    fitted.fit(X[train], delta[train], model__sample_weight=paper_weights(groups[train]))
    return fitted.predict(X[test])


def nested_predictions(kinds: tuple[str, ...], X: np.ndarray,
                       delta: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    papers = sorted(set(groups))
    outer = np.empty(len(delta))
    folds = []
    for held in papers:
        training_papers = [paper for paper in papers if paper != held]
        scores = []
        for kind in kinds:
            inner = np.full(len(delta), np.nan)
            for inner_held in training_papers:
                inner[groups == inner_held] = predict_held_out(
                    kind, X, delta, groups, inner_held, training_papers)
            paper_mae = [mean_absolute_error(delta[groups == paper],
                                             inner[groups == paper])
                         for paper in training_papers]
            scores.append((float(np.mean(paper_mae)), kind))
        score, selected = min(scores)
        outer[groups == held] = predict_held_out(selected, X, delta, groups, held)
        folds.append({"held_out_paper": held, "selected_model": selected,
                      "inner_delta_MAE": float(score)})
        print(f"nested {held}: {selected}", flush=True)
    return outer, folds


def main() -> None:
    pairs = load_pairs(ROOT / "data/training_51.csv")
    kinds = ("ridge", "svr", "rf", "extra", "gb")
    X = np.asarray([features(p) for p in pairs], dtype=object)
    delta = np.asarray([p["delta"] for p in pairs])
    groups = np.asarray([p["paper"] for p in pairs])
    control_log = np.asarray([math.log10(float(p["control"]["icorr_A_cm2"])) for p in pairs])
    target_log = control_log + delta
    papers = sorted(set(groups))
    by_kind = {}
    for kind in kinds:
        predicted = np.empty(len(pairs))
        for paper in papers:
            predicted[groups == paper] = predict_held_out(kind, X, delta, groups, paper)
        by_kind[kind] = {"delta": metrics(delta, predicted),
                         "absolute_log": metrics(target_log, control_log + predicted)}

    nested, folds = nested_predictions(kinds, X, delta, groups)

    # The direct-control baseline makes the gain from modelling the RE effect
    # visible. A high absolute R2 alone can be due entirely to calibration.
    baseline = {"delta": metrics(delta, np.zeros(len(delta))),
                "absolute_log": metrics(target_log, control_log)}
    output = {"scope": "Matched-control prediction; one held-out-paper control measurement is provided",
              "method_reference": {
                  "paper": "Al-Nouti and Yaseen (2025), Integration of hybrid machine learning model with differential evolution algorithm for underground Pipeline corrosion prediction",
                  "journal": "Engineering Applications of Artificial Intelligence, 158, 111511",
                  "doi": "10.1016/j.engappai.2025.111511",
                  "journal_rank": "2025 SJR Artificial Intelligence Q1",
                  "adopted_part": "SVR as a corrosion regression candidate; this project independently uses a matched-control log ratio, its own feature set, and paper-held-out validation",
                  "not_reproduced": "The paper's underground-pipeline dataset, differential-evolution optimization, and reported R2 are not used here",
              },
              "paired_rows": len(pairs), "papers": len(papers),
              "excluded": "UCS23 cast RE row lacks a matching cast control",
              "baseline_control_unchanged": baseline, "models_exploratory": by_kind,
              "nested_selected": {"selection_metric": "mean paper-level delta MAE",
                                  "delta": metrics(delta, nested),
                                  "absolute_log": metrics(target_log, control_log + nested),
                                  "folds": folds}}
    out = ROOT / "results/paired_benchmark.json"
    out.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    with (ROOT / "results/paired_heldout_predictions.csv").open(
            "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "reference_id", "record_id", "control_record_id", "control_icorr_A_cm2",
            "actual_icorr_A_cm2", "predicted_icorr_A_cm2", "actual_delta_log10",
            "predicted_delta_log10", "prediction_scope"])
        writer.writeheader()
        for i, pair in enumerate(pairs):
            writer.writerow({"reference_id": pair["paper"],
                             "record_id": pair["variant"]["record_id"],
                             "control_record_id": pair["control"]["record_id"],
                             "control_icorr_A_cm2": 10 ** control_log[i],
                             "actual_icorr_A_cm2": 10 ** target_log[i],
                             "predicted_icorr_A_cm2": 10 ** (control_log[i] + nested[i]),
                             "actual_delta_log10": delta[i],
                             "predicted_delta_log10": nested[i],
                             "prediction_scope": "held-out paper with measured matched control"})
    # Inner selection chose SVR on every outer fold. Save a model trained on all
    # available pairs for future calibrated predictions; training performance is
    # never reported as validation performance.
    fitted = model("svr")
    fitted.fit(X, delta, model__sample_weight=paper_weights(groups))
    numeric_ranges = {}
    for i, name in enumerate(NUMERIC):
        values = np.asarray(X[:, i], dtype=float)
        valid = values[np.isfinite(values)]
        numeric_ranges[name] = [float(valid.min()), float(valid.max())] if len(valid) else None
    known_categories = {name: sorted(set(X[:, len(NUMERIC) + j]))
                        for j, name in enumerate(CATEGORICAL)}
    joblib.dump({"model": fitted, "numeric_columns": NUMERIC,
                 "categorical_columns": CATEGORICAL,
                 "numeric_ranges": numeric_ranges,
                 "known_categories": known_categories,
                 "target": "log10(variant_i_corr/control_i_corr)",
                 "scope": output["scope"]}, ROOT / "models/paired_model.joblib")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
