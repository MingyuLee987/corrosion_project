"""Evidence-aware corrosion i_corr benchmark and calibrated prediction.

This is a new, small-data pipeline independent of the older train.py workflow.
Run ``python corrosion_v2.py audit|benchmark|predict`` from the project root.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from build_v2_31 import COLUMNS as CURATED_COLUMNS


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data/v2_curated_31.csv"
OUT = ROOT / "results/v2"
MODEL = ROOT / "models/v2_paired.joblib"

# Only information available before the variant's polarization test is allowed.
NUMERIC = ["RE_wt_pct", "C_wt_pct", "Cr_wt_pct", "Ni_wt_pct",
           "temperature_C", "NaCl_wt_pct", "Cl_mol_L", "H2SO4_mol_L",
           "NaHSO3_mol_L", "pre_exposure_h"]
CATEGORICAL = ["steel_family", "RE_element_cat", "electrolyte_family",
               "processing_state_cat", "surface_state_cat"]
GROUP = "reference_id"
TARGET = "icorr_A_cm2"
EXCLUDED_FROM_MODEL = ["record_id", "sample_id", "doi", "paper_title", "journal",
                       "source_url", "target_locator", "Ecorr_V", "log_icorr",
                       "inclusion_number_density_mm2", "inclusion_mean_area_um2"]
CANDIDATES = ["control_only", "median_delta", "ridge", "svr"]


def load_data() -> pd.DataFrame:
    data = pd.read_csv(DATA, encoding="utf-8-sig")
    if list(data.columns) != CURATED_COLUMNS:
        raise ValueError("Expected the exact 31-column v2 schema; run build_v2_31.py")
    required = ["record_id", GROUP, TARGET, "material_class"] + NUMERIC + [
        col for col in CATEGORICAL if col != "steel_family"]
    missing = set(required) - set(data)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.record_id.duplicated().any():
        raise ValueError("record_id must be unique")
    if data[GROUP].isna().any() or not np.isfinite(data[TARGET]).all() or (data[TARGET] <= 0).any():
        raise ValueError("Each record needs a paper ID and a positive, finite i_corr")
    data["steel_family"] = data.material_class.apply(steel_family)
    return data


def steel_family(value: object) -> str:
    material = str(value).lower()
    if "stainless" in material:
        return "stainless"
    if "bearing" in material:
        return "bearing"
    if "carbon" in material:
        return "carbon"
    return "low_alloy_or_unspecified"


def audit(data: pd.DataFrame) -> dict:
    coverage = {name: {"reported": int(data[name].notna().sum()),
                       "missing": int(data[name].isna().sum())}
                for name in NUMERIC + CATEGORICAL}
    return {"rows": len(data), "papers": int(data[GROUP].nunique()),
            "target_unit": "A/cm2", "coverage": coverage,
            "input_features": NUMERIC + CATEGORICAL,
            "excluded_examples": EXCLUDED_FROM_MODEL,
            "warning": "Missing is kept missing; no source value is silently replaced by a physical zero"}


def review_queue(data: pd.DataFrame) -> pd.DataFrame:
    critical = ["temperature_C", "C_wt_pct", "Cr_wt_pct", "Ni_wt_pct",
                "Cl_mol_L", "NaCl_wt_pct", "pre_exposure_h"]
    rows = []
    for _, row in data.iterrows():
        missing = [col for col in critical if pd.isna(row[col])]
        flags = str(row.get("quality_flags") or "")
        priority = len(missing) + (3 if flags not in ("", "nan") else 0)
        rows.append({"record_id": row.record_id, "reference_id": row.reference_id,
                     "doi": row.get("doi"), "target_locator": row.get("target_locator"),
                     "quality_flags": row.get("quality_flags"),
                     "missing_critical_columns": "|".join(missing),
                     "review_priority": priority})
    return pd.DataFrame(rows).sort_values(
        ["review_priority", "reference_id", "record_id"], ascending=[False, True, True])


def pair_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Create treatment/control pairs without consulting the variant target."""
    pairs = []
    match_cols = ["electrolyte_family", "processing_state_cat", "surface_state_cat",
                  "pre_exposure_h"]
    for paper, group in data.groupby(GROUP, sort=True):
        controls = group[group.RE_wt_pct.eq(0)]
        variants = group[group.RE_wt_pct.gt(0)]
        for _, variant in variants.iterrows():
            if paper == "UCS23" and variant.sample_id != "UCS-Q235BRE(R)":
                continue  # Its cast variant has no matched cast control.
            candidates = controls.copy()
            if paper == "UCS23":
                candidates = candidates[candidates.sample_id.eq("UCS-Q235B(R)")]
            else:
                for col in match_cols:
                    v = variant[col]
                    candidates = candidates[candidates[col].isna()] if pd.isna(v) else candidates[candidates[col].eq(v)]
            if len(candidates) != 1:
                raise ValueError(f"{variant.record_id}: expected one matched control, found {len(candidates)}")
            control = candidates.iloc[0]
            row = {col: variant[col] for col in NUMERIC + CATEGORICAL}
            row.update(record_id=variant.record_id, reference_id=paper,
                       control_record_id=control.record_id,
                       control_icorr_A_cm2=float(control[TARGET]),
                       variant_icorr_A_cm2=float(variant[TARGET]))
            pairs.append(row)
    result = pd.DataFrame(pairs)
    result["control_log10"] = np.log10(result.control_icorr_A_cm2)
    result["delta_log10"] = np.log10(result.variant_icorr_A_cm2 / result.control_icorr_A_cm2)
    return result


def model_columns() -> list[str]:
    return NUMERIC + CATEGORICAL


def design(data: pd.DataFrame) -> pd.DataFrame:
    result = data[model_columns()].copy()
    result["RE_wt_pct"] = np.log1p(pd.to_numeric(result.RE_wt_pct, errors="coerce") * 10000)
    result["pre_exposure_h"] = np.log1p(pd.to_numeric(result.pre_exposure_h, errors="coerce"))
    for col in NUMERIC:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    for col in CATEGORICAL:
        result[col] = result[col].fillna("not_reported").astype(str)
    return result


def make_model(name: str):
    if name == "control_only":
        return None
    if name == "median_delta":
        return DummyRegressor(strategy="median")
    numeric = NUMERIC
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True,
                                 add_indicator=True)),
        ("scale", StandardScaler())])
    prep = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)],
        sparse_threshold=0)
    estimator = Ridge(alpha=10) if name == "ridge" else SVR(C=0.3, epsilon=0.1)
    return Pipeline([("pre", prep), ("regressor", estimator)])


def group_weights(groups: np.ndarray) -> np.ndarray:
    counts = pd.Series(groups).value_counts()
    return np.asarray([1.0 / counts[group] for group in groups], dtype=float)


def fit_predict(name: str, x: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                train: np.ndarray, test: np.ndarray) -> np.ndarray:
    if name == "control_only":
        return np.zeros(len(test))
    model = make_model(name)
    if name in ("ridge", "svr"):
        model.fit(x.iloc[train], y[train],
                  regressor__sample_weight=group_weights(groups[train]))
    else:
        model.fit(x.iloc[train], y[train])
    return np.asarray(model.predict(x.iloc[test]), dtype=float)


def paper_mae(y: np.ndarray, predictions: np.ndarray,
              groups: np.ndarray, papers: list[str]) -> float:
    return float(np.mean([mean_absolute_error(y[groups == paper],
                                            predictions[groups == paper]) for paper in papers]))


def inner_choice(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray,
                 pool: list[str]) -> tuple[str, dict[str, float]]:
    scores = {}
    for name in CANDIDATES:
        predictions = np.full(len(y), np.nan)
        for held in pool:
            train = np.flatnonzero(np.isin(groups, [p for p in pool if p != held]))
            test = np.flatnonzero(groups == held)
            predictions[test] = fit_predict(name, x, y, groups, train, test)
        scores[name] = paper_mae(y, predictions, groups, pool)
    # In a tie prefer the simpler model, following the declared order.
    selected = min(CANDIDATES, key=lambda name: (scores[name], CANDIDATES.index(name)))
    return selected, scores


def benchmark(pairs: pd.DataFrame) -> dict:
    x = design(pairs)
    y = pairs.delta_log10.to_numpy(float)
    groups = pairs.reference_id.to_numpy(str)
    papers = sorted(set(groups))
    outer = np.full(len(y), np.nan)
    folds = []
    for held in papers:
        pool = [paper for paper in papers if paper != held]
        selected, scores = inner_choice(x, y, groups, pool)
        train = np.flatnonzero(groups != held)
        test = np.flatnonzero(groups == held)
        outer[test] = fit_predict(selected, x, y, groups, train, test)
        folds.append({"held_out_paper": held, "selected_model": selected,
                      "inner_paper_MAE": scores, "outer_rows": len(test),
                      "outer_delta_MAE": float(mean_absolute_error(y[test], outer[test]))})
        print(f"{held}: {selected}", flush=True)
    actual_abs = pairs.control_log10.to_numpy(float) + y
    predicted_abs = pairs.control_log10.to_numpy(float) + outer
    predictions = pairs[["record_id", "reference_id", "control_record_id",
                         "control_icorr_A_cm2", "variant_icorr_A_cm2"]].copy()
    predictions["actual_delta_log10"] = y
    predictions["predicted_delta_log10"] = outer
    predictions["predicted_icorr_A_cm2"] = 10 ** predicted_abs
    predictions["absolute_error_log10"] = np.abs(actual_abs - predicted_abs)
    fixed_svr = np.full(len(y), np.nan)
    for held in papers:
        train = np.flatnonzero(groups != held)
        test = np.flatnonzero(groups == held)
        fixed_svr[test] = fit_predict("svr", x, y, groups, train, test)
    OUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUT / "heldout_predictions.csv", index=False, encoding="utf-8-sig")
    # Descriptive error distribution only: the same data established the model recipe.
    q90 = float(np.quantile(predictions.absolute_error_log10, 0.9, method="higher"))
    return {"scope": "Measured matched control is required for every new prediction",
            "validation": "Nested leave-one-paper-out; model choice uses only training papers; fixed feature schema",
            "rows": len(pairs), "papers": len(papers),
            "absolute_log_R2": float(r2_score(actual_abs, predicted_abs)),
            "absolute_log_MAE": float(mean_absolute_error(actual_abs, predicted_abs)),
            "delta_R2": float(r2_score(y, outer)),
            "delta_MAE": float(mean_absolute_error(y, outer)),
            "control_only_log_R2": float(r2_score(actual_abs, pairs.control_log10)),
            "fixed_SVR_diagnostic": {
                "absolute_log_R2": float(r2_score(actual_abs, pairs.control_log10 + fixed_svr)),
                "absolute_log_MAE": float(mean_absolute_error(y, fixed_svr)),
                "note": "Fixed SVR was informed by prior analysis on these same papers; nested result is the primary model-selection estimate"},
            "observed_error_p90_log10": q90,
            "error_interval_note": "Descriptive held-out error quantile, not a validated prediction interval",
            "folds": folds}


def fit_final(pairs: pd.DataFrame) -> dict:
    x = design(pairs)
    y = pairs.delta_log10.to_numpy(float)
    groups = pairs.reference_id.to_numpy(str)
    selected, scores = inner_choice(x, y, groups, sorted(set(groups)))
    model = make_model(selected)
    if model is not None:
        if selected in ("ridge", "svr"):
            model.fit(x, y, regressor__sample_weight=group_weights(groups))
        else:
            model.fit(x, y)
    value_ranges = {col: [float(x[col].min()), float(x[col].max())]
                    for col in NUMERIC if x[col].notna().any()}
    value_ranges["control_log10"] = [float(pairs.control_log10.min()),
                                      float(pairs.control_log10.max())]
    categories = {col: sorted(x[col].unique().tolist()) for col in CATEGORICAL}
    package = {"model": model, "selected": selected, "columns": model_columns(),
               "value_ranges": value_ranges, "categories": categories,
               "training_rows": len(pairs), "training_papers": sorted(set(groups)),
               "scope": "requires measured same-condition control i_corr"}
    MODEL.parent.mkdir(exist_ok=True)
    joblib.dump(package, MODEL)
    return {"selected_model": selected, "all_paper_LOPO_delta_MAE": scores}


def predict(input_path: Path, output_path: Path) -> None:
    package = joblib.load(MODEL)
    data = pd.read_csv(input_path, encoding="utf-8-sig")
    needed = NUMERIC + CATEGORICAL + ["control_icorr_A_cm2"]
    missing = set(needed) - set(data)
    if missing:
        raise ValueError(f"Prediction input lacks: {sorted(missing)}")
    control = pd.to_numeric(data.control_icorr_A_cm2, errors="coerce")
    if control.isna().any() or (control <= 0).any():
        raise ValueError("A positive measured control_icorr_A_cm2 is required for each row")
    data["control_log10"] = np.log10(control)
    x = design(data)
    if package["selected"] == "control_only":
        delta = np.zeros(len(data))
    else:
        delta = package["model"].predict(x)
    warnings = []
    for _, row in x.iterrows():
        issues = []
        for col, (minimum, maximum) in package["value_ranges"].items():
            value = data.loc[row.name, col] if col == "control_log10" else row[col]
            if pd.notna(value) and (value < minimum or value > maximum):
                issues.append(f"{col} outside training range")
        for col, known in package["categories"].items():
            if row[col] not in known:
                issues.append(f"new {col}: {row[col]}")
        warnings.append("; ".join(issues))
    output = data.copy()
    output["predicted_delta_log10"] = delta
    output["predicted_icorr_A_cm2"] = control.to_numpy() * 10 ** delta
    output["applicability_warnings"] = warnings
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(output[["predicted_icorr_A_cm2", "applicability_warnings"]].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "benchmark", "predict"])
    parser.add_argument("--input", type=Path, default=ROOT / "v2_prediction_input.csv")
    parser.add_argument("--output", type=Path, default=OUT / "new_predictions.csv")
    args = parser.parse_args()
    if args.mode == "predict":
        predict(args.input, args.output)
        return
    data = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    profile = audit(data)
    (OUT / "data_audit.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")
    review_queue(data).to_csv(OUT / "source_review_queue.csv", index=False, encoding="utf-8-sig")
    if args.mode == "audit":
        print(json.dumps(profile, ensure_ascii=False, indent=2))
        return
    pairs = pair_rows(data)
    result = benchmark(pairs)
    result["final_model"] = fit_final(pairs)
    result["method_paper_Q1_AI"] = {"title": "Integration of hybrid machine learning model with differential evolution algorithm for underground Pipeline corrosion prediction",
                                     "journal": "Engineering Applications of Artificial Intelligence",
                                     "year": 2025, "doi": "10.1016/j.engappai.2025.111511",
                                     "used": "SVR as a regression candidate; no claim of reproducing the paper's dataset or optimization"}
    result["evaluation_paper_Q1_AI"] = {"title": "A framework to evaluate machine learning crystal stability predictions",
                                         "journal": "Nature Machine Intelligence", "year": 2025,
                                         "doi": "10.1038/s42256-025-01055-1",
                                         "used": "task-aligned, paper-held-out evaluation"}
    (OUT / "benchmark.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["rows", "papers", "absolute_log_R2",
                                               "absolute_log_MAE", "delta_R2", "delta_MAE",
                                               "final_model"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
