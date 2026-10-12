# Attached paper applicability and model experiment

Reference: 10.1038/s41529-026-00859-9, A supervised machine learning approach with dual GAN augmentation and Bayesian HPO for corrosion prediction of Mg-Li alloys.

Decision: use as a methodological reference, not a steel training-data source. The paper describes 450 Mg-Li records and 70 external records. Table 6 separates PDP-derived icorr from weight-loss CR and hydrogen evolution. Its Data availability statement says datasets are not public. No Mg-Li rows or synthetic observations were added to our steel CSV.

Table 4 reports icorr R2 0.814 for random splitting, 0.735 for source-grouped validation and 0.768 for external validation. These are not performance targets transferable to 18 steel observations from 5 sources.

Adopted: log10 icorr, RobustScaler fitted only on training folds, model comparison including CatBoost and Random Forest, Optuna TPE tuning and a train/validation gap penalty (weight 0.5, tolerance 0.15). Adaptations: 24 trials per outer fold rather than 100, shallow trees for tiny data, nested leave-one-publication-out rather than random splitting, pooled inner out-of-source R2 because tiny individual validation groups have unstable R2. Equal total training weight per paper is normalized to mean sample weight 1.

Not adopted: GAN augmentation (only 18 rows / 5 independent papers); Mg-Li phase encodings; thermodynamic descriptors requiring full alloy compositions and element property assumptions. Missing fields remain missing. Additional pre-exposure temperature/pressure and OCP candidates undergo the same <=20% training-fold missingness rule and are not automatically admitted.

Same 18-row dataset baseline with time: log10 R2 -0.805259, log10 MAE 1.506481. Paper-guided nested experiment: log10 R2 -0.734054, log10 MAE 1.472201. Improvement in R2 is 0.071205. All outer predictions are from held-out publications. This remains worse than the pooled-mean R2 reference and is not a validated deployment model. It is an exploratory comparison: outer scores have been observed during development, so independent new publications are required to confirm improvement. No source was excluded because of model error.

Run: .venv/Scripts/python.exe corrosion_time_verified.py then .venv/Scripts/python.exe corrosion_paper_guided.py. Dependencies: requirements_paper_guided.txt. Fold-specific parameters and missingness are in benchmark.json; all 120 optimization trials and 18 out-of-source predictions are retained alongside this review.
