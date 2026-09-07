# Model Card

Numbers below are read directly from `models/registry.json` and
`results/*.csv` (the exact same fitted objects the application serves) --
regenerate with `python scripts/train_and_persist.py`; if you change any
code, re-run before trusting a stale copy of this document.

## Intended use

Computational research on the association between ferroptosis-related gene
expression and (1) microsatellite-instability status in colorectal tumors,
(2) in-vitro drug-sensitivity summary statistics in colorectal cancer cell
lines. **Not** intended, tested, or validated for clinical diagnosis,
treatment selection, or patient-specific dosing. See `docs/LIMITATIONS.md`.

## Deployed models (served by the Streamlit application)

| Task | Experiment | Model | Selection reason |
|---|---|---|---|
| Classification | `exp1_msi_raw_genes` | LogisticRegression (default), RandomForest, SVM, XGBoost all selectable | Highest ROC-AUC, non-circular task -- see `configs/model_config.yaml::deployed_models` |
| Dose-response | `pooled_LN_IC50` | RandomForest (default), LinearRegression, XGBoost all selectable | Highest R² / lowest MAE on held-out cell lines |
| Dose-response | `pooled_AUC` | RandomForest (default), LinearRegression, XGBoost all selectable | Highest R² / lowest MAE on held-out cell lines |

The circular `exp3_riskgroup_raw_genes` classification models are persisted
(for reproducibility/comparison) but are **not** exposed on the interactive
Ferroptosis Classification page -- only on the read-only Model Performance
page, with an explicit circularity warning.

## Classification models -- MSI status (Experiment 1, raw ferroptosis genes, n_train=473, n_test=119)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| LogisticRegression | 0.849 | 0.500 | 0.889 | 0.640 | **0.917** | 0.677 |
| RandomForest | 0.815 | 0.433 | 0.722 | 0.542 | 0.902 | 0.604 |
| SVM | 0.765 | 0.375 | 0.833 | 0.517 | 0.904 | 0.713 |
| XGBoost | 0.840 | 0.400 | 0.111 | 0.174 | 0.861 | 0.475 |

Threshold: probability ≥ 0.5 → MSI-H. Class imbalance handled via
`class_weight="balanced"`, not resampling. Trained with 5-fold stratified
CV hyperparameter search (`GridSearchCV`, scoring=`roc_auc`) on the 80%
train/val portion; the numbers above are the single held-out 20% test
evaluation.

## Dose-response regression models -- pooled, held-out cell lines (n_train=9143, n_test=2368 rows across 268 drugs)

**LN(IC50):**

| Model | MAE | RMSE | R² | Pearson r | Spearman ρ |
|---|---|---|---|---|---|
| LinearRegression | 2.446 | 2.970 | -0.263 | 0.349 | 0.271 |
| **RandomForest** | **1.283** | **1.636** | **0.617** | **0.792** | 0.659 |
| XGBoost | 1.305 | 1.650 | 0.610 | 0.785 | 0.647 |

**AUC:**

| Model | MAE | RMSE | R² | Pearson r | Spearman ρ |
|---|---|---|---|---|---|
| LinearRegression | 0.082 | 0.113 | 0.062 | 0.307 | 0.289 |
| **RandomForest** | **0.061** | **0.088** | **0.436** | **0.668** | 0.585 |
| XGBoost | 0.063 | 0.089 | 0.427 | 0.657 | 0.560 |

Evaluated on entire cell lines never seen during training (`GroupShuffleSplit`
by `COSMIC_ID`), i.e. the model has never seen these cell lines' expression
profile in any capacity.

## Training data

- Classification: `data/processed/tcga_coadread_features.csv` (594 TCGA
  COADREAD patients, 555 with an MSI label; see `docs/DATASET.md`).
- Dose-response: `data/processed/gdsc_colorectal_dose_response_processed.csv`
  (44 GDSC2 colorectal cell lines x 268 drugs; see `docs/DATASET.md`).

## Inputs

- Classification: 35-gene ferroptosis panel, log2-transformed RNA-seq
  expression (or raw counts, transformed identically at inference time --
  see `src/inference.py::predict_msi_from_raw_expression`).
- Dose-response: same gene panel (raw log2-RMA scale) + a drug's pathway
  annotation and screening concentration range (looked up from the real
  GDSC drug metadata, never fabricated).

## Outputs

- Classification: `MSI-H` / `MSS/MSI-L` label + class probabilities +
  (for linear models) an exact per-sample feature-contribution breakdown.
- Dose-response: a scalar LN(IC50) or AUC prediction + (for RandomForest) an
  empirical ensemble-spread statistic (std/min/max across the fitted trees'
  individual predictions on that exact input) as an uncertainty proxy --
  explicitly NOT a formal statistical confidence interval.

## Additional evaluation added this session

- **Whole-transcriptome baseline (Experiment 4):** the same MSI target,
  classified from ~19,200 genes (PCA-50, train-fold-only) instead of the
  35-gene panel. Best ROC-AUC 0.986 (Random Forest) vs. 0.917 for the panel
  — see `RESULTS.md` §4c. Not a deployed/interactive model; evaluation-only.
- **External validation (`exp1_msi_raw_genes_cptac28`):** a 28-gene
  intersection variant of Experiment 1, scored on the independent CPTAC-2
  cohort with zero retraining/tuning on that cohort. External ROC-AUC
  0.94–0.96; internal (TCGA) ROC-AUC for this same variant 0.84–0.90. See
  `RESULTS.md` §4d and the application's External Validation page. This
  variant is registered in `models/registry.json` under task
  `classification`, experiment `exp1_msi_raw_genes_cptac28` — separate from,
  and not used by, the deployed 35-gene Experiment 1 models.

## Caveats

- Feature importance/SHAP/coefficient values describe association with the
  model's output on this dataset, never biological causality.
- No external (cross-cohort) validation has been performed -- reported
  metrics are internal (same-cohort held-out) validation only.
- The MSI classifier was trained and evaluated exclusively on TCGA COADREAD;
  applying it to expression data from a different sequencing platform,
  normalization pipeline, or tumor type is out of scope and untested.
- The dose-response regressor's drug representation is limited to GDSC's own
  pathway/concentration-range annotation; it cannot score a drug outside the
  268-drug training panel (the application enforces this and reports a
  clear error rather than a silent/incorrect prediction).
