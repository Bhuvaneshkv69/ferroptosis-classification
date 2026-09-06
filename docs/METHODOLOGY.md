# Methodology

## 1. Overview

```
FERROPTOSIS LITERATURE  ──▶  UNIFIED GENE SIGNATURE (35 genes, documented)
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                         ▼
        TCGA COADREAD (patients, RNA-seq)          GDSC2 COAD/READ (cell lines, microarray)
                    │                                         │
            PREPROCESSING                              PREPROCESSING
     (QC, dedup, log2, MSI label,                (QC, gene-array resolution,
      Cox risk-score label)                        drug-coverage filter)
                    │                                         │
        FEATURE ENGINEERING                                   │
   (raw genes vs signature score/PC1)                          │
                    │                                         │
        ┌───────────┴───────────┐                             │
        ▼                       ▼                             ▼
 MSI CLASSIFICATION     FERROPTOSIS RISK-GROUP        DOSE-RESPONSE REGRESSION
 (primary, non-circular) CLASSIFICATION (exploratory) (pooled + per-drug)
        └───────────┬───────────┘                             │
                    ▼                                         ▼
              EVALUATION (Acc/Prec/Rec/F1/ROC/PR)   EVALUATION (MAE/RMSE/R2/corr)
                    └───────────────────┬───────────────────┘
                                        ▼
                              INTERPRETABILITY
                    (permutation importance + SHAP, association only)
```

## 2. Why two separate classification targets

A prognosis/risk label built by regressing survival time on the very genes
later used as predictive features is **circular by construction** — the model
can partly "recover" the label because the label is a linear function of the
features. This is a known property of the LASSO-Cox ferroptosis-signature
literature itself (all four CRC papers reviewed in `DATASET.md` do exactly
this). To give the project one genuinely non-circular result, **MSI status**
(MSI-H vs MSS/MSI-L) is used as the **primary** classification target: it is
measured independently (from indel/microsatellite genotyping, MANTIS score),
not derived from the ferroptosis expression panel. The Cox-derived risk group
is kept as a **secondary, exploratory** experiment that mirrors the published
methodology for direct comparability, always reported with the circularity
caveat attached.

**MSI threshold**: MANTIS score ≥ 0.4 ⇒ MSI-H, else MSS/MSI-L, per Kautto et
al., *Oncotarget* 2017 (doi:10.18632/oncotarget.13918), which established 0.4
as the accuracy-optimal cutoff for this exact score.

## 3. TCGA preprocessing pipeline (`src/preprocessing.py::build_tcga_dataset`)

1. **Load** raw clinical (patient- and sample-level) and RNA-seq V2 expression snapshots.
2. **Duplicate/sample-type handling**: keep only `SAMPLE_TYPE == "Primary"`; if a patient has >1 primary sample, keep the alphabetically first sample barcode (standard TCGA convention).
3. **Merge** clinical + expression by `sampleId`; samples with no expression record are dropped (logged).
4. **Missing-value analysis**: per-gene and per-sample missingness computed and logged; samples missing >20% of the 35-gene panel are dropped (none were, in practice — see `docs/review_evidence/tcga_preprocessing_log.txt`); remaining values are the raw cBioPortal RSEM values (no imputation was needed on the final gene panel).
5. **Transformation**: `log2(RSEM + 1)` on all gene expression columns (standard for right-skewed RNA-seq counts-like data).
6. **Target construction**: MSI label (see §2); Cox risk-score label built in `feature_engineering.py`.
7. **Train/val/test split**: stratified 70/10/20 (`preprocessing.split_tcga`), independent per target (a sample missing one target, e.g. MSI score, is simply excluded from that task's split, not imputed).
8. **Leakage prevention**: any scaling/standardization used downstream is fit with `sklearn.Pipeline` inside the CV loop / on the training fold only, never on the full dataset before splitting.

## 4. GDSC preprocessing pipeline (`src/preprocessing.py::build_gdsc_dose_response_dataset`)

1. **Filter** cell-line annotation to `Cancer Type (matching TCGA label) == "COAD/READ"` with expression data available.
2. **Gene resolution**: each of the 35 signature genes is mapped to its Ensembl gene ID(s) via MyGene.info (`src/fetch_gene_maps.py`); every candidate ID is tried against the actual array, and whichever is physically present is kept (recovers e.g. DRD4 via its secondary Ensembl ID). Genes with no ID present on the array (4/35) are dropped and logged, not silently substituted.
3. **Standardization**: each gene z-scored across the colorectal cell-line panel only (GDSC and TCGA expression are never scaled together).
4. **Drug coverage filter**: keep drugs screened on ≥ 40 of the 46 available colorectal lines, to avoid extremely sparse per-drug samples.
5. **Duplicate handling**: drop duplicate (drug, cell line) fitted-curve rows, keep first.
6. **Merge** dose-response with expression by `COSMIC_ID`; drop rows with missing `LN_IC50`/`AUC`.
7. **Train/test split**: `GroupShuffleSplit` on `COSMIC_ID` (80/20) — entire cell lines held out, so no cell line's expression profile appears in both folds. This is the leakage-prevention mechanism explicitly required for cell-line-level data (a row-random split would let the model see the same cell line's baseline expression at train time under a different drug and "cheat" at test time).

## 5. Feature engineering (`src/feature_engineering.py`)

- **Raw features**: the 35 (TCGA) / 31 (GDSC) individual gene expression values.
- **Engineered features**: `ferroptosis_score_mean` (mean per-gene z-score) and `ferroptosis_score_pc1` (first principal component of the standardized gene matrix; explains ~13.5% of variance — see `data/processed/feature_engineering_metadata.json`).
- **Cox risk score**: univariate Cox screen (p < 0.10) on each gene vs `OS_MONTHS`/`OS_EVENT`, then a penalized multivariate `CoxPHFitter` (lifelines, `penalizer=0.1`) on the screened genes; risk score = z-scored expression · Cox coefficients; median split into High/Low risk.
- **Experiment 2 vs Experiment 1 finding**: the compact engineered score performs **substantially worse** than the raw 35-gene panel for MSI classification (ROC-AUC ≈ 0.65 vs ≈ 0.90–0.92) — collapsing 35 genes into 1–2 summary numbers loses gene-specific signal relevant to MSI status. This is reported as a genuine (negative) finding, not smoothed over.

## 6. Models

**Classification** (`src/classification.py`): Logistic Regression, Random Forest, SVM (linear/RBF), XGBoost — each wrapped in a `StandardScaler` pipeline, tuned by `GridSearchCV` (5-fold stratified CV, `scoring="roc_auc"`) on the training set only, evaluated once on the held-out test set.

**Regression** (`src/dose_response.py`): Linear Regression, Random Forest Regressor, XGBoost Regressor. Pooled model uses `ColumnTransformer` (gene z-scores + concentration range + one-hot drug pathway); hyperparameters tuned with `GroupKFold`(5) on the training cell lines only (never mixing folds across a cell line). Representative single-drug models use only the 31 ferroptosis genes with leave-one-out CV (samples per drug: 42–44, too small for a held-out split).

## 7. Evaluation

Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, ROC/PR curves, calibration curve, per-model comparison table/chart. Regression: MAE, RMSE, R², Pearson r, Spearman ρ, predicted-vs-actual plot, residual plot, per-model comparison table/chart.

## 8. Interpretability (`src/interpretability.py`)

Permutation importance (30 repeats, held-out test data) and SHAP `TreeExplainer` values are computed for a fixed-hyperparameter Random Forest fit specifically for interpretability (independent of the hyperparameter-tuned scored models, per standard practice). All outputs are reported as **statistical association with the model's prediction**, never as evidence of biological causality — see `LIMITATIONS.md`.

## 9. Reproducibility

`RANDOM_SEED = 42` fixed everywhere splits/models use randomness. Every preprocessing decision is logged to `docs/review_evidence/*_preprocessing_log.txt` and machine-readable metadata (`data/processed/*_metadata.json`). Re-running `python main.py` regenerates every processed file, table, and figure from the cached raw snapshots; `python main.py --fetch` re-pulls the TCGA/gene-map data from source first.
