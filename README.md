# Designing a Pipeline for Ferroptosis Classification and Dose-Response Prediction

B.Tech Project-I/II | VIT | Team: Mayank Bhardwaj (23BKT0106), Sovan Sekar Sarma
(23BKT0057), Bhuvanesh K.V. (23BKT0138) | Faculty Guide: Dr. Adrija Bhattacharya

## Overview

A reproducible machine-learning pipeline that uses a literature-traceable
ferroptosis-related gene signature to (1) classify colorectal-cancer patient
tumors by MSI status and by an exploratory ferroptosis survival-risk group,
and (2) predict quantitative in-vitro drug-response (`LN_IC50`, `AUC`) for
colorectal cancer cell lines, with model interpretability throughout.

**Scope statement:** this project predicts an experimentally measured
in-vitro drug-sensitivity summary statistic for cancer cell lines, not a
human patient's clinical drug dosage. See `docs/LIMITATIONS.md`.

## Motivation

Ferroptosis — an iron-dependent, lipid-peroxidation-driven form of regulated
cell death — is mechanistically well characterized and has been linked to
colorectal cancer prognosis in several independent 2021 studies (see
`docs/RESEARCH_GAP.md`). Those studies stop at survival-risk stratification
using a label built from the same features being modeled. No reviewed study
extends a documented ferroptosis gene panel to a genuine quantitative
drug-response prediction task. This project does both, honestly, on the same
gene panel, with a non-circular classification target added alongside the
literature-style one.

## Architecture

```
Literature (4 CRC papers + 1 mechanistic review)
        -> unified ferroptosis gene signature (35 genes, documented)
                |
    -----------------------------
    |                           |
TCGA COADREAD (594 patients)   GDSC2 COAD/READ (44-51 cell lines)
    -> preprocessing            -> preprocessing
    -> feature engineering          |
    -> MSI classification       -> pooled + per-drug dose-response regression
    -> risk-group classification    |
    -----------------------------
                |
        Evaluation + Interpretability (permutation importance, SHAP)
```
Full diagram and stage-by-stage description: `docs/METHODOLOGY.md`.

## Dataset sources

- **TCGA COADREAD (PanCancer Atlas)** via the public cBioPortal REST API — 594 patients, RNA-seq expression + clinical/MSI/survival data.
- **GDSC2 colorectal cell lines** — Sanger Institute FTP mirror — 44-51 cell lines, fitted `LN_IC50`/`AUC` for 268 drugs, RMA microarray expression.

Full documentation (accession IDs, exact files, citations, license notes):
`docs/DATASET.md`. Ferroptosis gene provenance: `data/gene_selection_method.md`.

## Installation

```bash
git init                     # optional, if you want version control
python -m venv .venv
source .venv/Scripts/activate    # Windows Git-Bash; use .venv\Scripts\activate.bat for cmd.exe
pip install -r requirements.txt
```

GDSC files must be present under `data/raw/gdsc/` (see `docs/DATASET.md` for
exact URLs — they are downloaded once, manually, from the Sanger FTP mirror
since no API is provided by GDSC):
- `GDSC2_fitted_dose_response.csv`
- `Cell_Lines_Details.xlsx`
- `sanger1018_brainarray_ensemblgene_rma.txt` (decompressed from the `.gz` on the FTP mirror)

TCGA data is fetched programmatically (no manual download needed).

## Usage

```bash
python main.py                # run the full pipeline using cached raw data
python main.py --fetch        # also re-fetch TCGA + gene-map data from source first
pytest tests/ -q               # run the automated test suite (29 tests)
```

Individual stages can also be run directly, e.g. `python src/dose_response.py`
(useful for iterating on one module without re-running everything).

## Pipeline stages

1. `src/ferroptosis.py` — builds the 35-gene unified signature from documented literature sources.
2. `src/fetch_tcga.py`, `src/fetch_gene_maps.py` — raw data acquisition (network calls).
3. `src/preprocessing.py` — QC, cleaning, transformation, target construction, leakage-safe splitting for both datasets.
4. `src/feature_engineering.py` — raw vs engineered ferroptosis features; Cox-regression risk score.
5. `src/classification.py` — Logistic Regression / Random Forest / SVM / XGBoost, 3 experiments.
6. `src/dose_response.py` — Linear / Random Forest / XGBoost regression, pooled + 4 representative drugs.
7. `src/interpretability.py` — permutation importance + SHAP for both tasks.
8. `src/evaluation.py` — shared metrics and plotting utilities used by 5 and 6.

## Models

Classification: Logistic Regression, Random Forest, SVM, XGBoost (5-fold
stratified CV hyperparameter search). Regression: Linear Regression, Random
Forest Regressor, XGBoost Regressor (GroupKFold CV by cell line). Full
hyperparameter grids: `src/classification.py::_model_grid`,
`src/dose_response.py::run_pooled_experiment`.

## Evaluation

Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion
matrix, ROC/PR curves, calibration curve. Regression: MAE, RMSE, R², Pearson
r, Spearman ρ, predicted-vs-actual plot, residual plot. See
`docs/METHODOLOGY.md` §7 and `RESULTS.md`.

## Results (headline numbers — full detail in `RESULTS.md`)

- **MSI classification** (primary, non-circular): Logistic Regression, ROC-AUC **0.917** using 35 raw ferroptosis genes.
- **Feature-engineering finding**: collapsing 35 genes into a compact signature score *hurts* MSI classification (ROC-AUC drops to 0.65) — a genuine negative result.
- **Pooled dose-response regression**: Random Forest, R²=**0.617** (LN_IC50), R²=0.436 (AUC), on entirely held-out cell lines.
- **Representative single-drug models** (ferroptosis genes only, n≈44, LOOCV): Random Forest R² 0.21–0.32 across 5-Fluorouracil, Oxaliplatin, Irinotecan, Sorafenib.
- **Interpretability**: `SLC7A11` (a canonical core ferroptosis regulator) ranks among the top predictive genes in both the classification and regression tasks.

## Limitations

No external validation cohort; no whole-transcriptome baseline; GDSC dose-
response curves are illustrative reconstructions from the fitted IC50, not
raw viability data; the survival-risk-group classification is circular by
construction (label built from the same features). Full detail:
`docs/LIMITATIONS.md`. Future work: `docs/FUTURE_WORK.md`.

## Reproducibility

`RANDOM_SEED = 42` everywhere. All preprocessing decisions logged to
`docs/review_evidence/`. All processed data, results, figures, and tables are
regenerated by `python main.py` from the cached raw snapshots in `data/raw/`.

## Repository structure

```
data/
  raw/{tcga,gdsc}/          raw, unmodified data snapshots
  processed/                cleaned, feature-engineered datasets
  ferroptosis_genes_literature.csv
  unified_gene_signature.csv
  gene_selection_method.md
src/                        pipeline modules (see "Pipeline stages" above)
tests/                      pytest test suite (29 tests)
results/                    machine-readable CSV results
tables/                     rounded comparison tables (CSV)
figures/                    all generated plots (PNG)
docs/                       DATASET.md, METHODOLOGY.md, RESEARCH_GAP.md,
                             LIMITATIONS.md, FUTURE_WORK.md,
                             REVIEW_2_CONTENT.md, VIVA.md, RUBRIC_MAPPING.md,
                             review_evidence/
main.py                     pipeline entry point
requirements.txt
PROJECT_STATUS.md            what exists / what's missing (Phase 1 audit)
IMPLEMENTATION_STATUS.md     honest per-module implementation percentage
RESULTS.md                   full results with discussion
```

## References

Full literature list with per-gene provenance:
`data/ferroptosis_genes_literature.csv`. Key sources:

1. Jiang X, Stockwell BR, Conrad M. "Ferroptosis: mechanisms, biology and role in disease." *Nature Reviews Molecular Cell Biology*, 2021.
2. Qi X, Wang R, Lin Y, Yan D, Zuo J, Chen J, Shen B. "A Ferroptosis-Related Gene Signature Identified as a Novel Prognostic Biomarker for Colon Cancer." *Frontiers in Genetics*, 2021. doi:10.3389/fgene.2021.692426
3. Shi C, Xie Y, Li X, Li G, Liu W, Pei W, Liu J, Yu X, Liu T. "Identification of Ferroptosis-Related Genes Signature Predicting the Efficiency of Invasion and Metastasis Ability in Colon Adenocarcinoma." *Frontiers in Cell and Developmental Biology*, 2021. doi:10.3389/fcell.2021.815104
4. Shao Y, Jia H, Huang L, et al. "An Original Ferroptosis-Related Gene Signature Effectively Predicts the Prognosis and Clinical Status for Colorectal Cancer Patients." *Frontiers in Oncology*, 2021. doi:10.3389/fonc.2021.711776
5. Wang Y, Xia H, Chen Z, Meng L, Xu A. "Identification of a ferroptosis-related gene signature predictive model in colon cancer." *World Journal of Surgical Oncology*, 2021. PMC8086290
6. Kautto EA, et al. "Performance evaluation for rapid detection of pan-cancer microsatellite instability with MANTIS." *Oncotarget*, 2017. doi:10.18632/oncotarget.13918
7. Iorio F, et al. "A Landscape of Pharmacogenomic Interactions in Cancer." *Cell*, 2016.
8. cBioPortal for Cancer Genomics — https://www.cbioportal.org
9. Genomics of Drug Sensitivity in Cancer (GDSC) — https://www.cancerrxgene.org
