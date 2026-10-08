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
python main.py                       # run the full pipeline using cached raw data
python main.py --fetch               # also re-fetch TCGA + gene-map data from source first
python scripts/run_benchmark.py                     # classification: nested-CV model selection, locked test, external cohort
python scripts/run_benchmark.py --task regression   # drug-response benchmark (registers the production models)
python scripts/write_model_selection_doc.py         # regenerate docs/MODEL_SELECTION.md from the results
pytest tests/ -q                     # run the automated test suite (121 tests)
streamlit run app/main.py            # launch the interactive application
```

Individual stages can also be run directly, e.g. `python src/dose_response.py`
(useful for iterating on one module without re-running everything). Full
command reference: `docs/API_OR_PIPELINE_GUIDE.md`. Guided click-through of
the application: `docs/DEMO_GUIDE.md`.

## Application

A Streamlit application (`app/`) is the interactive layer on top of the
pipeline: dataset/preprocessing statistics, an interactive MSI classifier,
dose-response predictor, and drug-resistance classifier, a non-circular
prognosis-validation view (concordance index + Kaplan-Meier curves), full
model-performance/interpretability browsing, and a downloadable, dynamically
generated HTML report. A sample can be provided three ways — an existing
TCGA/GDSC record, a validated CSV upload (see `docs/CSV_FORMAT.md`), or a
guided "Research Sample Input" form with one numeric field per required
gene, pre-filled with real cohort medians. All three paths feed the
identical persisted model; none of them involves a symptom/lifestyle
questionnaire or an invented mapping. It never re-trains anything — every
prediction comes from a persisted model in `models/registry.json`, the
exact object `classification.py`/`dose_response.py`/`drug_resistance.py`
evaluated. See `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `UI_AUDIT.md`,
`docs/REFERENCES.md`.

**Result popup and appearance.** When a classification finishes, a result popup shows the predicted class
("Predicted: MSI-H" / "Predicted: MSS / MSI-L"), the model's *prediction probability* (its probability of MSI-H; omitted
if the model were not probability-calibrated), how many of the combined model's members agree with the combined result
(computed from their real predictions; single models show none), and a plain-language explanation. Model
performance (locked-test accuracy and AUC) is shown separately and is read from `results/final_model_registry.json`; it
describes the model on held-out data, not this sample. The latest analysis stays at the top of the workspace. A Light / Dark
switch in the header applies one centralized theme (`app/theme.py`) to every page and chart and is remembered across refreshes.
Browser acceptance: `python scripts/ui_acceptance.py <input_dir>` (app running).

## Pipeline stages

1. `src/ferroptosis.py` — builds the 35-gene unified signature from documented literature sources.
2. `src/fetch_tcga.py`, `src/fetch_gene_maps.py` — raw data acquisition (network calls).
3. `src/preprocessing.py` — QC, cleaning, transformation, target construction, leakage-safe splitting for both datasets.
4. `src/feature_engineering.py` — raw vs engineered ferroptosis features; Cox-regression risk score.
5. `src/classification.py` — Logistic Regression / Random Forest / SVM / XGBoost, 3 experiments.
6. `src/dose_response.py` — Linear / Random Forest / XGBoost regression, pooled + 4 representative drugs.
7. `src/drug_resistance.py` — Resistant/Sensitive classification (pooled + oxaliplatin-specific), median-split from GDSC LN_IC50.
8. `src/interpretability.py` — permutation importance + SHAP for both tasks.
9. `src/evaluation.py` — shared metrics and plotting utilities used by 5, 6, and 7.

## Models

Classification: Logistic Regression, Random Forest, SVM, XGBoost (5-fold
stratified CV hyperparameter search). Regression: Linear Regression, Random
Forest Regressor, XGBoost Regressor (GroupKFold CV by cell line).
Drug-resistance classification: Logistic Regression, Random Forest (same
GroupShuffleSplit-by-cell-line discipline as the regression task). Full
hyperparameter grids: `src/classification.py::_model_grid`,
`src/dose_response.py::run_pooled_experiment`,
`src/drug_resistance.py::run_pooled_resistance_experiment`.

## Evaluation

Classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion
matrix, ROC/PR curves, calibration curve. Regression: MAE, RMSE, R², Pearson
r, Spearman ρ, predicted-vs-actual plot, residual plot. See
`docs/METHODOLOGY.md` §7 and `RESULTS.md`.

## Model selection (current, supersedes the headline numbers below)

Classification of MSI status from the 35 ferroptosis genes was redesigned around a locked test set and nested
cross-validation (`ML_AUDIT.md`, `docs/MODEL_SELECTION.md`, `configs/model_selection.yaml`). Honest outcome: the winner
is a calibrated L2 logistic regression (the reference baseline); CV AUC 0.877 ± 0.032, locked-test AUC 0.826 / accuracy 0.871 (majority
baseline 0.840), external CPTAC-2 AUC 0.970 / accuracy 0.886. The served Ferroptosis Core Classifier is the ensemble that later passed the
predefined ensemble gate (test accuracy 0.863; see the table below and `results/final_model_registry.json`). **The 90% accuracy target was not achieved.**
The drug-response regression (single Extra Trees model, target ln IC50) is weak: held-out MAE 0.866 vs 0.868 for the per-drug-mean baseline on 9 cell lines (about 6% better in cross-validation).
The application's production models are looked up from `models/champions.json`, never hard-coded.

## Core classifier vs. extended models (read this before quoting any number)

| Model | Genes | CV acc. | Locked-test acc. | Status |
|---|---|---|---|---|
| **Ferroptosis Core Classifier** (served by default) | 35 ferroptosis genes | 0.877 ± 0.032 | 0.863 (best single model: 0.871) | combined (soft-voting) model |
| 35 genes + mismatch repair (served as the optional extended model) | 35 + MLH1/MSH2/MSH6/PMS2 | 0.930 ± 0.017 | 0.914 | single logistic regression |
| **Extended Molecular Model** (benchmarked) | 35 + 4 MMR + 14 immune genes | 0.931 ± 0.024 | 0.921 | combined model; not the served one |
| Whole-transcriptome model | ~16k genes | not completed | not completed | experiment incomplete: no result is claimed |

The 35-gene ferroptosis core does **not** reach 90%. The extended models do, but the gain is associated substantially with
mismatch-repair genes, which define the MSI-H label biologically; they are not ferroptosis genes, and the extended models must not be
described as ferroptosis classifiers. Source: `results/final_model_report.md` (generated from stored results by `scripts/write_final_report.py`).

## Results of the earlier pipeline (superseded; kept for history — they predate the label fix and the locked protocol)

- **MSI classification** (primary, non-circular): Logistic Regression, ROC-AUC **0.917** using 35 raw ferroptosis genes.
- **Feature-engineering finding**: collapsing 35 genes into a compact signature score *hurts* MSI classification (ROC-AUC drops to 0.67) — a genuine negative result.
- **Whole-transcriptome baseline** (~19,200 genes, PCA-reduced): Random Forest ROC-AUC **0.986**, (pre-correction figure, NOT a current result: it predates the corrected labels and the locked protocol; the re-run under the current protocol did not complete and is reported as such).
- **External validation** (independent CPTAC-2 cohort, 28-gene intersection variant): ROC-AUC 0.94–0.96, matching or exceeding internal held-out performance, though default-threshold accuracy did not transfer as cleanly (see `RESULTS.md` §4d).
- **Pooled dose-response regression**: Random Forest, R²=**0.617** (LN_IC50), R²=0.436 (AUC), on entirely held-out cell lines.
- **Real (measured) dose-response curves**: fit from the raw 2.1GB GDSC per-well viability file for the 4 representative drugs, replacing the earlier illustrative reconstructions.
- **Representative single-drug models** (ferroptosis genes only, n≈44, LOOCV): Random Forest R² 0.21–0.32 across 5-Fluorouracil, Oxaliplatin, Irinotecan, Sorafenib.
- **Interpretability**: `SLC7A11` (a canonical core ferroptosis regulator) ranks among the top predictive genes in both the classification and regression tasks.
- **Prognosis validation** (non-circular; new this session): the Cox ferroptosis risk score, fit on training patients only, scores concordance index **0.692** and log-rank p=**0.039** against held-out patients' actual survival (`RESULTS.md` §9).
- **Drug-resistance classification** (new this session): the pooled cross-drug model is close to chance (ROC-AUC 0.45–0.54); an oxaliplatin-specific variant recovers real signal (ROC-AUC ~0.68) — an honest split result, reported both ways (`RESULTS.md` §10).

## Limitations

GDSC dose-response curves for non-representative drugs are illustrative
reconstructions from the fitted IC50, not raw viability data; the
survival-risk-group *classification* (Experiment 3) is circular by
construction (label built from the same features it is then classified
on) — the non-circular prognostic-value check is the concordance
index/log-rank result above; the pooled drug-resistance classifier performs
close to chance. Full detail: `docs/LIMITATIONS.md`. Future work:
`docs/FUTURE_WORK.md`.

## Reproducibility

`RANDOM_SEED = 42` everywhere. All preprocessing decisions logged to
`docs/review_evidence/`. All processed data, results, figures, and tables are
regenerated by `python main.py` from the cached raw snapshots in `data/raw/`.

## Repository structure

```
configs/                    data_config.yaml, model_config.yaml (thresholds, paths, seed,
                             hyperparameter documentation, deployed-model choice)
data/
  README.md                 role of each data/ subdirectory
  raw/{tcga,gdsc}/          raw, unmodified data snapshots
  processed/                cleaned, feature-engineered datasets
  external/, metadata/      reserved for future use (see data/README.md)
  ferroptosis_genes_literature.csv
  unified_gene_signature.csv
  gene_selection_method.md
src/                        pipeline modules (see "Pipeline stages" above) +
                             config.py, model_registry.py, inference.py,
                             validation.py, reporting.py
scripts/
  train_and_persist.py      trains + persists every deployable model
app/                        Streamlit application (main.py + pages/, see "Application" above)
models/                     persisted model artifacts (models/registry.json + .joblib files,
                             gitignored/regenerable — run scripts/train_and_persist.py)
tests/                      pytest test suite (95 tests: unit, integration, Streamlit AppTest e2e)
results/                    machine-readable CSV results + reports/ (generated HTML reports)
tables/                     rounded comparison tables (CSV)
figures/                    all generated plots (PNG)
docs/                       DATASET.md, METHODOLOGY.md, ARCHITECTURE.md, MODEL_CARD.md,
                             DATA_LEAKAGE_AUDIT.md, DEMO_GUIDE.md, API_OR_PIPELINE_GUIDE.md,
                             DATA_GUIDE.md, SCIENTIFIC_METHODS.md, RESEARCH_GAP.md,
                             LIMITATIONS.md, FUTURE_WORK.md, REVIEW_2_CONTENT.md, VIVA.md,
                             RUBRIC_MAPPING.md, REFERENCES.md, CSV_FORMAT.md, review_evidence/
main.py                     pipeline entry point
requirements.txt
PROJECT_AUDIT.md             audit performed before this session's upgrade
PROJECT_STATUS.md            what exists / what's missing, phase-by-phase
IMPLEMENTATION_STATUS.md     honest per-module implementation percentage
RESULTS.md                   full results with discussion
UI_AUDIT.md                  application UX audit + changes made (2026-09-17 session)
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

This project's broader literature basis (7 papers selected for patient
risk classification, prognosis, drug-response, and drug-resistance
prediction, and an honest paper-by-paper mapping of what is directly
implemented vs. conceptually related but not built) is in
`docs/REFERENCES.md`.
