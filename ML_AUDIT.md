# ML Audit — before the model-selection redesign

Scope: the existing classification (MSI status from 35 ferroptosis genes) and drug-response pipelines.
Every number below is either read from `results/*.csv` of the *previous* pipeline or from
`results/ml_audit_facts.json` (produced by `scripts/ml_audit.py`). The audit found real defects; they
are listed first because they change how the old results should be read.

## 1. Headline findings

| # | Finding | Severity | Status |
|---|---|---|---|
| 1 | **37 patients without an MSI score were silently labelled "MSS/MSI-L"** (`np.where(NaN >= 0.4, …)` is `False`). Negative class was 503 instead of 466; these 37 were also in the old train/test sets. | High (label noise) | Fixed in `src/preprocessing.py`; processed data regenerated |
| 2 | **The old "best model" was chosen by looking at the test set.** `configs/model_config.yaml` deployed Logistic Regression because it had "the highest ROC-AUC" on the test split; the feature set (raw genes over the 2-feature signature) was also decided from test results. | High (selection bias) | Replaced: selection now uses development-set nested CV only |
| 3 | **Old test set is tiny for the class that matters**: 119 patients, ~18 MSI-H. Sensitivity 0.889 means 16/18; its 95% interval is about ±0.15. Differences between old models (e.g. AUC 0.917 vs 0.904) are noise. | High (precision of estimate) | Locked test is reported with bootstrap CIs; CV is the primary estimate |
| 4 | **Accuracy 0.849 ≈ the majority-class baseline** (always predicting "MSS/MSI-L" scores ≈ 0.84–0.85). Reported precision was 0.50. Accuracy was not a meaningful headline. | High (misleading metric) | Majority baseline now printed next to every accuracy; AUC / MCC / balanced accuracy emphasised |
| 5 | **Experiment 3 ("risk group", accuracy 0.92–0.98) is circular**: the label is a thresholded linear function of the same 35 genes. It must not be read as evidence of a good model. | High if quoted | Excluded from production, selection and UI |
| 6 | Drug-resistance classifier (pooled GDSC) scored AUC 0.45–0.54 — chance. | Medium | No longer exposed in the UI |
| 7 | Pooled drug-response regression R² mostly reflects **which drug** is being predicted (drug identity explains ~75% of LN(IC50) variance in-sample), not the cell line's expression. No baseline was reported. | Medium | New benchmark adds a per-drug-mean baseline and a "what does expression add" test |
| 8 | Only **44 colorectal cell lines** underlie the drug-response model; the old test held out 9. | Medium (inherent) | Documented; locked test is reported with cell-line bootstrap CIs |

## 2. CURRENT PERFORMANCE (previous pipeline, test split 80/20, seed 42, n_test = 119)

| Task | Model | Accuracy | AUC | F1 | Note |
|---|---|---|---|---|---|
| MSI (exp1, 35 raw genes) | Logistic Regression | 0.849 | 0.917 | 0.640 | deployed; chosen on test AUC |
| | Random Forest | 0.815 | 0.902 | 0.542 | |
| | SVM | 0.765 | 0.904 | 0.517 | |
| | XGBoost | 0.840 | 0.861 | 0.174 | recall 0.11 |
| MSI (exp2, 2 engineered scores) | best AUC 0.672 | | | | signature collapse loses the information |
| Risk group (exp3) | LR | 0.983 | 1.000 | | **circular**, ignore |
| External (CPTAC-2, 28/35 genes) | LR | 0.819 | 0.959 | 0.716 | models trained on the 28-gene intersection |
| Whole transcriptome (18 302 genes → PCA 50) | RF | 0.941 | 0.986 | 0.774 | context only: the ceiling is set by the feature panel |

## 3. CURRENT PIPELINE

* **Data**: TCGA COAD/READ, 592 patients after QC (555 with an MSI score), 35 literature-curated ferroptosis genes,
  log2(RSEM+1). GDSC2 colorectal: 44 cell lines × 268 drugs = 11 511 rows, LN(IC50) target.
* **Preprocessing**: `StandardScaler` inside an sklearn `Pipeline` (fit on the training fold only). No missing values in the 35-gene panel.
* **Feature selection**: none — the 35 genes were fixed from the literature before any modelling (not data-driven, so no selection leakage).
  Exp2 recomputes z-score/PCA on the training split only (a previous whole-cohort fit was already fixed).
* **Models**: LogisticRegression, RandomForest, SVM, XGBoost. Small `GridSearchCV` (≤ 12 settings, 5-fold, ROC-AUC) on the 80% training split.
* **Validation**: single stratified 80/20 split, no nested CV, no repeated CV, no variance estimate, no CIs.
* **Registry**: `models/registry.json` append-only; the deployed model was named in `model_config.yaml` (hard-coded).

## 4. POTENTIAL LEAKAGE — the eleven questions

| Question | Answer |
|---|---|
| 1. Is there data leakage? | Not through preprocessing or feature selection. **Yes through model/feature-set choice made on the test set (finding 2)** and label noise (finding 1). |
| 2. Is feature selection done correctly? | There was none (fixed gene panel). The new pipeline compares eight selection strategies, all fit inside the training folds. |
| 3. Is preprocessing fitted only on training folds? | Yes (scaler is a pipeline step; exp2/exp3 transforms were fixed in an earlier session). Verified again by unit tests. |
| 4. Is the test set untouched? | Not in the scientific sense: it was used to rank models and pick the deployed one. A **new locked split** is now written to disk and its use is logged. |
| 5. Is the accuracy trustworthy? | The AUC/CV-grid parts are sound; the headline accuracy is not (finding 4) and the test-based choice is optimistic (finding 2). |
| 6. Is class imbalance affecting the metrics? | Yes: 16.0% MSI-H (89 / 555). Accuracy is dominated by the majority class. |
| 7. Too small / high-dimensional for some models? | 555 samples, 89 positives, 35 features = 2.5 positive events per feature. Fine for regularised linear models and kernels; risky for boosted trees, and clearly too small for deep nets. |
| 8. Duplicates / near-duplicates? | None: 0 exact duplicate rows, 0 duplicate patients, maximum pairwise correlation between two samples 0.85 (`results/ml_audit_facts.json`). |
| 9. Same patient/cohort in train and test? | One sample per patient, so no patient straddles a split. |
| 10. Batch / cohort leakage? | TCGA is one cohort but 37 tissue-source sites; MSI-H frequency differs by site (χ² p = 0.005) and by tumour location (p = 1e-4). Random splits therefore mix sites. The benchmark adds a leave-sites-out CV as a sensitivity check. CPTAC-2 is used as the independent cohort. |
| 11. Is the label derived from the features? | The MSI label comes from MANTIS, a DNA-sequencing score; the strongest single-gene Spearman correlation with it is 0.28. **Not circular** (unlike exp3, finding 5). |

## 5. DATA QUALITY

* 555 labelled patients; 89 MSI-H (16.0%), 466 MSS/MSI-L. 37 patients without a MANTIS score are now excluded, not mislabelled.
* No missing values in the 35 gene columns; no near-duplicates; one row per patient.
* Largest single tissue source site holds 24.5% of patients; 37 sites in total.
* External cohort: CPTAC-2 colon, 105 samples (24 MSI-H), but only **28 of the 35 genes** are reported there, so the external model must be a refit on the shared genes.

## 6. MODEL LIMITATIONS

* The panel itself is the limit: the same samples and target reach AUC 0.986 with the whole transcriptome (includes MLH1/immune genes), versus ~0.88–0.92 with the 35 ferroptosis genes. No amount of model tuning on 35 genes is expected to close that gap; honest accuracy above 90% on this panel is plausible only because the minority class is 16%, and it must be judged against the 84% majority baseline.
* 18–22 positives in any held-out test set → wide confidence intervals; single-split conclusions are fragile.
* `SVC(probability=True)` is deprecated in scikit-learn 1.9 and will be removed in 1.11 (pin the version or migrate to `CalibratedClassifierCV`).
* Drug response: 44 cell lines; cell-line microarray vs tumour RNA-seq is a platform mismatch that no validated mapping bridges.

## 7. RECOMMENDED PIPELINE (implemented in this change)

1. Fix the label bug; freeze a **locked test set** (25%, stratified, written to `data/processed/locked_split_*.json`, every use logged).
2. On the development set only: **nested cross-validation** (5 outer folds × 2 repeats; inner 5-fold `RandomizedSearchCV`).
   Calibration and the decision threshold (MCC objective) are fit inside the training folds as well.
3. Compare, **without taking a full cross-product**: 9 feature strategies on three reference learners → imbalance handling (none / class-weight / SMOTE-in-fold) → 13 learners + voting + stacking → calibration (none / Platt / isotonic).
4. Pick each winner with a configurable rule (`configs/model_selection.yaml`): best primary metric, but the **simplest model within one standard error** wins, and unstable models are excluded.
5. Refit the chosen procedure on the development set; score the locked test **once**; refit on the 28 shared genes and score CPTAC-2 **once**.
6. Leakage canaries: label-permutation nested CV (AUC must be ≈ 0.5), leave-sites-out CV, patient/sample overlap checks.
7. Registry stores CV / locked-test / external metrics, features, hyper-parameters, calibration; the production model is a pointer (`models/champions.json`) moved only by a promotion gate (CV gain **and** test no worse **and** stability **and** leakage checks).
8. Drug response: same protocol with whole-cell-line grouping, plus drug-mean baselines.
