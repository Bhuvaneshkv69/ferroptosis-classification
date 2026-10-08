# Scientific summary (factual source for the paper)

_Generated from the frozen registry by `scripts/write_paper_artifacts.py`. Not the paper: a checked set of facts and wording._

## 1. Problem

Can microsatellite-instability status (MSI-H vs MSS/MSI-L) of colorectal tumours be classified from the expression of a literature-derived ferroptosis gene panel, and does adding mismatch-repair and immune genes help? A secondary, exploratory question is whether the same panel carries cell-line drug-sensitivity signal (ln IC50). Research use only: outputs are model classifications and computational in-vitro estimates, not diagnoses, doses or treatment recommendations.

## 2. Dataset

* **Classification:** TCGA PanCancer Atlas COAD/READ (cBioPortal); 555 patients with an MSI MANTIS score (89 MSI-H, 16%); 37 patients without a score were excluded. MSI-H = MANTIS ≥ 0.4 (DNA-based label, not computed from expression).
* **Split:** one locked split (fingerprint `9fbc975e2a0b…`, seed 42): 416 development / 139 locked test (22 MSI-H).
* **External (supplementary):** External-cohort data were evaluated separately from the primary locked-test evaluation and were not used for model selection or the primary performance claims. (CPTAC-2 colon: 110 samples, 105 with a cohort MSI_STATUS label and expression: 81 MSS, 24 MSI-H; the cohort's own MSI call, whereas TCGA labels are MANTIS ≥ 0.4; models refit on the shared genes: core 28 of 35, core_plus_mmr 32 of 39, extended_molecular 43 of 53.)
* **Drug response:** GDSC colorectal cell lines (44 lines), target ln IC50.

## 3. Ferroptosis feature set

35 genes, derived from the literature; no gene was chosen using this study's expression data or MSI labels. Construction: union of 8 core-mechanism genes (a published mechanistic review) and 27 genes from four published colorectal-cancer ferroptosis prognostic/metastasis signatures; src/ferroptosis.py reads only the literature table (data/ferroptosis_genes_literature.csv). Caveat: docs/DATASET.md states that TCGA COADREAD is the cohort used by all four source signature papers, so the panel is literature-derived but not independent of the TCGA cohort (the original selection was for prognosis/metastasis, not MSI). Model name: **Ferroptosis Core Classifier**.

## 4. Classification methodology

* Learners: logistic regression (L2, elastic net), linear and RBF SVM, random forest, extra trees, histogram and gradient boosting, XGBoost, LightGBM (CatBoost on the core).
* Nested stratified cross-validation (5 outer folds × 2 repeats, 5-fold inner tuning); preprocessing, tuning, sigmoid calibration, ensemble weights and decision thresholds (MCC objective) are all fit inside training folds.
* Ensembles (soft vote, hard vote, stacking on out-of-fold predictions) are adopted only if they pass a predefined gate (CV AUC gain, balanced accuracy, minority recall, stability, member complementarity, locked test, leakage canary, reproducibility).
* The Ferroptosis Core Classifier is an equal-weight soft-voting model of 5 calibrated learners (logreg_l2, svm_rbf, random_forest, extra_trees, xgboost); its CV AUC gain over the best single model was only about +0.004.

## 5. Extended molecular features

* **35-gene + MMR Extended Model (39 genes):** the 35 genes + MLH1, MSH2, MSH6, PMS2.
* **Extended Molecular Model (53 features):** the 39 + 14 immune / interferon-γ genes (CD274, PDCD1, CTLA4, LAG3, TIGIT, HAVCR2, CD8A, CD8B, GZMA, GZMB, PRF1, IFNG, CXCL9, CXCL10).
* Rationale: MSI-H is mismatch-repair deficiency by definition, so MMR expression is a direct biological correlate of the label. These are **not** ferroptosis genes and the models are not ferroptosis classifiers.

## 6. Validation strategy

* Cross-validation on the development set for every choice; one locked test of 139 patients with 22 MSI-H, so positive-class metrics are imprecise. 95% interval = class-stratified nonparametric percentile bootstrap (2,000 resamples of the locked-test patients; the fitted model is fixed, so it reflects test-set sampling only). CV ± = standard deviation across the 10 outer folds (5 folds × 2 repeats), not an interval.
with 22 MSI-H, so positive-class metrics are imprecise.
* Permutation-label canary (AUC ≈ 0.5 expected): Ferroptosis Core Classifier 0.48, 35-gene + MMR Extended Model 0.52, Extended Molecular Model 0.52.
* External cohort: External-cohort data were evaluated separately from the primary locked-test evaluation and were not used for model selection or the primary performance claims. (CPTAC-2 colon: 110 samples, 105 with a cohort MSI_STATUS label and expression: 81 MSS, 24 MSI-H; the cohort's own MSI call, whereas TCGA labels are MANTIS ≥ 0.4; models refit on the shared genes: core 28 of 35, core_plus_mmr 32 of 39, extended_molecular 43 of 53.)
* Drug response: whole cell lines held out (grouped CV) plus a separate locked set of 9 cell lines.

## 7. Final classification results

| Model | CV accuracy | Locked-test accuracy (95% interval) | Test AUC | MSI-H recall | Status |
|---|---|---|---|---|---|
| Ferroptosis Core Classifier | 0.877 ± 0.032 | 0.863 (0.82–0.91) | 0.832 | 0.364 | In production (default) |
| 35-gene + MMR Extended Model | 0.930 ± 0.017 | 0.914 (0.88–0.95) | 0.886 | 0.545 | In production (optional extended slot) |
| Extended Molecular Model | 0.931 ± 0.024 | 0.921 (0.88–0.96) | 0.886 | 0.591 | Benchmarked candidate; not serving |

* Majority-class accuracy is 0.840; MSI-H recall on the locked test is 0.36 for the core and 0.59–0.55 for the extended models: most gain over the majority class comes from specificity.
* The 35-gene core did **not** reach 90% accuracy in cross-validation or on the locked test. The extended models exceed 90%, but through mismatch-repair (and immune) genes.
* The Extended Molecular Model's advantage over the 35-gene + MMR Extended Model is one patient on the locked test (inside overlapping 95% intervals); the one-SE rule therefore serves the simpler 39-gene model.
* Observation (not corrected): locked-test AUC is below the cross-validated AUC for every model (gap in CV SDs: Ferroptosis Core Classifier 1.7, 35-gene + MMR Extended Model 1.1, Extended Molecular Model 1.1), while accuracy barely changes; with 22 MSI-H test patients the AUC interval is wide. The 53-feature model's higher CV AUC than the 39-gene model's is not reproduced on the locked test.
* Whole-transcriptome: The whole-transcriptome experimental arm was not completed within the validated evaluation protocol and was therefore excluded from the final performance claims.

## 8. Drug-response results

* Model: extra trees regressor (single model; not an ensemble) on 35 ferroptosis-gene expression values + drug identity; target ln IC50 (in vitro).
* **Cross-validation** (development cell lines only (35 lines)): MAE 1.009 ± 0.168 vs per-drug-mean baseline 1.070: 5.7% better, in 5 of 5 folds.
* **Locked test** (9 cell lines (2368 line-drug rows) held out before any modelling; disjoint from the cross-validation data): MAE 0.866 vs 0.868: 0.2% better, well inside the 95% interval of the MAE.
* Characterisation: preliminary / weak. Most accuracy comes from knowing which drug it is. The multi-model drug-response ensemble was not completed.

## 9. Limitations

* Only 22 MSI-H patients in the locked test; one cohort for development; external cohort needs a refit on shared genes.
* MMR-containing models are partly circular with the label biology; they are not evidence about ferroptosis.
* Panel provenance: docs/DATASET.md states that TCGA COADREAD is the cohort used by all four source signature papers, so the panel is literature-derived but not independent of the TCGA cohort (the original selection was for prognosis/metastasis, not MSI).
* The external cohort uses its own MSI_STATUS call (not MANTIS), and models are refit on the shared genes, so it does not evaluate the served models directly; patient independence from TCGA rests on the cohort description plus a check that no sample or patient IDs overlap.
* Locked-test discrimination (AUC) is below the cross-validated AUC for all three classifiers (see section 7).
* The core ensemble's gain over its best single model is within noise; the single model is a defensible equivalent.
* Only 44 colorectal cell lines for drug response; no clinical dosing or treatment claim is made.
* Whole-transcriptome and the drug-response ensemble were not completed.

## 10. Reproducibility and provenance

* Frozen results: `results/final_model_registry.json` (one authoritative result per model); report: `results/final_model_report.md`; paper table: `results/paper_results_table.md`.
* Seed 42; locked split recorded in `data/processed/locked_split_*.json` with a fingerprint and a log of every locked-test scoring.
* Every authoritative classifier has exactly one logged locked-test scoring, and its stored metrics equal that scoring (max difference 6e-17).
* Disclosure: one crash (floating-point rounding, before the locked test was touched for that model) occurred while building the extended models; probabilities are now clipped to [0, 1] in the metric function, and the rerun scored the locked test once, was not used for tuning, and no unlogged re-scoring was found. The 35- and 39-gene results predate that numerical fix and were not recomputed.
* Every candidate in the screening tables was scored once on the locked test for reporting; selection used cross-validation only.
* Artifacts: `models/` (see the registry for paths); commands: `python scripts/run_ensemble_lab.py`, `python scripts/write_final_report.py`, `python scripts/write_paper_artifacts.py`; tests: `pytest tests`.

## 11. Final contribution

A leakage-controlled, fully audited comparison of single and ensemble classifiers for MSI status from a ferroptosis gene panel, showing that the panel alone reaches about 86% locked-test accuracy (majority baseline 84%), that adding mismatch-repair genes lifts it above 90%, and that a preliminary drug-response regression adds little beyond per-drug means. The contribution is the honest, reproducible quantification of what the ferroptosis panel does and does not carry, not a claim that ferroptosis genes classify MSI at 90%.
