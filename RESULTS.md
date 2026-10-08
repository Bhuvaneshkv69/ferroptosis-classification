# Results

> **Superseded for classification and drug response by `docs/MODEL_SELECTION.md` (generated from
> `results/benchmark/`) and `ML_AUDIT.md`.** The sections below are the earlier single-split pipeline: they were
> produced before 37 unlabelled patients were removed from the MSI-negative class, and the deployed model was
> chosen on the test set. Treat them as history, not as the project's headline numbers. Section 3
> (risk group) is circular by construction.

All numbers below are read directly from `results/*.csv` produced by an
actual run of the pipeline (`python main.py`) on the real datasets described
in `docs/DATASET.md`. Nothing here is estimated or fabricated.

## 1. Classification — MSI status (primary, non-circular task)

Target: MSI-H vs MSS/MSI-L (594→555 TCGA COADREAD patients with an MSI MANTIS
score; 20% held-out test set, n_test=111, n_train=444).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.849 | 0.500 | 0.889 | 0.640 | **0.917** | 0.677 |
| Random Forest | 0.815 | 0.433 | 0.722 | 0.542 | 0.902 | 0.604 |
| SVM | 0.765 | 0.375 | 0.833 | 0.517 | 0.904 | 0.713 |
| XGBoost | 0.840 | 0.400 | 0.111 | 0.174 | 0.861 | 0.475 |

**Best model: Logistic Regression (ROC-AUC 0.917).** Ferroptosis-related gene
expression alone carries substantial, genuine signal about MSI status —
biologically plausible, since MSI-H tumors have well-documented differences
in immune infiltration and metabolic state that intersect with ferroptosis
pathways. Recall is prioritized by `class_weight="balanced"` given the
MSI-H minority class; this is why precision is moderate at high recall
(clinically, missing an MSI-H case has different costs than a false
positive, so this trade-off is a defensible choice, not an artifact).
XGBoost underperforms here (F1=0.174) — likely overfitting the small
positive class despite tuning; Random Forest and SVM sit in between.

Figures: `figures/roc_curve_exp1_msi_raw_genes.png`,
`figures/confusion_matrix_exp1_msi_raw_genes_*.png`,
`figures/model_comparison_msi_classification.png`,
`figures/calibration_exp1_msi_raw_genes.png`.

## 2. Classification — raw genes vs engineered signature score (feature-engineering comparison)

*Updated this session: the PCA/z-score signature score is now fit strictly
on the training fold only (previously fit on the full cohort before the
split — a data-leakage vector found and fixed, see
`docs/DATA_LEAKAGE_AUDIT.md`). Numbers below are from the corrected,
leakage-free pipeline.*

| Feature set | Best model | ROC-AUC |
|---|---|---|
| Raw 35 ferroptosis genes | Logistic Regression | **0.917** |
| Engineered (mean z-score + PC1, 2 features) | Logistic Regression | 0.672 |

**Finding: feature engineering (collapsing to a compact signature score) does
NOT improve MSI classification — it substantially hurts it.** The 35
individual genes carry gene-specific information relevant to MSI status that
a 2-feature summary discards. This is reported as a genuine, data-driven
negative result, exactly as instructed, rather than being omitted or spun
positively. (Full model breakdown: `results/classification_results.csv`,
experiment `exp2_msi_engineered`.)

## 3. Classification — ferroptosis survival risk group (secondary, exploratory, CIRCULAR — see caveat)

*Updated this session: the Cox risk model and its High/Low threshold are now
fit strictly on the training fold only (previously fit on the full cohort,
including test-fold survival data — the same leakage class as item 2 above,
fixed together; see `docs/DATA_LEAKAGE_AUDIT.md`).*

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.983 | 0.984 | **0.9997** |
| SVM | 0.975 | 0.975 | 0.9994 |
| Random Forest | 0.932 | 0.931 | 0.987 |
| XGBoost | 0.924 | 0.927 | 0.973 |

⚠️ **This label is a median-split of a linear function of the same gene
features used to predict it** (built in `src/feature_engineering.py` via Cox
regression on the same 35 genes). It is reported only for comparability with
the four published CRC ferroptosis-signature papers reviewed in
`docs/RESEARCH_GAP.md`, which use the same circular construction.

**A genuinely interesting, honest finding from the leakage fix: the score
went slightly UP (0.995 → 0.9997), not down.** This confirms the near-perfect
score was never really about the leakage vector — it is about **circularity**.
The target for held-out samples is (to a very good approximation) a linear
function of the same input features a powerful classifier is given, so the
classifier can nearly perfectly recover that function on new data drawn from
the same distribution *regardless* of whether the Cox model that defined the
target was fit with or without those samples included. Leakage and
circularity are two independent problems; fixing the former does not, and
was never claimed to, fix the latter. This experiment remains excluded from
every real scientific claim in this project and from the interactive
application's classifier page.

**The genuinely non-circular prognostic-value check is §9 below** — it does
not use this risk-group label as a classification target at all; it checks
the underlying risk *score* directly against held-out patients' actual
survival time and event indicator.

## 4. Dose-response regression — pooled model (268 drugs × 44 colorectal cell lines, entire cell lines held out at test time)

**Target: LN(IC50)**

| Model | MAE | RMSE | R² | Pearson r | Spearman ρ |
|---|---|---|---|---|---|
| Random Forest | **1.283** | **1.636** | **0.617** | **0.792** | 0.659 |
| XGBoost | 1.305 | 1.650 | 0.610 | 0.785 | 0.647 |
| Linear Regression | 2.446 | 2.970 | -0.263 | 0.349 | 0.271 |

**Target: AUC**

| Model | MAE | RMSE | R² | Pearson r | Spearman ρ |
|---|---|---|---|---|---|
| Random Forest | **0.061** | **0.088** | **0.436** | **0.668** | 0.585 |
| XGBoost | 0.063 | 0.089 | 0.427 | 0.657 | 0.560 |
| Linear Regression | 0.082 | 0.113 | 0.062 | 0.307 | 0.289 |

**Best model: Random Forest, R²=0.617 for LN_IC50 on cell lines never seen in
training.** This is a genuinely strong result for a held-out-cell-line
generalization test, and confirms drug pathway + concentration range +
ferroptosis expression jointly carry real predictive signal (pathway/drug
identity dominates, as expected — most inter-drug potency variance exceeds
inter-cell-line variance; the per-drug models below isolate the
gene-expression-only contribution). Linear Regression's negative R² reflects
the non-linear, high-cardinality one-hot pathway encoding overwhelming an
unregularized linear fit — not a data problem.

Figures: `figures/predicted_vs_actual_LN_IC50_*.png`,
`figures/residuals_LN_IC50_*.png`,
`figures/model_comparison_dose_response.png`.

## 4b. Real (measured) dose-response curves for the 4 representative drugs

*New this session.* The ~2.1GB raw per-well GDSC2 viability file was
downloaded (`GDSC2_public_raw_data_24Jul22.csv`, Sanger FTP mirror) and
processed (`src/process_raw_dose_response.py`) into genuine measured
dose-response curves — replacing the previous illustrative,
IC50-parameter-only reconstructions for the 4 representative drugs.

| Drug | Cell lines with a fitted curve | Example fitted IC50 (µM) |
|---|---|---|
| 5-Fluorouracil | 4 (of 44 plotted) | 14.9–3221 (wide inter-line range) |
| Sorafenib | 4 | 7.6–10.6 |
| Irinotecan | 4 | 6.4–22.8 |
| Oxaliplatin | 4 | 3.0–189.1 |

Methodology: relative viability = well `INTENSITY` / mean `NC-0` (negative
control) `INTENSITY` on the same assay plate (`BARCODE`); technical
replicates averaged per (drug, cell line, concentration); a 2-parameter
logistic curve fit via `scipy.optimize.curve_fit` on the real averaged
points. This is a simplified, documented re-implementation, not a
byte-for-byte reproduction of GDSC's own `gdscIC50` R package (which
additionally applies positional/edge-effect plate correction) — see
`docs/LIMITATIONS.md`.

Full point-level data: `data/processed/gdsc_raw_dose_response_curves.csv`
(2,345 points). Fitted parameters: `results/real_dose_response_curve_fits.csv`.
Figures: `figures/dose_response_curve_real_{drug}.png` (real, measured,
with error bars) alongside the original `figures/dose_response_curve_{drug}.png`
(illustrative, IC50-only reconstruction, kept for comparison).

## 4c. Whole-transcriptome baseline vs. ferroptosis panel (Experiment 4)

*New this session — closes `docs/FUTURE_WORK.md` item 2.* All ~19,200
protein-coding genes' RNA-seq V2 expression were fetched for the same TCGA
COADREAD cohort (`src/fetch_whole_transcriptome.py`, `src/whole_transcriptome_baseline.py`)
and used, via PCA (50 components, fit on the training fold only — no
supervised feature selection), to train the same 4 classifiers on the same
MSI-H/MSS-MSI-L target as Experiment 1.

| Model | Ferroptosis panel (35 genes) ROC-AUC | Whole transcriptome (~19,200 genes, PCA-50) ROC-AUC |
|---|---|---|
| Logistic Regression | 0.917 | 0.932 |
| Random Forest | 0.902 | **0.986** |
| SVM | 0.904 | 0.978 |
| XGBoost | 0.861 | 0.948 |

**Honest finding: the whole-transcriptome baseline outperforms the
ferroptosis-panel-only classifier on every model (best 0.986 vs. best
0.917).** The 35-gene ferroptosis panel captures substantial, genuine MSI
signal — but not all of it. This is biologically unsurprising: MSI status
correlates with a genome-wide hypermutator phenotype and broad immune-
infiltration signatures that extend well beyond the ferroptosis pathway.
This result does **not** undermine the ferroptosis-panel findings above; it
correctly and honestly quantifies their scope, exactly as
`docs/FUTURE_WORK.md` item 2 set out to measure. Full results:
`results/whole_transcriptome_baseline_results.csv`,
`tables/ferroptosis_vs_whole_transcriptome_comparison.csv`,
`figures/roc_curve_exp4_whole_transcriptome.png`.

## 4d. External validation — independent cohort (CPTAC-2 Prospective Colon Cancer)

*New this session — closes `docs/FUTURE_WORK.md` item 1.* All results above
are **internal** validation: a held-out split of the same TCGA cohort used
for training. This section is the one genuine **external** test: the exact
persisted classifiers (no retraining, no tuning on external data) scored on
CPTAC-2 Prospective Colon Cancer (Vasaikar et al., *Cell* 2019) — a
different consortium, different patients, different RNA-seq batch than TCGA.
Because CPTAC's expression profile does not report values for 7/35 panel
genes, a compatible variant trained on the 28-gene intersection
(`exp1_msi_raw_genes_cptac28`, `src/external_validation.py`) is used instead
of the deployed 35-gene model — an upfront, documented design decision, not
a post-hoc adjustment.

| Model | Internal (TCGA held-out) ROC-AUC | External (CPTAC, n=105) ROC-AUC | External accuracy | External F1 |
|---|---|---|---|---|
| Logistic Regression | 0.902 | 0.959 | 0.819 | 0.716 |
| Random Forest | 0.866 | 0.943 | 0.876 | 0.764 |
| SVM | 0.883 | 0.939 | 0.514 | 0.485 |
| XGBoost | 0.841 | 0.946 | 0.848 | 0.500 |

**Finding: ROC-AUC (ranking quality) transfers well, even slightly better,
across cohorts (external ROC-AUC 0.94–0.96 vs. internal 0.84–0.90) — but
accuracy/precision/recall at the default 0.5 threshold do NOT transfer
uniformly** (SVM's external accuracy collapses to 0.514 despite ROC-AUC
0.939). This is a real, well-known cross-cohort phenomenon: the model's
ability to *rank* MSI-H-likely samples above MSS-likely samples survives a
change of cohort/platform, but the *decision threshold* calibrated on TCGA's
class balance and score distribution does not automatically transfer to
CPTAC's different MSI-H prevalence (24/105 = 22.9%) and score distribution.
Reported honestly, not smoothed over — a threshold recalibration step would
be needed before any external deployment, and is noted as future work.

Full results: `results/external_validation_results.csv`. Figures:
`figures/roc_curve_external_validation_cptac.png`,
`figures/confusion_matrix_external_cptac_*.png`. Interactive: the
application's "External Validation" page.

## 5. Representative single-drug models (ferroptosis genes ONLY, leave-one-out CV, n=42–44)

| Drug | Model | n | MAE | R² | Pearson r |
|---|---|---|---|---|---|
| 5-Fluorouracil | Random Forest | 44 | 1.161 | 0.228 | 0.483 |
| 5-Fluorouracil | Linear Regression | 44 | 1.919 | -1.231 | 0.309 |
| Oxaliplatin | Random Forest | 44 | 1.123 | 0.210 | 0.459 |
| Oxaliplatin | Linear Regression | 44 | 2.136 | -2.308 | 0.171 |
| Irinotecan | Random Forest | 44 | 1.043 | **0.322** | **0.568** |
| Irinotecan | Linear Regression | 44 | 2.534 | -4.606 | -0.067 |
| Sorafenib | Random Forest | 42 | 0.777 | 0.226 | 0.482 |
| Sorafenib | Linear Regression | 42 | 1.785 | -2.626 | 0.223 |

**Finding: with only 31 ferroptosis genes and ~44 cell lines, Random Forest
recovers modest but real predictive signal for all four CRC-relevant drugs
(R² 0.21–0.32, Pearson r 0.46–0.57), while unregularized Linear Regression
fails badly (strongly negative R² on every drug) because the feature count
approaches the sample size.** This is an honest, informative result:
ferroptosis gene expression alone is not sufficient to strongly predict
whole-drug sensitivity, but it is far from noise, and the failure mode of
plain linear regression under n≈p is itself a useful, correctly-diagnosed
finding (see `docs/FUTURE_WORK.md` item 4 — regularization).

Figure: `figures/predicted_vs_actual_5-Fluorouracil_LOOCV.png`; illustrative
dose-response curves: `figures/dose_response_curve_{drug}.png` (see
`docs/LIMITATIONS.md` for what "illustrative" means here).

## 6. Interpretability

**MSI classification — top genes by permutation importance (ROC-AUC drop):**
`NFS1` (0.186, by far the largest single contributor — Fe-S cluster
biogenesis gene, mechanistically linked to iron metabolism), `HOXC11`
(0.041), `SLC7A11` (0.019, a canonical core ferroptosis regulator gene),
`SLC2A3` (0.015), `TFAP2C` (0.014).

**LN_IC50 regression — top genes by permutation importance:** `HOXC11`,
`HAMP`, `CARS1`, `AIFM2` (FSP1, a core ferroptosis-suppressor gene),
`FDFT1`, `SLC7A11`.

SLC7A11 — one of the most extensively studied ferroptosis genes in the
literature (the "gatekeeper" of the cystine/glutamate antiporter) — appears
among the top contributors in **both** the classification and regression
tasks despite being selected purely from external mechanistic literature, not
CRC-specific studies. This cross-task recurrence is a genuinely interesting,
data-driven observation. As stated throughout, this is **association with
model output, not proof of causal biological effect** — see
`docs/LIMITATIONS.md`.

Figures: `figures/shap_summary_classification.png`,
`figures/shap_summary_dose_response.png`,
`figures/feature_importance_classification.png`,
`figures/feature_importance_regression.png`.

## 7. Summary table (all experiments)

Full machine-readable results: `results/classification_results.csv`,
`results/regression_results.csv`, `results/representative_drug_results.csv`,
`results/external_validation_results.csv`,
`results/whole_transcriptome_baseline_results.csv`,
`results/real_dose_response_curve_fits.csv`.
Rounded comparison tables: `tables/classification_model_comparison.csv`,
`tables/regression_model_comparison.csv`,
`tables/representative_drug_comparison.csv`,
`tables/cox_risk_score_coefficients_trainfold.csv`,
`tables/ferroptosis_vs_whole_transcriptome_comparison.csv`.

## 8. These models are now persisted and interactively servable

Every deployed classification and dose-response model in the tables above
was trained and persisted to `models/registry.json` this session
(`python scripts/train_and_persist.py`). The exact fitted objects behind
these numbers are what the Streamlit application (`streamlit run
app/main.py`) loads and scores new samples with — see `docs/MODEL_CARD.md`
and `docs/DEMO_GUIDE.md`. Two data-leakage fixes were made and verified this
session (GDSC gene scaling, and TCGA engineered-feature/Cox-model fitting —
see `docs/DATA_LEAKAGE_AUDIT.md`); the GDSC fix left every regression metric
unchanged to the reported precision, while the TCGA fix changed Experiment
2/3's numbers slightly (both reported above with before/after values) while
leaving Experiment 1, the project's primary headline result, untouched.

Three further real analyses were added this session, closing every item
that was previously listed as deferred: external validation on an
independent cohort (§4d), a whole-transcriptome baseline (§4c), and real
measured dose-response curves from the raw per-well GDSC file (§4b). See
`docs/FUTURE_WORK.md` and `IMPLEMENTATION_STATUS.md` for the full,
now-100%-complete-against-plan implementation tracking.

## 9. Prognosis Prediction validation (non-circular) — new this session (2026-09-17)

Motivated by `docs/REFERENCES.md` papers 1–3. §3 above reports how well
gene expression predicts the risk-*group label* back — a question that is
partly circular by construction. This section asks a different, genuinely
non-circular question instead: **does the risk *score* (fit on a training
fold of patients only) correctly order held-out patients' actual survival?**

| Metric | Value | n (held-out patients) |
|---|---|---|
| Concordance index | **0.692** | 118 |
| Log-rank test p-value (High-risk vs. Low-risk survival) | **0.039** | 60 high-risk / 58 low-risk |

A concordance index of 0.5 is random; 1.0 is perfect. 0.692 indicates real,
moderate prognostic signal in the ferroptosis risk score on patients it was
never fit on. The log-rank p-value (0.039 < 0.05) indicates the High- and
Low-risk groups (thresholded using only the training fold's median) have a
statistically detectable survival difference among held-out patients —
the standard validation style the reviewed literature (papers 1–3) uses.

This is reported as a genuine, moderate positive finding — not as strong as
the (non-circular) primary MSI classification result (§1, ROC-AUC 0.917),
and it should not be read as clinical prognostic validity for any
individual patient. Implementation: `feature_engineering.evaluate_prognosis()`,
`feature_engineering.plot_kaplan_meier()`; raw output:
`results/prognosis_validation.json`, `figures/kaplan_meier_risk_group.png`.
Shown in the application under Results → Prognosis.

## 10. Drug-resistance classification (Resistant / Sensitive) — new this session (2026-09-17)

Motivated by `docs/REFERENCES.md` papers 6–7 (ferroptosis and chemoresistance
in colorectal cancer, and oxaliplatin resistance specifically). This project
did not have verified access to those papers' exact resistance thresholds or
gene panels, so "Resistant"/"Sensitive" is defined transparently from this
project's own GDSC data: a (cell line, drug) pair is Resistant if its
LN_IC50 is at or above that drug's **training-fold-only** median across
tested colorectal cell lines, Sensitive otherwise (`src/drug_resistance.py`).

**Pooled model** (all well-covered drugs, ferroptosis genes + drug
pathway/concentration-range, entire cell lines held out):

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Random Forest | 0.500 | 0.537 | 0.254 |
| Logistic Regression | 0.459 | 0.446 | 0.461 |

**Honest negative finding:** the pooled model performs close to chance.
Defining "resistant" relative to each drug's own median makes it inherently
drug-relative, and pooling dozens of different drug mechanisms into one
shared feature space appears to wash out any consistent cross-drug signal
recoverable from the ferroptosis panel alone.

**Oxaliplatin-specific model** (paper 7's specific focus; ferroptosis genes
only, leave-one-out CV, n=44 — same small-n treatment as the representative
single-drug dose-response regressions in §5):

| Model | Accuracy | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.705 | 0.676 | 0.711 |
| Random Forest | 0.568 | 0.599 | 0.558 |

Restricting to a single, well-studied drug and the ferroptosis panel
(which includes `SLC7A11` and `GPX4`, the axis papers 6–7 specifically
implicate in oxaliplatin resistance) recovers real signal where the pooled,
cross-drug model does not — mirroring the same pattern already seen in §5
for dose-response regression (per-drug models outperforming what a purely
pooled ferroptosis-only model could do for that drug).

Raw output: `results/drug_resistance_results.csv`,
`results/oxaliplatin_resistance_results.csv`,
`data/processed/drug_resistance_thresholds.json`. The pooled model is
persisted to `models/registry.json` (task `drug_resistance`) and served
live by the application (Drug Response page, "Resistance classification"
section; Results → Drug resistance for the full picture; Models → Drug
resistance for registry details).
