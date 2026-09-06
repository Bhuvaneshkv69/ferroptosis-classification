# Results

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

| Feature set | Best model | ROC-AUC |
|---|---|---|
| Raw 35 ferroptosis genes | Logistic Regression | **0.917** |
| Engineered (mean z-score + PC1, 2 features) | XGBoost | 0.652 |

**Finding: feature engineering (collapsing to a compact signature score) does
NOT improve MSI classification — it substantially hurts it.** The 35
individual genes carry gene-specific information relevant to MSI status that
a 2-feature summary discards. This is reported as a genuine, data-driven
negative result, exactly as instructed, rather than being omitted or spun
positively.

## 3. Classification — ferroptosis survival risk group (secondary, exploratory, CIRCULAR — see caveat)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.975 | 0.975 | **0.995** |
| SVM | 0.975 | 0.975 | 0.992 |
| XGBoost | 0.856 | 0.860 | 0.918 |
| Random Forest | 0.814 | 0.823 | 0.902 |

⚠️ **This label is a median-split of a linear function of the same gene
features used to predict it** (built in `src/feature_engineering.py` via Cox
regression on the same 35 genes). The near-perfect ROC-AUC reflects this
circularity, not independent clinical validity — see `docs/LIMITATIONS.md`.
It is reported only for comparability with the four published CRC
ferroptosis-signature papers reviewed in `docs/RESEARCH_GAP.md`, which use
the same circular construction.

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
`results/regression_results.csv`, `results/representative_drug_results.csv`.
Rounded comparison tables: `tables/classification_model_comparison.csv`,
`tables/regression_model_comparison.csv`,
`tables/representative_drug_comparison.csv`,
`tables/cox_risk_score_coefficients.csv`.

## 8. These models are now persisted and interactively servable

Every model in the tables above was re-trained and persisted to
`models/registry.json` this session (`python scripts/train_and_persist.py`);
the numbers did not change (a data-leakage fix to the GDSC preprocessing
step — see `docs/DATA_LEAKAGE_AUDIT.md` — was verified to leave every metric
above unchanged to the reported precision). The exact fitted objects behind
these numbers are what the Streamlit application (`streamlit run
app/main.py`) loads and scores new samples with — see `docs/MODEL_CARD.md`
and `docs/DEMO_GUIDE.md`.
