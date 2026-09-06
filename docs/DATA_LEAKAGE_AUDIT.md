# Data Leakage Audit

This document records a systematic check of every point in the pipeline
where information could leak from a validation/test fold into a training
transformation, and states, for each pipeline, what was found.

## Method

For each dataset (TCGA classification, GDSC dose-response), every
transformation between raw data and a model's `.fit()` call was inspected
for whether its parameters (mean/std, PCA axes, feature selection, gene
list, hyperparameters) were computed using only the training fold, or using
any information from the validation/test fold or the target label itself.

## 1. TCGA classification pipeline

| Step | Fit on | Leakage risk? |
|---|---|---|
| Ferroptosis gene list selection (`src/ferroptosis.py`) | Literature only, before any data was even downloaded | None -- genes were chosen before any model existed |
| Missing-sample filtering, log2 transform (`src/preprocessing.py::build_tcga_dataset`) | Deterministic, sample-independent function (`log2(x+1)`); no fitted parameters | None -- log2 has no fitted parameters, so applying it before the split cannot leak |
| Train/val/test split (`split_tcga`) | `train_test_split(..., stratify=y, random_state=42)` | None -- standard stratified split, done once, before any transform with fitted parameters |
| `StandardScaler` (inside every model's `sklearn.Pipeline`) | `GridSearchCV.fit(X_trainval, y_trainval)` -- only the 80% train/val portion | None -- scaler is refit inside every CV fold via the Pipeline, never sees the test set |
| Hyperparameter search (`GridSearchCV`, `StratifiedKFold` on train/val only) | Train/val only | None -- test set is never touched until the single final evaluation call |
| Feature engineering (`ferroptosis_score_mean`, `ferroptosis_score_pc1`, Cox risk score) | **Fit on the FULL dataset** (`src/feature_engineering.py::add_engineered_features`, `build_ferroptosis_risk_score`) — see finding below | See below |

**Finding (pre-existing, documented, not fixed this session):** the PCA and
per-gene z-scoring used to build `ferroptosis_score_mean` /
`ferroptosis_score_pc1` (Experiment 2) and the Cox-regression risk score
(Experiment 3) are fit on the *entire* processed TCGA cohort, before any
train/test split for those specific experiments. This is a mild leakage
vector: the PCA axis and Cox coefficients are influenced by samples that
later appear in Experiment 2/3's test fold.

Severity assessment: **low, and does not change either headline claim**.
Experiment 2 already underperforms Experiment 1 (this is reported as a
negative result); leakage would, if anything, inflate Experiment 2's score,
making the true gap even larger, not smaller — so the negative-result
conclusion is unaffected (fixing the leak could only strengthen it).
Experiment 3 (`ferroptosis_risk_group`) is already flagged throughout this
project as **circular by construction** (the label is a function of the
same features used to predict it) and is explicitly excluded from every
scientific claim this project makes; this leakage vector is a second,
independent reason (in addition to circularity) not to treat Experiment 3's
near-perfect score as evidence of anything, and is documented here rather
than silently left unmentioned. **Not fixed this session** because doing so
correctly requires re-deriving the Cox risk score per-fold inside the
classification CV loop, which would meaningfully restructure
`feature_engineering.py`; tracked as future work
(`docs/FUTURE_WORK.md`) rather than rushed.

**Experiment 1 (the project's primary, headline result, ROC-AUC 0.917) uses
none of these engineered features** — only the 35 raw genes, scaled inside
each model's Pipeline on the training fold only — so it is unaffected by
this finding.

## 2. GDSC dose-response pipeline

| Step | Fit on | Leakage risk? |
|---|---|---|
| Ferroptosis gene list, drug coverage filter (≥40 cell lines/drug) | Fixed before any model existed / based on drug metadata only, not response values | None |
| Gene-expression scaling | **Previously:** z-scored across ALL colorectal cell lines before the split. **Fixed this session** — see below | Fixed |
| Train/test split (`split_gdsc`, `GroupShuffleSplit` by `COSMIC_ID`) | Whole-cell-line group split, so no cell line appears in both folds | None |
| `ColumnTransformer` (`StandardScaler` on genes+conc, `OneHotEncoder` on pathway) | `GridSearchCV.fit(X_train, y_train)` -- train fold only, refit inside every CV fold | None |
| Hyperparameter search (`GroupKFold` by `COSMIC_ID`, train fold only) | Train fold only | None |
| Representative single-drug models (LOOCV) | `StandardScaler` inside `Pipeline`, refit on each leave-one-out training fold | None |

**Finding (found and fixed this session):**
`src/preprocessing.py::_resolve_gdsc_gene_expression` used to z-score every
gene's expression across **all** colorectal cell lines (train + test
combined) before `split_gdsc()` ran the train/test split — exactly the
"scale before split" pattern called out as a leakage anti-pattern. Fixed by
removing that global scaling step entirely (`build_gdsc_dose_response_dataset`
now stores raw, unscaled log2-RMA values); all scaling now happens
exclusively inside the `ColumnTransformer`/`Pipeline`, fit on the training
fold only, for every experiment.

**Impact of the fix:** the pipeline was re-run end-to-end after the fix
(`python main.py`) and every reported metric in `RESULTS.md` was
regenerated from the corrected data. The numbers are **unchanged to the
reported precision** (e.g. pooled LN_IC50 Random Forest R² = 0.617 before
and after). This is expected and itself a useful finding: Random Forest and
XGBoost are invariant to a monotonic global rescaling of an individual
feature, and Linear Regression's poor performance was already dominated by
the one-hot drug-pathway encoding, not gene scaling — so removing this
leakage vector improves methodological correctness without being a
"we changed the headline number" event.

## 3. What was NOT tested against the held-out set

- No hyperparameter was ever chosen, and no feature was ever selected or
  dropped, based on test-set performance, in any experiment.
- The ferroptosis gene panel itself was fixed by literature review
  (`data/gene_selection_method.md`) before any TCGA/GDSC data was even
  downloaded, so it cannot reflect test-set overfitting.

## 4. Summary

| Leakage vector | Status |
|---|---|
| Scaling before train/test split (TCGA classification, Experiment 1) | Not present |
| Scaling before train/test split (GDSC dose-response) | **Found and fixed this session** |
| Feature selection using test-set performance | Not present anywhere |
| Hyperparameter tuning using the test set | Not present anywhere (CV always restricted to train fold) |
| Target leaking into feature engineering | Present in the circular, explicitly-flagged Experiment 3 only (already excluded from all claims); low-severity variant also affects Experiment 2's engineered-feature fit (does not change its already-negative conclusion) |
| Drug-response information leaking into its own input features | Not applicable — features are gene expression + drug pathway/concentration metadata, never the response value itself |
