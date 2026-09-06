# Viva Preparation — Questions and Answers

## Project basics

**1. What is the one-sentence summary of this project?**
A reproducible ML pipeline that uses a literature-traceable ferroptosis gene
signature to classify colorectal-cancer patients (by MSI status) and predict
quantitative drug-response in colorectal cancer cell lines.

**2. Why colorectal cancer specifically?**
It is the project's stated application domain, ferroptosis has documented
prognostic relevance in CRC (four independent 2021 studies), and both TCGA
(patient) and GDSC (cell-line) public datasets have well-annotated colorectal
cohorts, making the two-dataset design feasible.

**3. What are the two independent tasks in this project, and why are they separate?**
(1) Classification of TCGA patient tumors by MSI status / ferroptosis risk
group; (2) regression of drug-response (LN_IC50, AUC) in GDSC colorectal cell
lines. They use different biological units (patient tumors vs. cultured cell
lines) on different measurement technologies (RNA-seq vs. microarray), so
merging them would conflate incompatible data — see `docs/DATASET.md`.

## Ferroptosis

**4. What is ferroptosis?**
An iron-dependent form of regulated cell death driven by uncontrolled lipid
peroxidation, distinct from apoptosis/necrosis, first formally named by
Dixon et al. (2012).

**5. Name three core ferroptosis regulator genes used in this project and their roles.**
`GPX4` (detoxifies lipid hydroperoxides using glutathione), `SLC7A11`
(imports cystine for glutathione synthesis), `ACSL4` (enriches membranes with
oxidizable polyunsaturated lipids, sensitizing cells to ferroptosis).

**6. Why is SLC7A11 highlighted in your results?**
It ranks among the top genes by permutation importance in *both* the MSI
classification and LN_IC50 regression tasks, despite being selected from
mechanistic (non-CRC-specific) literature — a cross-task recurrence worth
noting, though only as an association, not proof of causal role.

**7. What does FerrDb/FSP1/AIFM2 refer to?**
FerrDb is a curated ferroptosis-regulator database referenced by the CRC
literature we reviewed (used by Wang et al. 2021 as their starting gene
pool); AIFM2 (a.k.a. FSP1) is a GPX4-independent ferroptosis suppressor via
the CoQ10/NAD(P)H pathway, included in our core-mechanism gene tier.

**8. Did you use FerrDb directly?**
No — FerrDb V2's web interface could not be reliably scraped programmatically
in this session, so the gene signature was built instead directly from the
published, retrievable gene lists of four peer-reviewed CRC studies plus one
mechanistic review, each with full citation — arguably more traceable than
re-scraping a live database snapshot.

## Colorectal cancer / MSI

**9. What is MSI (microsatellite instability) and why does it matter clinically?**
MSI arises from defective DNA mismatch repair, causing instability in
repetitive DNA sequences. MSI-H colorectal tumors have distinct prognosis and
respond differently to immunotherapy versus MSS tumors — a clinically
consequential molecular subtype.

**10. How did you define MSI-H vs MSS in this project?**
Using the MSI MANTIS score with a threshold of 0.4 (score ≥ 0.4 = MSI-H),
per Kautto et al., *Oncotarget* 2017, which established this as the
accuracy-optimal cutoff for the MANTIS algorithm.

**11. Why is MSI status your "primary" classification target rather than the ferroptosis risk group?**
Because MSI status is measured independently of the ferroptosis gene
expression panel used as features — a genuinely non-circular prediction task,
unlike the risk-group label (see Q15).

## Datasets

**12. What is TCGA and what did you retrieve from it?**
The Cancer Genome Atlas — a large public cancer genomics program. We
retrieved 594 colorectal adenocarcinoma (COADREAD) patients' RNA-seq
expression (for our 35 ferroptosis genes) and clinical/MSI/survival
annotations, via the cBioPortal REST API.

**13. What is GDSC and what did you retrieve from it?**
Genomics of Drug Sensitivity in Cancer — a pharmacogenomic screen of cancer
cell lines against many compounds. We retrieved fitted dose-response
statistics (LN_IC50, AUC) and gene-expression data for 44-51 colorectal
(COAD/READ) cell lines, downloaded from the Sanger Institute FTP mirror.

**14. Why didn't you download the raw per-well GDSC viability data?**
It is ~2 GB (`GDSC2_public_raw_data_24Jul22.csv`) and was out of scope for
this iteration; the fitted `LN_IC50`/`AUC` summary statistics we used are the
standard ML targets in essentially all published GDSC-based papers anyway.
This is documented as a limitation, and the dose-response curve figures are
explicitly labeled as illustrative reconstructions from the fitted IC50, not
raw viability data.

**15. What does "circular" mean for your ferroptosis risk-group experiment, and why did you still include it?**
The risk-group label is a median split of a linear combination (Cox
regression coefficients) of the same 35 gene features later used to classify
it, so a model can partly "recover" the label by construction, inflating
apparent accuracy (we observed ROC-AUC up to 0.995). We include it, clearly
flagged, because it mirrors exactly how the four published CRC ferroptosis
papers validate their signatures — it lets us show the same circularity
exists in the literature we're comparing against, not just in our own work.

**16. Why 4 genes are missing from the GDSC expression data?**
`ATP6V1G2`, `NCOA4`, `SLC2A3`, `TXNIP` have no probe on the specific 2013-era
microarray platform (`sanger1018_brainarray`) used for GDSC cell-line
expression — verified by direct lookup against the raw file, not a mapping
bug. All GDSC-side models use the remaining 31/35 genes.

## Preprocessing

**17. How did you prevent data leakage in the TCGA classification task?**
Stratified train/val/test splitting per target; any scaling is fit inside a
scikit-learn `Pipeline` on the training fold only (via cross-validation), and
hyperparameter search uses only the training set.

**18. How did you prevent data leakage in the GDSC regression task?**
`GroupShuffleSplit`/`GroupKFold` grouped by `COSMIC_ID` (cell line identity),
so an entire cell line — not just individual (drug, cell-line) rows — is held
out at test time. A row-random split would let the model see a cell line's
expression profile during training (paired with a different drug) and
"cheat" when that same cell line reappears at test time with another drug.

**19. Why log2-transform the TCGA expression values?**
RNA-seq RSEM values are right-skewed, count-like data; log2(x+1) is the
standard transform to stabilize variance and make the distribution more
suitable for linear/distance-based models.

**20. Why z-score the GDSC expression values per gene?**
To standardize each gene's expression to comparable scale across the 44-49
colorectal cell lines before combining with drug/pathway/concentration
features and feeding into scale-sensitive models (Linear Regression, SVM).

**21. How did you handle duplicate samples/records?**
TCGA: kept only "Primary" tumor samples, and the alphabetically first sample
barcode for any patient with more than one. GDSC: dropped duplicate
(drug, cell-line) fitted-curve rows, keeping the first occurrence.

## Feature engineering

**22. What's the difference between "raw" and "engineered" features in your classification comparison?**
Raw = the 35 individual gene expression values. Engineered = two compact
derived summaries (mean per-gene z-score, and PC1 of the standardized gene
matrix).

**23. Did feature engineering help?**
No — for MSI classification, the engineered 2-feature summary performed
substantially worse (ROC-AUC ≈0.65) than the raw 35-gene panel (ROC-AUC
≈0.90–0.92). This is reported as a genuine negative finding: collapsing genes
into a single score discards gene-specific signal relevant to MSI status.

**24. How did you build the Cox-regression risk score?**
Univariate Cox regression (`lifelines.CoxPHFitter`) per gene against overall
survival, screening to genes with p<0.10, then a penalized multivariate Cox
model on the screened genes; the risk score is the weighted sum of z-scored
expression using the fitted coefficients, median-split into High/Low risk.

## Machine learning / classification

**25. Which models did you use for classification, and why these four?**
Logistic Regression (interpretable linear baseline), Random Forest (robust
non-linear ensemble), SVM (margin-based, handles both linear and RBF
kernels), XGBoost (gradient-boosted trees, often state-of-the-art on tabular
data) — a standard, complementary spread of model families as required.

**26. How did you tune hyperparameters?**
`GridSearchCV` with 5-fold stratified cross-validation on the training set
only, scoring by ROC-AUC; the best configuration per model is then evaluated
once on the untouched held-out test set.

**27. Why did XGBoost underperform Logistic Regression for MSI classification?**
XGBoost's F1 was only 0.174 on this task — likely overfitting the minority
MSI-H class despite the small tuning grid; with only ~89 MSI-H patients
total, tree ensembles can be less stable than a well-regularized linear
model. This is reported honestly rather than cherry-picking a better run.

**28. How did you handle class imbalance?**
`class_weight="balanced"` for Logistic Regression, Random Forest, and SVM,
which reweights the loss inversely proportional to class frequency.

## Regression / dose-response

**29. What exactly are you predicting in the dose-response module?**
`LN_IC50` (natural log of the half-maximal inhibitory concentration, µM) and
`AUC` (area under the fitted dose-response curve, 0-1) — both experimentally
fitted, GDSC-released summary statistics for a cell line's response to a
drug, not a raw measurement and not a clinical dosage.

**30. Why do you say this is NOT clinical dosage prediction?**
IC50/AUC describe an in-vitro cultured cell line's sensitivity to a compound
under standardized lab conditions; they are not validated against, nor
directly translatable to, a human patient's therapeutic dose, which depends
on pharmacokinetics, body physiology, and clinical trial-derived dosing that
this data cannot capture.

**31. What features did the pooled dose-response model use?**
31 ferroptosis gene expression z-scores, drug pathway (one-hot), and the
screening concentration range (MIN_CONC, MAX_CONC) — combined via a
`ColumnTransformer`.

**32. Why did Linear Regression perform so poorly (negative R²) on the per-drug models?**
With only ~44 cell lines and 31 gene features, the sample size barely
exceeds the feature count; an unregularized linear model overfits badly
under leave-one-out CV. Random Forest, being less sensitive to this ratio and
capable of implicit feature selection, achieved modest positive R² (0.21-0.32)
instead. This motivates trying Ridge/Lasso regularization as future work.

**33. Why leave-one-out CV for the representative single-drug models?**
Each drug has only 42-44 colorectal cell lines tested — too small to carve
out a separate held-out test split without making the test set trivially
tiny and high-variance; LOOCV uses every sample for both training and
(one-at-a-time) testing, maximizing data use for such a small n.

**34. Why did you pick 5-Fluorouracil, Oxaliplatin, Irinotecan, and Sorafenib as representative drugs?**
The first three are the backbone chemotherapy agents for colorectal cancer
(FOLFOX/FOLFIRI regimens), directly relevant to the project's clinical
domain; Sorafenib is a documented pharmacological ferroptosis inducer (via
system xc⁻ inhibition), tying the drug-response task back to ferroptosis
biology. Canonical ferroptosis inducers (Erastin, RSL3) are not in the GDSC
screening panel — a documented limitation.

## Metrics

**35. Explain ROC-AUC vs PR-AUC and why you report both.**
ROC-AUC measures ranking quality across all classification thresholds using
true/false positive rates; PR-AUC focuses on precision-recall trade-off and
is more informative under class imbalance (our MSI-H class is a minority),
so reporting both gives a fuller picture than either alone.

**36. Why report both Pearson and Spearman correlation for regression?**
Pearson captures linear correlation; Spearman captures monotonic
(rank-based) correlation and is more robust to outliers/non-linearity —
useful since IC50 relationships need not be strictly linear.

## Research gap / novelty

**37. What exactly is new about your approach compared to the literature?**
Not the ML algorithms or the biology, but the *combination*: a single
literature-traceable ferroptosis gene panel applied to both a non-circular
classification task and a genuine dose-response regression task, evaluated
and interpreted end-to-end in one reproducible pipeline — something none of
the four reviewed CRC ferroptosis papers attempt (they stop at circular
survival-risk stratification).

**38. Do you claim this is the first ever integration of ferroptosis + ML + drug response?**
No absolute novelty is claimed anywhere in this project; see
`docs/RESEARCH_GAP.md`, which explicitly states this is a combination of
existing threads, not new algorithms or new biology.

## Architecture / engineering

**39. Why is the codebase split into separate modules instead of one big script/notebook?**
Reusability and testability — each module (`data_loader.py`,
`preprocessing.py`, `feature_engineering.py`, `classification.py`,
`dose_response.py`, `evaluation.py`, `interpretability.py`) can be tested,
re-run, and iterated on independently, and `main.py` composes them into one
reproducible end-to-end run.

**40. How would someone reproduce your exact results?**
`pip install -r requirements.txt`, place the three GDSC files per
`docs/DATASET.md`, then run `python main.py` (add `--fetch` to also
re-download TCGA data). `RANDOM_SEED = 42` is fixed everywhere, and every
preprocessing decision is logged to `docs/review_evidence/`.

## Limitations / overfitting / interpretability

**41. What is the single biggest limitation of this project?**
No external validation cohort — both classification tasks are evaluated on a
held-out split of the *same* TCGA dataset they were trained on, not an
independent cohort (e.g. a GEO series), so generalization beyond TCGA is
untested. Documented in `docs/LIMITATIONS.md` and `docs/FUTURE_WORK.md`.

**42. How do you know your models aren't just overfitting?**
Classification uses a held-out test split never seen during hyperparameter
tuning; dose-response regression holds out entire cell lines (not just rows);
representative single-drug models use leave-one-out CV. The honestly-reported
failures (XGBoost's low F1, Linear Regression's negative R² under n≈p) are
themselves evidence we are not selectively hiding poor results.

**43. What does SHAP tell you, and what does it NOT tell you?**
SHAP quantifies each feature's contribution to a specific model's prediction
on specific samples — a measure of statistical association within this
dataset and model. It does NOT establish that a gene biologically causes the
outcome; that requires independent wet-lab/functional validation, explicitly
out of scope here (`docs/LIMITATIONS.md`).

**44. If a reviewer asks "is this ready for clinical use?" — what do you say?**
No. This is a retrospective computational research project using public
datasets, without external validation, wet-lab confirmation, or prospective
clinical testing. It demonstrates a reproducible analytical pipeline and
genuine, honestly-reported statistical associations — not a validated
clinical tool. See the scope statement in `README.md` and `docs/LIMITATIONS.md`.
