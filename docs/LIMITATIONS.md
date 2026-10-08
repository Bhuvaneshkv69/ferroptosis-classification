# Limitations

This is a computational research project using retrospective public data. It
is **not a clinical decision-making system** and makes no claim to be ready
for patient treatment or clinical dosing decisions.

## Dataset limitations

- **Retrospective, public cohorts.** TCGA COADREAD (594 patients) and GDSC2
  colorectal cell lines (44–51 lines) are retrospective, already-curated
  public datasets, not prospectively collected for this project. Findings
  reflect associations within these specific cohorts.
- **Sample size.** 555 TCGA patients with an MSI label, and 42–46 cell lines
  per representative drug, are modest by modern ML standards, especially for
  the per-drug regression models (n ≈ p, the number of ferroptosis-gene
  features), which increases variance and overfitting risk (visible directly
  in the unregularized Linear Regression results — strongly negative R² on
  several drugs — see `RESULTS.md`).
- **Dataset bias.** TCGA is enriched for surgically resectable tumors from
  a specific set of contributing institutions; GDSC cell lines are long-term
  cultured lines that have undergone genomic drift from the original tumor
  and lack a tumor microenvironment, immune cells, or stroma.
- **Cell-line vs patient biology.** GDSC drug-sensitivity results describe
  in-vitro cell-line pharmacology; they are not validated to predict a real
  patient's clinical response to the same drug.
- **GDSC expression-array gene coverage.** The 2013-era microarray used for
  GDSC cell-line expression lacks probes for 4/35 unified signature genes
  (`ATP6V1G2`, `NCOA4`, `SLC2A3`, `TXNIP`); all GDSC-side results use the
  remaining 31 genes only.
- **Real dose-response curves (added this session) use a simplified
  normalization.** `src/process_raw_dose_response.py` normalizes each well's
  raw intensity to its own plate's mean `NC-0` control intensity only. GDSC's
  own `gdscIC50` R package additionally applies positional/edge-effect and
  dye-based corrections; this project's curves are real, measured data but
  not a byte-for-byte reproduction of GDSC's published fitting pipeline —
  see `RESULTS.md` §4b. The original illustrative (IC50-only-reconstructed)
  curves are kept alongside for comparison, clearly labeled as illustrative.
- **No canonical ferroptosis-inducer drugs in GDSC.** Erastin and RSL3 (the
  textbook pharmacological ferroptosis inducers) are not part of the GDSC
  screening panel. Sorafenib (a documented ferroptosis inducer via system
  xc⁻ inhibition) is used as the closest available proxy.

## Methodological limitations

- **Circularity of the ferroptosis risk-group experiment.** The Cox-derived
  risk-group label (`ferroptosis_risk_group`) is a thresholded linear
  function of the same gene features used to classify it, so its very high
  reported accuracy (ROC-AUC up to 0.995) reflects the model recovering an
  internally consistent label, not independent clinical validation. This
  mirrors a known property of the published literature it is modeled on
  (see `docs/RESEARCH_GAP.md`) and is reported for comparability only — it is
  **not** the project's evidentiary claim.
- **External validation (added this session) uses a 28/35-gene intersection
  variant, and its classification threshold does not transfer as cleanly as
  its ranking does.** The MSI classifier was validated on CPTAC-2 Prospective
  Colon Cancer, an independent cohort. CPTAC's expression profile lacks
  values for 7 of the 35 panel genes, so a compatible model trained on the
  28-gene intersection is used for this comparison (not the deployed 35-gene
  model). ROC-AUC (ranking quality) transferred well (earlier September pipeline: 0.94–0.96 externally
  vs. 0.84–0.90 internally for this variant); default-threshold accuracy did
  not transfer uniformly (SVM's external accuracy was 0.51 despite ROC-AUC
  0.94) — a real, honestly-reported cross-cohort calibration gap, not a
  hidden failure. See `RESULTS.md` §4d, `docs/FUTURE_WORK.md` item 12.
- **Feature scope limited to a documented gene panel (with a whole-
  transcriptome baseline comparison added this session).** All classification/
  regression modeling in this project uses the 35-gene literature-derived
  ferroptosis signature (see `data/gene_selection_method.md`), not the whole
  transcriptome. Experiment 4 (`RESULTS.md` §4c) shows a whole-transcriptome
  (PCA-reduced) classifier meaningfully **outperforms** the ferroptosis-panel
  classifier (ROC-AUC 0.986 vs. 0.917) — i.e. the panel captures real but
  incomplete MSI signal. This baseline exists only for the MSI classification
  task, not for the dose-response regression task.
- **Association, not causation.** All feature-importance and SHAP results
  (`src/interpretability.py`) describe statistical association between a
  gene's expression and the model's output on this dataset. They are **not**
  evidence that the gene causally drives MSI status, survival, or drug
  sensitivity; such claims require independent wet-lab / functional
  validation, which is outside this project's scope.
- **Multiple-testing / small-panel univariate Cox screen.** The univariate
  Cox screening step (p < 0.10, no multiple-testing correction) mirrors the
  published literature's own approach but is statistically permissive; it is
  used only to build the exploratory, explicitly-caveated risk score.
- **Prognosis validation is moderate, not strong (added this session).** The
  non-circular check of the Cox risk score against held-out patient survival
  (`RESULTS.md` §9) gives a concordance index of 0.692 and a log-rank
  p-value of 0.039 on 118 held-out patients — real, statistically detectable
  signal, but well short of the primary MSI classification result's strength
  (ROC-AUC 0.917) and based on a modest held-out sample size. It should be
  read as "this risk score carries real but moderate prognostic signal,"
  not as a validated clinical prognostic tool.
- **The pooled drug-resistance classifier performs close to chance (added
  this session).** `src/drug_resistance.py`'s cross-drug pooled model
  (ROC-AUC 0.45–0.54, `RESULTS.md` §10) does not reliably distinguish
  Resistant from Sensitive when many different drug mechanisms are pooled
  together; only the oxaliplatin-specific, evaluation-only variant recovers
  real signal (ROC-AUC ~0.68). The pooled model is still deployed/servable
  for completeness, and the application surfaces this limitation directly
  rather than hiding it (Results → Drug resistance). Its "Resistant"/
  "Sensitive" labels are also this project's own median-split definition,
  not a value reproduced from the literature that motivated the analysis
  (`docs/REFERENCES.md` papers 6-7) -- see that document for why.

## Scope statement (dose-response)

Everywhere in this project, "dose-response prediction" means predicting an
**experimentally measured, in-vitro drug-sensitivity summary statistic**
(`LN_IC50` or `AUC`, as fitted and released by the GDSC consortium) for a
cancer cell line from its ferroptosis-related gene expression. It does **not**
mean, and this project never claims, prediction of a human patient's clinical
drug dosage or treatment recommendation.

## Addendum — model-selection redesign

- Only 22 MSI-H patients in the locked test set; all positive-class metrics have wide intervals (see `docs/MODEL_SELECTION.md`).
- At the MCC-tuned threshold sensitivity is low (7/22 on the locked test). Do not use the classifier to rule MSI-High out.
- The 35-gene panel caps performance; the 90% accuracy target was not achieved (majority baseline 84%).
- External validation uses a refit on the 28 genes shared with CPTAC-2, not the production 35-gene model.
- Drug response: 44 cell lines; model ≈ per-compound mean on held-out lines; tumour RNA-seq to cell-line microarray mapping is exploratory and unvalidated.
- `SVC(probability=True)` is deprecated in scikit-learn 1.9 (removed in 1.11); the SVM candidates need migrating then.
- The `drug_resistance` classifier (AUC ≈ chance) and the circular risk-group models are no longer exposed in the app.
