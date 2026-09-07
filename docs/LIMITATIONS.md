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
  model). ROC-AUC (ranking quality) transferred well (0.94–0.96 externally
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

## Scope statement (dose-response)

Everywhere in this project, "dose-response prediction" means predicting an
**experimentally measured, in-vitro drug-sensitivity summary statistic**
(`LN_IC50` or `AUC`, as fitted and released by the GDSC consortium) for a
cancer cell line from its ferroptosis-related gene expression. It does **not**
mean, and this project never claims, prediction of a human patient's clinical
drug dosage or treatment recommendation.
