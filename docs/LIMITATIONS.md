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
- **No raw dose-response curves.** The ~2 GB raw per-well GDSC2 viability file
  was not downloaded (see `DATASET.md`). The dose-response curve figures are
  mathematically reconstructed from the fitted `LN_IC50` parameter for
  illustration and are explicitly labeled as such; they are not measured
  viability data.
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
- **No external validation cohort.** Both classification tasks are evaluated
  on a held-out split of the same TCGA cohort used for training, not an
  independent GEO or other external CRC cohort. This is flagged explicitly as
  future work (see `FUTURE_WORK.md`), not silently omitted.
- **Feature scope limited to a documented gene panel.** All modeling uses the
  35-gene literature-derived ferroptosis signature (see
  `data/gene_selection_method.md`), not the whole transcriptome. This means
  no whole-transcriptome vs ferroptosis-only baseline comparison is available
  in this iteration (a genuine constraint of the network-fetch approach used
  here, not a hidden gap — see `FUTURE_WORK.md`).
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
