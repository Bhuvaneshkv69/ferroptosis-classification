# Project Status

**Project:** Designing a Pipeline for Ferroptosis Classification and Dose-Response Prediction
**Team:** Mayank Bhardwaj (23BKT0106), Sovan Sekar Sarma (23BKT0057), Bhuvanesh K.V. (23BKT0138)
**Faculty Guide:** Dr. Adrija Bhattacharya
**Status date:** 2026-09-06

## Starting condition (Phase 1 finding)

No prior Review-1 PPT/PDF, source code, dataset, or documentation was present
in the working directory (`D:\small` was empty at the start of this build).
This project was therefore built **from scratch**: independent literature
research, dataset selection, gene-signature curation, full implementation,
real execution, and documentation, all performed in this session using only
the project title, team, and guide information supplied by the user. No prior
review claims exist to reconcile or contradict.

## What exists now (this session's output)

- A literature-traceable, 35-gene ferroptosis signature with full source
  provenance (`data/ferroptosis_genes_literature.csv`,
  `data/unified_gene_signature.csv`, `data/gene_selection_method.md`).
- Two real, working data-acquisition pipelines pulling from public sources:
  TCGA COADREAD via the cBioPortal REST API (`src/fetch_tcga.py`), and GDSC2
  colorectal cell-line data downloaded from the Sanger FTP mirror.
- Full preprocessing, feature engineering, classification, dose-response
  regression, and interpretability modules (`src/*.py`), all executed
  end-to-end and producing real (not fabricated) results.
- 29 passing automated tests (`tests/`).
- Documentation: `README.md`, `docs/DATASET.md`, `docs/METHODOLOGY.md`,
  `docs/RESEARCH_GAP.md`, `docs/LIMITATIONS.md`, `docs/FUTURE_WORK.md`,
  `RESULTS.md`, `IMPLEMENTATION_STATUS.md`, `docs/RUBRIC_MAPPING.md`,
  `docs/REVIEW_2_CONTENT.md`, `docs/VIVA.md`.

## What is explicitly NOT done (see `docs/FUTURE_WORK.md` for detail)

- No external (non-TCGA) validation cohort for the classifier.
- No whole-transcriptome baseline comparison (ferroptosis-panel-only features).
- No raw per-well GDSC dose-response curves (illustrative curves only, clearly labeled).
- No manuscript submitted, no patent filed, no product deployed — see
  "Progress toward expected outcome" in `docs/REVIEW_2_CONTENT.md`.

## Rubric component mapping (see `docs/RUBRIC_MAPPING.md` for full detail with evidence files)

| Rubric item | Status |
|---|---|
| Title, abstract, problem statement, objectives | Complete — `README.md`, `docs/REVIEW_2_CONTENT.md` |
| Updated literature survey + research gap | Complete — `docs/RESEARCH_GAP.md` |
| Proposed methodology + system architecture | Complete — `docs/METHODOLOGY.md` |
| Dataset, tools, software, hardware | Complete — `docs/DATASET.md`, `README.md` |
| ≥75% implementation with demonstration | Complete — see `IMPLEMENTATION_STATUS.md` for the honest percentage and per-module breakdown |
| Preliminary results, observations, performance analysis | Complete — `RESULTS.md`, `results/`, `figures/`, `tables/` |
| Comparison with existing methods | Complete — `docs/RESEARCH_GAP.md` comparison table |
| Progress toward IEEE paper/journal/product/patent | Honestly reported as not yet started beyond this pipeline — `docs/REVIEW_2_CONTENT.md` |
| Updated documentation, source code, presentation, evidence | Complete — this repository + `docs/review_evidence/` |
| Citations/references | Complete — `README.md` References section, `data/ferroptosis_genes_literature.csv` |
| Individual contribution | Template provided in `docs/REVIEW_2_CONTENT.md` (team to fill in actual task split) |
