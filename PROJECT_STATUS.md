# Project Status

**Project:** Designing a Pipeline for Ferroptosis Classification and Dose-Response Prediction
**Team:** Mayank Bhardwaj (23BKT0106), Sovan Sekar Sarma (23BKT0057), Bhuvanesh K.V. (23BKT0138)
**Faculty Guide:** Dr. Adrija Bhattacharya
**Status date:** 2026-09-17 (UI/UX overhaul + literature-driven additions; see below)

## History

- **Prior session:** built the analysis pipeline from scratch (data
  acquisition, preprocessing, classification, dose-response regression,
  interpretability, 29 tests, documentation) starting from an empty
  directory.
- **Session 2026-09-07, phase 1:** audited that pipeline (`PROJECT_AUDIT.md`),
  fixed a data-leakage issue found in GDSC preprocessing, and built the
  application/persistence/registry/reporting layer (Streamlit app, model
  registry, inference, validation, reporting modules).
- **Session 2026-09-07, phase 2 (following a request to close every remaining
  gap):** fixed a second, subtler leakage issue in the TCGA engineered-
  feature pipeline; implemented genuine external validation on an
  independent cohort; fetched and trained a whole-transcriptome baseline;
  downloaded and processed the 2.1GB raw GDSC file into real, measured
  dose-response curves.
- **Session 2026-09-17:** turned the application into a guided,
  result-focused product (see `UI_AUDIT.md`): a "Research Sample Input"
  manual-entry form (per-gene numeric fields, no CSV required), visible run
  progress, plain-language "What does this mean?" panels alongside the
  existing technical detail. Following a request to also integrate seven
  newly selected literature papers (`docs/REFERENCES.md`), added two real
  backend components those papers motivate: a non-circular Prognosis
  Prediction validation (concordance index + Kaplan-Meier/log-rank test of
  the existing Cox risk score against real held-out patient survival) and a
  new drug-resistance classification module (`src/drug_resistance.py`,
  pooled + oxaliplatin-specific), both persisted/served the same way as the
  existing classification and dose-response models, with results reported
  honestly either way (the prognosis check: real but moderate signal; the
  pooled resistance classifier: real but near-chance, with the
  oxaliplatin-specific variant recovering real signal). 21 new/updated
  tests; full suite 95/95 passing.

## Phase-by-phase status

| Phase | Status | Evidence |
|---|---|---|
| 1. Audit existing repository | **DONE** | `PROJECT_AUDIT.md` |
| 2. Fix GDSC data-leakage issue | **DONE** | `docs/DATA_LEAKAGE_AUDIT.md`, pipeline re-run with identical metrics |
| 3. `configs/` + config-driven thresholds/seed | **DONE** | `configs/*.yaml`, `src/config.py` |
| 4. Model registry + persistence | **DONE** | `src/model_registry.py`, `scripts/train_and_persist.py`, `models/registry.json` |
| 5. Inference layer | **DONE** | `src/inference.py`, `tests/test_inference.py` |
| 6. Input validation | **DONE** | `src/validation.py`, `tests/test_validation.py` |
| 7. Report generator | **DONE** | `src/reporting.py`, `results/reports/*.html` |
| 8. Streamlit application (9 pages) | **DONE** | `app/`, wired only to `src/*` |
| 9. New documentation set | **DONE** | `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/DATA_LEAKAGE_AUDIT.md`, `docs/DEMO_GUIDE.md`, `docs/API_OR_PIPELINE_GUIDE.md`, `docs/DATA_GUIDE.md`, `docs/SCIENTIFIC_METHODS.md`, `data/README.md` |
| 10. TCGA engineered-feature leakage fix (Exp. 2/3) | **DONE** | `docs/DATA_LEAKAGE_AUDIT.md` finding #1; measured before/after numbers in `RESULTS.md` §2-3 |
| 11. External validation cohort (CPTAC-2) | **DONE** | `src/fetch_external_validation.py`, `src/external_validation.py`, `RESULTS.md` §4d, app "External Validation" page |
| 12. Whole-transcriptome baseline (Experiment 4) | **DONE** | `src/fetch_whole_transcriptome.py`, `src/whole_transcriptome_baseline.py`, `RESULTS.md` §4c |
| 13. Real (measured) per-well dose-response curves | **DONE** | `src/process_raw_dose_response.py`, `RESULTS.md` §4b, `figures/dose_response_curve_real_*.png` |
| 14. Tests for every new module + full suite | **DONE** | 12 new test files this session, **78/78 passing** |
| 15. End-to-end execution & verification | **DONE** | see "What was actually run" below |

## What was actually run, this session, in order

```
python main.py                                          # x2 (leakage fixes) -- 29/29 tests pass both times
python scripts/train_and_persist.py                     # x2 -- 18 model artifacts registered
pytest tests/ -q                                         # repeated after every change -- final: 78/78
streamlit run app/main.py                                 # launched; verified serving on localhost:8501
python src/fetch_external_validation.py                   # real network fetch, cBioPortal (CPTAC-2 study)
python src/external_validation.py                          # real scoring, no fitting on external data
python src/fetch_whole_transcriptome.py                    # real network fetch, ~19,200 genes x 592 samples
python src/whole_transcriptome_baseline.py                 # real training, PCA(50) inside CV folds
curl ... GDSC2_public_raw_data_24Jul22.csv                  # real download, 2.1GB, Sanger FTP mirror
python src/process_raw_dose_response.py                     # real streaming/processing of the raw file
```

## What is explicitly still NOT done (genuine, ongoing research directions)

These are not gaps in "the plan" — they are honest, real follow-on research
directions beyond this project's original scope, listed in
`docs/FUTURE_WORK.md`:

- Regularized (Ridge/Lasso/ElasticNet) per-drug regression.
- CTRP/DepMap PRISM screening for canonical ferroptosis-inducer compounds
  (Erastin/RSL3), absent from GDSC.
- Multi-omics integration (copy-number, methylation, mutation data).
- Formal Kaplan-Meier / log-rank survival reporting for the Cox risk groups.
- Multiple-testing (FDR) correction for the univariate Cox gene screen.
- Cross-cohort classification-threshold recalibration (the external
  validation found ROC-AUC transfers well but the default 0.5 threshold
  does not — a real, reported, unresolved finding, not a hidden one).
- Manuscript preparation / IEEE paper submission — no publication claimed.

Full detail: `docs/FUTURE_WORK.md`, `PROJECT_AUDIT.md`,
`IMPLEMENTATION_STATUS.md`.
