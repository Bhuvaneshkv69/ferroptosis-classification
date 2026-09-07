# Project Audit

**Date:** 2026-09-06
**Auditor:** Claude Sonnet 5, in-session code audit (no prior audit existed)
**Method:** every file in the repository was read, the environment was
rebuilt, the full test suite was executed, and the full pipeline (`main.py`)
was re-run before any code was changed.

## 1. Starting condition

This was **not** an empty or mock project. Before this session's changes it
already contained:

- A real, working, script-based pipeline (`main.py` → `src/*.py`, 1,734 lines)
  that fetches TCGA COADREAD data from the live cBioPortal API and reads a
  manually-downloaded GDSC2 snapshot, preprocesses both, engineers features,
  trains 4 classifiers × 3 experiments and 3 regressors × 2 targets + 4
  per-drug models, computes real metrics, and writes real figures/tables.
- 29 passing pytest tests (`pytest tests/ -q` → `29 passed`), all of which
  read from files the pipeline actually produced (no test asserts on
  hard-coded/fabricated numbers).
- Honest documentation (`RESULTS.md`, `IMPLEMENTATION_STATUS.md`,
  `docs/LIMITATIONS.md`, `docs/DATASET.md`, `docs/METHODOLOGY.md`,
  `docs/RESEARCH_GAP.md`, `docs/FUTURE_WORK.md`) that explicitly separates
  what was implemented from what was not, and flags a circular label
  (`ferroptosis_risk_group`) as circular rather than hiding it.
- A literature-traceable 35-gene ferroptosis signature with per-gene source
  provenance (`data/ferroptosis_genes_literature.csv` →
  `data/unified_gene_signature.csv`, built by `src/ferroptosis.py`).

Re-running `pytest tests/ -q` at the start of this session reproduced
**29 passed** with no changes. Nothing was broken, mocked, or fabricated in
the pre-existing code.

## 2. What this audit found missing (the actual gap vs. the request)

The prior work is a **correct, tested, real analysis pipeline**. It is not,
and was never built as, a **software system with a persistence layer or a
user-facing application**. Specifically, missing before this session:

| Gap | Detail |
|---|---|
| No application | No Streamlit/web UI of any kind — only CLI scripts and notebooks. |
| No model persistence | Every run re-trains every model from scratch; nothing is saved to disk (no `joblib`/pickle anywhere in `src/`). An app cannot serve predictions without this. |
| No model registry / versioning | No metadata store of model version, hyperparameters, training timestamp, or metrics per artifact. |
| No configs directory | Hyperparameter grids, thresholds (`MSI_MANTIS_THRESHOLD`, `MIN_DRUG_COVERAGE`), and paths are hard-coded as Python constants scattered across modules rather than centralized in versioned config files. |
| No inference-only path | Prediction always requires re-running the full training pipeline; there is no "load model, score one new sample" function. |
| No report generator | `RESULTS.md` is a hand-written summary of one run, not a reproducibly regenerated artifact. |
| One data-leakage issue found | See §3 below — global (pre-split) feature scaling in the GDSC dose-response pipeline. |
| No `data/README.md`, `ARCHITECTURE.md`, `MODEL_CARD.md`, `DATA_LEAKAGE_AUDIT.md`, `DEMO_GUIDE.md` | Requested documentation set was partially present under different names/scope. |
| No external (cross-cohort) validation | Explicitly and honestly flagged as not done in `docs/FUTURE_WORK.md` already. |
| No raw per-well GDSC dose-response curves | Explicitly and honestly flagged already — the ~2GB raw file was never downloaded; existing curves are labeled "illustrative." |

## 3. Data-leakage audit finding (fixed this session)

`src/preprocessing.py::_resolve_gdsc_gene_expression` z-scored every
ferroptosis gene's expression **across all colorectal cell lines** (train
and test) *before* `split_gdsc()` performed the train/test `GroupShuffleSplit`.
This is exactly the leakage pattern the project brief calls out by name
("BAD: Scaling the entire dataset before train/test split.") — the global
mean/std baked into `gdsc_colorectal_dose_response_processed.csv` was
computed using cell lines that later end up in the held-out test fold.

The severity is mild (it leaks only marginal distribution statistics, not
the target `LN_IC50`/`AUC`, and the downstream model pipeline still fits a
*second*, train-only `StandardScaler` inside a `sklearn.Pipeline`/
`ColumnTransformer`), but it is a genuine, avoidable leakage vector and is
fixed in this session: `build_gdsc_dose_response_dataset` now stores **raw
(unscaled) log2-scale RMA expression** in the processed CSV, and all scaling
happens exclusively inside the train-only-fit `Pipeline`/`ColumnTransformer`
used by `dose_response.py` and `interpretability.py`. Full writeup:
`docs/DATA_LEAKAGE_AUDIT.md`. The pipeline was re-run after this fix and all
regression numbers in `RESULTS.md`/`README.md` were regenerated from the
corrected data (not hand-edited).

No other leakage was found: TCGA train/val/test split is stratified and
disjoint (tested), GDSC train/test split is by `COSMIC_ID` group so no cell
line appears in both folds (tested), all hyperparameter search uses CV folds
built only from the training portion, and feature/gene selection was fixed
by literature review before any model was ever fit (not tuned against test
performance).

## 4. Scientific-validity notes (already mostly handled correctly)

- Patient-level (TCGA) and cell-line-level (GDSC) data are correctly never
  pooled into one model (`docs/DATASET.md` §"Why these two datasets are
  never merged").
- The circular `ferroptosis_risk_group` classification experiment is clearly
  labeled circular and kept separate from the primary, non-circular MSI
  classification task.
- "Dose-response prediction" is precisely scoped to mean predicting a
  GDSC-fitted in-vitro summary statistic (`LN_IC50`/`AUC`), never a human
  clinical dose — stated in `README.md`, `docs/LIMITATIONS.md`, and (new)
  restated on every relevant page of the application built this session.
- Interpretability outputs are consistently phrased as "associated with,"
  never "causes."

## 5. Proposed final architecture (implemented this session)

```
configs/                   data_config.yaml, model_config.yaml (centralizes
                            thresholds, hyperparameter grids, paths)
data/
  README.md                 (new) role of each data/ subdirectory
  raw/, processed/, external/, metadata/ (metadata/ new)
models/                     (new) persisted joblib Pipelines, one file per
                            (task, experiment, model), never overwritten
  registry.json              (new) model registry: version, metrics, hyper-
                              params, dataset, timestamp per persisted model
src/
  model_registry.py          (new) register/list/load persisted models
  inference.py                (new) load-model + score-one-sample functions
                              used by both the app and the CLI
  reporting.py                 (new) builds an HTML computational report from
                              real result files
  (existing modules unchanged in behavior, only the GDSC z-scoring leakage
   fix and a light config-driven read of thresholds/paths)
app/
  main.py, pages/             (new) 8-page Streamlit application, calling
                              only src/* functions and reading only files the
                              pipeline actually produced — no hard-coded
                              numbers in the UI layer
scripts/
  train_and_persist.py        (new) trains + persists the deployable models
                              used by the app (same fit calls already used
                              for evaluation — the persisted model IS the
                              evaluated model, not a separately retrained one)
docs/ (existing + new: ARCHITECTURE.md, DATA_LEAKAGE_AUDIT.md, MODEL_CARD.md,
       DEMO_GUIDE.md, DATA_GUIDE.md, SCIENTIFIC_METHODS.md,
       API_OR_PIPELINE_GUIDE.md)
```

## 6. Implementation plan (phases, tracked live in `PROJECT_STATUS.md`)

1. Fix the GDSC leakage issue, regenerate processed data + all downstream
   results/figures/tables, update numbers in docs to match.
2. `configs/` + `src/model_registry.py` + persist the exact fitted models
   already evaluated by `classification.py`/`dose_response.py`.
3. `src/inference.py` — reusable scoring functions for new samples.
4. `src/reporting.py` — real, dynamically generated HTML report.
5. Streamlit application (8 pages), wired only to real backend calls.
6. New documentation set.
7. Tests for the new modules; full-suite re-run.
8. End-to-end execution: fresh install → pipeline → tests → app launch →
   click through every page → generate a report → re-run tests.
9. Honest completion percentage in the final report (not based on lines of
   code).

**Update (phase 2 of this session):** external validation, a whole-
transcriptome baseline, and real raw dose-response curves — all three
originally scoped above as future work — were subsequently implemented in
full following a request to close every remaining gap. See
`docs/FUTURE_WORK.md`, `RESULTS.md` §4b/4c/4d, and
`IMPLEMENTATION_STATUS.md` for what was actually done and the honest
results (including two findings that cut against a simpler, more flattering
narrative: the whole-transcriptome baseline outperforming the ferroptosis
panel, and external-cohort classification thresholds not transferring as
cleanly as ROC-AUC ranking does).

Molecular fingerprint/SMILES-based drug descriptors remain out of scope —
GDSC's own pathway/target/concentration-range annotations are used as the
drug representation instead, which is scientifically defensible and already
what the pooled dose-response model uses.
