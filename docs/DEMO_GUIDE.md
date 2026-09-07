# Demo Guide

A reproducible, real-data walkthrough of the full pipeline, from a clean
environment to a downloadable report. Every step uses real, already-included
data (`data/raw/`) — nothing here is synthetic.

## 0. Setup (once)

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows: .venv\Scripts\activate.bat
pip install -r requirements.txt
```

## 1. Run the pipeline (data → preprocessing → classification → dose-response → interpretability)

```bash
python main.py
```

Produces (all real, freshly computed): `data/processed/*`, `results/*.csv`,
`tables/*.csv`, `figures/*.png` (40 figures), `docs/review_evidence/*.txt`
(preprocessing logs). Takes ~5-7 minutes on a normal laptop CPU.

## 2. Persist the deployable models

```bash
python scripts/train_and_persist.py
```

Trains the same models `main.py` just evaluated and writes them to
`models/<task>/<experiment>/<Model>__<version>.joblib` + registers them in
`models/registry.json`. Takes ~4 minutes. **The application will not have
anything to serve until this step has run at least once.**

## 3. Run the tests

```bash
pytest tests/ -q
```

78 tests: dataset/preprocessing/feature-engineering/classification/
dose-response unit and integration tests (from the original pipeline),
model-registry/inference/input-validation unit tests, an end-to-end
Streamlit AppTest suite that runs every page and exercises two real
predictions and a real report generation, and tests for the external-
validation, whole-transcriptome-baseline, and real-dose-response-curve
results (steps 6-8 below; skipped automatically if those optional,
larger downloads haven't been run yet).

## 4. Launch the application

```bash
streamlit run app/main.py
```

Opens at `http://localhost:8501`. Suggested click-through for a live demo:

1. **Home** -- read the scope/limitations banner.
2. **Data** -- Overview tab (dataset sources, gene signature table); Upload &
   validate tab: download the CSV template, re-upload it unmodified to see a
   green "valid" result, then try deleting a column and re-uploading to see
   the validation catch it.
3. **Preprocessing** -- expand the real preprocessing run log for either
   dataset.
4. **Ferroptosis Classification** -- pick a real TCGA sample from the
   dropdown, click "Run prediction," compare the predicted label to the
   true label shown alongside it, inspect the per-gene contribution chart
   (exact for LogisticRegression).
5. **Drug Response** -- pick a drug + cell line pair, click "Run
   prediction," compare predicted vs. actual LN(IC50)/AUC and the
   RandomForest ensemble-spread indicator.
6. **Model Performance** -- browse ROC/PR curves, confusion matrices,
   predicted-vs-actual and residual plots for every model.
7. **Interpretability** -- SHAP summary plots + permutation importance
   tables, with the association-not-causation banner.
8. **Report** -- click "Generate report now," download the resulting
   self-contained HTML file, open it in a browser to confirm it stands alone.
9. **External Validation** -- compare internal (TCGA held-out) vs. external
   (independent CPTAC-2 cohort) performance for the same persisted models
   (requires step 6 below to have been run first).

## 5. Optional: the three larger, previously-deferred real analyses

These are real, substantial pieces of work (large downloads / longer
compute) kept out of the default `main.py` flow so a first run stays fast.
Each is independently runnable and documented in `RESULTS.md` §4b/4c/4d.

```bash
# 6. External validation on an independent cohort (~1 min, small download)
python src/fetch_external_validation.py
python src/external_validation.py

# 7. Whole-transcriptome baseline (~5 min fetch + ~5-10 min training)
python src/fetch_whole_transcriptome.py
python src/whole_transcriptome_baseline.py

# 8. Real (measured) dose-response curves (requires manually downloading
#    GDSC2_public_raw_data_24Jul22.csv, 2.1GB, to data/raw/gdsc/ first --
#    see docs/DATASET.md for the exact URL; then ~2-5 min to process)
python src/process_raw_dose_response.py
```

## 6. Re-running after a change

Any change to `src/preprocessing.py`, `src/classification.py`, or
`src/dose_response.py` requires re-running steps 1-3 (in order) before the
application reflects it -- the app never re-trains anything itself, by
design (see `docs/ARCHITECTURE.md`).

## What this demo does NOT show (by design, not oversight)

- Any clinical dosing recommendation (explicitly out of scope everywhere).
- A byte-for-byte reproduction of GDSC's own `gdscIC50` curve-fitting
  pipeline (the real curves in step 8 use a simplified, documented
  normalization — see `docs/LIMITATIONS.md`).
- Cross-cohort classification-threshold recalibration (step 6 shows that
  ROC-AUC ranking transfers well to the external cohort but the default
  0.5 threshold does not — reported honestly as an open item, not fixed
  here; see `docs/FUTURE_WORK.md` item 12).
