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

61 tests: dataset/preprocessing/feature-engineering/classification/
dose-response unit and integration tests (from the original pipeline),
plus model-registry, inference, input-validation unit tests, and an
end-to-end Streamlit AppTest suite that runs every page and exercises two
real predictions and a real report generation.

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

## 5. Re-running after a change

Any change to `src/preprocessing.py`, `src/classification.py`, or
`src/dose_response.py` requires re-running steps 1-3 (in order) before the
application reflects it -- the app never re-trains anything itself, by
design (see `docs/ARCHITECTURE.md`).

## What this demo does NOT show (by design, not oversight)

- A truly independent external validation cohort (not yet implemented --
  `docs/FUTURE_WORK.md`).
- Raw per-well GDSC viability curves (the ~2GB raw file was never
  downloaded; the dose-response curve figures are labeled "illustrative").
- Any clinical dosing recommendation (explicitly out of scope everywhere).
