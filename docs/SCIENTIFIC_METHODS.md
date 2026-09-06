# Scientific Methods

This is a short index; full methodology, results, and caveats live in the
documents below so they are maintained in exactly one place each.

- **Full stage-by-stage methodology and system architecture:**
  `docs/METHODOLOGY.md`, `docs/ARCHITECTURE.md`
- **Literature survey and research gap this project addresses:**
  `docs/RESEARCH_GAP.md`
- **Statistical/ML methods used** (stratified/group CV, `GridSearchCV`
  hyperparameter search, permutation importance, SHAP, Cox regression):
  `docs/METHODOLOGY.md` §7, `src/classification.py`, `src/dose_response.py`,
  `src/interpretability.py`, `src/feature_engineering.py` module docstrings.
- **Data-leakage prevention, audited systematically:**
  `docs/DATA_LEAKAGE_AUDIT.md`
- **What "dose-response prediction" scientifically means in this project
  (and does not mean):** `docs/LIMITATIONS.md` §"Scope statement," restated
  on every relevant page of the application.
- **Model performance, with discussion:** `RESULTS.md`, `docs/MODEL_CARD.md`
- **Full, honest list of limitations:** `docs/LIMITATIONS.md`
- **What remains unimplemented and why:** `docs/FUTURE_WORK.md`,
  `PROJECT_AUDIT.md`, `PROJECT_STATUS.md`
