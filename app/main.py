"""
Streamlit application entry point (Page 1 -- Home).

Run with:
    streamlit run app/main.py

Every other page lives in app/pages/ and is auto-discovered by Streamlit.
This file and every page import only from src/* (via app/common.py) -- there
is no independent "UI copy" of any number; everything shown is read live
from data/processed/, results/, tables/, figures/, or models/registry.json.
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403

st.set_page_config(page_title="Ferroptosis Pipeline", page_icon="🧬", layout="wide")

st.title("🧬 Designing a Pipeline for Ferroptosis Classification and Dose-Response Prediction")
st.caption("Computational research pipeline · TCGA COADREAD + GDSC2 colorectal cell lines")

sidebar_status()

st.warning(DISCLAIMER, icon="⚠️")

st.markdown("""
## What this system does

This application is the interactive layer on top of a real, end-to-end computational pipeline:
a literature-traceable ferroptosis gene signature is used (1) to **classify** colorectal-cancer
patient tumors by microsatellite-instability (MSI) status from ferroptosis-related gene expression,
and (2) to **predict** an in-vitro drug-sensitivity summary statistic (LN(IC50) / AUC) for
colorectal-cancer cell lines, with model interpretability throughout.

Every number, prediction, and figure in this application is computed from the pipeline's actual
output files -- nothing is hard-coded in the UI layer.
""")

st.markdown("### Pipeline")
st.markdown("""
```
Literature (4 CRC papers + 1 mechanistic review)
        -> unified 35-gene ferroptosis signature (data/unified_gene_signature.csv)
                |
    -----------------------------
    |                           |
TCGA COADREAD (patients)    GDSC2 COAD/READ (cell lines)
    -> preprocessing            -> preprocessing
    -> feature engineering          |
    -> MSI classification       -> pooled dose-response regression (LN_IC50, AUC)
    -----------------------------
                |
    Evaluation + Interpretability (permutation importance, SHAP)
                |
    Model registry (models/registry.json) -> this application
```
""")

col1, col2, col3 = st.columns(3)
if data_available():
    tcga_meta = load_tcga_meta()
    gdsc_meta = load_gdsc_meta()
    col1.metric("TCGA COADREAD samples", tcga_meta["n_samples"])
    col2.metric("GDSC colorectal cell lines", gdsc_meta["n_cell_lines"])
    col3.metric("Ferroptosis genes in signature", len(load_gene_signature()))
else:
    st.error("Processed data not found. Run `python main.py` from the project root first.")

st.markdown("### Scientific scope and limitations (read before interpreting any result)")
st.markdown("""
- **Two datasets, never merged.** TCGA (primary human tumors, RNA-seq) and GDSC (immortalized
  cell lines, microarray) are different biological units and measurement technologies; classification
  runs entirely within TCGA, dose-response regression entirely within GDSC.
- **"Dose-response prediction" = an in-vitro summary statistic**, not a human clinical dose. See the
  warning banner above and `docs/LIMITATIONS.md`.
- **One classification experiment is circular by construction** (the "ferroptosis risk group" label
  is a thresholded function of the same genes used to predict it) and is deliberately excluded from
  the interactive prediction page; it is shown, clearly labeled, only on the Model Performance page
  for comparability with the published literature it mirrors.
- **Feature importance / SHAP show association, not causation.**
- **No external (cross-cohort) validation cohort** has been run yet -- see `docs/FUTURE_WORK.md` and
  `PROJECT_STATUS.md`.

Full detail: `docs/LIMITATIONS.md`, `docs/DATA_LEAKAGE_AUDIT.md`, `PROJECT_AUDIT.md`.
""")

st.markdown("### Pages")
st.markdown("""
1. **Home** (this page)
2. **Data** -- dataset statistics, gene signature, upload &amp; validate a new expression file
3. **Preprocessing** -- before/after data-quality summary from the actual preprocessing run
4. **Ferroptosis Classification** -- run the deployed MSI classifier on a real or uploaded sample
5. **Drug Response** -- run the deployed dose-response regressor on a real or uploaded sample
6. **Model Performance** -- every model's real metrics, ROC/PR curves, confusion matrices, residuals
7. **Interpretability** -- permutation importance and SHAP, with the association-not-causation caveat
8. **Report** -- generate and download a self-contained HTML report from the current pipeline outputs
9. **External Validation** -- the classifier scored on an independent cohort (CPTAC-2), no retraining
""")
