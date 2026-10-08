import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403
from theme import page_header  # noqa: E402

page_header("Documentation", "Methodology, scope, and reproducibility")

st.info(DISCLAIMER, icon=":material/info:")

tab_about, tab_method, tab_audit, tab_data, tab_refs, tab_limits, tab_repro = st.tabs(
    ["About", "Methodology", "ML audit", "Dataset", "References", "Limitations", "Reproducibility"]
)


def _md_file(relpath: str):
    path = ROOT / relpath
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.caption(f"{relpath} not found.")


with tab_about:
    st.markdown("""
A computational research workspace for ferroptosis-associated molecular profiling in
colorectal cancer: a literature-traceable ferroptosis gene signature is used to (1) classify
tumor samples by microsatellite-instability (MSI) status from ferroptosis-related gene
expression, and (2) predict an in-vitro drug-sensitivity summary statistic (LN(IC50) / AUC)
for colorectal-cancer cell lines, with interpretability throughout.

Every number, prediction, and figure in this workspace is computed from the pipeline's
actual output files -- nothing is hard-coded in the interface layer.
    """)
    st.markdown("**Pipeline**")
    st.code(
        "Literature (4 CRC papers + 1 mechanistic review)\n"
        "        -> unified 35-gene ferroptosis signature\n"
        "                |\n"
        "    -----------------------------\n"
        "    |                           |\n"
        "TCGA COADREAD (patients)    GDSC2 COAD/READ (cell lines)\n"
        "    -> preprocessing            -> preprocessing\n"
        "    -> feature engineering          |\n"
        "    -> MSI classification       -> pooled dose-response regression\n"
        "    -----------------------------\n"
        "                |\n"
        "    Evaluation + interpretability (permutation importance, SHAP)\n"
        "                |\n"
        "    Model registry (models/registry.json) -> this workspace",
        language=None,
    )
    st.caption("Project: Designing a Pipeline for Ferroptosis Classification and Dose-Response Prediction.")

with tab_method:
    _md_file("docs/METHODOLOGY.md")

with tab_audit:
    _md_file("ML_AUDIT.md")
    _md_file("docs/MODEL_SELECTION.md")
    with st.expander("Final model report (ensembles, extended features, 90% investigation)"):
        _md_file("results/final_model_report.md")

with tab_data:
    _md_file("docs/DATASET.md")
    with st.expander("CSV upload format (Analysis page)"):
        _md_file("docs/CSV_FORMAT.md")
    with st.expander("Data leakage audit"):
        _md_file("docs/DATA_LEAKAGE_AUDIT.md")

with tab_refs:
    _md_file("docs/REFERENCES.md")

with tab_limits:
    _md_file("docs/LIMITATIONS.md")

with tab_repro:
    st.markdown("**Regenerate every output from scratch**")
    st.code(
        "python main.py                          # full pipeline (data -> features -> models -> figures)\n"
        "python scripts/train_and_persist.py      # persist deployable models to models/registry.json\n"
        "python src/fetch_external_validation.py  # fetch the CPTAC-2 external cohort\n"
        "python src/external_validation.py         # score persisted models on the external cohort\n"
        "pytest tests/ -q                          # run the test suite",
        language="bash",
    )
    st.markdown("**Launch this workspace**")
    st.code("streamlit run app/main.py", language="bash")
    with st.expander("Model card"):
        _md_file("docs/MODEL_CARD.md")
