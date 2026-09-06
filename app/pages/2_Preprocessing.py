import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403

st.set_page_config(page_title="Preprocessing | Ferroptosis Pipeline", page_icon="🧬", layout="wide")
sidebar_status()
st.title("🧹 Preprocessing")
st.caption(
    "Every number on this page is read from the metadata JSON and text logs written by the last real "
    "run of src/preprocessing.py (`python main.py`) -- nothing here is a description of what the code "
    "does in the abstract; it is what actually happened on the actual data."
)

if not data_available():
    st.error("Processed data not found. Run `python main.py` from the project root first.")
    st.stop()

tab1, tab2 = st.tabs(["TCGA COADREAD", "GDSC2 cell lines"])

with tab1:
    meta = load_tcga_meta()
    st.subheader("Before → after")
    st.markdown(f"""
    - **MSI MANTIS threshold:** ≥ {meta['msi_mantis_threshold']} → `MSI-H` (Kautto et al., *Oncotarget* 2017)
    - **Ferroptosis genes retained:** {meta['n_ferroptosis_genes_used']} / 35
      (`{len(meta['ferroptosis_genes_missing_from_profile'])}` not returned by the RNA-seq V2 profile:
      {meta['ferroptosis_genes_missing_from_profile'] or 'none'})
    - **Samples after cleaning:** {meta['n_samples']}
    - **Transform applied:** log2(RSEM value, clipped ≥ 0, + 1) per gene
    - **Random seed:** {meta['random_seed']}
    """)
    st.write("**Final MSI label counts:**")
    st.json(meta["msi_label_counts"])

    log_path = DOCS_DIR / "review_evidence" / "tcga_preprocessing_log.txt"
    if log_path.exists():
        with st.expander("Full preprocessing run log (docs/review_evidence/tcga_preprocessing_log.txt)"):
            st.code(log_path.read_text(encoding="utf-8"), language=None)

with tab2:
    meta = load_gdsc_meta()
    st.subheader("Before → after")
    st.markdown(f"""
    - **Minimum drug coverage filter:** ≥ {meta['min_drug_coverage']} colorectal cell lines tested per drug kept
    - **Ferroptosis genes retained:** {len(meta['ferroptosis_genes_used'])} / 35
      (`{len(meta['ferroptosis_genes_missing_from_array'])}` with no probe on the 2013-era array:
      {meta['ferroptosis_genes_missing_from_array'] or 'none'})
    - **Final rows (drug × cell line):** {meta['n_rows']}
    - **Cell lines / drugs:** {meta['n_cell_lines']} / {meta['n_drugs']}
    - **Scaling:** NONE applied at this stage (data-leakage fix, see `docs/DATA_LEAKAGE_AUDIT.md`) --
      gene expression is stored as raw log2-scale RMA values; all scaling happens inside the
      train-only-fit sklearn Pipeline used for modeling.
    - **Random seed:** {meta['random_seed']}
    """)

    log_path = DOCS_DIR / "review_evidence" / "gdsc_preprocessing_log.txt"
    if log_path.exists():
        with st.expander("Full preprocessing run log (docs/review_evidence/gdsc_preprocessing_log.txt)"):
            st.code(log_path.read_text(encoding="utf-8"), language=None)

st.info(
    "See `docs/DATA_LEAKAGE_AUDIT.md` for the full leakage-prevention audit: why scaling/feature "
    "selection is always fit on the training fold only, and the one leakage vector found and fixed "
    "this session.",
    icon="🔒",
)
