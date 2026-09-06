import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403

st.set_page_config(page_title="Data | Ferroptosis Pipeline", page_icon="🧬", layout="wide")
sidebar_status()
st.title("📊 Data")

if not data_available():
    st.error("Processed data not found. Run `python main.py` from the project root first.")
    st.stop()

tcga_meta = load_tcga_meta()
gdsc_meta = load_gdsc_meta()
tcga = load_tcga_features()
gdsc = load_gdsc_processed()
genes = load_gene_signature()

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "TCGA COADREAD", "GDSC2 cell lines", "Upload & validate"])

with tab1:
    st.subheader("Dataset sources")
    st.table(pd.DataFrame([
        {"Dataset": "TCGA COADREAD (PanCancer Atlas)", "Level": "Patient", "Access": "cBioPortal REST API",
         "Samples": tcga_meta["n_samples"], "Role": "Ferroptosis-gene MSI classification"},
        {"Dataset": "GDSC2 colorectal cell lines", "Level": "Cell line", "Access": "Sanger FTP mirror",
         "Samples": f'{gdsc_meta["n_cell_lines"]} lines / {gdsc_meta["n_rows"]} (drug,line) rows',
         "Role": "Dose-response (LN_IC50, AUC) regression"},
    ]))
    st.caption("Full dataset documentation, citations and license notes: docs/DATASET.md, data/README.md")

    st.subheader("Ferroptosis gene signature")
    st.write(f"{len(genes)} genes, literature-traceable to 4 CRC prognostic-signature papers + 1 mechanistic review "
             f"(see `data/gene_selection_method.md`).")
    st.dataframe(genes, width="stretch", hide_index=True)

with tab2:
    st.subheader("TCGA COADREAD sample/gene statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", tcga_meta["n_samples"])
    c2.metric("Ferroptosis genes used", tcga_meta["n_ferroptosis_genes_used"])
    c3.metric("Genes missing from RNA-seq profile", len(tcga_meta["ferroptosis_genes_missing_from_profile"]))
    if tcga_meta["ferroptosis_genes_missing_from_profile"]:
        st.caption(f"Missing: {tcga_meta['ferroptosis_genes_missing_from_profile']}")

    st.write("**MSI label distribution** (MANTIS score ≥ %.2f = MSI-H):" % tcga_meta["msi_mantis_threshold"])
    st.bar_chart(pd.Series(tcga_meta["msi_label_counts"]))

    st.write("**Ferroptosis risk-group distribution** (exploratory, circular label -- see Home page caveats):")
    if "ferroptosis_risk_group" in tcga.columns:
        st.bar_chart(tcga["ferroptosis_risk_group"].value_counts())

    with st.expander("Raw processed table (first 50 rows)"):
        st.dataframe(tcga.head(50), width="stretch")

with tab3:
    st.subheader("GDSC2 colorectal cell-line / drug statistics")
    c1, c2, c3 = st.columns(3)
    c1.metric("Cell lines", gdsc_meta["n_cell_lines"])
    c2.metric("Drugs (well-covered)", gdsc_meta["n_drugs"])
    c3.metric("Genes missing from array", len(gdsc_meta["ferroptosis_genes_missing_from_array"]))
    if gdsc_meta["ferroptosis_genes_missing_from_array"]:
        st.caption(f"Missing (no probe on the 2013-era array): {gdsc_meta['ferroptosis_genes_missing_from_array']}")

    st.write("**Rows per drug pathway:**")
    st.bar_chart(gdsc["PATHWAY_NAME"].value_counts().head(15))

    with st.expander("Raw processed table (first 50 rows)"):
        st.dataframe(gdsc.head(50), width="stretch")

with tab4:
    st.subheader("Upload and validate a new gene-expression file")
    st.caption(
        "Checks the file against the ferroptosis gene panel for duplicate columns, missing genes, "
        "non-numeric/malformed values, and zero-variance columns -- before any model ever sees it."
    )
    required_genes = val.load_required_gene_panel()
    template = val.make_template_csv(required_genes)
    st.download_button(
        "Download CSV template (35 gene columns, one example row)",
        template.to_csv(index=False).encode("utf-8"),
        file_name="ferroptosis_expression_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload expression CSV (rows = samples, columns = gene symbols)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse the uploaded file as CSV: {e}")
            st.stop()
        result = val.validate_expression_dataframe(df, required_genes)
        if result["is_valid"]:
            st.success(f"Valid: {result['n_samples']} sample(s), {result['n_genes_usable']}/{len(required_genes)} "
                       f"required genes present and numeric.")
        else:
            st.error("Validation failed:")
            for e in result["errors"]:
                st.write(f"- {e}")
        for w in result["warnings"]:
            st.warning(w)
        st.caption("Use this file on the Ferroptosis Classification or Drug Response pages to run a real prediction.")
