import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403
import plotly.express as px  # noqa: E402
from theme import page_header, styled_figure, chart, tokens  # noqa: E402

CATEGORICAL = tokens()["categorical"]   # series colours of the active theme

page_header("Data", "Cohorts, gene signature, and expression statistics behind every analysis")

if not data_available():
    st.error("Processed data not found. Run `python main.py` from the project root first.")
    st.stop()

tcga_meta = load_tcga_meta()
gdsc_meta = load_gdsc_meta()
tcga = load_tcga_features()
gdsc = load_gdsc_processed()
genes = load_gene_signature()

GENE_LIST = genes["gene"].tolist()

tab_tcga, tab_gdsc, tab_genes, tab_upload = st.tabs(
    ["TCGA cohort", "GDSC cell lines", "Gene signature", "Upload & validate"]
)

with tab_tcga:
    c1, c2, c3 = st.columns(3)
    c1.metric("Samples", tcga_meta["n_samples"])
    c2.metric("Ferroptosis genes used", tcga_meta["n_ferroptosis_genes_used"])
    c3.metric("Genes missing from profile", len(tcga_meta["ferroptosis_genes_missing_from_profile"]))

    cc1, cc2 = st.columns(2)
    with cc1:
        st.write("**MSI status distribution**")
        counts = pd.Series(tcga_meta["msi_label_counts"])
        fig = px.bar(x=counts.index, y=counts.values, labels={"x": "", "y": "samples"},
                     color=counts.index, color_discrete_sequence=CATEGORICAL)
        fig.update_layout(showlegend=False)
        chart(fig, 260, width="stretch", config={"displayModeBar": False})
        st.caption(f"MANTIS score ≥ {tcga_meta['msi_mantis_threshold']} = MSI-H (Kautto et al., Oncotarget 2017).")
    with cc2:
        if "ferroptosis_risk_group" in tcga.columns:
            st.write("**Ferroptosis risk-group distribution** (exploratory)")
            rc = tcga["ferroptosis_risk_group"].value_counts()
            fig = px.bar(x=rc.index, y=rc.values, labels={"x": "", "y": "samples"},
                         color=rc.index, color_discrete_sequence=CATEGORICAL)
            fig.update_layout(showlegend=False)
            chart(fig, 260, width="stretch", config={"displayModeBar": False})
            st.caption("Circular label (thresholded from the same genes) -- exploratory only, not used in Analysis.")

    with st.expander("Expression statistics (computed live from the processed cohort)"):
        gene_cols = [g for g in GENE_LIST if g in tcga.columns]
        stats = tcga[gene_cols].agg(["mean", "median", "std"]).T
        stats["missing"] = tcga[gene_cols].isna().sum().values
        stats["samples_detected"] = (len(tcga) - stats["missing"]).astype(int)
        stats = stats.rename(columns={"mean": "Mean", "median": "Median", "std": "Std"}).round(3)
        stats.index.name = "Gene"
        st.dataframe(stats.reset_index(), width="stretch", hide_index=True)

    with st.expander("Preprocessing details"):
        st.markdown(f"""
        - Transform: log2(RSEM value, clipped ≥ 0, + 1) per gene
        - Genes retained: {tcga_meta['n_ferroptosis_genes_used']} / {len(genes)}
          (missing from RNA-seq V2 profile: {tcga_meta['ferroptosis_genes_missing_from_profile'] or 'none'})
        - Random seed: {tcga_meta['random_seed']}
        """)
        log_path = DOCS_DIR / "review_evidence" / "tcga_preprocessing_log.txt"
        if log_path.exists():
            st.code(log_path.read_text(encoding="utf-8"), language=None)

    with st.expander("Raw processed table"):
        st.dataframe(tcga.head(200), width="stretch")

with tab_gdsc:
    c1, c2, c3 = st.columns(3)
    c1.metric("Cell lines", gdsc_meta["n_cell_lines"])
    c2.metric("Drugs (well-covered)", gdsc_meta["n_drugs"])
    c3.metric("Genes missing from array", len(gdsc_meta["ferroptosis_genes_missing_from_array"]))

    st.write("**Rows per drug pathway**")
    pathway_counts = gdsc["PATHWAY_NAME"].value_counts().head(15)
    fig = px.bar(x=pathway_counts.values, y=pathway_counts.index, orientation="h",
                 labels={"x": "rows", "y": ""}, color_discrete_sequence=[CATEGORICAL[0]])
    fig.update_layout(yaxis=dict(autorange="reversed"))
    chart(fig, 420, width="stretch", config={"displayModeBar": False})

    with st.expander("Expression statistics (computed live from the processed cohort)"):
        gene_cols = [g for g in GENE_LIST if g in gdsc.columns]
        stats = gdsc[gene_cols].agg(["mean", "median", "std"]).T
        stats["missing"] = gdsc[gene_cols].isna().sum().values
        stats["samples_detected"] = (len(gdsc) - stats["missing"]).astype(int)
        stats = stats.rename(columns={"mean": "Mean", "median": "Median", "std": "Std"}).round(3)
        stats.index.name = "Gene"
        st.dataframe(stats.reset_index(), width="stretch", hide_index=True)

    with st.expander("Preprocessing details"):
        st.markdown(f"""
        - Minimum drug coverage filter: ≥ {gdsc_meta['min_drug_coverage']} colorectal cell lines tested per drug
        - Genes retained: {len(gdsc_meta['ferroptosis_genes_used'])} / {len(genes)}
          (missing from array: {gdsc_meta['ferroptosis_genes_missing_from_array'] or 'none'})
        - Scaling: none applied at this stage -- fit only inside the train-only sklearn Pipeline (see Documentation)
        - Random seed: {gdsc_meta['random_seed']}
        """)
        log_path = DOCS_DIR / "review_evidence" / "gdsc_preprocessing_log.txt"
        if log_path.exists():
            st.code(log_path.read_text(encoding="utf-8"), language=None)

    with st.expander("Raw processed table"):
        st.dataframe(gdsc.head(200), width="stretch")

with tab_genes:
    st.write(f"{len(genes)} genes, literature-traceable to 4 CRC prognostic-signature papers + 1 mechanistic review.")
    search = st.text_input("Filter by gene symbol", "", placeholder="e.g. GPX4")
    view = genes[genes["gene"].str.contains(search, case=False, na=False)] if search else genes
    st.dataframe(view, width="stretch", hide_index=True)

with tab_upload:
    st.caption(
        "Checks the file against the ferroptosis gene panel for duplicate columns, missing genes, "
        "non-numeric values, and zero-variance columns before any model sees it."
    )
    required_genes = val.load_required_gene_panel()
    template = val.make_template_csv(required_genes)
    st.download_button(
        "Download CSV template",
        template.to_csv(index=False).encode("utf-8"),
        file_name="ferroptosis_expression_template.csv",
        mime="text/csv",
    )
    _tf = load_tcga_features()
    st.download_button(
        f"Download full TCGA dataset ({len(_tf)} samples, raw scale)",
        full_cohort_csv(_tf, required_genes, raw_scale=True),
        file_name="tcga_coadread_all_samples_raw_expression.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload expression profile CSV (rows = samples, columns = genes)", type=["csv"])
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
            st.error("Validation failed.")
            for e in result["errors"]:
                st.write(f"- {e}")
        for w in result["warnings"]:
            st.warning(w)
        if result["is_valid"]:
            st.caption("Use this file in Analysis or Drug Response to run a prediction on it.")
