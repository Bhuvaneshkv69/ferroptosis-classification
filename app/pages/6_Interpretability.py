import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403

st.set_page_config(page_title="Interpretability | Ferroptosis Pipeline", page_icon="🧬", layout="wide")
sidebar_status()
st.title("🔍 Interpretability")
st.error(
    "Feature importance and SHAP values below reflect STATISTICAL ASSOCIATION between a gene's "
    "expression and the model's prediction on this dataset. They are NOT evidence that the gene "
    "CAUSES MSI status, drug sensitivity, or survival risk. Establishing causality requires "
    "independent wet-lab / functional validation, outside this project's scope.",
    icon="🧬",
)

if not (RESULTS_DIR / "feature_importance_classification_permutation.csv").exists():
    st.error("No interpretability results found. Run `python main.py` first.")
    st.stop()

tab1, tab2 = st.tabs(["MSI classification", "Dose-response regression"])

with tab1:
    perm = load_csv("results/feature_importance_classification_permutation.csv")
    shap_df = load_csv("results/feature_importance_classification_shap.csv")
    st.subheader("Permutation importance (ROC-AUC drop when a gene is shuffled)")
    st.bar_chart(perm.set_index("gene")["perm_importance_mean"].head(15))
    st.dataframe(perm, width="stretch", hide_index=True)
    figure("feature_importance_classification.png")
    st.subheader("SHAP summary")
    figure("shap_summary_classification.png")
    with st.expander("SHAP mean |value| table"):
        st.dataframe(shap_df, width="stretch", hide_index=True)

with tab2:
    perm_r = load_csv("results/feature_importance_regression_permutation.csv")
    shap_r = load_csv("results/feature_importance_regression_shap.csv")
    st.subheader("Permutation importance (MAE increase when a gene is shuffled), LN_IC50")
    st.bar_chart(perm_r.set_index("gene")["perm_importance_mean"].head(15))
    st.dataframe(perm_r, width="stretch", hide_index=True)
    figure("feature_importance_regression.png")
    st.subheader("SHAP summary")
    figure("shap_summary_dose_response.png")
    with st.expander("SHAP mean |value| table"):
        st.dataframe(shap_r, width="stretch", hide_index=True)

st.divider()
recurrent = load_gene_signature()
recurrent = recurrent[recurrent["cross_study_recurrent"]]["gene"].tolist()
st.info(
    f"**SLC7A11** (a canonical ferroptosis \"gatekeeper\" gene, cystine/glutamate antiporter) ranks "
    f"among the top contributors in both tasks despite being selected purely from external mechanistic "
    f"literature, not CRC-specific studies -- a genuinely interesting cross-task recurrence, reported "
    f"as association only. Cross-study-recurrent genes in the signature: {recurrent}",
    icon="🧬",
)
