import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403

st.set_page_config(page_title="Model Performance | Ferroptosis Pipeline", page_icon="🧬", layout="wide")
sidebar_status()
st.title("📈 Model Performance")
st.caption("All tables and figures below are read directly from results/, tables/ and figures/ -- "
           "produced by the last real run of `python main.py`.")

if not (RESULTS_DIR / "classification_results.csv").exists():
    st.error("No results found. Run `python main.py` first.")
    st.stop()

tab1, tab2 = st.tabs(["Classification (MSI status)", "Dose-response regression"])

with tab1:
    clf = load_csv("results/classification_results.csv")

    st.subheader("Experiment 1 -- raw ferroptosis genes (primary, non-circular)")
    exp1 = clf[clf["experiment"] == "exp1_msi_raw_genes"]
    st.dataframe(exp1[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "n_train", "n_test"]]
                 .set_index("model").round(3), width="stretch")
    figure("model_comparison_msi_classification.png")

    c1, c2 = st.columns(2)
    with c1:
        figure("roc_curve_exp1_msi_raw_genes.png", "ROC curves")
    with c2:
        figure("pr_curve_exp1_msi_raw_genes.png", "Precision-recall curves")
    figure("calibration_exp1_msi_raw_genes.png", "Calibration")

    model_pick = st.selectbox("Confusion matrix for model:", sorted(exp1["model"].unique()), key="cm1")
    figure(f"confusion_matrix_exp1_msi_raw_genes_{model_pick}.png")

    with st.expander("Experiment 2 -- engineered signature score (feature-engineering comparison)"):
        exp2 = clf[clf["experiment"] == "exp2_msi_engineered"]
        st.dataframe(exp2[["model", "accuracy", "roc_auc", "f1"]].set_index("model").round(3))
        st.caption("Finding: collapsing 35 genes into a 2-feature signature score HURTS MSI classification "
                   "(best ROC-AUC drops from 0.917 to ~0.65) -- a genuine negative result.")
        figure("roc_curve_exp2_msi_engineered.png")

    with st.expander("Experiment 3 -- ferroptosis risk group (⚠️ CIRCULAR label, exploratory only)"):
        st.warning("This label is a thresholded linear function of the same genes used to predict it. "
                   "Near-perfect scores reflect circularity, not independent validity. Reported only for "
                   "comparability with the published literature it mirrors. See docs/LIMITATIONS.md.")
        exp3 = clf[clf["experiment"] == "exp3_riskgroup_raw_genes"]
        st.dataframe(exp3[["model", "accuracy", "roc_auc", "f1"]].set_index("model").round(3))
        figure("roc_curve_exp3_riskgroup_raw_genes.png")

    wt_path = RESULTS_DIR / "whole_transcriptome_baseline_results.csv"
    if wt_path.exists():
        with st.expander("Experiment 4 -- whole-transcriptome baseline (~19,200 genes, PCA-50) vs. ferroptosis panel"):
            comparison = load_csv("tables/ferroptosis_vs_whole_transcriptome_comparison.csv")
            st.dataframe(comparison.set_index("model"), width="stretch")
            st.info(
                "Honest finding: the whole-transcriptome baseline OUTPERFORMS the ferroptosis-panel-only "
                "classifier on every model (best ROC-AUC 0.986 vs. 0.917). The panel captures real but "
                "incomplete MSI signal -- biologically expected, since MSI status correlates with a "
                "genome-wide hypermutator phenotype extending well beyond ferroptosis. See RESULTS.md §4c.",
                icon="🧬",
            )
            figure("roc_curve_exp4_whole_transcriptome.png")

with tab2:
    reg_results = load_csv("results/regression_results.csv")

    target_pick = st.selectbox("Target", ["LN_IC50", "AUC"], key="reg_target")
    exp_name = f"pooled_{target_pick}"
    sub = reg_results[reg_results["experiment"] == exp_name]
    st.dataframe(sub[["model", "MAE", "RMSE", "R2", "pearson_r", "spearman_r", "n_train", "n_test",
                      "n_test_cell_lines"]].set_index("model").round(3), width="stretch")
    figure("model_comparison_dose_response.png")

    model_pick2 = st.selectbox("Model for predicted-vs-actual / residual plots:", sorted(sub["model"].unique()))
    c1, c2 = st.columns(2)
    with c1:
        figure(f"predicted_vs_actual_{target_pick}_{model_pick2}.png")
    with c2:
        figure(f"residuals_{target_pick}_{model_pick2}.png")

    with st.expander("Representative single-drug models (ferroptosis genes only, leave-one-out CV)"):
        drug_results = load_csv("results/representative_drug_results.csv")
        st.dataframe(drug_results.round(3), width="stretch")
        st.caption("With ~44 cell lines and 31 genes (n≈p), unregularized Linear Regression fails badly "
                   "(strongly negative R²) while Random Forest recovers modest real signal -- an honest, "
                   "diagnostic finding about sample-size limits, not a hidden failure.")
        drug_pick = st.selectbox("Dose-response curve:", ["5-Fluorouracil", "Oxaliplatin",
                                                            "Irinotecan", "Sorafenib"])
        real_fig = FIG_DIR / f"dose_response_curve_real_{drug_pick}.png"
        if real_fig.exists():
            c1, c2 = st.columns(2)
            with c1:
                figure(f"dose_response_curve_real_{drug_pick}.png", "REAL measured (raw per-well viability)")
            with c2:
                figure(f"dose_response_curve_{drug_pick}.png", "Illustrative (fitted-IC50 reconstruction)")
            st.caption("Real curves: normalized per-well viability from the 2.1GB raw GDSC2 file, averaged "
                       "over technical replicates, fit with a 2-parameter logistic (src/process_raw_dose_response.py). "
                       "A simplified re-implementation, not byte-identical to GDSC's own fitting pipeline -- "
                       "see docs/LIMITATIONS.md.")
        else:
            figure(f"dose_response_curve_{drug_pick}.png")
            st.caption("Reconstructed from the fitted LN_IC50 for illustration -- NOT raw per-well viability "
                       "data. Run `python src/process_raw_dose_response.py` to add the real measured curves "
                       "(requires the 2.1GB raw GDSC2 file). See docs/LIMITATIONS.md.")
