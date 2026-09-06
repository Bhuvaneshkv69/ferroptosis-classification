import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403

st.set_page_config(page_title="Drug Response | Ferroptosis Pipeline", page_icon="🧬", layout="wide")
sidebar_status()
st.title("💊 Dose-Response Prediction (in-vitro cell-line sensitivity)")
st.warning(
    "Predicts an experimentally measured, in-vitro drug-sensitivity summary statistic "
    "(LN(IC50) or AUC, GDSC-fitted) for a colorectal-cancer CELL LINE. This is NOT a human "
    "patient's clinical dosage and must never be used as one.",
    icon="⚠️",
)

if not models_available():
    st.error("No persisted models found. Run `python scripts/train_and_persist.py` first.")
    st.stop()

target = st.selectbox("Predict which response measure?", ["LN_IC50", "AUC"],
                       format_func=lambda t: "LN(IC50)" if t == "LN_IC50" else "AUC")
experiment = "pooled_LN_IC50" if target == "LN_IC50" else "pooled_AUC"
available = inf.available_dose_response_models(target)
if not available:
    st.error(f"No persisted models for experiment '{experiment}'.")
    st.stop()

model_names = sorted({e["name"] for e in available})
model_name = st.selectbox("Model", model_names, index=model_names.index("RandomForest")
                           if "RandomForest" in model_names else 0)

pipeline, entry = get_dose_response_model(model_name, target)
with st.expander("Model card (from models/registry.json)"):
    st.json({k: v for k, v in entry.items() if k != "feature_columns"})

st.markdown("### Choose an input")
mode = st.radio("Input mode", ["Pick a real (cell line, drug) pair (ground truth known)",
                                "Upload gene expression + pick a drug (new sample)"])

result = None

if mode.startswith("Pick"):
    gdsc = load_gdsc_processed()
    drug = st.selectbox("Drug", sorted(gdsc["DRUG_NAME"].unique()))
    subset = gdsc[gdsc["DRUG_NAME"] == drug]
    cell_line = st.selectbox("Cell line", sorted(subset["CELL_LINE_NAME"].unique()))
    row = subset[subset["CELL_LINE_NAME"] == cell_line].iloc[0]
    if st.button("Run prediction", type="primary"):
        try:
            result = inf.predict_dose_response_from_processed_row(row, drug, model_name, target)
        except inf.MissingGenesError as e:
            st.error(str(e))

else:
    gene_cols = [c for c in entry["feature_columns"] if c not in ("MIN_CONC", "MAX_CONC", "PATHWAY_NAME")]
    template = val.make_template_csv(gene_cols)
    st.download_button("Download CSV template (gene expression, log2-RMA scale)",
                        template.to_csv(index=False).encode("utf-8"),
                        file_name="dose_response_input_template.csv", mime="text/csv")
    drugs_df = inf.get_available_drugs()
    drug = st.selectbox("Drug (its real pathway/concentration-range annotation from GDSC is used)",
                         sorted(drugs_df["DRUG_NAME"]))
    uploaded = st.file_uploader("Upload gene-expression CSV (one row per sample)", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            df = None
        if df is not None:
            v = val.validate_expression_dataframe(df, gene_cols)
            if not v["is_valid"]:
                for e in v["errors"]:
                    st.error(e)
            else:
                row_idx = 0
                if len(df) > 1:
                    row_idx = st.number_input("Row to predict", min_value=0, max_value=len(df) - 1, value=0)
                if st.button("Run prediction", type="primary"):
                    gene_expr = df.iloc[row_idx][gene_cols].to_dict()
                    try:
                        result = inf.predict_dose_response_from_raw_expression(gene_expr, drug, model_name, target)
                    except (inf.MissingGenesError, ValueError) as e:
                        st.error(str(e))

if result is not None:
    st.markdown("### Prediction")
    label = "LN(IC50)" if target == "LN_IC50" else "AUC"
    c1, c2 = st.columns(2)
    c1.metric(f"Predicted {label}", f"{result['predicted_value']:.3f}")
    if "actual_value" in result:
        c2.metric(f"Actual {label} (measured by GDSC)", f"{result['actual_value']:.3f}",
                   delta=f"{-result['absolute_error']:.3f} error", delta_color="inverse")
    if "ensemble_spread_std" in result:
        st.caption(
            f"Ensemble spread across the {model_name} trees' individual predictions on this exact input: "
            f"std={result['ensemble_spread_std']:.3f}, range=[{result['ensemble_spread_min']:.3f}, "
            f"{result['ensemble_spread_max']:.3f}]. This is a real, computed empirical spread, not a "
            f"formal statistical confidence interval."
        )

    st.write("**Reported test-set performance of this exact model** (held-out cell lines, never seen in "
             "training):")
    st.json(result["reported_test_metrics"])

    st.markdown("### Explanation (global feature importance, ferroptosis genes)")
    st.caption(
        "Precomputed permutation importance for the LN_IC50 Random-Forest model on held-out cell lines "
        "(association with prediction, not causal effect on drug sensitivity)."
    )
    perm = load_csv("results/feature_importance_regression_permutation.csv")
    st.dataframe(perm.head(15), width="stretch")

st.divider()
st.caption(DISCLAIMER)
