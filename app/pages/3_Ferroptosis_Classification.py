import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403

st.set_page_config(page_title="Classification | Ferroptosis Pipeline", page_icon="🧬", layout="wide")
sidebar_status()
st.title("🧪 Ferroptosis-related Gene Expression → MSI Classification")
st.caption(
    "Predicts MSI-H vs MSS/MSI-L from ferroptosis-related gene expression, using the deployed model "
    "trained and evaluated in Experiment 1 (raw genes, non-circular task -- see RESULTS.md §1)."
)

if not models_available():
    st.error("No persisted models found. Run `python scripts/train_and_persist.py` first.")
    st.stop()

EXPERIMENT = "exp1_msi_raw_genes"
available = inf.available_classification_models(EXPERIMENT)
if not available:
    st.error(f"No persisted models for experiment '{EXPERIMENT}'.")
    st.stop()

model_names = sorted({e["name"] for e in available})
model_name = st.selectbox("Model", model_names, index=model_names.index("LogisticRegression")
                           if "LogisticRegression" in model_names else 0)

pipeline, entry = get_classification_model(model_name, EXPERIMENT)
with st.expander("Model card (from models/registry.json)"):
    st.json({k: v for k, v in entry.items() if k != "feature_columns"})
    st.caption(f"Feature columns ({len(entry['feature_columns'])}): {entry['feature_columns']}")

st.markdown("### Choose an input sample")
mode = st.radio("Input mode", ["Pick a real TCGA patient sample (ground truth known)",
                                "Upload gene-expression CSV (new sample)"], horizontal=False)

result = None
truth = None

if mode.startswith("Pick"):
    samples = inf.get_tcga_test_samples()
    idx = st.selectbox(
        "TCGA sample",
        samples.index,
        format_func=lambda i: f"{samples.loc[i, 'sampleId']} (true label: {samples.loc[i, 'msi_label']})",
    )
    row = samples.loc[idx]
    truth = row["msi_label"]
    if st.button("Run prediction", type="primary"):
        try:
            result = inf.predict_msi_from_processed_row(row, model_name, EXPERIMENT)
        except inf.MissingGenesError as e:
            st.error(str(e))

else:
    required_genes = entry["feature_columns"]
    template = val.make_template_csv(required_genes)
    st.download_button("Download CSV template", template.to_csv(index=False).encode("utf-8"),
                        file_name="msi_classification_input_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Upload raw (untransformed) expression values, one row per sample", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not parse CSV: {e}")
            df = None
        if df is not None:
            v = val.validate_expression_dataframe(df, required_genes)
            if not v["is_valid"]:
                for e in v["errors"]:
                    st.error(e)
            else:
                row_idx = 0
                if len(df) > 1:
                    row_idx = st.number_input("Row to predict", min_value=0, max_value=len(df) - 1, value=0)
                if st.button("Run prediction", type="primary"):
                    raw_expr = df.iloc[row_idx][required_genes].to_dict()
                    try:
                        result = inf.predict_msi_from_raw_expression(raw_expr, model_name, EXPERIMENT)
                    except inf.MissingGenesError as e:
                        st.error(str(e))

if result is not None:
    st.markdown("### Prediction")
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Predicted label", result["predicted_label"])
        if truth is not None:
            correct = "✅ matches true label" if result["predicted_label"] == truth else "❌ differs from true label"
            st.write(f"True label: **{truth}** — {correct}")
        st.write("Class probabilities:")
        st.bar_chart(pd.Series(result["class_probabilities"]))
    with c2:
        st.write(f"**Model:** {result['model_name']} ({result['model_version']}, experiment `{result['experiment']}`)")
        st.write("**Reported test-set performance of this exact model** (from the held-out test split used "
                 "during evaluation — this describes the model's general reliability, not a per-prediction "
                 "confidence interval):")
        st.json(result["reported_test_metrics"])

    st.markdown("### Explanation")
    if result["explanation_type"] == "exact_linear_contribution":
        st.caption("Exact per-sample contribution = fitted coefficient × standardized feature value "
                   "(this model is linear, so this is not an approximation).")
        contrib = pd.DataFrame(result["local_explanation"]).set_index("gene")
        st.bar_chart(contrib["contribution"].head(15))
        st.dataframe(contrib, width="stretch")
    else:
        st.caption(
            f"{result['model_name']} is non-linear, so a per-sample explanation is not shown here; "
            "showing the model-level (global) permutation importance computed on the held-out test set instead."
        )
        perm = load_csv("results/feature_importance_classification_permutation.csv")
        st.dataframe(perm.head(15), width="stretch")

    st.caption("Feature importance/contribution reflects association with the model's output on this "
              "dataset, not proof of biological causality. See docs/LIMITATIONS.md.")

st.divider()
st.caption(DISCLAIMER)
