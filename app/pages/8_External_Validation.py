import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))
from common import *  # noqa: E402,F401,F403

st.set_page_config(page_title="External Validation | Ferroptosis Pipeline", page_icon="🧬", layout="wide")
sidebar_status()
st.title("🌍 External Validation")
st.info(
    "Every result elsewhere in this application is INTERNAL validation: a held-out split of the "
    "same TCGA COADREAD cohort used for training. This page is the one genuine EXTERNAL test: the "
    "already-persisted, already-evaluated MSI classifiers (no retraining, no tuning, no feature "
    "selection performed here) are scored on CPTAC-2 Prospective Colon Cancer (Vasaikar et al., "
    "Cell 2019) -- an independent cohort from a different research consortium, different patients, "
    "and a different RNA-seq processing batch than TCGA.",
    icon="🌍",
)

ext_results_path = RESULTS_DIR / "external_validation_results.csv"
if not ext_results_path.exists():
    st.error(
        "External validation has not been run yet. From the project root: "
        "`python src/fetch_external_validation.py && python src/external_validation.py`"
    )
    st.stop()

ext_meta_path = PROCESSED_DIR / "external_cptac_coad_2019_metadata.json"
if ext_meta_path.exists():
    with open(ext_meta_path) as f:
        ext_meta = json.load(f)
    c1, c2 = st.columns(2)
    c1.metric("External cohort samples (MSI-labeled)", ext_meta["n_samples"])
    c2.write("**MSI label distribution (CPTAC's own MSI_STATUS call):**")
    c2.json(ext_meta["msi_label_counts"])

results = load_csv("results/external_validation_results.csv")
st.subheader("Internal (TCGA held-out) vs. external (CPTAC) performance -- same exact model")
compare = results[["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
                    "internal_test_roc_auc"]].rename(columns={
    "roc_auc": "external_roc_auc", "internal_test_roc_auc": "internal_roc_auc"})
compare = compare[["model", "internal_roc_auc", "external_roc_auc", "accuracy", "precision", "recall",
                    "f1", "pr_auc"]]
st.dataframe(compare.round(3), width="stretch", hide_index=True)

drop = (compare["internal_roc_auc"] - compare["external_roc_auc"]).mean()
st.caption(
    f"Average ROC-AUC drop from internal to external validation across models: {drop:+.3f}. "
    "A drop here is normal and expected -- it reflects real cross-cohort, cross-platform "
    "generalization, not a bug. Read the number honestly rather than as a failure."
)

figure("roc_curve_external_validation_cptac.png", "ROC curves on the external CPTAC-2 cohort")

model_pick = st.selectbox("Confusion matrix for model:", sorted(results["model"].unique()))
figure(f"confusion_matrix_external_cptac_{model_pick}.png")

st.markdown("### Why this counts as genuine external validation")
st.markdown("""
- **Different consortium and patients**: CPTAC (NCI Clinical Proteomic Tumor Analysis Consortium)
  vs. TCGA -- no patient overlap.
- **Different label source**: CPTAC's `MSI_STATUS` is called independently by that study, not
  derived from the same MANTIS-score pipeline used for the TCGA training label (see `docs/DATASET.md`).
- **No fitting on this cohort, ever**: `src/external_validation.py` only loads already-persisted
  models from `models/registry.json` and calls `.predict()` -- no scaler, gene list, hyperparameter,
  or threshold was touched using CPTAC data.
""")
st.caption(DISCLAIMER)
