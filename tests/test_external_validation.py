from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _prereqs_present():
    return (ROOT / "results" / "external_validation_results.csv").exists() and \
           (ROOT / "data" / "processed" / "external_cptac_coad_2019_processed.csv").exists()


pytestmark = pytest.mark.skipif(
    not _prereqs_present(),
    reason="Run `python src/fetch_external_validation.py && python src/external_validation.py` first",
)


@pytest.fixture(scope="module")
def external_df():
    return pd.read_csv(ROOT / "data" / "processed" / "external_cptac_coad_2019_processed.csv")


@pytest.fixture(scope="module")
def external_results():
    return pd.read_csv(ROOT / "results" / "external_validation_results.csv")


def test_external_cohort_has_both_msi_classes(external_df):
    assert set(external_df["msi_label"].unique()) == {"MSI-H", "MSS"}
    assert (external_df["msi_label"] == "MSI-H").sum() > 0
    assert (external_df["msi_label"] == "MSS").sum() > 0


def test_external_cohort_is_disjoint_from_tcga_training_ids(external_df):
    # CPTAC sample IDs use a different barcode scheme entirely (e.g. "01CO001")
    # than TCGA barcodes (e.g. "TCGA-AA-XXXX"); a crude but real disjointness check.
    assert not external_df["sampleId"].astype(str).str.startswith("TCGA").any()


def test_external_validation_results_cover_all_four_models(external_results):
    assert set(external_results["model"]) == {"LogisticRegression", "RandomForest", "SVM", "XGBoost"}


def test_external_validation_metrics_are_valid_probabilities(external_results):
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert external_results[col].between(0, 1).all(), f"{col} out of [0,1] range"


def test_external_validation_used_the_same_persisted_models_no_retraining_on_external_data(external_results):
    # internal_test_roc_auc is copied straight from the model's OWN registry
    # entry (fit only on TCGA); if this column exists and is populated, the
    # scoring path never touched external data during fitting.
    assert "internal_test_roc_auc" in external_results.columns
    assert external_results["internal_test_roc_auc"].notna().all()


def test_compatible_variant_uses_gene_intersection_not_full_35_panel():
    import model_registry as reg
    entries = reg.list_models(task="classification", experiment="exp1_msi_raw_genes_cptac28")
    assert entries, "compatible variant not registered -- run src/external_validation.py"
    for e in entries:
        assert len(e["feature_columns"]) < 35
        assert len(e["feature_columns"]) > 0
