from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def _prereqs_present():
    return (ROOT / "results" / "whole_transcriptome_baseline_results.csv").exists() and \
           (ROOT / "tables" / "ferroptosis_vs_whole_transcriptome_comparison.csv").exists()


pytestmark = pytest.mark.skipif(
    not _prereqs_present(),
    reason="Run `python src/fetch_whole_transcriptome.py && python src/whole_transcriptome_baseline.py` first",
)


@pytest.fixture(scope="module")
def wt_results():
    return pd.read_csv(ROOT / "results" / "whole_transcriptome_baseline_results.csv")


@pytest.fixture(scope="module")
def comparison():
    return pd.read_csv(ROOT / "tables" / "ferroptosis_vs_whole_transcriptome_comparison.csv")


def test_whole_transcriptome_covers_all_four_models(wt_results):
    assert set(wt_results["model"]) == {"LogisticRegression", "RandomForest", "SVM", "XGBoost"}


def test_whole_transcriptome_metrics_are_valid_probabilities(wt_results):
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert wt_results[col].between(0, 1).all(), f"{col} out of [0,1] range"


def test_whole_transcriptome_used_many_more_genes_than_the_ferroptosis_panel(wt_results):
    assert (wt_results["n_genes_used"] > 1000).all()


def test_whole_transcriptome_used_leakage_safe_pca_component_count(wt_results):
    assert (wt_results["n_pca_components"] == 50).all()


def test_comparison_table_has_both_feature_sets_for_every_model(comparison):
    expected_cols = {"model", "roc_auc_ferroptosis_panel_35_genes", "roc_auc_whole_transcriptome_pca50"}
    assert expected_cols.issubset(comparison.columns)
    assert len(comparison) == 4
