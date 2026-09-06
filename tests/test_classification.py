from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_model_grid_defines_all_four_required_model_families():
    import classification
    grid = classification._model_grid()
    assert set(grid.keys()) == {"LogisticRegression", "RandomForest", "SVM", "XGBoost"}
    for name, (pipe, params) in grid.items():
        assert hasattr(pipe, "fit") and hasattr(pipe, "predict")


@pytest.fixture(scope="module")
def results():
    path = ROOT / "results/classification_results.csv"
    if not path.exists():
        pytest.skip("Run src/classification.py first")
    return pd.read_csv(path)


def test_classification_results_cover_all_experiments_and_models(results):
    assert set(results["experiment"]) == {"exp1_msi_raw_genes", "exp2_msi_engineered", "exp3_riskgroup_raw_genes"}
    assert set(results["model"]) == {"LogisticRegression", "RandomForest", "SVM", "XGBoost"}
    assert len(results) == 12


def test_classification_metrics_are_valid_probabilities(results):
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert results[col].between(0, 1).all(), f"{col} out of [0,1] range"


def test_test_set_is_never_larger_than_training_set(results):
    assert (results["n_test"] < results["n_train"]).all()
