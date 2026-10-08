from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def resistance_results():
    path = ROOT / "results/drug_resistance_results.csv"
    if not path.exists():
        pytest.skip("Run src/drug_resistance.py first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def oxaliplatin_results():
    path = ROOT / "results/oxaliplatin_resistance_results.csv"
    if not path.exists():
        pytest.skip("Run src/drug_resistance.py first")
    return pd.read_csv(path)


def test_pooled_resistance_results_cover_both_models(resistance_results):
    assert set(resistance_results["model"]) == {"LogisticRegression", "RandomForest"}
    assert (resistance_results["experiment"] == "pooled_resistance").all()


def test_pooled_resistance_metrics_are_valid_probabilities_and_rates(resistance_results):
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert (resistance_results[col] >= 0).all() and (resistance_results[col] <= 1).all()


def test_oxaliplatin_resistance_results_cover_both_models(oxaliplatin_results):
    assert set(oxaliplatin_results["model"]) == {"LogisticRegression", "RandomForest"}
    assert (oxaliplatin_results["drug"] == "Oxaliplatin").all()
    assert (oxaliplatin_results["n"] >= 15).all()


def test_thresholds_file_has_per_drug_and_fallback_values():
    path = ROOT / "data/processed/drug_resistance_thresholds.json"
    if not path.exists():
        pytest.skip("Run src/drug_resistance.py first")
    import json
    with open(path) as f:
        thresholds = json.load(f)
    assert "_global_fallback" in thresholds["pooled_thresholds"]
    assert "Oxaliplatin" in thresholds["pooled_thresholds"]


def test_label_resistance_uses_only_the_training_fold_threshold():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import drug_resistance as dr

    train = pd.DataFrame({"DRUG_NAME": ["DrugA"] * 4, "LN_IC50": [1.0, 2.0, 3.0, 4.0]})
    test = pd.DataFrame({"DRUG_NAME": ["DrugA"] * 2, "LN_IC50": [0.5, 100.0]})

    train_label, thresholds = dr.label_resistance(train, train)
    test_label, thresholds_test = dr.label_resistance(train, test)

    # threshold must be the TRAIN median (2.5), independent of the values in `test`
    assert thresholds["DrugA"] == pytest.approx(2.5)
    assert thresholds_test["DrugA"] == pytest.approx(2.5)
    assert list(test_label) == ["Sensitive", "Resistant"]


def test_label_resistance_falls_back_to_global_median_for_an_unseen_drug():
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import drug_resistance as dr

    train = pd.DataFrame({"DRUG_NAME": ["DrugA", "DrugA", "DrugB", "DrugB"], "LN_IC50": [1.0, 2.0, 10.0, 20.0]})
    test = pd.DataFrame({"DRUG_NAME": ["DrugC"], "LN_IC50": [8.25]})  # global median of train = 6.0

    label, thresholds = dr.label_resistance(train, test)
    assert thresholds["_global_fallback"] == pytest.approx(6.0)
    assert list(label) == ["Resistant"]
