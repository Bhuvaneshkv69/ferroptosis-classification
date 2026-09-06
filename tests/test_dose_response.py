from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def regression_results():
    path = ROOT / "results/regression_results.csv"
    if not path.exists():
        pytest.skip("Run src/dose_response.py first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def drug_results():
    path = ROOT / "results/representative_drug_results.csv"
    if not path.exists():
        pytest.skip("Run src/dose_response.py first")
    return pd.read_csv(path)


def test_pooled_results_cover_both_targets_and_three_models(regression_results):
    assert set(regression_results["experiment"]) == {"pooled_LN_IC50", "pooled_AUC"}
    assert set(regression_results["model"]) == {"LinearRegression", "RandomForest", "XGBoost"}


def test_pooled_regression_errors_are_non_negative(regression_results):
    assert (regression_results["MAE"] >= 0).all()
    assert (regression_results["RMSE"] >= 0).all()
    assert (regression_results["RMSE"] >= regression_results["MAE"]).all()


def test_representative_drugs_cover_expected_set(drug_results):
    expected = {"5-Fluorouracil", "Oxaliplatin", "Irinotecan", "Sorafenib"}
    assert set(drug_results["drug"]).issubset(expected)
    assert set(drug_results["drug"]) == expected


def test_representative_drug_sample_sizes_reasonable(drug_results):
    assert (drug_results["n"] >= 15).all()
