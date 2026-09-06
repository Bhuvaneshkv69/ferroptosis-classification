from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def features():
    path = ROOT / "data/processed/tcga_coadread_features.csv"
    if not path.exists():
        pytest.skip("Run src/feature_engineering.py first")
    return pd.read_csv(path)


def test_engineered_columns_present(features):
    for col in ["ferroptosis_score_mean", "ferroptosis_score_pc1", "ferroptosis_risk_group", "ferroptosis_risk_score"]:
        assert col in features.columns


def test_risk_group_is_binary_and_reasonably_balanced(features):
    counts = features["ferroptosis_risk_group"].value_counts()
    assert set(counts.index) == {"High risk", "Low risk"}
    ratio = counts.min() / counts.max()
    assert ratio > 0.8, "median split should produce roughly balanced groups"


def test_ferroptosis_score_mean_is_standardized_scale(features):
    # mean z-score across genes should itself be roughly centered around 0
    assert abs(features["ferroptosis_score_mean"].mean()) < 0.5
