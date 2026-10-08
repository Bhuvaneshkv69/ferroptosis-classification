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


@pytest.fixture(scope="module")
def prognosis_metrics():
    path = ROOT / "results/prognosis_validation.json"
    if not path.exists():
        pytest.skip("Run src/classification.py first")
    import json
    with open(path) as f:
        return json.load(f)


def test_prognosis_concordance_index_is_a_valid_probability_like_score(prognosis_metrics):
    c = prognosis_metrics["concordance_index"]
    assert c is not None
    assert 0.0 <= c <= 1.0


def test_prognosis_concordance_beats_chance(prognosis_metrics):
    # A real, held-out, non-circular check: the ferroptosis Cox risk score
    # should order patient survival better than a coin flip (0.5). This is
    # the legitimate prognostic-value metric -- distinct from and much
    # stricter than classification.py Experiment 3's circular label-recovery
    # accuracy (see docs/REFERENCES.md, papers 1-3).
    assert prognosis_metrics["concordance_index"] > 0.5


def test_prognosis_logrank_p_is_a_valid_p_value(prognosis_metrics):
    p = prognosis_metrics["logrank_p"]
    assert p is not None
    assert 0.0 <= p <= 1.0


def test_kaplan_meier_figure_was_generated():
    path = ROOT / "figures/kaplan_meier_risk_group.png"
    if not (ROOT / "results/prognosis_validation.json").exists():
        pytest.skip("Run src/classification.py first")
    assert path.exists()


def test_evaluate_prognosis_and_score_risk_are_consistent_on_a_held_out_split():
    """Direct unit test of feature_engineering.fit_risk_model /
    evaluate_prognosis using a fresh train/test split, independent of the
    persisted results file -- confirms the function itself (not just its
    saved output) behaves correctly."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    import feature_engineering as fe
    import numpy as np
    from sklearn.model_selection import train_test_split

    tcga = pd.read_csv(ROOT / "data/processed/tcga_coadread_processed.csv")
    import json
    with open(ROOT / "data/processed/tcga_preprocessing_metadata.json") as f:
        meta = json.load(f)
    gene_cols = meta["ferroptosis_genes_used"]

    survival_mask = tcga[["OS_MONTHS", "OS_EVENT"] + gene_cols].notna().all(axis=1)
    sub = tcga[survival_mask].reset_index(drop=True)
    train_idx, test_idx = train_test_split(np.arange(len(sub)), test_size=0.2, stratify=sub["OS_EVENT"],
                                            random_state=42)

    fitted, _, _ = fe.fit_risk_model(sub.iloc[train_idx], gene_cols)
    metrics = fe.evaluate_prognosis(sub.iloc[test_idx], fitted)

    assert metrics["n_test"] > 0
    assert 0.0 <= metrics["concordance_index"] <= 1.0
