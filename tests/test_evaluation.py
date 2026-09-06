import numpy as np

import evaluation as ev


def test_evaluate_classification_perfect_predictions():
    y_true = np.array(["MSI-H", "MSI-H", "MSS/MSI-L", "MSS/MSI-L"])
    y_pred = y_true.copy()
    y_proba = np.array([0.9, 0.8, 0.1, 0.2])
    m = ev.evaluate_classification(y_true, y_pred, y_proba, pos_label="MSI-H")
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["roc_auc"] == 1.0


def test_evaluate_classification_metrics_are_bounded():
    rng = np.random.default_rng(0)
    y_true = rng.choice(["A", "B"], size=50)
    y_pred = rng.choice(["A", "B"], size=50)
    y_proba = rng.random(50)
    m = ev.evaluate_classification(y_true, y_pred, y_proba, pos_label="A")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        assert 0.0 <= m[key] <= 1.0


def test_evaluate_regression_perfect_predictions():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = y_true.copy()
    m = ev.evaluate_regression(y_true, y_pred)
    assert m["MAE"] == 0.0
    assert m["RMSE"] == 0.0
    assert m["R2"] == 1.0
    assert abs(m["pearson_r"] - 1.0) < 1e-9


def test_evaluate_regression_known_error():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([2.0, 3.0, 4.0, 5.0])  # constant +1 offset
    m = ev.evaluate_regression(y_true, y_pred)
    assert abs(m["MAE"] - 1.0) < 1e-9
    assert abs(m["pearson_r"] - 1.0) < 1e-9  # perfectly correlated despite offset
