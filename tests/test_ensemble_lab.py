"""Tests for the ensemble lab: leakage safety of ensembles, complementarity analysis, the promotion gates,
feature-set definitions, and (when the lab has run) consistency of the stored results."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import benchmark as bm  # noqa: E402
import ensemble_lab as el  # noqa: E402
import ensemble_zoo as ez  # noqa: E402
import extended_data as ed  # noqa: E402
import model_zoo as mz  # noqa: E402

CFG = bm.load_cfg()


def small_cfg():
    cfg = json.loads(json.dumps(CFG))
    cfg["classification"].update(search_iterations=2, inner_folds=3, outer_folds=3)
    return cfg


def synth(n=260, p=8, seed=0, signal=True, noise=0.9):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=[f"G{i}" for i in range(p)])
    y = (X["G0"] + 0.6 * X["G1"] + rng.normal(size=n) * noise > 1.0).astype(int).to_numpy()
    if not signal:
        y = rng.permutation(y)
    return X, y


def suite(X, y, members=("logreg_l2", "svm_rbf", "random_forest"), cfg=None):
    cfg = cfg or small_cfg()
    return el._make_suite(list(members), "raw", "none", list(X.columns), cfg, "sigmoid")


# ------------------------------------------------------------------------------------ ensemble mechanics
def test_suite_exposes_all_four_methods_and_valid_scores():
    X, y = synth()
    s = suite(X, y).fit(X.iloc[:200], y[:200])
    for m in ez.ENSEMBLE_METHODS:
        sc = s.scores(X.iloc[200:], m)
        assert sc.shape == (60,) and ((sc >= 0) & (sc <= 1)).all()
        assert set(np.unique(s.predict_method(X.iloc[200:], m))) <= {0, 1}
        assert set(s.thresholds_[m]) == set(ez.OBJECTIVES)


def test_weights_and_meta_model_come_from_out_of_fold_predictions():
    X, y = synth()
    s = suite(X, y).fit(X.iloc[:200], y[:200])
    assert s.oof_.shape == (200, 3)                              # one cross-fitted column per member
    assert s.weights_.min() >= 0 and s.weights_.sum() == pytest.approx(1.0)
    assert s.meta_.coef_.shape == (1, 3)
    # OOF probabilities are NOT the in-sample probabilities of the refit members
    in_sample = s.member_proba(X.iloc[:200])
    assert not np.allclose(in_sample, s.oof_)


def test_hard_vote_is_strict_majority_of_member_votes():
    X, y = synth()
    s = suite(X, y).fit(X.iloc[:200], y[:200])
    Xt = X.iloc[200:]
    votes = (s.member_proba(Xt) >= s.member_thr_).sum(axis=1)
    assert (s.predict_method(Xt, "hard") == (votes > 1.5).astype(int)).all()


def test_ensemble_model_wrapper_matches_suite_and_is_sklearn_compatible():
    X, y = synth()
    s = suite(X, y).fit(X.iloc[:200], y[:200])
    em = ez.EnsembleModel(s, "stacking")
    assert np.allclose(em.predict_proba(X.iloc[200:])[:, 1], s.scores(X.iloc[200:], "stacking"))
    assert em.threshold_ == s.thresholds_["stacking"]["mcc"]
    from sklearn.base import is_classifier
    assert is_classifier(em)


def test_thresholds_are_learned_from_cross_fitted_training_predictions_only():
    X, y = synth()
    s = suite(X, y).fit(X.iloc[:200], y[:200])
    # recompute the equal-vote threshold from the stored OOF matrix: must match exactly (test rows never involved)
    assert s.thresholds_["soft_equal"]["mcc"] == pytest.approx(ez.threshold_for(y[:200], s.oof_.mean(axis=1), "mcc"))


def test_ensemble_permutation_canary_is_near_chance(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "CACHE", tmp_path)
    cfg = small_cfg()
    X, y = synth(n=240, signal=True, noise=0.3)
    yp = np.random.default_rng(3).permutation(y)
    real = el.nested_suite(["logreg_l2", "svm_rbf"], "raw", "none", X, y, cfg, 1, "sigmoid", tag="t1", n_jobs=1)
    perm = el.nested_suite(["logreg_l2", "svm_rbf"], "raw", "none", X, yp, cfg, 1, "sigmoid", tag="t2", n_jobs=1)
    for m in ez.ENSEMBLE_METHODS:
        assert real["methods"][m]["fold_metrics"]["roc_auc"].mean() > 0.8
        assert abs(perm["methods"][m]["fold_metrics"]["roc_auc"].mean() - 0.5) < 0.13


# ----------------------------------------------------------------------------------------- high-dimensional
@pytest.mark.parametrize("strategy", ez.WT_STRATEGIES)
def test_whole_transcriptome_strategies_fit_with_missing_values(strategy):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(140, 300)))
    X.iloc[3, 5] = np.nan
    y = (X[0] + X[1] > 0).astype(int).to_numpy()
    pipe, _ = mz.build_pipeline("logreg_l2", strategy, list(X.columns), 0)
    pipe.fit(X.iloc[:100], y[:100])
    assert pipe.predict_proba(X.iloc[100:]).shape == (40, 2)


def test_top_variance_k_uses_training_rows_only():
    rng = np.random.default_rng(1)
    tr = rng.normal(size=(80, 6)) * np.array([6, 1, 1, 1, 1, 0.1])
    te = rng.normal(size=(40, 6)) * np.array([0.1, 1, 1, 1, 1, 6])
    f = ez.TopVarianceK(2).fit(tr)
    assert 0 in f.keep_ and 5 not in f.keep_ and f.transform(te).shape == (40, 2)


# ------------------------------------------------------------------------------------------ complementarity
def _fake(pred_a, proba_a, n):
    oof = pd.DataFrame({"repeat": 0, "idx": np.arange(n), "proba": proba_a, "pred": pred_a})
    return {"oof": oof}


def test_complementarity_identifies_duplicates_and_diverse_models():
    rng = np.random.default_rng(0)
    n = 200
    y = (rng.random(n) < 0.3).astype(int)
    pa = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.2, n), 0, 1)
    pb = np.clip(pa + rng.normal(0, 0.01, n), 0, 1)            # near-duplicate of a
    pc = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.35, n), 0, 1)  # noisier, different errors
    res = {"a": _fake((pa > .5).astype(int), pa, n), "b": _fake((pb > .5).astype(int), pb, n),
           "c": _fake((pc > .5).astype(int), pc, n)}
    comp = el.complementarity(res, y)
    assert comp["proba_corr"].loc["a", "b"] > 0.98 and comp["disagreement"].loc["a", "b"] < 0.03
    assert comp["disagreement"].loc["a", "c"] > comp["disagreement"].loc["a", "b"]
    assert comp["oracle_accuracy"] >= comp["best_single_accuracy_oof"]
    table = pd.DataFrame({"name": ["a", "b", "c"], "roc_auc_mean": [0.90, 0.899, 0.88]})
    chosen, log = el.auto_members(table, comp, CFG)
    assert chosen == ["a", "c"]                                  # b excluded as a near-duplicate
    assert any(r["model"] == "b" and "duplicate" in r["why"] for r in log)


# ------------------------------------------------------------------------------------------------ the gates
def _run(auc, acc, bal, rec, std=0.03, n=10, offset=0.0):
    fm = pd.DataFrame({"roc_auc": np.full(n, auc) + offset + np.linspace(-std, std, n),
                       "accuracy": np.full(n, acc), "balanced_accuracy": np.full(n, bal), "recall": np.full(n, rec),
                       "precision": 0.6, "f1": 0.6, "mcc": 0.5, "specificity": 0.9, "pr_auc": 0.6, "brier": 0.09,
                       "ece": 0.03})
    return {"name": "x", "fold_metrics": fm, "n_folds": n, "secs_per_fold": 1.0}


def test_cv_gate_requires_every_condition():
    single = _run(0.880, 0.88, 0.75, 0.55)
    assert el.cv_gate(_run(0.890, 0.89, 0.76, 0.56, offset=0.0), single, 0.05, CFG)[0] is True
    assert el.cv_gate(_run(0.881, 0.88, 0.75, 0.55), single, 0.05, CFG)[0] is False            # no AUC gain
    assert el.cv_gate(_run(0.890, 0.89, 0.70, 0.56), single, 0.05, CFG)[0] is False            # balanced acc drops
    assert el.cv_gate(_run(0.890, 0.91, 0.76, 0.30), single, 0.05, CFG)[0] is False            # recall collapse
    assert el.cv_gate(_run(0.890, 0.89, 0.76, 0.56), single, 0.001, CFG)[0] is False           # near-duplicate members


def test_test_gate_checks_locked_test_without_choosing_the_candidate():
    ok, _ = el.test_gate({"roc_auc": 0.83, "balanced_accuracy": 0.70}, {"roc_auc": 0.82, "balanced_accuracy": 0.69}, CFG)
    bad, _ = el.test_gate({"roc_auc": 0.75, "balanced_accuracy": 0.70}, {"roc_auc": 0.82, "balanced_accuracy": 0.69}, CFG)
    assert ok and not bad


def test_threshold_tradeoff_uses_development_predictions_only():
    rng = np.random.default_rng(0)
    yd = (rng.random(300) < 0.2).astype(int)
    pd_ = np.clip(0.25 + 0.4 * yd + rng.normal(0, 0.2, 300), 0, 1)
    yt = (rng.random(100) < 0.2).astype(int)
    pt = np.clip(0.25 + 0.4 * yt + rng.normal(0, 0.2, 100), 0, 1)
    tr = el.threshold_tradeoff(pd_, yd, pt, yt).set_index("objective")
    for obj in ez.OBJECTIVES:
        assert tr.loc[obj, "threshold"] == pytest.approx(ez.threshold_for(yd, pd_, obj))
    pt2 = np.clip(pt + 0.3, 0, 1)                                 # change test predictions: thresholds must not move
    tr2 = el.threshold_tradeoff(pd_, yd, pt2, yt).set_index("objective")
    assert (tr["threshold"] == tr2["threshold"]).all()


# ------------------------------------------------------------------------------------------ feature sets
def test_feature_sets_are_nested_and_from_the_config():
    core, mmr, ext = ed.feature_set_genes("core"), ed.feature_set_genes("core_plus_mmr"), ed.feature_set_genes("extended_molecular")
    cfg = ed.feature_config()
    assert len(core) == 35 and len(mmr) == 39 and len(ext) == 53
    assert set(core) < set(mmr) < set(ext)
    assert set(cfg["mmr"]["genes"]) <= set(mmr) and set(cfg["immune"]["genes"]) <= set(ext)
    assert len(cfg["immune"]["justification"]) > 50


def test_extended_matrices_align_with_the_labelled_samples():
    if not ed.WT_CACHE.exists() and not (ed.RAW / "whole_transcriptome_expression_long.csv").exists():
        pytest.skip("whole-transcriptome data not available")
    df, genes, y = bm.load_msi_data()
    X = ed.load_matrix("extended_molecular", df["sampleId"])
    assert X.shape == (len(df), 53) and list(X.columns[:35]) == genes
    assert not X.isna().any().any()
    wt = ed.load_matrix("whole_transcriptome", df["sampleId"])
    assert len(wt) == len(df) and wt.shape[1] > 10000


# --------------------------------------------------------------------------- stored results are consistent
LABF = ROOT / "results" / "ensemble_lab" / "lab_summary.json"
needs_lab = pytest.mark.skipif(not LABF.exists(), reason="run scripts/run_ensemble_lab.py first")


@needs_lab
def test_stored_locked_test_metrics_match_stored_predictions_for_every_feature_set():
    lab = json.load(open(LABF))
    for r in lab["table"]:
        d = ROOT / "results" / "ensemble_lab" / r["feature_set"]
        s = json.load(open(d / "summary.json"))
        p = pd.read_csv(d / "locked_test_predictions.csv")
        m = bm.clf_metrics(p["y_true"], p["proba"], p["pred"])
        for k in ("accuracy", "roc_auc", "recall", "mcc"):
            assert m[k] == pytest.approx(s["final"]["test"][k], abs=1e-9)
        assert r["test_accuracy"] == pytest.approx(m["accuracy"], abs=1e-9)


@needs_lab
def test_every_feature_set_uses_the_same_locked_test_samples():
    lab = json.load(open(LABF))
    files = [f for f in sorted((ROOT / "data" / "processed").glob("locked_split_*.json")) if "gdsc" not in f.name]
    test_ids = set(json.load(open(files[-1]))["test_ids"])
    for r in lab["table"]:
        p = pd.read_csv(ROOT / "results" / "ensemble_lab" / r["feature_set"] / "locked_test_predictions.csv")
        assert set(p["sampleId"]) == test_ids


@needs_lab
def test_ensemble_is_adopted_only_if_every_gate_passed():
    lab = json.load(open(LABF))
    for r in lab["table"]:
        s = json.load(open(ROOT / "results" / "ensemble_lab" / r["feature_set"] / "summary.json"))
        if s["final"]["kind"] == "ensemble":
            g = s["gate"]
            assert g["adopted"] and all(c["passed"] for c in g["cv"]) and all(c["passed"] for c in g["test"])
            assert g["leakage"]["passed"]


@needs_lab
def test_extended_models_are_labelled_and_never_registered_as_the_core_model():
    import model_registry as reg
    core = reg.current_best("classification")
    ext = reg.current_best("classification_extended")
    assert core["extra"]["feature_set"] == "core" and core["extra"]["n_features"] == 35
    if ext:
        assert ext["extra"]["feature_set"] in ("extended_molecular", "core_plus_mmr")
        assert ext["extra"]["n_features"] > 35 and "ferroptosis core" not in ext["display_name"].lower().replace("extended", "")
