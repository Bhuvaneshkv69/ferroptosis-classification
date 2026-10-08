"""
Tests for the model-selection engine: leakage safety, the selection rule, the promotion gate, the champion
registry, and (when a benchmark has been run) consistency between reported metrics and stored predictions.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import benchmark as bm  # noqa: E402
import benchmark_run as br  # noqa: E402
import model_registry as reg  # noqa: E402
import model_zoo as mz  # noqa: E402
import validation as val  # noqa: E402

CFG = bm.load_cfg()


def synth(n=240, p=12, signal=True, seed=0, pos_rate=0.2):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pos_rate).astype(int)
    X = rng.normal(size=(n, p))
    if signal:
        X[:, 0] += 2.2 * y
        X[:, 1] -= 1.5 * y
    return pd.DataFrame(X, columns=[f"G{i}" for i in range(p)]), y


def small_cfg():
    cfg = json.loads(json.dumps(CFG))
    cfg["classification"].update(search_iterations=3, inner_folds=3, outer_folds=3)
    return cfg


# ---------------------------------------------------------------- leakage safety ------------------------
def test_signature_scores_use_training_statistics_only():
    X, _ = synth()
    tr, te = X.iloc[:150].to_numpy(), X.iloc[150:].to_numpy()
    s = mz.SignatureScores(groups={"driver": [0, 1], "suppressor": [2, 3], "other": [4]}, append=True).fit(tr)
    assert np.allclose(s.mu_, tr.mean(axis=0))                      # not the pooled mean
    assert not np.allclose(s.mu_, np.vstack([tr, te]).mean(axis=0))
    assert s.transform(te).shape[0] == len(te)


def test_top_variance_filter_is_fit_on_training_rows():
    rng = np.random.default_rng(1)
    tr = rng.normal(size=(100, 6)) * np.array([5, 1, 1, 1, 1, 0.1])
    te = rng.normal(size=(50, 6)) * np.array([0.1, 1, 1, 1, 1, 5])  # opposite variance pattern
    f = mz.TopVariance(0.5).fit(tr)
    assert 0 in f.keep_ and 5 not in f.keep_
    assert f.transform(te).shape == (50, len(f.keep_))


def test_smote_is_a_pipeline_step_applied_to_training_data_only():
    X, y = synth(n=200)
    pipe, _ = mz.build_pipeline("logreg_l2", "raw", list(X.columns), 0, imbalance="smote")
    assert [n for n, _ in pipe.steps] == ["scale", "smote", "clf"]
    pipe.fit(X.iloc[:150], y[:150])
    assert len(pipe.predict(X.iloc[150:])) == 50                    # no resampling at predict time
    resampled = pipe.named_steps["smote"].fit_resample(pipe.named_steps["scale"].transform(X.iloc[:150]), y[:150])
    assert np.bincount(resampled[1])[0] == np.bincount(resampled[1])[1]  # balanced -- inside the fold only


@pytest.mark.parametrize("strategy", ["raw", "select_k_best", "mutual_info", "elastic_net_select", "rfe", "pca",
                                      "variance_filter", "raw_plus_signature"])
def test_every_feature_strategy_fits_and_transforms_unseen_rows(strategy):
    X, y = synth(n=160)
    pipe, _ = mz.build_pipeline("logreg_l2", strategy, list(X.columns), 0)
    pipe.fit(X.iloc[:120], y[:120])
    p = pipe.predict_proba(X.iloc[120:])
    assert p.shape == (40, 2) and np.isfinite(p).all()


def test_label_permutation_canary_is_near_chance_and_real_signal_is_detected():
    cfg = small_cfg()
    spec = {"kind": "single", "learner": "logreg_l2", "features": "select_k_best", "imbalance": "smote"}
    X, y = synth(n=220, signal=True)
    real = bm.nested_cv(spec, X, y, cfg, repeats=1, n_jobs=1, cache_tag="test_real")["fold_metrics"]["roc_auc"].mean()
    perm = bm.nested_cv(spec, X, np.random.default_rng(5).permutation(y), cfg, repeats=1, n_jobs=1,
                        cache_tag="test_perm")["fold_metrics"]["roc_auc"].mean()
    assert real > 0.85
    assert abs(perm - 0.5) < 0.12      # feature selection + SMOTE + tuning leak nothing under shuffled labels


def test_search_tune_calibrate_threshold_and_calibration_run_inside_fit():
    cfg = small_cfg()
    X, y = synth(n=200)
    spec = {"kind": "single", "learner": "logreg_l2", "features": "raw", "imbalance": "none", "calibration": "sigmoid"}
    est = bm.build_procedure(spec, list(X.columns), cfg).fit(X.iloc[:150], y[:150])
    assert 0.0 < est.threshold_ < 1.0 and est.best_params_
    p = est.predict_proba(X.iloc[150:])[:, 1]
    assert ((p >= 0) & (p <= 1)).all()
    assert set(np.unique(est.predict(X.iloc[150:]))) <= {0, 1}


def test_stacking_meta_model_is_trained_on_out_of_fold_predictions():
    cfg = small_cfg()
    cfg["classification"]["ensembles"]["stacking"]["members"] = ["logreg_l2", "svm_rbf"]
    X, y = synth(n=200)
    est = bm.build_procedure({"kind": "stacking", "features": "raw", "imbalance": "none"}, list(X.columns), cfg)
    est.fit(X.iloc[:150], y[:150])
    stack = est.model_.estimator_
    assert stack.cv.n_splits == cfg["classification"]["inner_folds"]   # meta-learner sees cross-fitted predictions
    assert est.predict_proba(X.iloc[150:]).shape == (50, 2)


# ---------------------------------------------------------------- locked split --------------------------
def test_locked_split_is_disjoint_stable_and_stratified(tmp_path, monkeypatch):
    monkeypatch.setattr(bm, "PROCESSED", tmp_path)
    n = 300
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"sampleId": [f"S{i}" for i in range(n)], "patientId": [f"P{i}" for i in range(n)]})
    y = (rng.random(n) < 0.18).astype(int)
    dev, test, spec, path = bm.locked_split(df, y, CFG)
    assert not set(dev) & set(test) and len(dev) + len(test) == n
    assert abs(y[test].mean() - y.mean()) < 0.03
    dev2, test2, _, _ = bm.locked_split(df, y, CFG)
    assert (test == test2).all() and (dev == dev2).all()           # reloaded verbatim from disk
    y_changed = y.copy()
    y_changed[0] = 1 - y_changed[0]
    _, test3, _, path3 = bm.locked_split(df, y_changed, CFG)       # changed data => different fingerprint, new file
    assert path3 != path
    bm.log_test_evaluation(path, "unit-test")
    assert json.load(open(path))["test_evaluations"][0]["tag"] == "unit-test"


# ---------------------------------------------------------------- metrics + selection -------------------
def test_metrics_and_bootstrap_ci():
    rng = np.random.default_rng(0)
    y = (rng.random(300) < 0.2).astype(int)
    proba = np.clip(0.5 * y + rng.random(300) * 0.6, 0, 1)
    pred = (proba > 0.5).astype(int)
    m = bm.clf_metrics(y, proba, pred)
    assert m["tp"] + m["fn"] == y.sum() and m["tn"] + m["fp"] == (1 - y).sum()
    assert m["accuracy"] == pytest.approx((m["tp"] + m["tn"]) / len(y))
    assert m["balanced_accuracy"] == pytest.approx((m["recall"] + m["specificity"]) / 2)
    ci = bm.bootstrap_ci(y, proba, pred, n=200)
    assert ci["roc_auc"][0] <= m["roc_auc"] <= ci["roc_auc"][1]


def _table(rows):
    return pd.DataFrame(rows, columns=["name", "rank", "n_folds", "roc_auc_mean", "roc_auc_std"])


def test_selection_prefers_simpler_model_within_one_se():
    t = _table([("complex", 9, 10, 0.910, 0.030), ("simple", 1, 10, 0.905, 0.030), ("bad", 2, 10, 0.80, 0.03)])
    win, why = bm.select_best(t, "roc_auc", CFG)
    assert win == "simple" and "simplest" in why


def test_selection_takes_clear_winner_and_respects_stability_gate():
    t = _table([("complex", 9, 10, 0.95, 0.02), ("simple", 1, 10, 0.85, 0.02)])
    assert bm.select_best(t, "roc_auc", CFG)[0] == "complex"
    t = _table([("spiky", 5, 10, 0.97, 0.15), ("steady", 6, 10, 0.93, 0.02)])
    assert bm.select_best(t, "roc_auc", CFG)[0] == "steady"        # std 0.15 > max_cv_std


def test_primary_metric_is_configurable_not_hardcoded():
    cfg = json.loads(json.dumps(CFG))
    t = pd.DataFrame({"name": ["a", "b"], "rank": [1, 2], "n_folds": [10, 10], "brier_mean": [0.2, 0.1],
                      "brier_std": [0.005, 0.005]})
    assert bm.select_best(t, "brier", cfg)[0] == "b"                # lower-is-better handled
    assert CFG["primary_metric"]["classification"] in bm.HIGHER_BETTER


# ---------------------------------------------------------------- promotion gate ------------------------
def _cand(auc, std, test_auc, leak=True):
    return {"cv_metrics": {"roc_auc_mean": auc, "roc_auc_std": std}, "test_metrics": {"roc_auc": test_auc},
            "leakage_checks": {"passed": leak}}


def test_promotion_requires_all_criteria():
    inc = _cand(0.88, 0.03, 0.90)
    assert br.evaluate_promotion(_cand(0.90, 0.03, 0.91), inc, CFG)[0]
    assert not br.evaluate_promotion(_cand(0.881, 0.03, 0.95), inc, CFG)[0]   # CV gain too small
    assert not br.evaluate_promotion(_cand(0.95, 0.03, 0.85), inc, CFG)[0]    # test worse
    assert not br.evaluate_promotion(_cand(0.95, 0.09, 0.95), inc, CFG)[0]    # unstable
    assert not br.evaluate_promotion(_cand(0.95, 0.03, 0.95, leak=False), inc, CFG)[0]
    assert br.evaluate_promotion(_cand(0.5, 0.5, 0.5), None, CFG)[0]          # no incumbent


# ---------------------------------------------------------------- registry / champion -------------------
def test_registry_champion_is_looked_up_not_hardcoded(tmp_path, monkeypatch):
    monkeypatch.setattr(reg, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(reg, "REGISTRY_PATH", tmp_path / "registry.json")
    monkeypatch.setattr(reg, "CHAMPIONS_PATH", tmp_path / "champions.json")
    monkeypatch.setattr(reg, "ROOT", tmp_path)
    assert reg.current_best("classification") is None
    metrics = {"roc_auc_mean": 0.9}
    e1 = reg.register_benchmark_model(task="classification", name="a", display_name="A", model_object={"x": 1},
                                      feature_columns=["g"], feature_pipeline="raw", dataset="d", hyperparameters={},
                                      cv_metrics=metrics, test_metrics={"roc_auc": 0.9}, external_metrics=None,
                                      calibration="none", version="v1")
    assert reg.current_best("classification") is None               # registering alone never promotes
    reg.promote_champion("classification", e1, "first")
    assert reg.current_best("classification")["name"] == "a"
    e2 = reg.register_benchmark_model(task="classification", name="b", display_name="B", model_object={"x": 2},
                                      feature_columns=["g"], feature_pipeline="raw", dataset="d", hyperparameters={},
                                      cv_metrics=metrics, test_metrics={"roc_auc": 0.95}, external_metrics=None,
                                      calibration="none", version="v2")
    assert reg.current_best("classification")["name"] == "a"        # better model still not auto-promoted
    reg.promote_champion("classification", e2, "gate passed")
    assert reg.current_best("classification")["name"] == "b"
    assert reg.champion_history()["classification"]["previous"]["name"] == "a"
    with pytest.raises(FileExistsError):
        reg.register_benchmark_model(task="classification", name="b", display_name="B", model_object={},
                                     feature_columns=["g"], feature_pipeline="raw", dataset="d", hyperparameters={},
                                     cv_metrics=metrics, test_metrics={}, external_metrics=None, calibration="none",
                                     version="v2")


# ---------------------------------------------------------------- training-data validation ---------------
def test_validate_training_dataframe():
    genes = [f"G{i}" for i in range(5)]
    n = 200
    df = pd.DataFrame(np.random.default_rng(0).normal(8, 1, (n, 5)), columns=genes)
    df["sampleId"] = [f"S{i}" for i in range(n)]
    df["patientId"] = [f"P{i}" for i in range(n)]
    df["msi_label"] = ["MSI-H" if i % 5 == 0 else "MSS/MSI-L" for i in range(n)]
    assert val.validate_training_dataframe(df, genes)["is_valid"]
    bad = df.copy()
    bad["patientId"] = "P1"
    assert not val.validate_training_dataframe(bad, genes)["is_valid"]
    bad = df.copy()
    bad["msi_label"] = "MSS/MSI-L"
    assert not val.validate_training_dataframe(bad, genes)["is_valid"]
    assert not val.validate_training_dataframe(df.drop(columns=["G0"]), genes)["is_valid"]


# ---------------------------------------------------------------- regression pieces ---------------------
def test_regression_group_split_holds_out_whole_cell_lines():
    import regression_benchmark as rb
    p = ROOT / "data" / "processed" / "gdsc_colorectal_dose_response_processed.csv"
    if not p.exists():
        pytest.skip("GDSC processed data missing")
    df, _, _ = rb.load_gdsc()
    dev, test, _ = rb.locked_cell_line_split(df, CFG)
    assert not set(df["COSMIC_ID"].iloc[dev]) & set(df["COSMIC_ID"].iloc[test])


def test_regression_metrics_and_drug_mean_baseline():
    import regression_benchmark as rb
    X = pd.DataFrame({"DRUG_NAME": ["a", "a", "b", "b", "c"]})
    est = rb.DrugMeanBaseline().fit(X, np.array([1.0, 3.0, 10.0, 12.0, 5.0]))
    out = est.predict(pd.DataFrame({"DRUG_NAME": ["a", "b", "zzz"]}))
    assert out[0] == 2.0 and out[1] == 11.0 and out[2] == pytest.approx(31 / 5)  # unseen drug -> global mean
    m = rb.reg_metrics([1, 2, 3, 4], [1, 2, 3, 5])
    assert m["mae"] == 0.25 and m["rmse"] == pytest.approx(0.5)


def test_grouped_search_never_sees_the_group_column_as_a_feature():
    import regression_benchmark as rb
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"COSMIC_ID": np.repeat(np.arange(12), 10), "g": rng.normal(size=120)})
    y = X["g"].to_numpy() * 2 + rng.normal(size=120) * 0.1
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline
    est = rb.GroupedSearchRegressor(Pipeline([("m", LinearRegression())]), None).fit(X, y)
    assert est.model_.named_steps["m"].n_features_in_ == 1
    assert np.corrcoef(est.predict(X), y)[0, 1] > 0.99


# ---------------------------------------------------------------- reported results are real --------------
BENCH = ROOT / "results" / "benchmark"
needs_run = pytest.mark.skipif(not (BENCH / "benchmark_summary.json").exists(),
                               reason="run scripts/run_benchmark.py first")


@needs_run
def test_reported_locked_test_metrics_match_stored_predictions():
    s = json.load(open(BENCH / "benchmark_summary.json"))
    p = pd.read_csv(BENCH / "locked_test_predictions.csv")
    m = bm.clf_metrics(p["y_true"], p["proba"], p["pred"])
    for k in ("accuracy", "roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy"):
        assert m[k] == pytest.approx(s["locked_test_metrics"][k], abs=1e-9)


@needs_run
def test_locked_test_rows_never_in_development_data():
    s = json.load(open(BENCH / "benchmark_summary.json"))
    files = sorted((ROOT / "data" / "processed").glob("locked_split_*.json"))
    files = [f for f in files if "gdsc" not in f.name]
    spec = json.load(open(files[-1]))
    oof = pd.read_csv(BENCH / "cv_oof_predictions.csv")
    assert not set(spec["test_ids"]) & set(oof["sampleId"])
    assert len(spec["test_ids"]) == s["n_test"]
    assert len(oof) == s["n_dev"]


@needs_run
def test_champion_in_registry_matches_summary_and_is_chosen_by_cv_not_test():
    s = json.load(open(BENCH / "benchmark_summary.json"))
    cur = reg.current_best("classification")
    assert cur is not None
    # production is either the single-model benchmark champion, or an ensemble that the ensemble lab promoted
    # through its predefined gate (the reason is recorded in champions.json)
    promoted_by_lab = cur["name"].startswith(("reference_", "auto_")) and         "ensemble gate" in reg.champion_history()["classification"]["reason"]
    assert cur["name"] == s["champion"] or promoted_by_lab
    zoo = pd.read_csv(BENCH / "model_zoo_cv.csv")
    chosen, _ = bm.select_best(zoo.drop(columns=["selected", "reason_not_selected"]),
                               CFG["primary_metric"]["classification"], CFG)
    assert chosen == s["champion"]                                  # recomputable from CV table alone


@needs_run
def test_verdict_is_consistent_with_numbers():
    v = json.load(open(BENCH / "benchmark_summary.json"))["verdict"]
    achieved = v["cv_accuracy_mean"] >= v["target"] and v["locked_test_accuracy"] >= v["target"]
    assert v["achieved"] == achieved
    assert v["statement"] == ("90% target achieved" if achieved else "90% target not achieved")
