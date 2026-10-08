"""
Ensemble machinery and high-dimensional (whole-transcriptome) feature strategies.

* Every wt_* strategy is a sequence of pipeline steps (impute -> variance filter -> scale -> supervised selection),
  so ALL of it is fit inside the training fold.
* EnsembleSuite fits its member models ONCE (hyper-parameters tuned on the training data only) and derives four
  combination methods -- equal soft vote, learned-weight soft vote, hard (majority) vote, stacking -- from the same
  CROSS-FITTED member probabilities. Weights, the stacking meta-learner and decision thresholds are learned from
  those out-of-fold probabilities only, never from predictions of models that saw the same rows.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import randint
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

OBJECTIVES = ["accuracy", "balanced_accuracy", "f1", "mcc"]
ENSEMBLE_METHODS = ["soft_equal", "soft_weighted", "hard", "stacking"]
WT_STRATEGIES = ["wt_f", "wt_mutual_info", "wt_l1", "wt_elastic_net", "wt_rfe", "wt_pca"]


class TopVarianceK(TransformerMixin, BaseEstimator):
    """Unsupervised: keeps the k highest-variance columns of the TRAINING fold."""

    def __init__(self, k=2000):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_ = np.sort(np.argsort(X.var(axis=0))[::-1][:min(self.k, X.shape[1])])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def wt_feature_steps(strategy: str, seed: int, top_k: int = 2000):
    from model_zoo import _MutualInfo
    pre = [("impute", SimpleImputer(strategy="median")), ("var", TopVarianceK(top_k)), ("scale", StandardScaler())]
    if strategy == "wt_f":
        return pre + [("sel", SelectKBest(f_classif, k=100))], {"sel__k": randint(20, 301)}
    if strategy == "wt_mutual_info":
        return pre + [("sel", SelectKBest(_MutualInfo(seed), k=100))], {"sel__k": randint(20, 301)}
    if strategy in ("wt_l1", "wt_elastic_net"):
        ratio = 1.0 if strategy == "wt_l1" else 0.5
        sel = SelectFromModel(LogisticRegression(l1_ratio=ratio, solver="saga", C=0.05, max_iter=3000,
                                                 random_state=seed), threshold=1e-6)
        return pre + [("sel", sel)], {}
    if strategy == "wt_rfe":
        return pre + [("sel", SelectKBest(f_classif, k=300)),
                      ("rfe", RFE(LogisticRegression(C=0.05, max_iter=2000, random_state=seed),
                                  n_features_to_select=50, step=0.2))], {"rfe__n_features_to_select": randint(20, 121)}
    if strategy == "wt_pca":
        return pre + [("pca", PCA(n_components=40, random_state=seed))], {"pca__n_components": randint(20, 81)}
    raise ValueError(strategy)


def _obj_score(y, pred, objective):
    if objective == "accuracy":
        return accuracy_score(y, pred)
    if objective == "balanced_accuracy":
        return balanced_accuracy_score(y, pred)
    if objective == "f1":
        return f1_score(y, pred, zero_division=0)
    return matthews_corrcoef(y, pred)


def threshold_for(y, score, objective="mcc") -> float:
    """Decision threshold chosen on CROSS-FITTED training predictions only (middle of ties, for stability)."""
    grid = np.linspace(0.02, 0.98, 97)
    vals = np.array([_obj_score(y, score >= t, objective) for t in grid])
    return float(np.median(grid[vals >= vals.max() - 1e-12]))


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class EnsembleSuite(ClassifierMixin, BaseEstimator):
    """members: list of (name, SearchTuneCalibrate built with threshold_objective=None)."""

    def __init__(self, members=None, calibration="sigmoid", inner_folds=5, seed=42):
        self.members = members
        self.calibration = calibration
        self.inner_folds = inner_folds
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        names, fitted, cols = [], [], []
        for name, m in self.members:
            best = clone(m).fit(X, y).best_pipeline_
            cal = CalibratedClassifierCV(clone(best), method=self.calibration, cv=3) \
                if self.calibration in ("sigmoid", "isotonic") else clone(best)
            cols.append(cross_val_predict(cal, X, y, cv=cv, method="predict_proba")[:, 1])
            fitted.append(clone(cal).fit(X, y))
            names.append(name)
        O = np.column_stack(cols)
        self.names_, self.fitted_, self.oof_ = names, fitted, O
        self.member_thr_ = np.array([threshold_for(y, O[:, j], "mcc") for j in range(O.shape[1])])
        auc = np.array([roc_auc_score(y, O[:, j]) for j in range(O.shape[1])])
        w = np.clip(auc - 0.5, 0, None)
        self.weights_ = w / w.sum() if w.sum() > 0 else np.full(len(w), 1 / len(w))
        L = _logit(O)
        self.meta_ = LogisticRegression(C=1.0, max_iter=2000).fit(L, y)
        meta_oof = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), L, y,
                                     cv=StratifiedKFold(5, shuffle=True, random_state=self.seed),
                                     method="predict_proba")[:, 1]
        scores = {"soft_equal": O.mean(axis=1), "soft_weighted": O @ self.weights_, "stacking": meta_oof}
        self.thresholds_ = {m: {o: threshold_for(y, s, o) for o in OBJECTIVES} for m, s in scores.items()}
        self.thresholds_["hard"] = {o: 0.5 for o in OBJECTIVES}   # strict majority of member votes
        self.classes_ = np.array([0, 1])
        return self

    def member_proba(self, X) -> np.ndarray:
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self.fitted_])

    def scores(self, X, method: str) -> np.ndarray:
        P = self.member_proba(X)
        if method == "soft_equal":
            return P.mean(axis=1)
        if method == "soft_weighted":
            return P @ self.weights_
        if method == "stacking":
            return self.meta_.predict_proba(_logit(P))[:, 1]
        if method == "hard":
            return (P >= self.member_thr_).mean(axis=1)           # share of members voting MSI-High
        raise ValueError(method)

    def predict_method(self, X, method: str, objective: str = "mcc") -> np.ndarray:
        s = self.scores(X, method)
        thr = self.thresholds_[method][objective]
        return (s > thr).astype(int) if method == "hard" else (s >= thr).astype(int)

    def predict_proba(self, X):
        s = self.scores(X, "soft_equal")
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return self.predict_method(X, "soft_equal")


class EnsembleModel(ClassifierMixin, BaseEstimator):
    """A fitted EnsembleSuite seen through ONE combination method: the production ensemble."""

    def __init__(self, suite=None, method="stacking", objective="mcc"):
        self.suite = suite
        self.method = method
        self.objective = objective
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    @property
    def threshold_(self):
        return self.suite.thresholds_[self.method][self.objective]

    def predict_proba(self, X):
        s = self.suite.scores(X, self.method)
        return np.column_stack([1 - s, s])

    def predict(self, X):
        return self.suite.predict_method(X, self.method, self.objective)
