"""
Model zoo for the model-selection engine: leakage-safe feature strategies,
classifiers, regressors, hyper-parameter search spaces and ensembles.

Design rules
------------
* Every object that LEARNS from data (scaler, selector, PCA, SMOTE, signature
  z-scoring, calibrator, decision threshold) lives inside an sklearn/imblearn
  Pipeline or meta-estimator, so it is refit on each training fold and only
  applied (transform / predict) to validation and test rows.
* SMOTE is a Pipeline step (imblearn), which means it only ever resamples the
  training portion of a fold -- never the data handed to cross-validation.
* Search spaces are deliberately small (RandomizedSearchCV, not big grids).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,
                              GradientBoostingRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor, VotingRegressor)
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, TunedThresholdClassifierCV)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent

try:  # optional libraries -- the benchmark records "not evaluated" if absent
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except Exception:  # pragma: no cover
    HAS_LIGHTGBM = False
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CATBOOST = True
except Exception:  # pragma: no cover
    HAS_CATBOOST = False


# ---------------------------------------------------------------------------
# Feature strategies (all fitted inside the training fold)
# ---------------------------------------------------------------------------

class TopVariance(TransformerMixin, BaseEstimator):
    """Unsupervised variance filter: keeps the `keep_fraction` highest-variance columns of the TRAINING fold."""

    def __init__(self, keep_fraction=0.7):
        self.keep_fraction = keep_fraction

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        k = max(2, int(round(self.keep_fraction * X.shape[1])))
        self.keep_ = np.sort(np.argsort(X.var(axis=0))[::-1][:k])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


class SignatureScores(TransformerMixin, BaseEstimator):
    """Ferroptosis pathway scores from FerrDb v2 gene roles.

    Per-gene z-scoring uses TRAINING-fold statistics; the score is the mean z of
    each role group (driver / suppressor / other) plus driver-minus-suppressor.
    `groups` maps role -> list of column indices; `append=True` keeps the raw
    genes alongside the scores.
    """

    def __init__(self, groups=None, append=False):
        self.groups = groups
        self.append = append

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.mu_, self.sd_ = X.mean(axis=0), X.std(axis=0) + 1e-9
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        Z = (X - self.mu_) / self.sd_
        cols = []
        for role in ("driver", "suppressor", "other"):
            idx = (self.groups or {}).get(role, [])
            cols.append(Z[:, idx].mean(axis=1) if idx else np.zeros(len(Z)))
        cols.append(cols[0] - cols[1])
        S = np.column_stack(cols)
        return np.hstack([Z, S]) if self.append else S


def ferrdb_gene_groups(feature_names: list[str]) -> dict[str, list[int]]:
    """FerrDb v2 driver/suppressor role for each feature (a literature annotation -- uses NO sample data,
    so it cannot leak). Genes listed as both driver and suppressor, or in neither, go to 'other'."""
    d = ROOT / "data" / "raw" / "ferrdb"
    drivers, supp = set(), set()
    for f in ("ferrdb_v2_driver.csv", "ferrdb_v2_driver_suppressor_marker.csv"):
        t = pd.read_csv(d / f)
        drivers |= set(t.loc[t["gene_type"].astype(str).str.contains("driver"), "symbol"].str.upper())
        supp |= set(t.loc[t["gene_type"].astype(str).str.contains("suppressor"), "symbol"].str.upper())
    supp |= set(pd.read_csv(d / "ferrdb_v2_suppressor.csv")["symbol"].str.upper())
    drivers |= set(pd.read_csv(d / "ferrdb_v2_driver.csv")["symbol"].str.upper())
    groups = {"driver": [], "suppressor": [], "other": []}
    for i, g in enumerate(feature_names):
        g = g.upper()
        if g in drivers and g not in supp:
            groups["driver"].append(i)
        elif g in supp and g not in drivers:
            groups["suppressor"].append(i)
        else:
            groups["other"].append(i)
    return groups


class _MutualInfo:
    """Picklable score function for SelectKBest (a lambda would break joblib persistence)."""

    def __init__(self, seed):
        self.seed = seed

    def __call__(self, X, y):
        return mutual_info_classif(X, y, random_state=self.seed)


def feature_steps(strategy: str, feature_names: list[str], seed: int) -> tuple[list, dict]:
    """Returns (pipeline steps, search-space additions) for a feature strategy."""
    if strategy.startswith("wt_"):  # whole-transcriptome strategies live in ensemble_zoo (lazy import)
        from ensemble_zoo import wt_feature_steps
        return wt_feature_steps(strategy, seed)
    scaler = ("scale", StandardScaler())
    if strategy == "raw":
        return [scaler], {}
    if strategy == "variance_filter":
        return [("var", TopVariance()), scaler], {"var__keep_fraction": uniform(0.5, 0.5)}
    if strategy == "select_k_best":
        return [scaler, ("sel", SelectKBest(f_classif, k=15))], {"sel__k": randint(5, 31)}
    if strategy == "mutual_info":
        return [scaler, ("sel", SelectKBest(_MutualInfo(seed), k=15))], {"sel__k": randint(5, 31)}
    if strategy == "elastic_net_select":
        sel = SelectFromModel(LogisticRegression(l1_ratio=0.5, solver="saga", C=0.1, max_iter=5000,
                                                 random_state=seed), threshold=1e-6)
        return [scaler, ("sel", sel)], {}
    if strategy == "rfe":
        sel = RFE(LogisticRegression(C=0.1, max_iter=2000, random_state=seed), n_features_to_select=15, step=2)
        return [scaler, ("sel", sel)], {"sel__n_features_to_select": randint(8, 28)}
    if strategy == "pca":
        return [scaler, ("pca", PCA(n_components=0.9, random_state=seed))], \
               {"pca__n_components": uniform(0.7, 0.28)}
    if strategy in ("signature_scores", "raw_plus_signature"):
        g = ferrdb_gene_groups(feature_names)
        return [("sig", SignatureScores(groups=g, append=strategy == "raw_plus_signature")), scaler], {}
    raise ValueError(f"unknown feature strategy: {strategy}")


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

class PlattSVC(ClassifierMixin, BaseEstimator):
    """SVC with Platt-scaled probabilities, built the way scikit-learn >= 1.9 recommends
    (CalibratedClassifierCV(SVC(), ensemble=False)) instead of the deprecated SVC(probability=True).
    Exposes C / gamma directly so the search spaces keep their names."""

    def __init__(self, C=1.0, kernel="rbf", gamma="scale", class_weight=None, random_state=None):
        self.C, self.kernel, self.gamma = C, kernel, gamma
        self.class_weight, self.random_state = class_weight, random_state

    def fit(self, X, y):
        base = SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, class_weight=self.class_weight)
        self.model_ = CalibratedClassifierCV(base, method="sigmoid", ensemble=False,
                                             cv=StratifiedKFold(5, shuffle=True, random_state=self.random_state))
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


class BalancedXGB(XGBClassifier):
    """XGBClassifier whose scale_pos_weight is computed from the TRAINING labels at fit time."""

    def __init__(self, balanced=False, **kw):
        super().__init__(**kw)
        self.balanced = balanced

    def fit(self, X, y, **kw):
        if self.balanced:
            y = np.asarray(y)
            self.set_params(scale_pos_weight=float((y == 0).sum() / max((y == 1).sum(), 1)))
        return super().fit(X, y, **kw)


def classifier_spec(name: str, seed: int, balanced: bool):
    """(estimator, search distributions, supports_sample_balance_natively)."""
    cw = "balanced" if balanced else None
    if name == "logreg_l2":
        return (LogisticRegression(l1_ratio=0.0, max_iter=5000, class_weight=cw, random_state=seed),
                {"clf__C": loguniform(1e-3, 1e2)})
    if name == "logreg_l1":
        return (LogisticRegression(l1_ratio=1.0, solver="saga", max_iter=8000, class_weight=cw, random_state=seed),
                {"clf__C": loguniform(1e-2, 1e2)})
    if name == "logreg_elasticnet":
        return (LogisticRegression(l1_ratio=0.5, solver="saga", max_iter=8000, class_weight=cw, random_state=seed),
                {"clf__C": loguniform(1e-2, 1e2), "clf__l1_ratio": uniform(0.1, 0.8)})
    if name == "svm_linear":
        return (PlattSVC(kernel="linear", class_weight=cw, random_state=seed),
                {"clf__C": loguniform(1e-3, 1e1)})
    if name == "svm_rbf":
        return (PlattSVC(kernel="rbf", class_weight=cw, random_state=seed),
                {"clf__C": loguniform(1e-1, 1e2), "clf__gamma": loguniform(1e-4, 1e-1)})
    tree_space = {"clf__n_estimators": randint(200, 601), "clf__max_depth": [3, 5, 8, 12, None],
                  "clf__min_samples_split": randint(2, 13), "clf__min_samples_leaf": randint(1, 9),
                  "clf__max_features": ["sqrt", "log2", 0.5]}
    if name == "random_forest":
        return (RandomForestClassifier(class_weight="balanced_subsample" if balanced else None,
                                       random_state=seed, n_jobs=1), tree_space)
    if name == "extra_trees":
        return (ExtraTreesClassifier(class_weight="balanced_subsample" if balanced else None,
                                     random_state=seed, n_jobs=1), tree_space)
    if name == "hist_gradient_boosting":
        return (HistGradientBoostingClassifier(class_weight=cw, early_stopping=False, random_state=seed),
                {"clf__learning_rate": loguniform(0.01, 0.2), "clf__max_iter": randint(50, 301),
                 "clf__max_leaf_nodes": randint(4, 25), "clf__l2_regularization": loguniform(1e-3, 10),
                 "clf__min_samples_leaf": randint(5, 31)})
    if name == "gradient_boosting":
        return (GradientBoostingClassifier(random_state=seed),
                {"clf__n_estimators": randint(50, 301), "clf__learning_rate": loguniform(0.01, 0.2),
                 "clf__max_depth": randint(2, 5), "clf__subsample": uniform(0.6, 0.4),
                 "clf__min_samples_leaf": randint(3, 21)})
    if name == "xgboost":
        return (BalancedXGB(balanced=balanced, eval_metric="logloss", random_state=seed, n_jobs=1, verbosity=0),
                {"clf__learning_rate": loguniform(0.01, 0.3), "clf__n_estimators": randint(50, 401),
                 "clf__max_depth": randint(2, 6), "clf__subsample": uniform(0.6, 0.4),
                 "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 8),
                 "clf__reg_lambda": loguniform(1e-2, 10), "clf__reg_alpha": loguniform(1e-3, 5)})
    if name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm not installed")
        return (LGBMClassifier(class_weight=cw, random_state=seed, n_jobs=1, verbose=-1),
                {"clf__learning_rate": loguniform(0.01, 0.2), "clf__n_estimators": randint(50, 401),
                 "clf__num_leaves": randint(4, 25), "clf__min_child_samples": randint(5, 31),
                 "clf__subsample": uniform(0.6, 0.4), "clf__subsample_freq": [1],
                 "clf__colsample_bytree": uniform(0.5, 0.5), "clf__reg_lambda": loguniform(1e-2, 10)})
    if name == "catboost":
        if not HAS_CATBOOST:
            raise ImportError("catboost not installed")
        return (CatBoostClassifier(auto_class_weights="Balanced" if balanced else None, random_seed=seed,
                                   verbose=0, thread_count=1, allow_writing_files=False),
                {"clf__iterations": randint(100, 401), "clf__depth": randint(3, 7),
                 "clf__learning_rate": loguniform(0.02, 0.2), "clf__l2_leaf_reg": loguniform(1, 20)})
    if name == "mlp":
        return (MLPClassifier(hidden_layer_sizes=(32,), early_stopping=True, n_iter_no_change=15,
                              max_iter=600, random_state=seed),
                {"clf__alpha": loguniform(1e-3, 1e0), "clf__hidden_layer_sizes": [(16,), (32,), (32, 16)],
                 "clf__learning_rate_init": loguniform(1e-3, 3e-2)})
    raise ValueError(f"unknown classifier: {name}")


def build_pipeline(learner: str, feature_strategy: str, feature_names: list[str], seed: int,
                   imbalance: str = "none"):
    """(pipeline, search space) for one classifier with one feature strategy and one imbalance strategy."""
    steps, fspace = feature_steps(feature_strategy, feature_names, seed)
    est, space = classifier_spec(learner, seed, balanced=imbalance == "class_weight")
    steps = list(steps)
    if imbalance == "smote":
        steps.append(("smote", SMOTE(k_neighbors=5, random_state=seed)))
    steps.append(("clf", est))
    return Pipeline(steps), {**fspace, **space}


class SearchTuneCalibrate(ClassifierMixin, BaseEstimator):
    """The COMPLETE model-building procedure as one estimator:

        1. RandomizedSearchCV over the pipeline's hyper-parameters (inner CV, training data only)
        2. optional probability calibration (Platt / isotonic, cross-fitted on the training data)
        3. decision-threshold tuning on cross-validated training predictions

    The nested-CV benchmark calls .fit on each outer-training fold and .predict_proba / .predict on the
    outer-test fold; the final model is the same .fit on the whole development set. The outer test rows
    therefore never influence any of the three steps.
    """

    def __init__(self, pipeline=None, space=None, n_iter=15, inner_folds=5, search_metric="roc_auc",
                 calibration="none", threshold_objective="mcc", seed=42):
        self.pipeline = pipeline
        self.space = space
        self.n_iter = n_iter
        self.inner_folds = inner_folds
        self.search_metric = search_metric
        self.calibration = calibration
        self.threshold_objective = threshold_objective
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        cv = StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed)
        if self.space:
            search = RandomizedSearchCV(self.pipeline, self.space, n_iter=self.n_iter, cv=cv,
                                        scoring=self.search_metric, random_state=self.seed, n_jobs=1,
                                        error_score=np.nan, refit=True)
            search.fit(X, y)
            best, self.best_params_, self.inner_cv_score_ = search.best_estimator_, search.best_params_, \
                float(search.best_score_)
        else:
            best = clone(self.pipeline).fit(X, y)
            self.best_params_, self.inner_cv_score_ = {}, float("nan")
        base = best
        if self.calibration in ("sigmoid", "isotonic"):
            base = CalibratedClassifierCV(clone(best), method=self.calibration, cv=3)
        if self.threshold_objective is None:  # plain 0.5 cut-off (used for ensemble members)
            self.model_ = clone(base).fit(X, y)
            self.proba_model_ = self.model_
            self.threshold_ = 0.5
        else:
            scorer = make_scorer(matthews_corrcoef) if self.threshold_objective == "mcc"                 else self.threshold_objective
            self.model_ = TunedThresholdClassifierCV(base, scoring=scorer, cv=3, refit=True, thresholds=101,
                                                     response_method="predict_proba", random_state=self.seed)
            self.model_.fit(X, y)
            self.proba_model_ = self.model_.estimator_
            self.threshold_ = float(self.model_.best_threshold_)
        self.best_pipeline_ = best
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        return self.proba_model_.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


class VotingProcedure(ClassifierMixin, BaseEstimator):
    """Soft voting over members that each run their OWN hyper-parameter search on the training data."""

    def __init__(self, members=None, calibration="none", threshold_objective="mcc", seed=42):
        self.members = members  # list[(name, SearchTuneCalibrate-with-no-calibration)]
        self.calibration = calibration
        self.threshold_objective = threshold_objective
        self.seed = seed

    def fit(self, X, y):
        self.fitted_ = [(n, clone(m).fit(X, y)) for n, m in self.members]
        self.classes_ = np.array([0, 1])
        # threshold from cross-validated ensemble predictions on the training data
        oof = np.zeros(len(y))
        y = np.asarray(y)
        for tr, va in StratifiedKFold(3, shuffle=True, random_state=self.seed).split(X, y):
            ps = [clone(m.best_pipeline_).fit(self._rows(X, tr), y[tr]).predict_proba(self._rows(X, va))[:, 1]
                  for _, m in self.fitted_]
            oof[va] = np.mean(ps, axis=0)
        self.threshold_ = _best_threshold(y, oof, self.threshold_objective)
        return self

    @staticmethod
    def _rows(X, idx):
        return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    def predict_proba(self, X):
        p = np.mean([m.predict_proba(X)[:, 1] for _, m in self.fitted_], axis=0)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold_).astype(int)


class StackingProcedure(ClassifierMixin, BaseEstimator):
    """Stacking: each base learner is tuned on the training data, then StackingClassifier trains the
    logistic-regression meta-model on CROSS-VALIDATED (out-of-fold) base predictions only."""

    def __init__(self, members=None, inner_folds=5, threshold_objective="mcc", seed=42):
        self.members = members
        self.inner_folds = inner_folds
        self.threshold_objective = threshold_objective
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        tuned = [(n, clone(m).fit(X, y).best_pipeline_) for n, m in self.members]
        stack = StackingClassifier(
            estimators=tuned, final_estimator=LogisticRegression(max_iter=2000, random_state=self.seed),
            cv=StratifiedKFold(self.inner_folds, shuffle=True, random_state=self.seed),
            stack_method="predict_proba", passthrough=False, n_jobs=1)
        scorer = make_scorer(matthews_corrcoef) if self.threshold_objective == "mcc" else self.threshold_objective
        self.model_ = TunedThresholdClassifierCV(stack, scoring=scorer, cv=3, refit=True, thresholds=101,
                                                 response_method="predict_proba", random_state=self.seed)
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1])
        self.threshold_ = float(self.model_.best_threshold_)
        return self

    def predict_proba(self, X):
        return self.model_.estimator_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def _best_threshold(y, p, objective):
    if objective != "mcc":
        raise ValueError("ensemble threshold tuning supports objective='mcc' only")
    grid = np.linspace(0.02, 0.98, 97)
    return float(grid[int(np.argmax([matthews_corrcoef(y, p >= t) for t in grid]))])


# ---------------------------------------------------------------------------
# Regressors
# ---------------------------------------------------------------------------

def regressor_spec(name: str, seed: int):
    tree = {"model__n_estimators": randint(150, 401), "model__max_depth": [4, 8, 12, None],
            "model__min_samples_leaf": randint(1, 21), "model__max_features": ["sqrt", 0.3, 0.6]}
    if name == "elastic_net":
        return (ElasticNet(max_iter=20000, random_state=seed),
                {"model__alpha": loguniform(1e-3, 1.0), "model__l1_ratio": uniform(0.05, 0.9)})
    if name == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=1), tree
    if name == "extra_trees":
        return ExtraTreesRegressor(random_state=seed, n_jobs=1), tree
    if name == "hist_gradient_boosting":
        return (HistGradientBoostingRegressor(early_stopping=False, random_state=seed),
                {"model__learning_rate": loguniform(0.02, 0.2), "model__max_iter": randint(80, 301),
                 "model__max_leaf_nodes": randint(6, 40), "model__l2_regularization": loguniform(1e-3, 10),
                 "model__min_samples_leaf": randint(10, 80)})
    if name == "gradient_boosting":
        return (GradientBoostingRegressor(random_state=seed),
                {"model__n_estimators": randint(80, 301), "model__learning_rate": loguniform(0.02, 0.2),
                 "model__max_depth": randint(2, 6), "model__subsample": uniform(0.6, 0.4)})
    if name == "xgboost":
        return (XGBRegressor(random_state=seed, n_jobs=1, verbosity=0),
                {"model__learning_rate": loguniform(0.02, 0.3), "model__n_estimators": randint(80, 401),
                 "model__max_depth": randint(3, 8), "model__subsample": uniform(0.6, 0.4),
                 "model__colsample_bytree": uniform(0.4, 0.6), "model__min_child_weight": randint(1, 10),
                 "model__reg_lambda": loguniform(1e-2, 10), "model__reg_alpha": loguniform(1e-3, 5)})
    if name == "lightgbm":
        if not HAS_LIGHTGBM:
            raise ImportError("lightgbm not installed")
        return (LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1),
                {"model__learning_rate": loguniform(0.02, 0.2), "model__n_estimators": randint(80, 401),
                 "model__num_leaves": randint(6, 40), "model__min_child_samples": randint(10, 80),
                 "model__subsample": uniform(0.6, 0.4), "model__subsample_freq": [1],
                 "model__colsample_bytree": uniform(0.4, 0.6)})
    if name == "catboost":
        if not HAS_CATBOOST:
            raise ImportError("catboost not installed")
        return (CatBoostRegressor(random_seed=seed, verbose=0, thread_count=1, allow_writing_files=False),
                {"model__iterations": randint(150, 501), "model__depth": randint(3, 8),
                 "model__learning_rate": loguniform(0.02, 0.2), "model__l2_leaf_reg": loguniform(1, 20)})
    raise ValueError(f"unknown regressor: {name}")


PLAIN_ENGLISH = {
    "logreg_l2": "Logistic regression (L2)", "logreg_l1": "Logistic regression (L1)",
    "logreg_elasticnet": "Logistic regression (elastic net)", "svm_linear": "Linear support-vector model",
    "svm_rbf": "Kernel support-vector model", "random_forest": "Random forest",
    "extra_trees": "Extra-trees ensemble", "hist_gradient_boosting": "Gradient-boosting model",
    "gradient_boosting": "Gradient-boosting model", "xgboost": "Gradient-boosting model (XGBoost)",
    "lightgbm": "Gradient-boosting model (LightGBM)", "catboost": "Gradient-boosting model (CatBoost)",
    "mlp": "Small neural network", "voting": "Voting ensemble", "stacking": "Stacked ensemble",
    "elastic_net": "Elastic-net regression",
}
