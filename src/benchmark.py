"""
Model-selection engine for the MSI classification task (ferroptosis-gene features).

PROTOCOL (every number it reports comes from this file's code path)
------------------------------------------------------------------
    all labelled TCGA samples
        |-- LOCKED TEST SET (stratified, created once, written to disk, never used for any choice)
        '-- DEVELOPMENT SET
              |-- NESTED CROSS-VALIDATION
              |       outer folds : unbiased estimate of the whole selection procedure
              |       inner folds : RandomizedSearchCV, calibration, decision-threshold tuning
              |-- feature strategy / imbalance strategy / learner / ensemble / calibration choice
              |       (made from outer-CV scores by the rule in configs/model_selection.yaml)
              '-- FINAL MODEL = chosen procedure refit on the whole development set
                      |-- locked test set (once)
                      '-- external cohort (CPTAC-2; refit on the shared genes, never tuned on it)

Anything that learns from data (scaling, feature selection, PCA, SMOTE, signature z-scoring,
calibration, threshold) is a step of the procedure being fit, so it only ever sees training rows.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import Parallel, delayed
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, matthews_corrcoef, precision_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold, train_test_split

import model_zoo as mz

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "results" / "benchmark"
CACHE = OUT / "cache"
FIG = ROOT / "figures"
CFG_PATH = ROOT / "configs" / "model_selection.yaml"

HIGHER_BETTER = {"roc_auc": True, "pr_auc": True, "f1": True, "balanced_accuracy": True, "mcc": True,
                 "accuracy": True, "brier": False, "ece": False, "mae": False, "rmse": False, "r2": True,
                 "pearson": True, "spearman": True}
SEARCH_SCORING = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1",
                  "balanced_accuracy": "balanced_accuracy", "accuracy": "accuracy"}


def load_cfg() -> dict:
    with open(CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Data + locked split
# ---------------------------------------------------------------------------

def load_msi_data(path: Path | None = None):
    """Labelled TCGA samples (MANTIS-scored only), gene list, 0/1 target (1 = MSI-H)."""
    df = pd.read_csv(path or PROCESSED / "tcga_coadread_processed.csv")
    genes = json.load(open(PROCESSED / "tcga_preprocessing_metadata.json"))["ferroptosis_genes_used"]
    df = df[df["msi_label"].notna()].reset_index(drop=True)
    y = (df["msi_label"] == "MSI-H").astype(int).to_numpy()
    return df, genes, y


def _fingerprint(ids, y) -> str:
    h = hashlib.sha256()
    for i, v in sorted(zip(map(str, ids), map(int, y))):
        h.update(f"{i}:{v};".encode())
    return h.hexdigest()


def locked_split(df: pd.DataFrame, y: np.ndarray, cfg: dict, seed: int | None = None):
    """Creates (once) or loads the locked train/test split. The file is the single source of truth:
    if it exists it is reused verbatim, so the test rows can never drift between runs. If the underlying
    data changed (different fingerprint) it refuses rather than silently re-splitting."""
    c = cfg["classification"]
    seed = cfg["random_seed"] if seed is None else seed
    fp = _fingerprint(df["sampleId"], y)
    path = PROCESSED / f"locked_split_{fp[:8]}.json"
    if path.exists():
        spec = json.load(open(path))
        ids = df["sampleId"].tolist()
        pos = {s: i for i, s in enumerate(ids)}
        test_idx = np.array(sorted(pos[s] for s in spec["test_ids"]))
        dev_idx = np.array(sorted(set(range(len(df))) - set(test_idx)))
        return dev_idx, test_idx, spec, path
    dev_idx, test_idx = train_test_split(np.arange(len(df)), test_size=c["locked_test_fraction"],
                                         stratify=y, random_state=seed)
    dev_idx, test_idx = np.sort(dev_idx), np.sort(test_idx)
    spec = {"fingerprint": fp, "seed": seed, "created_at": datetime.now(timezone.utc).isoformat(),
            "n_dev": int(len(dev_idx)), "n_test": int(len(test_idx)),
            "test_positives": int(y[test_idx].sum()), "dev_positives": int(y[dev_idx].sum()),
            "test_ids": df["sampleId"].iloc[test_idx].tolist(), "test_evaluations": []}
    path.write_text(json.dumps(spec, indent=2))
    return dev_idx, test_idx, spec, path


def log_test_evaluation(path: Path, tag: str, note: str = "") -> None:
    spec = json.load(open(path))
    spec["test_evaluations"].append({"at": datetime.now(timezone.utc).isoformat(), "tag": tag, "note": note})
    path.write_text(json.dumps(spec, indent=2))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def expected_calibration_error(y, p, bins: int = 5) -> float:
    y, p = np.asarray(y), np.asarray(p)
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -1e-9, 1 + 1e-9
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi)
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def clf_metrics(y, proba, pred) -> dict:
    y, proba, pred = np.asarray(y), np.clip(np.asarray(proba, dtype=float), 0.0, 1.0), np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": tp / (tp + fn) if tp + fn else np.nan,          # = sensitivity
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, proba) if both else np.nan,
        "pr_auc": average_precision_score(y, proba) if both else np.nan,
        "brier": brier_score_loss(y, proba),
        "ece": expected_calibration_error(y, proba),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def bootstrap_ci(y, proba, pred, n: int = 2000, seed: int = 0, keys=("accuracy", "roc_auc", "pr_auc", "f1",
                                                                  "balanced_accuracy", "mcc", "recall", "specificity")):
    """Stratified percentile bootstrap 95% CI of the held-out predictions."""
    rng = np.random.default_rng(seed)
    y, proba, pred = map(np.asarray, (y, proba, pred))
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    draws = {k: [] for k in keys}
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = clf_metrics(y[idx], proba[idx], pred[idx])
        for k in keys:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


# ---------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------

def build_procedure(spec: dict, feature_names: list[str], cfg: dict, tune_threshold: bool = True):
    """spec = {kind: single|voting|stacking, learner?, features, imbalance, calibration}"""
    c, seed = cfg["classification"], cfg["random_seed"]
    primary = cfg["primary_metric"]["classification"]
    search_metric = SEARCH_SCORING.get(primary, "roc_auc")
    obj = c["threshold_objective"] if tune_threshold else None

    def single(learner, calibration="none", objective=obj):
        pipe, space = mz.build_pipeline(learner, spec["features"], feature_names, seed, spec["imbalance"])
        return mz.SearchTuneCalibrate(pipe, space, n_iter=c["search_iterations"], inner_folds=c["inner_folds"],
                                      search_metric=search_metric, calibration=calibration,
                                      threshold_objective=objective, seed=seed)

    kind = spec["kind"]
    if kind == "single":
        return single(spec["learner"], spec.get("calibration", "none"))
    members_cfg = c["ensembles"][kind]["members"]
    members = [(m, single(m, "none", None)) for m in members_cfg]
    if kind == "voting":
        return mz.VotingProcedure(members, threshold_objective=obj or "mcc", seed=seed)
    return mz.StackingProcedure(members, inner_folds=c["inner_folds"], threshold_objective=obj or "mcc", seed=seed)


def spec_name(spec: dict) -> str:
    return spec["learner"] if spec["kind"] == "single" else spec["kind"]


def _outer_fold(spec, X, y, tr, te, feature_names, cfg):
    t0 = time.time()
    est = build_procedure(spec, feature_names, cfg)
    est.fit(X.iloc[tr], y[tr])
    proba = est.predict_proba(X.iloc[te])[:, 1]
    pred = est.predict(X.iloc[te])
    return {"metrics": clf_metrics(y[te], proba, pred), "te": te, "proba": proba, "pred": pred,
            "params": getattr(est, "best_params_", {}), "threshold": float(getattr(est, "threshold_", 0.5)),
            "secs": time.time() - t0}


def nested_cv(spec: dict, X: pd.DataFrame, y, cfg: dict, repeats: int, groups=None, n_jobs: int = 8,
              cache_tag: str = "", progress=None) -> dict:
    """Outer CV of the complete procedure (search + calibration + threshold all fit inside each outer-train)."""
    c, seed = cfg["classification"], cfg["random_seed"]
    feature_names = list(X.columns)
    if groups is None:
        splits = list(RepeatedStratifiedKFold(n_splits=c["outer_folds"], n_repeats=repeats,
                                              random_state=seed).split(X, y))
    else:
        splits = []
        for r in range(repeats):
            splits += list(StratifiedGroupKFold(c["outer_folds"], shuffle=True, random_state=seed + r)
                           .split(X, y, groups))
    key = hashlib.sha1(json.dumps([spec, cache_tag, repeats, cfg["classification"], cfg["primary_metric"],
                                   feature_names, len(y), int(np.sum(y)), cfg["random_seed"],
                                   groups is not None], sort_keys=True, default=str).encode()).hexdigest()[:16]
    cache_file = CACHE / f"{key}.pkl"
    if cache_file.exists():
        return pickle.load(open(cache_file, "rb"))
    folds = Parallel(n_jobs=min(n_jobs, len(splits)))(
        delayed(_outer_fold)(spec, X, y, tr, te, feature_names, cfg) for tr, te in splits)
    metrics = pd.DataFrame([f["metrics"] for f in folds])
    n_per_repeat = len(splits) // repeats
    oof_rows = []
    for i, f in enumerate(folds):
        for idx, p, d in zip(f["te"], f["proba"], f["pred"]):
            oof_rows.append((i // n_per_repeat, int(idx), float(p), int(d)))
    res = {"spec": spec, "name": spec_name(spec), "fold_metrics": metrics, "n_folds": len(folds),
           "oof": pd.DataFrame(oof_rows, columns=["repeat", "idx", "proba", "pred"]),
           "params": [f["params"] for f in folds], "thresholds": [f["threshold"] for f in folds],
           "secs_per_fold": float(np.mean([f["secs"] for f in folds]))}
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(cache_file, "wb"))
    return res


def summarize(res: dict, metrics=("roc_auc", "pr_auc", "f1", "balanced_accuracy", "mcc", "accuracy",
                                  "recall", "specificity", "brier", "ece")) -> dict:
    fm = res["fold_metrics"]
    row = {"name": res["name"], "n_folds": res["n_folds"], "secs_per_fold": round(res["secs_per_fold"], 1)}
    for m in metrics:
        row[f"{m}_mean"], row[f"{m}_std"] = float(fm[m].mean()), float(fm[m].std(ddof=1))
    return row


# ---------------------------------------------------------------------------
# Selection rule
# ---------------------------------------------------------------------------

def select_best(table: pd.DataFrame, primary: str, cfg: dict, rank_col: str = "rank") -> tuple[str, str]:
    """Highest primary score wins unless the one-standard-error rule lets a SIMPLER candidate (lower
    complexity rank) through; candidates whose fold-to-fold std exceeds the stability gate are excluded
    when at least one candidate passes it. Returns (winner, human-readable justification)."""
    sel = cfg["selection"]
    hb = HIGHER_BETTER[primary]
    mean, std = table[f"{primary}_mean"], table[f"{primary}_std"]
    score = mean if hb else -mean
    stable = std <= sel["max_cv_std"]
    pool = table[stable] if stable.any() else table
    s = score.loc[pool.index]
    best_i = s.idxmax()
    se = std.loc[best_i] / np.sqrt(table.loc[best_i, "n_folds"])
    if sel["one_se_rule"]:
        eligible = pool[s >= s.loc[best_i] - se]
        win_i = eligible.sort_values([rank_col, f"{primary}_mean"], ascending=[True, not hb]).index[0]
    else:
        win_i = best_i
    why = (f"top {primary} = {table.loc[best_i, 'name']} ({mean.loc[best_i]:.3f}); "
           f"winner {table.loc[win_i, 'name']} ({mean.loc[win_i]:.3f} +/- {std.loc[win_i]:.3f})"
           + (f"; chosen as the simplest candidate within 1 SE ({se:.3f}) of the best" if win_i != best_i else "")
           + ("" if stable.all() else f"; {int((~stable).sum())} candidate(s) excluded by the stability gate "
                                       f"(std > {sel['max_cv_std']})"))
    return table.loc[win_i, "name"], why


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _say(progress, msg, frac=None):
    print(msg, flush=True)
    if progress:
        progress(msg, frac)


def run_feature_strategy_stage(X, y, cfg, quick, progress):
    c = cfg["classification"]
    primary = cfg["primary_metric"]["classification"]
    rows, per_strategy_folds = [], {}
    strategies = c["feature_strategies"][:3] if quick else c["feature_strategies"]
    for s in strategies:
        fold_vals = []
        for learner in c["reference_learners"]:
            _say(progress, f"[features] {s} x {learner}")
            r = nested_cv({"kind": "single", "learner": learner, "features": s, "imbalance": "class_weight"},
                          X, y, cfg, repeats=1, cache_tag="stage1")
            sm = summarize(r)
            rows.append({"feature_strategy": s, "learner": learner, **sm})
            fold_vals.append(r["fold_metrics"][primary].to_numpy())
        per_strategy_folds[s] = np.mean(fold_vals, axis=0)
    detail = pd.DataFrame(rows)
    agg = pd.DataFrame([{"name": s, "rank": strategies.index(s), "n_folds": len(v),
                         f"{primary}_mean": float(v.mean()), f"{primary}_std": float(v.std(ddof=1))}
                        for s, v in per_strategy_folds.items()])
    return detail, agg


def run_imbalance_stage(X, y, cfg, feature_strategy, quick, progress):
    c = cfg["classification"]
    primary = cfg["primary_metric"]["classification"]
    rows, folds = [], {}
    strategies = c["imbalance_strategies"]
    for s in strategies:
        fv = []
        for learner in c["reference_learners"]:
            _say(progress, f"[imbalance] {s} x {learner}")
            r = nested_cv({"kind": "single", "learner": learner, "features": feature_strategy, "imbalance": s},
                          X, y, cfg, repeats=1, cache_tag="stage2")
            rows.append({"imbalance": s, "learner": learner, **summarize(r)})
            fv.append(r["fold_metrics"][primary].to_numpy())
        folds[s] = np.mean(fv, axis=0)
    detail = pd.DataFrame(rows)
    agg = pd.DataFrame([{"name": s, "rank": strategies.index(s), "n_folds": len(v),
                         f"{primary}_mean": float(v.mean()), f"{primary}_std": float(v.std(ddof=1))}
                        for s, v in folds.items()])
    return detail, agg


def available_learners(cfg, quick):
    out, skipped = [], []
    for name, meta in cfg["classification"]["learners"].items():
        if not meta.get("enabled", True):
            continue
        try:
            mz.classifier_spec(name, 0, False)
        except ImportError as e:
            skipped.append({"name": name, "reason": str(e)})
            continue
        out.append(name)
    if quick:
        out = [n for n in out if n in ("logreg_l2", "svm_rbf", "random_forest")]
    return out, skipped


def run_zoo_stage(X, y, cfg, feature_strategy, imbalance, repeats, quick, progress):
    c = cfg["classification"]
    learners, skipped = available_learners(cfg, quick)
    results = {}
    for i, l in enumerate(learners):
        _say(progress, f"[zoo] {l}", i / max(len(learners), 1))
        results[l] = nested_cv({"kind": "single", "learner": l, "features": feature_strategy,
                                "imbalance": imbalance}, X, y, cfg, repeats=repeats, cache_tag="zoo")
    if not quick:
        for kind in ("voting", "stacking"):
            _say(progress, f"[zoo] {kind} ensemble")
            results[kind] = nested_cv({"kind": kind, "features": feature_strategy, "imbalance": imbalance},
                                      X, y, cfg, repeats=repeats, cache_tag="zoo")
    rank = {**{k: v["rank"] for k, v in c["learners"].items()}, **{k: v["rank"] for k, v in c["ensembles"].items()}}
    table = pd.DataFrame([{**summarize(r), "rank": rank[n]} for n, r in results.items()])
    return results, table, skipped


def run_calibration_stage(X, y, cfg, base_spec, repeats, progress):
    rows, folds = [], {}
    methods = cfg["classification"]["calibration_methods"]
    primary = "brier"
    for m in methods:
        _say(progress, f"[calibration] {m}")
        r = nested_cv({**base_spec, "calibration": m}, X, y, cfg, repeats=repeats, cache_tag="calib")
        rows.append({**summarize(r), "name": m, "rank": methods.index(m)})
        folds[m] = r
    return folds, pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Final fit, locked test, external cohorts, leakage canaries
# ---------------------------------------------------------------------------

def fit_final(spec, X_dev, y_dev, cfg):
    est = build_procedure(spec, list(X_dev.columns), cfg)
    est.fit(X_dev, y_dev)
    return est


def permutation_canary(spec, X, y, cfg, runs):
    """Nested CV on shuffled labels: any honest pipeline must score ~0.5 AUC. A value far above 0.5 means
    information is leaking from the held-out fold into the fit."""
    aucs = []
    for r in range(runs):
        yp = np.random.default_rng(1000 + r).permutation(y)
        res = nested_cv(spec, X, yp, cfg, repeats=1, cache_tag=f"perm{r}")
        aucs.append(float(res["fold_metrics"]["roc_auc"].mean()))
    return aucs


def load_cptac():
    df = pd.read_csv(PROCESSED / "external_cptac_coad_2019_processed.csv")
    y = (df["msi_label"] == "MSI-H").astype(int).to_numpy()
    return df, y
