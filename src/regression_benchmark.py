"""
Drug-response (LN_IC50) regression benchmark on GDSC colorectal cell lines, same philosophy as
src/benchmark.py:

    cell lines -> LOCKED TEST cell lines (whole lines held out, written to disk)
              '-> development cell lines -> nested GROUP cross-validation (outer: GroupKFold by cell line,
                  inner: grouped RandomizedSearchCV) -> chosen model -> refit -> locked test once

Whole cell lines are always held out together (never rows), because rows of one line share an expression
profile and would otherwise leak identity across folds.

Two baselines make the result honest: GDSC drug effects dominate LN_IC50 (drug identity alone explains ~75%
of the variance), so every model is compared with "predict the drug's training mean". Whether expression adds
anything beyond drug identity is the scientifically relevant question and is reported explicitly.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import time
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import benchmark as bm
import model_registry as reg
import model_zoo as mz

warnings.filterwarnings("ignore")
OUT, CACHE, PROCESSED, ROOT = bm.OUT, bm.CACHE, bm.PROCESSED, bm.ROOT
ID = "COSMIC_ID"


def load_gdsc(target="LN_IC50"):
    df = pd.read_csv(PROCESSED / "gdsc_colorectal_dose_response_processed.csv")
    genes = json.load(open(PROCESSED / "gdsc_preprocessing_metadata.json"))["ferroptosis_genes_used"]
    return df, genes, df[target].to_numpy(float)


def reg_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"mae": float(mean_absolute_error(y, p)), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "r2": float(r2_score(y, p)), "pearson": float(pearsonr(y, p)[0]) if p.std() > 0 else np.nan,
            "spearman": float(spearmanr(y, p)[0]) if p.std() > 0 else np.nan}


def locked_cell_line_split(df, cfg):
    lines = np.sort(df[ID].unique())
    fp = hashlib.sha256(",".join(map(str, lines)).encode()).hexdigest()[:8]
    path = PROCESSED / f"locked_split_gdsc_{fp}.json"
    if path.exists():
        test_lines = json.load(open(path))["test_cell_lines"]
    else:
        gss = GroupShuffleSplit(1, test_size=cfg["regression"]["locked_test_fraction"],
                                random_state=cfg["random_seed"])
        _, te = next(gss.split(lines, groups=lines))
        test_lines = [int(x) for x in lines[te]]
        path.write_text(json.dumps({"seed": cfg["random_seed"], "created_at": datetime.now(timezone.utc).isoformat(),
                                    "test_cell_lines": test_lines, "test_evaluations": []}, indent=2))
    is_test = df[ID].isin(test_lines).to_numpy()
    return np.where(~is_test)[0], np.where(is_test)[0], path


def make_pre(genes, variant):
    drug = [("drug", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["DRUG_NAME"])]
    if variant == "drug_identity":          # genes + drug identity (known compounds)
        return ColumnTransformer([("genes", StandardScaler(), genes)] + drug)
    if variant == "pathway_conc":           # the legacy representation: pathway + concentration range
        return ColumnTransformer([("genes", StandardScaler(), genes),
                                  ("conc", StandardScaler(), ["MIN_CONC", "MAX_CONC"]),
                                  ("pathway", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                                   ["PATHWAY_NAME"])])
    if variant == "drug_only":              # no expression at all: isolates what expression adds
        return ColumnTransformer(drug)
    raise ValueError(variant)


class GroupedSearchRegressor(RegressorMixin, BaseEstimator):
    """Grouped hyper-parameter search + refit. X must carry the COSMIC_ID column; it is used ONLY to build
    inner group folds and is dropped before the model sees the data."""

    def __init__(self, pipeline=None, space=None, n_iter=8, inner_folds=3, seed=42):
        self.pipeline, self.space, self.n_iter, self.inner_folds, self.seed = pipeline, space, n_iter, inner_folds, seed

    def fit(self, X, y):
        g = X[ID].to_numpy()
        Xf = X.drop(columns=[ID])
        cv = list(GroupKFold(self.inner_folds).split(Xf, y, g))
        if self.space:
            s = RandomizedSearchCV(self.pipeline, self.space, n_iter=self.n_iter, cv=cv,
                                   scoring="neg_mean_absolute_error", random_state=self.seed, n_jobs=1,
                                   error_score=np.nan, refit=True)
            s.fit(Xf, y)
            self.model_, self.best_params_ = s.best_estimator_, s.best_params_
        else:
            self.model_, self.best_params_ = clone(self.pipeline).fit(Xf, y), {}
        return self

    def predict(self, X):
        return self.model_.predict(X.drop(columns=[ID]) if ID in X else X)


class DrugMeanBaseline(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        d = pd.Series(np.asarray(y), index=X["DRUG_NAME"].to_numpy())
        self.means_, self.global_ = d.groupby(level=0).mean(), float(np.mean(y))
        return self

    def predict(self, X):
        return X["DRUG_NAME"].map(self.means_).fillna(self.global_).to_numpy()


class VotingRegProcedure(RegressorMixin, BaseEstimator):
    def __init__(self, members=None):
        self.members = members

    def fit(self, X, y):
        self.fitted_ = [(n, clone(m).fit(X, y)) for n, m in self.members]
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for _, m in self.fitted_], axis=0)


def build(name, genes, variant, cfg, quick=False):
    r, seed = cfg["regression"], cfg["random_seed"]
    n_iter = 3 if quick else r["search_iterations"]

    def single(learner):
        model, space = mz.regressor_spec(learner, seed)
        return GroupedSearchRegressor(Pipeline([("pre", make_pre(genes, variant)), ("model", model)]), space,
                                      n_iter, r["inner_folds"], seed)
    if name == "drug_mean_baseline":
        return DrugMeanBaseline()
    if name == "voting":
        return VotingRegProcedure([(m, single(m)) for m in r["ensemble"]["members"]])
    if name.startswith("voting:"):   # voting:extra_trees,lightgbm,xgboost
        return VotingRegProcedure([(m, single(m)) for m in name.split(":", 1)[1].split(",")])
    return single(name)


def _fold(name, genes, variant, cfg, quick, X, y, tr, te):
    t = time.time()
    est = build(name, genes, variant, cfg, quick)
    est.fit(X.iloc[tr], y[tr])
    p = est.predict(X.iloc[te])
    return {"m": reg_metrics(y[te], p), "te": te, "p": p, "secs": time.time() - t,
            "params": getattr(est, "best_params_", {})}


def nested(name, genes, variant, cfg, X, y, quick=False, n_jobs=8):
    r = cfg["regression"]
    groups = X[ID].to_numpy()
    splits = list(GroupKFold(r["outer_folds"]).split(X, y, groups))
    key = hashlib.sha1(json.dumps([name, variant, quick, r, genes, len(y), cfg["random_seed"]],
                                  sort_keys=True).encode()).hexdigest()[:16]
    f = CACHE / f"reg_{key}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    folds = Parallel(n_jobs=min(n_jobs, len(splits)))(
        delayed(_fold)(name, genes, variant, cfg, quick, X, y, tr, te) for tr, te in splits)
    fm = pd.DataFrame([x["m"] for x in folds])
    res = {"name": name, "variant": variant, "fold_metrics": fm, "n_folds": len(folds),
           "secs_per_fold": float(np.mean([x["secs"] for x in folds])),
           "oof": pd.DataFrame({"idx": np.concatenate([x["te"] for x in folds]),
                                "pred": np.concatenate([x["p"] for x in folds])})}
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    return res


def expression_value_added(runs, champion) -> dict:
    """Paired fold-level comparison with the per-drug-mean predictor (the right yardstick: drug identity alone
    explains most of the variance, so a positive gain means expression carried extra information)."""
    gain = (runs["drug_mean_baseline"]["fold_metrics"]["mae"] - runs[champion]["fold_metrics"]["mae"]).to_numpy()
    return {"champion_vs_drug_mean_mae_gain_mean": float(gain.mean()),
            "champion_vs_drug_mean_mae_gain_std": float(gain.std(ddof=1)),
            "folds_where_expression_helped": int((gain > 0).sum()), "n_folds": int(len(gain))}


def summ(res):
    fm = res["fold_metrics"]
    row = {"name": res["name"], "variant": res["variant"], "n_folds": res["n_folds"],
           "secs_per_fold": round(res["secs_per_fold"], 1)}
    for m in ("mae", "rmse", "r2", "pearson", "spearman"):
        row[f"{m}_mean"], row[f"{m}_std"] = float(fm[m].mean()), float(fm[m].std(ddof=1))
    return row


def cell_line_bootstrap(df_test, y, p, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    lines = df_test[ID].to_numpy()
    uniq = np.unique(lines)
    idx_by = {l: np.where(lines == l)[0] for l in uniq}
    draws = {k: [] for k in ("mae", "rmse", "r2", "pearson", "spearman")}
    for _ in range(n):
        pick = np.concatenate([idx_by[l] for l in rng.choice(uniq, len(uniq))])
        m = reg_metrics(y[pick], p[pick])
        for k in draws:
            draws[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in draws.items()}


def run_regression_benchmark(quick=False, progress=None, promote=False) -> dict:
    t0 = time.time()
    cfg = bm.load_cfg()
    r = cfg["regression"]
    primary = cfg["primary_metric"]["regression"]
    say = lambda m: bm._say(progress, m)  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    df, genes, y = load_gdsc(r["target"])
    dev, test, lock = locked_cell_line_split(df, cfg)
    cols = [ID, "DRUG_NAME", "PATHWAY_NAME", "MIN_CONC", "MAX_CONC"] + genes
    X = df[cols]
    Xdev, ydev, Xtest, ytest = X.iloc[dev].reset_index(drop=True), y[dev], X.iloc[test].reset_index(drop=True), y[test]
    say(f"GDSC: {len(df)} (cell line, drug) rows, {df[ID].nunique()} cell lines "
        f"({Xdev[ID].nunique()} dev / {Xtest[ID].nunique()} locked-test lines), {df['DRUG_NAME'].nunique()} drugs")

    learners = [n for n in r["learners"] if _available(n)]
    skipped = [n for n in r["learners"] if n not in learners]
    if quick:
        learners = ["elastic_net", "hist_gradient_boosting"]

    runs = {}
    say("[baseline] per-drug mean")
    runs["drug_mean_baseline"] = nested("drug_mean_baseline", genes, "drug_only", cfg, Xdev, ydev, quick)
    for n in learners:
        say(f"[zoo] {n}")
        runs[n] = nested(n, genes, "drug_identity", cfg, Xdev, ydev, quick)
    if not quick:
        say("[zoo] voting ensemble")
        runs["voting"] = nested("voting", genes, "drug_identity", cfg, Xdev, ydev, quick)
    say("[representation] legacy pathway+concentration features vs drug identity (reference: elastic net)")
    runs["elastic_net__pathway_conc"] = nested("elastic_net", genes, "pathway_conc", cfg, Xdev, ydev, quick)
    runs["elastic_net__pathway_conc"]["name"] = "elastic_net__pathway_conc"

    rank = {**{k: v["rank"] for k, v in r["learners"].items()}, "voting": r["ensemble"]["rank"]}
    cand_names = [n for n in runs if n in rank]
    table = pd.DataFrame([{**summ(runs[n]), "rank": rank[n]} for n in cand_names])
    all_table = pd.DataFrame([summ(v) for v in runs.values()])
    champion, why = bm.select_best(table, primary, {**cfg, "selection": {**cfg["selection"], "max_cv_std": 9.9}})
    base = all_table.set_index("name").loc["drug_mean_baseline"]
    all_table["mae_gain_vs_drug_mean"] = (base["mae_mean"] - all_table["mae_mean"]) / base["mae_mean"]
    all_table["selected"] = all_table["name"] == champion
    all_table.to_csv(OUT / "regression_cv.csv", index=False)
    say(f"Regression champion -> {champion} ({why})")

    expr_added = expression_value_added(runs, champion)
    summary = {"quick": quick, "champion": champion, "selection": why, "skipped_learners": skipped,
               "cv": summ(runs[champion]), "baseline_drug_mean_cv": summ(runs["drug_mean_baseline"]),
               "expression_value_added": expr_added, "n_cell_lines_total": int(df[ID].nunique()),
               "caution": f"Only {df[ID].nunique()} colorectal cell lines: effective sample size for expression-driven "
                          "effects is tiny; the locked test set holds only a handful of lines."}
    if quick:
        (OUT / "regression_summary_quick.json").write_text(json.dumps(summary, indent=2, default=float))
        return summary

    say("[final] refit on development cell lines; locked test cell lines scored once")
    final = build(champion, genes, "drug_identity", cfg).fit(Xdev, ydev)
    p = final.predict(Xtest)
    base_est = DrugMeanBaseline().fit(Xdev, ydev)
    pb = base_est.predict(Xtest)
    tm, bmx = reg_metrics(ytest, p), reg_metrics(ytest, pb)
    lock_spec = json.load(open(lock))
    lock_spec["test_evaluations"].append({"at": datetime.now(timezone.utc).isoformat(), "tag": f"regression_{champion}"})
    lock.write_text(json.dumps(lock_spec, indent=2))
    ci = cell_line_bootstrap(Xtest, ytest, p)
    pd.DataFrame({"COSMIC_ID": Xtest[ID], "DRUG_NAME": Xtest["DRUG_NAME"], "y_true": ytest, "pred": p,
                  "baseline_pred": pb}).to_csv(OUT / "regression_locked_test_predictions.csv", index=False)
    oof = runs[champion]["oof"].sort_values("idx")
    pd.DataFrame({"COSMIC_ID": Xdev[ID].to_numpy()[oof["idx"]], "DRUG_NAME": Xdev["DRUG_NAME"].to_numpy()[oof["idx"]],
                  "y_true": ydev[oof["idx"]], "pred": oof["pred"].to_numpy()}).to_csv(
        OUT / "regression_cv_oof_predictions.csv", index=False)

    test_metrics = {**tm, "ci95": ci, "n_rows": int(len(ytest)), "n_cell_lines": int(Xtest[ID].nunique()),
                    "baseline_drug_mean": bmx, "mae_gain_vs_drug_mean": float((bmx["mae"] - tm["mae"]) / bmx["mae"])}
    summary.update({"locked_test_metrics": test_metrics, "created_at": datetime.now(timezone.utc).isoformat(),
                    "hyperparameters": getattr(final, "best_params_", {}),
                    "runtime_min": (time.time() - t0) / 60})

    drugs = (df.groupby("DRUG_NAME").agg(PATHWAY_NAME=("PATHWAY_NAME", lambda s: s.mode().iloc[0] if s.notna().any() else "unknown"),
                                          PUTATIVE_TARGET=("PUTATIVE_TARGET", lambda s: str(s.mode().iloc[0]) if s.notna().any() else "unknown"),
                                          MIN_CONC=("MIN_CONC", "median"), MAX_CONC=("MAX_CONC", "median"),
                                          n_cell_lines=(ID, "nunique")).reset_index())
    bundle = {"model": final, "genes": genes, "drugs": drugs, "target": r["target"],
              "background": Xdev[genes].sample(100, random_state=cfg["random_seed"]).reset_index(drop=True)}
    cv_metrics = {k: v for k, v in summ(runs[champion]).items() if k != "name"}
    entry = reg.register_benchmark_model(
        task="dose_response", name=champion, display_name=mz.PLAIN_ENGLISH.get(champion, champion),
        model_object=bundle, feature_columns=genes + ["DRUG_NAME"], feature_pipeline="expression+drug_identity",
        dataset="gdsc_colorectal_dose_response_processed",
        hyperparameters={k: (v.item() if hasattr(v, "item") else v) for k, v in getattr(final, "best_params_", {}).items()},
        cv_metrics=cv_metrics, test_metrics=test_metrics, external_metrics=None, calibration="n/a",
        seed=cfg["random_seed"], extra={"target_column": r["target"]})
    summary["registry_entry"] = {"name": entry["name"], "version": entry["version"], "path": entry["path"]}
    if promote:
        inc = reg.current_best("dose_response")
        ok, reasons = evaluate_regression_promotion(entry, inc, cfg, explicit_action=promote)
        summary["promotion"] = {"promoted": ok, "reasons": reasons}
        if ok:
            reg.promote_champion("dose_response", entry, "; ".join(reasons))
    (OUT / "regression_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    say(f"Done in {summary['runtime_min']:.1f} min.")
    return summary


def _available(name):
    try:
        mz.regressor_spec(name, 0)
        return True
    except ImportError:
        return False


def evaluate_regression_promotion(entry: dict, incumbent: dict | None, cfg: dict, *, explicit_action: bool,
                                  root=None) -> tuple[bool, list[str]]:
    """Drug-response promotion gate. ALL must hold: explicit researcher action; metadata, metrics and the model artifact
    present; same feature set / preprocessing / response target as the production model; the model beats the per-drug-mean
    baseline on the locked test; then the score criteria (CV improvement, locked-test not worse, stability)."""
    import promotion as pr
    if not explicit_action:
        return False, ["promotion needs an explicit researcher action"]
    kw = {"root": root} if root is not None else {}
    bad = pr.structural_failures(entry, incumbent, required_cv=("mae_mean", "mae_std"),
                                 required_test=("mae",), **kw)
    base = (entry.get("test_metrics") or {}).get("baseline_drug_mean", {}).get("mae")
    if not pr._finite(base):
        bad.append("per-drug-mean baseline MAE missing: cannot show the model beats the baseline")
    elif entry["test_metrics"]["mae"] > base and "required validation metrics" not in " ".join(bad):
        bad.append(f"locked-test MAE {entry['test_metrics']['mae']:.3f} is worse than the per-drug-mean baseline {base:.3f}")
    if bad:
        return False, bad
    return _promotion(entry["cv_metrics"], entry["test_metrics"], incumbent, cfg)


def _promotion(cv_new, test_new, inc, cfg):
    p = cfg["selection"]["promotion"]
    if inc is None or "cv_metrics" not in inc:
        return True, ["no comparable incumbent: first validated model is promoted"]
    ok, why = True, []
    gain = inc["cv_metrics"]["mae_mean"] - cv_new["mae_mean"]
    if gain < p["min_cv_improvement"]:
        ok = False
        why.append(f"CV MAE improvement {gain:.3f} < {p['min_cv_improvement']}")
    if test_new["mae"] > inc["test_metrics"]["mae"] + p["max_test_drop"]:
        ok = False
        why.append("locked-test MAE worse than incumbent")
    if cv_new["mae_std"] > inc["cv_metrics"]["mae_std"] * p["max_std_ratio"] and cv_new["mae_std"] > 0.05:
        ok = False
        why.append("less stable across folds")
    return ok, why or ["all promotion criteria satisfied"]
