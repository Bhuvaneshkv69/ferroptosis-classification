"""
Ensemble lab: does a properly designed multi-model ensemble beat the best single model, and does an extended
feature set lift accuracy past 90% on genuinely unseen data?

For each feature set (35-gene ferroptosis core, +mismatch repair, Extended molecular model, whole transcriptome):
  1. nested-CV benchmark of the base learners (same folds for every model)
  2. complementarity analysis from the shared out-of-fold predictions
  3. nested-CV evaluation of ensembles (soft equal / soft learned-weight / hard / stacking)
  4. CV-ONLY promotion gate chooses at most ONE ensemble candidate
  5. only then: locked test scored once; the chosen candidate must also pass the test / leakage / reproducibility
     checks. If it fails, the best single model stays -- the next-best ensemble is NOT tried (no test snooping).
Nothing in steps 1-4 touches the locked test set.
"""
from __future__ import annotations

import json
import pickle
import time
import hashlib
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold

import benchmark as bm
import ensemble_zoo as ez
import extended_data as ed
import model_registry as reg
import model_zoo as mz

ROOT = bm.ROOT
LAB = ROOT / "results" / "ensemble_lab"
METHOD_RANK = {"soft_equal": 20, "hard": 21, "soft_weighted": 22, "stacking": 23}
TASK_OF = {"core": "classification", "core_plus_mmr": "classification_extended",
           "extended_molecular": "classification_extended", "whole_transcriptome": "classification_wt"}
LABEL_OF = {"core": "Ferroptosis Core Classifier (35 genes)", "core_plus_mmr": "35-gene + MMR Extended Model (39 genes)",
            "extended_molecular": "Extended Molecular Model (53 features)",
            "whole_transcriptome": "Whole-transcriptome model (experimental)"}


# ---------------------------------------------------------------------------------------------- data
def feature_matrices(fs: str, df: pd.DataFrame, dev, test, cfg):
    X_all = ed.load_matrix(fs, df["sampleId"])
    if fs == "whole_transcriptome":
        kept = ed.dev_gene_filter(X_all.iloc[dev], ed.feature_config()["whole_transcriptome"]["max_missing_fraction_dev"])
        X_all = X_all[kept]
    return X_all.iloc[dev].reset_index(drop=True), X_all.iloc[test].reset_index(drop=True)


def learners_for(fs: str, cfg: dict) -> list[str]:
    el = cfg["ensemble_lab"]
    names = list(el["learners_whole_transcriptome"] if fs == "whole_transcriptome" else el["learners_default"])
    if fs == "core":
        names += el["learners_core_extra"]
    ok = []
    for n in names:
        try:
            mz.classifier_spec(n, 0, False)
            ok.append(n)
        except ImportError:
            pass
    return ok


# ------------------------------------------------------------------------------------ complementarity
def complementarity(results: dict, y) -> dict:
    """Pairwise diversity of base learners from their out-of-fold predictions on the SAME folds."""
    names = list(results)
    reps = sorted(results[names[0]]["oof"]["repeat"].unique())
    n = len(names)
    corr = np.zeros((n, n)); dis = np.zeros((n, n)); jac = np.zeros((n, n)); q = np.zeros((n, n))
    oracle = []
    for r in reps:
        P, D, E = [], [], []
        for nm in names:
            o = results[nm]["oof"]
            o = o[o["repeat"] == r].sort_values("idx")
            P.append(o["proba"].to_numpy()); D.append(o["pred"].to_numpy()); E.append((o["pred"].to_numpy() != y[o["idx"].to_numpy()]))
        P, D, E = np.array(P), np.array(D), np.array(E)
        for i in range(n):
            for j in range(n):
                corr[i, j] += spearmanr(P[i], P[j])[0] / len(reps)
                dis[i, j] += (D[i] != D[j]).mean() / len(reps)
                either = (E[i] | E[j]).sum()
                jac[i, j] += ((E[i] & E[j]).sum() / either if either else 1.0) / len(reps)
                n11, n00 = (~E[i] & ~E[j]).sum(), (E[i] & E[j]).sum()
                n10, n01 = (~E[i] & E[j]).sum(), (E[i] & ~E[j]).sum()
                den = n11 * n00 + n01 * n10
                q[i, j] += ((n11 * n00 - n01 * n10) / den if den else 0.0) / len(reps)
        oracle.append(float((~E.all(axis=0)).mean()))
    mk = lambda M: pd.DataFrame(M, index=names, columns=names)  # noqa: E731
    off = ~np.eye(n, dtype=bool)
    summary = pd.DataFrame({"model": names,
                            "mean_proba_corr": [corr[i][off[i]].mean() for i in range(n)],
                            "mean_disagreement": [dis[i][off[i]].mean() for i in range(n)],
                            "mean_error_overlap": [jac[i][off[i]].mean() for i in range(n)]})
    best_acc = max(1 - results[nm]["oof"].assign(
        err=lambda d: (d["pred"].to_numpy() != y[d["idx"].to_numpy()])).groupby("repeat")["err"].mean().mean()
        for nm in names)
    return {"proba_corr": mk(corr), "disagreement": mk(dis), "error_overlap": mk(jac), "q_statistic": mk(q),
            "summary": summary, "oracle_accuracy": float(np.mean(oracle)), "best_single_accuracy_oof": float(best_acc)}


def auto_members(table: pd.DataFrame, comp: dict, cfg: dict) -> tuple[list[str], list[dict]]:
    """Complementarity rule (dev data only): walk learners by CV AUC; skip near-duplicates (probability correlation
    above max_proba_corr with an already chosen member) and weak learners (AUC more than auc_margin below the best)."""
    a = cfg["ensemble_lab"]["auto_members"]
    t = table.sort_values("roc_auc_mean", ascending=False)
    best = t["roc_auc_mean"].iloc[0]
    chosen, log = [], []
    for nm, auc in zip(t["name"], t["roc_auc_mean"]):
        if len(chosen) >= a["max_members"]:
            log.append({"model": nm, "decision": "skipped", "why": "member cap reached"})
            continue
        if auc < best - a["auc_margin"]:
            log.append({"model": nm, "decision": "excluded", "why": f"AUC {auc:.3f} more than {a['auc_margin']} below best"})
            continue
        dup = [c for c in chosen if comp["proba_corr"].loc[nm, c] > a["max_proba_corr"]]
        if dup:
            log.append({"model": nm, "decision": "excluded", "why": f"near-duplicate of {dup[0]} "
                                                                   f"(rho={comp['proba_corr'].loc[nm, dup[0]]:.3f})"})
            continue
        chosen.append(nm)
        log.append({"model": nm, "decision": "included", "why": "complementary"})
    return chosen, log


# ----------------------------------------------------------------------------------- ensemble nested CV
def _make_suite(member_names, features, imbalance, names, cfg, calibration):
    members = [(l, bm.build_procedure({"kind": "single", "learner": l, "features": features, "imbalance": imbalance},
                                      names, cfg, tune_threshold=False)) for l in member_names]
    return ez.EnsembleSuite(members, calibration=calibration, inner_folds=cfg["classification"]["inner_folds"],
                            seed=cfg["random_seed"])


def _suite_fold(member_names, features, imbalance, names, cfg, calibration, X, y, tr, te):
    t0 = time.time()
    s = _make_suite(member_names, features, imbalance, names, cfg, calibration).fit(X.iloc[tr], y[tr])
    Xte = X.iloc[te]
    P = s.member_proba(Xte)
    out = {"te": te, "secs": time.time() - t0, "methods": {}, "members": {}, "names": s.names_,
           "weights": s.weights_.tolist()}
    for m in ez.ENSEMBLE_METHODS:
        sc = s.scores(Xte, m)
        out["methods"][m] = {o: (bm.clf_metrics(y[te], sc, s.predict_method(Xte, m, o)),
                                 s.predict_method(Xte, m, o), sc) for o in ez.OBJECTIVES}
    for j, nm in enumerate(s.names_):
        pred = (P[:, j] >= s.member_thr_[j]).astype(int)
        out["members"][nm] = (bm.clf_metrics(y[te], P[:, j], pred), pred, P[:, j])
    return out


def nested_suite(member_names, features, imbalance, X, y, cfg, repeats, calibration="sigmoid", tag="", n_jobs=8,
                 groups_tag=""):
    c, seed = cfg["classification"], cfg["random_seed"]
    names = list(X.columns)
    splits = list(RepeatedStratifiedKFold(n_splits=c["outer_folds"], n_repeats=repeats, random_state=seed).split(X, y))
    key = hashlib.sha1(json.dumps([member_names, features, imbalance, calibration, tag, repeats, c, names, len(y),
                                   int(np.sum(y)), seed, "suite_v1"], sort_keys=True, default=str).encode()).hexdigest()[:16]
    f = bm.CACHE / f"suite_{key}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    folds = Parallel(n_jobs=min(n_jobs, len(splits)))(
        delayed(_suite_fold)(member_names, features, imbalance, names, cfg, calibration, X, y, tr, te)
        for tr, te in splits)
    per_rep = len(splits) // repeats
    res = {"member_names": member_names, "n_folds": len(folds), "secs_per_fold": float(np.mean([x["secs"] for x in folds])),
           "weights": np.mean([x["weights"] for x in folds], axis=0).tolist()}

    def assemble(get):
        rows = []
        for i, fo in enumerate(folds):
            m, pred, sc = get(fo)
            rows += [(i // per_rep, int(a), float(b), int(d)) for a, b, d in zip(fo["te"], sc, pred)]
        return pd.DataFrame(rows, columns=["repeat", "idx", "proba", "pred"])

    res["methods"] = {}
    for m in ez.ENSEMBLE_METHODS:
        res["methods"][m] = {
            "name": f"suite:{m}", "n_folds": len(folds), "secs_per_fold": res["secs_per_fold"],
            "fold_metrics": pd.DataFrame([fo["methods"][m]["mcc"][0] for fo in folds]),
            "by_objective": {o: pd.DataFrame([fo["methods"][m][o][0] for fo in folds]) for o in ez.OBJECTIVES},
            "oof": assemble(lambda fo, m=m: fo["methods"][m]["mcc"])}
    res["members"] = {}
    for nm in folds[0]["names"]:
        res["members"][nm] = {"name": nm, "n_folds": len(folds), "secs_per_fold": res["secs_per_fold"],
                              "fold_metrics": pd.DataFrame([fo["members"][nm][0] for fo in folds]),
                              "oof": assemble(lambda fo, nm=nm: fo["members"][nm])}
    bm.CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(res, open(f, "wb"))
    return res


# ------------------------------------------------------------------------------------------- the gate
def cv_gate(ens: dict, single: dict, member_disagreement: float, cfg: dict) -> tuple[bool, list[dict]]:
    g = cfg["ensemble_lab"]["gate"]
    se, ss = bm.summarize(ens), bm.summarize(single)
    paired = (ens["fold_metrics"]["roc_auc"].to_numpy() - single["fold_metrics"]["roc_auc"].to_numpy())
    checks = [
        ("CV AUC improves", se["roc_auc_mean"] - ss["roc_auc_mean"], f">= {g['min_cv_auc_gain']}",
         se["roc_auc_mean"] - ss["roc_auc_mean"] >= g["min_cv_auc_gain"]),
        ("CV balanced accuracy maintained", se["balanced_accuracy_mean"] - ss["balanced_accuracy_mean"],
         f">= -{g['max_cv_balanced_accuracy_drop']}",
         se["balanced_accuracy_mean"] - ss["balanced_accuracy_mean"] >= -g["max_cv_balanced_accuracy_drop"]),
        ("MSI-High recall not collapsed", se["recall_mean"] / max(ss["recall_mean"], 1e-9), f">= {g['min_recall_ratio']}",
         se["recall_mean"] >= g["min_recall_ratio"] * ss["recall_mean"]),
        ("Stable across folds", se["roc_auc_std"] / max(ss["roc_auc_std"], 1e-9), f"<= {g['max_std_ratio']}",
         se["roc_auc_std"] <= max(ss["roc_auc_std"] * g["max_std_ratio"], 0.02)),
        ("Members are complementary (disagreement)", member_disagreement, f">= {g['min_member_disagreement']}",
         member_disagreement >= g["min_member_disagreement"]),
        ("Ensemble beats the single model in most folds", float((paired > 0).mean()),
         f">= {g['min_fraction_folds_improved']}", float((paired > 0).mean()) >= g["min_fraction_folds_improved"]),
    ]
    out = [{"check": c, "value": float(v), "requirement": r, "passed": bool(p)} for c, v, r, p in checks]
    return all(x["passed"] for x in out), out


def test_gate(ens_test: dict, single_test: dict, cfg: dict) -> tuple[bool, list[dict]]:
    g = cfg["ensemble_lab"]["gate"]
    d_auc = ens_test["roc_auc"] - single_test["roc_auc"]
    d_ba = ens_test["balanced_accuracy"] - single_test["balanced_accuracy"]
    checks = [("Locked-test AUC comparable or better", d_auc, f">= -{g['max_test_auc_drop']}", d_auc >= -g["max_test_auc_drop"]),
              ("Locked-test balanced accuracy comparable or better", d_ba, f">= -{g['max_test_balanced_accuracy_drop']}",
               d_ba >= -g["max_test_balanced_accuracy_drop"])]
    out = [{"check": c, "value": float(v), "requirement": r, "passed": bool(p)} for c, v, r, p in checks]
    return all(x["passed"] for x in out), out


# ------------------------------------------------------------------------------------ candidates / fits
def fit_candidate(spec: dict, X, y, cfg):
    if spec["kind"] == "single":
        return bm.fit_final(spec, X, y, cfg)
    suite = _make_suite(spec["members"], spec["features"], spec["imbalance"], list(X.columns), cfg,
                        spec["calibration"]).fit(X, y)
    return ez.EnsembleModel(suite, spec["method"], cfg["ensemble_lab"]["decision_objective"])


def score_candidate(model, X):
    return model.predict_proba(X)[:, 1], model.predict(X)


def threshold_tradeoff(oof_proba, y_dev, test_proba, y_test) -> pd.DataFrame:
    """Thresholds chosen on CROSS-VALIDATED development predictions only, then applied unchanged to the locked test."""
    rows = []
    for obj in ez.OBJECTIVES:
        thr = ez.threshold_for(y_dev, oof_proba, obj)
        cv_m = bm.clf_metrics(y_dev, oof_proba, (oof_proba >= thr).astype(int))
        te_m = bm.clf_metrics(y_test, test_proba, (test_proba >= thr).astype(int))
        rows.append({"objective": obj, "threshold": thr, "cv_accuracy": cv_m["accuracy"], "cv_recall": cv_m["recall"],
                     "cv_balanced_accuracy": cv_m["balanced_accuracy"], "cv_mcc": cv_m["mcc"],
                     "test_accuracy": te_m["accuracy"], "test_recall": te_m["recall"],
                     "test_balanced_accuracy": te_m["balanced_accuracy"], "test_mcc": te_m["mcc"],
                     "test_precision": te_m["precision"], "test_specificity": te_m["specificity"]})
    return pd.DataFrame(rows)


def load_cptac_matrix(fs: str):
    """CPTAC-2 expression for the genes this feature set needs (None if the raw fetch is missing)."""
    base = pd.read_csv(bm.PROCESSED / "external_cptac_coad_2019_processed.csv")
    y = (base["msi_label"] == "MSI-H").astype(int).to_numpy()
    X = base.set_index("sampleId")
    ext_dir = ROOT / "data" / "external" / "cptac_coad_2019"
    frames = [X]
    for fname in (["mrna_expression_all_genes_long.csv"] if fs == "whole_transcriptome"
                  else ["mrna_expression_extended_long.csv"]):
        p = ext_dir / fname
        if p.exists():
            long = pd.read_csv(p, usecols=["sampleId", "hugoGeneSymbol", "value"])
            wide = long.pivot_table(index="sampleId", columns="hugoGeneSymbol", values="value", aggfunc="first")
            wide.columns.name = None
            frames.append(wide.loc[wide.index.intersection(X.index)])
    M = pd.concat(frames, axis=1)
    M = M.loc[:, ~M.columns.duplicated()]
    return M.loc[base["sampleId"]].reset_index(drop=True), y


# ---------------------------------------------------------------------------------- one feature set
def run_feature_set(fs: str, df, y, dev, test, lock_path, cfg, progress=None, quick=False) -> dict:
    say = lambda m: bm._say(progress, f"[{fs}] {m}")  # noqa: E731
    c, el = cfg["classification"], cfg["ensemble_lab"]
    primary = cfg["primary_metric"]["classification"]
    out_dir = LAB / fs
    out_dir.mkdir(parents=True, exist_ok=True)
    Xdev, Xtest = feature_matrices(fs, df, dev, test, cfg)
    ydev, ytest = y[dev], y[test]
    say(f"{Xdev.shape[1]} features; dev {len(Xdev)}, test {len(Xtest)}")

    # ---- feature strategy (only the high-dimensional set needs a selection stage)
    features, imbalance = "raw", "none"
    wt_stage = None
    if fs == "whole_transcriptome":
        rows, folds = [], {}
        for s in ez.WT_STRATEGIES:
            fv = []
            for l in c["reference_learners"]:
                say(f"selection strategy {s} x {l}")
                r = bm.nested_cv({"kind": "single", "learner": l, "features": s, "imbalance": "none"}, Xdev, ydev, cfg,
                                 repeats=1, cache_tag="wt_stage")
                rows.append({"strategy": s, "learner": l, **bm.summarize(r)})
                fv.append(r["fold_metrics"][primary].to_numpy())
            folds[s] = np.mean(fv, axis=0)
        agg = pd.DataFrame([{"name": s, "rank": i, "n_folds": len(v), f"{primary}_mean": float(v.mean()),
                             f"{primary}_std": float(v.std(ddof=1))} for i, (s, v) in enumerate(folds.items())])
        features, why = bm.select_best(agg, primary, cfg)
        pd.DataFrame(rows).to_csv(out_dir / "selection_strategy_detail.csv", index=False)
        agg.assign(selected=agg["name"] == features).to_csv(out_dir / "selection_strategy_summary.csv", index=False)
        wt_stage = {"selected": features, "why": why}
        say(f"selection strategy -> {features} ({why})")

    # ---- base learners (identical folds for all)
    learners = learners_for(fs, cfg)
    if quick:
        learners = [l for l in learners if l in ("logreg_l2", "svm_rbf", "random_forest", "xgboost", "extra_trees")]
    results = {}
    for l in learners:
        say(f"base learner {l}")
        results[l] = bm.nested_cv({"kind": "single", "learner": l, "features": features, "imbalance": imbalance},
                                  Xdev, ydev, cfg, repeats=c["outer_repeats"], cache_tag="zoo")
    rank = {k: v["rank"] for k, v in c["learners"].items()}
    zoo = pd.DataFrame([{**bm.summarize(r), "rank": rank[n]} for n, r in results.items()])
    best_single, why_single = bm.select_best(zoo, primary, cfg)
    comp = complementarity(results, ydev)
    for k in ("proba_corr", "disagreement", "error_overlap", "q_statistic"):
        comp[k].round(4).to_csv(out_dir / f"complementarity_{k}.csv")
    comp["summary"].round(4).to_csv(out_dir / "complementarity_summary.csv", index=False)

    # ---- the best single model in its FINAL form (calibrated, same as its ensemble rivals)
    single_spec = {"kind": "single", "learner": best_single, "features": features, "imbalance": imbalance,
                   "calibration": el["calibration"]}
    single_res = bm.nested_cv(single_spec, Xdev, ydev, cfg, repeats=c["outer_repeats"], cache_tag="calib")

    # ---- ensembles
    auto, auto_log = auto_members(zoo, comp, cfg)
    member_sets = {"reference": [m for m in el["reference_members"] if m in results], "auto": auto}
    if member_sets["auto"] == member_sets["reference"]:
        member_sets.pop("auto")
    suites, ens_rows = {}, []
    for set_name, mem in member_sets.items():
        say(f"ensemble '{set_name}' members {mem}")
        suites[set_name] = nested_suite(mem, features, imbalance, Xdev, ydev, cfg, c["outer_repeats"],
                                        el["calibration"], tag="lab")
        idx = [list(results).index(m) for m in mem]
        member_dis = float(np.mean([comp["disagreement"].iloc[i, j] for a, i in enumerate(idx) for j in idx[a + 1:]])) \
            if len(idx) > 1 else 0.0
        for m, r in suites[set_name]["methods"].items():
            ok, checks = cv_gate(r, single_res, member_dis, cfg)
            ens_rows.append({**bm.summarize(r), "name": f"{set_name}:{m}", "set": set_name, "method": m,
                             "members": ",".join(mem), "rank": METHOD_RANK[m] + (0.5 if set_name == "auto" else 0),
                             "mean_member_disagreement": member_dis, "cv_gate_passed": ok,
                             "cv_gate_checks": json.dumps(checks)})
    ens_table = pd.DataFrame(ens_rows)

    # ---- CV-ONLY decision (locked test untouched so far)
    passing = ens_table[ens_table["cv_gate_passed"]]
    chosen_ens = None
    if len(passing):
        win, why_ens = bm.select_best(passing.reset_index(drop=True), primary, cfg)
        chosen_ens = ens_table[ens_table["name"] == win].iloc[0]
    say(f"best single: {best_single}; CV-gate-passing ensembles: {len(passing)}"
        + (f"; chosen for test check: {chosen_ens['name']}" if chosen_ens is not None else ""))

    # ---- calibration study for the reference ensemble (Phase 5)
    cal_rows = []
    if fs in ("core", "extended_molecular") and not quick:
        for cal in ("none", "sigmoid", "isotonic"):
            r = nested_suite(member_sets["reference"], features, imbalance, Xdev, ydev, cfg, c["outer_repeats"], cal,
                             tag="lab")
            for m in ("soft_equal", "stacking"):
                s = bm.summarize(r["methods"][m])
                cal_rows.append({"calibration": cal, "method": m, "brier": s["brier_mean"], "ece": s["ece_mean"],
                                 "roc_auc": s["roc_auc_mean"]})
        pd.DataFrame(cal_rows).to_csv(out_dir / "ensemble_calibration.csv", index=False)

    if quick:   # smoke run: CV only, the locked test set is never touched
        return {"quick": True, "feature_set": fs, "best_single": best_single, "ensembles": ens_table.drop(
            columns=["cv_gate_checks"]).to_dict(orient="records"), "chosen_ensemble_by_cv": None if chosen_ens is None
            else chosen_ens["name"]}

    # ===================================================================== locked test (single use per run)
    say("fitting candidates on the development set; scoring the locked test set")
    cand_specs = {f"single:{l}": {"kind": "single", "learner": l, "features": features, "imbalance": imbalance,
                                  "calibration": "none"} for l in learners}
    cand_specs["single_final"] = single_spec
    suite_fits = {}
    for set_name, mem in member_sets.items():
        suite_fits[set_name] = _make_suite(mem, features, imbalance, list(Xdev.columns), cfg, el["calibration"]).fit(Xdev, ydev)
    singles = Parallel(n_jobs=min(8, len(cand_specs)))(
        delayed(lambda sp: (lambda m: (m, *score_candidate(m, Xtest)))(fit_candidate(sp, Xdev, ydev, cfg)))(sp)
        for sp in cand_specs.values())
    single_fit = dict(zip(cand_specs, singles))
    test_rows = []
    for name, (m, pr, pd_) in single_fit.items():
        test_rows.append({"name": name, **{k: v for k, v in bm.clf_metrics(ytest, pr, pd_).items()}})
    for set_name, suite in suite_fits.items():
        for meth in ez.ENSEMBLE_METHODS:
            sc, pdc = suite.scores(Xtest, meth), suite.predict_method(Xtest, meth)
            test_rows.append({"name": f"{set_name}:{meth}", **bm.clf_metrics(ytest, sc, pdc)})
    test_all = pd.DataFrame(test_rows)
    test_all.to_csv(out_dir / "locked_test_all_candidates.csv", index=False)
    bm.log_test_evaluation(lock_path, f"ensemble_lab_{fs}_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
                           f"best_single={best_single}; chosen_ensemble={None if chosen_ens is None else chosen_ens['name']}")

    final, final_kind, gate_log = "single", "single", {"cv": None, "test": None}
    sm, spred, sproba = single_fit["single_final"][0], single_fit["single_final"][2], single_fit["single_final"][1]
    single_test = bm.clf_metrics(ytest, sproba, spred)
    final_model, final_spec, final_res, final_test_proba, final_test_pred = sm, single_spec, single_res, sproba, spred
    leak = {}
    if chosen_ens is not None:
        set_name, meth = chosen_ens["set"], chosen_ens["method"]
        suite = suite_fits[set_name]
        ens_model = ez.EnsembleModel(suite, meth, el["decision_objective"])
        ep, ed_ = ens_model.predict_proba(Xtest)[:, 1], ens_model.predict(Xtest)
        ens_test = bm.clf_metrics(ytest, ep, ed_)
        ok_cv, cv_checks = True, json.loads(chosen_ens["cv_gate_checks"])
        ok_test, test_checks = test_gate(ens_test, single_test, cfg)
        # leakage canary + reproducibility, only for this one candidate
        say("leakage canary + reproducibility check for the chosen ensemble")
        perm_aucs = []
        for k in range(c["permutation_canary_runs"]):
            yp = np.random.default_rng(2000 + k).permutation(ydev)
            pr_ = nested_suite(member_sets[set_name], features, imbalance, Xdev, yp, cfg, 1, el["calibration"], tag=f"perm{k}")
            perm_aucs.append(float(pr_["methods"][meth]["fold_metrics"]["roc_auc"].mean()))
        suite2 = _make_suite(member_sets[set_name], features, imbalance, list(Xdev.columns), cfg, el["calibration"]).fit(Xdev, ydev)
        reproducible = bool(np.allclose(suite2.scores(Xtest, meth), suite.scores(Xtest, meth), atol=1e-9))
        leak = {"permutation_auc": perm_aucs, "permutation_auc_mean": float(np.mean(perm_aucs)),
                "permutation_pass": bool(abs(np.mean(perm_aucs) - 0.5) < el["gate"]["permutation_canary_tolerance"]),
                "reproducible": reproducible}
        leak["passed"] = leak["permutation_pass"] and reproducible
        ok_all = ok_cv and ok_test and leak["passed"]
        gate_log = {"cv": cv_checks, "test": test_checks, "leakage": leak, "adopted": bool(ok_all)}
        if ok_all:
            final, final_kind = chosen_ens["name"], "ensemble"
            final_model = ens_model
            final_spec = {"kind": "suite", "members": member_sets[set_name], "method": meth, "features": features,
                          "imbalance": imbalance, "calibration": el["calibration"]}
            final_res = suites[set_name]["methods"][meth]
            final_test_proba, final_test_pred = ep, ed_
    if not leak:  # single model: run the same canary so every final model has one
        perm_aucs = []
        for k in range(c["permutation_canary_runs"]):
            yp = np.random.default_rng(1000 + k).permutation(ydev)
            perm_aucs.append(float(bm.nested_cv(single_spec, Xdev, yp, cfg, repeats=1, cache_tag=f"perm{k}")
                                   ["fold_metrics"]["roc_auc"].mean()))
        leak = {"permutation_auc": perm_aucs, "permutation_auc_mean": float(np.mean(perm_aucs)),
                "permutation_pass": bool(abs(np.mean(perm_aucs) - 0.5) < el["gate"]["permutation_canary_tolerance"])}
        leak["passed"] = leak["permutation_pass"]

    final_cv = bm.summarize(final_res)
    final_test = bm.clf_metrics(ytest, final_test_proba, final_test_pred)
    final_ci = bm.bootstrap_ci(ytest, final_test_proba, final_test_pred)
    oof = final_res["oof"].groupby("idx", as_index=False)["proba"].mean().sort_values("idx")
    tradeoff = threshold_tradeoff(oof["proba"].to_numpy(), ydev[oof["idx"].to_numpy()], final_test_proba, ytest)
    tradeoff.round(4).to_csv(out_dir / "threshold_tradeoff.csv", index=False)
    pd.DataFrame({"sampleId": df["sampleId"].iloc[test].to_numpy(), "y_true": ytest, "proba": final_test_proba,
                  "pred": final_test_pred}).to_csv(out_dir / "locked_test_predictions.csv", index=False)

    # ---- external cohort
    external = None
    try:
        Mx, yx = load_cptac_matrix(fs)
        shared = [g for g in Xdev.columns if g in Mx.columns and Mx[g].notna().mean() > 0.9]
        if len(shared) >= 20:
            say(f"external CPTAC-2: refit on {len(shared)}/{Xdev.shape[1]} shared genes")
            m_ext = fit_candidate(final_spec, Xdev[shared], ydev, cfg)
            xp, xd = score_candidate(m_ext, Mx[shared])
            external = {"cohort": "CPTAC-2 colon (Cell 2019)", "n": int(len(yx)), "n_msi_h": int(yx.sum()),
                        "n_shared_features": len(shared), "n_features": int(Xdev.shape[1]),
                        "majority_accuracy": float(1 - yx.mean()), "metrics": bm.clf_metrics(yx, xp, xd),
                        "ci95": bm.bootstrap_ci(yx, xp, xd)}
    except Exception as e:  # noqa: BLE001
        external = {"error": repr(e)}

    # ---- explainability
    imp = permutation_importance(final_model, Xtest, ytest, scoring="roc_auc", n_repeats=15 if fs == "whole_transcriptome" else 30,
                                 random_state=cfg["random_seed"], n_jobs=1)
    imp_df = pd.DataFrame({"feature": Xtest.columns, "importance_mean": imp.importances_mean,
                           "importance_std": imp.importances_std}).sort_values("importance_mean", ascending=False)
    imp_df.head(50).to_csv(out_dir / "feature_importance_permutation.csv", index=False)

    ens_table.drop(columns=["cv_gate_checks"]).to_csv(out_dir / "ensemble_cv.csv", index=False)
    zoo.assign(selected=zoo["name"] == best_single).to_csv(out_dir / "base_learners_cv.csv", index=False)

    summary = {"feature_set": fs, "label": LABEL_OF[fs], "n_features": int(Xdev.shape[1]), "features_strategy": features,
               "wt_selection": wt_stage, "best_single": best_single, "best_single_reason": why_single,
               "single_cv": bm.summarize(single_res), "single_test": single_test,
               "ensemble_member_sets": member_sets, "auto_member_log": auto_log,
               "oracle_accuracy": comp["oracle_accuracy"], "best_single_accuracy_oof": comp["best_single_accuracy_oof"],
               "chosen_ensemble_by_cv": None if chosen_ens is None else chosen_ens["name"], "gate": gate_log,
               "final": {"name": final, "kind": final_kind, "spec": final_spec, "cv": final_cv,
                         "test": {**final_test, "ci95": final_ci, "n": int(len(ytest)), "n_positive": int(ytest.sum())},
                         "leakage": leak, "external": external, "decision_threshold": float(final_model.threshold_)},
               "skipped_learners": [n for n in (el["learners_default"] if fs != "whole_transcriptome"
                                                else el["learners_whole_transcriptome"]) if n not in learners]
               + ([] if fs == "core" else el["learners_core_extra"]),
               "created_at": datetime.now(timezone.utc).isoformat()}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    # ---- persist the FS final as a registry candidate
    bundle = {"model": final_model, "features": list(Xdev.columns), "threshold": float(final_model.threshold_),
              "spec": final_spec, "pos_label": "MSI-H", "neg_label": "MSS/MSI-L",
              "background": Xdev.sample(min(100, len(Xdev)), random_state=cfg["random_seed"]).reset_index(drop=True),
              "calibration": el["calibration"] if final_kind == "ensemble" else final_spec.get("calibration", "none"),
              "kind": final_kind, "feature_set": fs, "feature_set_label": LABEL_OF[fs],
              "members": member_sets[chosen_ens["set"]] if final_kind == "ensemble" else [best_single],
              "method": final_spec.get("method", "single")}
    ent = reg.register_benchmark_model(
        task=TASK_OF[fs], name=(best_single if final_kind == "single" else final.replace(":", "_")), display_name=f"{LABEL_OF[fs]}: " + (
            "ensemble" if final_kind == "ensemble" else mz.PLAIN_ENGLISH.get(best_single, best_single)),
        model_object=bundle, feature_columns=list(Xdev.columns), feature_pipeline=f"{fs}|{features}|{imbalance}",
        dataset="tcga_coadread_processed", hyperparameters={"kind": final_kind, "spec": final_spec},
        cv_metrics={k: v for k, v in final_cv.items() if k != "name"},
        test_metrics={**{k: float(v) for k, v in final_test.items()}, "ci95": final_ci, "n": int(len(ytest)),
                      "n_positive": int(ytest.sum())},
        external_metrics=({**external["metrics"], "n": external["n"]} if external and "metrics" in external else None),
        calibration=bundle["calibration"], seed=cfg["random_seed"],
        extra={"pos_label": "MSI-H", "neg_label": "MSS/MSI-L", "feature_set": fs, "feature_set_label": LABEL_OF[fs],
               "kind": final_kind, "leakage_checks": leak, "n_features": int(Xdev.shape[1])})
    summary["registry_entry"] = {"name": ent["name"], "version": ent["version"], "task": ent["task"]}
    summary["test_table"] = test_all.to_dict(orient="records")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


# --------------------------------------------------------------------------------------------- driver
def run_ensemble_lab(progress=None, quick=False, feature_sets=None, reuse_finished=True) -> dict:
    t0 = time.time()
    cfg = bm.load_cfg()
    LAB.mkdir(parents=True, exist_ok=True)
    df, genes, y = bm.load_msi_data()
    dev, test, lock_spec, lock_path = bm.locked_split(df, y, cfg)
    sets = feature_sets or cfg["ensemble_lab"]["feature_sets"]
    summaries = {}
    for fs in sets:
        done = LAB / fs / "summary.json"
        if not quick and reuse_finished and done.exists() and "registry_entry" in json.load(open(done)):
            bm._say(progress, f"[{fs}] finished earlier: reusing stored results (locked test is not scored again)")
            summaries[fs] = json.load(open(done))
            continue
        summaries[fs] = run_feature_set(fs, df, y, dev, test, lock_path, cfg, progress, quick)
    if quick:
        return summaries

    # ---- production routing (decisions from CV tables; the locked test only confirmed the per-set gate)
    primary = cfg["primary_metric"]["classification"]
    routing = {}
    if "core" in summaries:
        s = summaries["core"]
        inc = reg.current_best("classification")
        cand = reg._load_registry()
        entry = next(e for e in reversed(cand) if e["task"] == "classification" and e["version"] == s["registry_entry"]["version"]
                     and e["name"] == s["registry_entry"]["name"])
        if s["final"]["kind"] == "ensemble":
            ok, why = (True, ["no incumbent"]) if inc is None else (True, ["ensemble passed the full ensemble gate "
                                                                          "(CV, balanced accuracy, recall, stability, complementarity, locked test, leakage, reproducibility)"])
            reg.promote_champion("classification", entry, "; ".join(why))
            routing["core"] = {"promoted": True, "reason": why}
        else:
            routing["core"] = {"promoted": False, "reason": ["best single model retained: no ensemble passed the gate"
                                                             if s["chosen_ensemble_by_cv"] is None or not s["gate"].get("adopted", False)
                                                             else "single retained"]}
    ext_sets = [fs for fs in ("core_plus_mmr", "extended_molecular") if fs in summaries]
    if ext_sets:
        tab = pd.DataFrame([{"name": fs, "rank": i, "n_folds": summaries[fs]["final"]["cv"]["n_folds"],
                             f"{primary}_mean": summaries[fs]["final"]["cv"][f"{primary}_mean"],
                             f"{primary}_std": summaries[fs]["final"]["cv"][f"{primary}_std"]} for i, fs in enumerate(ext_sets)])
        win, why = bm.select_best(tab, primary, cfg)
        s = summaries[win]
        entry = next(e for e in reversed(reg._load_registry()) if e["task"] == "classification_extended"
                     and e["version"] == s["registry_entry"]["version"] and e["name"] == s["registry_entry"]["name"])
        reg.promote_champion("classification_extended", entry, f"Extended-model choice by CV: {why}")
        routing["extended"] = {"promoted": win, "reason": why}
    if "whole_transcriptome" in summaries:
        s = summaries["whole_transcriptome"]
        entry = next(e for e in reversed(reg._load_registry()) if e["task"] == "classification_wt"
                     and e["version"] == s["registry_entry"]["version"])
        reg.promote_champion("classification_wt", entry, "research benchmark only (not served)")
    lab = {"created_at": datetime.now(timezone.utc).isoformat(), "runtime_min": (time.time() - t0) / 60,
           "routing": routing,
           "table": [{"feature_set": fs, "label": LABEL_OF[fs], "n_features": s["n_features"], "final": s["final"]["name"],
                      "kind": s["final"]["kind"], "cv_accuracy": s["final"]["cv"]["accuracy_mean"],
                      "cv_accuracy_std": s["final"]["cv"]["accuracy_std"], "cv_auc": s["final"]["cv"]["roc_auc_mean"],
                      "cv_balanced_accuracy": s["final"]["cv"]["balanced_accuracy_mean"],
                      "cv_recall": s["final"]["cv"]["recall_mean"],
                      "test_accuracy": s["final"]["test"]["accuracy"], "test_auc": s["final"]["test"]["roc_auc"],
                      "test_recall": s["final"]["test"]["recall"], "test_mcc": s["final"]["test"]["mcc"],
                      "best_single": s["best_single"], "single_cv_accuracy": s["single_cv"]["accuracy_mean"],
                      "single_test_accuracy": s["single_test"]["accuracy"],
                      "external_accuracy": (s["final"]["external"] or {}).get("metrics", {}).get("accuracy"),
                      "external_auc": (s["final"]["external"] or {}).get("metrics", {}).get("roc_auc")}
                     for fs, s in summaries.items()],
           "majority_baseline": float(1 - y.mean()), "target": cfg["classification"]["target_accuracy"],
           # configured feature sets with no finished result: recorded explicitly, never given metrics
           "not_completed": [{"feature_set": fs, "label": LABEL_OF[fs], "status": "not_completed",
                              "reason": "experiment did not finish; no metrics are reported for it"}
                             for fs in cfg["ensemble_lab"]["feature_sets"] if fs not in summaries]}
    (LAB / "lab_summary.json").write_text(json.dumps(lab, indent=2, default=float))
    return lab
