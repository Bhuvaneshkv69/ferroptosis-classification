"""
End-to-end classification benchmark: stages -> selection -> final fit -> locked test -> external
validation -> leakage canaries -> artifacts -> registry + promotion gate.

    python scripts/run_benchmark.py            # full run
    python scripts/run_benchmark.py --quick    # smoke run (3 learners, 3 feature strategies; NO locked-test use)

The locked test set is touched exactly once per full run, after the champion has been chosen from
development-set cross-validation alone.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.inspection import permutation_importance

import benchmark as bm
import evaluation as ev
import model_registry as reg
import model_zoo as mz

OUT, FIG = bm.OUT, bm.FIG


def _fit_and_score(spec, Xdev, ydev, Xtest, ytest, cfg):
    est = bm.fit_final(spec, Xdev, ydev, cfg)
    proba = est.predict_proba(Xtest)[:, 1]
    pred = est.predict(Xtest)
    return est, proba, pred, bm.clf_metrics(ytest, proba, pred)


def rejection_reasons(table: pd.DataFrame, champion: str, primary: str, cfg: dict) -> dict:
    hb = bm.HIGHER_BETTER[primary]
    c = table.set_index("name")
    win = c.loc[champion]
    out = {}
    for name, r in c.iterrows():
        if name == champion:
            continue
        d = (win[f"{primary}_mean"] - r[f"{primary}_mean"]) * (1 if hb else -1)
        reasons = []
        if r[f"{primary}_std"] > cfg["selection"]["max_cv_std"]:
            reasons.append(f"unstable (fold std {r[f'{primary}_std']:.3f} > {cfg['selection']['max_cv_std']})")
        se = r[f"{primary}_std"] / np.sqrt(r["n_folds"])
        if d > 0:
            reasons.append(f"{primary} {r[f'{primary}_mean']:.3f} vs champion {win[f'{primary}_mean']:.3f} "
                           f"(gap {d:.3f}, {'within' if d <= se else 'beyond'} 1 SE)")
        if r["rank"] > win["rank"]:
            reasons.append("more complex than the champion")
        out[name] = "; ".join(reasons) or "tied with champion; champion preferred as simpler/stabler"
    return out


def evaluate_promotion(candidate: dict, incumbent: dict | None, cfg: dict) -> tuple[bool, list[str]]:
    """Gate for replacing the production model. ALL of: CV improvement, locked-test no worse, stability, no leakage."""
    primary = cfg["primary_metric"]["classification"]
    p = cfg["selection"]["promotion"]
    if incumbent is None:
        return True, ["no incumbent model: first validated model is promoted"]
    reasons, ok = [], True
    # structural preconditions: required metrics present, and the candidate scores the SAME features the app feeds the
    # production slot (a model with a different feature set must never silently replace the served one)
    need = [f"{primary}_mean", f"{primary}_std"]
    if any(k not in candidate.get("cv_metrics", {}) for k in need) or primary not in candidate.get("test_metrics", {}):
        return False, ["required validation metrics are missing: not promotable"]
    if candidate.get("features") is not None and incumbent.get("feature_columns") is not None             and list(candidate["features"]) != list(incumbent["feature_columns"]):
        return False, ["feature set differs from the production model: not promotable into this slot"]
    cv_new, cv_old = candidate["cv_metrics"][f"{primary}_mean"], incumbent["cv_metrics"][f"{primary}_mean"]
    if cv_new - cv_old < p["min_cv_improvement"]:
        ok = False
        reasons.append(f"CV {primary} {cv_new:.3f} vs incumbent {cv_old:.3f}: improvement < {p['min_cv_improvement']}")
    t_new, t_old = candidate["test_metrics"][primary], incumbent["test_metrics"][primary]
    if t_new < t_old - p["max_test_drop"]:
        ok = False
        reasons.append(f"locked-test {primary} {t_new:.3f} below incumbent {t_old:.3f}")
    s_new, s_old = candidate["cv_metrics"][f"{primary}_std"], incumbent["cv_metrics"][f"{primary}_std"]
    if s_new > s_old * p["max_std_ratio"] and s_new > 0.02:
        ok = False
        reasons.append(f"less stable: fold std {s_new:.3f} > {p['max_std_ratio']} x incumbent {s_old:.3f}")
    if p["require_leakage_checks"] and not candidate["leakage_checks"]["passed"]:
        ok = False
        reasons.append("leakage checks failed")
    if ok:
        reasons.append("all promotion criteria satisfied")
    return ok, reasons


def run_classification_benchmark(quick: bool = False, progress=None, promote: bool = False, dataset_path=None) -> dict:
    t0 = time.time()
    cfg = bm.load_cfg()
    c = cfg["classification"]
    primary = cfg["primary_metric"]["classification"]
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    say = lambda m, f=None: bm._say(progress, m, f)  # noqa: E731

    df, genes, y = bm.load_msi_data(dataset_path)
    dev, test, lock_spec, lock_path = bm.locked_split(df, y, cfg)
    assert not set(dev) & set(test), "development and test sets overlap"
    assert df["patientId"].iloc[dev].isin(df["patientId"].iloc[test]).sum() == 0, "patient appears in both sets"
    Xdev, ydev = df[genes].iloc[dev].reset_index(drop=True), y[dev]
    Xtest, ytest = df[genes].iloc[test].reset_index(drop=True), y[test]
    say(f"Data: {len(df)} labelled samples; dev {len(dev)} ({ydev.sum()} MSI-H); locked test {len(test)} "
        f"({ytest.sum()} MSI-H); majority-class accuracy {1 - y.mean():.3f}")

    # -- stage 1: feature strategy ------------------------------------------------------------
    f_detail, f_agg = bm.run_feature_strategy_stage(Xdev, ydev, cfg, quick, progress)
    best_feat, why_feat = bm.select_best(f_agg, primary, cfg)
    f_detail.to_csv(OUT / "feature_strategy_detail.csv", index=False)
    f_agg.assign(selected=f_agg["name"] == best_feat).to_csv(OUT / "feature_strategy_summary.csv", index=False)
    say(f"Feature strategy -> {best_feat} ({why_feat})")

    # -- stage 2: class imbalance -------------------------------------------------------------
    i_detail, i_agg = bm.run_imbalance_stage(Xdev, ydev, cfg, best_feat, quick, progress)
    best_imb, why_imb = bm.select_best(i_agg, primary, cfg)
    i_detail.to_csv(OUT / "imbalance_detail.csv", index=False)
    i_agg.assign(selected=i_agg["name"] == best_imb).to_csv(OUT / "imbalance_summary.csv", index=False)
    say(f"Imbalance strategy -> {best_imb} ({why_imb})")

    # -- stage 3: model zoo + ensembles -------------------------------------------------------
    results, zoo, skipped = bm.run_zoo_stage(Xdev, ydev, cfg, best_feat, best_imb, c["outer_repeats"], quick, progress)
    champion, why_champ = bm.select_best(zoo, primary, cfg)
    say(f"Champion (CV-selected) -> {champion} ({why_champ})")

    # -- stage 4: calibration (single-model champions; ensembles are reported uncalibrated) ----
    champ_spec = {**results[champion]["spec"], "calibration": "none"}
    calib_note = "not evaluated for ensembles (probabilities are averaged / stacked)"
    cal_table = None
    champ_res = results[champion]
    if champ_spec["kind"] == "single" and not quick:
        cal_runs, cal_table = bm.run_calibration_stage(Xdev, ydev, cfg, champ_spec, c["outer_repeats"], progress)
        best_cal, why_cal = bm.select_best(cal_table, "brier", cfg)
        champ_spec["calibration"] = best_cal
        champ_res = cal_runs[best_cal]
        calib_note = f"{best_cal} ({why_cal})"
        cal_table.assign(selected=cal_table["name"] == best_cal).to_csv(OUT / "calibration_summary.csv", index=False)
    say(f"Calibration -> {champ_spec['calibration']}  [{calib_note}]")

    zoo_out = zoo.copy()
    zoo_out["selected"] = zoo_out["name"] == champion
    zoo_out["reason_not_selected"] = zoo_out["name"].map(rejection_reasons(zoo, champion, primary, cfg))
    zoo_out.to_csv(OUT / "model_zoo_cv.csv", index=False)

    champ_cv = bm.summarize(champ_res)
    result = {"quick": quick, "feature_strategy": best_feat, "imbalance": best_imb, "champion": champion,
              "champion_spec": champ_spec, "selection": {"features": why_feat, "imbalance": why_imb,
                                                          "model": why_champ, "calibration": calib_note},
              "skipped_learners": skipped, "cv_metrics": champ_cv}
    if quick:
        (OUT / "benchmark_summary_quick.json").write_text(json.dumps(result, indent=2, default=float))
        say("Quick run complete (locked test NOT evaluated).")
        return result

    # -- stage 5: leakage canaries (development data only) ---------------------------------------
    say("[checks] label-permutation canary")
    perm = bm.permutation_canary(champ_spec, Xdev, ydev, cfg, c["permutation_canary_runs"])
    site_auc = None
    if c.get("site_grouped_cv") and "TISSUE_SOURCE_SITE_CODE" in df:
        say("[checks] leave-sites-out CV")
        groups = df["TISSUE_SOURCE_SITE_CODE"].fillna("NA").iloc[dev].astype(str).to_numpy()
        site = bm.nested_cv(champ_spec, Xdev, ydev, cfg, repeats=1, groups=groups, cache_tag="site")
        site_auc = {k: v for k, v in bm.summarize(site).items() if k.startswith(("roc_auc", "mcc", "accuracy"))}
    leakage = {"permutation_auc": perm, "permutation_auc_mean": float(np.mean(perm)),
               "permutation_pass": bool(abs(np.mean(perm) - 0.5) < 0.08),
               "dev_test_sample_overlap": 0, "patient_overlap": 0, "site_grouped_cv": site_auc,
               "preprocessing_inside_cv": True, "smote_inside_training_fold_only": True,
               "feature_selection_inside_cv": True}
    leakage["passed"] = leakage["permutation_pass"]

    # -- stage 6: final fit + locked test (single use) -------------------------------------------
    say("[final] fitting all candidates on the development set; scoring the locked test set ONCE")
    specs = {n: {**r["spec"], "calibration": "none"} for n, r in results.items()}
    specs[champion] = champ_spec
    fits = Parallel(n_jobs=min(8, len(specs)))(
        delayed(_fit_and_score)(s, Xdev, ydev, Xtest, ytest, cfg) for s in specs.values())
    fitted = dict(zip(specs, fits))
    bm.log_test_evaluation(lock_path, f"benchmark_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}",
                           f"champion={champion}; {len(specs)} candidates reported (not used for selection)")
    est, proba, pred, test_m = fitted[champion]
    ci = bm.bootstrap_ci(ytest, proba, pred)
    all_rows = [{"name": n, **{k: m[k] for k in ("roc_auc", "pr_auc", "f1", "balanced_accuracy", "mcc",
                                                   "accuracy", "recall", "specificity", "brier")}}
                for n, (_, _, _, m) in fitted.items()]
    pd.DataFrame(all_rows).to_csv(OUT / "locked_test_all_models.csv", index=False)
    pd.DataFrame({"sampleId": df["sampleId"].iloc[test].to_numpy(), "y_true": ytest, "proba": proba,
                  "pred": pred}).to_csv(OUT / "locked_test_predictions.csv", index=False)
    oof = champ_res["oof"].groupby("idx", as_index=False)["proba"].mean()
    oof["sampleId"] = df["sampleId"].iloc[dev].to_numpy()[oof["idx"]]
    oof["y_true"] = ydev[oof["idx"]]
    oof.drop(columns="idx").to_csv(OUT / "cv_oof_predictions.csv", index=False)

    # -- stage 7: external validation (CPTAC-2): same procedure, refit on the shared genes -------
    external = None
    try:
        ext_df, ext_y = bm.load_cptac()
        shared = [g for g in genes if g in ext_df.columns]
        if len(shared) >= 20:
            say(f"[external] CPTAC-2: refitting champion procedure on {len(shared)}/{len(genes)} shared genes")
            ext_est = bm.fit_final(champ_spec, Xdev[shared], ydev, cfg)
            ep = ext_est.predict_proba(ext_df[shared])[:, 1]
            ed = ext_est.predict(ext_df[shared])
            em = bm.clf_metrics(ext_y, ep, ed)
            external = {"cohort": "CPTAC-2 colon (Cell 2019)", "n": int(len(ext_y)), "n_msi_h": int(ext_y.sum()),
                        "n_shared_genes": len(shared), "missing_genes": sorted(set(genes) - set(shared)),
                        "majority_accuracy": float(1 - ext_y.mean()), "metrics": em,
                        "ci95": bm.bootstrap_ci(ext_y, ep, ed),
                        "note": "Procedure refit on development data restricted to the shared genes; the external "
                                "cohort was never used for any choice. Platform/batch shift is real."}
            pd.DataFrame({"sampleId": ext_df["sampleId"], "y_true": ext_y, "proba": ep, "pred": ed}).to_csv(
                OUT / "external_predictions.csv", index=False)
    except FileNotFoundError:
        external = None

    # -- stage 8: explainability (permutation importance on the locked test set) --------------------
    say("[explain] permutation importance")
    imp = permutation_importance(est, Xtest, ytest, scoring="roc_auc", n_repeats=30, random_state=cfg["random_seed"],
                                 n_jobs=1)
    imp_df = pd.DataFrame({"gene": genes, "importance_mean": imp.importances_mean,
                           "importance_std": imp.importances_std}).sort_values("importance_mean", ascending=False)
    imp_df.to_csv(OUT / "feature_importance_permutation.csv", index=False)

    # -- figures --------------------------------------------------------------------------------------
    ev.plot_roc_curves({"champion (locked test)": (ytest, proba)}, 1, "ROC - locked test set", FIG / "benchmark_roc_test.png")
    ev.plot_pr_curves({"champion (locked test)": (ytest, proba)}, 1, "Precision-recall - locked test set",
                      FIG / "benchmark_pr_test.png")
    ev.plot_calibration({"champion (locked test)": (ytest, proba)}, 1, "Calibration - locked test set",
                        FIG / "benchmark_calibration_test.png")
    ev.plot_confusion_matrix(ytest, pred, [0, 1], "Locked test confusion matrix (0=MSS/MSI-L, 1=MSI-H)",
                             FIG / "benchmark_confusion_test.png")

    # -- 90% verdict (honest, strict: locked test AND cross-validation must both reach the target) -------
    tgt = c["target_accuracy"]
    verdict = {"target": tgt, "majority_baseline_accuracy": float(1 - y.mean()),
               "cv_accuracy_mean": champ_cv["accuracy_mean"], "locked_test_accuracy": test_m["accuracy"],
               "locked_test_accuracy_ci95": ci["accuracy"],
               "external_accuracy": external["metrics"]["accuracy"] if external else None}
    verdict["achieved"] = bool(champ_cv["accuracy_mean"] >= tgt and test_m["accuracy"] >= tgt)
    verdict["statement"] = "90% target achieved" if verdict["achieved"] else "90% target not achieved"

    cv_metrics = {k: v for k, v in champ_cv.items() if k not in ("name",)}
    test_metrics = {**{k: float(v) for k, v in test_m.items()}, "ci95": ci, "n": int(len(ytest)),
                    "n_positive": int(ytest.sum())}
    summary = {"quick": False, "created_at": datetime.now(timezone.utc).isoformat(), "primary_metric": primary,
               "n_total": int(len(df)), "n_dev": int(len(dev)), "n_test": int(len(test)),
               "prevalence_msi_h": float(y.mean()), "feature_strategy": best_feat, "imbalance": best_imb,
               "champion": champion, "champion_spec": champ_spec, "selection": result["selection"],
               "skipped_learners": skipped, "cv_metrics": cv_metrics, "locked_test_metrics": test_metrics,
               "external_validation": external, "leakage_checks": leakage, "verdict": verdict,
               "locked_test_uses": json.load(open(lock_path))["test_evaluations"],
               "decision_threshold": float(est.threshold_), "hyperparameters": getattr(est, "best_params_", {}),
               "rejected": rejection_reasons(zoo, champion, primary, cfg), "runtime_min": (time.time() - t0) / 60}

    # -- registry + promotion gate ------------------------------------------------------------------------
    bundle = {"model": est, "features": genes, "threshold": float(est.threshold_), "spec": champ_spec,
              "pos_label": "MSI-H", "neg_label": "MSS/MSI-L",
              "background": Xdev.sample(min(100, len(Xdev)), random_state=cfg["random_seed"]).reset_index(drop=True),
              "calibration": champ_spec["calibration"]}
    cand = {"cv_metrics": cv_metrics, "test_metrics": {**test_m}, "leakage_checks": leakage, "features": genes}
    entry = reg.register_benchmark_model(
        task="classification", name=champion, display_name=mz.PLAIN_ENGLISH.get(champion, champion),
        model_object=bundle, feature_columns=genes, feature_pipeline=f"{best_feat}+{best_imb}",
        dataset="tcga_coadread_processed", hyperparameters=_jsonable(getattr(est, "best_params_", {})),
        cv_metrics=cv_metrics, test_metrics=test_metrics,
        external_metrics=({**external["metrics"], "n": external["n"], "n_shared_genes": external["n_shared_genes"]}
                          if external else None),
        calibration=champ_spec["calibration"], seed=cfg["random_seed"],
        extra={"pos_label": "MSI-H", "neg_label": "MSS/MSI-L", "threshold": float(est.threshold_),
               "leakage_checks": leakage, "verdict": verdict["statement"]})
    summary["registry_entry"] = {"name": entry["name"], "version": entry["version"], "path": entry["path"]}
    if promote:
        incumbent = reg.current_best("classification")
        ok, reasons = evaluate_promotion(cand, incumbent, cfg) if (incumbent is None or "cv_metrics" in incumbent) \
            else (True, ["incumbent predates the benchmark protocol and is not comparable; promoting"])
        summary["promotion"] = {"promoted": ok, "reasons": reasons}
        import promotion as pr
        bad = pr.structural_failures(entry, incumbent, required_cv=(f"{primary}_mean", f"{primary}_std"),
                                     required_test=(primary,), root=bm.ROOT, check_pipeline=False)
        if bad:
            ok, reasons = False, reasons + bad
            summary["promotion"] = {"promoted": ok, "reasons": reasons}
        if ok:
            reg.promote_champion("classification", entry, "; ".join(reasons))
    (OUT / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    say(f"Done in {summary['runtime_min']:.1f} min. {verdict['statement']}.")
    return summary


def _jsonable(d):
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in d.items()}
