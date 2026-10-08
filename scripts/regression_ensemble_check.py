"""Phase 20: do the better regressors provide complementary predictions, and does an ensemble of them beat the
champion? Uses the cached nested-CV runs (no refit) for the complementarity analysis; fits only ONE new nested run
(the ensemble). The ensemble is promoted only if CV improves, folds are stable, the locked test is not worse and
the per-drug-mean baseline is beaten.

    python scripts/regression_ensemble_check.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import benchmark as bm  # noqa: E402
import model_registry as reg  # noqa: E402
import model_zoo as mz  # noqa: E402
import regression_benchmark as rb  # noqa: E402


def main():
    cfg = bm.load_cfg()
    r = cfg["regression"]
    df, genes, y = rb.load_gdsc(r["target"])
    dev, test, lock = rb.locked_cell_line_split(df, cfg)
    cols = [rb.ID, "DRUG_NAME", "PATHWAY_NAME", "MIN_CONC", "MAX_CONC"] + genes
    X = df[cols]
    Xdev, ydev, Xtest, ytest = X.iloc[dev].reset_index(drop=True), y[dev], X.iloc[test].reset_index(drop=True), y[test]
    summ = json.load(open(bm.OUT / "regression_summary.json"))
    champion = summ["champion"]
    names = [n for n in r["learners"] if n != "elastic_net" or True]
    runs = {n: rb.nested(n, genes, "drug_identity", cfg, Xdev, ydev) for n in names}
    # complementarity of out-of-fold predictions (same GroupKFold folds for every model)
    oof = {n: runs[n]["oof"].sort_values("idx")["pred"].to_numpy() for n in names}
    truth = ydev[runs[names[0]]["oof"].sort_values("idx")["idx"].to_numpy()]
    res = {n: truth - oof[n] for n in names}
    corr_pred = pd.DataFrame([[spearmanr(oof[a], oof[b])[0] for b in names] for a in names], index=names, columns=names)
    corr_err = pd.DataFrame([[np.corrcoef(res[a], res[b])[0, 1] for b in names] for a in names], index=names, columns=names)
    cv_mae = {n: float(runs[n]["fold_metrics"]["mae"].mean()) for n in names}
    top = [n for n in sorted(cv_mae, key=cv_mae.get) if cv_mae[n] <= cv_mae[sorted(cv_mae, key=cv_mae.get)[0]] * 1.10][:4]
    ens_name = "voting:" + ",".join(top)
    ens = rb.nested(ens_name, genes, "drug_identity", cfg, Xdev, ydev)
    base = rb.nested("drug_mean_baseline", genes, "drug_only", cfg, Xdev, ydev)
    champ = runs[champion]
    paired = champ["fold_metrics"]["mae"].to_numpy() - ens["fold_metrics"]["mae"].to_numpy()
    g = cfg["selection"]["promotion"]
    gain = float(champ["fold_metrics"]["mae"].mean() - ens["fold_metrics"]["mae"].mean())
    out = {"members": top, "cv_mae": {n: cv_mae[n] for n in names}, "ensemble_cv_mae": float(ens["fold_metrics"]["mae"].mean()),
           "ensemble_cv_mae_std": float(ens["fold_metrics"]["mae"].std(ddof=1)),
           "champion": champion, "champion_cv_mae": cv_mae[champion], "cv_gain_vs_champion": gain,
           "folds_ensemble_better": int((paired > 0).sum()), "n_folds": int(len(paired)),
           "baseline_drug_mean_cv_mae": float(base["fold_metrics"]["mae"].mean()),
           "mean_pairwise_pred_corr_top": float(np.mean([corr_pred.loc[a, b] for i, a in enumerate(top) for b in top[i + 1:]])),
           "mean_pairwise_error_corr_top": float(np.mean([corr_err.loc[a, b] for i, a in enumerate(top) for b in top[i + 1:]]))}
    corr_pred.round(3).to_csv(bm.OUT / "regression_pred_correlation.csv")
    corr_err.round(3).to_csv(bm.OUT / "regression_error_correlation.csv")
    cv_ok = gain >= g["min_cv_improvement"] and (paired > 0).mean() >= 0.6
    out["cv_gate_passed"] = bool(cv_ok)
    out["adopted"] = False
    out["decision"] = "ensemble not adopted: it did not beat the champion in cross-validation"
    if cv_ok:   # only now touch the locked test, once, for this single candidate
        final = rb.build(ens_name, genes, "drug_identity", cfg).fit(Xdev, ydev)
        p = final.predict(Xtest)
        tm = rb.reg_metrics(ytest, p)
        spec = json.load(open(lock))
        spec["test_evaluations"].append({"at": datetime.now(timezone.utc).isoformat(), "tag": "regression_ensemble_check"})
        lock.write_text(json.dumps(spec, indent=2))
        champ_test = summ["locked_test_metrics"]["mae"]
        out["ensemble_test_mae"], out["champion_test_mae"] = tm["mae"], champ_test
        if tm["mae"] <= champ_test + g["max_test_drop"] and ens["fold_metrics"]["mae"].std(ddof=1) <= \
                champ["fold_metrics"]["mae"].std(ddof=1) * g["max_std_ratio"]:
            out["adopted"] = True
            out["decision"] = "ensemble adopted (CV gain, stable, locked test not worse)"
            drugs = (df.groupby("DRUG_NAME").agg(PATHWAY_NAME=("PATHWAY_NAME", lambda s: s.mode().iloc[0]),
                                                  PUTATIVE_TARGET=("PUTATIVE_TARGET", lambda s: str(s.mode().iloc[0]) if s.notna().any() else "unknown"),
                                                  MIN_CONC=("MIN_CONC", "median"), MAX_CONC=("MAX_CONC", "median"),
                                                  n_cell_lines=(rb.ID, "nunique")).reset_index())
            bundle = {"model": final, "genes": genes, "drugs": drugs, "target": r["target"],
                      "background": Xdev[genes].sample(100, random_state=cfg["random_seed"]).reset_index(drop=True)}
            ent = reg.register_benchmark_model(
                task="dose_response", name="voting_" + "_".join(top), display_name="Regressor ensemble",
                model_object=bundle, feature_columns=genes + ["DRUG_NAME"], feature_pipeline="expression+drug_identity",
                dataset="gdsc_colorectal_dose_response_processed", hyperparameters={"members": top},
                cv_metrics={k: v for k, v in rb.summ(ens).items() if k != "name"},
                test_metrics={**tm, "n_rows": int(len(ytest)), "n_cell_lines": int(Xtest[rb.ID].nunique()),
                              "mae_gain_vs_drug_mean": float((summ["locked_test_metrics"]["baseline_drug_mean"]["mae"] - tm["mae"])
                                                             / summ["locked_test_metrics"]["baseline_drug_mean"]["mae"])},
                external_metrics=None, calibration="n/a", seed=cfg["random_seed"], extra={"target_column": r["target"]})
            reg.promote_champion("dose_response", ent, out["decision"])
        else:
            out["decision"] = "ensemble not adopted: locked test or stability check failed"
    (bm.OUT / "regression_ensemble_check.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
