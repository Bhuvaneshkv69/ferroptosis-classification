"""Phase 1: reproduce the 35-gene baseline FROM SCRATCH and verify it against the stored benchmark.

Re-derives labels, sample IDs and the locked split, refits the production procedure (nested CV with a fresh,
uncached tag, then a fresh final fit), scores the locked test set, and checks every number against
results/benchmark/benchmark_summary.json and locked_test_predictions.csv. Output: results/baseline_35_gene/.

    python scripts/reproduce_baseline.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import benchmark as bm  # noqa: E402

OUT = ROOT / "results" / "baseline_35_gene"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = bm.load_cfg()
    stored = json.load(open(ROOT / "results" / "benchmark" / "benchmark_summary.json"))
    raw = pd.read_csv(bm.PROCESSED / "tcga_coadread_processed.csv")
    df, genes, y = bm.load_msi_data()
    dev, test, lock_spec, lock_path = bm.locked_split(df, y, cfg)

    checks = {}
    checks["n_patients_in_table"] = int(len(raw))
    checks["n_unlabelled_excluded"] = int(raw["msi_label"].isna().sum())
    checks["n_labelled"] = int(len(df))
    checks["label_counts"] = {"MSI-H": int(y.sum()), "MSS/MSI-L": int((1 - y).sum())}
    checks["labels_match_mantis_threshold"] = bool(
        (df["msi_label"] == np.where(df["MSI_SCORE_MANTIS"] >= 0.4, "MSI-H", "MSS/MSI-L")).all())
    checks["sample_ids_unique"] = bool(df["sampleId"].is_unique and df["patientId"].is_unique)
    checks["split_dev_test_disjoint"] = not set(dev) & set(test)
    checks["split_fingerprint_matches_file"] = bool(lock_spec["fingerprint"] == bm._fingerprint(df["sampleId"], y))
    checks["n_dev"], checks["n_test"] = int(len(dev)), int(len(test))
    checks["test_positives"] = int(y[test].sum())
    checks["gene_columns"] = len(genes)
    checks["missing_gene_values"] = int(df[genes].isna().sum().sum())

    spec = stored["champion_spec"]
    Xdev, ydev = df[genes].iloc[dev].reset_index(drop=True), y[dev]
    Xtest, ytest = df[genes].iloc[test].reset_index(drop=True), y[test]
    cv = bm.nested_cv(spec, Xdev, ydev, cfg, repeats=cfg["classification"]["outer_repeats"], cache_tag="baseline_repro_fresh")
    cvs = bm.summarize(cv)
    est = bm.fit_final(spec, Xdev, ydev, cfg)
    proba, pred = est.predict_proba(Xtest)[:, 1], est.predict(Xtest)
    m = bm.clf_metrics(ytest, proba, pred)
    bm.log_test_evaluation(lock_path, "baseline_reproduction", "Phase 1: from-scratch reproduction of the 35-gene baseline")

    old = pd.read_csv(ROOT / "results" / "benchmark" / "locked_test_predictions.csv")
    checks["test_predictions_identical_to_stored"] = bool(np.allclose(old["proba"].to_numpy(), proba, atol=1e-9))
    checks["cv_accuracy_matches_stored"] = bool(abs(cvs["accuracy_mean"] - stored["cv_metrics"]["accuracy_mean"]) < 1e-9)
    checks["test_accuracy_matches_stored"] = bool(abs(m["accuracy"] - stored["locked_test_metrics"]["accuracy"]) < 1e-9)
    checks["all_passed"] = all(v for k, v in checks.items() if isinstance(v, bool))

    keys = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "mcc", "specificity"]
    metrics = {"model": spec, "cv_mean": {k: cvs[f"{k}_mean"] for k in keys if f"{k}_mean" in cvs},
               "cv_std": {k: cvs[f"{k}_std"] for k in keys if f"{k}_std" in cvs},
               "locked_test": {k: float(m[k]) for k in keys}, "locked_test_ci95": bm.bootstrap_ci(ytest, proba, pred),
               "majority_baseline_accuracy": float(1 - y.mean())}
    (OUT / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2))
    pd.DataFrame([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], index=["true MSS/MSI-L", "true MSI-H"],
                 columns=["pred MSS/MSI-L", "pred MSI-H"]).astype(int).to_csv(OUT / "confusion_matrix.csv")
    pd.DataFrame({"sampleId": df["sampleId"].iloc[test].to_numpy(), "y_true": ytest, "proba": proba, "pred": pred}).to_csv(
        OUT / "locked_test_predictions.csv", index=False)
    lines = ["# 35-gene baseline (reproduced from scratch)", "",
             f"Verification passed: **{checks['all_passed']}** (see `verification.json`).", "",
             "| Metric | CV mean ± SD | Locked test |", "|---|---|---|"]
    for k in keys:
        cvm = f"{cvs[f'{k}_mean']:.3f} ± {cvs[f'{k}_std']:.3f}" if f"{k}_mean" in cvs else "-"
        lines.append(f"| {k} | {cvm} | {m[k]:.3f} |")
    lines += ["", f"Majority-class accuracy: {1 - y.mean():.3f}. Confusion matrix (locked test): TN {m['tn']:.0f}, FP {m['fp']:.0f}, "
              f"FN {m['fn']:.0f}, TP {m['tp']:.0f}."]
    (OUT / "baseline_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(checks, indent=2))
    print({k: round(v, 3) for k, v in metrics["locked_test"].items()})


if __name__ == "__main__":
    main()
